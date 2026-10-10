"""
Physics-informed regularisers for NMR blind source separation, and PhysicsPITLoss.

Why these terms exist
---------------------
MSE and Pearson are dominated by the tall peaks: with 16,384 points of which ~98% are
baseline, a 0.1-high ghost peak costs 0.01 in squared error but 0.1 in L1. The terms
below have non-vanishing gradients for small amplitudes, so crosstalk ("ghost peaks")
that leaks from other compounds into a predicted component is actually penalised.

All terms except `gt_baseline_l1` are LABEL-FREE: they use only the predicted spectra
`pred` of shape (B, N, L), never the ground truth or the Hungarian permutation. They
therefore also apply to real blind data and are permutation invariant by construction.

Terms
-----
sparsity_loss      L1 (Kopriva et al. 2009, eq. 16-17) or a log-sum reweighted-L1
                   surrogate that penalises small values strongly but barely shrinks
                   tall true peaks.
baseline_tv_loss   Pseudo-Huber total variation restricted to the baseline region
                   (far from any detected peak): penalises ripple, not peak shape.
linewidth_loss     Peaks of one compound share one Lorentzian linewidth (uniform
                   tumbling / T2). Local width is read from apex curvature,
                   gamma = sqrt(2 s / -s''); the loss is a tolerance-banded weighted
                   variance of log(gamma).
symmetry_loss      First-order multiplets are symmetric about their centre.
pascal_loss        Resolved multiplets must obey Pascal-triangle ratios and equal J
                   spacing. Gated by a resolvedness score, so it is a no-op on
                   unresolved lines (the 30-compound library has FWHM ~12.6 Hz > J).
disjoint_loss      Kopriva's per-frequency sparsity across sources: few sources are
                   active at any one frequency. Optional; real compounds do overlap.
gt_baseline_l1     SUPERVISED control arm (not physics): L1 on the ground-truth
                   baseline region. Upper bound for what the physics terms could do.
"""
from math import comb
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

from src.training.pit import PITLoss, match_sources, reorder

EPS = 1e-8
PEAK_THRESHOLD = 0.02      # absolute amplitude above which a point counts as "peak" (spectra max ~ 1)
BASELINE_DILATION = 96     # points (~0.06 ppm at L=16384) kept out of the baseline mask around peaks


# --------------------------------------------------------------------------- helpers
def _peak_mask(s: torch.Tensor, threshold: float = PEAK_THRESHOLD, dilation: int = BASELINE_DILATION) -> torch.Tensor:
    """Float mask (B,N,L): 1 within `dilation` points of any point above `threshold`. No gradient."""
    with torch.no_grad():
        m = (s > threshold).float()
        return F.max_pool1d(m, kernel_size=2 * dilation + 1, stride=1, padding=dilation)


def _local_maxima(s: torch.Tensor, threshold: float = PEAK_THRESHOLD, radius: int = 2) -> torch.Tensor:
    """Bool mask (B,N,L) of strict-ish local maxima above threshold. No gradient."""
    with torch.no_grad():
        pooled = F.max_pool1d(s, kernel_size=2 * radius + 1, stride=1, padding=radius)
        return (s >= pooled) & (s > threshold)


# --------------------------------------------------------------------------- sparsity
def sparsity_loss(pred: torch.Tensor, kind: str = "log", eps: float = 0.02) -> torch.Tensor:
    """
    Mean sparsity penalty over (B,N,L). Predictions are non-negative (Softplus), so L1 is the mean.
    kind='l1'  : mean(s)                     (Kopriva 2009)
    kind='log' : mean(log(1+s/eps)) / log(1+1/eps)   reweighted-L1 / log-sum surrogate;
                 normalised so that s==1 everywhere gives 1. Gradient 1/(eps+s): large for
                 small ghosts, small for tall peaks (less amplitude shrinkage than L1).
    """
    s = pred.clamp(min=0.0)
    if kind == "l1":
        return s.mean()
    if kind == "log":
        return torch.log1p(s / eps).mean() / float(np.log1p(1.0 / eps))
    raise ValueError(f"unknown sparsity kind {kind!r}")


# --------------------------------------------------------------------------- baseline ripple
def baseline_tv_loss(pred: torch.Tensor, delta: float = 1e-3) -> torch.Tensor:
    """Pseudo-Huber TV of the prediction on the baseline region (far from detected peaks)."""
    s = pred
    base = 1.0 - _peak_mask(s)
    b = base[..., 1:] * base[..., :-1]
    d = s[..., 1:] - s[..., :-1]
    ph = torch.sqrt(d * d + delta * delta) - delta
    return (b * ph).mean()


# --------------------------------------------------------------------------- linewidth
def linewidth_loss(
    pred: torch.Tensor,
    tolerance: float = 0.25,
    scope: str = "source",
    threshold: float = PEAK_THRESHOLD,
    min_log_gamma: float = 0.0,
    max_log_gamma: float = 6.0,
) -> torch.Tensor:
    """
    Linewidth consistency. At each local maximum of a Lorentzian of half-width gamma (points),
    s'' = -2 s / gamma^2, hence gamma = sqrt(2 s / -s''). We take log(gamma) at every
    detected maximum and penalise its deviation from the amplitude-weighted mean of log(gamma)
    (the mean is detached), beyond a tolerance band (unresolved multiplets look broader).

    scope='source': one linewidth per compound (Sir's constraint).
    scope='sample': one linewidth shared by all compounds in a sample (shimming dominates).
    """
    s = pred
    curv = -(s[..., 2:] - 2.0 * s[..., 1:-1] + s[..., :-2])           # -s'' in points^-2, (B,N,L-2)
    c = s[..., 1:-1]
    log_gamma = 0.5 * torch.log(2.0 * c.clamp(min=EPS) / curv.clamp(min=EPS))
    log_gamma = log_gamma.clamp(min_log_gamma, max_log_gamma)
    w = (_local_maxima(s, threshold)[..., 1:-1].float()) * c.detach() ** 2   # amplitude^2 weights, no grad through w
    dims = (-1,) if scope == "source" else (-2, -1)
    wsum = w.sum(dim=dims, keepdim=True)
    mu = ((w * log_gamma.detach()).sum(dim=dims, keepdim=True) / wsum.clamp(min=EPS))
    dev = F.relu((log_gamma - mu).abs() - tolerance)
    num = (w * dev * dev).sum(dim=dims)
    den = wsum.clamp(min=EPS).reshape(num.shape)
    return (num / den).mean()


# --------------------------------------------------------------------------- symmetry
def symmetry_loss(pred: torch.Tensor, top_k: int = 8, half_window: int = 96, threshold: float = PEAK_THRESHOLD) -> torch.Tensor:
    """
    Reflection asymmetry of the top-K peak clusters per source. A window (Hann-tapered, to
    limit contamination from neighbouring clusters) is cut around each peak, its centroid c is
    found, and the window is compared to its mirror image about c (linear interpolation).
    Loss = sum((w - mirror)^2) / sum(w^2), averaged over selected peaks. Valid at any resolution.
    """
    B, N, L = pred.shape
    K = min(top_k, L)
    W = half_window
    with torch.no_grad():
        cand = torch.where(_local_maxima(pred, threshold), pred, torch.zeros_like(pred))
        vals, idx = cand.topk(K, dim=-1)                                   # (B,N,K)
        valid = (vals > threshold).float()
    offs = torch.arange(-W, W + 1, device=pred.device)                      # (2W+1,)
    offs_f = offs.to(pred.dtype)
    taper = torch.hann_window(2 * W + 1, periodic=False, device=pred.device, dtype=pred.dtype)
    expanded = pred.unsqueeze(2).expand(B, N, K, L)

    def cut(centres: torch.Tensor) -> torch.Tensor:
        pos = (centres.unsqueeze(-1) + offs).clamp(0, L - 1)               # (B,N,K,2W+1)
        return expanded.gather(-1, pos) * taper

    # pass 1: window on the peak -> centroid; pass 2: re-cut on the centroid so an outer line of a
    # resolved multiplet sees the whole (symmetric) cluster rather than half of it
    with torch.no_grad():
        idx2 = idx
        for _ in range(5):   # fixed point: the tapered centroid is biased toward the window centre, so iterate
            w1 = cut(idx2)
            c1 = (w1 * offs_f).sum(-1) / w1.sum(-1).clamp(min=EPS)          # (B,N,K)
            idx2 = (idx2 + c1.round().long()).clamp(0, L - 1)
    win = cut(idx2)
    cen = ((win.detach() * offs_f).sum(-1, keepdim=True) / win.detach().sum(-1, keepdim=True).clamp(min=EPS))  # (B,N,K,1)
    q = (2.0 * cen - offs_f).clamp(-W, W) + W                               # mirrored coordinates in window index space
    lo = q.floor().long().clamp(0, 2 * W)
    hi = (lo + 1).clamp(max=2 * W)
    frac = q - lo.to(pred.dtype)
    mirror = win.gather(-1, lo) * (1 - frac) + win.gather(-1, hi) * frac
    num = ((win - mirror) ** 2).sum(-1)
    den = (win ** 2).sum(-1).clamp(min=EPS)
    per_peak = num / den
    return (per_peak * valid).sum() / valid.sum().clamp(min=1.0)


# --------------------------------------------------------------------------- Pascal
def _pascal_row(n: int) -> np.ndarray:
    r = np.array([comb(n - 1, k) for k in range(n)], dtype=np.float64)
    return r / r.sum()


def pascal_loss(
    pred: torch.Tensor,
    max_gap: int = 66,
    rel_height: float = 0.05,
    margin: float = 0.05,
    resolved_valley: float = 0.8,
    max_lines: int = 6,
) -> torch.Tensor:
    """
    Pascal-ratio + equal-spacing penalty on RESOLVED multiplets.

    Per source, lines are detected on detached values (scipy find_peaks, height >= rel_height*max),
    grouped into clusters when consecutive lines are closer than `max_gap` points (~0.04 ppm), and
    the line heights are read back differentiably. A cluster of n lines is compared with the
    n-th Pascal row (n is the observed line count). Each cluster is weighted by a stop-gradient
    resolvedness gate g = sigmoid((resolved_valley - valley/min(adjacent line heights)) / 0.1),
    so on unresolved data (valleys ~ line heights) the term vanishes. The gate is detached so the
    network cannot dodge the penalty by merging lines.
    Cost: one scipy call per (batch, source); acceptable for B*N ~ 10-20.
    """
    B, N, L = pred.shape
    total = pred.sum() * 0.0   # keeps a graph even when no resolved cluster exists
    count = 0.0
    arr = pred.detach().float().cpu().numpy()
    for b in range(B):
        for n in range(N):
            x = arr[b, n]
            mx = float(x.max())
            if mx <= PEAK_THRESHOLD:
                continue
            pk, _ = find_peaks(x, height=rel_height * mx)
            if len(pk) < 2:
                continue
            groups, cur = [], [pk[0]]
            for p in pk[1:]:
                if p - cur[-1] <= max_gap:
                    cur.append(p)
                else:
                    groups.append(cur)
                    cur = [p]
            groups.append(cur)
            for g in groups:
                k = len(g)
                if k < 2 or k > max_lines:
                    continue
                idx = torch.as_tensor(g, device=pred.device)
                h = pred[b, n].index_select(0, idx)
                hn = h / h.sum().clamp(min=EPS)
                # resolvedness gate (detached): worst valley relative to adjacent lines
                ratios = []
                for a, c in zip(g[:-1], g[1:]):
                    valley = float(x[a:c + 1].min())
                    ratios.append(valley / max(min(x[a], x[c]), EPS))
                gate = 1.0 / (1.0 + np.exp(-(resolved_valley - max(ratios)) / 0.1))
                if gate < 1e-3:
                    continue
                target = torch.as_tensor(_pascal_row(k), device=pred.device, dtype=pred.dtype)
                ratio_term = F.relu((hn - target).abs() - margin).pow(2).sum()
                spacing_term = pred.new_zeros(())
                if k >= 3:
                    # differentiable sub-pixel line positions (parabolic apex interpolation)
                    y0 = pred[b, n].index_select(0, idx)
                    ym = pred[b, n].index_select(0, (idx - 1).clamp(min=0))
                    yp = pred[b, n].index_select(0, (idx + 1).clamp(max=L - 1))
                    delta = 0.5 * (ym - yp) / (ym - 2.0 * y0 + yp).clamp(max=-EPS)
                    sp = (idx[1:].to(pred.dtype) + delta[1:]) - (idx[:-1].to(pred.dtype) + delta[:-1])
                    rel_sd = torch.sqrt(sp.var(unbiased=False) + EPS) / sp.mean().clamp(min=EPS)
                    spacing_term = F.relu(rel_sd - 0.05).pow(2)
                total = total + gate * (ratio_term + spacing_term)
                count += gate
    return total / max(count, 1.0)


def pascal_spacing_irregularity(pred: torch.Tensor, max_gap: int = 66, rel_height: float = 0.05) -> float:
    """Diagnostic (non-differentiable): mean relative variance of line spacing within clusters of >=3 lines."""
    vals = []
    arr = pred.detach().float().cpu().numpy()
    for x in arr.reshape(-1, arr.shape[-1]):
        mx = float(x.max())
        if mx <= PEAK_THRESHOLD:
            continue
        pk, _ = find_peaks(x, height=rel_height * mx)
        cur = [pk[0]] if len(pk) else []
        for p in pk[1:]:
            if p - cur[-1] <= max_gap:
                cur.append(p)
            else:
                if len(cur) >= 3:
                    sp = np.diff(cur)
                    vals.append(np.var(sp) / np.mean(sp) ** 2)
                cur = [p]
        if len(cur) >= 3:
            sp = np.diff(cur)
            vals.append(np.var(sp) / np.mean(sp) ** 2)
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------- disjointness
def disjoint_loss(pred: torch.Tensor) -> torch.Tensor:
    """Mean over points of sum_{i<j} s_i s_j = ((sum_i s_i)^2 - sum_i s_i^2) / 2."""
    s = pred.clamp(min=0.0)
    return 0.5 * ((s.sum(1) ** 2) - (s ** 2).sum(1)).mean()


# --------------------------------------------------------------------------- supervised control
def gt_baseline_l1(matched_pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """SUPERVISED control: mean of prediction on the ground-truth baseline region."""
    base = 1.0 - _peak_mask(true)
    return (base * matched_pred).mean()


# --------------------------------------------------------------------------- the loss
class PhysicsPITLoss(PITLoss):
    """
    PITLoss (Hungarian MSE + reconstruction) plus label-free physics regularisers.

    total = lambda_spectral*spec + lambda_recon*recon
          + ramp * ( lambda_sparse*sparse + lambda_tv*tv + lambda_linewidth*lw
                   + lambda_symmetry*sym + lambda_multiplet*pascal
                   + lambda_disjoint*disj + lambda_gt_baseline*gtb )

    With all physics lambdas == 0 this equals PITLoss exactly. `ramp` in [0,1] is set from the
    training loop (warm-up), because strong sparsity from step 0 pushes a Softplus output to the
    trivial all-zero solution.
    """

    def __init__(
        self,
        lambda_spectral: float = 1.0,
        lambda_recon: float = 0.5,
        lambda_sparse: float = 0.0,
        lambda_linewidth: float = 0.0,
        lambda_multiplet: float = 0.0,
        lambda_symmetry: float = 0.0,
        lambda_tv: float = 0.0,
        lambda_disjoint: float = 0.0,
        lambda_gt_baseline: float = 0.0,
        sparse_kind: str = "log",
        sparse_eps: float = 0.02,
        lw_tolerance: float = 0.25,
        lw_scope: str = "source",
    ):
        super().__init__(lambda_spectral, lambda_recon)
        self.lam = {
            "sparse": lambda_sparse, "lw": lambda_linewidth, "pascal": lambda_multiplet,
            "sym": lambda_symmetry, "tv": lambda_tv, "disj": lambda_disjoint, "gtb": lambda_gt_baseline,
        }
        self.sparse_kind, self.sparse_eps = sparse_kind, sparse_eps
        self.lw_tolerance, self.lw_scope = lw_tolerance, lw_scope
        self.ramp = 1.0

    def set_ramp(self, value: float) -> None:
        self.ramp = float(min(max(value, 0.0), 1.0))

    def forward(
        self, pred: torch.Tensor, true: torch.Tensor, concentrations: torch.Tensor, mixtures: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        base_total, parts = super().forward(pred, true, concentrations, mixtures)
        lam = self.lam
        terms: Dict[str, torch.Tensor] = {}
        if lam["sparse"]:
            terms["sparse"] = sparsity_loss(pred, self.sparse_kind, self.sparse_eps)
        if lam["tv"]:
            terms["tv"] = baseline_tv_loss(pred)
        if lam["lw"]:
            terms["lw"] = linewidth_loss(pred, self.lw_tolerance, self.lw_scope)
        if lam["sym"]:
            terms["sym"] = symmetry_loss(pred)
        if lam["pascal"]:
            terms["pascal"] = pascal_loss(pred)
        if lam["disj"]:
            terms["disj"] = disjoint_loss(pred)
        if lam["gtb"]:
            perm = match_sources(pred.detach(), true)
            terms["gtb"] = gt_baseline_l1(reorder(pred, perm), true)
        phys = sum((lam[k] * v for k, v in terms.items()), pred.new_zeros(()))
        total = base_total + self.ramp * phys
        parts = dict(parts)
        parts["loss"] = total.item()
        for k, v in terms.items():
            parts[k] = v.item()
        return total, parts


# --------------------------------------------------------------------------- CLI helpers
def add_physics_args(parser) -> None:
    """Shared flags for train.py and train_procedural.py."""
    g = parser.add_argument_group("physics-informed loss (used when --loss physics_pit)")
    g.add_argument("--loss", choices=["pit", "physics_pit"], default="pit")
    g.add_argument("--lambda-sparse", type=float, default=0.05, help="L1 / log-sum sparsity weight")
    g.add_argument("--lambda-linewidth", type=float, default=0.01, help="linewidth-consistency weight")
    g.add_argument("--lambda-multiplet", type=float, default=0.01, help="Pascal-ratio weight (gated on resolved lines)")
    g.add_argument("--lambda-symmetry", type=float, default=0.01, help="multiplet symmetry weight")
    g.add_argument("--lambda-tv", type=float, default=0.05, help="baseline-ripple TV weight")
    g.add_argument("--lambda-disjoint", type=float, default=0.0, help="cross-source disjointness weight (optional)")
    g.add_argument("--lambda-gt-baseline", type=float, default=0.0, help="SUPERVISED control arm, not physics")
    g.add_argument("--sparse-type", choices=["l1", "log"], default="log")
    g.add_argument("--sparse-eps", type=float, default=0.02)
    g.add_argument("--lw-tolerance", type=float, default=0.25, help="log-width tolerance band")
    g.add_argument("--lw-scope", choices=["source", "sample"], default="source")
    g.add_argument("--physics-warmup-epochs", type=int, default=5, help="linear ramp of physics weights from 0")


def build_criterion(args):
    """PITLoss or PhysicsPITLoss from parsed args (args.lambda_recon optional)."""
    lam_recon = getattr(args, "lambda_recon", 0.5)
    if args.loss == "pit":
        return PITLoss(lambda_recon=lam_recon)
    return PhysicsPITLoss(
        lambda_recon=lam_recon,
        lambda_sparse=args.lambda_sparse, lambda_linewidth=args.lambda_linewidth,
        lambda_multiplet=args.lambda_multiplet, lambda_symmetry=args.lambda_symmetry,
        lambda_tv=args.lambda_tv, lambda_disjoint=args.lambda_disjoint,
        lambda_gt_baseline=args.lambda_gt_baseline,
        sparse_kind=args.sparse_type, sparse_eps=args.sparse_eps,
        lw_tolerance=args.lw_tolerance, lw_scope=args.lw_scope,
    )


def warmup_ramp(epoch: int, warmup_epochs: int) -> float:
    """Ramp for epoch index (0-based): 0 at first epoch, reaching 1 after `warmup_epochs`."""
    return 1.0 if warmup_epochs <= 0 else min(1.0, epoch / warmup_epochs)
