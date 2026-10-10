"""
Ablation runner: does the physics-informed loss remove ghost peaks?

Trains the SAME model (same init seed, same data, same batch order) under each loss arm, then
evaluates every arm on the SAME held-out-compound test samples with matched Pearson AND the
ghost metrics (ghost-mass fraction, peak precision / recall / F1, spurious / missed peaks).

Arms
----
pit          standard Hungarian-MSE + reconstruction (baseline)
physics      PIT + label-free physics terms (sparsity + baseline TV + linewidth + symmetry + Pascal)
gt_baseline  PIT + SUPERVISED L1 on the ground-truth baseline region (control: how far a direct,
             label-using ghost penalty goes; NOT physics)
Optional extra arms (via --arms): sparse_only, tv_only, linewidth_only, symmetry_only, pascal_only,
disjoint

Reference rows: NMF (untrained) and the ground truth itself (the floor of the ghost metrics, which
is non-zero because of Lorentzian tails).

Every arm is selected at its own best validation loss. Arms optimise different objectives, so the
raw validation numbers are NOT comparable across arms; the test metrics are.

Usage:
    python ablation.py --model v1 --epochs 40 --train-size 400
    python ablation.py --model v1 --arms pit physics gt_baseline sparse_only --seeds 0 1 2
"""
import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.synthetic import MixtureSampler, load_library, split_compounds
from src.data.torch_dataset import SyntheticNMRDataset
from src.evaluation.baselines import matched_correlation, nmf_separate, summarize
from src.evaluation.benchmark import draw, predict
from src.evaluation.ghost import ghost_metrics
from src.models.factory import MODEL_NAMES, build_model
from src.training.physics import PhysicsPITLoss, warmup_ramp
from src.training.pit import PITLoss
from train import run_epoch

DEFAULT_ARMS = ["pit", "physics", "gt_baseline"]
ALL_ARMS = DEFAULT_ARMS + ["sparse_only", "tv_only", "linewidth_only", "symmetry_only", "pascal_only", "disjoint"]

# metric key -> (label, higher_is_better)
METRICS = {
    "matched_pearson": ("matched Pearson", True),
    "ghost_mass_fraction": ("ghost mass frac", False),
    "peak_precision": ("peak precision", True),
    "peak_recall": ("peak recall", True),
    "peak_f1": ("peak F1", True),
    "n_spurious_per_source": ("spurious/src", False),
    "n_missed_per_source": ("missed/src", False),
}


def make_criterion(arm: str, a: argparse.Namespace):
    """Loss for one arm. `a` carries the lambda values."""
    if arm == "pit":
        return PITLoss(lambda_recon=a.lambda_recon)
    common = dict(lambda_recon=a.lambda_recon, sparse_kind=a.sparse_type, sparse_eps=a.sparse_eps,
                  lw_tolerance=a.lw_tolerance, lw_scope=a.lw_scope)
    if arm == "physics":
        return PhysicsPITLoss(lambda_sparse=a.lambda_sparse, lambda_tv=a.lambda_tv, lambda_linewidth=a.lambda_linewidth,
                              lambda_symmetry=a.lambda_symmetry, lambda_multiplet=a.lambda_multiplet, **common)
    if arm == "gt_baseline":
        return PhysicsPITLoss(lambda_gt_baseline=a.lambda_gt_baseline, **common)
    single = {
        "sparse_only": {"lambda_sparse": a.lambda_sparse}, "tv_only": {"lambda_tv": a.lambda_tv},
        "linewidth_only": {"lambda_linewidth": a.lambda_linewidth}, "symmetry_only": {"lambda_symmetry": a.lambda_symmetry},
        "pascal_only": {"lambda_multiplet": a.lambda_multiplet}, "disjoint": {"lambda_disjoint": a.lambda_disjoint},
    }
    if arm in single:
        return PhysicsPITLoss(**single[arm], **common)
    raise ValueError(f"unknown arm {arm!r}; choose from {ALL_ARMS}")


def train_arm(arm: str, a: argparse.Namespace, library, pools, device, seed: int):
    """Train one arm; returns (model at best epoch, best_epoch, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_set = SyntheticNMRDataset(library, pools["train"], a.train_size, a.noise, seed=1, resample_each_epoch=True)
    val_set = SyntheticNMRDataset(library, pools["val"], a.val_size, a.noise, seed=2)
    tl = DataLoader(train_set, batch_size=a.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(val_set, batch_size=a.batch_size, shuffle=False, num_workers=0)
    model = build_model(a.model).to(device)
    crit = make_criterion(arm, a)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best, best_state, best_epoch, stale, history = float("inf"), None, 0, 0, []
    for ep in range(a.epochs):
        t0 = time.time()
        train_set.set_epoch(ep)
        if hasattr(crit, "set_ramp"):
            crit.set_ramp(warmup_ramp(ep, a.physics_warmup_epochs))
        tr = run_epoch(model, tl, crit, device, opt)
        if hasattr(crit, "set_ramp"):
            crit.set_ramp(1.0)      # validate at full weight so epochs are comparable
        va = run_epoch(model, vl, crit, device)
        sched.step()
        improved = va["loss"] < best
        if improved:
            best, best_epoch, stale = va["loss"], ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        history.append({"epoch": ep + 1, "train": tr, "val": va})
        print(f"  [{arm}] epoch {ep + 1:>3}/{a.epochs} train {tr['loss']:.5f} val {va['loss']:.5f} "
              f"{time.time() - t0:.0f}s {'<< best' if improved else f'(stale {stale}/{a.patience})'}", flush=True)
        if stale >= a.patience:
            print(f"  [{arm}] early stop")
            break
    model.load_state_dict(best_state)
    return model, best_epoch, history


def per_sample_metrics(samples: List[dict], predictor) -> Dict[str, np.ndarray]:
    """predictor(sample) -> (N, L) prediction. Returns metric -> per-sample array."""
    out: Dict[str, List[float]] = {k: [] for k in METRICS}
    for d in samples:
        pred = predictor(d)
        out["matched_pearson"].append(float(matched_correlation(d["compounds"], pred).mean()))
        for k, v in ghost_metrics(d["compounds"], pred).items():
            out[k].append(v)
    return {k: np.asarray(v) for k, v in out.items()}


def paired_delta(arm: Dict[str, np.ndarray], ref: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Per-sample paired difference (arm - ref) with 95% CI; same test samples for every arm."""
    return {k: summarize(arm[k] - ref[k]) for k in METRICS}


def _fmt(s: dict) -> str:
    return f"{s['mean']:.4f}±{s['ci95']:.4f}"


def format_comparison(results: Dict[str, Dict[str, np.ndarray]], ref_rows: Dict[str, Dict[str, float]]) -> str:
    cols = list(METRICS)
    head = f"{'arm':<16}" + "".join(f"{METRICS[c][0] + (' ↑' if METRICS[c][1] else ' ↓'):>22}" for c in cols)
    lines = [head, "-" * len(head)]
    for arm, m in results.items():
        lines.append(f"{arm:<16}" + "".join(f"{_fmt(summarize(m[c])):>22}" for c in cols))
    for name, row in ref_rows.items():
        lines.append(f"{name:<16}" + "".join(f"{(f'{row[c]:.4f}' if c in row else '-'):>22}" for c in cols))
    return "\n".join(lines)


def format_paired(paired: Dict[str, Dict[str, dict]], baseline: str) -> str:
    cols = list(METRICS)
    head = f"{'Δ vs ' + baseline:<16}" + "".join(f"{METRICS[c][0]:>22}" for c in cols)
    lines = [head, "-" * len(head)]
    for arm, d in paired.items():
        cells = []
        for c in cols:
            s = d[c]
            good = (s["mean"] > 0) == METRICS[c][1]
            sig = abs(s["mean"]) > s["ci95"]
            tag = ("+" if good else "-") if sig else "~"       # + significantly better, - worse, ~ within CI
            cells.append(f"{s['mean']:+.4f}±{s['ci95']:.4f} {tag}")
        lines.append(f"{arm:<16}" + "".join(f"{c:>22}" for c in cells))
    lines.append("(+ significantly better than baseline, - significantly worse, ~ within 95% CI)")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODEL_NAMES, default="v1")
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=ALL_ARMS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="init/shuffle seeds; the compound split uses the first")
    p.add_argument("--data-dir", default=os.path.join(ROOT, "NMR_PROJECT_FINAL_PACKAGE"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--lambda-recon", type=float, default=0.5)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--train-size", type=int, default=400)
    p.add_argument("--val-size", type=int, default=60)
    p.add_argument("--test-size", type=int, default=60)
    p.add_argument("--n-val-compounds", type=int, default=6)
    p.add_argument("--n-test-compounds", type=int, default=7)
    p.add_argument("--nmf-samples", type=int, default=30, help="0 disables the NMF reference row")
    g = p.add_argument_group("loss weights (physics arm uses the first five; gt_baseline uses --lambda-gt-baseline)")
    g.add_argument("--lambda-sparse", type=float, default=0.05)
    g.add_argument("--lambda-tv", type=float, default=0.05)
    g.add_argument("--lambda-linewidth", type=float, default=0.01)
    g.add_argument("--lambda-symmetry", type=float, default=0.01)
    g.add_argument("--lambda-multiplet", type=float, default=0.01)
    g.add_argument("--lambda-disjoint", type=float, default=0.01)
    g.add_argument("--lambda-gt-baseline", type=float, default=0.05)
    g.add_argument("--sparse-type", choices=["l1", "log"], default="log")
    g.add_argument("--sparse-eps", type=float, default=0.02)
    g.add_argument("--lw-tolerance", type=float, default=0.25)
    g.add_argument("--lw-scope", choices=["source", "sample"], default="source")
    g.add_argument("--physics-warmup-epochs", type=int, default=5)
    p.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--output", default=None, help="default: <results-dir>/ablation_<model>.json")
    a = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, library = load_library(a.data_dir)
    pools = split_compounds(len(names), a.n_val_compounds, a.n_test_compounds, seed=a.seeds[0])
    samples = draw(MixtureSampler(library, pools["test"], noise_std=a.noise, seed=3), a.test_size)
    print(f"device={device} model={a.model} arms={a.arms} seeds={a.seeds} test samples={len(samples)} (held-out compounds)")

    per_arm: Dict[str, List[Dict[str, np.ndarray]]] = {arm: [] for arm in a.arms}   # one entry per seed
    meta: Dict[str, List[dict]] = {arm: [] for arm in a.arms}
    for seed in a.seeds:
        for arm in a.arms:
            print(f"\n=== arm={arm} seed={seed} ===", flush=True)
            model, best_epoch, history = train_arm(arm, a, library, pools, device, seed)
            per_arm[arm].append(per_sample_metrics(samples, lambda d: predict(model, d["mixtures"], device)))
            meta[arm].append({"seed": seed, "best_epoch": best_epoch, "epochs_run": len(history), "last_val": history[-1]["val"]})
            del model

    # pool seeds: concatenate per-sample arrays (paired across arms by construction: same samples, same seed order)
    pooled = {arm: {k: np.concatenate([r[k] for r in runs]) for k in METRICS} for arm, runs in per_arm.items()}

    ref_rows: Dict[str, Dict[str, float]] = {}
    floor = per_sample_metrics(samples, lambda d: d["compounds"])
    ref_rows["ground truth"] = {k: float(v.mean()) for k, v in floor.items()}      # floor/ceiling of each metric
    nmf_res = None
    if a.nmf_samples > 0:
        nmf = per_sample_metrics(samples[: a.nmf_samples], lambda d: nmf_separate(d["mixtures"], d["compounds"].shape[0]))
        ref_rows["NMF"] = {k: float(v.mean()) for k, v in nmf.items()}
        nmf_res = {k: summarize(v) for k, v in nmf.items()}

    print("\n" + "=" * 100 + f"\nHELD-OUT COMPOUND TEST  (n={len(samples)} samples x {len(a.seeds)} seed(s); mean ± 95% CI)\n" + "=" * 100)
    print(format_comparison(pooled, ref_rows))
    baseline = "pit" if "pit" in pooled else a.arms[0]
    paired = {arm: paired_delta(pooled[arm], pooled[baseline]) for arm in a.arms if arm != baseline}
    if paired:
        print("\n" + format_paired(paired, baseline))

    out_path = a.output or os.path.join(a.results_dir, f"ablation_{a.model}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(a),
            "arms": {arm: {"metrics": {k: summarize(pooled[arm][k]) for k in METRICS}, "runs": meta[arm]} for arm in a.arms},
            "paired_vs_baseline": {"baseline": baseline, "delta": paired},
            "reference": {"ground_truth": ref_rows["ground truth"], "nmf": nmf_res},
        }, f, indent=2)
    print(f"\nwrote {out_path}")
    return pooled, paired


if __name__ == "__main__":
    main()
