import argparse
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

torch = pytest.importorskip("torch")

from src.data.synthetic import SPECTRAL_LENGTH as L
from src.data.synthetic import MixtureSampler, lorentz_multiplet, make_unseen_library, pascal_weights
from src.evaluation.ghost import ghost_metrics
from src.models.factory import build_model
from src.training import physics as P
from src.training.physics import PhysicsPITLoss, add_physics_args, build_criterion, warmup_ramp
from src.training.pit import PITLoss

PPM = np.linspace(10.0, 0.0, L)


def line(centers, width=0.03, n=1, j=7.0, binomial=True, amp=1.0):
    s = sum(amp * lorentz_multiplet(PPM, c, n, j, width, 1.0, binomial=binomial) for c in centers)
    return np.asarray(s, dtype=np.float32)


def batch(rows):
    """rows: list of (N,L) arrays -> (1, N, L) tensor"""
    return torch.tensor(np.stack(rows))[None]


@pytest.fixture(scope="module")
def clean():
    return batch([line([1.2, 3.6]), line([7.1]), line([2.1, 4.1]), line([6.0]), line([8.5, 0.9])])


def with_ghost(x, amp):
    g = x.clone()
    g[0, 0] += torch.tensor(line([7.1], amp=amp))       # crosstalk copy of source 1's peak into source 0
    g[0, 2] += torch.tensor(line([6.0], amp=amp))
    return g


# ----------------------------------------------------------------------------- sparsity
@pytest.mark.parametrize("kind", ["l1", "log"])
def test_sparsity_penalizes_ghosts_monotonically(clean, kind):
    vals = [P.sparsity_loss(with_ghost(clean, a), kind).item() for a in (0.0, 0.05, 0.1, 0.2)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_log_sparsity_is_more_sensitive_to_small_values_than_l1():
    """Reweighted-L1 gradient is ~1/(eps+s): larger on a 0.05 ghost than on a unit peak; L1 is flat."""
    x = torch.tensor([[[0.05, 1.0]]], requires_grad=True)
    P.sparsity_loss(x, "log", eps=0.02).backward()
    g_log = (x.grad[0, 0, 0] / x.grad[0, 0, 1]).item()
    y = torch.tensor([[[0.05, 1.0]]], requires_grad=True)
    P.sparsity_loss(y, "l1").backward()
    g_l1 = (y.grad[0, 0, 0] / y.grad[0, 0, 1]).item()
    assert g_log > 10.0 and abs(g_l1 - 1.0) < 1e-6


def test_sparsity_far_more_sensitive_than_mse_to_ghosts(clean):
    g = with_ghost(clean, 0.1)
    mse_rel = (torch.nn.functional.mse_loss(g, clean) / torch.nn.functional.mse_loss(clean, torch.zeros_like(clean))).item()
    sp_rel = ((P.sparsity_loss(g) - P.sparsity_loss(clean)) / P.sparsity_loss(clean)).item()
    # measured 23.8x for the log-sum penalty at ghost amplitude 0.1 (plain L1: 10.0x); asserted conservatively
    assert sp_rel > 10 * mse_rel


# ----------------------------------------------------------------------------- baseline ripple
def test_baseline_tv_penalizes_ripple_but_not_peak_shape(clean):
    ripple = clean + 0.003 * torch.sin(torch.arange(L) * 0.3).abs()[None, None]
    tv_clean, tv_ripple = P.baseline_tv_loss(clean).item(), P.baseline_tv_loss(ripple).item()
    assert tv_ripple > 1000 * tv_clean
    # scaling the true peaks leaves the baseline term essentially unchanged
    assert abs(P.baseline_tv_loss(clean * 3).item() - tv_clean) < 1e-6


def test_all_ripple_sensitive_terms_react(clean):
    ripple = clean + 0.003 * torch.sin(torch.arange(L) * 0.3).abs()[None, None]
    assert P.sparsity_loss(ripple) > P.sparsity_loss(clean)
    assert P.baseline_tv_loss(ripple) > P.baseline_tv_loss(clean)


# ----------------------------------------------------------------------------- linewidth
def test_linewidth_zero_for_equal_widths_and_positive_for_mixed(clean):
    assert P.linewidth_loss(clean).item() < 1e-4
    mixed = clean.clone()
    mixed[0, 0] += torch.tensor(line([5.0], width=0.12)) * 0.6        # one broad peak in a narrow-peaked source
    assert P.linewidth_loss(mixed).item() > 100 * max(P.linewidth_loss(clean).item(), 1e-6)


def test_linewidth_tolerance_band_ignores_modest_broadening(clean):
    mild = clean.clone()
    mild[0, 0] += torch.tensor(line([5.0], width=0.036)) * 0.6          # ~20% wider, inside the 25% band
    assert P.linewidth_loss(mild, tolerance=0.25).item() < 1e-3


def test_linewidth_sample_scope_flags_cross_source_mismatch():
    narrow_src = batch([line([2.0], width=0.03), line([6.0], width=0.03)])
    ok = P.linewidth_loss(narrow_src, scope="sample").item()
    mismatch = batch([line([2.0], width=0.03), line([6.0], width=0.12)])
    assert P.linewidth_loss(mismatch, scope="source").item() < 1e-3        # each source self-consistent
    assert P.linewidth_loss(mismatch, scope="sample").item() > 100 * max(ok, 1e-6)


# ----------------------------------------------------------------------------- multiplicity
def test_symmetry_orders_symmetric_below_asymmetric():
    sym = batch([line([5.0], width=0.004, n=3)])
    shoulder = batch([line([5.0], width=0.03) + 0.5 * line([5.04], width=0.03)])
    assert P.symmetry_loss(sym).item() < 0.01 < P.symmetry_loss(shoulder).item()


def test_pascal_resolved_binomial_vs_flat_vs_unresolved():
    binom = batch([line([5.0], width=0.004, n=3, binomial=True)])
    flat = batch([line([5.0], width=0.004, n=3, binomial=False)])
    unresolved = batch([line([5.0], width=0.03, n=3, binomial=True)])
    assert P.pascal_loss(binom).item() < 1e-4
    assert P.pascal_loss(flat).item() > 1e-3
    assert P.pascal_loss(unresolved).item() < 1e-4         # gate closes: lines not resolved


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_pascal_accepts_binomial_rows(n):
    x = batch([line([5.0], width=0.004, n=n, j=8.0, binomial=True)])
    assert P.pascal_loss(x).item() < 1e-4


def test_pascal_gradient_pushes_flat_triplet_toward_pascal():
    x = batch([line([5.0], width=0.004, n=3, binomial=False)]).clone().requires_grad_(True)
    base = P.pascal_loss(x)
    base.backward()
    with torch.no_grad():
        stepped = x - 0.5 * x.grad / x.grad.abs().max() * 0.05
    assert P.pascal_loss(stepped).item() < base.item()


def test_disjoint_prefers_non_overlapping_sources():
    sep = batch([line([2.0]), line([6.0])])
    over = batch([line([2.0]), line([2.0])])
    assert P.disjoint_loss(over) > 10 * P.disjoint_loss(sep)


# ----------------------------------------------------------------------------- gradient flow / stability
TERMS = {
    "sparse_log": lambda x: P.sparsity_loss(x, "log"),
    "sparse_l1": lambda x: P.sparsity_loss(x, "l1"),
    "tv": P.baseline_tv_loss,
    "linewidth": P.linewidth_loss,
    "symmetry": P.symmetry_loss,
    "pascal": P.pascal_loss,
    "disjoint": P.disjoint_loss,
}


@pytest.mark.parametrize("term", sorted(TERMS))
@pytest.mark.parametrize("case", ["clean", "ghost", "zeros", "tiny", "huge", "constant"])
def test_terms_finite_value_and_gradient(clean, term, case):
    base = {
        "clean": clean, "ghost": with_ghost(clean, 0.1), "zeros": torch.zeros_like(clean),
        "tiny": torch.full_like(clean, 1e-9), "huge": clean * 1e4, "constant": torch.full_like(clean, 0.5),
    }[case]
    x = base.clone().requires_grad_(True)
    v = TERMS[term](x)
    v.backward()
    assert torch.isfinite(v)
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("term,step", [("sparse_log", 1e-2), ("sparse_l1", 1e-2), ("tv", 3e-4), ("symmetry", 1e-2)])
def test_gradient_step_reduces_term_on_ghosted_input(clean, term, step):
    ripple = with_ghost(clean, 0.1) + 0.003 * torch.sin(torch.arange(L) * 0.3).abs()[None, None]
    x = ripple.clone().requires_grad_(True)
    v = TERMS[term](x)
    v.backward()
    with torch.no_grad():
        stepped = (x - step * x.grad / x.grad.abs().max()).clamp(min=0)   # TV is non-smooth: step << ripple amplitude (0.003)
    assert TERMS[term](stepped).item() < v.item()


# ----------------------------------------------------------------------------- PhysicsPITLoss
def sampler_batch(b=2, length=L):
    lib = np.stack([line([c, c + 1.7], width=0.03) for c in np.linspace(0.8, 7.6, 12)])
    lib = lib / lib.max(axis=1, keepdims=True)
    sampler = MixtureSampler(lib, np.arange(12), seed=11)
    ds = [sampler.sample(i) for i in range(b)]
    return tuple(torch.from_numpy(np.stack([d[k] for d in ds])) for k in ("mixtures", "compounds", "concentrations"))


def test_physics_pit_equals_pit_when_all_lambdas_zero():
    x, s, a = sampler_batch()
    pred = (s + 0.05 * torch.rand_like(s)).clone()
    ref, _ = PITLoss()(pred, s, a, x)
    got, parts = PhysicsPITLoss()(pred, s, a, x)
    assert torch.equal(ref, got) and "sparse" not in parts


def test_ramp_zero_equals_pit():
    x, s, a = sampler_batch()
    pred = (s + 0.05 * torch.rand_like(s))
    crit = PhysicsPITLoss(lambda_sparse=1.0, lambda_tv=1.0, lambda_linewidth=1.0)
    crit.set_ramp(0.0)
    assert torch.allclose(crit(pred, s, a, x)[0], PITLoss()(pred, s, a, x)[0])
    crit.set_ramp(1.0)
    assert crit(pred, s, a, x)[0] > PITLoss()(pred, s, a, x)[0]


def test_physics_pit_invariant_to_slot_permutation():
    x, s, a = sampler_batch()
    pred = s + 0.03 * torch.rand_like(s)
    crit = PhysicsPITLoss(lambda_sparse=0.1, lambda_tv=0.1, lambda_linewidth=0.1, lambda_symmetry=0.1,
                          lambda_multiplet=0.1, lambda_disjoint=0.1, lambda_gt_baseline=0.1)
    base, _ = crit(pred, s, a, x)
    perm, _ = crit(pred[:, torch.tensor([3, 0, 4, 1, 2])], s, a, x)
    assert torch.allclose(base, perm, atol=1e-6)


def test_physics_loss_separates_ghosts_far_more_than_pit():
    """Same predictions: the ghost-contaminated one costs relatively much more under PhysicsPIT."""
    x, s, a = sampler_batch()
    ghost = s.clone()
    ghost[:, 0] += 0.1 * s[:, 1]
    ghost[:, 2] += 0.1 * s[:, 3]
    pit = PITLoss()
    phys = PhysicsPITLoss(lambda_sparse=0.05, lambda_tv=0.05)
    pit_gap = (pit(ghost, s, a, x)[0] - pit(s, s, a, x)[0]).item()
    phys_gap = (phys(ghost, s, a, x)[0] - phys(s, s, a, x)[0]).item()
    assert phys_gap > 5 * pit_gap


def test_gt_baseline_control_arm_targets_baseline_only():
    x, s, a = sampler_batch()
    pred_ghost = s.clone()
    pred_ghost[:, 0] += 0.1 * s[:, 1]
    crit = PhysicsPITLoss(lambda_gt_baseline=1.0)
    _, p_clean = crit(s, s, a, x)
    _, p_ghost = crit(pred_ghost, s, a, x)
    assert p_clean["gtb"] < p_ghost["gtb"]


def test_physics_loss_backprop_through_model_all_terms_enabled():
    x, s, a = sampler_batch()
    model = build_model("v1")
    crit = PhysicsPITLoss(lambda_sparse=0.05, lambda_tv=0.05, lambda_linewidth=0.01, lambda_symmetry=0.01,
                          lambda_multiplet=0.01, lambda_disjoint=0.01, lambda_gt_baseline=0.01)
    loss, parts = crit(model(x), s, a, x)
    assert np.isfinite(parts["loss"]) and all(np.isfinite(v) for v in parts.values())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_one_optimizer_step_with_physics_loss_is_finite():
    x, s, a = sampler_batch()
    model = build_model("v1")
    crit = PhysicsPITLoss(lambda_sparse=0.05, lambda_tv=0.05, lambda_linewidth=0.01, lambda_symmetry=0.01)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(2):
        loss, _ = crit(model(x), s, a, x)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


# ----------------------------------------------------------------------------- CLI / trainer integration
def _parse(argv):
    p = argparse.ArgumentParser()
    add_physics_args(p)
    p.add_argument("--lambda-recon", type=float, default=0.5)
    return p.parse_args(argv)


def test_build_criterion_from_flags():
    assert type(build_criterion(_parse([]))) is PITLoss
    crit = build_criterion(_parse(["--loss", "physics_pit", "--lambda-sparse", "0.3", "--lambda-linewidth", "0.2"]))
    assert isinstance(crit, PhysicsPITLoss)
    assert crit.lam["sparse"] == 0.3 and crit.lam["lw"] == 0.2


def test_warmup_ramp():
    assert warmup_ramp(0, 5) == 0.0 and warmup_ramp(5, 5) == 1.0 and warmup_ramp(2, 0) == 1.0
    assert 0.0 < warmup_ramp(2, 5) < 1.0


def test_train_run_epoch_reports_physics_terms():
    import train as train_script

    x, s, a = sampler_batch()
    loader = [{"mixtures": x, "compounds": s, "concentrations": a}]
    crit = PhysicsPITLoss(lambda_sparse=0.05, lambda_tv=0.05)
    out = train_script.run_epoch(build_model("v1"), loader, crit, torch.device("cpu"))
    assert {"loss", "spec", "recon", "sparse", "tv"} <= set(out)
    out_pit = train_script.run_epoch(build_model("v1"), loader, PITLoss(), torch.device("cpu"))
    assert set(out_pit) == {"loss", "spec", "recon"}


# ----------------------------------------------------------------------------- data generator flag
def test_binomial_flag_default_off_is_legacy_flat():
    flat = lorentz_multiplet(PPM, 5.0, 3, 7.0, 0.004, 1.0)
    legacy = sum(1.0 / 3 * (0.002) ** 2 / ((PPM - 5.0 - off) ** 2 + 0.002 ** 2) for off in (np.arange(3) - 1.0) * 7.0 / 400.0)
    assert np.array_equal(flat, legacy)
    unseen_a = make_unseen_library(3, seed=2)
    unseen_b = make_unseen_library(3, seed=2, binomial=False)
    assert np.array_equal(unseen_a, unseen_b)


def test_binomial_multiplet_has_pascal_line_heights():
    from scipy.signal import find_peaks

    for n in (2, 3, 4):
        m = lorentz_multiplet(PPM, 5.0, n, 8.0, 0.004, 1.0, binomial=True)
        pk, _ = find_peaks(m)
        h = m[pk] / m[pk].sum()
        assert np.allclose(h, pascal_weights(n), atol=0.01)


# ----------------------------------------------------------------------------- ghost metrics
def test_ghost_metrics_detect_spurious_peaks_that_pearson_hides():
    from src.evaluation.baselines import matched_correlation

    truth = np.stack([line([1.2, 3.6]), line([7.1]), line([2.1, 4.1]), line([6.0]), line([8.5])])
    truth = truth / truth.max(axis=1, keepdims=True)
    pred = truth.copy()
    pred[0] += 0.08 * truth[1]
    pred[2] += 0.08 * truth[3]
    perfect = ghost_metrics(truth, truth)
    ghosts = ghost_metrics(truth, pred)
    assert perfect["peak_f1"] == 1.0 and perfect["n_spurious_per_source"] == 0.0
    assert ghosts["n_spurious_per_source"] > 0 and ghosts["peak_precision"] < 1.0
    assert ghosts["ghost_mass_fraction"] > perfect["ghost_mass_fraction"]
    assert matched_correlation(truth, pred).mean() > 0.97      # Pearson barely notices
