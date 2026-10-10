import argparse
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

torch = pytest.importorskip("torch")

import ablation as ab
from src.training.physics import PhysicsPITLoss
from src.training.pit import PITLoss


def _args(**over):
    d = dict(lambda_recon=0.5, lambda_sparse=0.1, lambda_tv=0.2, lambda_linewidth=0.3, lambda_symmetry=0.4,
             lambda_multiplet=0.5, lambda_disjoint=0.6, lambda_gt_baseline=0.7, sparse_type="log", sparse_eps=0.02,
             lw_tolerance=0.25, lw_scope="source")
    d.update(over)
    return argparse.Namespace(**d)


def test_arm_definitions():
    a = _args()
    assert type(ab.make_criterion("pit", a)) is PITLoss
    phys = ab.make_criterion("physics", a)
    assert phys.lam["sparse"] == 0.1 and phys.lam["tv"] == 0.2 and phys.lam["lw"] == 0.3 and phys.lam["gtb"] == 0
    gt = ab.make_criterion("gt_baseline", a)
    assert gt.lam["gtb"] == 0.7 and all(v == 0 for k, v in gt.lam.items() if k != "gtb")    # control arm uses NO physics term
    for arm in ab.ALL_ARMS:
        assert isinstance(ab.make_criterion(arm, a), PITLoss)
    only = ab.make_criterion("sparse_only", a)
    assert only.lam["sparse"] == 0.1 and sum(v != 0 for v in only.lam.values()) == 1
    with pytest.raises(ValueError):
        ab.make_criterion("nope", a)


def test_paired_delta_and_formatting():
    rng = np.random.RandomState(0)
    ref = {k: rng.rand(20) for k in ab.METRICS}
    better = {k: v - 0.2 if not ab.METRICS[k][1] else v + 0.2 for k, v in ref.items()}
    d = ab.paired_delta(better, ref)
    assert all(abs(abs(d[k]["mean"]) - 0.2) < 1e-9 for k in ab.METRICS)
    table = ab.format_comparison({"pit": ref, "physics": better}, {"ground truth": {k: 1.0 for k in ab.METRICS}})
    assert "pit" in table and "ground truth" in table
    paired = ab.format_paired({"physics": d}, "pit")
    assert paired.count("+") >= len(ab.METRICS)           # every metric flagged as significantly better


def test_per_sample_metrics_ground_truth_is_perfect():
    from src.data.synthetic import MixtureSampler

    lib = np.zeros((6, 2048), dtype=np.float32)
    for k in range(6):
        for c in (200 + 250 * k, 700 + 100 * k):
            lib[k] += np.exp(-0.5 * ((np.arange(2048) - c) / 6.0) ** 2)
        lib[k] /= lib[k].max()
    samples = [MixtureSampler(lib, np.arange(6), num_compounds=5, seed=1).sample(i) for i in range(3)]
    out = ab.per_sample_metrics(samples, lambda d: d["compounds"])
    assert np.allclose(out["matched_pearson"], 1.0) and np.allclose(out["peak_f1"], 1.0)
    assert np.allclose(out["n_spurious_per_source"], 0.0)


def test_end_to_end_tiny_run_writes_json(tmp_path):
    out = tmp_path / "abl.json"
    pooled, paired = ab.main([
        "--model", "v1", "--arms", "pit", "physics", "gt_baseline", "--epochs", "1", "--train-size", "2",
        "--val-size", "2", "--test-size", "2", "--nmf-samples", "2", "--physics-warmup-epochs", "1",
        "--results-dir", str(tmp_path), "--output", str(out),
    ])
    data = json.loads(out.read_text())
    assert set(data["arms"]) == {"pit", "physics", "gt_baseline"}
    assert data["paired_vs_baseline"]["baseline"] == "pit" and set(data["paired_vs_baseline"]["delta"]) == {"physics", "gt_baseline"}
    for arm in data["arms"].values():
        assert set(ab.METRICS) <= set(arm["metrics"]) and np.isfinite(arm["metrics"]["matched_pearson"]["mean"])
    assert "ground_truth" in data["reference"] and data["reference"]["nmf"] is not None
