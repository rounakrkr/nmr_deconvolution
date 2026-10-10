import json
import os

import numpy as np
import pytest

from src.evaluation.ablation_ckpt import compat_signature, load_runs, run_key, save_run

ARGS = {"model": "v1", "epochs": 40, "batch_size": 4, "lr": 1e-4, "train_size": 400, "seeds": [0, 1],
        "arms": ["pit", "physics"], "output": None, "results_dir": "/x", "data_dir": "/d", "resume": False,
        "nmf_samples": 30, "lambda_sparse": 0.05}


def _runs():
    return {run_key("pit", 0): {"metrics": {"matched_pearson": np.array([0.8, 0.9]), "peak_f1": np.array([0.5, 0.6])},
                                "meta": {"seed": 0, "best_epoch": 7, "epochs_run": 19, "last_val": {"loss": 0.1}}}}


def test_roundtrip_preserves_arrays_and_meta(tmp_path):
    p = str(tmp_path / "a.json.partial")
    save_run(p, ARGS, _runs())
    got = load_runs(p, ARGS)
    r = got[run_key("pit", 0)]
    np.testing.assert_allclose(r["metrics"]["matched_pearson"], [0.8, 0.9])
    assert r["meta"]["best_epoch"] == 7
    assert not os.path.exists(p + ".tmp")           # atomic write leaves no tmp file


def test_missing_checkpoint_returns_none(tmp_path):
    assert load_runs(str(tmp_path / "nope.partial"), ARGS) is None


def test_resume_may_change_arms_seeds_paths_but_not_training_settings(tmp_path):
    p = str(tmp_path / "a.partial")
    save_run(p, ARGS, _runs())
    ok = dict(ARGS, arms=["pit", "physics", "gt_baseline"], output="/y", results_dir="/z", resume=True,
              nmf_samples=0, seeds=[0, 1, 2])
    assert load_runs(p, ok) is not None             # more arms / seeds is the intended use
    for bad in (dict(ARGS, lambda_sparse=0.5), dict(ARGS, epochs=60), dict(ARGS, seeds=[5, 1])):
        with pytest.raises(ValueError):             # different weights / epochs / compound split
            load_runs(p, bad)


def test_failed_write_keeps_previous_checkpoint(tmp_path, monkeypatch):
    p = str(tmp_path / "a.partial")
    save_run(p, ARGS, _runs())
    before = open(p).read()
    monkeypatch.setattr(json, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        save_run(p, ARGS, _runs())
    assert open(p).read() == before                 # old checkpoint untouched


def test_signature_tracks_split_seed():
    assert compat_signature(ARGS)["split_seed"] == 0
