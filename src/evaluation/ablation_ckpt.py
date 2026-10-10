"""
Resumable checkpointing for ablation.py (torch-free so it is unit-testable anywhere).

One JSON file holds every finished (arm, seed) run: its per-sample metric arrays AND its
run metadata. A restarted job with --resume loads the file and skips finished runs, so a Colab
disconnect costs at most the run in progress. Writes are atomic (tmp file + os.replace), so an
interrupt mid-write can never corrupt the previous checkpoint.

Runs are only reused when the settings that affect training / evaluation are identical; otherwise
resuming would silently mix incompatible runs into one comparison.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np

SCHEMA = 1
# Settings that do NOT change what a single (arm, seed) run produces.
_IGNORED_FOR_COMPAT = {"arms", "seeds", "output", "results_dir", "data_dir", "resume", "nmf_samples"}


def run_key(arm: str, seed: int) -> str:
    return f"{arm}|{seed}"


def compat_signature(args: dict) -> dict:
    """Settings a stored run must match. The compound split depends on seeds[0], so keep it."""
    sig = {k: v for k, v in args.items() if k not in _IGNORED_FOR_COMPAT}
    sig["split_seed"] = args["seeds"][0]
    return json.loads(json.dumps(sig, sort_keys=True))     # normalise tuples/np scalars


def partial_path(out_path: str) -> str:
    return out_path + ".partial"


def save_run(path: str, args: dict, runs: Dict[str, dict]) -> None:
    """runs: run_key -> {"metrics": {name: ndarray}, "meta": dict}. Atomic overwrite."""
    payload = {
        "schema": SCHEMA,
        "signature": compat_signature(args),
        "runs": {
            k: {"metrics": {m: np.asarray(v, dtype=np.float64).tolist() for m, v in r["metrics"].items()},
                "meta": r["meta"]}
            for k, r in runs.items()
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_runs(path: str, args: dict) -> Optional[Dict[str, dict]]:
    """Return finished runs (metrics as ndarrays), or None if no checkpoint exists.
    Raises ValueError if the checkpoint was made with different settings."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported checkpoint schema {payload.get('schema')!r}")
    stored, now = payload["signature"], compat_signature(args)
    diff = sorted(k for k in set(stored) | set(now) if stored.get(k) != now.get(k))
    if diff:
        detail = ", ".join(f"{k}: {stored.get(k)!r} -> {now.get(k)!r}" for k in diff)
        raise ValueError(f"{path}: checkpoint was made with different settings ({detail}). "
                         f"Delete it or drop --resume.")
    return {
        k: {"metrics": {m: np.asarray(v, dtype=np.float64) for m, v in r["metrics"].items()},
            "meta": r["meta"]}
        for k, r in payload["runs"].items()
    }
