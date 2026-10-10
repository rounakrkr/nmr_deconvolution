"""
Ghost-peak metrics. Matched Pearson is dominated by tall peaks and is nearly blind to small
spurious peaks (a 94% Pearson spectrum can still carry several ghosts), so molecule-identification
quality needs peak-level measures:

ghost_mass_fraction : share of predicted spectral mass lying on the ground-truth baseline
                      (far from every true peak). 0 is ideal.
peak precision/recall/F1 : predicted peak list vs true peak list with a ppm tolerance.
n_spurious / n_missed    : peak count errors per source (Sir: "more or fewer peaks than ground
                           truth is highly problematic for molecule identification").
"""
from typing import Dict

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from src.evaluation.baselines import correlation_matrix

PPM_RANGE = 10.0


def _peaks(x: np.ndarray, abs_height: float) -> np.ndarray:
    pk, _ = find_peaks(x, height=abs_height)
    return pk


def ghost_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    tol_ppm: float = 0.015,
    peak_height: float = 0.05,
    baseline_threshold: float = 0.02,
    baseline_margin_ppm: float = 0.06,
) -> Dict[str, float]:
    """
    true, pred: (N, L). Spectra are assumed on a 10 -> 0 ppm axis and scaled so true max is ~1.
    Sources are matched with the same Hungarian-on-correlation rule as `matched_correlation`.
    """
    n, length = true.shape
    dppm = PPM_RANGE / (length - 1)
    tol = max(int(round(tol_ppm / dppm)), 1)
    margin = max(int(round(baseline_margin_ppm / dppm)), 1)
    r, c = linear_sum_assignment(-correlation_matrix(true, pred))
    pred = pred[c]
    true = true[r]

    tp = fp = fn = 0
    ghost_mass = total_mass = 0.0
    for t, p in zip(true, pred):
        near = maximum_filter1d((t > baseline_threshold).astype(np.uint8), 2 * margin + 1) > 0
        ghost_mass += float(p[~near].clip(min=0).sum())
        total_mass += float(p.clip(min=0).sum())
        tpk, ppk = _peaks(t, peak_height * max(t.max(), 1e-12)), _peaks(p, peak_height)
        used = set()
        for q in tpk:
            if len(ppk) == 0:
                break
            d = np.abs(ppk - q)
            j = int(np.argmin(d))
            if d[j] <= tol and j not in used:
                used.add(j)
        tp += len(used)
        fp += len(ppk) - len(used)
        fn += len(tpk) - len(used)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "ghost_mass_fraction": ghost_mass / max(total_mass, 1e-12),
        "peak_precision": prec, "peak_recall": rec, "peak_f1": f1,
        "n_spurious_per_source": fp / n, "n_missed_per_source": fn / n,
    }
