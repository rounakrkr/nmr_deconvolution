from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


def nmf_separate(
    x: np.ndarray, rank: int, iters: int = 800, restarts: int = 3, seed: int = 0, eps: float = 1e-12
) -> np.ndarray:
    """Multiplicative-update NMF (Frobenius). x: (M, L) -> sources (rank, L)."""
    rng = np.random.RandomState(seed)
    x = x.astype(np.float64)
    best_err, best_h = np.inf, None
    for _ in range(restarts):
        w = rng.rand(x.shape[0], rank) + 1e-3
        h = rng.rand(rank, x.shape[1]) + 1e-3
        for _ in range(iters):
            h *= (w.T @ x) / (w.T @ w @ h + eps)
            w *= (x @ h.T) / (w @ h @ h.T + eps)
        err = np.linalg.norm(x - w @ h)
        if err < best_err:
            best_err, best_h = err, h
    return best_h.astype(np.float32)


def mean_spectrum_baseline(x: np.ndarray, rank: int) -> np.ndarray:
    return np.tile(x.mean(axis=0, keepdims=True), (rank, 1))


def correlation_matrix(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    t = true - true.mean(axis=1, keepdims=True)
    p = pred - pred.mean(axis=1, keepdims=True)
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    pn = np.linalg.norm(p, axis=1, keepdims=True)
    return (t / np.maximum(tn, 1e-12)) @ (p / np.maximum(pn, 1e-12)).T


def matched_correlation(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Hungarian-matched Pearson correlation per true source, shape (N,)."""
    c = correlation_matrix(true, pred)
    r, col = linear_sum_assignment(-c)
    out = np.zeros(true.shape[0])
    out[r] = c[r, col]
    return out


def slot_correlation(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Fixed-slot (no matching) Pearson correlation per slot, shape (N,)."""
    t = true - true.mean(axis=1, keepdims=True)
    p = pred - pred.mean(axis=1, keepdims=True)
    num = (t * p).sum(axis=1)
    den = np.linalg.norm(t, axis=1) * np.linalg.norm(p, axis=1)
    return num / np.maximum(den, 1e-12)


def summarize(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64).ravel()
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return {"mean": float(v.mean()), "ci95": float(1.96 * se), "min": float(v.min()), "max": float(v.max()), "n": n}
