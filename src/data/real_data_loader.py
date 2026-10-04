"""
Real NMR Data Loader for Sir's CSV format.

Expected CSV format:
    Column 0 (index): row number
    Column 'ppm': chemical shift values (high to low, e.g. 10.0 to -1.0)
    Column 'replicate_1' ... 'replicate_5': 5 repeated measurements of the mixture
    Column 'replicate_mean': average of all 5 replicates

Usage:
    from src.data.real_data_loader import load_mixture_csv, load_mixture_csv_as_tensors
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Tuple, Optional
import torch


def load_mixture_csv(
    csv_path: str,
    ppm_min: float = 0.0,
    ppm_max: float = 10.0,
    target_length: int = 16384,
    use_replicates: bool = True,
    clip_negative: bool = True,
) -> dict:
    """
    Load a mixture NMR spectrum from sir's CSV format.

    Steps:
        1. Read CSV
        2. Crop to [ppm_min, ppm_max]
        3. Resample to target_length via linear interpolation
        4. Optionally clip negative values to 0 (instrument noise artefacts)

    Args:
        csv_path: Path to the CSV file.
        ppm_min: Lower bound of ppm range to keep (default 0.0).
        ppm_max: Upper bound of ppm range to keep (default 10.0).
        target_length: Number of points in the output spectrum (default 16384).
        use_replicates: If True, return all 5 replicates as separate rows.
                        If False, return only replicate_mean.
        clip_negative: If True, clip small negative values to 0.

    Returns:
        dict with keys:
            'ppm_axis'    : np.ndarray, shape (target_length,)  — uniform ppm grid
            'mean'        : np.ndarray, shape (target_length,)  — replicate mean
            'replicates'  : np.ndarray, shape (5, target_length) — 5 replicates
                           (only if use_replicates=True)
            'original_length': int — number of points before resampling
            'ppm_range'   : tuple — actual (min, max) used
    """
    df = pd.read_csv(csv_path)

    # --- Validate columns ---
    required = ['ppm', 'replicate_mean']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. Columns: {list(df.columns)}")

    ppm_raw = df['ppm'].values
    mean_raw = df['replicate_mean'].values

    replicate_cols = [c for c in df.columns if c.startswith('replicate_') and c != 'replicate_mean']
    replicates_raw = df[replicate_cols].values  # shape (N_points, n_replicates)

    # --- Crop to [ppm_min, ppm_max] ---
    # Note: ppm axis in NMR goes high→low (reversed), so we handle both directions
    if ppm_raw[0] > ppm_raw[-1]:
        # Descending: flip to ascending for interpolation, flip back after
        ppm_raw = ppm_raw[::-1].copy()
        mean_raw = mean_raw[::-1].copy()
        replicates_raw = replicates_raw[::-1, :].copy()

    mask = (ppm_raw >= ppm_min) & (ppm_raw <= ppm_max)
    ppm_crop = ppm_raw[mask]
    mean_crop = mean_raw[mask]
    replicates_crop = replicates_raw[mask, :]

    original_length = len(ppm_crop)

    # --- Resample to target_length via linear interpolation ---
    # New uniform grid from ppm_min to ppm_max
    ppm_uniform = np.linspace(ppm_min, ppm_max, target_length)

    interp_mean = interp1d(ppm_crop, mean_crop, kind='linear',
                           bounds_error=False, fill_value=0.0)
    mean_resampled = interp_mean(ppm_uniform)

    replicates_resampled = np.zeros((replicates_crop.shape[1], target_length), dtype=np.float32)
    for i in range(replicates_crop.shape[1]):
        interp_rep = interp1d(ppm_crop, replicates_crop[:, i], kind='linear',
                              bounds_error=False, fill_value=0.0)
        replicates_resampled[i] = interp_rep(ppm_uniform)

    # --- Clip negatives (instrument baseline noise) ---
    if clip_negative:
        mean_resampled = np.clip(mean_resampled, 0.0, None)
        replicates_resampled = np.clip(replicates_resampled, 0.0, None)

    return {
        'ppm_axis': ppm_uniform.astype(np.float32),
        'mean': mean_resampled.astype(np.float32),
        'replicates': replicates_resampled if use_replicates else None,
        'original_length': original_length,
        'ppm_range': (ppm_min, ppm_max),
        'n_replicates': replicates_resampled.shape[0],
    }


def load_mixture_csv_as_tensors(
    csv_path: str,
    ppm_min: float = 0.0,
    ppm_max: float = 10.0,
    target_length: int = 16384,
    clip_negative: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load sir's CSV and return as PyTorch tensors, ready for model input.

    Returns:
        mixtures : torch.Tensor, shape (n_replicates, target_length)
                   — each row is one replicate (use as M different mixture measurements)
        ppm_axis : torch.Tensor, shape (target_length,)
    """
    data = load_mixture_csv(
        csv_path=csv_path,
        ppm_min=ppm_min,
        ppm_max=ppm_max,
        target_length=target_length,
        use_replicates=True,
        clip_negative=clip_negative,
    )

    mixtures = torch.from_numpy(data['replicates'])  # (n_replicates, L)
    ppm_axis = torch.from_numpy(data['ppm_axis'])    # (L,)

    return mixtures, ppm_axis


def normalize_spectrum(spectrum: np.ndarray, method: str = 'max') -> np.ndarray:
    """
    Normalize a spectrum.

    Args:
        spectrum: 1D or 2D (N, L) array.
        method: 'max' (divide by max), 'minmax' (0-1 scale), 'l2' (unit norm).

    Returns:
        Normalized array, same shape.
    """
    eps = 1e-8
    if method == 'max':
        m = spectrum.max(axis=-1, keepdims=True)
        return spectrum / (m + eps)
    elif method == 'minmax':
        lo = spectrum.min(axis=-1, keepdims=True)
        hi = spectrum.max(axis=-1, keepdims=True)
        return (spectrum - lo) / (hi - lo + eps)
    elif method == 'l2':
        norm = np.linalg.norm(spectrum, axis=-1, keepdims=True)
        return spectrum / (norm + eps)
    else:
        raise ValueError(f"Unknown method: {method}")


# ─── Quick verification ───────────────────────────────────────────────────────
if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    CSV_PATH = os.path.join(REPO_ROOT, "continuous_simulated_spectra_without_peak_shift.csv")

    print("Loading CSV...")
    data = load_mixture_csv(CSV_PATH, target_length=16384)

    print(f"\n=== LOADED DATA ===")
    print(f"PPM axis   : {data['ppm_axis'].shape}  range [{data['ppm_axis'].min():.2f}, {data['ppm_axis'].max():.2f}]")
    print(f"Mean       : {data['mean'].shape}  min={data['mean'].min():.4f}  max={data['mean'].max():.4f}")
    print(f"Replicates : {data['replicates'].shape}")
    print(f"Original points (before resample): {data['original_length']}")
    print(f"All non-negative (mean): {(data['mean'] >= 0).all()}")

    # Test as tensors
    mix_t, ppm_t = load_mixture_csv_as_tensors(CSV_PATH, target_length=16384)
    print(f"\nAs tensors  : mixtures={mix_t.shape}  ppm={ppm_t.shape}  dtype={mix_t.dtype}")

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Sir\'s Mixture CSV — loaded at 16384 points', fontsize=14)

    for i in range(data['replicates'].shape[0]):
        axes[0].plot(data['ppm_axis'], data['replicates'][i], alpha=0.6, label=f'Replicate {i+1}')
    axes[0].set_title('5 Replicates')
    axes[0].set_ylabel('Intensity')
    axes[0].legend(fontsize=8)
    axes[0].invert_xaxis()

    axes[1].plot(data['ppm_axis'], data['mean'], color='black', linewidth=1.5)
    axes[1].set_title('Replicate Mean')
    axes[1].set_xlabel('Chemical Shift (ppm)')
    axes[1].set_ylabel('Intensity')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "real_data_preview.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
