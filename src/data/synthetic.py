import os
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SPECTRAL_LENGTH = 16384
NUM_MIXTURES = 20
NUM_COMPOUNDS = 5


def load_library(data_dir: str) -> Tuple[List[str], np.ndarray]:
    comp_dir = os.path.join(data_dir, "components", "npy")
    files = sorted(f for f in os.listdir(comp_dir) if f.endswith(".npy"))
    names = [f[:-4] for f in files]
    spectra = np.stack([np.load(os.path.join(comp_dir, f)).astype(np.float32) for f in files])
    spectra = spectra / spectra.max(axis=1, keepdims=True)
    return names, spectra


def split_compounds(
    num_compounds: int,
    n_val: int = 6,
    n_test: int = 7,
    seed: int = 0,
    min_pool: int = NUM_COMPOUNDS,
) -> Dict[str, np.ndarray]:
    if min(n_val, n_test) < min_pool or num_compounds - n_val - n_test < min_pool:
        raise ValueError(f"every split needs at least {min_pool} compounds to draw a mixture from")
    perm = np.random.RandomState(seed).permutation(num_compounds)
    return {
        "test": np.sort(perm[:n_test]),
        "val": np.sort(perm[n_test:n_test + n_val]),
        "train": np.sort(perm[n_test + n_val:]),
    }


def sample_concentrations(
    rng: np.random.RandomState, m: int, n: int, alpha: float = 1.0, min_fraction: float = 0.02
) -> np.ndarray:
    a = rng.dirichlet(np.full(n, alpha), size=m)
    a = np.maximum(a, min_fraction)
    return (a / a.sum(axis=1, keepdims=True)).astype(np.float32)


def make_mixtures(
    compounds: np.ndarray,
    rng: np.random.RandomState,
    num_mixtures: int = NUM_MIXTURES,
    noise_std: float = 0.0,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    a = sample_concentrations(rng, num_mixtures, compounds.shape[0], alpha)
    x = a @ compounds
    if noise_std > 0:
        x = np.maximum(x + rng.normal(0.0, noise_std, x.shape), 0.0)
    return x.astype(np.float32), a


class MixtureSampler:
    """
    Draws random compound subsets from a restricted pool, in random slot order,
    and mixes them linearly with Dirichlet concentrations. Targets and inputs
    share one scale, so A @ S == X holds exactly in the noise-free case.
    """

    def __init__(
        self,
        library: np.ndarray,
        pool: Sequence[int],
        num_compounds: int = NUM_COMPOUNDS,
        num_mixtures: int = NUM_MIXTURES,
        noise_std: float = 0.0,
        alpha: float = 1.0,
        seed: int = 0,
    ):
        self.library = library
        self.pool = np.asarray(pool)
        if len(self.pool) < num_compounds:
            raise ValueError("pool smaller than num_compounds")
        self.num_compounds = num_compounds
        self.num_mixtures = num_mixtures
        self.noise_std = noise_std
        self.alpha = alpha
        self.seed = seed

    def sample(self, index: int) -> Dict[str, np.ndarray]:
        rng = np.random.RandomState(self.seed * 1_000_003 + index)
        ids = rng.choice(self.pool, size=self.num_compounds, replace=False)
        s = self.library[ids]
        x, a = make_mixtures(s, rng, self.num_mixtures, self.noise_std, self.alpha)
        return {"mixtures": x, "compounds": s, "concentrations": a, "compound_ids": ids}


def pascal_weights(n_lines: int) -> np.ndarray:
    """Binomial (Pascal-triangle) line weights for a first-order multiplet, summing to 1."""
    w = np.array([comb(n_lines - 1, k) for k in range(n_lines)], dtype=np.float64)
    return w / w.sum()


def lorentz_multiplet(ppm: np.ndarray, center: float, n_lines: int, j_hz: float,
                      width_ppm: float, height: float, mhz: float = 400.0,
                      binomial: bool = False) -> np.ndarray:
    """
    Sum of Lorentzian lines spaced by J. Default (binomial=False) gives every line
    the same height, which is the legacy behaviour all existing benchmarks use.
    With binomial=True the line heights follow Pascal's triangle (1:2:1, 1:3:3:1, ...),
    as for a first-order multiplet. Total height (sum of line heights) is `height` either way.
    """
    spacing = j_hz / mhz
    offsets = (np.arange(n_lines) - (n_lines - 1) / 2.0) * spacing
    out = np.zeros_like(ppm)
    if not binomial:   # legacy expression kept verbatim so default benchmarks stay bit-identical
        for off in offsets:
            out += height / n_lines * (width_ppm / 2) ** 2 / ((ppm - center - off) ** 2 + (width_ppm / 2) ** 2)
        return out
    for off, w in zip(offsets, pascal_weights(n_lines)):
        out += height * w * (width_ppm / 2) ** 2 / ((ppm - center - off) ** 2 + (width_ppm / 2) ** 2)
    return out


def make_unseen_library(
    num: int = 5, length: int = SPECTRAL_LENGTH, seed: int = 0, width_ppm: float = 0.01,
    binomial: bool = False,
) -> np.ndarray:
    """
    Novel compounds with random Lorentzian multiplets (J-couplings in Hz at 400 MHz),
    linewidth matched to the training library's order of magnitude.
    """
    rng = np.random.RandomState(seed)
    ppm = np.linspace(10.0, 0.0, length)
    out = np.zeros((num, length), dtype=np.float32)
    for k in range(num):
        s = np.zeros(length)
        for _ in range(rng.randint(2, 5)):
            s += lorentz_multiplet(
                ppm,
                center=rng.uniform(0.5, 9.5),
                n_lines=int(rng.choice([1, 2, 3, 4])),
                j_hz=rng.uniform(5.0, 9.0),
                width_ppm=width_ppm * rng.uniform(0.7, 1.5),
                height=rng.uniform(0.3, 1.0),
                binomial=binomial,
            )
        out[k] = s / s.max()
    return out
