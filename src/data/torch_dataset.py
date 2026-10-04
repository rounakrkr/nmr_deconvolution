from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.synthetic import MixtureSampler


class SyntheticNMRDataset(Dataset):
    def __init__(
        self,
        library: np.ndarray,
        pool: Sequence[int],
        length: int,
        noise_std: float = 0.0,
        seed: int = 0,
        resample_each_epoch: bool = False,
    ):
        self.sampler = MixtureSampler(library, pool, noise_std=noise_std, seed=seed)
        self.length = length
        self.resample_each_epoch = resample_each_epoch
        self.offset = 0

    def set_epoch(self, epoch: int) -> None:
        self.offset = epoch * self.length if self.resample_each_epoch else 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        d = self.sampler.sample(self.offset + index)
        return {
            "mixtures": torch.from_numpy(d["mixtures"]),
            "compounds": torch.from_numpy(d["compounds"]),
            "concentrations": torch.from_numpy(d["concentrations"]),
        }
