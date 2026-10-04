import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Callable, Any

class NMRMixtureDataset(Dataset):
    """
    PyTorch Dataset for loading NMR mixture data.
    Scans all files on init to determine global max M and N for consistent padding.
    """
    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            data_dir (str): Directory containing .npz files.
            transform (Callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.transform = transform
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")
        
        # Scan all files to find global max M and N
        # This ensures consistent tensor shapes across ALL batches
        self.global_max_m = 0
        self.global_max_n = 0
        for fp in self.file_paths:
            data = np.load(fp)
            m = int(data['num_mixtures'])
            n = int(data['num_compounds'])
            self.global_max_m = max(self.global_max_m, m)
            self.global_max_n = max(self.global_max_n, n)
        
        # Get spectral length from first file
        data = np.load(self.file_paths[0])
        self.spectral_length = data['mixtures'].shape[1]
        
        print(f"  Dataset: {len(self.file_paths)} samples, "
              f"global_max_M={self.global_max_m}, global_max_N={self.global_max_n}, "
              f"L={self.spectral_length}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns pre-padded tensors with consistent shapes across all samples.
        """
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        
        mixtures_raw = data['mixtures']       # (m, L)
        compounds_raw = data['compounds']     # (n, L)
        concentrations_raw = data['concentrations']  # (m, n)
        
        m, L = mixtures_raw.shape
        n = compounds_raw.shape[0]
        
        # Pad to global max dimensions
        mixtures = np.zeros((self.global_max_m, L), dtype=np.float32)
        compounds = np.zeros((self.global_max_n, L), dtype=np.float32)
        concentrations = np.zeros((self.global_max_m, self.global_max_n), dtype=np.float32)
        
        mixtures[:m, :] = mixtures_raw
        compounds[:n, :] = compounds_raw
        concentrations[:m, :n] = concentrations_raw
        
        # Masks: True = valid, False = padding
        mixture_mask = np.zeros(self.global_max_m, dtype=np.float32)
        compound_mask = np.zeros(self.global_max_n, dtype=np.float32)
        mixture_mask[:m] = 1.0
        compound_mask[:n] = 1.0
        
        sample = {
            'mixtures': torch.from_numpy(mixtures),
            'compounds': torch.from_numpy(compounds),
            'concentrations': torch.from_numpy(concentrations),
            'mixture_mask': torch.from_numpy(mixture_mask),
            'compound_mask': torch.from_numpy(compound_mask),
            'num_mixtures': m,
            'num_compounds': n,
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Simple collate — since samples are already padded to global max in __getitem__,
    we just stack them.
    """
    return {
        'mixtures': torch.stack([s['mixtures'] for s in batch]),
        'compounds': torch.stack([s['compounds'] for s in batch]),
        'concentrations': torch.stack([s['concentrations'] for s in batch]),
        'mixture_mask': torch.stack([s['mixture_mask'] for s in batch]),
        'compound_mask': torch.stack([s['compound_mask'] for s in batch]),
    }
