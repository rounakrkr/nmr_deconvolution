"""
Real NMR Dataset Loader for MixNet Training.

Loads the NMR_PROJECT_FINAL_PACKAGE data:
  - 1000 datasets, each with 20 mixtures of 5 compounds
  - 30 unique compounds total
  - 16384 spectral points per spectrum

Each "sample" is one DATASET (20 mixtures → 5 compound spectra).
  Input:  (20, 16384) = 20 mixture spectra
  Target: (5, 16384)  = 5 pure compound spectra
  Concentrations: (20, 5) = mixing matrix
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class RealNMRDataset(Dataset):
    """
    Each sample = 1 dataset = 20 mixtures of 5 compounds.
    
    Returns:
        mixtures: (20, 16384) tensor — mixture spectra (INPUT)
        compounds: (5, 16384) tensor — pure compound spectra (TARGET/ANSWER)
        concentrations: (20, 5) tensor — mixing matrix A
        dataset_id: string — identifier
    """
    
    def __init__(self, data_dir, metadata_csv=None, compound_dir=None, 
                 normalize=True, max_datasets=None):
        """
        Args:
            data_dir: Path to NMR_PROJECT_FINAL_PACKAGE folder
            metadata_csv: Path to mixture_metadata.csv (default: inside data_dir)
            compound_dir: Path to components/npy/ folder (default: inside data_dir)
            normalize: If True, normalize spectra to [0, 1] range
            max_datasets: If set, only load this many datasets (for quick testing)
        """
        self.data_dir = data_dir
        self.normalize = normalize
        
        # Metadata
        if metadata_csv is None:
            metadata_csv = os.path.join(data_dir, 'mixture_metadata.csv')
        self.metadata = pd.read_csv(metadata_csv)
        
        # Compound spectra directory
        if compound_dir is None:
            compound_dir = os.path.join(data_dir, 'components', 'npy')
        
        # Load all 30 pure compound spectra into memory (small — ~2 MB)
        self.compound_spectra = {}
        for f in glob.glob(os.path.join(compound_dir, '*.npy')):
            name = os.path.basename(f).replace('.npy', '')
            spec = np.load(f).astype(np.float32)
            if normalize:
                spec = spec / spec.max()  # Normalize to [0, 1]
            self.compound_spectra[name] = spec
        
        # Get unique dataset IDs
        self.dataset_ids = sorted(self.metadata['dataset_id'].unique())
        if max_datasets:
            self.dataset_ids = self.dataset_ids[:max_datasets]
        
        # Pre-group metadata by dataset for fast access
        self.dataset_meta = {}
        for ds_id in self.dataset_ids:
            self.dataset_meta[ds_id] = self.metadata[
                self.metadata['dataset_id'] == ds_id
            ].reset_index(drop=True)
        
        self.spectral_length = 16384
        self.num_mixtures = 20  # per dataset
        self.num_compounds = 5  # per mixture
        
    def __len__(self):
        return len(self.dataset_ids)
    
    def __getitem__(self, idx):
        ds_id = self.dataset_ids[idx]
        ds_meta = self.dataset_meta[ds_id]
        
        # Load mixture spectra from NPZ
        npz_path = os.path.join(self.data_dir, 'mixtures', f'{ds_id}.npz')
        npz_data = np.load(npz_path)
        mixtures = npz_data['spectra'].astype(np.float32)  # (20, 16384)
        
        if self.normalize:
            # Normalize each mixture independently to [0, 1]
            mix_max = mixtures.max(axis=1, keepdims=True)
            mix_max = np.where(mix_max > 0, mix_max, 1.0)  # avoid div by zero
            mixtures = mixtures / mix_max
        
        # Get the 5 compound names for this dataset
        row0 = ds_meta.iloc[0]
        compound_names = [row0[f'compound_{k}'] for k in range(1, 6)]
        
        # Stack pure compound spectra (5, 16384)
        compounds = np.stack([
            self.compound_spectra[name] for name in compound_names
        ], axis=0)
        
        # Build concentration matrix (20, 5)
        concentrations = np.zeros((self.num_mixtures, self.num_compounds), dtype=np.float32)
        for i in range(len(ds_meta)):
            for j in range(self.num_compounds):
                concentrations[i, j] = ds_meta.iloc[i][f'fraction_{j+1}']
        
        return {
            'mixtures': torch.from_numpy(mixtures),           # (20, 16384)
            'compounds': torch.from_numpy(compounds),          # (5, 16384)
            'concentrations': torch.from_numpy(concentrations), # (20, 5)
            'dataset_id': ds_id
        }


def create_data_loaders(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                        batch_size=1, seed=42, max_datasets=None):
    """
    Create train/val/test DataLoaders.
    
    Note: batch_size=1 is recommended because each "sample" is already 
    a full dataset of 20 mixtures. With batch_size=1, each batch has 
    (1, 20, 16384) mixtures.
    
    Args:
        data_dir: Path to NMR_PROJECT_FINAL_PACKAGE
        train_ratio, val_ratio, test_ratio: Split ratios
        batch_size: Batch size (1 = one dataset per batch)
        seed: Random seed for reproducible splits
        max_datasets: Limit total datasets (for testing)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    full_dataset = RealNMRDataset(data_dir, max_datasets=max_datasets)
    total = len(full_dataset)
    
    # Split
    np.random.seed(seed)
    indices = np.random.permutation(total)
    
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_set = torch.utils.data.Subset(full_dataset, train_indices)
    val_set = torch.utils.data.Subset(full_dataset, val_indices)
    test_set = torch.utils.data.Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Dataset splits:")
    print(f"  Total datasets: {total}")
    print(f"  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print(f"  Each dataset = {full_dataset.num_mixtures} mixtures x {full_dataset.spectral_length} points")
    print(f"  Compounds per dataset: {full_dataset.num_compounds}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import time
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'NMR_PROJECT_FINAL_PACKAGE')
    
    print("Loading dataset...")
    t0 = time.time()
    dataset = RealNMRDataset(data_dir, max_datasets=5)
    print(f"Loaded in {time.time()-t0:.2f}s")
    print(f"Datasets: {len(dataset)}")
    print(f"Compounds loaded: {len(dataset.compound_spectra)}")
    
    # Test one sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  mixtures:       {sample['mixtures'].shape}")
    print(f"  compounds:      {sample['compounds'].shape}")
    print(f"  concentrations: {sample['concentrations'].shape}")
    print(f"  dataset_id:     {sample['dataset_id']}")
    
    # Physics check
    recon = sample['concentrations'] @ sample['compounds']  # (20,5) @ (5,16384) = (20,16384)
    corr = float(torch.corrcoef(torch.stack([recon[0], sample['mixtures'][0]]))[0, 1])
    print(f"  Physics check (corr): {corr:.6f}")
