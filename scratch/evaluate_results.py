"""Evaluate trained MixNet on test data and show results."""
import os, sys, torch, numpy as np
sys.path.insert(0, r'c:\Extra Programs\Files\BioTech')

from src.models.unet1d import MixNet
from src.data.real_nmr_loader import RealNMRDataset

# Load trained model
model = MixNet(in_channels=20, out_channels=5)
ckpt = torch.load(r'c:\Extra Programs\Files\BioTech\checkpoints\best_model_real.pth', 
                  map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load dataset
data_dir = r'c:\Extra Programs\Files\BioTech\NMR_PROJECT_FINAL_PACKAGE'
dataset = RealNMRDataset(data_dir)

# Use test split (same seed as training)
np.random.seed(42)
indices = np.random.permutation(len(dataset))
test_indices = indices[900:]  # Last 100 = test

print("=" * 60)
print("  MixNet EVALUATION ON TEST DATA")
print("=" * 60)
print("Best epoch:", ckpt["epoch"], "| Val loss:", round(ckpt["val_loss"], 6))
print()

all_corrs = []
all_mses = []

for t_idx in test_indices[:20]:  # First 20 test datasets
    sample = dataset[t_idx]
    mixtures = sample['mixtures'].unsqueeze(0)
    true_compounds = sample['compounds']
    concentrations = sample['concentrations']
    
    with torch.no_grad():
        pred_compounds = model(mixtures).squeeze(0)
    
    for i in range(5):
        true_i = true_compounds[i].numpy()
        pred_i = pred_compounds[i].numpy()
        corr = float(np.corrcoef(true_i, pred_i)[0, 1])
        mse_val = float(np.mean((true_i - pred_i) ** 2))
        all_corrs.append(corr)
        all_mses.append(mse_val)

print("Tested on 20 datasets (100 compound predictions)")
print("-" * 50)
print("Correlation (predicted vs true compound spectra):")
print("  Min:  ", round(min(all_corrs), 6))
print("  Mean: ", round(np.mean(all_corrs), 6))
print("  Max:  ", round(max(all_corrs), 6))
print()
print("MSE (predicted vs true):")
print("  Min:  ", round(min(all_mses), 6))
print("  Mean: ", round(np.mean(all_mses), 6))
print("  Max:  ", round(max(all_mses), 6))
print()

# Detailed look at 1 dataset
sample = dataset[test_indices[0]]
ds_id = sample['dataset_id']
mixtures = sample['mixtures'].unsqueeze(0)
true_compounds = sample['compounds']

with torch.no_grad():
    pred_compounds = model(mixtures).squeeze(0)

print("=" * 60)
print("  DETAILED: " + ds_id)
print("=" * 60)
for i in range(5):
    true_i = true_compounds[i].numpy()
    pred_i = pred_compounds[i].numpy()
    corr = float(np.corrcoef(true_i, pred_i)[0, 1])
    mse_val = float(np.mean((true_i - pred_i) ** 2))
    peak_true = int(np.sum(true_i > 0.1))
    peak_pred = int(np.sum(pred_i > 0.1))
    print("  Compound " + str(i+1) + ": corr=" + str(round(corr, 4)) + 
          " mse=" + str(round(mse_val, 6)) + 
          " peaks_true=" + str(peak_true) + " peaks_pred=" + str(peak_pred))
