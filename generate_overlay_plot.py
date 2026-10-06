"""
Generates publication-quality overlay plot of Ground Truth vs Predicted spectra
from MixNet V1 on test dataset_0831 (Average Correlation: 94.17%).
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models.unet1d import MixNet
from src.data.synthetic import SPECTRAL_LENGTH

PPM = np.linspace(10.0, 0.0, SPECTRAL_LENGTH)

def generate_overlay():
    data_dir = os.path.join(PROJECT_ROOT, "NMR_PROJECT_FINAL_PACKAGE")
    meta = pd.read_csv(os.path.join(data_dir, "mixture_metadata.csv"))
    
    # Dataset 0831 from Test Split (avg correlation = 94.17%)
    ds_id = "dataset_0831"
    ds_meta = meta[meta["dataset_id"] == ds_id].reset_index(drop=True)
    row0 = ds_meta.iloc[0]
    comp_names = [row0[f"compound_{k}"] for k in range(1, 6)]
    
    comp_spectra = [np.load(os.path.join(data_dir, "components", "npy", f"{name}.npy")).astype(np.float32) for name in comp_names]
    comp_spectra = [s / s.max() for s in comp_spectra]
    true_spectra = np.stack(comp_spectra, axis=0) # (5, 16384)
    
    npz = np.load(os.path.join(data_dir, "mixtures", f"{ds_id}.npz"))
    mixtures = npz["spectra"].astype(np.float32)
    mix_max = mixtures.max(axis=1, keepdims=True)
    mix_max = np.where(mix_max > 0, mix_max, 1.0)
    mixtures = mixtures / mix_max # (20, 16384)
    
    # Load Model
    model = MixNet(in_channels=20, out_channels=5)
    ckpt = torch.load(os.path.join(PROJECT_ROOT, "checkpoints", "best_model_real.pth"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        x = torch.from_numpy(mixtures).unsqueeze(0).float()
        pred = model(x).squeeze(0).numpy() # (5, 16384)
        
    cost = -((true_spectra - true_spectra.mean(1, keepdims=True)) @ 
             (pred - pred.mean(1, keepdims=True)).T)
    r_idx, c_idx = linear_sum_assignment(cost)
    matched_pred = pred[c_idx]
    
    # 5 Subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    plt.subplots_adjust(hspace=0.38)
    
    for i in range(5):
        ax = axes[i]
        true_s = true_spectra[i]
        pred_s = matched_pred[i]
        
        t_c = true_s - true_s.mean()
        p_c = pred_s - pred_s.mean()
        corr = np.dot(t_c, p_c) / (np.linalg.norm(t_c) * np.linalg.norm(p_c) + 1e-12)
        
        comp_display = comp_names[i].replace("_", " ")
        
        ax.plot(PPM, true_s, label="Ground Truth (Pure Spectrum)", color="#111111", lw=1.6, alpha=0.9)
        ax.plot(PPM, pred_s, label=f"Predicted by MixNet V1", color="#d62828", lw=1.3, linestyle="--", alpha=0.95)
        
        ax.set_title(f"Component {i+1}: {comp_display}   |   Pearson Correlation: {corr*100:.2f}%", 
                     fontsize=12, fontweight="bold", loc="left", color="#1d3557")
        ax.set_ylabel("Intensity", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True, fontsize=10)
        ax.set_xlim(10.0, 0.0) # Invert PPM
        ax.set_ylim(-0.05, 1.15)
        
    axes[-1].set_xlabel("Chemical Shift $\\delta$ (ppm)", fontsize=12, fontweight="bold")
    
    fig.suptitle(f"MixNet V1 Blind Deconvolution on Test Mixture ({ds_id})\nMean Pearson Correlation: 94.17%", 
                 fontsize=15, fontweight="bold", y=0.995, color="#03045e")
    
    out_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "v1_predicted_vs_ground_truth_overlay.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Overlay plot saved to: {out_path}")

if __name__ == "__main__":
    generate_overlay()
