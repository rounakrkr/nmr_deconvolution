"""
Quick integration test — verify the full pipeline works end-to-end.
Trains on a tiny subset for a few epochs to verify loss decreases.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.data.dataset import NMRMixtureDataset, collate_fn
from src.models.unet1d import MixNet
from src.training.losses import CombinedLoss

def main():
    print("=" * 60)
    print("  INTEGRATION TEST — Quick Overfit Check")
    print("=" * 60)
    
    # Load config
    with open("src/configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load tiny subset (just 8 samples)
    dataset = NMRMixtureDataset("data/dummy/train")
    tiny_dataset = Subset(dataset, range(8))
    loader = DataLoader(tiny_dataset, batch_size=4, collate_fn=collate_fn)
    
    # Get first batch to verify shapes
    batch = next(iter(loader))
    max_m = dataset.global_max_m
    max_n = dataset.global_max_n
    L = batch['mixtures'].shape[2]
    
    print(f"Batch shapes: mixtures={batch['mixtures'].shape}, compounds={batch['compounds'].shape}")
    print(f"Global Max M={max_m}, Max N={max_n}, L={L}")
    
    # Create model
    model = MixNet(
        in_channels=max_m,
        out_channels=max_n,
        encoder_channels=config['model']['encoder_channels']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    
    # Loss & optimizer
    criterion = CombinedLoss(lambda_spectral=1.0, lambda_reconstruction=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Quick training — 10 steps
    print("\nTraining for 10 steps (overfit test)...")
    model.train()
    losses = []
    
    for step in range(10):
        for batch in loader:
            mixtures = batch['mixtures'].to(device)
            compounds = batch['compounds'].to(device)
            concentrations = batch['concentrations'].to(device)
            comp_mask = batch['compound_mask'].to(device)
            mix_mask = batch['mixture_mask'].to(device)
            
            optimizer.zero_grad()
            pred = model(mixtures)
            loss, loss_dict = criterion(pred, compounds, concentrations, mixtures, comp_mask, mix_mask)
            loss.backward()
            optimizer.step()
            
            losses.append(loss_dict['loss'])
            
        print(f"  Step {step+1}/10 — Loss: {losses[-1]:.6f}")
    
    # Check if loss decreased
    first_loss = losses[0]
    last_loss = losses[-1]
    decreased = last_loss < first_loss
    
    print(f"\n{'='*40}")
    print(f"First loss: {first_loss:.6f}")
    print(f"Last loss:  {last_loss:.6f}")
    print(f"Decreased:  {'YES!' if decreased else 'NO'}")
    
    if decreased:
        print("\nINTEGRATION TEST PASSED!")
        print("The full pipeline works: data -> model -> loss -> backprop -> learning")
    else:
        print("\nWARNING: Loss didn't decrease -- may need more steps or debugging")
    
    # Quick output check
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        mixtures = batch['mixtures'].to(device)
        pred = model(mixtures)
        print(f"\nOutput stats:")
        print(f"  Min: {pred.min().item():.4f}")
        print(f"  Max: {pred.max().item():.4f}")
        print(f"  Mean: {pred.mean().item():.4f}")
        print(f"  All non-negative: {(pred >= 0).all().item()}")

if __name__ == "__main__":
    main()
