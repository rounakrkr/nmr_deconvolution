"""
MixNet Training on Real NMR Data.

Usage:
    python train_real.py                    # Full training (for Colab)
    python train_real.py --quick            # Quick test (5 datasets, 3 epochs)
    python train_real.py --max_datasets 50  # Medium test
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models.unet1d import MixNet
from src.data.real_nmr_loader import RealNMRDataset, create_data_loaders


def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch. Returns average losses."""
    model.train()
    total_spec_loss = 0
    total_recon_loss = 0
    total_loss = 0
    batches = 0
    
    mse = nn.MSELoss()
    
    for batch in loader:
        mixtures = batch['mixtures'].to(device)        # (B, 20, 16384)
        compounds = batch['compounds'].to(device)       # (B, 5, 16384)
        concentrations = batch['concentrations'].to(device)  # (B, 20, 5)
        
        # Squeeze batch dim if batch_size=1
        if mixtures.dim() == 3 and mixtures.shape[0] == 1:
            mixtures = mixtures.squeeze(0)        # (20, 16384)
            compounds = compounds.squeeze(0)       # (5, 16384)
            concentrations = concentrations.squeeze(0)  # (20, 5)
            
            # Add batch dim for model
            mixtures = mixtures.unsqueeze(0)       # (1, 20, 16384)
            compounds = compounds.unsqueeze(0)      # (1, 5, 16384)
            concentrations = concentrations.unsqueeze(0)  # (1, 20, 5)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_compounds = model(mixtures)  # (B, 5, 16384)
        
        # Loss 1: Spectral loss (predicted compounds vs true compounds)
        spec_loss = mse(pred_compounds, compounds)
        
        # Loss 2: Reconstruction loss (A @ S_pred should ≈ X)
        recon = torch.bmm(concentrations, pred_compounds)  # (B, 20, 16384)
        recon_loss = mse(recon, mixtures)
        
        # Combined loss
        loss = 1.0 * spec_loss + 0.5 * recon_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_spec_loss += spec_loss.item()
        total_recon_loss += recon_loss.item()
        total_loss += loss.item()
        batches += 1
    
    return {
        'loss': total_loss / batches,
        'spec_loss': total_spec_loss / batches,
        'recon_loss': total_recon_loss / batches
    }


def validate(model, loader, device):
    """Validate. Returns average losses."""
    model.eval()
    total_spec_loss = 0
    total_recon_loss = 0
    total_loss = 0
    batches = 0
    
    mse = nn.MSELoss()
    
    with torch.no_grad():
        for batch in loader:
            mixtures = batch['mixtures'].to(device)
            compounds = batch['compounds'].to(device)
            concentrations = batch['concentrations'].to(device)
            
            if mixtures.dim() == 3 and mixtures.shape[0] == 1:
                mixtures = mixtures.squeeze(0).unsqueeze(0)
                compounds = compounds.squeeze(0).unsqueeze(0)
                concentrations = concentrations.squeeze(0).unsqueeze(0)
            
            pred_compounds = model(mixtures)
            
            spec_loss = mse(pred_compounds, compounds)
            recon = torch.bmm(concentrations, pred_compounds)
            recon_loss = mse(recon, mixtures)
            loss = 1.0 * spec_loss + 0.5 * recon_loss
            
            total_spec_loss += spec_loss.item()
            total_recon_loss += recon_loss.item()
            total_loss += loss.item()
            batches += 1
    
    if batches == 0:
        return {'loss': 0, 'spec_loss': 0, 'recon_loss': 0}
    return {
        'loss': total_loss / batches,
        'spec_loss': total_spec_loss / batches,
        'recon_loss': total_recon_loss / batches
    }


def main():
    parser = argparse.ArgumentParser(description='MixNet Training on Real NMR Data')
    parser.add_argument('--quick', action='store_true', help='Quick test (5 datasets, 3 epochs)')
    parser.add_argument('--max_datasets', type=int, default=None, help='Limit number of datasets')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Where to save model')
    args = parser.parse_args()
    
    if args.quick:
        args.max_datasets = 10
        args.epochs = 3
        args.patience = 99
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    
    # Data
    data_dir = os.path.join(PROJECT_ROOT, 'NMR_PROJECT_FINAL_PACKAGE')
    print(f'\nLoading data from: {data_dir}')
    
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=1,  # 1 dataset per batch (20 mixtures)
        max_datasets=args.max_datasets
    )
    
    # Model: 20 mixtures in → 5 compounds out
    model = MixNet(in_channels=20, out_channels=5)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: MixNet')
    print(f'  Input:  20 mixtures x 16384 points')
    print(f'  Output: 5 compounds x 16384 points')
    print(f'  Parameters: {total_params:,}')
    
    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    print(f'\n{"="*60}')
    print(f'  TRAINING START')
    print(f'  Epochs: {args.epochs} | LR: {args.lr} | Patience: {args.patience}')
    print(f'{"="*60}\n')
    
    os.makedirs(os.path.join(PROJECT_ROOT, args.checkpoint_dir), exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        # Train
        train_losses = train_one_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_losses = validate(model, val_loader, device)
        
        scheduler.step()
        elapsed = time.time() - t0
        
        # Print
        lr_now = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1:>3}/{args.epochs} | '
              f'Train: {train_losses["loss"]:.6f} (spec={train_losses["spec_loss"]:.6f}, recon={train_losses["recon_loss"]:.6f}) | '
              f'Val: {val_losses["loss"]:.6f} | '
              f'LR: {lr_now:.6f} | '
              f'{elapsed:.1f}s', end='')
        
        # Early stopping check
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            patience_counter = 0
            # Save best model
            ckpt_path = os.path.join(PROJECT_ROOT, args.checkpoint_dir, 'best_model_real.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': train_losses['loss'],
            }, ckpt_path)
            print(' << BEST (saved)')
        else:
            patience_counter += 1
            print(f' (patience: {patience_counter}/{args.patience})')
            if patience_counter >= args.patience:
                print(f'\nEarly stopping at epoch {epoch+1}!')
                break
    
    # Final test
    print(f'\n{"="*60}')
    print(f'  TRAINING COMPLETE')
    print(f'{"="*60}')
    print(f'Best validation loss: {best_val_loss:.6f}')
    
    # Load best model and test
    ckpt = torch.load(os.path.join(PROJECT_ROOT, args.checkpoint_dir, 'best_model_real.pth'),
                       map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    test_losses = validate(model, test_loader, device)
    print(f'Test loss: {test_losses["loss"]:.6f} (spec={test_losses["spec_loss"]:.6f}, recon={test_losses["recon_loss"]:.6f})')
    

if __name__ == '__main__':
    main()
