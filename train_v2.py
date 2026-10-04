"""
MixNet V2 Training — With Cross-Mixture Attention.

Usage:
    python train_v2.py --epochs 100 --patience 25
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models.unet1d_v2 import MixNetV2
from src.data.real_nmr_loader import create_data_loaders


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_spec, total_recon, total_loss, batches = 0, 0, 0, 0
    mse = nn.MSELoss()
    
    for batch in loader:
        m = batch['mixtures'].to(device)
        c = batch['compounds'].to(device)
        a = batch['concentrations'].to(device)
        
        if m.dim() == 3 and m.shape[0] == 1:
            m = m.squeeze(0).unsqueeze(0)
            c = c.squeeze(0).unsqueeze(0)
            a = a.squeeze(0).unsqueeze(0)
        
        optimizer.zero_grad()
        pred = model(m)
        
        spec_loss = mse(pred, c)
        recon = torch.bmm(a, pred)
        recon_loss = mse(recon, m)
        loss = 1.0 * spec_loss + 0.5 * recon_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_spec += spec_loss.item()
        total_recon += recon_loss.item()
        total_loss += loss.item()
        batches += 1
    
    return {'loss': total_loss/batches, 'spec_loss': total_spec/batches, 'recon_loss': total_recon/batches}


def validate(model, loader, device):
    model.eval()
    total_spec, total_recon, total_loss, batches = 0, 0, 0, 0
    mse = nn.MSELoss()
    
    with torch.no_grad():
        for batch in loader:
            m = batch['mixtures'].to(device)
            c = batch['compounds'].to(device)
            a = batch['concentrations'].to(device)
            
            if m.dim() == 3 and m.shape[0] == 1:
                m = m.squeeze(0).unsqueeze(0)
                c = c.squeeze(0).unsqueeze(0)
                a = a.squeeze(0).unsqueeze(0)
            
            pred = model(m)
            spec_loss = mse(pred, c)
            recon = torch.bmm(a, pred)
            recon_loss = mse(recon, m)
            loss = 1.0 * spec_loss + 0.5 * recon_loss
            
            total_spec += spec_loss.item()
            total_recon += recon_loss.item()
            total_loss += loss.item()
            batches += 1
    
    if batches == 0:
        return {'loss': 0, 'spec_loss': 0, 'recon_loss': 0}
    return {'loss': total_loss/batches, 'spec_loss': total_spec/batches, 'recon_loss': total_recon/batches}


def main():
    parser = argparse.ArgumentParser(description='MixNet V2 Training')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--max_datasets', type=int, default=None)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    data_dir = os.path.join(PROJECT_ROOT, 'NMR_PROJECT_FINAL_PACKAGE')
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, batch_size=1, max_datasets=args.max_datasets
    )
    
    model = MixNetV2(in_channels=20, out_channels=5).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: MixNet V2 (with Cross-Mixture Attention)')
    print(f'  Parameters: {params:,}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f'\n{"="*60}')
    print(f'  TRAINING V2 START')
    print(f'  Epochs: {args.epochs} | LR: {args.lr} | Patience: {args.patience}')
    print(f'{"="*60}\n')
    
    os.makedirs(os.path.join(PROJECT_ROOT, 'checkpoints'), exist_ok=True)
    best_val = float('inf')
    patience_ctr = 0
    
    for epoch in range(args.epochs):
        t0 = time.time()
        train_l = train_one_epoch(model, train_loader, optimizer, device)
        val_l = validate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1:>3}/{args.epochs} | '
              f'Train: {train_l["loss"]:.6f} (spec={train_l["spec_loss"]:.6f}, recon={train_l["recon_loss"]:.6f}) | '
              f'Val: {val_l["loss"]:.6f} | LR: {lr_now:.6f} | {elapsed:.1f}s', end='')
        
        if val_l['loss'] < best_val:
            best_val = val_l['loss']
            patience_ctr = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val,
                'train_loss': train_l['loss'],
            }, os.path.join(PROJECT_ROOT, 'checkpoints', 'best_model_v2.pth'))
            print(' << BEST (saved)')
        else:
            patience_ctr += 1
            print(f' (patience: {patience_ctr}/{args.patience})')
            if patience_ctr >= args.patience:
                print(f'\nEarly stopping at epoch {epoch+1}!')
                break
    
    print(f'\n{"="*60}')
    print(f'  V2 TRAINING COMPLETE')
    print(f'{"="*60}')
    print(f'Best validation loss: {best_val:.6f}')
    
    ckpt = torch.load(os.path.join(PROJECT_ROOT, 'checkpoints', 'best_model_v2.pth'),
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    test_l = validate(model, test_loader, device)
    print(f'Test loss: {test_l["loss"]:.6f} (spec={test_l["spec_loss"]:.6f}, recon={test_l["recon_loss"]:.6f})')


if __name__ == '__main__':
    main()
