import os
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Dict, Tuple

from src.training.losses import CombinedLoss
from src.training.metrics import compute_all_metrics

class Trainer:
    """
    Trainer class for the NMR Spectral Decomposition model.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device(config.get("device", "cpu"))
        if self.device.type == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.model.to(self.device)
        
        train_config = config["training"]
        self.lr = train_config["learning_rate"]
        self.epochs = train_config["epochs"]
        self.patience = train_config["patience"]
        self.checkpoint_dir = train_config["checkpoint_dir"]
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=train_config["weight_decay"])
        
        # Cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.warmup_epochs = train_config["warmup_epochs"]
        
        self.criterion = CombinedLoss(
            lambda_spectral=train_config["lambda_spectral"],
            lambda_reconstruction=train_config["lambda_reconstruction"]
        )
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def _warmup_lr(self, epoch: int):
        """Linearly warms up the learning rate for the specified number of epochs."""
        if epoch < self.warmup_epochs:
            lr = self.lr * ((epoch + 1) / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train_one_epoch(self) -> Dict[str, float]:
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_spec_loss = 0.0
        total_recon_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            # Unpack batch dict from collate_fn
            mixtures = batch['mixtures'].to(self.device)
            spectra = batch['compounds'].to(self.device)
            concentrations = batch['concentrations'].to(self.device)
            comp_mask = batch['compound_mask'].to(self.device)
            mix_mask = batch['mixture_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            pred_spectra = self.model(mixtures)
            
            loss, losses_dict = self.criterion(
                pred_spectra, spectra, concentrations, mixtures, comp_mask, mix_mask
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += losses_dict["loss"]
            total_spec_loss += losses_dict["spectral_loss"]
            total_recon_loss += losses_dict["reconstruction_loss"]
            
        num_batches = len(self.train_loader)
        return {
            "loss": total_loss / num_batches,
            "spectral_loss": total_spec_loss / num_batches,
            "reconstruction_loss": total_recon_loss / num_batches
        }

    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Evaluates the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        total_spec_loss = 0.0
        total_recon_loss = 0.0
        
        all_metrics = []
        # Recreate dummy ppm axis for metrics computation
        ppm_axis = torch.linspace(
            self.config["data"]["ppm_range"][0], 
            self.config["data"]["ppm_range"][1], 
            self.config["data"]["spectral_length"]
        ).to(self.device)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                mixtures = batch['mixtures'].to(self.device)
                spectra = batch['compounds'].to(self.device)
                concentrations = batch['concentrations'].to(self.device)
                comp_mask = batch['compound_mask'].to(self.device)
                mix_mask = batch['mixture_mask'].to(self.device)
                
                pred_spectra = self.model(mixtures)
                
                loss, losses_dict = self.criterion(
                    pred_spectra, spectra, concentrations, mixtures, comp_mask, mix_mask
                )
                
                total_loss += losses_dict["loss"]
                total_spec_loss += losses_dict["spectral_loss"]
                total_recon_loss += losses_dict["reconstruction_loss"]
                
                metrics = compute_all_metrics(pred_spectra, spectra, concentrations, mixtures, ppm_axis)
                all_metrics.append(metrics)
                
        num_batches = len(self.val_loader)
        val_loss_dict = {
            "loss": total_loss / num_batches,
            "spectral_loss": total_spec_loss / num_batches,
            "reconstruction_loss": total_recon_loss / num_batches
        }
        
        avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0].keys()}
        
        return val_loss_dict, avg_metrics

    def save_checkpoint(self, path: str):
        """Saves model checkpoint."""
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        """Loads model checkpoint."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def train(self):
        """Main training loop with early stopping and lr scheduling."""
        print("Starting training...")
        for epoch in range(self.epochs):
            self._warmup_lr(epoch)
            
            train_loss = self.train_one_epoch()
            val_loss, val_metrics = self.validate()
            
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
                
            print(f"Epoch {epoch+1}/{self.epochs} - "
                  f"Train Loss: {train_loss['loss']:.4f} - "
                  f"Val Loss: {val_loss['loss']:.4f} - "
                  f"Val MSE: {val_metrics['mse']:.4f}")
                  
            if val_loss['loss'] < self.best_val_loss - self.config["training"]["min_delta"]:
                self.best_val_loss = val_loss['loss']
                self.early_stop_counter = 0
                if self.config["training"]["save_best_only"]:
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pth"))
            else:
                self.early_stop_counter += 1
                
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

if __name__ == "__main__":
    # --- Quick Test Block ---
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Dummy Model for testing the training loop
    class DummyModel(nn.Module):
        def __init__(self, in_features, out_features, num_out_compounds):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.num_out_compounds = num_out_compounds
            
        def forward(self, x):
            # x shape: (B, M, L)
            B, M, L = x.shape
            x = self.linear(x) # (B, M, L)
            # Just return a slice to match N
            # In a real model, it would output (B, N, L)
            if M >= self.num_out_compounds:
                return x[:, :self.num_out_compounds, :]
            else:
                pad = torch.zeros(B, self.num_out_compounds - M, L, device=x.device)
                return torch.cat([x, pad], dim=1)

    spectral_length = config["data"]["spectral_length"]
    B, M, N = 2, 5, 3
    
    model = DummyModel(spectral_length, spectral_length, N)
    
    # Create tiny dummy data
    mixtures = torch.rand(B, M, spectral_length)
    spectra = torch.rand(B, N, spectral_length)
    concentrations = torch.rand(B, M, N)
    comp_mask = torch.ones(B, N)
    mix_mask = torch.ones(B, M)
    
    dataset = TensorDataset(mixtures, spectra, concentrations, comp_mask, mix_mask)
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)
    
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Override epochs for quick test
    trainer.epochs = 5
    trainer.train()
