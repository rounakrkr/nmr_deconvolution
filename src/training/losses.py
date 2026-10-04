import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class SpectralLoss(nn.Module):
    """
    MSE loss between predicted and true individual spectra, with masking for variable N.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, predicted_spectra: torch.Tensor, true_spectra: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predicted_spectra: Tensor of shape (Batch, N_max, L)
            true_spectra: Tensor of shape (Batch, N_max, L)
            mask: Optional boolean tensor of shape (Batch, N_max) or (Batch, N_max, 1), 
                  True for valid compounds, False for padded ones.
        Returns:
            Scalar loss
        """
        loss = self.mse(predicted_spectra, true_spectra) # (B, N_max, L)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1) # (B, N_max, 1)
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1e-8) / loss.size(-1)
        return loss.mean()

class ReconstructionLoss(nn.Module):
    """
    Consistency check by reconstructing mixtures from predicted components and known concentrations.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, predicted_spectra: torch.Tensor, concentrations: torch.Tensor, input_mixtures: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predicted_spectra: Tensor of shape (Batch, N_max, L)
            concentrations: Tensor of shape (Batch, M_max, N_max) - mixing matrix A
            input_mixtures: Tensor of shape (Batch, M_max, L) - true mixture X
            mask: Optional boolean tensor of shape (Batch, M_max) or (Batch, M_max, 1),
                  True for valid mixtures, False for padded ones.
        Returns:
            Scalar loss
        """
        # X_recon = A @ S_pred
        # concentrations: (B, M_max, N_max)
        # predicted_spectra: (B, N_max, L)
        # reconstructed: (B, M_max, L)
        reconstructed = torch.bmm(concentrations, predicted_spectra)
        loss = self.mse(reconstructed, input_mixtures)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1) # (B, M_max, 1)
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1e-8) / loss.size(-1)
        return loss.mean()

class CombinedLoss(nn.Module):
    """
    Combined Spectral and Reconstruction loss.
    """
    def __init__(self, lambda_spectral: float = 1.0, lambda_reconstruction: float = 0.5):
        super().__init__()
        self.lambda_spectral = lambda_spectral
        self.lambda_reconstruction = lambda_reconstruction
        self.spectral_loss = SpectralLoss()
        self.reconstruction_loss = ReconstructionLoss()

    def forward(self, predicted_spectra: torch.Tensor, true_spectra: torch.Tensor, 
                concentrations: torch.Tensor, input_mixtures: torch.Tensor,
                comp_mask: Optional[torch.Tensor] = None, mix_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        l_spec = self.spectral_loss(predicted_spectra, true_spectra, comp_mask)
        l_recon = self.reconstruction_loss(predicted_spectra, concentrations, input_mixtures, mix_mask)
        
        total_loss = self.lambda_spectral * l_spec + self.lambda_reconstruction * l_recon
        
        losses = {
            "loss": total_loss.item(),
            "spectral_loss": l_spec.item(),
            "reconstruction_loss": l_recon.item()
        }
        return total_loss, losses
