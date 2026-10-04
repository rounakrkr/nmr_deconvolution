from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def match_sources(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Optimal output-slot assignment per batch element (Hungarian, MSE cost).
    pred, true: (B, N, L). Returns perm (B, N) with perm[b, i] = pred slot matched to true source i.
    """
    length = pred.shape[-1]
    cost = (
        true.pow(2).mean(-1).unsqueeze(2)
        + pred.pow(2).mean(-1).unsqueeze(1)
        - 2.0 * torch.bmm(true, pred.transpose(1, 2)) / length
    )
    cost_np = cost.float().cpu().numpy()
    perm = np.stack([linear_sum_assignment(c)[1] for c in cost_np])
    return torch.from_numpy(perm).to(pred.device)


def reorder(pred: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    idx = perm.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
    return pred.gather(1, idx)


class PITLoss(nn.Module):
    """
    Permutation-invariant spectral MSE plus reconstruction consistency.
    The mixing matrix columns are indexed by true source order, so the
    reconstruction uses the matched (reordered) predictions.
    """

    def __init__(self, lambda_spectral: float = 1.0, lambda_recon: float = 0.5):
        super().__init__()
        self.lambda_spectral = lambda_spectral
        self.lambda_recon = lambda_recon

    def forward(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        concentrations: torch.Tensor,
        mixtures: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        perm = match_sources(pred.detach(), true)
        matched = reorder(pred, perm)
        spec = F.mse_loss(matched, true)
        recon = F.mse_loss(torch.bmm(concentrations, matched), mixtures)
        total = self.lambda_spectral * spec + self.lambda_recon * recon
        return total, {"loss": total.item(), "spec": spec.item(), "recon": recon.item()}
