import torch
import numpy as np
from typing import Dict
from scipy.stats import pearsonr
from scipy.signal import find_peaks

def compute_mse(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Computes Mean Squared Error."""
    return torch.nn.functional.mse_loss(predicted, target).item()

def compute_pearson_r(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Computes average Pearson correlation coefficient across compounds."""
    pred_np = predicted.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    r_vals = []
    # Loop over batch and compounds
    for i in range(pred_np.shape[0]):
        for j in range(pred_np.shape[1]):
            p = pred_np[i, j].flatten()
            t = target_np[i, j].flatten()
            if np.std(t) > 1e-8 and np.std(p) > 1e-8:
                r, _ = pearsonr(p, t)
                r_vals.append(r)
    
    return float(np.mean(r_vals)) if r_vals else 0.0

def compute_reconstruction_error(predicted_spectra: torch.Tensor, concentrations: torch.Tensor, input_mixtures: torch.Tensor) -> float:
    """Computes MSE between reconstructed mixtures and true mixtures."""
    reconstructed = torch.bmm(concentrations, predicted_spectra)
    return compute_mse(reconstructed, input_mixtures)

def compute_peak_position_error(predicted: torch.Tensor, target: torch.Tensor, ppm_axis: torch.Tensor) -> float:
    """
    Computes average peak position error in ppm.
    Finds peaks in both and matches them.
    """
    pred_np = predicted.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ppm_np = ppm_axis.detach().cpu().numpy()
    
    errors = []
    for i in range(pred_np.shape[0]):
        for j in range(pred_np.shape[1]):
            p = pred_np[i, j]
            t = target_np[i, j]
            
            p_peaks, _ = find_peaks(p, height=np.max(p)*0.1)
            t_peaks, _ = find_peaks(t, height=np.max(t)*0.1)
            
            if len(p_peaks) > 0 and len(t_peaks) > 0:
                # Find closest match for each true peak
                for tp in t_peaks:
                    closest_p = p_peaks[np.argmin(np.abs(p_peaks - tp))]
                    error_ppm = np.abs(ppm_np[closest_p] - ppm_np[tp])
                    errors.append(error_ppm)
                    
    return float(np.mean(errors)) if errors else 0.0

def compute_all_metrics(predicted: torch.Tensor, target: torch.Tensor, concentrations: torch.Tensor, mixtures: torch.Tensor, ppm_axis: torch.Tensor) -> Dict[str, float]:
    """Computes all evaluation metrics."""
    return {
        "mse": compute_mse(predicted, target),
        "pearson_r": compute_pearson_r(predicted, target),
        "reconstruction_error": compute_reconstruction_error(predicted, concentrations, mixtures),
        "peak_position_error": compute_peak_position_error(predicted, target, ppm_axis)
    }
