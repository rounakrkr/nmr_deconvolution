import numpy as np
import torch
from typing import Union

TensorOrArray = Union[np.ndarray, torch.Tensor]

def normalize_spectrum(spectrum: TensorOrArray, method: str = 'minmax') -> TensorOrArray:
    """
    Normalize the given spectrum or batch of spectra.
    
    Args:
        spectrum (np.ndarray or torch.Tensor): Input spectrum. Shape can be varied (L,) or (Batch, C, L).
        method (str): 'minmax' (0 to 1) or 'zscore' (mean 0, std 1).
        
    Returns:
        Normalized spectrum of same type as input.
    """
    is_torch = isinstance(spectrum, torch.Tensor)
    eps = 1e-8
    
    if is_torch:
        if method == 'minmax':
            min_val = spectrum.min(dim=-1, keepdim=True)[0]
            max_val = spectrum.max(dim=-1, keepdim=True)[0]
            return (spectrum - min_val) / (max_val - min_val + eps)
        elif method == 'zscore':
            mean_val = spectrum.mean(dim=-1, keepdim=True)
            std_val = spectrum.std(dim=-1, keepdim=True)
            return (spectrum - mean_val) / (std_val + eps)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    else:
        if method == 'minmax':
            min_val = spectrum.min(axis=-1, keepdims=True)
            max_val = spectrum.max(axis=-1, keepdims=True)
            return (spectrum - min_val) / (max_val - min_val + eps)
        elif method == 'zscore':
            mean_val = spectrum.mean(axis=-1, keepdims=True)
            std_val = spectrum.std(axis=-1, keepdims=True)
            return (spectrum - mean_val) / (std_val + eps)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

def add_noise(spectrum: TensorOrArray, noise_std: float) -> TensorOrArray:
    """
    Add Gaussian noise to the spectrum.
    
    Args:
        spectrum (np.ndarray or torch.Tensor): Input spectrum.
        noise_std (float): Standard deviation of the Gaussian noise.
        
    Returns:
        Spectrum with added noise, non-negative. Same type as input.
    """
    is_torch = isinstance(spectrum, torch.Tensor)
    
    if is_torch:
        noise = torch.randn_like(spectrum) * noise_std
        return torch.clamp(spectrum + noise, min=0.0)
    else:
        noise = np.random.normal(0, noise_std, spectrum.shape)
        return np.clip(spectrum + noise, a_min=0.0, a_max=None)

def baseline_correction(spectrum: TensorOrArray) -> TensorOrArray:
    """
    Perform simple baseline correction by subtracting the minimum value.
    
    Args:
        spectrum (np.ndarray or torch.Tensor): Input spectrum.
        
    Returns:
        Baseline corrected spectrum.
    """
    is_torch = isinstance(spectrum, torch.Tensor)
    
    if is_torch:
        min_val = spectrum.min(dim=-1, keepdim=True)[0]
        return spectrum - min_val
    else:
        min_val = spectrum.min(axis=-1, keepdims=True)
        return spectrum - min_val

def align_spectra(spectra: TensorOrArray, reference_idx: int = 0) -> TensorOrArray:
    """
    Perform basic alignment using cross-correlation.
    Note: A more robust approach might be needed for real NMR data.
    
    Args:
        spectra (np.ndarray or torch.Tensor): 2D array of spectra (N, L).
        reference_idx (int): Index of the reference spectrum for alignment.
        
    Returns:
        Aligned spectra.
    """
    is_torch = isinstance(spectra, torch.Tensor)
    
    if is_torch:
        device = spectra.device
        spectra_np = spectra.detach().cpu().numpy()
    else:
        spectra_np = spectra.copy()
        
    if spectra_np.ndim != 2:
        raise ValueError("align_spectra expects a 2D array (N, L)")
        
    num_spectra, length = spectra_np.shape
    reference = spectra_np[reference_idx]
    aligned = np.zeros_like(spectra_np)
    aligned[reference_idx] = reference
    
    for i in range(num_spectra):
        if i == reference_idx:
            continue
            
        target = spectra_np[i]
        # Cross correlation
        correlation = np.correlate(target, reference, mode='full')
        # Find the shift that maximizes correlation
        shift = np.argmax(correlation) - (length - 1)
        
        # Shift the target array
        if shift > 0:
            aligned[i, shift:] = target[:-shift]
        elif shift < 0:
            aligned[i, :shift] = target[-shift:]
        else:
            aligned[i] = target
            
    if is_torch:
        return torch.tensor(aligned, dtype=spectra.dtype, device=device)
    else:
        return aligned
