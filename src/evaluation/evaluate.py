import os
import json
import torch
import numpy as np
from typing import Dict, Any, List
from scipy.stats import pearsonr
from scipy.signal import find_peaks

# Assuming visualize.py is in the same directory
from .visualize import (
    plot_decomposition_result,
    plot_reconstruction_comparison,
    plot_correlation
)

def compute_mse(true_compounds: np.ndarray, pred_compounds: np.ndarray) -> float:
    return float(np.mean((true_compounds - pred_compounds) ** 2))

def compute_pearson_r(true_compounds: np.ndarray, pred_compounds: np.ndarray) -> float:
    true_flat = true_compounds.flatten()
    pred_flat = pred_compounds.flatten()
    r, _ = pearsonr(true_flat, pred_flat)
    return float(r)

def compute_reconstruction_error(input_mixtures: np.ndarray, pred_compounds: np.ndarray) -> float:
    # A simple reconstruction approximation: sum of compounds vs sum of mixtures
    sum_mixtures = np.sum(input_mixtures, axis=0)
    sum_compounds = np.sum(pred_compounds, axis=0)
    return float(np.mean((sum_mixtures - sum_compounds) ** 2))

def compute_peak_position_error(true_compounds: np.ndarray, pred_compounds: np.ndarray) -> float:
    errors: List[float] = []
    for i in range(true_compounds.shape[0]):
        # Find highest peak as a proxy for peak positions
        threshold_true = np.max(true_compounds[i]) * 0.1
        threshold_pred = np.max(pred_compounds[i]) * 0.1
        
        true_peaks, _ = find_peaks(true_compounds[i], height=threshold_true)
        pred_peaks, _ = find_peaks(pred_compounds[i], height=threshold_pred)
        
        if len(true_peaks) > 0 and len(pred_peaks) > 0:
            true_main = true_peaks[np.argmax(true_compounds[i][true_peaks])]
            pred_main = pred_peaks[np.argmax(pred_compounds[i][pred_peaks])]
            errors.append(float(abs(true_main - pred_main)))
    
    return float(np.mean(errors)) if errors else 0.0

def evaluate_single_sample(model: torch.nn.Module, sample: Dict[str, torch.Tensor], device: torch.device, ppm_axis: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates the model on a single sample.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model.
        sample (Dict[str, torch.Tensor]): Dictionary containing 'mixtures' and 'compounds'.
        device (torch.device): Device to run the evaluation on.
        ppm_axis (np.ndarray): The ppm axis array.
        
    Returns:
        Dict[str, Any]: Dictionary with predictions and metrics.
    """
    model.eval()
    
    # Move to device and add batch dimension if needed
    mixtures = sample['mixtures'].unsqueeze(0).to(device) # (1, M, L)
    true_compounds = sample['compounds'].numpy() # (N, L)
    input_mixtures = sample['mixtures'].numpy() # (M, L)
    
    with torch.no_grad():
        preds = model(mixtures)
        # Squeeze batch dimension
        pred_compounds = preds.squeeze(0).cpu().numpy() # (N, L)
        
    metrics = {
        'mse': compute_mse(true_compounds, pred_compounds),
        'pearson_r': compute_pearson_r(true_compounds, pred_compounds),
        'reconstruction_error': compute_reconstruction_error(input_mixtures, pred_compounds),
        'peak_position_error': compute_peak_position_error(true_compounds, pred_compounds)
    }
    
    return {
        'metrics': metrics,
        'predictions': pred_compounds,
        'true_compounds': true_compounds,
        'input_mixtures': input_mixtures
    }

def evaluate_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device, ppm_axis: np.ndarray, results_dir: str) -> Dict[str, float]:
    """
    Runs model evaluation on the entire test set.
    
    Args:
        model (torch.nn.Module): Trained model.
        test_loader (torch.utils.data.DataLoader): DataLoader for test set.
        device (torch.device): Device to run on.
        ppm_axis (np.ndarray): Numpy array for the ppm axis.
        results_dir (str): Directory to save plots and JSON results.
        
    Returns:
        Dict[str, float]: Dictionary of averaged metrics.
    """
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    
    all_metrics: Dict[str, List[float]] = {'mse': [], 'pearson_r': [], 'reconstruction_error': [], 'peak_position_error': []}
    
    plot_num_samples = 5
    samples_plotted = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            mixtures = batch['mixtures'].to(device)
            true_compounds_batch = batch['compounds'].cpu().numpy()
            input_mixtures_batch = batch['mixtures'].cpu().numpy()
            
            preds = model(mixtures)
            pred_compounds_batch = preds.cpu().numpy()
            
            for i in range(mixtures.size(0)):
                true_compounds = true_compounds_batch[i]
                pred_compounds = pred_compounds_batch[i]
                input_mixtures = input_mixtures_batch[i]
                
                mse = compute_mse(true_compounds, pred_compounds)
                pearson = compute_pearson_r(true_compounds, pred_compounds)
                recon = compute_reconstruction_error(input_mixtures, pred_compounds)
                peak_err = compute_peak_position_error(true_compounds, pred_compounds)
                
                all_metrics['mse'].append(mse)
                all_metrics['pearson_r'].append(pearson)
                all_metrics['reconstruction_error'].append(recon)
                all_metrics['peak_position_error'].append(peak_err)
                
                if samples_plotted < plot_num_samples:
                    save_path_decomp = os.path.join(results_dir, f'sample_{samples_plotted}_decomposition')
                    plot_decomposition_result(ppm_axis, input_mixtures, true_compounds, pred_compounds, sample_idx=samples_plotted, save_path=save_path_decomp)
                    
                    save_path_corr = os.path.join(results_dir, f'sample_{samples_plotted}_correlation')
                    plot_correlation(true_compounds, pred_compounds, title=f'Sample {samples_plotted}: True vs Predicted', save_path=save_path_corr)
                    
                    samples_plotted += 1

    avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    
    results_file = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(avg_metrics, f, indent=4)
        
    print("=" * 40)
    print("Evaluation Results Summary")
    print("=" * 40)
    for metric, value in avg_metrics.items():
        print(f"{metric.ljust(25)}: {value:.6f}")
    print("=" * 40)
    
    return avg_metrics

if __name__ == "__main__":
    import yaml
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device_str = config.get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
        
    print(f"Running evaluation on device: {device}")
    
    spectral_length = config['data']['spectral_length']
    ppm_range = config['data']['ppm_range']
    ppm_axis = np.linspace(ppm_range[1], ppm_range[0], spectral_length)
    
    class DummyModel(torch.nn.Module):
        def __init__(self, n_compounds):
            super().__init__()
            self.n_compounds = n_compounds
            
        def forward(self, x):
            batch_size = x.size(0)
            length = x.size(2)
            return torch.rand((batch_size, self.n_compounds, length), device=x.device)
            
    n_compounds = config['data']['num_compounds_range'][0]
    model = DummyModel(n_compounds=n_compounds).to(device)
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, num_mixtures, num_compounds, length):
            self.num_samples = num_samples
            self.num_mixtures = num_mixtures
            self.num_compounds = num_compounds
            self.length = length
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            return {
                'mixtures': torch.rand(self.num_mixtures, self.length),
                'compounds': torch.rand(self.num_compounds, self.length)
            }
            
    num_mixtures = config['data']['num_mixtures_range_offset'][0] + n_compounds
    test_dataset = DummyDataset(10, num_mixtures, n_compounds, spectral_length)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', config['evaluation']['results_dir'])
    
    evaluate_model(model, test_loader, device, ppm_axis, results_dir)
