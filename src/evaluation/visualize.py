import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
from typing import Optional, List, Tuple

# Publication quality settings
mpl.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.prop_cycle': plt.cycler('color', plt.cm.tab10.colors)
})

def plot_spectrum(ppm_axis: np.ndarray, spectrum: np.ndarray, title: str = '', color: str = 'blue', ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plots a single NMR spectrum.

    Args:
        ppm_axis (np.ndarray): The ppm axis (x-axis).
        spectrum (np.ndarray): The intensity values (y-axis).
        title (str): Title for the plot.
        color (str): Color of the plot line.
        ax (Optional[plt.Axes]): Matplotlib axes to plot on.

    Returns:
        plt.Axes: The axes on which the spectrum was plotted.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(ppm_axis, spectrum, color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel('Chemical Shift (ppm)')
    ax.set_ylabel('Intensity')
    
    # NMR convention: Reverse the x-axis
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
        
    ax.grid(True, linestyle='--', alpha=0.6)
    return ax

def plot_decomposition_result(ppm_axis: np.ndarray, input_mixtures: np.ndarray, true_compounds: np.ndarray, predicted_compounds: np.ndarray, sample_idx: int = 0, save_path: Optional[str] = None):
    """
    Plots a 3-row figure showing input mixtures, true compounds, and predicted compounds.

    Args:
        ppm_axis (np.ndarray): The ppm axis.
        input_mixtures (np.ndarray): Array of shape (M, L) containing mixture spectra.
        true_compounds (np.ndarray): Array of shape (N, L) containing true individual compound spectra.
        predicted_compounds (np.ndarray): Array of shape (N, L) containing predicted individual compound spectra.
        sample_idx (int): Index of the sample for the title.
        save_path (Optional[str]): Path to save the plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Sample {sample_idx}: NMR Spectral Decomposition', fontsize=16)

    # Row 1: Input Mixtures
    for i in range(input_mixtures.shape[0]):
        axes[0].plot(ppm_axis, input_mixtures[i], label=f'Mixture {i+1}', alpha=0.7)
    axes[0].set_title('Input Mixture Spectra')
    axes[0].set_ylabel('Intensity')
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Row 2: True Compounds
    for i in range(true_compounds.shape[0]):
        axes[1].plot(ppm_axis, true_compounds[i], label=f'Compound {i+1}')
    axes[1].set_title('True Individual Compound Spectra')
    axes[1].set_ylabel('Intensity')
    axes[1].legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Row 3: Predicted Compounds
    for i in range(predicted_compounds.shape[0]):
        axes[2].plot(ppm_axis, predicted_compounds[i], label=f'Pred Compound {i+1}', linestyle='--')
    axes[2].set_title('Predicted Individual Compound Spectra')
    axes[2].set_xlabel('Chemical Shift (ppm)')
    axes[2].set_ylabel('Intensity')
    axes[2].legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    if not axes[2].xaxis_inverted():
        axes[2].invert_xaxis()
        
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    if save_path:
        base, _ = os.path.splitext(save_path)
        plt.savefig(f"{base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{base}.pdf", bbox_inches='tight')
    plt.close()

def plot_reconstruction_comparison(ppm_axis: np.ndarray, original_mixtures: np.ndarray, reconstructed_mixtures: np.ndarray, save_path: Optional[str] = None):
    """
    Overlays original vs reconstructed mixture spectra with a residual plot.

    Args:
        ppm_axis (np.ndarray): The ppm axis.
        original_mixtures (np.ndarray): Array of shape (M, L).
        reconstructed_mixtures (np.ndarray): Array of shape (M, L).
        save_path (Optional[str]): Path to save the plot.
    """
    num_mixtures = original_mixtures.shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    fig.suptitle('Mixture Reconstruction', fontsize=16)

    # Plot original and reconstructed
    for i in range(num_mixtures):
        color = plt.cm.tab10(i % 10)
        axes[0].plot(ppm_axis, original_mixtures[i], color=color, alpha=0.6, label=f'Original {i+1}')
        axes[0].plot(ppm_axis, reconstructed_mixtures[i], color=color, linestyle='--', label=f'Reconstructed {i+1}')
        
    axes[0].set_title('Original vs Reconstructed')
    axes[0].set_ylabel('Intensity')
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Residuals
    residuals = original_mixtures - reconstructed_mixtures
    for i in range(num_mixtures):
        color = plt.cm.tab10(i % 10)
        axes[1].plot(ppm_axis, residuals[i], color=color, alpha=0.6)
        
    axes[1].set_title('Residuals (Original - Reconstructed)')
    axes[1].set_xlabel('Chemical Shift (ppm)')
    axes[1].set_ylabel('Residual')
    
    if not axes[1].xaxis_inverted():
        axes[1].invert_xaxis()
        
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    if save_path:
        base, _ = os.path.splitext(save_path)
        plt.savefig(f"{base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{base}.pdf", bbox_inches='tight')
    plt.close()

def plot_training_history(train_losses: List[float], val_losses: List[float], save_path: Optional[str] = None):
    """
    Plots the training and validation loss curves.

    Args:
        train_losses (List[float]): Training losses per epoch.
        val_losses (List[float]): Validation losses per epoch.
        save_path (Optional[str]): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', color='blue')
    ax.plot(epochs, val_losses, label='Validation Loss', color='orange')
    
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        base, _ = os.path.splitext(save_path)
        plt.savefig(f"{base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{base}.pdf", bbox_inches='tight')
    plt.close()

def plot_correlation(true_spectrum: np.ndarray, predicted_spectrum: np.ndarray, title: str = 'True vs Predicted Intensity', save_path: Optional[str] = None):
    """
    Scatter plot of true vs predicted values with Pearson r and R^2.

    Args:
        true_spectrum (np.ndarray): True intensity values.
        predicted_spectrum (np.ndarray): Predicted intensity values.
        title (str): Plot title.
        save_path (Optional[str]): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Flatten arrays
    true_flat = true_spectrum.flatten()
    pred_flat = predicted_spectrum.flatten()
    
    ax.scatter(true_flat, pred_flat, alpha=0.5, marker='.', color='teal')
    
    # Identity line
    min_val = min(true_flat.min(), pred_flat.min())
    max_val = max(true_flat.max(), pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x (Identity)')
    
    # Correlation
    r, _ = pearsonr(true_flat, pred_flat)
    r2 = r ** 2
    
    ax.text(0.05, 0.95, f'Pearson r = {r:.4f}\n$R^2$ = {r2:.4f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
    ax.set_title(title)
    ax.set_xlabel('True Intensity')
    ax.set_ylabel('Predicted Intensity')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        base, _ = os.path.splitext(save_path)
        plt.savefig(f"{base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{base}.pdf", bbox_inches='tight')
    plt.close()
