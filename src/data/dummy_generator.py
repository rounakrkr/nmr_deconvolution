import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

def generate_gaussian_peak(ppm_axis: np.ndarray, center: float, amplitude: float, sigma: float) -> np.ndarray:
    """
    Generate a 1D array representing a single Gaussian peak.

    Args:
        ppm_axis (np.ndarray): The x-axis (ppm).
        center (float): Center position of the peak.
        amplitude (float): Amplitude/height of the peak.
        sigma (float): Standard deviation (width) of the peak.

    Returns:
        np.ndarray: The generated 1D Gaussian peak.
    """
    return amplitude * np.exp(-((ppm_axis - center) ** 2) / (2 * sigma ** 2))

def generate_compound_spectrum(
    ppm_axis: np.ndarray,
    num_peaks_range: Tuple[int, int] = (1, 8),
    amp_range: Tuple[float, float] = (0.1, 1.0),
    sigma_range: Tuple[float, float] = (0.005, 0.05)
) -> np.ndarray:
    """
    Generate a random compound spectrum composed of multiple Gaussian peaks.

    Args:
        ppm_axis (np.ndarray): The x-axis (ppm).
        num_peaks_range (Tuple[int, int]): Range for random number of peaks.
        amp_range (Tuple[float, float]): Range for random amplitude.
        sigma_range (Tuple[float, float]): Range for random sigma.

    Returns:
        np.ndarray: The compound spectrum array.
    """
    num_peaks = np.random.randint(num_peaks_range[0], num_peaks_range[1] + 1)
    spectrum = np.zeros_like(ppm_axis)

    for _ in range(num_peaks):
        # Random center within the ppm range limits
        center = np.random.uniform(ppm_axis.min(), ppm_axis.max())
        amplitude = np.random.uniform(amp_range[0], amp_range[1])
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        spectrum += generate_gaussian_peak(ppm_axis, center, amplitude, sigma)

    return spectrum

def generate_mixture_set(
    ppm_axis: np.ndarray,
    num_compounds: int,
    num_mixtures: int,
    noise_std: float = 0.005
) -> Dict[str, np.ndarray]:
    """
    Generate a set of compound spectra and mixture spectra.

    Args:
        ppm_axis (np.ndarray): The x-axis (ppm).
        num_compounds (int): Number of pure compounds.
        num_mixtures (int): Number of mixture spectra to generate.
        noise_std (float): Standard deviation of Gaussian noise added to mixtures.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'mixtures': (num_mixtures, len(ppm_axis))
            - 'compounds': (num_compounds, len(ppm_axis))
            - 'concentrations': (num_mixtures, num_compounds)
            - 'ppm_axis': (len(ppm_axis),)
    """
    # Generate compound spectra
    compounds = np.array([
        generate_compound_spectrum(ppm_axis) for _ in range(num_compounds)
    ])

    # Generate random concentrations using Dirichlet distribution for each mixture
    # Adding 1 to alpha parameter to avoid very sparse concentrations by default
    alphas = np.ones(num_compounds) 
    concentrations = np.random.dirichlet(alphas, size=num_mixtures)

    # Create mixtures (Linear combination)
    mixtures = concentrations @ compounds

    # Add Gaussian noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, mixtures.shape)
        mixtures = np.clip(mixtures + noise, 0, None)  # Ensure non-negative

    return {
        'mixtures': mixtures,
        'compounds': compounds,
        'concentrations': concentrations,
        'ppm_axis': ppm_axis
    }

def generate_dataset(
    num_samples: int,
    save_dir: str,
    spectral_length: int = 4096,
    ppm_range: Tuple[float, float] = (0.5, 10.0),
    num_compounds_range: Tuple[int, int] = (2, 10),
    num_mixtures_range_offset: Tuple[int, int] = (3, 15),
    noise_std: float = 0.005,
    seed: Optional[int] = None
) -> None:
    """
    Generate multiple mixture sets and save as .npz files.

    Args:
        num_samples (int): Number of sample sets to generate.
        save_dir (str): Directory to save the dataset.
        spectral_length (int): Number of points in the ppm axis.
        ppm_range (Tuple[float, float]): Range of the ppm axis.
        num_compounds_range (Tuple[int, int]): Range for random number of compounds.
        num_mixtures_range_offset (Tuple[int, int]): Offset to determine number of mixtures (M = N + offset).
        noise_std (float): Standard deviation of added noise.
        seed (Optional[int]): Random seed.
    """
    if seed is not None:
        np.random.seed(seed)
        
    os.makedirs(save_dir, exist_ok=True)
    ppm_axis = np.linspace(ppm_range[0], ppm_range[1], spectral_length)
    
    for i in range(num_samples):
        # Determine N and M for this sample
        num_compounds = np.random.randint(num_compounds_range[0], num_compounds_range[1] + 1)
        offset = np.random.randint(num_mixtures_range_offset[0], num_mixtures_range_offset[1] + 1)
        num_mixtures = num_compounds + offset
        
        data = generate_mixture_set(
            ppm_axis=ppm_axis,
            num_compounds=num_compounds,
            num_mixtures=num_mixtures,
            noise_std=noise_std
        )
        
        file_path = os.path.join(save_dir, f"sample_{i:06d}.npz")
        np.savez_compressed(
            file_path,
            mixtures=data['mixtures'],
            compounds=data['compounds'],
            concentrations=data['concentrations'],
            ppm_axis=data['ppm_axis'],
            num_compounds=num_compounds,
            num_mixtures=num_mixtures
        )
        
    print(f"Generated {num_samples} samples in {save_dir}")

if __name__ == "__main__":
    # Generate 10 test samples
    save_directory = os.path.join(os.path.dirname(__file__), "test_dummy_data")
    generate_dataset(
        num_samples=10, 
        save_dir=save_directory,
        seed=42
    )
    
    # Plot one sample to visualize
    sample_file = os.path.join(save_directory, "sample_000000.npz")
    data = np.load(sample_file)
    
    mixtures = data['mixtures']
    compounds = data['compounds']
    ppm_axis = data['ppm_axis']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot mixtures
    for i in range(min(5, len(mixtures))):
        axes[0].plot(ppm_axis, mixtures[i], label=f'Mixture {i+1}')
    axes[0].set_title("Mixtures")
    axes[0].legend()
    
    # Plot compounds
    for i in range(len(compounds)):
        axes[1].plot(ppm_axis, compounds[i], label=f'Compound {i+1}')
    axes[1].set_title("Pure Compounds")
    axes[1].legend()
    
    axes[1].set_xlabel("PPM")
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "sample_000000_plot.png"))
    print(f"Plot saved to {os.path.join(save_directory, 'sample_000000_plot.png')}")
