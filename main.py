"""
MixNet — NMR Spectral Decomposition
Main entry point for training and evaluation.

Usage:
    python main.py --mode generate    # Generate dummy training data
    python main.py --mode train       # Train the model
    python main.py --mode evaluate    # Evaluate trained model
    python main.py --mode all         # Generate data + Train + Evaluate
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data.dummy_generator import generate_dataset
from src.data.dataset import NMRMixtureDataset, collate_fn
from src.models.unet1d import MixNet
from src.training.train import Trainer


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "src", "configs", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device(config: dict) -> torch.device:
    """Determine compute device."""
    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    return device


def generate_data(config: dict):
    """Generate dummy training, validation, and test data."""
    data_config = config["data"]
    dummy_config = config["dummy"]
    
    print("=" * 60)
    print("GENERATING DUMMY DATA")
    print("=" * 60)
    
    base_dir = os.path.join(PROJECT_ROOT, dummy_config["save_dir"])
    
    for split, num_samples in [
        ("train", dummy_config["num_train_samples"]),
        ("val", dummy_config["num_val_samples"]),
        ("test", dummy_config["num_test_samples"])
    ]:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir) and len(os.listdir(split_dir)) > 0:
            print(f"  {split}/ already exists with {len(os.listdir(split_dir))} files, skipping...")
            continue
            
        print(f"  Generating {num_samples} {split} samples...")
        generate_dataset(
            num_samples=num_samples,
            save_dir=split_dir,
            spectral_length=data_config["spectral_length"],
            ppm_range=tuple(data_config["ppm_range"]),
            num_compounds_range=tuple(data_config["num_compounds_range"]),
            num_mixtures_range_offset=tuple(data_config["num_mixtures_range_offset"]),
            noise_std=data_config["noise_std"],
            seed=config["seed"]
        )
    
    print("Data generation complete!\n")


def train_model(config: dict):
    """Train the MixNet model."""
    print("=" * 60)
    print("TRAINING MIXNET")
    print("=" * 60)
    
    device = get_device(config)
    data_config = config["data"]
    train_config = config["training"]
    dummy_config = config["dummy"]
    
    # Load datasets
    base_dir = os.path.join(PROJECT_ROOT, dummy_config["save_dir"])
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    if not os.path.exists(train_dir):
        print("ERROR: Training data not found! Run with --mode generate first.")
        return
    
    print(f"Loading training data from {train_dir}...")
    train_dataset = NMRMixtureDataset(train_dir)
    val_dataset = NMRMixtureDataset(val_dir)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Use global max dimensions from dataset for consistent model size
    max_m = train_dataset.global_max_m
    max_n = train_dataset.global_max_n
    spectral_length = train_dataset.spectral_length
    
    print(f"  Global Max mixtures (M): {max_m}")
    print(f"  Global Max compounds (N): {max_n}")
    print(f"  Spectral length (L): {spectral_length}")
    
    # Create model
    model_config = config["model"]
    model = MixNet(
        in_channels=max_m,
        out_channels=max_n,
        encoder_channels=model_config["encoder_channels"],
        kernel_size=model_config["kernel_size"],
        pool_kernel=model_config["pool_kernel"],
        pool_stride=model_config["pool_stride"],
        use_batch_norm=model_config["use_batch_norm"],
        activation=model_config["activation"]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_config['name']}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Input: ({max_m}, {spectral_length}) → Output: ({max_n}, {spectral_length})")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
    
    print("\nTraining complete!")
    print(f"Best model saved to: {os.path.join(train_config['checkpoint_dir'], 'best_model.pth')}")


def evaluate_model(config: dict):
    """Evaluate the trained model."""
    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    device = get_device(config)
    dummy_config = config["dummy"]
    train_config = config["training"]
    
    checkpoint_path = os.path.join(PROJECT_ROOT, train_config["checkpoint_dir"], "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No trained model found at {checkpoint_path}")
        print("Run with --mode train first.")
        return
    
    # Load test data
    test_dir = os.path.join(PROJECT_ROOT, dummy_config["save_dir"], "test")
    test_dataset = NMRMixtureDataset(test_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Get dimensions from data
    first_batch = next(iter(test_loader))
    max_m = first_batch['mixtures'].shape[1]
    max_n = first_batch['compounds'].shape[1]
    
    # Create model and load weights
    model_config = config["model"]
    model = MixNet(
        in_channels=max_m,
        out_channels=max_n,
        encoder_channels=model_config["encoder_channels"],
        kernel_size=model_config["kernel_size"],
        pool_kernel=model_config["pool_kernel"],
        pool_stride=model_config["pool_stride"],
        use_batch_norm=model_config["use_batch_norm"],
        activation=model_config["activation"]
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Run evaluation
    from src.evaluation.evaluate import evaluate_model as run_eval
    ppm_axis = np.linspace(
        config["data"]["ppm_range"][0],
        config["data"]["ppm_range"][1],
        config["data"]["spectral_length"]
    )
    results_dir = os.path.join(PROJECT_ROOT, config["evaluation"]["results_dir"])
    
    results = run_eval(model, test_loader, device, ppm_axis, results_dir)
    
    print("\n" + "=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    for metric, value in results.items():
        print(f"  {metric}: {value:.6f}")


def main():
    parser = argparse.ArgumentParser(description="MixNet — NMR Spectral Decomposition")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["generate", "train", "evaluate", "all"],
                        help="Mode to run: generate data, train model, evaluate, or all")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Set seed
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("=" * 60)
    print("  MixNet — NMR Spectral Decomposition")
    print("  Blind Source Separation for NMR Spectra")
    print("=" * 60)
    print()
    
    if args.mode in ["generate", "all"]:
        generate_data(config)
    
    if args.mode in ["train", "all"]:
        train_model(config)
    
    if args.mode in ["evaluate", "all"]:
        evaluate_model(config)


if __name__ == "__main__":
    main()
