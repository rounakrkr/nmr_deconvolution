"""
Open-world blind test on novel compounds (random Lorentzian multiplets, realistic J-couplings),
repeated over many seeds, with NMF and mean-spectrum baselines.

Usage:
    python blind_test.py --checkpoints checkpoints/best_v1.pth checkpoints/best_v2.pth checkpoints/best_v3.pth
    python blind_test.py --seeds 50 --noise 0.003
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.synthetic import make_mixtures, make_unseen_library
from src.evaluation.benchmark import evaluate, format_table
from src.models.factory import build_model


def build_samples(seeds: int, noise: float, binomial: bool = False):
    samples = []
    for seed in range(seeds):
        compounds = make_unseen_library(5, seed=seed, binomial=binomial)
        mixtures, conc = make_mixtures(compounds, np.random.RandomState(10_000 + seed), noise_std=noise)
        samples.append({"mixtures": mixtures, "compounds": compounds, "concentrations": conc})
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--binomial-multiplets", action="store_true", help="opt-in Pascal-weighted multiplets (default: legacy flat)")
    p.add_argument("--nmf-samples", type=int, default=30)
    p.add_argument("--output", default=os.path.join(ROOT, "results", "blind_test.json"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = build_samples(args.seeds, args.noise, args.binomial_multiplets)
    report = {}

    baselines = evaluate(samples, None, nmf_samples=args.nmf_samples)
    report["baselines"] = baselines
    print("baselines (novel compounds)")
    print(format_table(baselines))

    for path in args.checkpoints:
        state = torch.load(path, map_location=device, weights_only=True)
        model = build_model(state["model_name"]).to(device)
        model.load_state_dict(state["model_state_dict"])
        entry = evaluate(samples, model, device, baselines=False)
        report[state["model_name"]] = entry
        print(f"\n{state['model_name']} ({os.path.basename(path)})")
        print(format_table(entry))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"seeds": args.seeds, "noise": args.noise, "report": report}, f, indent=2)


if __name__ == "__main__":
    main()
