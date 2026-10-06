"""
Train on endlessly generated random compounds (zero real library compounds in training),
then test on ALL 30 real library compounds (every one is unseen) and on a random blind set.
Evaluates at several noise levels against NMF.

Usage:
    python train_procedural.py --model v1 --steps 6000 --batch-size 4
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.synthetic import (SPECTRAL_LENGTH, MixtureSampler, load_library, lorentz_multiplet,
                                make_mixtures, make_unseen_library)
from src.evaluation.benchmark import draw, evaluate, format_table
from src.models.factory import MODEL_NAMES, build_model
from src.training.pit import PITLoss

PPM = np.linspace(10.0, 0.0, SPECTRAL_LENGTH)


def random_compound(rng: np.random.RandomState) -> np.ndarray:
    """One random compound: 1-6 multiplets, mixed linewidths, optional broad peak."""
    s = np.zeros(SPECTRAL_LENGTH)
    width = rng.uniform(0.008, 0.04)  # library lines are ~0.03 ppm wide
    for _ in range(rng.randint(1, 7)):
        s += lorentz_multiplet(
            PPM, center=rng.uniform(0.3, 9.7), n_lines=int(rng.choice([1, 1, 2, 3, 4, 5, 6])),
            j_hz=rng.uniform(4.0, 12.0), width_ppm=width * rng.uniform(0.7, 1.6),
            height=rng.uniform(0.2, 1.0))
    if rng.rand() < 0.25:  # broad exchangeable OH/NH-like peak
        s += lorentz_multiplet(PPM, rng.uniform(1.0, 8.0), 1, 0.0, rng.uniform(0.08, 0.3), rng.uniform(0.2, 0.8))
    return (s / s.max()).astype(np.float32)


class ProceduralDataset(torch.utils.data.Dataset):
    def __init__(self, length, noise_max=0.01, seed=0, n_comp=5, n_mix=20):
        self.length, self.noise_max, self.seed = length, noise_max, seed
        self.n_comp, self.n_mix, self.offset = n_comp, n_mix, 0

    def set_epoch(self, e):
        self.offset = e * self.length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.RandomState(self.seed * 1_000_003 + self.offset + i)
        s = np.stack([random_compound(rng) for _ in range(self.n_comp)])
        noise = rng.uniform(0, self.noise_max) if rng.rand() < 0.7 else 0.0
        x, a = make_mixtures(s, rng, self.n_mix, noise_std=noise, alpha=float(rng.choice([0.5, 1.0, 2.0])))
        return {"mixtures": torch.from_numpy(x), "compounds": torch.from_numpy(s),
                "concentrations": torch.from_numpy(a)}


def run_epoch(model, loader, crit, device, opt=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for b in loader:
            m, c, a = (b[k].to(device) for k in ("mixtures", "compounds", "concentrations"))
            loss, parts = crit(model(m), c, a, m)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += parts["loss"]
            n += 1
    return tot / max(n, 1)


def real_library_samples(library, n, noise, seed):
    sampler = MixtureSampler(library, np.arange(len(library)), noise_std=noise, seed=seed)
    return draw(sampler, n)


def blind_samples(n, noise):
    out = []
    for seed in range(n):
        comp = make_unseen_library(5, seed=seed, width_ppm=0.02)
        x, a = make_mixtures(comp, np.random.RandomState(10_000 + seed), noise_std=noise)
        out.append({"mixtures": x, "compounds": comp, "concentrations": a})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_NAMES, required=True)
    p.add_argument("--data-dir", default=os.path.join(ROOT, "NMR_PROJECT_FINAL_PACKAGE"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--samples-per-epoch", type=int, default=400)
    p.add_argument("--val-size", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--noise-max", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--test-size", type=int, default=60)
    p.add_argument("--nmf-samples", type=int, default=20)
    p.add_argument("--noise-levels", type=float, nargs="*", default=[0.0, 0.003, 0.01])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-dir", default=os.path.join(ROOT, "checkpoints"))
    p.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, library = load_library(args.data_dir)

    train_set = ProceduralDataset(args.samples_per_epoch, args.noise_max, seed=1)
    val_set = ProceduralDataset(args.val_size, args.noise_max, seed=2)
    tl = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    vl = torch.utils.data.DataLoader(val_set, args.batch_size, num_workers=args.num_workers)

    model = build_model(args.model).to(device)
    print(f"device={device} model={args.model} params={sum(q.numel() for q in model.parameters()):,}")
    crit = PITLoss(lambda_recon=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    ckpt = os.path.join(args.checkpoint_dir, f"proc_{args.model}.pth")
    best, stale = float("inf"), 0
    for ep in range(args.epochs):
        t0 = time.time()
        train_set.set_epoch(ep)
        tr = run_epoch(model, tl, crit, device, opt)
        va = run_epoch(model, vl, crit, device)
        sched.step()
        imp = va < best
        if imp:
            best, stale = va, 0
            torch.save({"epoch": ep + 1, "model_name": args.model, "model_state_dict": model.state_dict(),
                        "val_loss": va, "args": vars(args)}, ckpt)
        else:
            stale += 1
        print(f"epoch {ep + 1:>3}/{args.epochs} train {tr:.6f} val {va:.6f} {time.time() - t0:.0f}s "
              f"{'<< best' if imp else f'(stale {stale}/{args.patience})'}", flush=True)
        if stale >= args.patience:
            print("early stop")
            break

    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    report = {}
    for noise in args.noise_levels:
        for tag, samples in (("real_library", real_library_samples(library, args.test_size, noise, seed=7)),
                             ("random_blind", blind_samples(args.test_size, noise))):
            res = evaluate(samples, model, device, nmf_samples=args.nmf_samples)
            report[f"{tag}_noise{noise}"] = res
            print(f"\n[{tag}] noise={noise}  (trained on 0 real compounds)")
            print(format_table(res), flush=True)
    with open(os.path.join(args.results_dir, f"proc_{args.model}.json"), "w") as f:
        json.dump({"args": vars(args), "best_epoch": state["epoch"], "report": report}, f, indent=2)


if __name__ == "__main__":
    main()
