"""
Train MixNet V1/V2/V3 with a permutation-invariant loss on freshly sampled mixtures.

Compounds are split into disjoint train / val / test pools, so the test score
measures separation of compounds never seen during training.

Usage:
    python train.py --model v1 --epochs 60 --patience 15
    python train.py --model v3 --epochs 60 --batch-size 2 --noise 0.003
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.synthetic import MixtureSampler, load_library, split_compounds
from src.data.torch_dataset import SyntheticNMRDataset
from src.evaluation.benchmark import draw, evaluate, format_table
from src.models.factory import MODEL_NAMES, build_model
from src.training.pit import PITLoss


def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "spec": 0.0, "recon": 0.0}
    batches = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            mixtures = batch["mixtures"].to(device)
            compounds = batch["compounds"].to(device)
            conc = batch["concentrations"].to(device)
            pred = model(mixtures)
            loss, parts = criterion(pred, compounds, conc, mixtures)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            for k in totals:
                totals[k] += parts[k]
            batches += 1
    return {k: v / max(batches, 1) for k, v in totals.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_NAMES, required=True)
    p.add_argument("--data-dir", default=os.path.join(ROOT, "NMR_PROJECT_FINAL_PACKAGE"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--lambda-recon", type=float, default=0.5)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--train-size", type=int, default=800)
    p.add_argument("--val-size", type=int, default=100)
    p.add_argument("--test-size", type=int, default=100)
    p.add_argument("--n-val-compounds", type=int, default=6)
    p.add_argument("--n-test-compounds", type=int, default=7)
    p.add_argument("--nmf-samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-dir", default=os.path.join(ROOT, "checkpoints"))
    p.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    names, library = load_library(args.data_dir)
    pools = split_compounds(len(names), args.n_val_compounds, args.n_test_compounds, seed=args.seed)
    for split, ids in pools.items():
        print(f"{split:>5} compounds ({len(ids)}): {', '.join(names[i] for i in ids)}")

    train_set = SyntheticNMRDataset(library, pools["train"], args.train_size, args.noise, seed=1, resample_each_epoch=True)
    val_set = SyntheticNMRDataset(library, pools["val"], args.val_size, args.noise, seed=2)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model).to(device)
    print(f"device={device} model={args.model} parameters={sum(q.numel() for q in model.parameters()):,}")

    criterion = PITLoss(lambda_recon=args.lambda_recon)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"best_{args.model}.pth")
    best_val, stale = float("inf"), 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_set.set_epoch(epoch)
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        improved = va["loss"] < best_val
        if improved:
            best_val, stale = va["loss"], 0
            torch.save(
                {"epoch": epoch + 1, "model_name": args.model, "model_state_dict": model.state_dict(),
                 "val_loss": best_val, "args": vars(args)},
                ckpt_path,
            )
        else:
            stale += 1
        print(
            f"epoch {epoch + 1:>3}/{args.epochs} train {tr['loss']:.6f} (spec {tr['spec']:.6f} recon {tr['recon']:.6f}) "
            f"val {va['loss']:.6f} lr {optimizer.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s "
            f"{'<< best' if improved else f'(stale {stale}/{args.patience})'}"
        )
        if stale >= args.patience:
            print(f"early stop at epoch {epoch + 1}")
            break

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    test_sampler = MixtureSampler(library, pools["test"], noise_std=args.noise, seed=3)
    results = evaluate(draw(test_sampler, args.test_size), model, device, nmf_samples=args.nmf_samples)
    print(f"\nheld-out-compound test (best epoch {state['epoch']}, val loss {state['val_loss']:.6f})")
    print(format_table(results))
    with open(os.path.join(args.results_dir, f"{args.model}_heldout_benchmark.json"), "w") as f:
        json.dump({"args": vars(args), "best_epoch": state["epoch"], "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
