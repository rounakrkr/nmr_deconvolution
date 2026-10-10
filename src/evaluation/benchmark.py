from typing import Dict, List, Optional

import numpy as np
import torch

from src.data.synthetic import MixtureSampler
from src.evaluation.ghost import ghost_metrics
from src.evaluation.baselines import (
    matched_correlation,
    mean_spectrum_baseline,
    nmf_separate,
    slot_correlation,
    summarize,
)


def draw(sampler: MixtureSampler, n: int) -> List[Dict[str, np.ndarray]]:
    return [sampler.sample(i) for i in range(n)]


@torch.no_grad()
def predict(model: torch.nn.Module, mixtures: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(mixtures).unsqueeze(0).to(device)
    return model(x).squeeze(0).cpu().numpy()


def evaluate(
    samples: List[Dict[str, np.ndarray]],
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    nmf_samples: int = 30,
    nmf_iters: int = 800,
    baselines: bool = True,
    ghost: bool = True,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if model is not None:
        device = device or torch.device("cpu")
        matched, slot = [], []
        ghosts = {}
        for d in samples:
            pred = predict(model, d["mixtures"], device)
            matched.append(matched_correlation(d["compounds"], pred).mean())
            slot.append(slot_correlation(d["compounds"], pred).mean())
            if ghost:
                for k, v in ghost_metrics(d["compounds"], pred).items():
                    ghosts.setdefault(k, []).append(v)
        out["model_matched"] = summarize(matched)
        out["model_slot"] = summarize(slot)
        for k, v in ghosts.items():
            out[f"model_{k}"] = summarize(v)
    if not baselines:
        return out
    subset = samples[:nmf_samples]
    n_src = samples[0]["compounds"].shape[0]
    out["mean_spectrum"] = summarize(
        [matched_correlation(d["compounds"], mean_spectrum_baseline(d["mixtures"], n_src)).mean() for d in subset]
    )
    out["nmf"] = summarize(
        [
            matched_correlation(d["compounds"], nmf_separate(d["mixtures"], n_src, iters=nmf_iters)).mean()
            for d in subset
        ]
    )
    return out


def format_table(results: Dict[str, dict]) -> str:
    lines = [f"{'method':<30}{'mean':>8}{'±95%CI':>9}{'min':>8}{'max':>8}{'n':>5}"]
    for name, r in results.items():
        lines.append(f"{name:<30}{r['mean']:>8.4f}{r['ci95']:>9.4f}{r['min']:>8.4f}{r['max']:>8.4f}{r['n']:>5}")
    return "\n".join(lines)
