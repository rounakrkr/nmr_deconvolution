import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

torch = pytest.importorskip("torch")

from src.data.synthetic import MixtureSampler
from src.models.factory import build_model
from src.training.pit import PITLoss, match_sources, reorder

L = 2048


@pytest.fixture(scope="module")
def library():
    rng = np.random.RandomState(0)
    lib = np.zeros((12, L), dtype=np.float32)
    for k in range(12):
        for c in rng.choice(L, size=4, replace=False):
            lib[k] += np.exp(-0.5 * ((np.arange(L) - c) / 6.0) ** 2)
        lib[k] /= lib[k].max()
    return lib



def _batch(library, b=2):
    sampler = MixtureSampler(library, np.arange(12), seed=7)
    ds = [sampler.sample(i) for i in range(b)]
    return tuple(
        torch.from_numpy(np.stack([d[k] for d in ds])) for k in ("mixtures", "compounds", "concentrations")
    )


def test_pit_invariant_to_target_permutation(library):
    x, s, a = _batch(library)
    pred = s + 0.05 * torch.randn_like(s)
    crit = PITLoss()
    base, _ = crit(pred, s, a, x)
    perm = torch.tensor([3, 0, 4, 1, 2])
    shuffled_pred = pred[:, perm]
    permuted, _ = crit(shuffled_pred, s, a, x)
    assert torch.allclose(base, permuted, atol=1e-6)


def test_pit_zero_for_perfect_permuted_prediction(library):
    x, s, a = _batch(library)
    loss, parts = PITLoss()(s[:, [4, 2, 0, 1, 3]], s, a, x)
    assert parts["spec"] < 1e-10 and parts["recon"] < 1e-10


def test_match_sources_recovers_permutation(library):
    _, s, _ = _batch(library)
    perm = torch.tensor([2, 0, 4, 1, 3])
    pred = s[:, perm]
    found = match_sources(pred, s)
    assert torch.allclose(reorder(pred, found), s)


@pytest.mark.parametrize("name", ["v1", "v2", "v3"])
def test_model_forward_and_grad(library, name):
    x, s, a = _batch(library)
    model = build_model(name)
    pred = model(x)
    assert pred.shape == (2, 5, L)
    assert (pred >= 0).all()
    loss, _ = PITLoss()(pred, s, a, x)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_v3_invariant_to_mixture_order(library):
    x, _, _ = _batch(library)
    model = build_model("v3").eval()
    with torch.no_grad():
        out = model(x)
        out_perm = model(x[:, torch.randperm(x.shape[1])])
    assert torch.allclose(out, out_perm, atol=1e-4)
