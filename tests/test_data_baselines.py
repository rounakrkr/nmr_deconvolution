import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data.synthetic import (
    MixtureSampler,
    make_mixtures,
    make_unseen_library,
    split_compounds,
)
from src.evaluation.baselines import matched_correlation, nmf_separate, slot_correlation

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


def test_split_disjoint():
    s = split_compounds(30, 4, 6, seed=1)
    all_ids = np.concatenate([s["train"], s["val"], s["test"]])
    assert len(set(all_ids)) == 30
    assert not set(s["test"]) & set(s["train"])


def test_sampler_deterministic_and_consistent(library):
    sampler = MixtureSampler(library, np.arange(8), seed=3)
    a, b = sampler.sample(5), sampler.sample(5)
    assert np.array_equal(a["compound_ids"], b["compound_ids"])
    assert np.allclose(a["concentrations"] @ a["compounds"], a["mixtures"], atol=1e-6)
    assert np.allclose(a["concentrations"].sum(1), 1.0, atol=1e-6)
    assert len(set(a["compound_ids"])) == 5


def test_sampler_respects_pool(library):
    pool = np.array([2, 3, 5, 7, 9, 11])
    sampler = MixtureSampler(library, pool, seed=4)
    for i in range(20):
        assert set(sampler.sample(i)["compound_ids"]) <= set(pool)


def test_slot_order_not_fixed(library):
    sampler = MixtureSampler(library, np.arange(12), seed=5)
    first_slots = {int(sampler.sample(i)["compound_ids"][0]) for i in range(40)}
    assert len(first_slots) > 3


def test_nmf_separates_unseen():
    comp = make_unseen_library(5, length=L, seed=1)
    x, _ = make_mixtures(comp, np.random.RandomState(0))
    assert matched_correlation(comp, nmf_separate(x, 5)).mean() > 0.9


def test_matched_vs_slot_correlation(library):
    t = library[:5]
    shuffled = t[[3, 1, 4, 0, 2]]
    assert matched_correlation(t, shuffled).min() > 0.999
    assert slot_correlation(t, shuffled).mean() < 0.9
