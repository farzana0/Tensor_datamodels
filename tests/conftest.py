"""Shared fixtures: small synthetic set functions and fitted datamodels."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

torch.set_num_threads(1)


def brute_force_shapley(values: np.ndarray, d: int) -> np.ndarray:
    """Shapley values by averaging marginal contributions over all d! orderings."""
    phi = np.zeros(d)
    orders = list(itertools.permutations(range(d)))
    for order in orders:
        mask = 0
        for player in order:
            phi[player] += values[mask | (1 << player)] - values[mask]
            mask |= 1 << player
    return phi / len(orders)


def brute_force_banzhaf(values: np.ndarray, d: int) -> np.ndarray:
    beta = np.zeros(d)
    for player in range(d):
        for mask in range(1 << d):
            if not mask & (1 << player):
                beta[player] += values[mask | (1 << player)] - values[mask]
    return beta / 2 ** (d - 1)


def discrete_derivative(values: np.ndarray, members: tuple[int, ...], background: int) -> float:
    """Delta_A v(S) = sum_{B subseteq A} (-1)^{|A|-|B|} v(S union B)."""
    total = 0.0
    for take in range(len(members) + 1):
        for chosen in itertools.combinations(members, take):
            mask = background
            for player in chosen:
                mask |= 1 << player
            total += (-1.0) ** (len(members) - take) * values[mask]
    return total


@pytest.fixture(scope="session")
def and_of_ors():
    """v(S) = 4 * [S hits {0,1}] * [S hits {2,3}] + small additive part, d = 6."""
    d = 6
    masks = np.arange(1 << d, dtype=np.int64)
    bits = ((masks[:, None] >> np.arange(d)) & 1).astype(float)
    values = 4.0 * np.maximum(bits[:, 0], bits[:, 1]) * np.maximum(bits[:, 2], bits[:, 3])
    values = values + bits @ np.array([0.3, -0.2, 0.1, 0.05, 0.4, -0.1])
    return d, masks, values


@pytest.fixture(scope="session")
def fitted(and_of_ors):
    """One fitted model per tensor-network family, on 48 of the 64 coalitions."""
    from tndm import fit_datamodel

    d, masks, values = and_of_ors
    rng = np.random.default_rng(0)
    observed = np.sort(rng.choice(masks, size=48, replace=False))
    y = np.column_stack([values[observed], 0.5 * values[observed] - 1.0])  # two outcomes
    orders = (tuple(range(d)), (0, 2, 4, 1, 3, 5))
    models = {}
    for family in ("cp", "tt", "ht"):
        models[family], _ = fit_datamodel(observed, y, d, family, split_seed=3,
                                          orders=orders if family == "tt" else None, ranks=(2, 4))
    return d, masks, models
