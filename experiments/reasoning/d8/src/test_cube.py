"""Checks of the exact functionals against the original implementation."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import cube  # noqa: E402
from experiments.reasoning.common.query_conditioned_tensor import (  # noqa: E402
    interaction_values,
)


def reference_marginal(values: np.ndarray, players: tuple[int, ...], t: float) -> float:
    """Directly summed E_{S ~ Bernoulli(t)}[Delta_T v(S)], no tensor algebra."""
    total = 0.0
    free = [player for player in range(cube.D) if player not in players]
    for present in itertools.product((0, 1), repeat=len(free)):
        weight = 1.0
        background = 0
        for player, indicator in zip(free, present):
            weight *= t if indicator else (1 - t)
            background |= indicator << player
        delta = 0.0
        for subset_size in range(len(players) + 1):
            for subset in itertools.combinations(players, subset_size):
                mask = background | sum(1 << player for player in subset)
                delta += (-1) ** (len(players) - subset_size) * values[mask]
        total += weight * delta
    return total


def main() -> None:
    rng = np.random.default_rng(20260912)
    values = rng.normal(size=1 << cube.D)
    tensor = cube.as_tensor(values)

    # axis convention: axis i must be the bit-i indicator
    for mask in rng.integers(0, 1 << cube.D, size=32):
        index = tuple((int(mask) >> player) & 1 for player in range(cube.D))
        assert tensor[index] == values[int(mask)], "axis convention broken"

    worst_curve = 0.0
    for players in [(3,), (1, 6), (0, 2, 5)]:
        for t in (0.0, 0.25, 0.5, 1.0):
            mine = float(cube.marginal_curve(tensor, players, np.asarray([t]))[0])
            worst_curve = max(worst_curve, abs(mine - reference_marginal(values, players, t)))

    # Shapley and Banzhaf interaction indices against query_conditioned_tensor
    worst_shapley = worst_banzhaf = 0.0
    for order in (1, 2, 3):
        combos, shapley = interaction_values(values, order, True)
        _, banzhaf = interaction_values(values, order, False)
        for players, reference in zip(combos, shapley):
            worst_shapley = max(worst_shapley,
                                abs(float(cube.shapley_interaction(tensor, players)) - reference))
        for players, reference in zip(combos, banzhaf):
            mine = float(cube.marginal_curve(tensor, players, np.asarray([0.5]))[0])
            worst_banzhaf = max(worst_banzhaf, abs(mine - reference))

    # LOO identity: g_T(1) is the finite difference at the full background
    worst_loo = 0.0
    for players in [(2,), (4, 7), (1, 3, 6)]:
        background = sum(1 << player for player in range(cube.D) if player not in players)
        delta = 0.0
        for subset_size in range(len(players) + 1):
            for subset in itertools.combinations(players, subset_size):
                mask = background | sum(1 << player for player in subset)
                delta += (-1) ** (len(players) - subset_size) * values[mask]
        mine = float(cube.marginal_curve(tensor, players, np.asarray([1.0]))[0])
        worst_loo = max(worst_loo, abs(mine - delta))

    # conditioning: g_i(t | j present) mixed with g_i(t | j absent) reproduces g_i(t)
    t = np.linspace(0, 1, 21)
    plain = cube.marginal_curve(tensor, (2,), t)
    present = cube.marginal_curve(tensor, (2,), t, condition={5: 1})
    absent = cube.marginal_curve(tensor, (2,), t, condition={5: 0})
    worst_condition = float(np.max(np.abs(plain - (t * present + (1 - t) * absent))))

    # multi-output arrays must behave like independent scalar games
    stacked = np.stack((values, rng.normal(size=1 << cube.D)), axis=1)
    stacked_tensor = cube.as_tensor(stacked)
    worst_stack = float(np.max(np.abs(
        cube.marginal_curve(stacked_tensor, (1, 4), t)[:, 0]
        - cube.marginal_curve(tensor, (1, 4), t))))

    print(f"curve vs direct sum      {worst_curve:.3e}")
    print(f"shapley vs original      {worst_shapley:.3e}")
    print(f"banzhaf vs original      {worst_banzhaf:.3e}")
    print(f"loo identity             {worst_loo:.3e}")
    print(f"conditioning mixture     {worst_condition:.3e}")
    print(f"multi-output consistency {worst_stack:.3e}")
    worst = max(worst_curve, worst_shapley, worst_banzhaf, worst_loo, worst_condition, worst_stack)
    assert worst < 1e-10, f"exact functionals disagree: {worst}"
    print("OK")


if __name__ == "__main__":
    main()
