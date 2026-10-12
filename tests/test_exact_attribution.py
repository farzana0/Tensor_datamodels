"""Exact Shapley / Banzhaf / interaction computations against exhaustive references."""
import itertools
from math import factorial

import numpy as np
import pytest

from conftest import brute_force_banzhaf, brute_force_shapley, discrete_derivative
from tndm.attribution import exact


@pytest.fixture(params=[3, 5, 6])
def game(request):
    d = request.param
    return d, np.random.default_rng(100 + d).normal(size=1 << d)


def test_shapley_matches_permutation_enumeration(game):
    d, values = game
    phi = exact.shapley_values(values)
    assert np.allclose(phi, brute_force_shapley(values, d))
    assert np.isclose(phi.sum(), values[-1] - values[0])          # efficiency


def test_banzhaf_matches_subset_enumeration(game):
    d, values = game
    assert np.allclose(exact.banzhaf_values(values), brute_force_banzhaf(values, d))


def test_leave_one_out(game):
    d, values = game
    full = (1 << d) - 1
    expected = [values[full] - values[full & ~(1 << i)] for i in range(d)]
    assert np.allclose(exact.loo_values(values), expected)


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("shapley", [True, False])
def test_interaction_indices_match_definition(game, order, shapley):
    """I_A = sum_{S subseteq N minus A} w(|S|) Delta_A v(S), with Shapley or Banzhaf weights."""
    d, values = game
    subsets = list(itertools.combinations(range(d), order))
    computed = (exact.pair_interactions if order == 2 else exact.triple_interactions)(values, shapley)
    for index, members in enumerate(subsets):
        rest = [p for p in range(d) if p not in members]
        total = 0.0
        for size in range(len(rest) + 1):
            weight = (factorial(size) * factorial(d - size - order) / factorial(d - order + 1)
                      if shapley else 2.0 ** -(d - order))
            for background in itertools.combinations(rest, size):
                mask = sum(1 << p for p in background)
                total += weight * discrete_derivative(values, members, mask)
        assert np.isclose(computed[index], total)


def test_mobius_coefficients_reconstruct_the_game(game):
    d, values = game
    coefficients = exact.mobius(values)
    masks = np.arange(1 << d)
    for mask in masks:
        subsets = [t for t in masks if t & mask == t]
        assert np.isclose(values[mask], coefficients[subsets].sum())


def test_multi_output_tables_are_handled_columnwise(game):
    d, values = game
    table = np.column_stack([values, 2.0 * values + 1.0])
    phi = exact.shapley_values(table)
    assert phi.shape == (d, 2)
    assert np.allclose(phi[:, 1], 2.0 * phi[:, 0])


def test_known_closed_forms():
    # three-way conjunction: every Shapley value is 1/3, the triple interaction is 1
    d = 3
    values = np.zeros(1 << d)
    values[-1] = 1.0
    assert np.allclose(exact.shapley_values(values), 1.0 / 3.0)
    assert np.allclose(exact.triple_interactions(values), 1.0)
    # redundant pair v = z0 OR z1: pairwise interaction is -1 (pure redundancy)
    table = np.array([0.0, 1.0, 1.0, 1.0])
    assert np.allclose(exact.pair_interactions(table), -1.0)
    assert np.allclose(exact.shapley_values(table), 0.5)
