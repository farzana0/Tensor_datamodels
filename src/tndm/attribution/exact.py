"""Exact game-theoretic quantities from a dense coalition table.

Everything here enumerates the full cube, so the values are exact up to floating
point.  All functions are vectorised over the output axis: a table of shape
``(2 ** d, K)`` yields attributions of shape ``(d, K)`` and pair interactions of
shape ``(n_pairs, K)``.  The same functions are applied to a surrogate's dense
prediction, which is what makes "fit once, query everything" comparable to a
direct estimator at equal budget.

Conventions (checked against shapiq's ExactComputer in ``validate_exact.py``):

* Shapley value   phi_i      = sum_{S not containing i} |S|!(d-|S|-1)!/d! * (v(S+i) - v(S))
* Banzhaf value   beta_i     = 2^-(d-1) sum_{S not containing i} (v(S+i) - v(S))
* LOO             Delta_i    = v(N) - v(N\\{i})   (positive = removing i hurts)
* SII order 2     I_ij       = sum_{S subset N\\{i,j}} |S|!(d-|S|-2)!/(d-1)! * delta_ij v(S)
* BII order 2     B_ij       = 2^-(d-2) sum_S delta_ij v(S)
* SII/BII order 3 use the analogous weights and the third-order difference.
"""
from __future__ import annotations

import itertools
from math import comb, factorial

import numpy as np

_POPCOUNT_CACHE: dict[int, np.ndarray] = {}


def popcounts(d: int) -> np.ndarray:
    if d not in _POPCOUNT_CACHE:
        masks = np.arange(1 << d, dtype=np.int64)
        size = np.zeros(1 << d, dtype=np.int64)
        for bit in range(d):
            size += (masks >> bit) & 1
        _POPCOUNT_CACHE[d] = size
    return _POPCOUNT_CACHE[d]


def as_matrix(values: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array[:, None], True
    if array.ndim != 2:
        raise ValueError("values must be (2**d,) or (2**d, outputs)")
    return array, False


def dimension(values: np.ndarray) -> int:
    rows = np.asarray(values).shape[0]
    d = int(round(np.log2(rows)))
    if (1 << d) != rows:
        raise ValueError(f"table length {rows} is not a power of two")
    return d


def shapley_weights(d: int) -> np.ndarray:
    """Weight attached to a background of each size, for a first-order index."""
    return np.asarray([factorial(s) * factorial(d - s - 1) / factorial(d) for s in range(d)])


def _first_order(values: np.ndarray, weight_by_size: np.ndarray) -> np.ndarray:
    matrix, scalar = as_matrix(values)
    d = dimension(matrix)
    size = popcounts(d)
    result = np.zeros((d, matrix.shape[1]))
    for player in range(d):
        bit = 1 << player
        without = np.flatnonzero((np.arange(1 << d) & bit) == 0)
        delta = matrix[without | bit] - matrix[without]
        result[player] = weight_by_size[size[without]] @ delta
    return result[:, 0] if scalar else result


def shapley_values(values: np.ndarray) -> np.ndarray:
    d = dimension(values)
    return _first_order(values, shapley_weights(d))


def banzhaf_values(values: np.ndarray) -> np.ndarray:
    d = dimension(values)
    return _first_order(values, np.full(d, 2.0 ** -(d - 1)))


def loo_values(values: np.ndarray) -> np.ndarray:
    """v(N) - v(N minus i): positive when removing the group hurts."""
    matrix, scalar = as_matrix(values)
    d = dimension(matrix)
    full = (1 << d) - 1
    result = np.stack([matrix[full] - matrix[full & ~(1 << player)] for player in range(d)])
    return result[:, 0] if scalar else result


def pairs(d: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(d), 2))


def triples(d: int) -> list[tuple[int, int, int]]:
    return list(itertools.combinations(range(d), 3))


def _interaction_weights(d: int, order: int, shapley: bool) -> np.ndarray:
    sizes = np.arange(d - order + 1)
    if not shapley:
        return np.full(len(sizes), 2.0 ** -(d - order))
    return np.asarray([factorial(s) * factorial(d - s - order) / factorial(d - order + 1)
                       for s in sizes])


def _higher_order(values: np.ndarray, subsets, order: int, shapley: bool) -> np.ndarray:
    matrix, scalar = as_matrix(values)
    d = dimension(matrix)
    size = popcounts(d)
    weight = _interaction_weights(d, order, shapley)
    result = np.zeros((len(subsets), matrix.shape[1]))
    all_masks = np.arange(1 << d)
    for index, members in enumerate(subsets):
        joint = 0
        for player in members:
            joint |= 1 << player
        background = np.flatnonzero((all_masks & joint) == 0)
        delta = np.zeros((len(background), matrix.shape[1]))
        for take in range(order + 1):
            sign = (-1.0) ** (order - take)
            for chosen in itertools.combinations(members, take):
                add = 0
                for player in chosen:
                    add |= 1 << player
                delta += sign * matrix[background | add]
        result[index] = weight[size[background]] @ delta
    return result[:, 0] if scalar else result


def pair_interactions(values: np.ndarray, shapley: bool = True) -> np.ndarray:
    d = dimension(values)
    return _higher_order(values, pairs(d), 2, shapley)


def triple_interactions(values: np.ndarray, shapley: bool = True) -> np.ndarray:
    d = dimension(values)
    return _higher_order(values, triples(d), 3, shapley)


def mobius(values: np.ndarray) -> np.ndarray:
    """Fast zeta/Moebius transform: a(S) = sum_{T subset S} (-1)^{|S|-|T|} v(T)."""
    matrix, scalar = as_matrix(values)
    d = dimension(matrix)
    result = matrix.astype(float).copy()
    for bit in range(d):
        step = 1 << bit
        index = np.flatnonzero((np.arange(1 << d) & step) != 0)
        result[index] -= result[index & ~step]
    return result[:, 0] if scalar else result


def degree_energy(values: np.ndarray, max_degree: int = 3) -> np.ndarray:
    """Fraction of squared Moebius mass at degrees 1..max_degree and above."""
    coefficients = mobius(values)
    matrix, _ = as_matrix(coefficients)
    d = dimension(matrix)
    size = popcounts(d)
    energy = (matrix ** 2)
    energy[0] = 0.0  # the empty-set offset carries no interaction information
    total = energy.sum(0)
    total[total == 0] = 1.0
    out = []
    for degree in range(1, max_degree + 1):
        out.append(energy[size == degree].sum(0) / total)
    out.append(energy[size > max_degree].sum(0) / total)
    return np.asarray(out)


def shapley_kernel(d: int) -> np.ndarray:
    """mu(|S|) for interior coalition sizes; 0 at the empty and grand coalitions."""
    weight = np.zeros(d + 1)
    for size in range(1, d):
        weight[size] = (d - 1) / (comb(d, size) * size * (d - size))
    return weight


def fsii(values: np.ndarray, order: int = 2) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Faithful Shapley Interaction Index: best weighted degree-<=order surrogate.

    Solves the Shapley-kernel weighted least squares over the *whole* cube under
    the two efficiency constraints (the empty coalition reproduces v(empty) and
    the full coalition reproduces v(N)).  Returns the interaction terms of the
    requested order together with their coefficients.
    """
    matrix, scalar = as_matrix(values)
    d = dimension(matrix)
    terms = [()] + [t for o in range(1, order + 1) for t in itertools.combinations(range(d), o)]
    all_masks = np.arange(1 << d)
    bits = ((all_masks[:, None] >> np.arange(d)) & 1).astype(float)
    design = np.column_stack([bits[:, list(t)].prod(1) if t else np.ones(1 << d) for t in terms])
    weight = shapley_kernel(d)[popcounts(d)]
    interior = weight > 0
    x = design[interior]
    w = weight[interior]
    gram = x.T @ (w[:, None] * x)
    rhs = x.T @ (w[:, None] * matrix[interior])
    constraint = np.vstack([design[0], design[(1 << d) - 1]])
    target = np.vstack([matrix[0], matrix[(1 << d) - 1]])
    size = gram.shape[0]
    kkt = np.zeros((size + 2, size + 2))
    kkt[:size, :size] = gram + 1e-12 * np.eye(size)
    kkt[:size, size:] = constraint.T
    kkt[size:, :size] = constraint
    right = np.vstack([rhs, target])
    solution = np.linalg.solve(kkt, right)[:size]
    selected = [index for index, term in enumerate(terms) if len(term) == order]
    coefficients = solution[selected]
    return [terms[index] for index in selected], (coefficients[:, 0] if scalar else coefficients)


def best_set_of_size(values: np.ndarray, k: int) -> np.ndarray:
    """Index of the highest-value coalition of size k, per output."""
    matrix, scalar = as_matrix(values)
    d = dimension(matrix)
    candidates = np.flatnonzero(popcounts(d) == k)
    best = candidates[np.argmax(matrix[candidates], axis=0)]
    return best[0] if scalar else best


def selection_regret(truth: np.ndarray, prediction: np.ndarray, k: int) -> np.ndarray:
    """True utility lost by taking the surrogate's best size-k set instead of the best one."""
    truth_matrix, scalar = as_matrix(truth)
    prediction_matrix, _ = as_matrix(prediction)
    chosen = np.atleast_1d(best_set_of_size(prediction_matrix, k))
    oracle = np.atleast_1d(best_set_of_size(truth_matrix, k))
    columns = np.arange(truth_matrix.shape[1])
    regret = truth_matrix[oracle, columns] - truth_matrix[chosen, columns]
    return regret[0] if scalar else regret
