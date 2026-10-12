"""Coalition sampling designs.  All of them are outcome-free.

A design returns exactly ``m`` distinct masks; those are the coalitions a method
is allowed to look up.  The cost unit everywhere in equal-budget study is one distinct coalition
evaluation, because in the retraining setting that is one retraining and repeated
lookups of the same coalition are free.
"""
from __future__ import annotations

import numpy as np

from . import exact


def uniform(d: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """m distinct masks drawn uniformly without replacement from all 2**d."""
    total = 1 << d
    if m > total:
        raise ValueError(f"cannot draw {m} distinct masks from {total}")
    return np.sort(rng.choice(total, size=m, replace=False))


def size_stratified(d: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """m distinct masks spread as evenly as possible over cardinalities 0..d."""
    size = exact.popcounts(d)
    strata = [np.flatnonzero(size == k) for k in range(d + 1)]
    capacity = np.asarray([len(s) for s in strata])
    take = np.zeros(d + 1, dtype=int)
    remaining = m
    while remaining > 0:
        room = np.flatnonzero(take < capacity)
        share = max(1, remaining // len(room))
        for stratum in room:
            add = min(share, capacity[stratum] - take[stratum], remaining)
            take[stratum] += add
            remaining -= add
            if remaining == 0:
                break
    chosen = [rng.choice(strata[k], size=take[k], replace=False) for k in range(d + 1) if take[k]]
    return np.sort(np.concatenate(chosen))


def _degree_terms(d: int, degree: int) -> list[tuple[int, ...]]:
    import itertools
    return [()] + [term for order in range(1, degree + 1)
                   for term in itertools.combinations(range(d), order)]


def monomial_design(masks: np.ndarray, d: int, degree: int) -> np.ndarray:
    bits = ((np.asarray(masks)[:, None] >> np.arange(d)) & 1).astype(float)
    terms = _degree_terms(d, degree)
    return np.column_stack([bits[:, list(term)].prod(1) if term else np.ones(len(masks))
                            for term in terms])


def d_optimal(d: int, m: int, rng: np.random.Generator, degree: int = 3) -> np.ndarray:
    """Greedy D-optimal (max log-det) design for the degree-``degree`` monomial basis.

    Outcome-free: it uses only the design matrix.  Seeded by a random start so the
    design still has sampling variability across repeats.
    """
    total = 1 << d
    full = monomial_design(np.arange(total), d, degree)
    p = full.shape[1]
    chosen = [int(i) for i in rng.choice(total, size=min(m, 2), replace=False)]
    inverse = np.eye(p) / 1e-6
    # Sherman-Morrison: add each row to the information matrix, keep its inverse
    # and every candidate's leverage x_i^T A^-1 x_i up to date in O(N p).
    gain = np.einsum("ij,ij->i", full @ inverse, full)
    available = np.ones(total, dtype=bool)

    def add(index: int) -> None:
        nonlocal inverse, gain
        row = full[index]
        projected = inverse @ row
        denominator = 1.0 + row @ projected
        inverse = inverse - np.outer(projected, projected) / denominator
        gain = gain - (full @ projected) ** 2 / denominator
        available[index] = False

    for index in chosen:
        add(index)
    while len(chosen) < m:
        score = np.where(available, gain, -np.inf)
        best = int(np.argmax(score))
        chosen.append(best)
        add(best)
    return np.sort(np.asarray(chosen))


DESIGNS = {"uniform": uniform, "size_stratified": size_stratified, "d_optimal": d_optimal}


def stratified_fit_validation(masks: np.ndarray, d: int, seed: int,
                              fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Split sampled masks into fit/validation within each cardinality stratum.

    This mirrors the split used by the frozen 14_ sample-efficiency replay so that
    hyperparameter selection never sees a coalition outside the sampled budget.
    """
    rng = np.random.default_rng(seed)
    fit, validation = [], []
    sizes = exact.popcounts(d)[masks]
    for size in np.unique(sizes):
        group = rng.permutation(masks[sizes == size])
        count = max(1, round(fraction * len(group))) if len(group) >= 4 else 0
        validation.extend(group[:count])
        fit.extend(group[count:])
    if not validation:
        validation.append(fit.pop())
    if not fit:
        fit.append(validation.pop())
    return np.sort(np.asarray(fit)), np.sort(np.asarray(validation))
