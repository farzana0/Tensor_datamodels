"""Linear Datamodeling Score (LDS) and native coalition-prediction Spearman for coalition games.

Definition (TRAK, Park et al. 2023, Defs. 2.2-2.3, App. C.4; repo policy
the project LDS convention): for a fixed attribution vector
tau, LDS is the Spearman correlation, over held-out coalitions S, between the retrained utility v(S) and
sum_{i in S} tau_i. Here players are groups, so every LDS is a *group-level adaptation*.

For a fitted surrogate, tau is the exact Shapley vector of the surrogate's own predicted game, computed
from predictions only (never from true labels). Native Spearman correlates the surrogate's own nonlinear
prediction v_hat(S) with v(S) on the same held-out coalitions.

Ties follow the repository implementation (`analyze_pair_holdout.correlation`): values within 1e-10 are one
tie group and receive their average rank; a constant prediction or constant truth gives an undefined
correlation (None).
"""
from __future__ import annotations

import math

import numpy as np

TIE_TOL = 1e-10


def popcount(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.int64)
    out = np.zeros(masks.shape, dtype=np.int64)
    m = masks.copy()
    while np.any(m):
        out += m & 1
        m >>= 1
    return out


def bits(masks: np.ndarray, d: int) -> np.ndarray:
    return ((np.asarray(masks, dtype=np.int64)[:, None] >> np.arange(d)) & 1).astype(np.int8)


def shapley_from_cube(cube: np.ndarray, d: int) -> np.ndarray:
    """Exact Shapley values of a fully enumerated game.

    cube: shape (2**d,) or (2**d, q), indexed by mask with bit i = player i. Cost O(d 2^d q).
    Refuses d > 20 (use a tensor-network contraction instead; never enumerate past d ~ 20).
    """
    if d > 20:
        raise ValueError("exact enumeration refused for d > 20")
    v = np.asarray(cube, dtype=float)
    squeeze = v.ndim == 1
    if squeeze:
        v = v[:, None]
    n = 1 << d
    assert v.shape[0] == n
    masks = np.arange(n, dtype=np.int64)
    size = popcount(masks)
    weights = np.array([math.factorial(s) * math.factorial(d - 1 - s) / math.factorial(d) for s in range(d)])
    phi = np.zeros((d, v.shape[1]))
    for i in range(d):
        without = masks[(masks >> i) & 1 == 0]
        w = weights[size[without]]
        phi[i] = (w[:, None] * (v[without | (1 << i)] - v[without])).sum(0)
    return phi[:, 0] if squeeze else phi


def _tie_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    sorted_vals = values[order]
    while i < len(values):
        j = i + 1
        while j < len(values) and abs(sorted_vals[j] - sorted_vals[i]) <= TIE_TOL:
            j += 1
        ranks[order[i:j]] = i + (j - i + 1) / 2.0
        i = j
    return ranks


def spearman(prediction, truth):
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    if len(prediction) < 3 or np.ptp(prediction) <= TIE_TOL or np.ptp(truth) <= TIE_TOL:
        return None
    return float(np.corrcoef(_tie_ranks(prediction), _tie_ranks(truth))[0, 1])


def tie_fraction(values) -> float:
    """Fraction of held-out values that share their tie group (|a-b| <= 1e-10) with another value."""
    values = np.sort(np.asarray(values, float))
    if len(values) < 2:
        return 0.0
    close = np.abs(np.diff(values)) <= TIE_TOL
    tied = np.zeros(len(values), bool)
    tied[:-1] |= close
    tied[1:] |= close
    return float(tied.mean())


def modal_fraction(values, width: float) -> float:
    """Largest fraction of values inside one window of the given width (near-tie diagnostic for AND games)."""
    values = np.sort(np.asarray(values, float))
    j, best = 0, 0
    for i in range(len(values)):
        while values[i] - values[j] > width:
            j += 1
        best = max(best, i - j + 1)
    return best / max(len(values), 1)


def lds_and_native(tau, pred_heldout, truth_heldout, heldout_masks, d):
    """Return (LDS, native Spearman) on one held-out set for one target."""
    attribution_sum = bits(heldout_masks, d) @ np.asarray(tau, float)
    return spearman(attribution_sum, truth_heldout), spearman(pred_heldout, truth_heldout)


def within_cardinality(score, truth, masks, min_count=5):
    """Spearman within each coalition size with >= min_count members; returns {size: rho or None}."""
    size = popcount(masks)
    out = {}
    for s in sorted(set(size.tolist())):
        idx = size == s
        if idx.sum() >= min_count:
            out[int(s)] = spearman(np.asarray(score)[idx], np.asarray(truth)[idx])
    return out


def bootstrap_mean_ci(values, units=None, n_boot=10000, seed=0, level=0.95):
    """Mean and percentile bootstrap CI; resamples `units` (e.g. query ids) with replacement if given."""
    values = np.asarray([v for v in values], dtype=float)
    ok = ~np.isnan(values)
    values = values[ok]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if units is None:
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        boots = values[idx].mean(1)
    else:
        units = np.asarray(units)[ok]
        labels = np.unique(units)
        groups = [values[units == u] for u in labels]
        picks = rng.integers(0, len(labels), size=(n_boot, len(labels)))
        sums = np.array([g.sum() for g in groups])
        counts = np.array([len(g) for g in groups])
        boots = sums[picks].sum(1) / counts[picks].sum(1)
    a = (1 - level) / 2
    return float(values.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def _selftest():
    rng = np.random.default_rng(1)
    d = 6
    coef = rng.normal(size=d)
    masks = np.arange(1 << d)
    cube = 0.7 + bits(masks, d) @ coef
    assert np.allclose(shapley_from_cube(cube, d), coef, atol=1e-12)
    # efficiency on a random game
    g = rng.normal(size=1 << d)
    assert abs(shapley_from_cube(g, d).sum() - (g[-1] - g[0])) < 1e-12
    # pair game: phi split evenly
    pair = bits(masks, d)[:, 0] * bits(masks, d)[:, 1] * 2.0
    phi = shapley_from_cube(pair, d)
    assert np.allclose(phi, [1, 1, 0, 0, 0, 0])
    assert spearman(np.ones(5), np.arange(5)) is None
    assert abs(spearman(np.arange(5), np.arange(5)) - 1) < 1e-12
    assert tie_fraction([0, 0, 1, 2]) == 0.5
    return True


if __name__ == "__main__":
    print("selftest", _selftest())
