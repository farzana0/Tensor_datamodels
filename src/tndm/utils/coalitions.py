"""Coalition encodings, sampling and fit/validation splits.

Conventions used throughout the package
---------------------------------------
* A coalition ``S`` of ``d`` training groups is an integer bitmask: bit ``i`` is
  set exactly when group ``i`` is in ``S``.  The empty coalition is ``0`` and the
  full coalition is ``2**d - 1``.
* A dense table of coalition values has shape ``(2**d,)`` or ``(2**d, K)`` and
  is indexed by that bitmask.
* ``bits_of(masks, d)`` returns the ``(n, d)`` inclusion matrix ``z`` with
  ``z[n, i] = 1`` iff group ``i`` is in coalition ``n``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["bits_of", "all_masks", "uniform_masks", "size_weights", "stratified_split"]


def bits_of(masks: np.ndarray, d: int) -> np.ndarray:
    return ((masks[:, None] >> np.arange(d)) & 1).astype(np.float64)


def all_masks(d: int) -> np.ndarray:
    """All ``2**d`` coalitions as integer bitmasks, in increasing order."""
    return np.arange(1 << d, dtype=np.int64)


def uniform_masks(d: int, n: int, rng: np.random.Generator, exclude: set[int] | None = None) -> np.ndarray:
    """Distinct i.i.d. Bernoulli(1/2) coalitions as integer masks.

    At small d the cube can be smaller than the request (d=12 has only 4,096
    coalitions), so the count is capped at what is available; when the request
    covers a large share of what is left, the remaining cube is enumerated and
    sampled without replacement instead of by rejection.
    """
    exclude = set() if exclude is None else set(exclude)
    available = (1 << d) - len(exclude)
    n = min(n, available)
    if d <= 24 and n > available / 4:
        pool = np.setdiff1d(np.arange(1 << d, dtype=np.int64),
                            np.asarray(sorted(exclude), dtype=np.int64))
        return np.sort(rng.choice(pool, size=n, replace=False))
    seen, out = set(exclude), []
    while len(out) < n:
        draw = rng.integers(0, 2, size=(max(n - len(out), 1) * 2, d))
        for row in draw:
            mask = int(np.dot(row, 1 << np.arange(d)))
            if mask not in seen:
                seen.add(mask); out.append(mask)
                if len(out) == n:
                    break
    return np.asarray(out, dtype=np.int64)


def size_weights(masks: np.ndarray) -> np.ndarray:
    sizes = np.asarray([int(mask).bit_count() for mask in masks])
    counts = {size: int(np.sum(sizes == size)) for size in np.unique(sizes)}
    value = np.asarray([1 / counts[size] for size in sizes])
    return value / value.mean()


def stratified_split(masks: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit, validation = [], []
    for size in sorted({int(mask).bit_count() for mask in masks}):
        group = rng.permutation([int(mask) for mask in masks if int(mask).bit_count() == size])
        count = max(1, round(.2 * len(group))) if len(group) >= 3 else 0
        validation.extend(group[:count]); fit.extend(group[count:])
    if len(validation) < 4:  # tiny budgets: fall back to a plain 20% split
        permuted = rng.permutation([int(mask) for mask in masks])
        cut = max(4, round(.2 * len(permuted)))
        return np.asarray(permuted[cut:], dtype=np.int64), np.asarray(permuted[:cut], dtype=np.int64)
    return np.asarray(fit, dtype=np.int64), np.asarray(validation, dtype=np.int64)
