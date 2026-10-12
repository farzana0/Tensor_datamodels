#!/usr/bin/env python3
"""Ideal k-hop AND-of-ORs coalition game, calibrated to the real d=12 transformer.

v(S, q) = BASE + JUMP * prod_{h=1..k} OR(providers of role h of query q present in S)

Structure: polynomial degree k*g, (2^g - 1)^k nonzero Moebius terms, TT rank <= 2
under a role-contiguous ordering.  Calibration constants come from
`analysis/d12_calibration.json` (606 real coalition cells).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

# Calibrated from the real d=12 game .
BASE = -11.2
JUMP = 11.2
NOISE_SD_COMPLETE = 7.8e-5
NOISE_SD_INCOMPLETE = 0.471


@dataclass(frozen=True)
class Game:
    k: int
    g: int
    d: int
    queries: tuple[tuple[tuple[int, ...], ...], ...]  # [query][role] -> provider groups

    @property
    def q(self) -> int:
        return len(self.queries)

    def relevant(self, query: int) -> tuple[int, ...]:
        return tuple(sorted(group for role in self.queries[query] for group in role))

    def values(self, bits: np.ndarray) -> np.ndarray:
        """Noiseless values; `bits` is (n, d) in {0,1}.  Returns (n, Q)."""
        out = np.empty((len(bits), self.q), dtype=float)
        for query, roles in enumerate(self.queries):
            covered = np.ones(len(bits), dtype=bool)
            for role in roles:
                covered &= bits[:, list(role)].any(1)
            out[:, query] = BASE + JUMP * covered
        return out

    def noise(self, values: np.ndarray, rng: np.random.Generator, multiplier: float) -> np.ndarray:
        complete = values > BASE + JUMP / 2
        sd = np.where(complete, NOISE_SD_COMPLETE, NOISE_SD_INCOMPLETE) * multiplier
        return values + rng.standard_normal(values.shape) * sd

    def role_ordering(self) -> tuple[int, ...]:
        """Groups ordered so that each role's providers are contiguous (TT rank <= 2)."""
        seen, order = set(), []
        for roles in self.queries:
            for role in roles:
                for group in role:
                    if group not in seen:
                        seen.add(group); order.append(group)
        order.extend(group for group in range(self.d) if group not in seen)
        return tuple(order)


def balanced_incidence(k: int, g: int, d: int, q: int, seed: int) -> Game:
    """Assign k*g provider slots per query so every group serves as evenly as possible.

    Slots are filled from a repeatedly reshuffled permutation of all d groups, which
    keeps per-group load within one of q*k*g/d, and each query's provider block is
    checked for duplicates against the blocks already accepted.
    """
    if k * g > d:
        raise ValueError(f"k*g={k * g} exceeds d={d}")
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        pool: list[int] = []
        queries, blocks, ok = [], set(), True
        for _query in range(q):
            chosen: list[int] = []
            while len(chosen) < k * g:
                if not pool:
                    pool = list(rng.permutation(d))
                candidate = pool.pop()
                if candidate not in chosen:
                    chosen.append(candidate)
                else:  # keep it for a later query rather than dropping the load
                    pool.insert(0, candidate)
                    if all(group in chosen for group in pool):
                        ok = False
                        break
            if not ok:
                break
            # Distinct queries: normally no two share the same provider set.  When
            # k*g == d every query must use all d groups, so the set is identical by
            # construction; then distinctness is required of the role partition
            # (which providers are interchangeable), the structure that defines the
            # game.  The k*g < d branch is unchanged, so its incidences are too.
            if k * g == d:
                block = tuple(sorted(tuple(sorted(chosen[role * g:(role + 1) * g])) for role in range(k)))
            else:
                block = tuple(sorted(chosen))
            if block in blocks:
                ok = False
                break
            blocks.add(block)
            queries.append(tuple(tuple(sorted(chosen[role * g:(role + 1) * g]))
                                 for role in range(k)))
        if ok and len(queries) == q:
            return Game(k=k, g=g, d=d, queries=tuple(queries))
    raise RuntimeError(f"could not build a balanced incidence for k={k} g={g} d={d} q={q}")


def bits_of(masks: np.ndarray, d: int) -> np.ndarray:
    return ((masks[:, None] >> np.arange(d)) & 1).astype(np.float64)


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


def exact_shapley(game: Game) -> np.ndarray:
    """Exact Shapley values of the ideal game, (Q, d).

    The game depends only on the k*g relevant groups of a query; groups outside
    that set are null players with value 0, and Shapley values of the relevant
    players are unchanged by removing null players, so enumerating 2^(k*g)
    subsets is exact.
    """
    out = np.zeros((game.q, game.d))
    for query, roles in enumerate(game.queries):
        members = list(game.relevant(query))
        t = len(members)
        index = {group: position for position, group in enumerate(members)}
        factorial = [float(math.factorial(i)) for i in range(t + 1)]
        for size in range(t + 1):
            weight = factorial[size] * factorial[t - size - 1] / factorial[t] if size < t else 0.0
            for subset in itertools.combinations(members, size):
                present = set(subset)
                base = all(any(group in present for group in role) for role in roles)
                for group in members:
                    if group in present:
                        continue
                    with_group = present | {group}
                    after = all(any(m in with_group for m in role) for role in roles)
                    out[query, group] += weight * JUMP * (after - base)
    return out


def exact_banzhaf(game: Game) -> np.ndarray:
    out = np.zeros((game.q, game.d))
    for query, roles in enumerate(game.queries):
        members = list(game.relevant(query))
        t = len(members)
        for subset_size in range(t + 1):
            for subset in itertools.combinations(members, subset_size):
                present = set(subset)
                base = all(any(group in present for group in role) for role in roles)
                for group in members:
                    if group in present:
                        continue
                    after = all(any(m in present | {group} for m in role) for role in roles)
                    out[query, group] += JUMP * (after - base) / (2 ** (t - 1))
    return out


def complete_sets(game: Game, query: int, size: int, rng: np.random.Generator,
                  limit: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Random size-`size` coalitions and whether each completes the query.

    Small slices are enumerated exactly; large ones are sampled in vectorised
    batches (random keys, take the `size` smallest) and deduplicated.
    """
    total = math.comb(game.d, size)
    if total <= limit:
        masks_array = np.asarray(sorted(sum(1 << g for g in combo)
                                        for combo in itertools.combinations(range(game.d), size)),
                                 dtype=np.int64)
    else:
        weights = np.int64(1) << np.arange(game.d, dtype=np.int64)
        collected = np.zeros(0, dtype=np.int64)
        while len(collected) < limit:
            chosen = np.argsort(rng.random((2 * limit, game.d)), axis=1)[:, :size]
            batch = weights[chosen].sum(1)
            collected = np.unique(np.concatenate((collected, batch)))
        masks_array = np.sort(rng.choice(collected, size=limit, replace=False))
    bits = bits_of(masks_array, game.d)
    covered = np.ones(len(bits), dtype=bool)
    for role in game.queries[query]:
        covered &= bits[:, list(role)].any(1)
    return masks_array, covered
