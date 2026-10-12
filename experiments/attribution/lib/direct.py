"""Direct attribution estimators under a distinct-evaluation budget.

Each estimator is allowed its natural sampling scheme but may look up at most
``m`` *distinct* coalitions.  Repeated lookups of a coalition it has already paid
for are free, which is the right accounting when one evaluation is one
retraining and results are cached.  Every estimator therefore reports the number
of distinct coalitions it actually used, and the runner checks it never exceeds
``m``.

The estimators are vectorised over the output axis: one sampling draw answers
every output of a cube at once, exactly as a cached retraining would.  Their
sampling never depends on the outputs, except for truncated Monte Carlo, which is
adaptive by construction and is driven by the cube's primary output.

Coverage: permutation (ApproShapley) and truncated Monte Carlo Data Shapley,
KernelSHAP-style regression Shapley, stratified Shapley, MSR Data Banzhaf, exact
LOO, and a regression estimator for pairwise Shapley interactions.  shapiq's own
approximators are run separately by ``run_shapiq.py`` on the primary output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb, factorial

import numpy as np

from . import exact


@dataclass
class Estimate:
    """What a direct estimator produced, and what it cost."""

    shapley: np.ndarray | None = None
    pair_fsii: np.ndarray | None = None
    banzhaf: np.ndarray | None = None
    loo: np.ndarray | None = None
    pair_sii: np.ndarray | None = None
    evaluations: int = 0
    detail: dict = field(default_factory=dict)


class Oracle:
    """Caching view of a cube: counts distinct coalitions, not lookups."""

    def __init__(self, values: np.ndarray, budget: int):
        self.values = values
        self.budget = budget
        self.seen: dict[int, int] = {}

    @property
    def used(self) -> int:
        return len(self.seen)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.budget

    def get(self, mask: int) -> np.ndarray | None:
        mask = int(mask)
        if mask in self.seen:
            return self.values[mask]
        if self.exhausted:
            return None
        self.seen[mask] = 1
        return self.values[mask]

    def masks(self) -> np.ndarray:
        return np.sort(np.asarray(list(self.seen), dtype=np.int64))


def permutation_shapley(values: np.ndarray, d: int, m: int, rng: np.random.Generator,
                        truncation: float | None = None) -> Estimate:
    """ApproShapley: average marginal contributions along random permutations.

    With ``truncation`` set this is Truncated Monte Carlo Data Shapley: once the
    running prefix is within ``truncation`` of the grand coalition's value (on the
    primary output) the remaining marginals in that permutation are taken to be
    zero and cost nothing.
    """
    oracle = Oracle(values, m)
    outputs = values.shape[1]
    total = np.zeros((d, outputs))
    count = np.zeros((d, outputs))
    full_value = oracle.get((1 << d) - 1)
    empty_value = oracle.get(0)
    if full_value is None or empty_value is None:
        return Estimate(evaluations=oracle.used, detail={"aborted": True})
    truncated_permutations = 0
    permutations = 0
    while not oracle.exhausted:
        order = rng.permutation(d)
        mask = 0
        previous = empty_value
        permutations += 1
        for player in order:
            nxt = mask | (1 << player)
            value = oracle.get(nxt)
            if value is None:
                break
            total[player] += value - previous
            count[player] += 1
            mask, previous = nxt, value
            if truncation is not None and abs(float(value[0] - full_value[0])) <= truncation:
                # remaining players in this permutation get a zero marginal for free
                remaining = [p for p in order if not mask >> p & 1]
                for other in remaining:
                    count[other] += 1
                truncated_permutations += 1
                break
    count[count == 0] = np.nan
    return Estimate(shapley=total / count, evaluations=oracle.used,
                    detail={"permutations": permutations,
                            "truncated_permutations": truncated_permutations})


def _sample_distinct(d: int, m: int, rng: np.random.Generator,
                     size_probabilities: np.ndarray | None = None) -> np.ndarray:
    """Draw m distinct masks; optionally with a prescribed distribution over sizes."""
    total = 1 << d
    if size_probabilities is None:
        return rng.choice(total, size=m, replace=False)
    size = exact.popcounts(d)
    chosen: set[int] = set()
    strata = {k: list(np.flatnonzero(size == k)) for k in range(d + 1)}
    guard = 0
    while len(chosen) < m and guard < 200 * m:
        guard += 1
        k = int(rng.choice(d + 1, p=size_probabilities))
        pool = strata[k]
        if not pool:
            continue
        candidate = int(pool[rng.integers(len(pool))])
        chosen.add(candidate)
    if len(chosen) < m:  # fall back to uniform completion
        remaining = [mask for mask in range(total) if mask not in chosen]
        chosen.update(rng.choice(remaining, size=m - len(chosen), replace=False).tolist())
    return np.asarray(sorted(chosen))


def kernel_shap(values: np.ndarray, d: int, m: int, rng: np.random.Generator) -> Estimate:
    """KernelSHAP: weighted least squares on Shapley-kernel-sampled coalitions.

    The grand and empty coalitions are always evaluated (they pin the efficiency
    constraint), and the remaining budget is drawn from the Shapley kernel over
    sizes 1..d-1 without replacement.
    """
    if m < d + 2:
        return Estimate(evaluations=0, detail={"aborted": "budget below d+2"})
    weights = np.zeros(d + 1)
    for size in range(1, d):
        weights[size] = (d - 1) / (comb(d, size) * size * (d - size))
    probabilities = weights / weights.sum()
    sample = _sample_distinct(d, m - 2, rng, probabilities)
    sample = np.asarray([mask for mask in sample if mask not in (0, (1 << d) - 1)])
    masks = np.concatenate([[0], sample, [(1 << d) - 1]])
    sizes = exact.popcounts(d)[masks]
    kernel = np.zeros(len(masks))
    interior = (sizes > 0) & (sizes < d)
    kernel[interior] = np.asarray([(d - 1) / (comb(d, int(s)) * int(s) * (d - int(s)))
                                   for s in sizes[interior]])
    z = ((masks[:, None] >> np.arange(d)) & 1).astype(float)
    y = values[masks]
    # Efficiency-constrained weighted least squares, eliminating the last player.
    empty = values[0]
    full = values[(1 << d) - 1]
    interior_index = np.flatnonzero(interior)
    if len(interior_index) < d:
        return Estimate(evaluations=len(masks), detail={"aborted": "too few interior coalitions"})
    zi = z[interior_index]
    yi = y[interior_index] - empty - zi[:, [d - 1]] * (full - empty)
    xi = zi[:, :d - 1] - zi[:, [d - 1]]
    w = kernel[interior_index]
    gram = xi.T @ (w[:, None] * xi) + 1e-10 * np.eye(d - 1)
    beta = np.linalg.solve(gram, xi.T @ (w[:, None] * yi))
    last = (full - empty) - beta.sum(0)
    shapley = np.vstack([beta, last])
    return Estimate(shapley=shapley, evaluations=len(masks),
                    detail={"interior": int(len(interior_index))})


def msr_banzhaf(values: np.ndarray, d: int, m: int, rng: np.random.Generator) -> Estimate:
    """Maximum-sample-reuse Banzhaf: every coalition informs all d players."""
    masks = _sample_distinct(d, m, rng)
    z = ((masks[:, None] >> np.arange(d)) & 1).astype(bool)
    y = values[masks]
    result = np.full((d, values.shape[1]), np.nan)
    for player in range(d):
        with_player = z[:, player]
        if with_player.any() and (~with_player).any():
            result[player] = y[with_player].mean(0) - y[~with_player].mean(0)
    return Estimate(banzhaf=result, evaluations=len(masks),
                    detail={"scheme": "uniform distinct masks, maximum sample reuse"})


def stratified_shapley(values: np.ndarray, d: int, m: int, rng: np.random.Generator) -> Estimate:
    """Stratified sampling: budget split over the d background strata per player.

    Each evaluated pair (S, S+i) costs at most two distinct coalitions; the cache
    is shared across players, which is what makes stratified sampling competitive.
    """
    oracle = Oracle(values, m)
    total = np.zeros((d, d, values.shape[1]))
    count = np.zeros((d, d))
    order = [(player, size) for size in range(d) for player in range(d)]
    rng.shuffle(order)
    while not oracle.exhausted:
        progressed = False
        for player, size in order:
            if oracle.exhausted:
                break
            others = [p for p in range(d) if p != player]
            background = rng.choice(others, size=size, replace=False) if size else []
            mask = 0
            for p in background:
                mask |= 1 << int(p)
            low = oracle.get(mask)
            high = oracle.get(mask | (1 << player))
            if low is None or high is None:
                break
            total[player, size] += high - low
            count[player, size] += 1
            progressed = True
        if not progressed:
            break
    weights = np.asarray([factorial(s) * factorial(d - s - 1) / factorial(d) * comb(d - 1, s)
                          for s in range(d)])
    shapley = np.zeros((d, values.shape[1]))
    for player in range(d):
        seen = count[player] > 0
        if not seen.any():
            shapley[player] = np.nan
            continue
        mean = np.zeros((d, values.shape[1]))
        mean[seen] = total[player][seen] / count[player][seen][:, None]
        weight = weights * seen
        shapley[player] = (weight[:, None] * mean).sum(0) / weight.sum()
    return Estimate(shapley=shapley, evaluations=oracle.used,
                    detail={"strata_filled": int((count > 0).sum())})


def direct_loo(values: np.ndarray, d: int, m: int, rng: np.random.Generator) -> Estimate:
    """Leave-one-out needs exactly d+1 evaluations and is then exact."""
    if m < d + 1:
        return Estimate(evaluations=0, detail={"aborted": "budget below d+1"})
    full = (1 << d) - 1
    masks = [full] + [full & ~(1 << player) for player in range(d)]
    loo = np.stack([values[full] - values[full & ~(1 << player)] for player in range(d)])
    return Estimate(loo=loo, evaluations=len(set(masks)),
                    detail={"exact": True})


def regression_pair_interactions(values: np.ndarray, d: int, m: int,
                                 rng: np.random.Generator) -> Estimate:
    """Faithful-interaction regression: weighted degree-2 fit, read off as SII.

    This is the KernelSHAP-IQ family: sample from the Shapley kernel, fit a
    degree-2 surface with those weights, and report the pairwise coefficients.  It
    is included so the interaction comparison has a direct estimator that is
    vectorised over outputs; shapiq's own SII/FSII/k-SII approximators are run
    separately on the primary output.
    """
    import itertools
    if m < 1 + d + comb(d, 2) // 4:
        pass  # under-determined fits are allowed; ridge keeps them finite
    weights = np.zeros(d + 1)
    for size in range(1, d):
        weights[size] = (d - 1) / (comb(d, size) * size * (d - size))
    probabilities = weights / weights.sum()
    masks = _sample_distinct(d, m, rng, probabilities)
    z = ((masks[:, None] >> np.arange(d)) & 1).astype(float)
    columns = [np.ones(len(masks))] + [z[:, i] for i in range(d)]
    pairs = list(itertools.combinations(range(d), 2))
    columns += [z[:, i] * z[:, j] for i, j in pairs]
    x = np.column_stack(columns)
    sizes = exact.popcounts(d)[masks]
    w = np.ones(len(masks))
    interior = (sizes > 0) & (sizes < d)
    w[interior] = np.asarray([(d - 1) / (comb(d, int(s)) * int(s) * (d - int(s)))
                              for s in sizes[interior]])
    w[~interior] = w[interior].max() * 1e3 if interior.any() else 1.0
    penalty = np.eye(x.shape[1]) * 1e-8
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ (w[:, None] * x) + penalty, x.T @ (w[:, None] * values[masks]))
    pair_block = beta[1 + d:]
    return Estimate(pair_fsii=pair_block, evaluations=len(masks),
                    detail={"scheme": "Shapley-kernel weighted degree-2 regression"})


def sampling_pair_sii(values: np.ndarray, d: int, m: int, rng: np.random.Generator) -> Estimate:
    """Unbiased stratified sampling estimator for the pairwise Shapley interaction.

    SII_ij is an expectation of the second difference over a background size drawn
    uniformly from 0..d-2 and then a uniform background of that size, so the
    estimator cycles over (pair, size) cells, draws a background, and evaluates the
    four masks it needs.  The evaluation cache is shared across pairs, which is
    where most of the sample reuse comes from.
    """
    import itertools
    oracle = Oracle(values, m)
    pair_list = list(itertools.combinations(range(d), 2))
    total = np.zeros((len(pair_list), d - 1, values.shape[1]))
    count = np.zeros((len(pair_list), d - 1))
    cells = [(index, size) for index in range(len(pair_list)) for size in range(d - 1)]
    rng.shuffle(cells)
    while not oracle.exhausted:
        progressed = False
        for index, size in cells:
            if oracle.exhausted:
                break
            i, j = pair_list[index]
            others = [p for p in range(d) if p not in (i, j)]
            background = rng.choice(others, size=size, replace=False) if size else []
            mask = 0
            for player in background:
                mask |= 1 << int(player)
            corners = [oracle.get(mask), oracle.get(mask | 1 << i),
                       oracle.get(mask | 1 << j), oracle.get(mask | 1 << i | 1 << j)]
            if any(corner is None for corner in corners):
                break
            total[index, size] += corners[3] - corners[1] - corners[2] + corners[0]
            count[index, size] += 1
            progressed = True
        if not progressed:
            break
    result = np.full((len(pair_list), values.shape[1]), np.nan)
    for index in range(len(pair_list)):
        filled = count[index] > 0
        if not filled.any():
            continue
        mean = total[index][filled] / count[index][filled][:, None]
        result[index] = mean.mean(0)  # sizes are equally weighted for SII
    return Estimate(pair_sii=result, evaluations=oracle.used,
                    detail={"cells_filled": int((count > 0).sum()),
                            "cells_total": int(count.size)})


ESTIMATORS = {
    "permutation_shapley": permutation_shapley,
    "kernel_shap": kernel_shap,
    "stratified_shapley": stratified_shapley,
    "msr_banzhaf": msr_banzhaf,
    "direct_loo": direct_loo,
    "sampling_sii": sampling_pair_sii,
    "fsii_regression": regression_pair_interactions,
}

# tmc_shapley is bound in the runner because its truncation tolerance is a
# property of the cube (a fraction of the utility range), not of the estimator.
TARGETS_ANSWERED = {
    "permutation_shapley": ("shapley",),
    "tmc_shapley": ("shapley",),
    "kernel_shap": ("shapley",),
    "stratified_shapley": ("shapley",),
    "msr_banzhaf": ("banzhaf",),
    "direct_loo": ("loo",),
    "sampling_sii": ("pair_sii",),
    "fsii_regression": ("pair_fsii",),
}
