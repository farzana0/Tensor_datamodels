"""Datamodel families fitted once per budget and queried for everything.

Every family implements the same contract:

    fit_<family>(masks, y, d, seed, effort) -> (prediction, choice)

``masks`` are the coalitions the family was allowed to evaluate, ``y`` is the
matching ``(m, K)`` block of the cube, and ``prediction`` is the dense
``(2**d, K)`` surface the family predicts for *every* coalition.  All attribution
targets are then computed exactly from that surface, which is the whole point of
the comparison: a surrogate pays its budget once and answers Shapley, Banzhaf,
LOO, interactions, held-out coalitions and set selection from the same fit.

Hyperparameters are chosen on a size-stratified validation split of the sampled
masks only, then the family is refitted on all ``m`` masks.  The effort grids are
deliberately generous and are the same shape for polynomial and tensor-network
families (full penalty grid, rank sweep, restarts).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .design import monomial_design, stratified_fit_validation

ALPHAS = tuple(float(a) for a in np.logspace(-6, 3, 19))
LADDER_ALPHAS = tuple(float(a) for a in np.logspace(-6, 3, 7))
LADDER_RANKS = (1, 2, 3, 4, 6, 8, 12, 16)
RANKS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
CP_RANKS = (1, 2, 3, 4, 6, 8, 12, 16)
TT_RANKS = (1, 2, 3, 4, 6, 8)
GAMMAS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
RESTARTS = 2
ALS_RIDGES = (1e-4, 1e-2, 1.0)
ALS_SWEEPS = 20
REFIT_SWEEPS = 3
ALS_RIDGE = 1e-6


def bits(masks: np.ndarray, d: int) -> np.ndarray:
    return ((np.asarray(masks)[:, None] >> np.arange(d)) & 1).astype(float)


def ridge_solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge with an unpenalised intercept column (column 0 must be all ones)."""
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


@dataclass
class Fit:
    prediction: np.ndarray  # (2**d, K)
    choice: dict


# --------------------------------------------------------------------------
# polynomial / kernel families
# --------------------------------------------------------------------------

def _cardinality_design(masks: np.ndarray, d: int) -> np.ndarray:
    size = bits(masks, d).sum(1)
    z = (size - d / 2) / (d / 2)
    return np.column_stack((np.ones(len(masks)), z, z ** 2, z ** 3))


def _fit_linear_family(masks, y, d, seed, design_fn):
    fit_masks, validation_masks = stratified_fit_validation(masks, d, seed)
    lookup = {int(mask): index for index, mask in enumerate(masks)}
    fit_index = [lookup[int(m)] for m in fit_masks]
    validation_index = [lookup[int(m)] for m in validation_masks]
    x_fit, x_validation = design_fn(fit_masks), design_fn(validation_masks)
    best = None
    for alpha in ALPHAS:
        coefficient = ridge_solve(x_fit, y[fit_index], alpha)
        score = _mse(x_validation @ coefficient, y[validation_index])
        if best is None or score < best[0]:
            best = (score, alpha)
    coefficient = ridge_solve(design_fn(masks), y, best[1])
    prediction = design_fn(np.arange(1 << d)) @ coefficient
    return Fit(prediction, {"alpha": best[1], "validation_mse": best[0]})


def fit_cardinality(masks, y, d, seed):
    return _fit_linear_family(masks, y, d, seed, lambda m: _cardinality_design(m, d))


def fit_cardinality_linear(masks, y, d, seed):
    """Cubic in |S| plus one coefficient per group."""
    return _fit_linear_family(masks, y, d, seed, lambda m: np.column_stack(
        [_cardinality_design(m, d), bits(m, d)]))


def fit_degree(masks, y, d, seed, degree):
    return _fit_linear_family(masks, y, d, seed, lambda m: monomial_design(m, d, degree))


def fit_omp(masks, y, d, seed, degree=3, sparsity=(2, 4, 8, 16, 32, 64, 128)):
    """Orthogonal matching pursuit over degree-<=3 monomials, sparsity by validation."""
    fit_masks, validation_masks = stratified_fit_validation(masks, d, seed)
    lookup = {int(mask): index for index, mask in enumerate(masks)}
    fit_index = [lookup[int(m)] for m in fit_masks]
    validation_index = [lookup[int(m)] for m in validation_masks]
    x_fit = monomial_design(fit_masks, d, degree)
    x_validation = monomial_design(validation_masks, d, degree)
    y_fit, y_validation = y[fit_index], y[validation_index]

    def run(x, target, k):
        chosen = [0]
        residual = target - x[:, [0]] @ np.linalg.lstsq(x[:, [0]], target, rcond=None)[0]
        for _ in range(min(k, x.shape[1] - 1, max(1, x.shape[0] - 1))):
            correlation = np.abs(x.T @ residual).sum(1)
            correlation[chosen] = -np.inf
            candidate = int(np.argmax(correlation))
            if not np.isfinite(correlation[candidate]):
                break
            chosen.append(candidate)
            coefficient = np.linalg.lstsq(x[:, chosen], target, rcond=None)[0]
            residual = target - x[:, chosen] @ coefficient
        coefficient = np.linalg.lstsq(x[:, chosen], target, rcond=None)[0]
        return chosen, coefficient

    best = None
    for k in sparsity:
        if k > x_fit.shape[0]:
            continue
        chosen, coefficient = run(x_fit, y_fit, k)
        score = _mse(x_validation[:, chosen] @ coefficient, y_validation)
        if best is None or score < best[0]:
            best = (score, k)
    if best is None:
        best = (np.inf, sparsity[0])
    x_all = monomial_design(masks, d, degree)
    chosen, coefficient = run(x_all, y, best[1])
    prediction = monomial_design(np.arange(1 << d), d, degree)[:, chosen] @ coefficient
    return Fit(prediction, {"sparsity": best[1], "validation_mse": best[0],
                            "n_selected": len(chosen)})


def fit_rbf(masks, y, d, seed):
    """Gaussian kernel ridge on the 0/1 indicator vectors (Hamming distance)."""
    fit_masks, validation_masks = stratified_fit_validation(masks, d, seed)
    lookup = {int(mask): index for index, mask in enumerate(masks)}
    fit_index = [lookup[int(m)] for m in fit_masks]
    validation_index = [lookup[int(m)] for m in validation_masks]

    def kernel(a, b, gamma):
        distance = np.abs(bits(a, d)[:, None, :] - bits(b, d)[None, :, :]).sum(-1)
        return np.exp(-gamma * distance)

    y_fit = y[fit_index]
    centre = y_fit.mean(0)
    best = None
    for gamma in GAMMAS:
        k_fit = kernel(fit_masks, fit_masks, gamma)
        k_validation = kernel(validation_masks, fit_masks, gamma)
        for alpha in ALPHAS:
            dual = np.linalg.solve(k_fit + alpha * np.eye(len(fit_masks)), y_fit - centre)
            score = _mse(k_validation @ dual + centre, y[validation_index])
            if best is None or score < best[0]:
                best = (score, gamma, alpha)
    _, gamma, alpha = best
    centre = y.mean(0)
    k_all = kernel(masks, masks, gamma)
    dual = np.linalg.solve(k_all + alpha * np.eye(len(masks)), y - centre)
    prediction = kernel(np.arange(1 << d), masks, gamma) @ dual + centre
    return Fit(prediction, {"gamma": gamma, "alpha": alpha, "validation_mse": best[0]})


# --------------------------------------------------------------------------
# tensor-network families (alternating least squares with validation early stop)
# --------------------------------------------------------------------------

def _cp_predict(factors, output, z):
    product = np.ones((len(z), factors[0].shape[1]))
    for axis, factor in enumerate(factors):
        product *= factor[z[:, axis].astype(int)]
    return product @ output


def _scaled_ridge(gram: np.ndarray, ridge: float = ALS_RIDGE) -> np.ndarray:
    """Ridge proportional to the scale of the block, so ALS cannot blow up."""
    scale = float(np.trace(gram)) / max(1, gram.shape[0])
    return (ridge * max(scale, 1e-12)) * np.eye(gram.shape[0])


def _cp_als(z, y, rank, seed, sweeps=ALS_SWEEPS, validation=None, ridge=ALS_RIDGE, init=None):
    """CP-ALS with an output factor; returns the sweep with the best validation MSE."""
    d = z.shape[1]
    rng = np.random.default_rng(seed)
    factors = [rng.normal(scale=0.3, size=(2, rank)) + 1.0 for _ in range(d)]
    output = rng.normal(scale=0.1, size=(rank, y.shape[1]))
    if init is not None:
        factors, output = [f.copy() for f in init[0]], init[1].copy()
    index = z.astype(int)
    best = (np.inf, [f.copy() for f in factors], output.copy(), 0)
    for sweep in range(1, sweeps + 1):
        for axis in range(d):
            rest = np.ones((len(z), rank))
            for other in range(d):
                if other != axis:
                    rest *= factors[other][index[:, other]]
            design = np.zeros((len(z), y.shape[1], 2 * rank))
            for value in (0, 1):
                select = index[:, axis] == value
                if not np.any(select):
                    continue
                design[select, :, value * rank:(value + 1) * rank] = (
                    rest[select][:, None, :] * output.T[None, :, :])
            flat = design.reshape(-1, 2 * rank)
            gram = flat.T @ flat
            try:
                solution = np.linalg.solve(gram + _scaled_ridge(gram, ridge), flat.T @ y.reshape(-1))
            except np.linalg.LinAlgError:
                return best[1], best[2], best[0], best[3]
            factors[axis] = solution.reshape(2, rank)
        product = np.ones((len(z), rank))
        for factor, column in zip(factors, index.T):
            product *= factor[column]
        gram = product.T @ product
        try:
            output = np.linalg.solve(gram + _scaled_ridge(gram, ridge), product.T @ y)
        except np.linalg.LinAlgError:
            return best[1], best[2], best[0], best[3]
        if validation is None:
            best = (np.nan, [f.copy() for f in factors], output.copy(), sweep)
            continue
        score = _mse(_cp_predict(factors, output, validation[0]), validation[1])
        if np.isfinite(score) and score < best[0]:
            best = (score, [f.copy() for f in factors], output.copy(), sweep)
    return best[1], best[2], best[0], best[3]


def fit_cp(masks, y, d, seed):
    fit_masks, validation_masks = stratified_fit_validation(masks, d, seed)
    lookup = {int(mask): index for index, mask in enumerate(masks)}
    fit_index = [lookup[int(m)] for m in fit_masks]
    validation_index = [lookup[int(m)] for m in validation_masks]
    z_fit, z_validation = bits(fit_masks, d), bits(validation_masks, d)
    validation = (z_validation, y[validation_index])
    best = None
    for rank in CP_RANKS:
        if rank * (2 * d + y.shape[1]) > 4 * len(fit_masks) * y.shape[1]:
            continue
        for ridge in ALS_RIDGES:
            for restart in range(RESTARTS):
                _, _, score, sweep = _cp_als(z_fit, y[fit_index], rank, seed * 131 + restart,
                                             validation=validation, ridge=ridge)
                if np.isfinite(score) and (best is None or score < best[0]):
                    best = (score, rank, restart, sweep, ridge)
    if best is None:
        best = (np.inf, CP_RANKS[0], 0, ALS_SWEEPS, ALS_RIDGES[-1])
    factors, output, _, _ = _cp_als(z_fit, y[fit_index], best[1], seed * 131 + best[2],
                                    sweeps=max(1, best[3]), ridge=best[4])
    # Warm-started refit on all m coalitions keeps the validated basin.
    factors, output, _, _ = _cp_als(bits(masks, d), y, best[1], 0, sweeps=REFIT_SWEEPS,
                                    ridge=best[4], init=(factors, output))
    prediction = _cp_predict(factors, output, bits(np.arange(1 << d), d))
    return Fit(prediction, {"rank": best[1], "restart": best[2], "sweeps": best[3],
                            "ridge": best[4], "validation_mse": best[0]})


def _tt_predict(cores, output, z):
    index = z.astype(int)
    value = np.ones((len(z), 1))
    for axis, core in enumerate(cores):
        value = np.einsum("na,nab->nb", value, core[:, index[:, axis], :].transpose(1, 0, 2))
    return value @ output


def _tt_als(z, y, rank, seed, sweeps=ALS_SWEEPS, validation=None, ridge=ALS_RIDGE, init=None):
    """TT-ALS over binary coalition legs with a trailing output matrix.

    One sweep is a single left-to-right pass: the right partial products are
    accumulated once before the pass (the cores to the right are untouched during
    it) and the left products are updated as each core is solved, so every core
    update is the exact least-squares solution given the others.
    """
    d = z.shape[1]
    rng = np.random.default_rng(seed)
    bonds = [1] + [min(rank, 2 ** min(k + 1, d - k - 1)) for k in range(d - 1)] + [rank]
    cores = [rng.normal(scale=1.0 / np.sqrt(max(1, bonds[k])), size=(bonds[k], 2, bonds[k + 1]))
             for k in range(d)]
    output = rng.normal(scale=0.1, size=(bonds[-1], y.shape[1]))
    if init is not None:
        cores, output = [c.copy() for c in init[0]], init[1].copy()
    index = z.astype(int)
    rows, outputs = len(z), y.shape[1]
    best = (np.inf, [c.copy() for c in cores], output.copy(), 0)

    def slice_core(axis):
        return cores[axis][:, index[:, axis], :].transpose(1, 0, 2)

    for sweep in range(1, sweeps + 1):
        right = [None] * (d + 1)
        right[d] = np.repeat(output[None], rows, axis=0)  # (rows, bond, outputs)
        for axis in range(d - 1, -1, -1):
            right[axis] = np.einsum("nab,nbj->naj", slice_core(axis), right[axis + 1])
        left = np.ones((rows, 1))
        for axis in range(d):
            a, b = cores[axis].shape[0], cores[axis].shape[2]
            block = np.einsum("na,nbj->njab", left, right[axis + 1]).reshape(rows, outputs, a * b)
            design = np.zeros((rows, outputs, 2 * a * b))
            for value in (0, 1):
                select = index[:, axis] == value
                if np.any(select):
                    design[select, :, value * a * b:(value + 1) * a * b] = block[select]
            flat = design.reshape(-1, 2 * a * b)
            gram = flat.T @ flat
            try:
                solution = np.linalg.solve(gram + _scaled_ridge(gram, ridge), flat.T @ y.reshape(-1))
            except np.linalg.LinAlgError:
                return best[1], best[2], best[0], best[3]
            cores[axis] = solution.reshape(2, a, b).transpose(1, 0, 2)
            left = np.einsum("na,nab->nb", left, slice_core(axis))
        gram = left.T @ left
        try:
            output = np.linalg.solve(gram + _scaled_ridge(gram, ridge), left.T @ y)
        except np.linalg.LinAlgError:
            return best[1], best[2], best[0], best[3]
        if validation is None:
            best = (np.nan, [c.copy() for c in cores], output.copy(), sweep)
            continue
        score = _mse(_tt_predict(cores, output, validation[0]), validation[1])
        if np.isfinite(score) and score < best[0]:
            best = (score, [c.copy() for c in cores], output.copy(), sweep)
    return best[1], best[2], best[0], best[3]


def fit_tt(masks, y, d, seed):
    fit_masks, validation_masks = stratified_fit_validation(masks, d, seed)
    lookup = {int(mask): index for index, mask in enumerate(masks)}
    fit_index = [lookup[int(m)] for m in fit_masks]
    validation_index = [lookup[int(m)] for m in validation_masks]
    z_fit, z_validation = bits(fit_masks, d), bits(validation_masks, d)
    validation = (z_validation, y[validation_index])
    best = None
    for rank in TT_RANKS:
        if 2 * rank * rank * d > 6 * len(fit_masks) * y.shape[1]:
            continue
        for ridge in ALS_RIDGES:
            for restart in range(RESTARTS):
                _, _, score, sweep = _tt_als(z_fit, y[fit_index], rank, seed * 977 + restart,
                                             validation=validation, ridge=ridge)
                if np.isfinite(score) and (best is None or score < best[0]):
                    best = (score, rank, restart, sweep, ridge)
    if best is None:
        best = (np.inf, TT_RANKS[0], 0, ALS_SWEEPS, ALS_RIDGES[-1])
    cores, output, _, _ = _tt_als(z_fit, y[fit_index], best[1], seed * 977 + best[2],
                                  sweeps=max(1, best[3]), ridge=best[4])
    # Warm-started refit on all m coalitions keeps the validated basin.
    cores, output, _, _ = _tt_als(bits(masks, d), y, best[1], 0, sweeps=REFIT_SWEEPS,
                                  ridge=best[4], init=(cores, output))
    prediction = _tt_predict(cores, output, bits(np.arange(1 << d), d))
    return Fit(prediction, {"rank": best[1], "restart": best[2], "sweeps": best[3],
                            "ridge": best[4], "validation_mse": best[0]})



# --------------------------------------------------------------------------
# degree ladders: truncate the fitted polynomial coefficient tensor
# --------------------------------------------------------------------------

def _coefficient_tensor(d: int, degree: int, coefficient: np.ndarray) -> np.ndarray:
    """Monomial coefficients laid out as a (2,)*d x K tensor.

    Entry ``e`` holds the coefficient of the monomial ``prod_{k in e} z_k``.  The
    flattened index is the term's own mask, so axis 0 of the tensor is player
    ``d-1`` and axis ``d-1`` is player 0.  That reverses the chain order used by
    the TT/HT truncations, which leaves the set of cuts (and therefore the rank
    spectrum) unchanged.
    """
    vector = np.zeros((1 << d, coefficient.shape[1]))
    terms = [()] + [term for order in range(1, degree + 1)
                    for term in itertools.combinations(range(d), order)]
    for index, term in enumerate(terms):
        mask = 0
        for player in term:
            mask |= 1 << player
        vector[mask] = coefficient[index]
    return vector.reshape((2,) * d + (coefficient.shape[1],))


def _tt_svd(tensor: np.ndarray, max_rank: int) -> np.ndarray:
    shape, cores, remainder, left = tensor.shape, [], tensor.copy(), 1
    for physical in shape[:-1]:
        matrix = remainder.reshape(left * physical, -1)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = min(max_rank, len(singular))
        cores.append(u[:, :rank].reshape(left, physical, rank))
        remainder, left = singular[:rank, None] * vh[:rank], rank
    cores.append(remainder.reshape(left, shape[-1], 1))
    value = cores[0]
    for core in cores[1:]:
        value = np.tensordot(value, core, axes=([-1], [0]))
    return value.reshape(shape)


def _ht_truncate(tensor: np.ndarray, max_rank: int, max_depth: int = 2) -> np.ndarray:
    """Balanced hierarchical-Tucker truncation on a binary tree of coalition modes.

    Each node truncates the matricisation that separates its own mode block from
    everything else (the remaining modes and the output mode) to ``max_rank``.
    The tree is cut at ``max_depth`` levels below the root, which is the usual
    three-level balanced HT and keeps the number of SVDs per fit small.
    """
    axes_all = list(range(tensor.ndim))

    def truncate_cut(block: np.ndarray, axes: list[int]) -> np.ndarray:
        rest = [axis for axis in axes_all if axis not in axes]
        order = list(axes) + rest
        permuted = np.transpose(block, order)
        shape = permuted.shape
        left = int(np.prod(shape[:len(axes)]))
        matrix = permuted.reshape(left, -1)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = min(max_rank, len(singular))
        approximation = (u[:, :rank] * singular[:rank]) @ vh[:rank]
        return np.transpose(approximation.reshape(shape), np.argsort(order))

    def recurse(block: np.ndarray, axes: list[int], depth: int) -> np.ndarray:
        if len(axes) <= 1 or depth > max_depth:
            return block
        block = truncate_cut(block, axes)
        half = len(axes) // 2
        block = recurse(block, axes[:half], depth + 1)
        return recurse(block, axes[half:], depth + 1)

    return recurse(tensor, list(range(tensor.ndim - 1)), 1)


def _contract_tensor(tensor: np.ndarray, d: int) -> np.ndarray:
    """Evaluate a monomial-coefficient tensor at every mask.

    Because the variables are binary, ``prod_{k in e} z_k`` is one exactly when
    ``e`` is a subset of the coalition, so evaluating the polynomial everywhere is
    the subset-sum (zeta) transform of the coefficients: O(d 2^d) rather than a
    contraction per coalition.
    """
    result = tensor.reshape(1 << d, -1).astype(float).copy()
    index = np.arange(1 << d)
    for bit in range(d):
        step = 1 << bit
        selected = np.flatnonzero((index & step) != 0)
        result[selected] += result[selected & ~step]
    return result


def _fit_ladder(masks, y, d, seed, truncate, degrees=(1, 2, 3), ranks=LADDER_RANKS):
    fit_masks, validation_masks = stratified_fit_validation(masks, d, seed)
    lookup = {int(mask): index for index, mask in enumerate(masks)}
    fit_index = [lookup[int(m)] for m in fit_masks]
    validation_index = [lookup[int(m)] for m in validation_masks]
    best = None
    for degree in degrees:
        x_fit = monomial_design(fit_masks, d, degree)
        for alpha in LADDER_ALPHAS:
            coefficient = ridge_solve(x_fit, y[fit_index], alpha)
            dense = _coefficient_tensor(d, degree, coefficient)
            if not np.all(np.isfinite(dense)):
                continue
            for rank in ranks:
                prediction = _contract_tensor(truncate(dense, rank), d)
                score = _mse(prediction[validation_masks], y[validation_index])
                if np.isfinite(score) and (best is None or score < best[0]):
                    best = (score, degree, alpha, rank)
    if best is None:
        best = (np.inf, 1, LADDER_ALPHAS[0], ranks[0])
    _, degree, alpha, rank = best
    coefficient = ridge_solve(monomial_design(masks, d, degree), y, alpha)
    dense = _coefficient_tensor(d, degree, coefficient)
    prediction = _contract_tensor(truncate(dense, rank), d)
    return Fit(prediction, {"degree": degree, "alpha": alpha, "rank": rank,
                            "validation_mse": best[0]})


def fit_tt_ladder(masks, y, d, seed):
    return _fit_ladder(masks, y, d, seed, _tt_svd)


def fit_ht_ladder(masks, y, d, seed):
    return _fit_ladder(masks, y, d, seed, _ht_truncate)


FAMILIES = {
    "cardinality": fit_cardinality,
    "cardinality_linear": fit_cardinality_linear,
    "linear": lambda masks, y, d, seed: fit_degree(masks, y, d, seed, 1),
    "degree2_ridge": lambda masks, y, d, seed: fit_degree(masks, y, d, seed, 2),
    "degree3_ridge": lambda masks, y, d, seed: fit_degree(masks, y, d, seed, 3),
    "degree3_omp": fit_omp,
    "rbf": fit_rbf,
    "cp": fit_cp,
    "tt": fit_tt,
    "ht_ladder": fit_ht_ladder,
    "tt_ladder": fit_tt_ladder,
}
