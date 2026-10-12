"""Cardinality baseline and degree-bounded / kernel datamodels.

Every fitted family is returned as a :class:`Model` whose ``predict(masks)``
maps integer coalition masks to an ``(n, Q)`` array (one column per outcome).
All families other than the cardinality baseline are *residual* models: they are
fitted to what remains after :func:`fit_cardinality` and wrapped in an
:class:`OffsetModel` (cardinality trend + residual), as described in Appendix
I.5 of the paper.  Hyperparameters are selected on the validation coalitions.
"""
from __future__ import annotations

import itertools

import numpy as np

from tndm.utils.coalitions import bits_of

__all__ = ["ALPHAS", "CHUNK", "Model", "OffsetModel", "ridge_solve", "terms", "poly_design",
           "cardinality_design", "fit_cardinality", "fit_polynomial", "fit_omp", "fit_rbf"]

ALPHAS = tuple(float(x) for x in np.logspace(-5, 2, 13))
CHUNK = 4096


def ridge_solve(x: np.ndarray, y: np.ndarray, weight: np.ndarray, alpha: float) -> np.ndarray:
    root = np.sqrt(weight)
    xw, yw = x * root[:, None], y * root[:, None]
    if x.shape[1] <= x.shape[0]:
        return np.linalg.solve(xw.T @ xw + alpha * np.eye(x.shape[1]), xw.T @ yw)
    return xw.T @ np.linalg.solve(xw @ xw.T + alpha * np.eye(x.shape[0]), yw)


def terms(d: int, degree: int) -> list[tuple[int, ...]]:
    return [term for order in range(1, degree + 1) for term in itertools.combinations(range(d), order)]


def poly_design(masks: np.ndarray, d: int, term_list: list[tuple[int, ...]]) -> np.ndarray:
    x = bits_of(masks, d)
    out = np.empty((len(masks), len(term_list)))
    for index, term in enumerate(term_list):
        out[:, index] = x[:, list(term)].prod(1)
    return out


def cardinality_design(masks: np.ndarray, d: int) -> np.ndarray:
    size = np.asarray([int(mask).bit_count() for mask in masks], dtype=float)
    z = (size - d / 2) / (d / 2)
    return np.column_stack((np.ones(len(masks)), z, z ** 2, z ** 3))


class Model:
    name = "model"

    def predict(self, masks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_chunked(self, masks: np.ndarray) -> np.ndarray:
        return np.vstack([self.predict(masks[start:start + CHUNK])
                          for start in range(0, len(masks), CHUNK)])


class OffsetModel(Model):
    """A base predictor (cardinality trend) plus a residual predictor."""

    def __init__(self, name: str, base, residual):
        self.name, self.base, self.residual = name, base, residual

    def predict(self, masks: np.ndarray) -> np.ndarray:
        return self.base(masks) + self.residual(masks)


def fit_cardinality(masks_fit, y_fit, masks_val, y_val, weight, d):
    best = None
    for alpha in ALPHAS:
        coef = ridge_solve(cardinality_design(masks_fit, d), y_fit, weight, alpha)
        mse = float(np.mean((cardinality_design(masks_val, d) @ coef - y_val) ** 2))
        if best is None or mse < best[0]:
            best = (mse, alpha, coef)
    _, alpha, coef = best

    def base(masks: np.ndarray) -> np.ndarray:
        return cardinality_design(masks, d) @ coef

    # Exact within-cardinality centring of the residual, as in the d=12 fitter.
    raw = y_fit - base(masks_fit)
    offsets = {}
    for size in {int(mask).bit_count() for mask in masks_fit}:
        rows = [int(mask).bit_count() == size for mask in masks_fit]
        offsets[size] = raw[rows].mean(0)
    zero = np.zeros(y_fit.shape[1])

    def base_with_offsets(masks: np.ndarray) -> np.ndarray:
        return base(masks) + np.vstack([offsets.get(int(mask).bit_count(), zero) for mask in masks])

    return base_with_offsets, {"alpha": alpha, "validation_mse": best[0]}


def fit_polynomial(degree, masks_fit, res_fit, masks_val, res_val, weight, d, base):
    term_list = terms(d, degree)
    xf, xv = poly_design(masks_fit, d, term_list), poly_design(masks_val, d, term_list)
    best = None
    for alpha in ALPHAS:
        coef = ridge_solve(xf, res_fit, weight, alpha)
        mse = float(np.mean((xv @ coef - res_val) ** 2))
        if best is None or mse < best[0]:
            best = (mse, alpha, coef)
    coef = best[2]

    def residual(masks: np.ndarray) -> np.ndarray:
        return poly_design(masks, d, term_list) @ coef

    name = "linear" if degree == 1 else f"degree{degree}_ridge"
    return OffsetModel(name, base, residual), {"alpha": best[1], "validation_mse": best[0],
                                               "parameters": len(term_list)}


def fit_omp(degree, masks_fit, res_fit, masks_val, res_val, d, base, max_terms=(4, 8, 16, 32, 64, 128)):
    """Sparse degree-`degree` regression by orthogonal matching pursuit, per query."""
    term_list = terms(d, degree)
    xf, xv = poly_design(masks_fit, d, term_list), poly_design(masks_val, d, term_list)
    xf_centred = xf - xf.mean(0)
    norms = np.linalg.norm(xf_centred, axis=0)
    norms[norms < 1e-12] = 1.0
    xn = xf_centred / norms
    coefficients = np.zeros((len(term_list), res_fit.shape[1]))
    intercepts = np.zeros(res_fit.shape[1])
    total_val = 0.0
    for query in range(res_fit.shape[1]):
        target = res_fit[:, query] - res_fit[:, query].mean()
        selected: list[int] = []
        residual = target.copy()
        best_query = None
        for count in range(1, max(max_terms) + 1):
            scores = np.abs(xn.T @ residual)
            scores[selected] = -np.inf
            selected.append(int(np.argmax(scores)))
            sub = xf[:, selected]
            solution, *_ = np.linalg.lstsq(np.column_stack((np.ones(len(sub)), sub)),
                                           res_fit[:, query], rcond=None)
            residual = res_fit[:, query] - (solution[0] + sub @ solution[1:])
            residual = residual - residual.mean()
            if count in max_terms:
                sub_val = xv[:, selected]
                prediction = solution[0] + sub_val @ solution[1:]
                mse = float(np.mean((prediction - res_val[:, query]) ** 2))
                if best_query is None or mse < best_query[0]:
                    best_query = (mse, list(selected), solution.copy())
            if count >= min(len(term_list), len(masks_fit) - 2):
                break
        mse, chosen, solution = best_query
        total_val += mse
        coefficients[chosen, query] = solution[1:]
        intercepts[query] = solution[0]

    def residual_fn(masks: np.ndarray) -> np.ndarray:
        return poly_design(masks, d, term_list) @ coefficients + intercepts

    return OffsetModel(f"degree{degree}_omp", base, residual_fn), {
        "validation_mse": total_val / res_fit.shape[1], "parameters": int((coefficients != 0).sum())}


def fit_rbf(masks_fit, res_fit, masks_val, res_val, d, base):
    xf, xv = bits_of(masks_fit, d), bits_of(masks_val, d)
    sq_fit = (xf ** 2).sum(1)
    dist_ff = sq_fit[:, None] + sq_fit[None] - 2 * xf @ xf.T
    dist_vf = (xv ** 2).sum(1)[:, None] + sq_fit[None] - 2 * xv @ xf.T
    median = float(np.median(dist_ff[dist_ff > 0])) if np.any(dist_ff > 0) else 1.0
    best = None
    for scale in (.25, .5, 1., 2., 4.):
        gamma = 1.0 / max(median * scale, 1e-9)
        kff, kvf = np.exp(-gamma * dist_ff), np.exp(-gamma * dist_vf)
        for alpha in ALPHAS:
            dual = np.linalg.solve(kff + alpha * np.eye(len(kff)), res_fit)
            mse = float(np.mean((kvf @ dual - res_val) ** 2))
            if best is None or mse < best[0]:
                best = (mse, gamma, dual)
    _, gamma, dual = best

    def residual(masks: np.ndarray) -> np.ndarray:
        x = bits_of(masks, d)
        dist = (x ** 2).sum(1)[:, None] + sq_fit[None] - 2 * x @ xf.T
        return np.exp(-gamma * dist) @ dual

    return OffsetModel("rbf_krr", base, residual), {"gamma": gamma, "validation_mse": best[0],
                                                    "parameters": int(dual.size)}
