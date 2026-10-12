"""TT degree ladder: degree-bounded ridge predictors distilled into tensor trains.

This is the "TT degree ladder" row of the paper's tables.  It is reported
separately from direct TT fitting (:mod:`tndm.models.tensor_networks`).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from tndm.baselines.regression import ALPHAS, CHUNK, OffsetModel
from tndm.models.tensor_networks import train_tn
from tndm.utils.coalitions import bits_of

__all__ = ["subset_kernel", "kernel_ridge_dual", "fit_ladder"]


def subset_kernel(xa: np.ndarray, xb: np.ndarray, degree: int) -> np.ndarray:
    """Inner product of degree-1..p monomial features: sum_{j=1..p} C(|a AND b|, j).

    The monomial x_T equals 1 exactly when T is a subset of the coalition, so
    two coalitions share C(c, j) monomials of order j, where c = |a AND b|.
    Ridge with an identity penalty on these features is therefore identical to
    kernel ridge with this kernel.
    """
    common = np.rint(xa @ xb.T).astype(np.int64)
    top = int(common.max()) if common.size else 0
    table = np.zeros(max(top, 0) + 1)
    for c in range(len(table)):
        table[c] = sum(math.comb(c, j) for j in range(1, degree + 1))
    return table[common]


def kernel_ridge_dual(kernel: np.ndarray, y: np.ndarray, weight: np.ndarray, alpha: float) -> np.ndarray:
    """Dual coefficients whose `K(new, fit) @ dual` equals weighted primal ridge."""
    root = np.sqrt(weight)
    system = root[:, None] * kernel * root[None, :] + alpha * np.eye(len(kernel))
    return root[:, None] * np.linalg.solve(system, root[:, None] * y)


def fit_ladder(d, q, masks_fit, res_fit, masks_val, res_val, weight, base, order,
               max_degree, distill_masks, ranks=(2, 4, 8), relative_gain=.01):
    """Degree ladder: raise the polynomial degree while compressing to a TT.

    At each degree the best ridge fit is distilled into a TT by regression on
    `distill_masks` (a large outcome-free set of coalitions), which is the
    scalable stand-in for the exact TT-SVD compression used at d=12.  The ladder
    stops when a higher degree fails to improve validation MSE by `relative_gain`.
    """
    previous, best_overall, path = None, None, []
    x_fit_bits = bits_of(masks_fit, d)
    x_val_bits = bits_of(masks_val, d)
    x_distill = bits_of(distill_masks, d)
    for degree in range(1, max_degree + 1):
        # Ridge over every monomial of degree 1..p, solved exactly in its dual via
        # the subset-count kernel (see `subset_kernel`), so no 2^d or C(d,p)
        # design is ever formed.
        k_ff = subset_kernel(x_fit_bits, x_fit_bits, degree)
        k_vf = subset_kernel(x_val_bits, x_fit_bits, degree)
        ridge_candidates = []
        for alpha in ALPHAS:
            dual = kernel_ridge_dual(k_ff, res_fit, weight, alpha)
            ridge_candidates.append((float(np.mean((k_vf @ dual - res_val) ** 2)), alpha, dual))
        candidates = []
        for _, alpha, dual in sorted(ridge_candidates, key=lambda row: row[0])[:1]:
            distilled_y = np.vstack([subset_kernel(x_distill[start:start + CHUNK], x_fit_bits, degree) @ dual
                                     for start in range(0, len(x_distill), CHUNK)])
            scale = np.maximum(distilled_y.std(0), .05)
            for rank in ranks:
                mse, model, _ = train_tn("tt", d, q, rank, 700 + rank, order,
                                         x_distill, distilled_y / scale,
                                         np.ones(len(distill_masks)), x_val_bits, res_val / scale,
                                         1e-3, 1e-4, steps=101)
                candidates.append((mse * float(np.mean(scale ** 2)), rank, alpha, model, scale))
        best = min(candidates, key=lambda row: row[0])
        gain = None if previous is None else (previous - best[0]) / max(previous, 1e-12)
        keep = degree <= min(3, max_degree) or (gain is not None and gain >= relative_gain)
        path.append({"degree": degree, "rank": best[1], "alpha": best[2],
                     "validation_mse": best[0], "relative_gain": gain, "accepted": bool(keep)})
        if not keep:
            break
        previous, best_overall = best[0], (degree, best)
    degree, best = best_overall
    _, rank, alpha, model, scale = best

    def residual(masks: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(bits_of(masks, d)[:, list(order)], dtype=torch.float64)
            return model(x).numpy() * scale

    parameters = int(sum(p.numel() for p in model.parameters()))
    return OffsetModel("ladder_tt", base, residual), {
        "degree": degree, "rank": rank, "alpha": alpha, "parameters": parameters,
        "validation_mse": best[0], "path": path}
