"""Surrogate families fitted on a sampled coalition budget.

Every family is fitted on the same sampled cells and then used to predict the
whole 256-cell cube, so that the exact functionals in `cube.py` can be recomputed
from its predictions.  Hyperparameters are chosen on an internal split of the
sampled cells and the model is then refitted on every sampled cell.

Polynomial families are implemented here; CP, TT and HT reuse the original
models (`experiments.reasoning.common.fit_query_paths`) and the degree
ladder reuses `interaction_degree_ladder.DegreeLadderTT`, so the paper's claims
are made about the paper's own implementations.
"""
from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import OrthogonalMatchingPursuit

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from experiments.reasoning.common.fit_query_paths import fit_tn, make_model  # noqa: E402
from experiments.reasoning.common.interaction_degree_ladder import (  # noqa: E402
    DegreeLadderTT,
    coefficient_tensor,
)

D = 8
ALL_MASKS = np.arange(1 << D)
ALPHAS = tuple(float(value) for value in np.logspace(-6, 3, 19))
OMP_NONZEROS = (4, 8, 16, 32, 48)
RBF_GAMMAS = (0.1, 0.3, 0.6, 1.0)
RBF_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0)
TN_RANKS = (1, 2, 3, 4, 6, 8)
TN_LAMBDAS = (1e-4, 1e-2, 1.0)
TN_RESTARTS = 2
LADDER_DEGREES = (1, 2, 3, 4, 8)
LADDER_RANKS = (1, 2, 3, 4, 6, 8)
LADDER_RELATIVE_GAIN = 0.01
MAIN_FAMILIES = ("linear", "degree2_ridge", "degree3_ridge", "tt_ladder")
FAMILIES = ("cardinality", "linear", "degree2_ridge", "degree3_ridge", "sparse_degree3",
            "rbf_kernel", "cp", "tt", "ht", "tt_ladder")


def bits(masks: np.ndarray) -> np.ndarray:
    return ((np.asarray(masks)[:, None] >> np.arange(D)) & 1).astype(np.float64)


def terms(degree: int) -> list[tuple[int, ...]]:
    return [term for order in range(1, degree + 1)
            for term in itertools.combinations(range(D), order)]


def design(masks: np.ndarray, term_list: list[tuple[int, ...]]) -> np.ndarray:
    x = bits(masks)
    columns = [np.ones(len(masks))] + [x[:, term].prod(1) for term in term_list]
    return np.column_stack(columns)


def cardinality_design(masks: np.ndarray) -> np.ndarray:
    size = np.asarray([int(mask).bit_count() for mask in masks], dtype=float)
    z = (size - D / 2) / (D / 2)
    return np.column_stack((np.ones(len(masks)), z, z ** 2, z ** 3))


def ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge with an unpenalized intercept in column 0."""
    penalty = alpha * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def stratified_split(sample: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    fit, validation = [], []
    for size in sorted({int(mask).bit_count() for mask in sample}):
        group = generator.permutation([int(mask) for mask in sample
                                       if int(mask).bit_count() == size])
        count = max(1, round(.2 * len(group))) if len(group) >= 4 else 0
        validation.extend(group[:count])
        fit.extend(group[count:])
    if not validation:
        validation.append(fit.pop())
    return np.asarray(sorted(fit)), np.asarray(sorted(validation))


def _mse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((prediction - truth) ** 2))


def fit_polynomial(term_list: list[tuple[int, ...]] | None, cube: np.ndarray,
                   fit_masks: np.ndarray, validation_masks: np.ndarray, sample: np.ndarray
                   ) -> tuple[np.ndarray, dict]:
    build = cardinality_design if term_list is None else (lambda m: design(m, term_list))
    candidates = []
    for alpha in ALPHAS:
        coefficient = ridge(build(fit_masks), cube[fit_masks], alpha)
        candidates.append((_mse(build(validation_masks) @ coefficient, cube[validation_masks]),
                           alpha))
    best = min(candidates, key=lambda item: item[:2])
    coefficient = ridge(build(sample), cube[sample], best[1])
    return build(ALL_MASKS) @ coefficient, {"alpha": best[1], "validation_mse": best[0]}


def fit_sparse_degree3(cube: np.ndarray, fit_masks: np.ndarray, validation_masks: np.ndarray,
                       sample: np.ndarray) -> tuple[np.ndarray, dict]:
    term_list = terms(3)
    xfit = design(fit_masks, term_list)[:, 1:]
    xvalidation = design(validation_masks, term_list)[:, 1:]
    xsample = design(sample, term_list)[:, 1:]
    xall = design(ALL_MASKS, term_list)[:, 1:]
    candidates = []
    for nonzeros in OMP_NONZEROS:
        if nonzeros >= len(fit_masks):
            continue
        prediction = np.column_stack([
            OrthogonalMatchingPursuit(n_nonzero_coefs=nonzeros).fit(xfit, cube[fit_masks, output])
            .predict(xvalidation) for output in range(cube.shape[1])])
        candidates.append((_mse(prediction, cube[validation_masks]), nonzeros))
    if not candidates:
        candidates = [(float("inf"), max(1, len(fit_masks) - 1))]
    best = min(candidates, key=lambda item: item[:2])
    prediction = np.column_stack([
        OrthogonalMatchingPursuit(n_nonzero_coefs=min(best[1], max(1, len(sample) - 1)))
        .fit(xsample, cube[sample, output]).predict(xall) for output in range(cube.shape[1])])
    return prediction, {"n_nonzero_coefs": best[1], "validation_mse": best[0]}


def fit_rbf(cube: np.ndarray, fit_masks: np.ndarray, validation_masks: np.ndarray,
            sample: np.ndarray) -> tuple[np.ndarray, dict]:
    candidates = []
    for gamma in RBF_GAMMAS:
        for alpha in RBF_ALPHAS:
            model = KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha)
            model.fit(bits(fit_masks), cube[fit_masks])
            candidates.append((_mse(model.predict(bits(validation_masks)),
                                    cube[validation_masks]), gamma, alpha))
    best = min(candidates, key=lambda item: item[:3])
    model = KernelRidge(kernel="rbf", gamma=best[1], alpha=best[2])
    model.fit(bits(sample), cube[sample])
    return model.predict(bits(ALL_MASKS)), {"gamma": best[1], "alpha": best[2],
                                            "validation_mse": best[0]}


def refit_tn(family: str, rank: int, sample: np.ndarray, cube: np.ndarray, seed: int,
             weight_decay: float, functional_penalty: float, steps: int) -> np.ndarray:
    """Refit the selected tensor-network configuration on every sampled cell.

    Selection used an internal split with early stopping; the final model reuses
    the selected configuration and the selected number of steps, so that every
    family is refitted on the full sampled budget.
    """
    model = make_model(family, rank, cube.shape[1], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=weight_decay)
    x = torch.as_tensor(bits(sample), dtype=torch.float64)
    y = torch.as_tensor(cube[sample], dtype=torch.float64)
    everything = torch.as_tensor(bits(ALL_MASKS), dtype=torch.float64)
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = (torch.mean((model(x) - y) ** 2)
                + functional_penalty * torch.mean(model(everything) ** 2))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
    with torch.no_grad():
        return model(everything).numpy()


def fit_tensor_network(family: str, cube: np.ndarray, fit_masks: np.ndarray,
                       validation_masks: np.ndarray, sample: np.ndarray, seed: int
                       ) -> tuple[np.ndarray, dict]:
    xfit, xvalidation = bits(fit_masks), bits(validation_masks)
    weights = np.ones(len(fit_masks))
    candidates = []
    for rank in TN_RANKS:
        for penalty in TN_LAMBDAS:
            for restart in range(TN_RESTARTS):
                mse, _, steps = fit_tn(family, rank, xfit, cube[fit_masks], weights,
                                       xvalidation, cube[validation_masks], bits(ALL_MASKS),
                                       seed * 131 + restart, penalty, penalty / 10)
                candidates.append((mse, rank, penalty, restart, steps))
    best = min(candidates, key=lambda item: item[:4])
    prediction = refit_tn(family, best[1], sample, cube, seed * 131 + best[3],
                          best[2], best[2] / 10, best[4])
    return prediction, {"rank": best[1], "lambda": best[2], "restart": best[3],
                        "steps": best[4], "validation_mse": best[0]}


def ladder_prediction(coefficient: np.ndarray, rank: int, degree: int,
                      masks: np.ndarray) -> np.ndarray:
    """Degree-capped contraction of the TT-compressed coefficient tensor."""
    term_list = terms(degree)
    x = bits(masks)
    columns = []
    for output in range(coefficient.shape[1]):
        dense = coefficient_tensor(D, term_list, coefficient[1:, output])
        ladder = DegreeLadderTT.from_tensor(dense, rank)
        columns.append(coefficient[0, output] + ladder.predict(x, degree))
    return np.column_stack(columns)


def fit_tt_ladder(cube: np.ndarray, fit_masks: np.ndarray, validation_masks: np.ndarray,
                  sample: np.ndarray) -> tuple[np.ndarray, dict]:
    """Climb the degree ladder while validation MSE improves by at least 1%."""
    accepted, path = None, []
    for degree in LADDER_DEGREES:
        term_list = terms(degree)
        candidates = []
        for alpha in ALPHAS:
            coefficient = ridge(design(fit_masks, term_list), cube[fit_masks], alpha)
            for rank in LADDER_RANKS:
                prediction = ladder_prediction(coefficient, rank, degree, validation_masks)
                candidates.append((_mse(prediction, cube[validation_masks]), rank, alpha))
        best = min(candidates, key=lambda item: item[:3])
        path.append({"degree": degree, "rank": best[1], "alpha": best[2],
                     "validation_mse": best[0]})
        if accepted is None:
            accepted = {"degree": degree, "rank": best[1], "alpha": best[2],
                        "validation_mse": best[0]}
            continue
        gain = (accepted["validation_mse"] - best[0]) / max(accepted["validation_mse"], 1e-12)
        if gain >= LADDER_RELATIVE_GAIN:
            accepted = {"degree": degree, "rank": best[1], "alpha": best[2],
                        "validation_mse": best[0]}
        else:
            break
    term_list = terms(accepted["degree"])
    coefficient = ridge(design(sample, term_list), cube[sample], accepted["alpha"])
    prediction = ladder_prediction(coefficient, accepted["rank"], accepted["degree"], ALL_MASKS)
    return prediction, {**accepted, "path": path}


def fit_all(cube: np.ndarray, sample: np.ndarray, seed: int) -> tuple[dict, dict]:
    """Fit every family on `sample` and predict all 256 cells."""
    fit_masks, validation_masks = stratified_split(sample, seed)
    predictions, choices = {}, {}
    plans = {
        "cardinality": lambda: fit_polynomial(None, cube, fit_masks, validation_masks, sample),
        "linear": lambda: fit_polynomial(terms(1), cube, fit_masks, validation_masks, sample),
        "degree2_ridge": lambda: fit_polynomial(terms(2), cube, fit_masks, validation_masks,
                                                sample),
        "degree3_ridge": lambda: fit_polynomial(terms(3), cube, fit_masks, validation_masks,
                                                sample),
        "sparse_degree3": lambda: fit_sparse_degree3(cube, fit_masks, validation_masks, sample),
        "rbf_kernel": lambda: fit_rbf(cube, fit_masks, validation_masks, sample),
        "cp": lambda: fit_tensor_network("cp", cube, fit_masks, validation_masks, sample, seed),
        "tt": lambda: fit_tensor_network("tt", cube, fit_masks, validation_masks, sample, seed),
        "ht": lambda: fit_tensor_network("ht", cube, fit_masks, validation_masks, sample, seed),
        "tt_ladder": lambda: fit_tt_ladder(cube, fit_masks, validation_masks, sample),
    }
    for name, plan in plans.items():
        prediction, choice = plan()
        predictions[name] = prediction
        choices[name] = choice
    return predictions, choices


def containment_check(cube: np.ndarray) -> float:
    """The ladder must reproduce a degree-3 ridge exactly at sufficient rank."""
    term_list = terms(3)
    coefficient = ridge(design(ALL_MASKS, term_list), cube, 1.0)
    direct = design(ALL_MASKS, term_list) @ coefficient
    mapped = ladder_prediction(coefficient, 2 ** (D // 2), 3, ALL_MASKS)
    return float(np.max(np.abs(direct - mapped)))


__all__ = ["FAMILIES", "MAIN_FAMILIES", "fit_all", "containment_check", "bits", "ALL_MASKS",
           "copy"]
