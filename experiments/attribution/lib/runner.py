"""Question A: one cube, every method, every budget, at equal evaluation cost.

For each budget m and each sampling repeat, every method is handed at most m
*distinct* coalition evaluations.  Direct estimators answer the query family they
were designed for; a datamodel is fitted once on the same m coalitions and then
answers every family from the same fit, which is exactly the asymmetry the
experiment is meant to quantify.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass

import numpy as np

from . import design, direct, exact, metrics, protocol, surrogates

TARGET_COLUMNS = [
    "shapley_rel_l2", "shapley_spearman",
    "banzhaf_rel_l2", "banzhaf_spearman",
    "loo_rel_l2", "loo_spearman",
    "sii_rel_l2", "sii_spearman", "sii_precision_at_k",
    "fsii_rel_l2",
    "sii3_rel_l2", "sii3_precision_at_k",
    "heldout_nmse", "selection_regret", "selection_regret_normalised",
]

ROW_FIELDS = ["cube", "d", "family", "method", "method_type", "design", "output", "output_name",
              "budget", "budget_multiple", "repeat", "evaluations", "fit_seconds",
              *TARGET_COLUMNS, "choice"]


@dataclass
class Truth:
    """Every exact target for one cube, computed once by enumeration."""

    shapley: np.ndarray
    banzhaf: np.ndarray
    loo: np.ndarray
    sii: np.ndarray
    fsii: np.ndarray
    sii3: np.ndarray
    values: np.ndarray
    selection_size: int
    selection_scale: np.ndarray


def exact_targets(values: np.ndarray, d: int) -> Truth:
    _, fsii = exact.fsii(values, 2)
    size = protocol.SELECTION_SIZE[d]
    candidates = np.flatnonzero(exact.popcounts(d) == size)
    block = values[candidates]
    scale = block.max(0) - block.min(0)
    scale[scale == 0] = 1.0
    return Truth(
        shapley=exact.shapley_values(values),
        banzhaf=exact.banzhaf_values(values),
        loo=exact.loo_values(values),
        sii=exact.pair_interactions(values, shapley=True),
        fsii=np.atleast_2d(fsii) if fsii.ndim == 1 else fsii,
        sii3=exact.triple_interactions(values, shapley=True),
        values=values,
        selection_size=size,
        selection_scale=scale,
    )


def _score_vector(estimate, truth, prefix, row, spearman=True, k=None):
    if estimate is None:
        return
    row[f"{prefix}_rel_l2"] = metrics.relative_l2(estimate, truth)
    if spearman:
        row[f"{prefix}_spearman"] = metrics.spearman(estimate, truth)
    if k is not None:
        row[f"{prefix}_precision_at_k"] = metrics.precision_at_k(estimate, truth, k)


def _blank_row(**kwargs) -> dict:
    row = {field: "" for field in ROW_FIELDS}
    row.update(kwargs)
    return row


def score_direct(estimate: direct.Estimate, truth: Truth, d: int, output: int) -> dict:
    row = {}
    if estimate.shapley is not None:
        _score_vector(estimate.shapley[:, output], truth.shapley[:, output], "shapley", row)
    if estimate.banzhaf is not None:
        _score_vector(estimate.banzhaf[:, output], truth.banzhaf[:, output], "banzhaf", row)
    if estimate.loo is not None:
        _score_vector(estimate.loo[:, output], truth.loo[:, output], "loo", row)
    if estimate.pair_sii is not None:
        _score_vector(estimate.pair_sii[:, output], truth.sii[:, output], "sii", row,
                      k=protocol.precision_k(d))
    if estimate.pair_fsii is not None:
        _score_vector(estimate.pair_fsii[:, output], truth.fsii[:, output], "fsii", row,
                      spearman=False)
    return row


def score_surrogate(prediction: np.ndarray, truth: Truth, d: int, output: int,
                    held_out: np.ndarray) -> dict:
    row = {}
    _score_vector(exact.shapley_values(prediction)[:, output], truth.shapley[:, output],
                  "shapley", row)
    _score_vector(exact.banzhaf_values(prediction)[:, output], truth.banzhaf[:, output],
                  "banzhaf", row)
    _score_vector(exact.loo_values(prediction)[:, output], truth.loo[:, output], "loo", row)
    _score_vector(exact.pair_interactions(prediction, shapley=True)[:, output],
                  truth.sii[:, output], "sii", row, k=protocol.precision_k(d))
    _, predicted_fsii = exact.fsii(prediction, 2)
    _score_vector(predicted_fsii[:, output], truth.fsii[:, output], "fsii", row, spearman=False)
    _score_vector(exact.triple_interactions(prediction, shapley=True)[:, output],
                  truth.sii3[:, output], "sii3", row, spearman=False,
                  k=protocol.precision_k(d))
    row["heldout_nmse"] = metrics.normalised_mse(prediction[:, output], truth.values[:, output],
                                                 held_out)
    regret = exact.selection_regret(truth.values[:, output], prediction[:, output],
                                    truth.selection_size)
    row["selection_regret"] = float(regret)
    row["selection_regret_normalised"] = float(regret / truth.selection_scale[output])
    return row


def run_cube(cube, writer: csv.DictWriter, verbose: bool = True) -> None:
    values = cube.values
    d = cube.d
    truth = exact_targets(values, d)
    all_masks = np.arange(1 << d)
    full_range = abs(float(values[(1 << d) - 1, cube.primary_output] - values[0, cube.primary_output]))
    tolerance = protocol.TMC_TOLERANCE_FRACTION * (full_range if full_range > 0 else 1.0)

    for budget in protocol.budgets(d):
        multiple = protocol.budget_multiple(d, budget)
        for repeat in range(protocol.REPEATS):
            # ---- direct estimators -------------------------------------------------
            for method in ["permutation_shapley", "tmc_shapley", "kernel_shap",
                           "stratified_shapley", "msr_banzhaf", "direct_loo",
                           "sampling_sii", "fsii_regression"]:
                rng = np.random.default_rng(protocol.seed_for(cube.name, budget, repeat, method))
                start = time.perf_counter()
                if method == "tmc_shapley":
                    estimate = direct.permutation_shapley(values, d, budget, rng,
                                                          truncation=tolerance)
                else:
                    estimate = direct.ESTIMATORS[method](values, d, budget, rng)
                elapsed = time.perf_counter() - start
                if estimate.evaluations > budget:
                    raise RuntimeError(f"{method} used {estimate.evaluations} > {budget}")
                for output in range(cube.n_outputs):
                    row = _blank_row(cube=cube.name, d=d, family=cube.family, method=method,
                                     method_type="direct", design="native", output=output,
                                     output_name=cube.output_names[output], budget=budget,
                                     budget_multiple=multiple, repeat=repeat,
                                     evaluations=estimate.evaluations,
                                     fit_seconds=round(elapsed, 4),
                                     choice=str(estimate.detail))
                    row.update(score_direct(estimate, truth, d, output))
                    writer.writerow(row)

            # ---- datamodels --------------------------------------------------------
            for design_name in protocol.DESIGNS:
                rng = np.random.default_rng(
                    protocol.seed_for(cube.name, budget, repeat, design_name))
                masks = np.asarray(design.DESIGNS[design_name](d, budget, rng))
                held_out = np.setdiff1d(all_masks, masks)
                y = values[masks]
                # One split seed per (cube, budget, repeat, design): every family sees the
                # same fit/validation partition, so validation MSEs are comparable and
                # can be used to select a family without touching held-out coalitions.
                seed = protocol.seed_for(cube.name, budget, repeat, design_name + "split")
                for family, fit_fn in surrogates.FAMILIES.items():
                    start = time.perf_counter()
                    try:
                        fit = fit_fn(masks, y, d, seed % (2 ** 31))
                        prediction = fit.prediction
                        choice = fit.choice
                    except np.linalg.LinAlgError as error:
                        prediction = np.full((1 << d, y.shape[1]), np.nan)
                        choice = {"error": str(error)}
                    elapsed = time.perf_counter() - start
                    for output in range(cube.n_outputs):
                        row = _blank_row(cube=cube.name, d=d, family=cube.family, method=family,
                                         method_type="datamodel", design=design_name,
                                         output=output, output_name=cube.output_names[output],
                                         budget=budget, budget_multiple=multiple, repeat=repeat,
                                         evaluations=len(masks), fit_seconds=round(elapsed, 4),
                                         choice=str(choice))
                        row.update(score_surrogate(prediction, truth, d, output, held_out))
                        writer.writerow(row)
            if verbose:
                print(f"  {cube.name} m={budget} repeat={repeat} done", flush=True)
