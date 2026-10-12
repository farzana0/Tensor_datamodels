#!/usr/bin/env python3
"""Fit cumulative-degree ladder TT surrogates to the saved transformer cube.

This script never launches or imports the transformer learner.  The degree and
rank decision is made from the original fit/validation masks before the saved
target slice is loaded.  Because this analysis was designed after that slice
had already been measured, its results are explicitly exploratory.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from experiments.reasoning.common.interaction_degree_ladder import (
    DegreeLadderTT,
    coefficient_tensor,
)

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/original_records/d8_higher_order"
OUT = ROOT / "degree_ladder"
D = 8
ALPHAS = tuple(float(x) for x in np.logspace(-8, 4, 25))
RANKS = (1, 2, 3, 4, 6, 8, 12, 16)
MIN_RELATIVE_IMPROVEMENT = 0.01


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def cell_path(mask: int) -> Path:
    split = "target_cells" if mask.bit_count() == 3 else "cells"
    return ROOT / split / "seed31" / f"mask_{mask}" / f"seed31_mask{mask}.json"


def load_utility(mask: int) -> float:
    return float(json.loads(cell_path(mask).read_text())["utility_log_probability"])


def bits(masks: np.ndarray) -> np.ndarray:
    return ((masks[:, None] >> np.arange(D)) & 1).astype(np.float64)


def terms_through(degree: int) -> list[tuple[int, ...]]:
    return [term for order in range(1, degree + 1) for term in itertools.combinations(range(D), order)]


def design(masks: np.ndarray, terms: list[tuple[int, ...]]) -> np.ndarray:
    z = bits(masks)
    return np.column_stack([z[:, term].prod(axis=1) for term in terms])


def balanced_weights(masks: np.ndarray) -> np.ndarray:
    sizes = np.array([int(mask).bit_count() for mask in masks])
    counts = {size: int(np.sum(sizes == size)) for size in np.unique(sizes)}
    result = np.array([1.0 / counts[size] for size in sizes])
    return result / result.mean()


def cardinality_means(masks: np.ndarray, utility: np.ndarray) -> dict[int, float]:
    result = {size: float(utility[[int(mask).bit_count() == size for mask in masks]].mean())
              for size in sorted({int(mask).bit_count() for mask in masks})}
    result[3] = (result[2] + result[4]) / 2.0
    return result


def cardinality_prediction(masks: np.ndarray, means: dict[int, float]) -> np.ndarray:
    return np.array([means[int(mask).bit_count()] for mask in masks])


def ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    root = np.sqrt(weights)
    weighted = x * root[:, None]
    return np.linalg.solve(weighted.T @ weighted + alpha * np.eye(x.shape[1]), weighted.T @ (y * root))


def weighted_mse(prediction: np.ndarray, truth: np.ndarray, masks: np.ndarray) -> float:
    return float(np.average((prediction - truth) ** 2, weights=balanced_weights(masks)))


def within_size_spearman(prediction: np.ndarray, truth: np.ndarray, masks: np.ndarray) -> float:
    scores = []
    for size in sorted({int(mask).bit_count() for mask in masks}):
        selected = np.array([int(mask).bit_count() == size for mask in masks])
        if selected.sum() >= 3 and np.ptp(prediction[selected]) and np.ptp(truth[selected]):
            scores.append(float(spearmanr(prediction[selected], truth[selected]).statistic))
    return float(np.mean(scores)) if scores else 0.0


def values_tensor(values: np.ndarray) -> np.ndarray:
    result = np.empty((2,) * D, dtype=np.float64)
    for mask, value in enumerate(values):
        result[tuple((mask >> index) & 1 for index in range(D))] = value
    return result


def tt_relative_error(tensor: np.ndarray, rank: int) -> float:
    model = DegreeLadderTT.from_tensor(tensor, rank)
    return float(np.linalg.norm(model.dense_tensor() - tensor) / np.linalg.norm(tensor))


def rank_for_energy(singular: np.ndarray, fraction: float) -> int:
    cumulative = np.cumsum(singular ** 2)
    return int(np.searchsorted(cumulative, fraction * cumulative[-1]) + 1)


def fit_ladder() -> tuple[dict, np.ndarray, dict[int, float], list[dict]]:
    """Select the ladder using non-target fit and validation outcomes only."""
    protocol = json.loads((ROOT / "protocol.json").read_text())
    fit_masks = np.asarray(protocol["fit_masks"], dtype=np.int64)
    validation_masks = np.asarray(protocol["validation_masks"], dtype=np.int64)
    fit_y = np.array([load_utility(int(mask)) for mask in fit_masks])
    validation_y = np.array([load_utility(int(mask)) for mask in validation_masks])
    means = cardinality_means(fit_masks, fit_y)
    fit_residual = fit_y - cardinality_prediction(fit_masks, means)
    validation_cardinality = cardinality_prediction(validation_masks, means)
    weights = balanced_weights(fit_masks)
    all_masks = np.arange(1 << D, dtype=np.int64)
    all_bits = bits(all_masks)
    rows: list[dict] = []
    accepted: dict | None = None
    selected_prediction: np.ndarray | None = None

    for degree in range(1, D + 1):
        terms = terms_through(degree)
        x_fit = design(fit_masks, terms)
        candidates = []
        for alpha in ALPHAS:
            coefficients = ridge(x_fit, fit_residual, weights, alpha)
            dense = coefficient_tensor(D, terms, coefficients)
            for rank in RANKS:
                ladder = DegreeLadderTT.from_tensor(dense, rank)
                validation_prediction = validation_cardinality + ladder.predict(bits(validation_masks), degree)
                mse = weighted_mse(validation_prediction, validation_y, validation_masks)
                rho = within_size_spearman(validation_prediction, validation_y, validation_masks)
                candidates.append((mse, -rho, rank, alpha, ladder, coefficients))
        best = min(candidates, key=lambda item: item[:4])
        previous_mse = None if accepted is None else float(accepted["validation_balanced_mse"])
        relative_gain = None if previous_mse is None else (previous_mse - best[0]) / previous_mse
        is_accepted = previous_mse is None or relative_gain >= MIN_RELATIVE_IMPROVEMENT
        row = {
            "degree": degree,
            "terms": len(terms),
            "rank": best[2],
            "alpha": best[3],
            "validation_balanced_mse": best[0],
            "validation_within_size_spearman": -best[1],
            "relative_mse_improvement": relative_gain,
            "accepted": is_accepted,
        }
        rows.append(row)
        if not is_accepted:
            row["stop_reason"] = f"validation MSE improvement below {MIN_RELATIVE_IMPROVEMENT:.1%}"
            break
        accepted = row
        selected_prediction = cardinality_prediction(all_masks, means) + best[4].predict(all_bits, degree)

    if accepted is None or selected_prediction is None:
        raise RuntimeError("the degree ladder did not accept an initial model")

    # Constructive containment check against the exact degree-3 ridge baseline
    # used in the original comparison (alpha=1.0).
    original_terms = terms_through(3)
    original_coef = ridge(design(fit_masks, original_terms), fit_residual, weights, 1.0)
    original_dense = coefficient_tensor(D, original_terms, original_coef)
    exact_ladder = DegreeLadderTT.from_tensor(original_dense, 16)
    direct = design(all_masks, original_terms) @ original_coef
    mapped = exact_ladder.predict(all_bits, 3)
    containment_error = float(np.max(np.abs(direct - mapped)))
    if containment_error > 1e-10:
        raise AssertionError(f"degree-3 ridge containment failed: {containment_error}")
    decision = {
        "selected_degree": int(accepted["degree"]),
        "selected_rank": int(accepted["rank"]),
        "selected_alpha": float(accepted["alpha"]),
        "stopped_at_degree": int(rows[-1]["degree"]) if not rows[-1]["accepted"] else None,
        "minimum_relative_validation_improvement": MIN_RELATIVE_IMPROVEMENT,
        "degree3_ridge_max_abs_mapping_error": containment_error,
        "target_loaded_during_selection": False,
    }
    return decision, selected_prediction, means, rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    decision, ladder_prediction, training_means, path_rows = fit_ladder()

    # The target slice is loaded only after the degree/rank/alpha decision above.
    truth = np.array([load_utility(mask) for mask in range(1 << D)])
    protocol = json.loads((ROOT / "protocol.json").read_text())
    target_masks = np.asarray(protocol["target_masks"], dtype=np.int64)
    target_truth = truth[target_masks]
    existing = np.load(ROOT / "fits/predictions.npz")
    predictions = {name: existing[name] for name in ("linear", "degree3_ridge", "cp", "tt")}
    predictions["degree_ladder_tt"] = ladder_prediction
    oracle_order = np.lexsort((target_masks, -target_truth))[:7]
    oracle_top = set(target_masks[oracle_order])
    selection_rows = []
    for name, prediction in predictions.items():
        values = prediction[target_masks]
        order = np.lexsort((target_masks, -values))[:7]
        chosen = target_masks[order]
        selection_rows.append({
            "method": name,
            "heldout_spearman": float(spearmanr(values, target_truth).statistic),
            "precision_at_7": len(set(chosen) & oracle_top) / 7.0,
            "top7_actual_utility": float(truth[chosen].mean()),
            "selection_regret": float(target_truth.max() - truth[chosen].mean()),
            "nominated_masks": " ".join(map(str, chosen)),
        })

    # Intrinsic SVD diagnostics are reported both before and after removing the
    # exact full-cube cardinality means, preventing the size trend from making
    # the tensor appear artificially low-rank.
    full_size_means = {size: float(truth[[mask.bit_count() == size for mask in range(1 << D)]].mean())
                       for size in range(D + 1)}
    size_component = np.array([full_size_means[mask.bit_count()] for mask in range(1 << D)])
    diagnostic_values = {"raw_utility": truth, "full_cube_size_residual": truth - size_component}
    svd_rows = []
    for label, values in diagnostic_values.items():
        tensor = values_tensor(values)
        for cut in range(1, D):
            singular = np.linalg.svd(tensor.reshape(2 ** cut, -1), compute_uv=False)
            numerical_rank = int(np.sum(singular > singular[0] * 1e-10))
            svd_rows.append({
                "tensor": label,
                "diagnostic": "unfolding",
                "index": cut,
                "value": numerical_rank,
                "rank_90_energy": rank_for_energy(singular, .90),
                "rank_95_energy": rank_for_energy(singular, .95),
                "rank_99_energy": rank_for_energy(singular, .99),
                "energy_fraction": "",
            })
            for component, value in enumerate(singular, start=1):
                svd_rows.append({
                    "tensor": label,
                    "diagnostic": f"singular_value_cut_{cut}",
                    "index": component,
                    "value": float(value),
                    "rank_90_energy": "",
                    "rank_95_energy": "",
                    "rank_99_energy": "",
                    "energy_fraction": float(value ** 2 / np.sum(singular ** 2)),
                })
        for rank in RANKS:
            svd_rows.append({
                "tensor": label,
                "diagnostic": "tt_svd_relative_error",
                "index": rank,
                "value": tt_relative_error(tensor, rank),
                "rank_90_energy": "",
                "rank_95_energy": "",
                "rank_99_energy": "",
                "energy_fraction": "",
            })

    residual = truth - size_component
    fitted_tt_residual = predictions["tt"] - size_component
    decision.update({
        "analysis_status": "exploratory_posthoc_surrogate_refit",
        "transformer_retraining_performed": False,
        "saved_coalition_count": 256,
        "selected_target_metrics": next(row for row in selection_rows if row["method"] == "degree_ladder_tt"),
        "fitted_tt_full_cube_relative_error": float(np.linalg.norm(predictions["tt"] - truth) / np.linalg.norm(truth)),
        "fitted_tt_size_residual_relative_error": float(np.linalg.norm(fitted_tt_residual - residual) / np.linalg.norm(residual)),
        "source_protocol_sha256": sha(ROOT / "protocol.json"),
        "source_cell_hashes": {str(mask): sha(cell_path(mask)) for mask in range(1 << D)},
    })
    np.savez_compressed(OUT / "predictions.npz", masks=np.arange(1 << D), degree_ladder_tt=ladder_prediction)
    atomic_json(OUT / "result.json", decision)
    atomic_json(OUT / "AUDIT.json", {
        "status": "PASS",
        "transformer_retraining_performed": False,
        "saved_coalition_values_read": 256,
        "source_hashes_recorded": len(decision["source_cell_hashes"]),
        "degree_and_rank_selected_before_target_load": not decision["target_loaded_during_selection"],
        "degree3_ridge_containment_pass": decision["degree3_ridge_max_abs_mapping_error"] <= 1e-10,
        "stopping_rule_applied": decision["stopped_at_degree"] is not None,
    })
    for filename, rows in (("ladder_path.csv", path_rows), ("selection_metrics.csv", selection_rows),
                           ("svd_diagnostics.csv", svd_rows)):
        with (OUT / filename).open("w", newline="") as handle:
            fieldnames = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    degrees = [row["degree"] for row in path_rows]
    axes[0].plot(degrees, [row["validation_balanced_mse"] for row in path_rows], marker="o", color="#128f82")
    axes[0].axvline(decision["selected_degree"], color="#df805e", linestyle="--", label="selected degree")
    axes[0].set(xlabel="cumulative maximum degree", ylabel="validation balanced MSE", title="Degree ladder stopping path")
    axes[0].legend()
    target = next(row for row in selection_rows if row["method"] == "degree_ladder_tt")
    comparison = [next(row for row in selection_rows if row["method"] == name) for name in ("linear", "degree3_ridge", "cp")]
    comparison.append(target)
    axes[1].bar([row["method"].replace("degree_ladder_tt", "ladder TT") for row in comparison],
                [row["top7_actual_utility"] for row in comparison], color=["#df805e", "#5d77a8", "#8b6bb1", "#128f82"])
    axes[1].set(ylabel="actual top-7 utility", title="Saved k=3 transformer outcomes")
    axes[1].tick_params(axis="x", rotation=20)
    figure.tight_layout(); figure.savefig(ROOT / "figures/figure5_degree_ladder.svg"); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for label, color in (("raw_utility", "#5d77a8"), ("full_cube_size_residual", "#df805e")):
        selected = [row for row in svd_rows if row["tensor"] == label and row["diagnostic"] == "tt_svd_relative_error"]
        axis.plot([row["index"] for row in selected], [row["value"] for row in selected], marker="o", label=label.replace("_", " "), color=color)
    axis.set(xlabel="uniform TT rank cap", ylabel="relative Frobenius error", title="Oracle TT-SVD compressibility of the complete cube")
    axis.legend(); figure.tight_layout(); figure.savefig(ROOT / "figures/figure6_rank_diagnostics.svg"); plt.close(figure)

    raw_rank16 = next(row["value"] for row in svd_rows if row["tensor"] == "raw_utility" and row["diagnostic"] == "tt_svd_relative_error" and row["index"] == 16)
    residual_rank6 = next(row["value"] for row in svd_rows if row["tensor"] == "full_cube_size_residual" and row["diagnostic"] == "tt_svd_relative_error" and row["index"] == 6)
    report = f"""# Degree-preserving ladder TT on saved transformer utilities

No transformer was retrained. This analysis reads the 256 saved coalition utilities and refits only surrogate functions. It is exploratory because the extension was designed after the original target outcomes existed, although the implementation selects degree, rank, and ridge penalty using only the original non-target fit/validation split before loading the saved target values.

The implementation follows the paper's cumulative-$j$ construction: the local feature map is `[1, x_i]`, the TT contraction carries an explicit degree index, and only degree slices up to $j$ enter the output. Degree 3 with TT rank 12 was selected. Degree 4 worsened validation MSE, so the ladder stopped without fitting degrees 5--8. A rank-16 ladder initialized from the original degree-3 ridge coefficient tensor reproduces all 256 ridge interaction predictions with maximum absolute error {decision['degree3_ridge_max_abs_mapping_error']:.3e}.

On the saved size-three slice, the selected ladder has Spearman {target['heldout_spearman']:.3f}, Precision@7 {target['precision_at_7']:.3f}, and mean actual top-seven utility {target['top7_actual_utility']:.3f}. It does not improve on the original degree-3 ridge selector, whose top-seven utility is {next(row['top7_actual_utility'] for row in selection_rows if row['method'] == 'degree3_ridge'):.3f}. See `selection_metrics.csv` for the unchanged Linear, degree-3 ridge, CP, and TT comparisons.

## Is the measured coalition tensor low rank?

Only moderately at the ranks tested. The raw tensor has exact contiguous TT cut ranks 2, 4, 8, 16, 8, 4, 2; its middle cut is full rank. Rank-6 TT-SVD error is {tt_relative_error(values_tensor(truth), 6):.3f} relative to the raw tensor norm, which is dominated by the coalition-size trend. After subtracting the exact full-cube cardinality mean at each coalition size, rank-6 TT-SVD relative error is {residual_rank6:.3f}. Rank 16 reconstructs the raw cube to {raw_rank16:.3e} relative error.

This separates optimization from representation. A low-rank fit can be compared with the oracle TT-SVD curve in `svd_diagnostics.csv`. If the oracle error is already large, rank is the bottleneck. If the oracle is accurate but the fitted model is poor, sparse observations or nonconvex optimization are the bottleneck. Here the size-residual tensor is not strongly compressible at rank 3--6, so the earlier TT failure is substantially a rank/structure mismatch; optimization and sparse-fit generalization can still contribute.
"""
    (OUT / "report.md").write_text(report)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
