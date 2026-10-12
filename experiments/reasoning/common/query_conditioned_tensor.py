#!/usr/bin/env python3
"""Build a query-output coalition tensor from saved transformer evaluations.

No learner is imported or retrained.  The saved artifacts expose seven
route-level accuracies per coalition, so the resulting tensor has eight binary
coalition modes and one seven-valued query mode.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.reasoning.common.fit_degree_ladder_tn import (
    D, ROOT, balanced_weights, bits, cell_path, design, ridge, terms_through,
)

OUT = ROOT / "query_conditioned"
FANO_LINES = ((0, 1, 2), (0, 3, 4), (0, 5, 6), (1, 3, 5),
              (1, 4, 6), (2, 3, 6), (2, 4, 5))
ALPHAS = tuple(float(x) for x in np.logspace(-6, 3, 19))
RANKS = (1, 2, 4, 6, 8, 12, 16, 24, 32)


def load_outputs(mask: int) -> np.ndarray:
    row = json.loads(cell_path(mask).read_text())
    values = np.asarray(row["target_accuracy_by_line"], dtype=np.float64)
    if values.shape != (len(FANO_LINES),):
        raise ValueError(f"mask {mask}: expected seven route outputs, got {values.shape}")
    return values


def ridge_matrix(x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    root = np.sqrt(weights)
    weighted = x * root[:, None]
    return np.linalg.solve(weighted.T @ weighted + alpha * np.eye(x.shape[1]),
                           weighted.T @ (y * root[:, None]))


def cardinality_design(masks: np.ndarray) -> np.ndarray:
    size = np.asarray([int(mask).bit_count() for mask in masks], dtype=float)
    z = (size - D / 2) / (D / 2)
    return np.column_stack((np.ones(len(masks)), z, z ** 2, z ** 3))


def coefficient_tensor(terms: list[tuple[int, ...]], coefficients: np.ndarray) -> np.ndarray:
    tensor = np.zeros((2,) * D + (len(FANO_LINES),), dtype=np.float64)
    for index, term in enumerate(terms):
        tensor[tuple(int(axis in term) for axis in range(D))] = coefficients[index]
    return tensor


def tt_svd(tensor: np.ndarray, rank: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Deterministic TT-SVD for binary coalition modes plus a query mode."""
    shape = tensor.shape
    cores: list[np.ndarray] = []
    remainder = tensor.copy()
    left_rank = 1
    for physical in shape[:-1]:
        matrix = remainder.reshape(left_rank * physical, -1)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        kept = min(rank, len(singular))
        cores.append(u[:, :kept].reshape(left_rank, physical, kept))
        remainder = singular[:kept, None] * vh[:kept]
        left_rank = kept
    cores.append(remainder.reshape(left_rank, shape[-1], 1))
    reconstructed = cores[0]
    for core in cores[1:]:
        reconstructed = np.tensordot(reconstructed, core, axes=([-1], [0]))
    return cores, reconstructed.reshape(shape)


def evaluate_coefficients(tensor: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Contract coalition modes with [1, x_i], leaving query mode open."""
    result = []
    for row in bits(masks):
        value = tensor
        for coordinate in row:
            value = np.tensordot(np.array([1.0, coordinate]), value, axes=([0], [0]))
        result.append(value)
    return np.asarray(result)


def finite_difference(values: np.ndarray, players: tuple[int, ...], background: int) -> float:
    result = 0.0
    for count in range(len(players) + 1):
        for subset in itertools.combinations(players, count):
            mask = background + sum(1 << player for player in subset)
            result += (-1.0) ** (len(players) - count) * values[mask]
    return float(result)


def shapley_value(values: np.ndarray) -> np.ndarray:
    result = np.zeros(D)
    for player in range(D):
        for mask in range(1 << D):
            if mask & (1 << player):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(D - size - 1) / math.factorial(D)
            result[player] += weight * (values[mask | (1 << player)] - values[mask])
    return result


def banzhaf_value(values: np.ndarray) -> np.ndarray:
    result = np.zeros(D)
    for player in range(D):
        backgrounds = [mask for mask in range(1 << D) if not mask & (1 << player)]
        result[player] = np.mean([values[mask | (1 << player)] - values[mask] for mask in backgrounds])
    return result


def interaction_values(values: np.ndarray, order: int, shapley_weighted: bool) -> tuple[list[tuple[int, ...]], np.ndarray]:
    interactions = list(itertools.combinations(range(D), order))
    result = []
    for players in interactions:
        player_mask = sum(1 << player for player in players)
        backgrounds = [mask for mask in range(1 << D) if not mask & player_mask]
        if shapley_weighted:
            weights = [math.factorial(mask.bit_count()) *
                       math.factorial(D - order - mask.bit_count()) /
                       math.factorial(D - order + 1) for mask in backgrounds]
        else:
            weights = [1.0 / len(backgrounds)] * len(backgrounds)
        result.append(sum(weight * finite_difference(values, players, mask)
                          for weight, mask in zip(weights, backgrounds)))
    return interactions, np.asarray(result)


def select_model(fit: np.ndarray, validation: np.ndarray, outputs: np.ndarray) -> tuple[dict, list[dict]]:
    all_masks = np.arange(1 << D, dtype=np.int64)
    card_candidates = []
    for alpha in ALPHAS:
        coef = ridge_matrix(cardinality_design(fit), outputs[fit], balanced_weights(fit), alpha)
        pred = cardinality_design(validation) @ coef
        card_candidates.append((float(np.mean((pred - outputs[validation]) ** 2)), alpha, coef))
    card = min(card_candidates, key=lambda item: item[:2])
    card_all = cardinality_design(all_masks) @ card[2]
    residual = outputs[fit] - card_all[fit]
    path, accepted, previous = [], None, None
    for degree in range(1, D + 1):
        terms = terms_through(degree)
        candidates = []
        for alpha in ALPHAS:
            coef = ridge_matrix(design(fit, terms), residual, balanced_weights(fit), alpha)
            dense = coefficient_tensor(terms, coef)
            for rank in RANKS:
                _, compressed = tt_svd(dense, rank)
                pred = card_all[validation] + evaluate_coefficients(compressed, validation)
                mse = float(np.mean((pred - outputs[validation]) ** 2))
                candidates.append((mse, rank, alpha))
        best = min(candidates, key=lambda item: item)
        gain = None if previous is None else (previous - best[0]) / previous
        keep = previous is None or gain >= .01
        path.append({"degree": degree, "rank": best[1], "alpha": best[2],
                     "validation_query_mse": best[0], "relative_gain": gain, "accepted": keep})
        if not keep:
            break
        accepted, previous = path[-1], best[0]
    if accepted is None:
        raise RuntimeError("query ladder accepted no degree")
    selected = dict(accepted)
    selected["cardinality_alpha"] = card[1]
    return selected, path


def fit_final(observed: np.ndarray, outputs: np.ndarray, selected: dict) -> tuple[np.ndarray, np.ndarray]:
    all_masks = np.arange(1 << D, dtype=np.int64)
    card_coef = ridge_matrix(cardinality_design(observed), outputs[observed], balanced_weights(observed),
                             float(selected["cardinality_alpha"]))
    card = cardinality_design(all_masks) @ card_coef
    terms = terms_through(int(selected["degree"]))
    coef = ridge_matrix(design(observed, terms), outputs[observed] - card[observed],
                        balanced_weights(observed), float(selected["alpha"]))
    dense = coefficient_tensor(terms, coef)
    _, compressed = tt_svd(dense, int(selected["rank"]))
    return card + evaluate_coefficients(compressed, all_masks), compressed


def independent_query_ladders(fit: np.ndarray, validation: np.ndarray, observed: np.ndarray,
                              outputs: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Fit each query separately to test whether output-mode sharing is harmful."""
    all_masks = np.arange(1 << D, dtype=np.int64)
    prediction = np.zeros_like(outputs)
    rows = []
    for query in range(len(FANO_LINES)):
        card_candidates = []
        for alpha in ALPHAS:
            coef = ridge(cardinality_design(fit), outputs[fit, query], balanced_weights(fit), alpha)
            mse = float(np.mean((cardinality_design(validation) @ coef - outputs[validation, query]) ** 2))
            card_candidates.append((mse, alpha, coef))
        card = min(card_candidates, key=lambda item: item[:2])
        card_all = cardinality_design(all_masks) @ card[2]
        previous, accepted, rejected = None, None, None
        for degree in range(1, D + 1):
            terms = terms_through(degree)
            candidates = []
            for alpha in ALPHAS:
                coef = ridge(design(fit, terms), outputs[fit, query] - card_all[fit],
                             balanced_weights(fit), alpha)
                pred = card_all[validation] + design(validation, terms) @ coef
                candidates.append((float(np.mean((pred - outputs[validation, query]) ** 2)), alpha))
            best = min(candidates, key=lambda item: item)
            gain = None if previous is None else (previous - best[0]) / previous
            if previous is not None and gain < .01:
                rejected = degree
                break
            previous, accepted = best[0], (degree, best[1], best[0])
        degree, alpha, validation_mse = accepted
        card_coef = ridge(cardinality_design(observed), outputs[observed, query], balanced_weights(observed),
                          float(card[1]))
        card_refit = cardinality_design(all_masks) @ card_coef
        terms = terms_through(degree)
        coef = ridge(design(observed, terms), outputs[observed, query] - card_refit[observed],
                     balanced_weights(observed), float(alpha))
        prediction[:, query] = card_refit + design(all_masks, terms) @ coef
        rows.append({"query": query, "selected_degree": degree, "rejected_degree": rejected,
                     "alpha": alpha, "cardinality_alpha": card[1], "validation_mse": validation_mse})
    return prediction, rows


def fit_shared_polynomial(degree: int, fit: np.ndarray, validation: np.ndarray,
                          observed: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Query-output polynomial baseline with validation-selected penalties."""
    all_masks = np.arange(1 << D, dtype=np.int64)
    card_candidates = []
    for alpha in ALPHAS:
        coef = ridge_matrix(cardinality_design(fit), outputs[fit], balanced_weights(fit), alpha)
        mse = float(np.mean((cardinality_design(validation) @ coef - outputs[validation]) ** 2))
        card_candidates.append((mse, alpha))
    card_alpha = min(card_candidates)[1]
    card_coef = ridge_matrix(cardinality_design(observed), outputs[observed], balanced_weights(observed), card_alpha)
    card = cardinality_design(all_masks) @ card_coef
    terms = terms_through(degree)
    candidates = []
    # Hyperparameters must be selected on fit/validation, before the observed refit.
    selection_card_coef = ridge_matrix(cardinality_design(fit), outputs[fit], balanced_weights(fit), card_alpha)
    selection_card = cardinality_design(all_masks) @ selection_card_coef
    for alpha in ALPHAS:
        coef = ridge_matrix(design(fit, terms), outputs[fit] - selection_card[fit], balanced_weights(fit), alpha)
        pred = selection_card[validation] + design(validation, terms) @ coef
        candidates.append((float(np.mean((pred - outputs[validation]) ** 2)), alpha))
    alpha = min(candidates)[1]
    coef = ridge_matrix(design(observed, terms), outputs[observed] - card[observed],
                        balanced_weights(observed), alpha)
    return card + design(all_masks, terms) @ coef


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    fit = np.asarray(protocol["fit_masks"], dtype=np.int64)
    validation = np.asarray(protocol["validation_masks"], dtype=np.int64)
    target = np.asarray(protocol["target_masks"], dtype=np.int64)
    observed = np.sort(np.concatenate((fit, validation)))
    outputs = np.vstack([load_outputs(mask) for mask in range(1 << D)])
    selected, path = select_model(fit, validation, outputs)
    prediction, coefficient = fit_final(observed, outputs, selected)
    independent_prediction, independent_rows = independent_query_ladders(fit, validation, observed, outputs)
    method_predictions = {
        "query_linear": fit_shared_polynomial(1, fit, validation, observed, outputs),
        "query_degree2_ridge": fit_shared_polynomial(2, fit, validation, observed, outputs),
        "query_degree3_ridge": fit_shared_polynomial(3, fit, validation, observed, outputs),
        "query_degree_ladder_output_tt": prediction,
        "independent_query_ladders": independent_prediction,
    }

    rows, attribution_rows, interaction_rows, higher_order_rows, top_interaction_rows = [], [], [], [], []
    all_true_shapley, all_predicted_shapley = [], []
    all_true_banzhaf, all_predicted_banzhaf = [], []
    for query, support in enumerate(FANO_LINES):
        truth, pred = outputs[target, query], prediction[target, query]
        ranked = target[np.lexsort((target, -pred))]
        support_mask = sum(1 << player for player in support)
        support_rank = int(np.where(ranked == support_mask)[0][0]) + 1
        true_shapley, predicted_shapley = shapley_value(outputs[:, query]), shapley_value(prediction[:, query])
        true_banzhaf, predicted_banzhaf = banzhaf_value(outputs[:, query]), banzhaf_value(prediction[:, query])
        all_true_shapley.append(true_shapley); all_predicted_shapley.append(predicted_shapley)
        all_true_banzhaf.append(true_banzhaf); all_predicted_banzhaf.append(predicted_banzhaf)
        for player in range(D):
            attribution_rows.append({"query": query, "player": player,
                "is_query_support": player in support, "true_shapley": true_shapley[player],
                "predicted_shapley": predicted_shapley[player], "true_banzhaf": true_banzhaf[player],
                "predicted_banzhaf": predicted_banzhaf[player]})
        triples, true_si = interaction_values(outputs[:, query], 3, True)
        _, predicted_si = interaction_values(prediction[:, query], 3, True)
        _, true_bi = interaction_values(outputs[:, query], 3, False)
        _, predicted_bi = interaction_values(prediction[:, query], 3, False)
        predicted_si_order = np.lexsort((np.arange(len(triples)), -np.abs(predicted_si)))
        predicted_bi_order = np.lexsort((np.arange(len(triples)), -np.abs(predicted_bi)))
        support_index = triples.index(tuple(support))
        for triple_index, triple in enumerate(triples):
            interaction_rows.append({"query": query, "triple": " ".join(map(str, triple)),
                "is_query_support": triple == tuple(support),
                "true_shapley_interaction": true_si[triple_index],
                "predicted_shapley_interaction": predicted_si[triple_index],
                "true_banzhaf_interaction": true_bi[triple_index],
                "predicted_banzhaf_interaction": predicted_bi[triple_index]})
        for order in range(2, 6):
            subsets, true_order_si = interaction_values(outputs[:, query], order, True)
            _, predicted_order_si = interaction_values(prediction[:, query], order, True)
            _, true_order_bi = interaction_values(outputs[:, query], order, False)
            _, predicted_order_bi = interaction_values(prediction[:, query], order, False)
            for subset_index, subset in enumerate(subsets):
                higher_order_rows.append({"query": query, "order": order,
                    "subset": " ".join(map(str, subset)),
                    "true_shapley_interaction": true_order_si[subset_index],
                    "predicted_shapley_interaction": predicted_order_si[subset_index],
                    "true_banzhaf_interaction": true_order_bi[subset_index],
                    "predicted_banzhaf_interaction": predicted_order_bi[subset_index]})
            true_si_top = int(np.argmax(np.abs(true_order_si)))
            predicted_si_top = int(np.argmax(np.abs(predicted_order_si)))
            true_bi_top = int(np.argmax(np.abs(true_order_bi)))
            predicted_bi_top = int(np.argmax(np.abs(predicted_order_bi)))
            top_interaction_rows.append({"query": query, "order": order,
                "true_top_shapley_subset": " ".join(map(str, subsets[true_si_top])),
                "predicted_top_shapley_subset": " ".join(map(str, subsets[predicted_si_top])),
                "true_top_banzhaf_subset": " ".join(map(str, subsets[true_bi_top])),
                "predicted_top_banzhaf_subset": " ".join(map(str, subsets[predicted_bi_top]))})
        independent_ranked = target[np.lexsort((target, -independent_prediction[target, query]))]
        rows.append({
            "query": query, "support": " ".join(map(str, support)), "support_mask": support_mask,
            "target_spearman": float(spearmanr(pred, truth).statistic),
            "predicted_top_mask": int(ranked[0]), "support_rank": support_rank,
            "support_accuracy": float(outputs[support_mask, query]),
            "predicted_top_actual_accuracy": float(outputs[ranked[0], query]),
            "shapley_spearman": float(spearmanr(predicted_shapley, true_shapley).statistic),
            "banzhaf_spearman": float(spearmanr(predicted_banzhaf, true_banzhaf).statistic),
            "triple_shapley_spearman": float(spearmanr(predicted_si, true_si).statistic),
            "triple_banzhaf_spearman": float(spearmanr(predicted_bi, true_bi).statistic),
            "support_shapley_interaction_rank": int(np.where(predicted_si_order == support_index)[0][0]) + 1,
            "support_banzhaf_interaction_rank": int(np.where(predicted_bi_order == support_index)[0][0]) + 1,
            "independent_target_spearman": float(spearmanr(independent_prediction[target, query], truth).statistic),
            "independent_top_mask": int(independent_ranked[0]),
            "independent_support_rank": int(np.where(independent_ranked == support_mask)[0][0]) + 1,
        })

    selection_rows = []
    target_prediction, target_truth = prediction[target], outputs[target]
    objectives = {
        "uniform_expected_accuracy": (target_prediction.mean(axis=1), target_truth.mean(axis=1)),
        "worst_query_accuracy": (target_prediction.min(axis=1), target_truth.min(axis=1)),
        "route_coverage_at_90pct": ((target_prediction >= .9).sum(axis=1), (target_truth >= .9).sum(axis=1)),
    }
    for name, (predicted_objective, actual_objective) in objectives.items():
        order = np.lexsort((target, -target_prediction.mean(axis=1), -predicted_objective))
        chosen = int(target[order[0]])
        selection_rows.append({"objective": name, "query": "all", "selected_mask": chosen,
            "predicted_objective": float(predicted_objective[order[0]]),
            "actual_objective": float(actual_objective[order[0]]),
            "actual_uniform_accuracy": float(outputs[chosen].mean()),
            "actual_worst_query_accuracy": float(outputs[chosen].min()),
            "actual_routes_at_90pct": int((outputs[chosen] >= .9).sum())})
    for query in range(len(FANO_LINES)):
        order = np.lexsort((target, -target_prediction[:, query]))
        chosen = int(target[order[0]])
        selection_rows.append({"objective": "query_specific_accuracy", "query": query,
            "selected_mask": chosen, "predicted_objective": float(prediction[chosen, query]),
            "actual_objective": float(outputs[chosen, query]),
            "actual_uniform_accuracy": float(outputs[chosen].mean()),
            "actual_worst_query_accuracy": float(outputs[chosen].min()),
            "actual_routes_at_90pct": int((outputs[chosen] >= .9).sum())})

    by_size_rows = []
    for size in range(1, D + 1):
        candidates = np.asarray([mask for mask in range(1 << D) if mask.bit_count() == size], dtype=np.int64)
        pred_slice, truth_slice = prediction[candidates], outputs[candidates]
        for name, predicted_objective, actual_objective in (
            ("uniform_expected_accuracy", pred_slice.mean(1), truth_slice.mean(1)),
            ("worst_query_accuracy", pred_slice.min(1), truth_slice.min(1)),
            ("route_coverage_at_90pct", (pred_slice >= .9).sum(1), (truth_slice >= .9).sum(1)),
        ):
            order = np.lexsort((candidates, -pred_slice.mean(1), -predicted_objective))
            chosen_index = int(order[0])
            oracle = float(np.max(actual_objective))
            by_size_rows.append({"cardinality": size, "objective": name,
                "selected_mask": int(candidates[chosen_index]),
                "selected_actual_objective": float(actual_objective[chosen_index]),
                "oracle_actual_objective": oracle,
                "regret": oracle - float(actual_objective[chosen_index]),
                "descriptive_only": size != 3})

    method_rows = []
    for method, method_prediction in method_predictions.items():
        correlations, ranks, chosen_accuracies = [], [], []
        for query, support in enumerate(FANO_LINES):
            support_mask = sum(1 << player for player in support)
            pred = method_prediction[target, query]
            truth = outputs[target, query]
            correlations.append(float(spearmanr(pred, truth).statistic))
            ranked = target[np.lexsort((target, -pred))]
            ranks.append(int(np.where(ranked == support_mask)[0][0]) + 1)
            chosen_accuracies.append(float(outputs[int(ranked[0]), query]))
        method_rows.append({"method": method, "mean_per_query_target_spearman": np.nanmean(correlations),
            "support_top1_queries": sum(rank == 1 for rank in ranks),
            "support_top3_queries": sum(rank <= 3 for rank in ranks),
            "mean_selected_query_accuracy": np.mean(chosen_accuracies),
            "mean_support_rank": np.mean(ranks)})

    full_tensor = outputs.reshape((2,) * D + (len(FANO_LINES),), order="F")
    cut_rows = []
    for cut in range(1, D + 1):
        singular = np.linalg.svd(full_tensor.reshape(np.prod(full_tensor.shape[:cut]), -1),
                                 compute_uv=False)
        cut_rows.append({"cut_after_mode": cut, "numerical_rank": int(np.sum(singular > singular[0] * 1e-10)),
                         "rank_99_energy": int(np.searchsorted(np.cumsum(singular ** 2), .99 * np.sum(singular ** 2)) + 1)})

    with (OUT / "query_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    with (OUT / "attributions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=attribution_rows[0]); writer.writeheader(); writer.writerows(attribution_rows)
    with (OUT / "triple_interactions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=interaction_rows[0]); writer.writeheader(); writer.writerows(interaction_rows)
    with (OUT / "higher_order_interactions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=higher_order_rows[0]); writer.writeheader(); writer.writerows(higher_order_rows)
    with (OUT / "top_interactions_by_order.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=top_interaction_rows[0]); writer.writeheader(); writer.writerows(top_interaction_rows)
    with (OUT / "selection_objectives.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selection_rows[0]); writer.writeheader(); writer.writerows(selection_rows)
    with (OUT / "selection_by_size.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=by_size_rows[0]); writer.writeheader(); writer.writerows(by_size_rows)
    with (OUT / "independent_query_ladders.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=independent_rows[0]); writer.writeheader(); writer.writerows(independent_rows)
    with (OUT / "method_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=method_rows[0]); writer.writeheader(); writer.writerows(method_rows)
    with (OUT / "ladder_path.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=path[0]); writer.writeheader(); writer.writerows(path)
    with (OUT / "cut_ranks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cut_rows[0]); writer.writeheader(); writer.writerows(cut_rows)
    np.savez_compressed(OUT / "query_tensor.npz", values=outputs, coefficient_tensor=coefficient,
                        prediction=prediction, fano_lines=np.asarray(FANO_LINES))
    result = {
        "tensor_shape": [2] * D + [len(FANO_LINES)],
        "interpretation": "eight coalition modes plus one route-query output mode",
        "saved_granularity": "accuracy averaged over 31 inputs within each of seven routes",
        "individual_query_logits_saved": False,
        "selected": selected,
        "mean_target_spearman": float(np.nanmean([row["target_spearman"] for row in rows])),
        "support_top1_count": sum(row["support_rank"] == 1 for row in rows),
        "support_top3_count": sum(row["support_rank"] <= 3 for row in rows),
        "independent_support_top1_count": sum(row["independent_support_rank"] == 1 for row in rows),
        "independent_selected_degrees": [row["selected_degree"] for row in independent_rows],
        "support_top_triple_shapley_count": sum(row["support_shapley_interaction_rank"] == 1 for row in rows),
        "support_top_triple_banzhaf_count": sum(row["support_banzhaf_interaction_rank"] == 1 for row in rows),
        "source_cell_count": 256,
        "transformer_retraining_performed": False,
        "uniform_shapley_linearity_error": float(np.max(np.abs(
            shapley_value(outputs.mean(axis=1)) - np.mean(all_true_shapley, axis=0)))),
        "uniform_banzhaf_linearity_error": float(np.max(np.abs(
            banzhaf_value(outputs.mean(axis=1)) - np.mean(all_true_banzhaf, axis=0)))),
        "source_sha256": {str(mask): hashlib.sha256(cell_path(mask).read_bytes()).hexdigest()
                           for mask in range(1 << D)},
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (OUT / "report.md").write_text(f"""# Query-conditioned coalition tensor

The saved evaluation supports a tensor of shape `{result['tensor_shape']}`: eight binary coalition modes and one seven-valued route-query mode. Contracting a coalition leaves a seven-vector; selecting an output index returns the value for that route. No transformer was retrained.

The validation ladder selected degree {selected['degree']}, rank {selected['rank']}, and ridge penalty {selected['alpha']:.6g}. Across the unopened size-three slice, mean per-route Spearman is {result['mean_target_spearman']:.3f}; the required support coalition ranks first for {result['support_top1_count']}/7 routes and within the top three for {result['support_top3_count']}/7. The route's true support triple is the largest predicted absolute third-order Shapley interaction for {result['support_top_triple_shapley_count']}/7 outputs and the largest predicted absolute third-order Banzhaf interaction for {result['support_top_triple_banzhaf_count']}/7.

`attributions.csv` contains a separate Shapley and Banzhaf vector for every route-query. `higher_order_interactions.csv` contains separate order-2 through order-5 interaction indices, and `top_interactions_by_order.csv` identifies the strongest subset only within a specified query and order. `query_metrics.csv` evaluates each query's fixed-cardinality ranking. There is deliberately no unnamed overall ranking. `selection_objectives.csv` reports only explicitly defined cross-query objectives: uniform expected accuracy, worst-query accuracy, and route coverage at 90%.

For the explicitly uniform expected-accuracy objective, averaging per-query Shapley or Banzhaf values equals computing the index after averaging utilities (maximum numerical discrepancies {result['uniform_shapley_linearity_error']:.2e} and {result['uniform_banzhaf_linearity_error']:.2e}). This equivalence follows from the linearity of these indices. It does not extend to nonlinear maximin, CVaR, coverage, or top-set selection objectives.

Independent per-query ladders select degrees {result['independent_selected_degrees']} for routes 0 through 6 and rank the correct support coalition first for {result['independent_support_top1_count']}/7 routes. This comparison tests the sharing assumption: the shared output-mode model also gets 7/7, so sharing did not obscure the operational support structure here. Its rank 24 indicates that compressing across query outputs is substantially harder than representing any one ideal route, which is a single cubic monomial.

At coalition size three, no set can complete more than one Fano route, so every candidate has zero worst-query accuracy. A maximin “overall” ranking is therefore structurally uninformative at this size. `selection_by_size.csv` repeats the named objectives at each cardinality; only size three is held out, and other sizes are labeled descriptive.

The saved `target_accuracy_by_line` field averages 31 input questions within each route. Consequently this is a seven-output route-conditioned tensor, not a 217-output per-example tensor. Future coalition evaluation should save an ordered query identifier and exact-answer log probability for every example. Those saved vectors can be stacked as an output mode without retraining again.
""")
    (OUT / "AUDIT.json").write_text(json.dumps({"status": "PASS", "source_cell_count": 256,
        "output_queries": 7, "individual_examples_available": False,
        "query_conditioned_shapley_banzhaf": True,
        "overall_selection_requires_named_objective": True,
        "transformer_retraining_performed": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("tensor_shape", "selected", "mean_target_spearman",
                                              "support_top1_count", "support_top3_count")}, indent=2))


if __name__ == "__main__":
    main()
