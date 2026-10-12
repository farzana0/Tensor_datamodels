#!/usr/bin/env python3
"""Analyze frozen D=12 predictions after target and pair-audit retraining."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.reasoning.common.transformer_query_paths_d12 import D, Q, completed_queries

ROOT = Path("results/original_records/d12")
BUDGETS = (32, 48, 64, 96, 128, 160)


def load_cell(mask: int) -> dict:
    for split in ("target", "interaction_audit", "gate", "observed"):
        path = ROOT / split / "seed31" / f"mask_{mask}" / f"seed31_mask{mask}.json"
        if path.exists(): return json.loads(path.read_text())
    raise FileNotFoundError(mask)


def average_precision(labels: np.ndarray, score: np.ndarray) -> float:
    order = np.lexsort((np.arange(len(score)), -score))
    ranked = labels[order].astype(float)
    if not ranked.sum(): return float("nan")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / ranked.sum())


def delta3(values: np.ndarray, mask: int) -> float:
    members = [i for i in range(D) if mask & (1 << i)]
    assert len(members) == 3
    singles = [1 << i for i in members]
    pairs = [mask ^ single for single in singles]
    return float(values[mask] - sum(values[p] for p in pairs)
                 + sum(values[s] for s in singles) - values[0])


def shapley_banzhaf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shapley, banzhaf = np.zeros(D), np.zeros(D)
    all_masks = np.arange(1 << D)
    for group in range(D):
        absent = all_masks[(all_masks & (1 << group)) == 0]
        marginal = values[absent | (1 << group)] - values[absent]
        banzhaf[group] = marginal.mean()
        shapley[group] = np.mean([marginal[[int(mask).bit_count() == size for mask in absent]].mean()
                                          for size in range(D)])
    return shapley, banzhaf


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    target_masks = np.asarray([m for m in range(1 << D) if m.bit_count() == 3], dtype=int)
    face_masks = [0] + [m for m in range(1 << D) if m.bit_count() in (1, 2)] + target_masks.tolist()
    actual = np.full((1 << D, Q), np.nan)
    for mask in face_masks:
        actual[mask] = np.asarray(load_cell(mask)["target_log_probability_by_query"], dtype=float)
    query_rows, summary_rows, interaction_rows, attribution_rows = [], [], [], []
    for budget in BUDGETS:
        frozen = json.loads((ROOT / "fits" / f"budget_{budget}" / "frozen_result.json").read_text())
        assert frozen["status"] == "FROZEN_BEFORE_TARGET" and not frozen["target_outcomes_loaded"]
        archive = np.load(ROOT / "fits" / f"budget_{budget}" / "frozen_predictions.npz")
        for method in sorted(key for key in archive.files if key != "masks"):
            prediction = archive[method]
            method_query_rows = []
            method_interactions = []
            for query in range(Q):
                y = actual[target_masks, query]
                p = prediction[target_masks, query]
                complete = np.asarray([completed_queries(int(mask))[query] for mask in target_masks])
                order = np.lexsort((target_masks, -p))
                top1, top8 = int(target_masks[order[0]]), target_masks[order[:8]]
                oracle = float(np.max(y))
                rho = float(spearmanr(p, y).statistic)
                row = {"new_budget": budget, "total_cost": budget + 19, "method": method, "query": query,
                       "top1_mask": top1, "top1_actual_utility": float(actual[top1, query]),
                       "oracle_utility": oracle, "top1_regret": oracle - float(actual[top1, query]),
                       "top1_complete": bool(completed_queries(top1)[query]),
                       "top8_complete_fraction": float(np.mean([completed_queries(int(m))[query] for m in top8])),
                       "top8_actual_utility": float(np.mean(actual[top8, query])),
                       "selection_ap": average_precision(complete, p), "target_spearman": rho}
                query_rows.append(row); method_query_rows.append(row)
                true_delta = np.asarray([delta3(actual[:, query], int(mask)) for mask in target_masks])
                pred_delta = np.asarray([delta3(prediction[:, query], int(mask)) for mask in target_masks])
                interaction_order = np.lexsort((target_masks, -np.abs(pred_delta)))
                interaction_top1 = int(target_masks[interaction_order[0]])
                interaction_top8 = target_masks[interaction_order[:8]]
                delta_rho = float(spearmanr(pred_delta, true_delta).statistic) if np.std(pred_delta) > 1e-12 else 0.0
                ir = {"new_budget": budget, "total_cost": budget + 19, "method": method, "query": query,
                      "delta3_mae": float(np.mean(np.abs(pred_delta - true_delta))),
                      "delta3_spearman": delta_rho,
                      "complete_interaction_ap": average_precision(complete, np.abs(pred_delta)),
                      "top1_interaction_mask": interaction_top1,
                      "top1_interaction_actual_utility": float(actual[interaction_top1, query]),
                      "top1_interaction_complete": bool(completed_queries(interaction_top1)[query]),
                      "top8_interaction_actual_utility": float(np.mean(actual[interaction_top8, query])),
                      "top8_interaction_complete_fraction": float(np.mean(complete[interaction_order[:8]])),
                      "true_complete_delta3_mean": float(np.mean(true_delta[complete])),
                      "predicted_complete_delta3_mean": float(np.mean(pred_delta[complete]))}
                interaction_rows.append(ir); method_interactions.append(ir)
                shapley, banzhaf = shapley_banzhaf(prediction[:, query])
                for criterion, scores in (("shapley", shapley), ("banzhaf", banzhaf)):
                    selected_groups = np.lexsort((np.arange(D), -scores))[:3]
                    selected_mask = int(sum(1 << int(group) for group in selected_groups))
                    attribution_rows.append({"new_budget": budget, "total_cost": budget + 19,
                        "method": method, "query": query, "criterion": criterion,
                        "selected_mask": selected_mask,
                        "selected_groups": " ".join(map(str, selected_groups.tolist())),
                        "actual_utility": float(actual[selected_mask, query]),
                        "complete": bool(completed_queries(selected_mask)[query])})
            summary_rows.append({"new_budget": budget, "total_cost": budget + 19, "method": method,
                "top1_complete_rate": float(np.mean([r["top1_complete"] for r in method_query_rows])),
                "top1_actual_utility": float(np.mean([r["top1_actual_utility"] for r in method_query_rows])),
                "top1_regret": float(np.mean([r["top1_regret"] for r in method_query_rows])),
                "top8_complete_fraction": float(np.mean([r["top8_complete_fraction"] for r in method_query_rows])),
                "selection_ap": float(np.mean([r["selection_ap"] for r in method_query_rows])),
                "target_spearman": float(np.nanmean([r["target_spearman"] for r in method_query_rows])),
                "delta3_mae": float(np.mean([r["delta3_mae"] for r in method_interactions])),
                "delta3_spearman": float(np.mean([r["delta3_spearman"] for r in method_interactions])),
                "complete_interaction_ap": float(np.mean([r["complete_interaction_ap"] for r in method_interactions])),
                "interaction_top1_complete_rate": float(np.mean([r["top1_interaction_complete"] for r in method_interactions])),
                "interaction_top1_actual_utility": float(np.mean([r["top1_interaction_actual_utility"] for r in method_interactions])),
                "top8_interaction_complete_fraction": float(np.mean([r["top8_interaction_complete_fraction"] for r in method_interactions]))})
    write_csv(ROOT / "analysis/query_selection.csv", query_rows)
    write_csv(ROOT / "analysis/method_summary.csv", summary_rows)
    write_csv(ROOT / "analysis/third_order_interactions.csv", interaction_rows)
    write_csv(ROOT / "analysis/attribution_selection.csv", attribution_rows)
    best = sorted(summary_rows, key=lambda r: (-r["top1_complete_rate"], r["top1_regret"], -r["complete_interaction_ap"]))[0]
    result = {"learner": "8-layer width-256 causal-GPT transformer retrained from scratch per coalition",
              "game": "query-specific three-hop compositional relation dataset, d=12",
              "target_slice": "all 220 size-three coalitions", "query_tensors": Q,
              "best_operational_row": best, "budgets": list(BUDGETS),
              "all_target_outcomes_from_retraining": True,
              "third_order_definition": "exact discrete third derivative at empty background using retrained empty, singleton, pair, and triple coalitions"}
    (ROOT / "analysis/result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
