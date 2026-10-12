#!/usr/bin/env python3
"""Multi-seed version of the the development run size-3 analysis.

Same metrics as `analyze_query_paths_d12.py` (top-1 completeness, regret, selection AP,
exact third discrete derivative at the empty background, Shapley/Banzhaf group selection),
computed per learner seed over the full budget ladder, with the added baselines.
`average_precision`, `delta3` and `shapley_banzhaf` are imported from the original module
so the metric definitions cannot drift.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.reasoning.common.transformer_query_paths_d12 import D, Q, completed_queries
from experiments.reasoning.common.analyze_query_paths_d12 import (
    average_precision, delta3, shapley_banzhaf,
)

BUDGETS = (32, 48, 64, 96, 128, 160)


SPLITS = ("target", "interaction_audit", "gate", "observed", "loo")
_CUBES: dict = {}


def _cube_cell(seed: int, mask: int) -> dict | None:
    """Outcome of one coalition from the compact export `cubes/d12_cells_seed<S>.csv`.

    The repository ships the retraining outcomes as these CSV cubes (written by
    `export_cubes.py`) instead of one JSON file per coalition.  Splits are searched
    in the same priority order as the per-coalition files.
    """
    if seed not in _CUBES:
        table: dict = {}
        path = Path(__file__).resolve().parent.parent / "cubes" / f"d12_cells_seed{seed}.csv"
        if path.exists():
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["split"] in SPLITS:
                        cell = table.setdefault((row["split"], int(row["mask"])), [None] * Q)
                        cell[int(row["query"])] = float(row["log_probability"])
        _CUBES[seed] = table
    for split in SPLITS:
        values = _CUBES[seed].get((split, mask))
        if values is not None:
            return {"target_log_probability_by_query": values}
    return None


def load_cell(roots: list[Path], seed: int, mask: int) -> dict:
    for root in roots:
        for split in SPLITS:
            path = root / split / f"seed{seed}" / f"mask_{mask}" / f"seed{seed}_mask{mask}.json"
            if path.exists():
                return json.loads(path.read_text())
    cell = _cube_cell(seed, mask)
    if cell is not None:
        return cell
    raise FileNotFoundError(f"seed {seed} mask {mask}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--fallback-root", type=Path,
                        help="read-only the development run root; used only for seed 31's size-3 and pair cells")
    args = parser.parse_args()

    target_masks = np.asarray([m for m in range(1 << D) if m.bit_count() == 3], dtype=int)
    face_masks = [0] + [m for m in range(1 << D) if m.bit_count() in (1, 2)] + target_masks.tolist()
    query_rows, summary_rows, interaction_rows, attribution_rows = [], [], [], []
    for seed in args.seeds:
        actual = np.full((1 << D, Q), np.nan)
        try:
            for mask in face_masks:
                roots = [args.root] + ([args.fallback_root] if args.fallback_root and seed == 31 else [])
                actual[mask] = np.asarray(load_cell(roots, seed, mask)["target_log_probability_by_query"],
                                          dtype=float)
        except FileNotFoundError as error:
            print(f"skipping seed {seed}: {error}")
            continue
        for budget in BUDGETS:
            frozen_path = args.root / "fits" / f"seed{seed}" / f"budget_{budget}" / "frozen_result.json"
            if not frozen_path.exists():
                continue
            frozen = json.loads(frozen_path.read_text())
            assert frozen["status"] == "FROZEN_BEFORE_TARGET" and not frozen["target_outcomes_loaded"]
            archive = np.load(args.root / "fits" / f"seed{seed}" / f"budget_{budget}" / "frozen_predictions.npz")
            for method in sorted(key for key in archive.files if key != "masks"):
                prediction = archive[method]
                method_query_rows, method_interactions = [], []
                for query in range(Q):
                    y = actual[target_masks, query]
                    p = prediction[target_masks, query]
                    complete = np.asarray([completed_queries(int(mask))[query] for mask in target_masks])
                    order = np.lexsort((target_masks, -p))
                    top1, top8 = int(target_masks[order[0]]), target_masks[order[:8]]
                    oracle = float(np.max(y))
                    row = {"seed": seed, "new_budget": budget, "total_cost": budget + 19,
                           "method": method, "query": query, "top1_mask": top1,
                           "top1_actual_utility": float(actual[top1, query]), "oracle_utility": oracle,
                           "top1_regret": oracle - float(actual[top1, query]),
                           "top1_complete": bool(completed_queries(top1)[query]),
                           "top8_complete_fraction": float(np.mean([completed_queries(int(m))[query] for m in top8])),
                           "top8_actual_utility": float(np.mean(actual[top8, query])),
                           "selection_ap": average_precision(complete, p),
                           "target_spearman": float(spearmanr(p, y).statistic)}
                    query_rows.append(row); method_query_rows.append(row)
                    true_delta = np.asarray([delta3(actual[:, query], int(mask)) for mask in target_masks])
                    pred_delta = np.asarray([delta3(prediction[:, query], int(mask)) for mask in target_masks])
                    interaction_order = np.lexsort((target_masks, -np.abs(pred_delta)))
                    interaction_top1 = int(target_masks[interaction_order[0]])
                    delta_rho = (float(spearmanr(pred_delta, true_delta).statistic)
                                 if np.std(pred_delta) > 1e-12 else 0.0)
                    entry = {"seed": seed, "new_budget": budget, "total_cost": budget + 19,
                             "method": method, "query": query,
                             "delta3_mae": float(np.mean(np.abs(pred_delta - true_delta))),
                             "delta3_spearman": delta_rho,
                             "complete_interaction_ap": average_precision(complete, np.abs(pred_delta)),
                             "top1_interaction_mask": interaction_top1,
                             "top1_interaction_complete": bool(completed_queries(interaction_top1)[query]),
                             "true_complete_delta3_mean": float(np.mean(true_delta[complete])),
                             "predicted_complete_delta3_mean": float(np.mean(pred_delta[complete]))}
                    interaction_rows.append(entry); method_interactions.append(entry)
                    shapley, banzhaf = shapley_banzhaf(prediction[:, query])
                    for criterion, scores in (("shapley", shapley), ("banzhaf", banzhaf)):
                        selected_groups = np.lexsort((np.arange(D), -scores))[:3]
                        selected_mask = int(sum(1 << int(group) for group in selected_groups))
                        attribution_rows.append({"seed": seed, "new_budget": budget,
                            "total_cost": budget + 19, "method": method, "query": query,
                            "criterion": criterion, "selected_mask": selected_mask,
                            "selected_groups": " ".join(map(str, selected_groups.tolist())),
                            "actual_utility": float(actual[selected_mask, query]),
                            "complete": bool(completed_queries(selected_mask)[query])})
                summary_rows.append({"seed": seed, "new_budget": budget, "total_cost": budget + 19,
                    "method": method,
                    "top1_complete_rate": float(np.mean([r["top1_complete"] for r in method_query_rows])),
                    "top1_actual_utility": float(np.mean([r["top1_actual_utility"] for r in method_query_rows])),
                    "top1_regret": float(np.mean([r["top1_regret"] for r in method_query_rows])),
                    "top8_complete_fraction": float(np.mean([r["top8_complete_fraction"] for r in method_query_rows])),
                    "selection_ap": float(np.mean([r["selection_ap"] for r in method_query_rows])),
                    "target_spearman": float(np.nanmean([r["target_spearman"] for r in method_query_rows])),
                    "delta3_mae": float(np.mean([r["delta3_mae"] for r in method_interactions])),
                    "delta3_spearman": float(np.mean([r["delta3_spearman"] for r in method_interactions])),
                    "complete_interaction_ap": float(np.mean([r["complete_interaction_ap"] for r in method_interactions])),
                    "interaction_top1_complete_rate": float(np.mean([r["top1_interaction_complete"] for r in method_interactions]))})
    if not summary_rows:
        raise SystemExit("no size-3 outcomes available yet")
    write_csv(args.out / "size3_query_selection.csv", query_rows)
    write_csv(args.out / "size3_method_summary.csv", summary_rows)
    write_csv(args.out / "size3_third_order_interactions.csv", interaction_rows)
    write_csv(args.out / "size3_attribution_selection.csv", attribution_rows)
    print(json.dumps({"seeds": sorted({r["seed"] for r in summary_rows}),
                      "rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
