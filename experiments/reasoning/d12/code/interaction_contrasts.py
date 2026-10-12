#!/usr/bin/env python3
"""Paired bootstrap for the secondary third-order interaction metrics.

Units are (query, learner seed) over the confirmatory seeds, as for the primary slice
comparisons. Contrasts are TT against every other method, at every budget, for
complete-path interaction AP and for top-1 interaction completeness.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BOOTSTRAP = 20000
METRICS = ("complete_interaction_ap", "top1_interaction_complete")
BASELINES = ("degree3_ridge", "degree_ladder_tt", "rbf_kernel_ridge", "sparse_degree3_omp",
             "degree2_ridge", "cp", "linear_plus_tt", "linear", "random")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--confirmatory-seeds", type=int, nargs="+", default=(43, 59))
    args = parser.parse_args()

    rows = list(csv.DictReader((args.analysis / "size3_third_order_interactions.csv").open()))
    value = {}
    for row in rows:
        key = (int(row["seed"]), int(row["new_budget"]), row["method"], int(row["query"]))
        value[key] = {"complete_interaction_ap": float(row["complete_interaction_ap"]),
                      "top1_interaction_complete": float(row["top1_interaction_complete"] == "True")}
    budgets = sorted({k[1] for k in value})
    out, table = {}, []
    for budget in budgets:
        for baseline in BASELINES:
            for metric in METRICS:
                pairs = [(value[(s, budget, "tt", q)][metric], value[(s, budget, baseline, q)][metric])
                         for s in args.confirmatory_seeds for q in range(10)
                         if (s, budget, "tt", q) in value and (s, budget, baseline, q) in value]
                if not pairs:
                    continue
                difference = np.asarray([a - b for a, b in pairs])
                rng = np.random.default_rng(20260912)
                draws = np.asarray([difference[rng.integers(0, len(difference), len(difference))].mean()
                                    for _ in range(BOOTSTRAP)])
                low, high = (float(x) for x in np.quantile(draws, [.025, .975]))
                entry = {"budget": budget, "baseline": baseline, "metric": metric,
                         "n_units": len(difference), "tt_mean": float(np.mean([a for a, _ in pairs])),
                         "baseline_mean": float(np.mean([b for _, b in pairs])),
                         "mean_difference": float(difference.mean()), "ci_low": low, "ci_high": high,
                         "status": "TT_BETTER" if low > 0 else ("TT_WORSE" if high < 0 else "INCONCLUSIVE")}
                table.append(entry)
                out[f"{budget}|{baseline}|{metric}"] = entry
    with (args.analysis / "size3_interaction_contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(table)
    print(json.dumps([e for e in table if e["budget"] == 160], indent=1))


if __name__ == "__main__":
    main()
