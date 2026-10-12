#!/usr/bin/env python3
"""Print the Qwen-probe panel of Table 17 and Table 19 from the saved attribution errors.

    python scripts/summarize_probe.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results" / "qwen_probe"
FAMILIES = {"linear": "Linear", "rbf_kernel": "RBF", "orderwise_cp": "Coefficient CP", "tensor_train": "TT"}
QUERIES = {"shapley": "Shapley", "banzhaf": "Banzhaf", "loo": "Leave-one-out",
           "pair_shapley_interaction": "Pairwise interaction"}


def table17_probe() -> pd.DataFrame:
    """Attribution RMSE against the complete game: mean and sample SD over seeds 31, 43, 59."""
    rows = pd.read_csv(RESULTS / "attribution_errors.csv")
    out = []
    for query, label in QUERIES.items():
        for family in ("linear", "rbf_kernel", "tensor_train"):
            values = rows[(rows["query"] == query) & (rows.family == family)].rmse
            out.append(dict(attribution=label, surrogate=FAMILIES[family], seeds=len(values),
                            mean=values.mean(), sd=values.std(ddof=1)))
    return pd.DataFrame(out)


def table19() -> pd.DataFrame:
    """Largest gap between the model-specific attribution routines and enumeration of the same fitted model."""
    rows = pd.read_csv(RESULTS / "attribution_errors.csv")
    out = []
    for family in ("linear", "orderwise_cp", "tensor_train"):
        gaps = rows[rows.family == family]
        pair = gaps["query"] == "pair_shapley_interaction"
        out.append(dict(surrogate=FAMILIES[family],
                        singleton=gaps[~pair].native_vs_enumerated_max_abs.max(),
                        pairwise=gaps[pair].native_vs_enumerated_max_abs.max()))
    return pd.DataFrame(out)


def main() -> None:
    print("Table 17, Qwen-probe panel: attribution RMSE, mean ± sample SD over three seeds")
    for _, r in table17_probe().iterrows():
        print(f"  {r['attribution']:<22}{r['surrogate']:<8}{r['mean']:.6f} ± {r['sd']:.6f}")
    print("\nTable 19: maximum absolute discrepancy, attribution routine vs enumeration")
    for _, r in table19().iterrows():
        print(f"  {r['surrogate']:<16}{r['singleton']:.2e}   {r['pairwise']:.2e}")


if __name__ == "__main__":
    main()
