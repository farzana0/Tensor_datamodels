#!/usr/bin/env python3
"""Print Table 16 and the UCI HAR panel of Table 17 from the saved result tables.

Nothing is fitted. Both tables are reductions of per-candidate / per-split
records saved when the experiments were run (``results/appendix_tables``).

    python scripts/summarize_appendix_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results" / "appendix_tables"

TABLE16_ROWS = [  # (label in the paper, family, ordering)
    ("Linear", "linear", "permutation_invariant"),
    ("RBF", "rbf_kernel", "permutation_invariant"),
    ("Coefficient CP", "orderwise_cp", "permutation_invariant"),
    ("TT, random", "tensor_train", "random"),
    ("TT, semantic", "tensor_train", "semantic"),
    ("Degree-graded TT, random", "degree_graded_tt", "random"),
    ("Degree-graded TT, semantic", "degree_graded_tt", "semantic"),
]
TABLE17_METHODS = {"additive": "Linear", "rbf": "RBF", "tt": "TT"}
TABLE17_METRICS = {"shapley_rmse": "Shapley", "banzhaf_rmse": "Banzhaf",
                   "pairwise_interaction_rmse": "Pairwise interaction"}


def table16() -> pd.DataFrame:
    """Mean and sample SD over seeds of the test RMSE of the validation-selected candidate."""
    rows = pd.read_csv(RESULTS / "table16_d500_candidates.csv")
    selected = rows[rows.selected == 1]
    out = []
    for label, family, ordering in TABLE16_ROWS:
        for task in ("coin_flip", "gsm8k"):
            for n_fit in (400, 800):
                values = selected[(selected.family == family) & (selected.ordering == ordering)
                                  & (selected.task == task) & (selected.n_fit == n_fit)].test_rmse
                out.append(dict(surrogate=label, task=task, n_fit=n_fit, seeds=len(values),
                                mean=values.mean(), sd=values.std(ddof=1)))
    return pd.DataFrame(out)


def table17_uci_har(budget: int = 320) -> pd.DataFrame:
    """Mean and sample SD over the ten coalition splits of each attribution RMSE."""
    rows = pd.read_csv(RESULTS / "table17_uci_har_heldout.csv")
    rows = rows[rows.requested_budget == budget]
    out = []
    for metric, label in TABLE17_METRICS.items():
        for method, name in TABLE17_METHODS.items():
            values = rows[rows.method == method][metric]
            out.append(dict(attribution=label, surrogate=name, splits=len(values),
                            mean=values.mean(), sd=values.std(ddof=1)))
    return pd.DataFrame(out)


def main() -> None:
    print("Table 16: test RMSE, mean ± sample SD over seeds 31 and 43")
    wide = table16()
    for label, *_ in TABLE16_ROWS:
        cells = [f"{r['mean']:.4f} ± {r['sd']:.4f}" for _, r in wide[wide.surrogate == label].iterrows()]
        print(f"  {label:<28}" + "   ".join(cells))
    print("  columns: coin flip n_fit=400, 800; GSM8K n_fit=400, 800")
    print("\nTable 17, UCI HAR panel: attribution RMSE, mean ± sample SD over ten splits (320 fitting, 32 validation)")
    for _, r in table17_uci_har().iterrows():
        print(f"  {r['attribution']:<22}{r['surrogate']:<8}{r['mean']:.6f} ± {r['sd']:.6f}")


if __name__ == "__main__":
    main()
