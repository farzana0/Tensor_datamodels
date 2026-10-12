#!/usr/bin/env python3
"""Retraining-noise ceiling: variance components for the d=12 query-path game.

Separates, at fixed coalition and fixed learner seed, the per-query utility spread caused
by (a) kernel nondeterminism at identical seeds, (b) weight initialisation, (c) training
data order, and (d) the GPU model.  Also reports the ceiling R^2 against the spread of
utilities across coalitions, and fixes the equivalence margin used by the
pre-registered C4 comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

Q = 10
ARMS = ("base", "rerun", "init1", "init2", "data1", "data2", "v100", "a6000")
# distinct stochastic replicates only: `rerun` is bit-identical to `base` on rtx8000 (D3)
RTX_ARMS = ("base", "init1", "init2", "data1", "data2")


def load(root: Path, seed: int, mask: int, arm: str) -> np.ndarray | None:
    folder = root / "ceiling" / f"seed{seed}" / arm / f"mask_{mask}"
    if not folder.exists():
        return None
    files = sorted(folder.glob("*.json"))
    if not files:
        return None
    return np.asarray(json.loads(files[0].read_text())["target_log_probability_by_query"], dtype=float)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(31, 43))
    args = parser.parse_args()

    masks = [int(x) for x in (args.configs / "ceiling_masks.txt").read_text().split()]
    cells, rows = {}, []
    for seed in args.seeds:
        for mask in masks:
            for arm in ARMS:
                value = load(args.root, seed, mask, arm)
                if value is None:
                    continue
                cells[(seed, mask, arm)] = value
                for query in range(Q):
                    rows.append({"seed": seed, "mask": mask, "cardinality": int(mask).bit_count(),
                                 "arm": arm, "query": query, "log_probability": float(value[query])})
    if not rows:
        raise SystemExit("no ceiling cells found")
    write_csv(args.out / "ceiling_cells.csv", rows)

    def spread(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        """SD implied by paired differences, plus the mean signed shift."""
        if not pairs:
            return {"n_pairs": 0}
        delta = np.concatenate([(a - b) for a, b in pairs])
        return {"n_pairs": len(pairs), "n_query_values": int(delta.size),
                "mean_signed_shift": float(delta.mean()),
                "mean_absolute_shift": float(np.abs(delta).mean()),
                "max_absolute_shift": float(np.abs(delta).max()),
                "implied_single_run_sd": float(delta.std(ddof=1) / np.sqrt(2))}

    components = {}
    for name, (a_arm, b_arms) in {
        "nondeterminism_same_seed_same_gpu": ("base", ("rerun",)),
        "initialisation": ("base", ("init1", "init2")),
        "data_order": ("base", ("data1", "data2")),
        "backend_v100_vs_rtx8000": ("base", ("v100",)),
        "backend_a6000_vs_rtx8000": ("base", ("a6000",)),
    }.items():
        pairs = [(cells[(s, m, b)], cells[(s, m, a_arm)])
                 for s in args.seeds for m in masks for b in b_arms
                 if (s, m, b) in cells and (s, m, a_arm) in cells]
        components[name] = spread(pairs)

    # Ceiling R^2: replicate variance against the spread of utilities across coalitions.
    base = np.vstack([cells[(s, m, "base")] for s in args.seeds for m in masks
                      if (s, m, "base") in cells])
    replicate_var = []
    for s in args.seeds:
        for m in masks:
            arm_values = [cells[(s, m, a)] for a in RTX_ARMS if (s, m, a) in cells]
            if len(arm_values) >= 2:
                replicate_var.append(np.var(np.vstack(arm_values), axis=0, ddof=1))
    replicate_var = np.mean(np.vstack(replicate_var), axis=0) if replicate_var else np.full(Q, np.nan)
    across_var = base.var(axis=0, ddof=1) if len(base) > 1 else np.full(Q, np.nan)
    ceiling_r2 = 1 - replicate_var / across_var

    pooled_sd = float(np.sqrt(np.mean(replicate_var)))
    margin = pooled_sd / np.sqrt(3)  # a top-3 mean averages three independent retrainings
    result = {
        "masks": masks, "seeds": list(args.seeds), "arms": list(ARMS),
        "cells_found": len(cells), "cells_expected": len(masks) * len(ARMS) * len(args.seeds),
        "variance_components_log_probability": components,
        "per_query_replicate_sd_rtx8000": np.sqrt(replicate_var).tolist(),
        "per_query_across_coalition_sd": np.sqrt(across_var).tolist(),
        "per_query_ceiling_r2": ceiling_r2.tolist(),
        "mean_ceiling_r2": float(np.nanmean(ceiling_r2)),
        "pooled_replicate_sd_single_cell": pooled_sd,
        "equivalence_margin_top3_mean_utility": float(margin),
        "margin_definition": "one pooled rtx8000 replicate SD of a single retrained cell, "
                             "divided by sqrt(3) because the primary metric averages three nominees",
    }
    (args.out / "ceiling.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("cells_found", "cells_expected",
        "variance_components_log_probability", "mean_ceiling_r2",
        "pooled_replicate_sd_single_cell", "equivalence_margin_top3_mean_utility")}, indent=2))


if __name__ == "__main__":
    main()
