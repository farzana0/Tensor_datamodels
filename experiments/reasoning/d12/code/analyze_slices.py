#!/usr/bin/env python3
"""Evaluate frozen fixed-size nominations against actual retraining, across seeds.

Generalises `analyze_query_paths_d12_size5.py` from the single post-hoc size-5 slice of
the development run to every pre-registered slice (3, 4, 5, 6) and every learner seed, and runs
the pre-registered comparisons C1-C5 on the confirmatory seeds only, pairing on
(query, seed).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.reasoning.common.transformer_query_paths_d12 import Q, completed_queries

BOOTSTRAP = 20000
BOOTSTRAP_SEED = 20260912
COMPARISONS = (("C1", "tt", "linear", "superiority"),
               ("C2", "tt", "random", "superiority"),
               ("C3", "tt", "influence_loo", "superiority"),
               ("C4", "tt", "degree3_ridge", "equivalence"),
               ("C5", "tt", "degree2_ridge", "descriptive"))


FALLBACK: Path | None = None


def cell(root: Path, seed: int, mask: int, size: int) -> dict | None:
    roots = [root] + ([FALLBACK] if FALLBACK is not None and seed == 31 and size == 3 else [])
    for base in roots:
        for split in (f"size{size}_eval", "target", "observed", "gate", "loo"):
            folder = base / split / f"seed{seed}" / f"mask_{mask}"
            if folder.exists():
                files = sorted(folder.glob(f"seed{seed}_mask{mask}.json"))
                if files:
                    return json.loads(files[0].read_text())
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(difference: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.asarray([difference[rng.integers(0, len(difference), len(difference))].mean()
                        for _ in range(BOOTSTRAP)])
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocols", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ceiling", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--confirmatory-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--slices", type=int, nargs="+", default=(3, 4, 5, 6))
    parser.add_argument("--primary-slice", type=int, default=4)
    parser.add_argument("--fallback-root", type=Path,
                        help="read-only the development run root; only seed 31's size-3 cells are read from it")
    args = parser.parse_args()
    global FALLBACK
    FALLBACK = args.fallback_root

    margin = None
    if args.ceiling and args.ceiling.exists():
        margin = json.loads(args.ceiling.read_text())["equivalence_margin_top3_mean_utility"]

    rows, summary, units, missing = [], [], {}, []
    for seed in args.seeds:
        protocol_path = args.protocols / f"SLICE_NOMINATION_PROTOCOL_seed{seed}.json"
        if not protocol_path.exists():
            continue
        protocol = json.loads(protocol_path.read_text())
        assert protocol["status"] == "FROZEN_BEFORE_SLICE_OUTCOMES"
        count = int(protocol["executed_per_query_count"])
        frozen = json.loads((args.root / "fits" / f"seed{seed}" / f"budget_{protocol['budget']}"
                             / "frozen_result.json").read_text())
        for size in args.slices:
            per_method = (protocol["slices"][str(size)]["nominations"]
                          if str(size) in protocol["slices"] else frozen["nominations"][str(size)])
            for method, queries in per_method.items():
                for query, masks in enumerate(queries):
                    masks = masks[:count]
                    utilities, complete = [], []
                    for rank, mask in enumerate(masks, 1):
                        record = cell(args.root, seed, mask, size)
                        if record is None:
                            missing.append({"seed": seed, "slice": size, "method": method,
                                            "query": query, "mask": mask})
                            continue
                        utility = float(record["target_log_probability_by_query"][query])
                        is_complete = bool(completed_queries(mask)[query])
                        utilities.append(utility); complete.append(is_complete)
                        rows.append({"seed": seed, "slice": size, "method": method, "query": query,
                                     "rank": rank, "mask": mask,
                                     "selected_groups": " ".join(str(i) for i in range(12) if mask & (1 << i)),
                                     "actual_retrained_utility": utility, "complete": is_complete,
                                     "gpu_name": record.get("gpu_name", "")})
                    if utilities:
                        units[(seed, size, method, query)] = {
                            "top3_mean_utility": float(np.mean(utilities)),
                            "top1_utility": utilities[0], "top1_complete": complete[0],
                            "top3_complete_fraction": float(np.mean(complete)),
                            "nominees_found": len(utilities)}
    if not rows:
        raise SystemExit("no slice outcomes found yet")
    write_csv(args.out / "slice_nominations.csv", rows)

    for seed in args.seeds:
        for size in args.slices:
            methods = sorted({m for (s, k, m, q) in units if s == seed and k == size})
            for method in methods:
                values = [units[(seed, size, method, q)] for q in range(Q)
                          if (seed, size, method, q) in units]
                if not values:
                    continue
                summary.append({"seed": seed, "slice": size, "method": method,
                    "queries": len(values),
                    "top1_complete_rate": float(np.mean([v["top1_complete"] for v in values])),
                    "top3_complete_fraction": float(np.mean([v["top3_complete_fraction"] for v in values])),
                    "top1_actual_utility": float(np.mean([v["top1_utility"] for v in values])),
                    "top3_actual_utility": float(np.mean([v["top3_mean_utility"] for v in values])),
                    "nominees_found": int(sum(v["nominees_found"] for v in values))})
    write_csv(args.out / "slice_summary.csv", summary)

    confirmatory = {"primary_slice": args.primary_slice, "confirmatory_seeds": args.confirmatory_seeds,
                    "equivalence_margin": margin, "bootstrap_draws": BOOTSTRAP,
                    "unit": "(query, learner seed)", "comparisons": {}, "by_slice": {}}
    for size in args.slices:
        slice_result = {}
        for name, left, right, kind in COMPARISONS:
            pairs = [(units[(s, size, left, q)]["top3_mean_utility"],
                      units[(s, size, right, q)]["top3_mean_utility"])
                     for s in args.confirmatory_seeds for q in range(Q)
                     if (s, size, left, q) in units and (s, size, right, q) in units]
            if not pairs:
                continue
            difference = np.asarray([a - b for a, b in pairs])
            low, high = bootstrap_ci(difference)
            entry = {"comparison": f"{left} - {right}", "kind": kind, "n_units": len(difference),
                     "mean_difference": float(difference.mean()),
                     "unit_win_rate": float(np.mean(difference > 0)),
                     "bootstrap_95_interval": [low, high]}
            if kind == "superiority":
                entry["status"] = "SUPPORTED" if low > 0 else ("REFUTED" if high < 0 else "INCONCLUSIVE")
            elif kind == "equivalence" and margin is not None:
                entry["margin"] = margin
                entry["status"] = ("INDISTINGUISHABLE" if low > -margin and high < margin else
                                   "SUPPORTED" if low > margin else
                                   "REFUTED" if high < -margin else "INCONCLUSIVE")
            slice_result[name] = entry
        # C5b: best nonadditive vs best additive, chosen by top-3 utility on this slice
        nonadditive = ("tt", "cp", "ht", "degree_ladder_tt", "linear_plus_tt", "degree2_ridge",
                       "degree3_ridge", "rbf_kernel_ridge", "sparse_degree3_omp")
        additive = ("linear", "cardinality", "influence_loo", "random")
        def family_best(family: tuple[str, ...]) -> str | None:
            scored = [(np.mean([units[(s, size, m, q)]["top3_mean_utility"]
                                for s in args.confirmatory_seeds for q in range(Q)
                                if (s, size, m, q) in units]), m)
                      for m in family
                      if any((s, size, m, q) in units for s in args.confirmatory_seeds for q in range(Q))]
            return max(scored)[1] if scored else None
        best_non, best_add = family_best(nonadditive), family_best(additive)
        if best_non and best_add:
            pairs = [(units[(s, size, best_non, q)]["top3_mean_utility"],
                      units[(s, size, best_add, q)]["top3_mean_utility"])
                     for s in args.confirmatory_seeds for q in range(Q)
                     if (s, size, best_non, q) in units and (s, size, best_add, q) in units]
            difference = np.asarray([a - b for a, b in pairs])
            low, high = bootstrap_ci(difference)
            slice_result["C5_family"] = {"comparison": f"{best_non} - {best_add}",
                "kind": "superiority", "n_units": len(difference),
                "mean_difference": float(difference.mean()),
                "unit_win_rate": float(np.mean(difference > 0)),
                "bootstrap_95_interval": [low, high],
                "status": "SUPPORTED" if low > 0 else ("REFUTED" if high < 0 else "INCONCLUSIVE")}
        confirmatory["by_slice"][str(size)] = slice_result
    confirmatory["comparisons"] = confirmatory["by_slice"].get(str(args.primary_slice), {})
    confirmatory["missing_nominee_cells"] = missing
    (args.out / "confirmatory.json").write_text(
        json.dumps(confirmatory, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"primary_slice": args.primary_slice,
                      "comparisons": confirmatory["comparisons"],
                      "missing": len(missing)}, indent=2))


if __name__ == "__main__":
    main()
