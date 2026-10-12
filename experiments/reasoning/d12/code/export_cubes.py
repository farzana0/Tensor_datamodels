#!/usr/bin/env python3
"""Export every retraining cell of this workstream to compact per-seed CSV tables.

The raw per-cell JSON records (per-input log-probabilities, training history) are not
part of the export. The export keeps what every analysis here
uses: per-query log-probability and accuracy, plus provenance (split, seeds, GPU, job).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

Q = 10
SPLITS = ("gate", "observed", "loo", "target", "interaction_audit",
          "size4_eval", "size5_eval", "size6_eval")


def rows_for(path: Path, split: str, arm: str) -> list[dict]:
    record = json.loads(path.read_text())
    base = {"split": split, "arm": arm, "mask": record["mask"], "cardinality": record["cardinality"],
            "init_seed": record.get("init_seed", record["seed"]),
            "data_seed": record.get("data_seed", record["seed"]),
            "gpu_name": record.get("gpu_name", ""), "slurm_job_id": record.get("slurm_job_id", "")}
    return [{**base, "query": q,
             "log_probability": repr(float(record["target_log_probability_by_query"][q])),
             "accuracy": repr(float(record["target_accuracy_by_query"][q])),
             "complete": bool(record["completed_queries"][q])} for q in range(Q)]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--here", type=Path, required=True)
    args = parser.parse_args()
    results, out = args.here / "results", args.here / "cubes"
    summary = {}
    for seed in (31, 43, 59):
        rows = []
        for split in SPLITS:
            for path in sorted((results / split / f"seed{seed}").glob("mask_*/*.json"),
                               key=lambda p: int(p.parent.name.split("_")[1])):
                rows += rows_for(path, split, "")
        if rows:
            write(out / f"d12_cells_seed{seed}.csv", rows)
            summary[f"seed{seed}"] = len(rows) // Q
    ceiling = []
    for seed in (31, 43):
        for arm_dir in sorted((results / "ceiling" / f"seed{seed}").iterdir()):
            for path in sorted(arm_dir.glob("mask_*/*.json"), key=lambda p: int(p.parent.name.split("_")[1])):
                ceiling += rows_for(path, "ceiling", arm_dir.name)
    if ceiling:
        write(out / "d12_ceiling_cells.csv", ceiling)
        summary["ceiling"] = len(ceiling) // Q
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
