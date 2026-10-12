#!/usr/bin/env python3
"""Question A: datamodels vs direct estimators at equal retraining budget.

One cube per invocation so the sweep can be spread over cores:

    python run_question_a.py --cube khop_d8
    python run_question_a.py --all --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cubes, jobs, protocol, runner  # noqa: E402

OUT = Path(__file__).resolve().parent / "results/question_a"


def run_one(name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    with jobs.claim(path) as partial:
        if partial is None:
            return f"{name}: skipped (done or claimed elsewhere)"
        cube = cubes.load(name)
        start = time.time()
        with open(partial, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=runner.ROW_FIELDS, lineterminator="\n")
            writer.writeheader()
            runner.run_cube(cube, writer, verbose=False)
        meta = {"cube": name, "d": cube.d, "outputs": cube.output_names,
                "budgets": protocol.budgets(cube.d), "repeats": protocol.REPEATS,
                "preregistration_sha256": protocol.preregistration_sha256(),
                "seconds": round(time.time() - start, 1)}
        (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return f"{name}: {meta['seconds']}s -> {path}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    names = cubes.all_names() if args.all else [args.cube]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for message in pool.map(run_one, names):
                print(message, flush=True)
    else:
        for name in names:
            print(run_one(name), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
