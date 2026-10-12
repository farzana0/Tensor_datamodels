"""Fit every surrogate family on one sampling repeat, at every pre-registered budget.

Usage: fit_budgets.py --repeat R          (R = 1..20 sampled budgets; R = 0 full-cube ceiling)

Writes results/fits/repeat_<R>.npz (predicted 256x8 cube per family and budget)
and results/fits/repeat_<R>.json (the selected hyperparameters).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cube  # noqa: E402
import surrogates  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "fits"
BUDGETS = (24, 32, 48, 64, 128)
BASE_SEED = 7000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True)
    arguments = parser.parse_args()
    repeat = arguments.repeat

    logp, _, _ = cube.load_query_cube()
    OUT.mkdir(parents=True, exist_ok=True)

    arrays, choices = {}, {}
    if repeat == 0:
        plan = [(1 << cube.D, np.arange(1 << cube.D))]
    else:
        generator = np.random.default_rng(BASE_SEED + repeat)
        plan = [(budget, np.sort(generator.choice(1 << cube.D, budget, replace=False)))
                for budget in BUDGETS]

    for budget, sample in plan:
        started = time.time()
        predictions, selected = surrogates.fit_all(logp, sample, BASE_SEED + repeat)
        for family, prediction in predictions.items():
            arrays[f"m{budget}__{family}"] = prediction
        arrays[f"m{budget}__sample"] = sample
        choices[str(budget)] = {
            "sample": [int(mask) for mask in sample],
            "selected": selected,
            "seconds": time.time() - started,
        }
        print(f"repeat {repeat} budget {budget}: {time.time() - started:.1f}s", flush=True)

    np.savez_compressed(OUT / f"repeat_{repeat}.npz", **arrays)
    (OUT / f"repeat_{repeat}.json").write_text(
        json.dumps({"repeat": repeat, "budgets": choices,
                    "ladder_containment_error": surrogates.containment_check(logp)},
                   indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
