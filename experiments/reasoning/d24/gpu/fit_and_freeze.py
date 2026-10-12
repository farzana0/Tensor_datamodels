#!/usr/bin/env python3
"""Fit every surrogate family on real d=24 retraining cells and freeze predictions.

Amendment A6: identical to the pilot's `build_models` (same grids, split seed 6001,
same 2,048-mask distillation set as pilot replicate 1), but the response is the
observed per-query log-probability from `gpu/observed/seed<S>/`.

Refuses to run if any evaluation-only cell (evaluation masks that are not gate
masks) already exists for this seed, so predictions are frozen before evaluation
outcomes exist.  Writes `gpu/fits/seed<S>/budget_<m>/frozen_predictions.npz` and
`frozen_result.json` with a freeze sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "pilot"))

import game as G  # noqa: E402
from run_arm import build_models  # noqa: E402

D, Q = 24, 10


def read(name: str) -> list[int]:
    return [int(x) for x in (HERE / "configs" / name).read_text().split()]


def cell(seed: int, mask: int) -> Path:
    return HERE / f"gpu/observed/seed{seed}/mask_{mask}/seed{seed}_mask{mask}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=(256, 512, 1024), required=True)
    args = parser.parse_args()

    training = read("gpu_training_masks.txt")[:args.budget]
    gate = set(read("gpu_gate_masks.txt"))
    evaluation = read("gpu_evaluation_masks.txt")
    visible = [m for m in evaluation if m not in gate and cell(args.seed, m).exists()]
    if visible:
        raise SystemExit(f"{len(visible)} evaluation-only cells already exist for seed {args.seed}; refusing to fit")
    missing = [m for m in training if not cell(args.seed, m).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} training cells missing for seed {args.seed}; first {missing[:5]}")

    rows = {m: json.loads(cell(args.seed, m).read_text()) for m in training}
    gpus = sorted({rows[m].get("gpu_name") for m in training})
    y = np.vstack([np.asarray(rows[m]["target_log_probability_by_query"], dtype=float) for m in training])
    masks = np.asarray(training, dtype=np.int64)

    ideal = G.Game(k=2, g=3, d=D, queries=G.balanced_incidence(2, 3, D, Q, 1001).queries)
    rng = np.random.default_rng(2001)  # reproduce pilot replicate 1's rng stream exactly
    design = G.uniform_masks(D, 1024, rng)
    assert [int(m) for m in design] == read("gpu_training_masks.txt")
    G.uniform_masks(D, 4096, rng, exclude=set(design.tolist()))
    distill = G.uniform_masks(D, 2048, rng)

    fitted, choices = build_models(ideal, masks, y, 6001, distill)

    size3 = np.asarray(sorted(sum(1 << g for g in c) for c in itertools.combinations(range(D), 3)), dtype=np.int64)
    eval_masks = np.asarray(evaluation, dtype=np.int64)
    arrays = {"evaluation_masks": eval_masks, "size3_masks": size3}
    nominations = {}
    for name, model in fitted.items():
        arrays[f"{name}__evaluation"] = model.predict_chunked(eval_masks)
        pred3 = model.predict_chunked(size3)
        arrays[f"{name}__size3"] = pred3
        nominations[name] = [[int(size3[i]) for i in np.lexsort((size3, -pred3[:, q]))[:3]] for q in range(Q)]

    out = HERE / f"gpu/fits/seed{args.seed}/budget_{args.budget}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "frozen_predictions.npz", **arrays)
    result = {
        "status": "FROZEN_BEFORE_EVALUATION_CELLS", "seed": args.seed, "budget": args.budget,
        "training_gpu_names": gpus, "choices": choices, "nominations_top3_size3": nominations,
        "training_source_sha256": hashlib.sha256(
            b"".join(hashlib.sha256(cell(args.seed, m).read_bytes()).digest() for m in training)).hexdigest(),
        "predictions_sha256": hashlib.sha256((out / "frozen_predictions.npz").read_bytes()).hexdigest(),
        "evaluation_outcomes_loaded": False,
    }
    encoded = json.dumps(result, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    result["freeze_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    (out / "frozen_result.json").write_text(json.dumps(result, indent=2, sort_keys=True,
                                                       default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    print(json.dumps({"seed": args.seed, "budget": args.budget, "gpus": gpus,
                      "freeze_sha256": result["freeze_sha256"]}))


if __name__ == "__main__":
    main()
