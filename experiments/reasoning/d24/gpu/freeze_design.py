#!/usr/bin/env python3
"""Freeze the outcome-free coalition design for the targeted GPU arm (k=2, g=3, d=24).

- training: the pilot's replicate-1 design, verbatim (`uniform_masks(24, 1024, rng(2001))`),
  so the nested budgets 256/512/1024 are the same prefixes the pilot fitted;
- test: the first 256 masks of the pilot's replicate-1 held-out draw (same rng stream,
  disjoint from training);
- LOO / leave-role-out / full / empty evaluation masks;
- gate masks (seed 31 only).

Writes `configs/gpu_*.txt` and `configs/GPU_DESIGN.json` with sha256 of every list.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "pilot"))
sys.path.insert(0, str(HERE / "gpu"))

import game as G  # noqa: E402

D = 24
FULL = (1 << D) - 1
PROVIDERS = G.balanced_incidence(2, 3, D, 10, 1001).queries


def write(name: str, masks: list[int]) -> str:
    text = "\n".join(str(int(m)) for m in masks) + "\n"
    (HERE / "configs" / name).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    rng = np.random.default_rng(2001)
    training = [int(m) for m in G.uniform_masks(D, 1024, rng)]
    test = [int(m) for m in G.uniform_masks(D, 4096, rng, exclude=set(training))][:256]

    singletons = [1 << g for g in range(D)]
    minimal_complete = [(1 << PROVIDERS[q][0][q % 3]) | (1 << PROVIDERS[q][1][(q + 1) % 3]) for q in range(10)]

    def without(groups) -> int:
        mask = FULL
        for g in groups:
            mask &= ~(1 << int(g))
        return mask

    gate_leave_role_out = [without(PROVIDERS[q][q % 2]) for q in range(10)]
    gate = list(dict.fromkeys([FULL, 0, *singletons, *minimal_complete, *gate_leave_role_out]))

    loo = [without([g]) for g in range(D)]
    leave_role_out = [without(PROVIDERS[q][r]) for q in range(10) for r in range(2)]
    evaluation = list(dict.fromkeys([*test, *loo, *leave_role_out, FULL, 0]))

    overlap_train_gate = sorted(set(training) & set(gate))
    overlap_train_eval = sorted(set(training) & set(evaluation))
    hashes = {
        "gpu_training_masks.txt": write("gpu_training_masks.txt", training),
        "gpu_test_masks.txt": write("gpu_test_masks.txt", test),
        "gpu_evaluation_masks.txt": write("gpu_evaluation_masks.txt", evaluation),
        "gpu_gate_masks.txt": write("gpu_gate_masks.txt", gate),
    }
    payload = {
        "status": "FROZEN_BEFORE_ANY_GPU_OUTCOME",
        "d": D, "k": 2, "g": 3, "queries": 10,
        "incidence": [[[int(x) for x in role] for role in query] for query in PROVIDERS],
        "training_count": len(training), "nested_budgets": [256, 512, 1024],
        "test_count": len(test), "evaluation_count": len(evaluation), "gate_count": len(gate),
        "loo_count": len(loo), "leave_role_out_count": len(leave_role_out),
        "overlap_training_gate": overlap_train_gate, "overlap_training_evaluation": overlap_train_eval,
        "sha256": hashes,
        "pilot_replicate_1_design_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(training, dtype=np.int64)).tobytes()).hexdigest(),
    }
    (HERE / "configs/GPU_DESIGN.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in ("training_count", "test_count", "evaluation_count",
                                              "gate_count", "overlap_training_gate",
                                              "overlap_training_evaluation",
                                              "pilot_replicate_1_design_sha256")}, indent=2))


if __name__ == "__main__":
    main()
