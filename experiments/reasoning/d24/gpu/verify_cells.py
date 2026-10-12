#!/usr/bin/env python3
"""Check that every listed coalition cell exists, ran on RTX8000 with the frozen runner, and is well formed."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main() -> None:
    masks_file, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    masks = [int(x) for x in (HERE / "configs" / masks_file).read_text().split()]
    gate = {int(x) for x in (HERE / "configs/gpu_gate_masks.txt").read_text().split()}
    evaluation = [int(x) for x in (HERE / "configs/gpu_evaluation_masks.txt").read_text().split()]
    for seed in seeds:
        gpus, runners, bad, missing = Counter(), Counter(), [], []
        for mask in masks:
            path = HERE / f"gpu/observed/seed{seed}/mask_{mask}/seed{seed}_mask{mask}.json"
            if not path.exists():
                missing.append(mask); continue
            row = json.loads(path.read_text())
            gpus[row.get("gpu_name")] += 1
            runners[row.get("runner_sha256", "")[:12]] += 1
            y = row["target_log_probability_by_query"]
            if row["mask"] != mask or row["seed"] != seed or len(y) != 10 or any(v != v for v in y):
                bad.append(mask)
        eval_only_present = sum((HERE / f"gpu/observed/seed{seed}/mask_{m}/seed{seed}_mask{m}.json").exists()
                                for m in evaluation if m not in gate)
        print(json.dumps({"seed": seed, "list": masks_file, "expected": len(masks), "missing": len(missing),
                          "malformed": len(bad), "gpus": dict(gpus), "runner_sha12": dict(runners),
                          "evaluation_only_cells_present": eval_only_present}))


if __name__ == "__main__":
    main()
