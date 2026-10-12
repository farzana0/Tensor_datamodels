#!/usr/bin/env python3
"""Value profile of the real d=24 training cells by role-completion pattern (cf. analysis/d12_calibration.json)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "gpu"))
from structure import PROVIDERS, Q  # noqa: E402

out = {}
masks = [int(x) for x in (HERE / "configs/gpu_training_masks.txt").read_text().split()]
for seed in (31, 43, 59):
    groups: dict = {}
    for mask in masks:
        y = json.loads((HERE / f"gpu/observed/seed{seed}/mask_{mask}/seed{seed}_mask{mask}.json").read_text())["target_log_probability_by_query"]
        for q in range(Q):
            covered = tuple(int(any(mask >> g & 1 for g in role)) for role in PROVIDERS[q])
            groups.setdefault(str(sorted(covered, reverse=True)), []).append(y[q])
    out[str(seed)] = {k: {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in sorted(groups.items())}
(HERE / "gpu/analysis").mkdir(parents=True, exist_ok=True)
(HERE / "gpu/analysis/training_value_profile.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=1))
