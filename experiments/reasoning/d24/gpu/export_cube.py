#!/usr/bin/env python3
"""Export all real d=24 coalition cells as a compact cube: gpu/cube_d24_k2_g3.npz.

For each seed: masks (sorted), per-query final-step log-probability and accuracy,
role-atomic recall, and a role label (gate / training / evaluation / confirmation).
Raw per-cell JSONs are not part of the export.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
LISTS = {"gate": "gpu_gate_masks.txt", "training": "gpu_training_masks.txt",
         "evaluation": "gpu_evaluation_masks.txt", "confirmation": "gpu_confirmation_masks.txt"}
arrays, manifest = {}, {}
for seed in (31, 43, 59):
    roles: dict[int, list[str]] = {}
    for role, name in LISTS.items():
        for m in (HERE / "configs" / name).read_text().split():
            roles.setdefault(int(m), []).append(role)
    present = sorted(m for m in roles if (HERE / f"gpu/observed/seed{seed}/mask_{m}/seed{seed}_mask{m}.json").exists())
    logp, acc, atomic, label = [], [], [], []
    for m in present:
        row = json.loads((HERE / f"gpu/observed/seed{seed}/mask_{m}/seed{seed}_mask{m}.json").read_text())
        logp.append(row["target_log_probability_by_query"]); acc.append(row["target_accuracy_by_query"])
        atomic.append(np.nan if row["selected_role_atomic_accuracy"] is None else row["selected_role_atomic_accuracy"])
        label.append("+".join(roles[m]))
    arrays[f"seed{seed}_masks"] = np.asarray(present, dtype=np.int64)
    arrays[f"seed{seed}_log_probability"] = np.asarray(logp)
    arrays[f"seed{seed}_accuracy"] = np.asarray(acc)
    arrays[f"seed{seed}_atomic_recall"] = np.asarray(atomic)
    arrays[f"seed{seed}_role"] = np.asarray(label)
    manifest[str(seed)] = len(present)
np.savez_compressed(HERE / "gpu/cube_d24_k2_g3.npz", **arrays)
digest = hashlib.sha256((HERE / "gpu/cube_d24_k2_g3.npz").read_bytes()).hexdigest()
print(json.dumps({"cells_per_seed": manifest, "sha256": digest}))
