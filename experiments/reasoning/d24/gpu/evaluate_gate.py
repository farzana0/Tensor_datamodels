#!/usr/bin/env python3
"""Evaluate the frozen d=24 learner gate (thresholds identical to 15_ GATE_PROTOCOL.json).

Checks, per gate cell and query: complete queries >= 15/31 accuracy, incomplete <= 2/31;
provided-role atomic recall > 0.9.  The gate masks include one minimal complete set per
query (any single provider per role suffices) and one leave-role-out mask per query
(removing a role's providers kills the query), so both OR checks are covered.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "gpu"))
from structure import Q, completed_queries  # noqa: E402


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    masks = [int(x) for x in (HERE / "configs/gpu_gate_masks.txt").read_text().split()]
    checks, missing, gpus, elapsed = [], [], set(), []
    for mask in masks:
        path = HERE / f"gpu/observed/seed{seed}/mask_{mask}/seed{seed}_mask{mask}.json"
        if not path.exists():
            missing.append(mask); continue
        row = json.loads(path.read_text())
        gpus.add(row.get("gpu_name")); elapsed.append(row["elapsed_seconds"])
        structure = completed_queries(mask)
        for query in range(Q):
            accuracy = float(row["target_accuracy_by_query"][query])
            threshold = 15 / 31 if structure[query] else 2 / 31
            ok = accuracy >= threshold if structure[query] else accuracy <= threshold
            checks.append({"mask": mask, "query": query, "complete": structure[query],
                           "accuracy": accuracy, "threshold": threshold, "passed": ok})
        if row["selected_role_atomic_accuracy"] is not None:
            recall = float(row["selected_role_atomic_accuracy"])
            checks.append({"mask": mask, "check": "provided-role atomic recall", "value": recall,
                           "threshold": 0.9, "passed": recall > 0.9})
    passed = not missing and all(c["passed"] for c in checks)
    failed = [c for c in checks if not c["passed"]]
    payload = {"status": "PASS" if passed else ("INCOMPLETE" if missing else "FAIL"), "seed": seed,
               "missing_masks": missing, "gpu_names": sorted(g for g in gpus if g),
               "median_cell_seconds": sorted(elapsed)[len(elapsed) // 2] if elapsed else None,
               "failed_checks": failed, "n_checks": len(checks), "checks": checks}
    out = HERE / f"gpu/gate/GATE_DECISION_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in ("status", "missing_masks", "gpu_names", "median_cell_seconds", "n_checks")}
                     | {"n_failed": len(failed), "first_failures": failed[:6]}, indent=2))


if __name__ == "__main__":
    main()
