#!/usr/bin/env python3
"""Evaluate the frozen D=12 transformer coalition gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.reasoning.common.transformer_query_paths_d12 import Q, completed_queries

REQUIRED_MASKS = (4095, 21, 7, 4092, 4083, 4047, *tuple(1 << i for i in range(12)), 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for path in args.input_dir.glob("mask_*/*.json"):
        row = json.loads(path.read_text())
        rows[int(row["mask"])] = row
    missing = sorted(set(REQUIRED_MASKS) - set(rows))
    checks = []
    if not missing:
        for mask in REQUIRED_MASKS:
            row = rows[mask]
            structure = completed_queries(mask)
            for query in range(Q):
                accuracy = float(row["target_accuracy_by_query"][query])
                threshold = 15 / 31 if structure[query] else 2 / 31
                passed = accuracy >= threshold if structure[query] else accuracy <= threshold
                checks.append({"mask": mask, "query": query, "complete": structure[query],
                               "accuracy": accuracy, "threshold": threshold, "passed": passed})
            if mask:
                recall = float(row["selected_role_atomic_accuracy"])
                checks.append({"mask": mask, "check": "provided-role atomic recall",
                               "value": recall, "threshold": 0.9, "passed": recall > 0.9})
    passed = not missing and all(check["passed"] for check in checks)
    payload = {"status": "PASS" if passed else "FAIL", "learner_gate_passed": passed,
               "missing_masks": missing, "required_masks": list(REQUIRED_MASKS), "checks": checks,
               "decision": ("Proceed to frozen sparse coalition sampling." if passed else
                            "Stop this configuration; do not fit coalition surrogates.")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "missing": missing,
                      "failed_checks": sum(not x["passed"] for x in checks)}))


if __name__ == "__main__":
    main()
