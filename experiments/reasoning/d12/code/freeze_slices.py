#!/usr/bin/env python3
"""Turn a seed's frozen budget-160 nominations into executable slice manifests.

Writes, per seed, `SLICE_NOMINATION_PROTOCOL_seed<S>.json` and one mask manifest per
slice.  Must run after `fit_d12.py` and before any slice cell is retrained; it refuses to
run if slice outcomes for that seed already exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SLICES = (4, 5, 6)
EXECUTED_PER_QUERY = 3


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=160)
    args = parser.parse_args()

    frozen_path = args.root / "fits" / f"seed{args.seed}" / f"budget_{args.budget}" / "frozen_result.json"
    frozen = json.loads(frozen_path.read_text())
    assert frozen["status"] == "FROZEN_BEFORE_TARGET" and not frozen["target_outcomes_loaded"]

    protocol = {"seed": args.seed, "budget": args.budget,
                "status": "FROZEN_BEFORE_SLICE_OUTCOMES",
                "freeze_sha256": frozen["freeze_sha256"],
                "frozen_result_sha256": sha(frozen_path),
                "preregistration_sha256": frozen.get("preregistration_sha256", ""),
                "executed_per_query_count": EXECUTED_PER_QUERY,
                "nominations_retained": frozen["nominations_retained"],
                "blind_to_outcomes": frozen["blind_to_outcomes"],
                "candidate_counts": frozen["nomination_candidate_counts"],
                "slices": {}}
    for size in SLICES:
        existing = list((args.root / f"size{size}_eval" / f"seed{args.seed}").glob("mask_*/*.json"))
        if existing:
            raise RuntimeError(f"seed {args.seed} slice {size}: {len(existing)} outcomes already exist")
        per_method = frozen["nominations"][str(size)]
        union = sorted({mask for masks in per_method.values()
                        for query in masks for mask in query[:EXECUTED_PER_QUERY]})
        manifest = args.configs / f"slice{size}_exec_masks_seed{args.seed}.txt"
        manifest.write_text("".join(f"{mask}\n" for mask in union))
        protocol["slices"][str(size)] = {
            "methods": sorted(per_method),
            "executed_union_size": len(union),
            "execution_mask_sha256": sha(manifest),
            "manifest": manifest.name,
            "nominations": per_method}
    out = args.root.parent / f"SLICE_NOMINATION_PROTOCOL_seed{args.seed}.json"
    out.write_text(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"seed": args.seed, "out": out.name,
                      "union_sizes": {k: v["executed_union_size"] for k, v in protocol["slices"].items()},
                      "candidate_counts": protocol["candidate_counts"],
                      "methods": len(protocol["slices"]["4"]["methods"])}, indent=2))


if __name__ == "__main__":
    main()
