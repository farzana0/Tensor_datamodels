#!/usr/bin/env python3
"""Emit the mask manifests this workstream adds, before any cell is run.

Gate and observed manifests are copied byte-for-byte from the development run and are not
regenerated here.  This script writes only the three new lists: the leave-one-group-out
influence cells, the noise-ceiling masks, and the paired size-3 target slice check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

D = 12
HERE = Path(__file__).resolve().parent.parent
CONFIGS = HERE / "configs"
CEILING_SEED = 20260912


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, masks: list[int]) -> Path:
    path = CONFIGS / name
    path.write_text("".join(f"{mask}\n" for mask in masks))
    return path


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    loo = [4095 ^ (1 << group) for group in range(D)]
    ceiling = [4095, 0]
    rng = np.random.default_rng(CEILING_SEED)
    size5 = rng.choice([m for m in range(1 << D) if m.bit_count() == 5], 6, replace=False)
    ceiling += sorted(int(m) for m in size5)
    pool = [m for m in range(1 << D) if 1 <= m.bit_count() <= 11 and m not in ceiling]
    ceiling += sorted(int(m) for m in rng.choice(pool, 12, replace=False))
    assert len(ceiling) == 20 and len(set(ceiling)) == 20
    manifest = {
        "loo_influence_masks": {"count": len(loo), "sha256": sha(write("loo_masks.txt", loo)),
                                "definition": "full coalition with exactly one group removed"},
        "ceiling_masks": {"count": len(ceiling), "sha256": sha(write("ceiling_masks.txt", ceiling)),
                          "rng_seed": CEILING_SEED,
                          "composition": "full, empty, 6 uniform size-5, 12 uniform over sizes 1-11"},
        "reused_from_development_run": {name: sha(CONFIGS / name) for name in
                                    ("all_observed_masks.txt", "new_observed_masks.txt",
                                     "frozen_target_masks.txt", "interaction_audit_masks.txt")},
    }
    (CONFIGS / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
