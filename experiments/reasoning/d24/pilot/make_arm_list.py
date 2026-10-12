#!/usr/bin/env python3
"""Write the frozen pilot arm list (one line per Slurm array task).

Primary grid: every pre-registered structure at every feasible d, noise multiplier
1, replicates 1-5.  Secondary robustness grid: noise multipliers 4 and 16 on the
two structures with g=2 at d in {12, 24}, replicates 1-3 (a subset, for cost; see the recorded deviations).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

STRUCTURES = ((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3))
DIMENSIONS = (12, 24, 36)
PRIMARY_REPLICATES = (1, 2, 3, 4, 5)
SECONDARY = {"structures": ((3, 2), (4, 2)), "dimensions": (12, 24),
             "noise": (4.0, 16.0), "replicates": (1, 2, 3)}


def main() -> None:
    rows = []
    for k, g in STRUCTURES:
        for d in DIMENSIONS:
            if k * g > d:
                continue
            for replicate in PRIMARY_REPLICATES:
                rows.append((k, g, d, 1.0, replicate))
    for k, g in SECONDARY["structures"]:
        for d in SECONDARY["dimensions"]:
            for noise in SECONDARY["noise"]:
                for replicate in SECONDARY["replicates"]:
                    rows.append((k, g, d, noise, replicate))
    lines = [f"{k} {g} {d} {noise} {replicate}" for k, g, d, noise, replicate in rows]
    text = "\n".join(lines) + "\n"
    out = Path(__file__).resolve().parent.parent / "configs/pilot_arms.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"{len(lines)} arms -> {out}")
    print("sha256", hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
