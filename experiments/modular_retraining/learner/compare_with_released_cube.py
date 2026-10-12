#!/usr/bin/env python3
"""Compare freshly retrained modular coalition outcomes with the released cube.

    python known_interaction_game.py freeze --root runs/check
    for i in 0 1 2; do python known_interaction_game.py worker --root runs/check --seed-index $i; done
    python compare_with_released_cube.py --root runs/check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    cube = np.load(REPO / "data/cubes/modular_relations_d8.npz", allow_pickle=False)
    names = [str(name) for name in cube["output_names"]]
    worst = 0.0
    for seed in (31, 43, 59):
        rows = np.flatnonzero(cube["seeds"] == seed)
        released = {int(cube["masks"][row]): cube["values"][row] for row in rows}
        for mask in range(256):
            cell = json.loads((args.root / "cells" / f"seed{seed}_mask{mask}.json").read_text())
            fresh = np.asarray([cell["metrics"][name]["utility_log_probability"] for name in names])
            worst = max(worst, float(np.abs(fresh - released[mask]).max()))
    print(f"768 coalition cells x {len(names)} outputs; max |retrained - released| = {worst:.3e}")


if __name__ == "__main__":
    main()
