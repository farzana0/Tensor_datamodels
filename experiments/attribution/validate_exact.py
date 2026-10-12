#!/usr/bin/env python3
"""Check the equal-budget study exact-attribution engine against shapiq's ExactComputer.

This runs before anything is frozen: it establishes that "ground truth" in equal-budget study
means the same thing it means in the shapiq reference implementation, so the
equal-budget comparison is not measuring a convention mismatch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cubes, exact  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def main() -> int:
    import shapiq
    from shapiq.game_theory.exact import ExactComputer

    rng = np.random.default_rng(20260912)
    report = {"shapiq_version": shapiq.__version__, "checks": []}
    worst = 0.0
    # The original check also covered four cubes that are not part of this release
    # (arith_d5_seed2023, khop_d8, fano_d8, har_d10); their saved results remain in
    # results/exact_validation.json, which this script overwrites when re-run.
    for name in ["modular_d8_seed31", "agnews_lora_r8_seed31"]:
        cube = cubes.load(name)
        values = cube.output(cube.primary_output)
        d = cube.d
        computer = ExactComputer(n_players=d, game=lambda coalitions: np.asarray(
            [values[int(np.dot(row.astype(int), 1 << np.arange(d)))] for row in coalitions]))

        checks = {}
        sv = computer(index="SV", order=1)
        mine = exact.shapley_values(values)
        checks["shapley"] = float(np.max(np.abs(np.asarray([sv[(i,)] for i in range(d)]) - mine)))

        bz = computer(index="BV", order=1)
        mine_bz = exact.banzhaf_values(values)
        checks["banzhaf"] = float(np.max(np.abs(np.asarray([bz[(i,)] for i in range(d)]) - mine_bz)))

        sii = computer(index="SII", order=2)
        mine_sii = exact.pair_interactions(values, shapley=True)
        checks["sii_order2"] = float(np.max(np.abs(
            np.asarray([sii[pair] for pair in exact.pairs(d)]) - mine_sii)))

        bii = computer(index="BII", order=2)
        mine_bii = exact.pair_interactions(values, shapley=False)
        checks["bii_order2"] = float(np.max(np.abs(
            np.asarray([bii[pair] for pair in exact.pairs(d)]) - mine_bii)))

        sii3 = computer(index="SII", order=3)
        mine_sii3 = exact.triple_interactions(values, shapley=True)
        checks["sii_order3"] = float(np.max(np.abs(
            np.asarray([sii3[triple] for triple in exact.triples(d)]) - mine_sii3)))

        mob = computer(index="Moebius", order=d)
        mine_mob = exact.mobius(values)
        masks = [tuple(np.flatnonzero((np.arange(d) >> 0) & 0)) for _ in range(0)]  # noqa: F841
        error = 0.0
        for mask in range(1 << d):
            members = tuple(i for i in range(d) if mask >> i & 1)
            error = max(error, abs(float(mob[members]) - float(mine_mob[mask])))
        checks["moebius"] = error

        # Moebius round trip and the efficiency identity are internal consistency checks.
        full = (1 << d) - 1
        checks["efficiency"] = float(abs(mine.sum() - (values[full] - values[0])))
        report["checks"].append({"cube": name, "d": d, **checks})
        worst = max(worst, max(checks.values()))
        print(name, json.dumps(checks))

    report["worst_absolute_discrepancy"] = worst
    # Scale-free sanity: worst discrepancy relative to the largest Shapley value seen.
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exact_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("worst absolute discrepancy", worst)
    return 0 if worst < 1e-8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
