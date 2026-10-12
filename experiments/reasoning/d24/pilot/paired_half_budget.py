#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered): uncensored half-budget test.

For every primary-noise arm and replicate, compare a low-rank family at m=512
against each rival at m=1024 on the primary metric (nMSE vs the noiseless game).
"Wins" means the low-rank family with half the retrainings has strictly lower
nMSE than the rival with twice as many.  This sidesteps the right-censoring of
the pre-registered m* (rivals that never reach tau=0.10 by m=1024).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LOW = ("tt", "cp", "ladder_tt")
RIVALS = ("degree3_ridge", "rbf_krr", "degree3_omp", "degree2_ridge")


def main() -> None:
    wins = defaultdict(lambda: [0, 0])
    gaps = defaultdict(list)
    for path in sorted((HERE / "pilot/results").glob("arm_*_c1.0_*.json")):
        payload = json.loads(path.read_text())
        arm = payload["arm"]
        by_budget = {entry["budget"]: entry["metrics"] for entry in payload["budgets"]}
        for low in LOW:
            for rival in RIVALS:
                key = (arm["k"], arm["g"], arm["d"], low, rival)
                a, b = by_budget[512][low]["nmse"], by_budget[1024][rival]["nmse"]
                wins[key][0] += int(a < b)
                wins[key][1] += 1
                gaps[key].append(b - a)
    lines = ["k,g,d,low_rank_at_512,rival_at_1024,replicates_won,replicates,mean_nmse_gap_rival_minus_low"]
    for key in sorted(wins):
        won, total = wins[key]
        lines.append(",".join(map(str, [*key, won, total, round(sum(gaps[key]) / len(gaps[key]), 4)])))
    (HERE / "tables/exploratory_half_budget_paired.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(line for line in lines
                    if line.startswith("k") or (",tt," in line and ",12," not in line.split("tt")[0][-4:])))


if __name__ == "__main__":
    main()
