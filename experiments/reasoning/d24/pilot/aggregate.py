#!/usr/bin/env python3
"""Aggregate pilot arms into budget curves, sample-complexity tables and the decision.

Applies the pre-registered decision rule  exactly once
and writes its verdict to `analysis/pilot_decision.json`.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
TAU = 0.10
PRIMARY = "nmse"
LOWRANK = ("ladder_tt", "tt")
RIVALS = ("degree3_ridge", "rbf_krr")
SECONDARIES = ("shapley_relative_error", "loo_relative_error", "triple_interaction_ap")


def load(results_dir: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(results_dir.glob("arm_*.json"))]


def arm_key(payload: dict) -> tuple:
    arm = payload["arm"]
    return (arm["k"], arm["g"], arm["d"], arm["noise_multiplier"])


def budget_curves(payloads: list[dict]) -> dict:
    """mean/sd of every metric over replicates, per (arm, budget, method).

    Triple AP is NaN by construction when k != 3; those all-NaN means are expected.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    gathered: dict = defaultdict(lambda: defaultdict(list))
    for payload in payloads:
        key = arm_key(payload)
        for entry in payload["budgets"]:
            for method, metrics in entry["metrics"].items():
                for metric, value in metrics.items():
                    if metric == "choice":
                        continue
                    gathered[(key, entry["budget"], method)][metric].append(value)
    out = {}
    for key, metrics in gathered.items():
        out[key] = {metric: {"mean": float(np.nanmean(values)),
                             "sd": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                             "n": int(np.sum(~np.isnan(values)))}
                    for metric, values in metrics.items()}
    return out


def sample_complexity(curves: dict, budgets: list[int]) -> dict:
    """Smallest budget whose mean primary metric reaches the threshold."""
    out: dict = {}
    for (arm, budget, method), metrics in curves.items():
        out.setdefault((arm, method), {})[budget] = metrics[PRIMARY]["mean"]
    result = {}
    for (arm, method), by_budget in out.items():
        reached = [b for b in sorted(by_budget) if by_budget[b] <= TAU]
        result[(arm, method)] = {"m_star": min(reached) if reached else math.inf,
                                 "curve": {b: by_budget[b] for b in sorted(by_budget)}}
    return result


def decide(complexity: dict) -> dict:
    """The pre-registered stop/go rule."""
    arms = sorted({arm for arm, _ in complexity}, key=lambda a: (a[0], a[1], a[2]))
    rows, wins = [], []
    for arm in arms:
        k, g, d, noise = arm
        entry = {"k": k, "g": g, "d": d, "noise_multiplier": noise}
        for method in set(LOWRANK) | set(RIVALS):
            value = complexity.get((arm, method), {}).get("m_star", math.inf)
            entry[f"m_star_{method}"] = None if math.isinf(value) else int(value)
        for low in LOWRANK:
            low_star = complexity.get((arm, low), {}).get("m_star", math.inf)
            ratios = {}
            beats_all = True
            for rival in RIVALS:
                rival_star = complexity.get((arm, rival), {}).get("m_star", math.inf)
                if math.isinf(low_star):
                    beats_all = False
                    ratios[rival] = None
                    continue
                # An unreachable rival counts as a win only if the low-rank family
                # reached the threshold at all; the ratio is then reported as inf.
                ratios[rival] = math.inf if math.isinf(rival_star) else rival_star / low_star
                if not (math.isinf(rival_star) or rival_star / low_star >= 2.0):
                    beats_all = False
            entry[f"{low}_budget_ratio"] = {r: (None if v is None else ("inf" if math.isinf(v) else round(v, 3)))
                                            for r, v in ratios.items()}
            entry[f"{low}_beats_both_by_2x"] = bool(beats_all)
            if beats_all and noise == 1.0 and d >= 24:
                wins.append({"arm": arm, "family": low, "ratios": ratios})
        rows.append(entry)
    return {"threshold": TAU, "primary_metric": PRIMARY, "arms": rows,
            "condition_1_arms": [{"k": w["arm"][0], "g": w["arm"][1], "d": w["arm"][2],
                                  "family": w["family"],
                                  "ratios": {r: ("inf" if math.isinf(v) else round(v, 3))
                                             for r, v in w["ratios"].items()}} for w in wins],
            "verdict_condition_1_met": bool(wins)}


def min_ratio(complexity: dict, arm: tuple, family: str) -> float:
    low = complexity.get((arm, family), {}).get("m_star", math.inf)
    if math.isinf(low):
        return 0.0
    values = []
    for rival in RIVALS:
        rival_star = complexity.get((arm, rival), {}).get("m_star", math.inf)
        values.append(math.inf if math.isinf(rival_star) else rival_star / low)
    return min(values)


def conditions_2_and_3(decision: dict, complexity: dict, curves: dict) -> dict:
    """Condition 2: advantage at d=36 not smaller than at d=24 (same k, g, family).
    Condition 3: at the low-rank family's m*, degree-3 ridge is strictly worse on at
    least one secondary (Shapley error, LOO error, triple AP, top-3 completeness)."""
    evaluated = []
    for win in decision["condition_1_arms"]:
        k, g, d, family = win["k"], win["g"], win["d"], win["family"]
        r24 = min_ratio(complexity, (k, g, 24, 1.0), family)
        r36 = min_ratio(complexity, (k, g, 36, 1.0), family)
        condition_2 = r36 >= r24
        arm = (k, g, d, 1.0)
        m_star = int(complexity[(arm, family)]["m_star"])
        low = curves.get((arm, m_star, family), {})
        rival = curves.get((arm, m_star, "degree3_ridge"), {})
        losses = []
        for metric, worse_is_higher in (("shapley_relative_error", True), ("loo_relative_error", True),
                                        ("triple_interaction_ap", False)):
            if metric in low and metric in rival:
                a, b = low[metric]["mean"], rival[metric]["mean"]
                if not (math.isnan(a) or math.isnan(b)) and ((b > a) if worse_is_higher else (b < a)):
                    losses.append(metric)
        for metric in low:
            if metric.startswith("top3_complete_size") and metric in rival:
                if rival[metric]["mean"] < low[metric]["mean"]:
                    losses.append(metric)
        evaluated.append({**win, "ratio_d24": "inf" if math.isinf(r24) else round(r24, 3),
                          "ratio_d36": "inf" if math.isinf(r36) else round(r36, 3),
                          "condition_2": bool(condition_2), "m_star_low_rank": m_star,
                          "degree3_loses_on": losses, "condition_3": bool(losses)})
    full = [e for e in evaluated if e["condition_2"] and e["condition_3"]]
    prediction_only = [e for e in evaluated if e["condition_2"] and not e["condition_3"]]
    verdict = "GO" if full else ("GO_PREDICTION_ONLY" if prediction_only else "STOP")
    return {"evaluated": evaluated, "verdict": verdict}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "pilot/results")
    args = parser.parse_args()
    payloads = load(args.results)
    if not payloads:
        raise SystemExit(f"no arm results under {args.results}")
    budgets = sorted({entry["budget"] for payload in payloads for entry in payload["budgets"]})
    curves = budget_curves(payloads)
    complexity = sample_complexity(curves, budgets)

    (HERE / "tables").mkdir(exist_ok=True)
    (HERE / "analysis").mkdir(exist_ok=True)

    methods = sorted({method for (_, _, method) in curves})
    lines = ["k,g,d,noise_multiplier,budget,method,nmse_mean,nmse_sd,replicates,"
             "shapley_relative_error,loo_relative_error,triple_interaction_ap,"
             "top3_complete_min_size,top3_complete_min_size_plus_one"]
    for (arm, budget, method), metrics in sorted(curves.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        k, g, d, noise = arm
        sizes = sorted(int(name.split("size")[-1]) for name in metrics if name.startswith("top3_complete_size"))
        selection = [metrics[f"top3_complete_size{size}"]["mean"] for size in sizes]
        lines.append(",".join(str(x) for x in [
            k, g, d, noise, budget, method,
            round(metrics[PRIMARY]["mean"], 6), round(metrics[PRIMARY]["sd"], 6), metrics[PRIMARY]["n"],
            *[round(metrics[name]["mean"], 6) if name in metrics else "" for name in SECONDARIES],
            *[round(value, 4) for value in selection]]))
    (HERE / "tables/pilot_budget_curves.csv").write_text("\n".join(lines) + "\n")

    complexity_lines = ["k,g,d,noise_multiplier,method,m_star," + ",".join(f"nmse_m{b}" for b in budgets)]
    for (arm, method), entry in sorted(complexity.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        k, g, d, noise = arm
        star = "" if math.isinf(entry["m_star"]) else int(entry["m_star"])
        complexity_lines.append(",".join(str(x) for x in [
            k, g, d, noise, method, star,
            *[round(entry["curve"].get(b, float("nan")), 6) for b in budgets]]))
    (HERE / "tables/pilot_sample_complexity.csv").write_text("\n".join(complexity_lines) + "\n")

    decision = decide(complexity)
    decision.update(conditions_2_and_3(decision, complexity, curves))
    decision["arms_run"] = len(payloads)
    decision["arms_expected"] = 144
    decision["budgets"] = budgets
    (HERE / "analysis/pilot_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"arms": len(payloads), "methods": methods,
                      "verdict": decision["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
