#!/usr/bin/env python3
"""Amendment A7 analysis for the targeted GPU arm (k=2, g=3, d=24).

Subcommands:
  confirm-list  write configs/gpu_confirmation_masks.txt from frozen seed-31 nominations
  analyze       compute the primary half-budget test and all secondaries
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "gpu"))
from structure import D, PROVIDERS, Q, completed_queries  # noqa: E402

SEEDS = (31, 43, 59)
BUDGETS = (256, 512, 1024)
RIVALS = ("degree3_ridge", "rbf_krr", "degree3_omp")
LOW = ("tt", "cp", "ladder_tt")
CONFIRM_METHODS = ("tt", "cp", "degree3_ridge", "rbf_krr", "degree3_omp", "linear")
TAU = 0.10
FULL = (1 << D) - 1


def read(name: str) -> list[int]:
    return [int(x) for x in (HERE / "configs" / name).read_text().split()]


def cell(seed: int, mask: int) -> Path:
    return HERE / f"gpu/observed/seed{seed}/mask_{mask}/seed{seed}_mask{mask}.json"


# The repository ships the retraining outcomes as one compact cube
# (gpu/cube_d24_k2_g3.npz, written by gpu/export_cube.py) rather than as ~4,100
# per-coalition JSON files.  Per-coalition files are used when present (i.e. after
# re-running the retraining jobs); otherwise values are read from the cube.
_CUBE: dict = {}


def _cube(seed: int) -> dict[int, np.ndarray]:
    if seed not in _CUBE:
        data = np.load(HERE / "gpu/cube_d24_k2_g3.npz")
        _CUBE[seed] = {int(m): row for m, row in zip(data[f"seed{seed}_masks"],
                                                     data[f"seed{seed}_log_probability"])}
    return _CUBE[seed]


def has_cell(seed: int, mask: int) -> bool:
    return cell(seed, mask).exists() or int(mask) in _cube(seed)


def utility(seed: int, mask: int) -> np.ndarray:
    if cell(seed, mask).exists():
        return np.asarray(json.loads(cell(seed, mask).read_text())["target_log_probability_by_query"], dtype=float)
    return np.asarray(_cube(seed)[int(mask)], dtype=float)


def frozen(seed: int, budget: int) -> tuple[dict, dict]:
    base = HERE / f"gpu/fits/seed{seed}/budget_{budget}"
    return dict(np.load(base / "frozen_predictions.npz")), json.loads((base / "frozen_result.json").read_text())


def confirm_list() -> None:
    masks = []
    for budget in (512, 1024):
        _, result = frozen(31, budget)
        for method in CONFIRM_METHODS:
            masks.extend(result["nominations_top3_size3"][method][q][0] for q in range(Q))
    unique = list(dict.fromkeys(int(m) for m in masks))
    text = "\n".join(map(str, unique)) + "\n"
    (HERE / "configs/gpu_confirmation_masks.txt").write_text(text)
    print(f"{len(unique)} confirmation masks, sha256 {hashlib.sha256(text.encode()).hexdigest()}")


def analyze() -> None:
    test = read("gpu_test_masks.txt")
    evaluation = read("gpu_evaluation_masks.txt")
    index = {m: i for i, m in enumerate(evaluation)}
    test_rows = [index[m] for m in test]
    out: dict = {"nmse": {}, "freeze_sha256": {}, "gpu_names": {}}

    # Primary metric: test nMSE per seed, budget, method.
    for seed in SEEDS:
        y = np.vstack([utility(seed, m) for m in test])
        var = y.var(0)
        var[var < 1e-12] = 1.0
        names = set()
        for mask in test[:32]:
            if cell(seed, mask).exists():  # GPU names are only recorded in per-coalition files
                names.add(json.loads(cell(seed, mask).read_text()).get("gpu_name"))
        out["gpu_names"][str(seed)] = sorted(n for n in names if n)
        for budget in BUDGETS:
            preds, result = frozen(seed, budget)
            out["freeze_sha256"][f"{seed}_{budget}"] = result["freeze_sha256"]
            for key, value in preds.items():
                if not key.endswith("__evaluation"):
                    continue
                method = key[: -len("__evaluation")]
                p = value[test_rows]
                out["nmse"].setdefault(method, {}).setdefault(str(budget), {})[str(seed)] = float(
                    np.mean(((p - y) ** 2).mean(0) / var))

    # Primary test.
    nmse = out["nmse"]
    primary = {}
    for low in LOW:
        table = {}
        for rival in RIVALS:
            wins = {str(s): bool(nmse[low]["512"][str(s)] < nmse[rival]["1024"][str(s)]) for s in SEEDS}
            table[rival] = {"wins_by_seed": wins, "n_wins": sum(wins.values())}
        all_nine = all(v["n_wins"] == 3 for v in table.values())
        partial = all(v["n_wins"] >= 2 for v in table.values())
        primary[low] = {"comparisons": table,
                        "verdict": "SUPPORTED" if all_nine else ("PARTIAL" if partial else "FAILED")}
    out["primary_half_budget_test"] = primary
    out["primary_verdict_tt"] = primary["tt"]["verdict"]

    # Pre-registered m* with explicit censoring.
    mstar = {}
    for method, by_budget in nmse.items():
        means = {b: float(np.mean(list(by_budget[str(b)].values()))) for b in BUDGETS}
        reached = [b for b in BUDGETS if means[b] <= TAU]
        mstar[method] = {"mean_nmse": means, "m_star": min(reached) if reached else ">1024 (censored)"}
    out["m_star_tau_0.10"] = mstar

    # Secondary: LOO and leave-role-out effects, predicted vs observed.
    loo_masks = [FULL & ~(1 << g) for g in range(D)]
    lro_masks = [FULL & ~sum(1 << g for g in PROVIDERS[q][r]) for q in range(Q) for r in range(2)]
    effects = {}
    for seed in SEEDS:
        full = utility(seed, FULL)
        obs_loo = np.vstack([full - utility(seed, m) for m in loo_masks])
        obs_lro = np.vstack([full - utility(seed, m) for m in lro_masks])
        for budget in BUDGETS:
            preds, _ = frozen(seed, budget)
            for key, value in preds.items():
                if not key.endswith("__evaluation"):
                    continue
                method = key[: -len("__evaluation")]
                pf = value[index[FULL]]
                p_loo = np.vstack([pf - value[index[m]] for m in loo_masks])
                p_lro = np.vstack([pf - value[index[m]] for m in lro_masks])
                rec = effects.setdefault(method, {}).setdefault(str(budget), {})
                rec[str(seed)] = {
                    "loo_rmse": float(np.sqrt(np.mean((p_loo - obs_loo) ** 2))),
                    "leave_role_out_rmse": float(np.sqrt(np.mean((p_lro - obs_lro) ** 2))),
                }
    out["effects"] = effects

    # Secondary: structural completeness of frozen top-3 size-3 nominations.
    completeness = {}
    for seed in SEEDS:
        for budget in BUDGETS:
            _, result = frozen(seed, budget)
            for method, per_query in result["nominations_top3_size3"].items():
                value = float(np.mean([[completed_queries(m)[q] for m in per_query[q]] for q in range(Q)]))
                completeness.setdefault(method, {}).setdefault(str(budget), {})[str(seed)] = value
    out["top3_size3_structural_completeness"] = completeness

    # Secondary: retrained utility of the top-1 nominations (seed 31).
    confirm_path = HERE / "configs/gpu_confirmation_masks.txt"
    if confirm_path.exists():
        confirmation = {}
        for budget in (512, 1024):
            _, result = frozen(31, budget)
            for method in CONFIRM_METHODS:
                values, complete = [], []
                for q in range(Q):
                    mask = result["nominations_top3_size3"][method][q][0]
                    if has_cell(31, mask):
                        values.append(float(utility(31, mask)[q]))
                        complete.append(bool(completed_queries(mask)[q]))
                confirmation.setdefault(method, {})[str(budget)] = {
                    "n": len(values), "mean_retrained_utility": float(np.mean(values)) if values else None,
                    "complete_fraction": float(np.mean(complete)) if complete else None}
        out["seed31_top1_retrained_confirmation"] = confirmation

    # Retraining-seed variability on identical test masks.
    stack = np.stack([np.vstack([utility(s, m) for m in test]) for s in SEEDS])
    out["between_seed_sd_on_test_cells"] = float(stack.std(0, ddof=1).mean())
    out["test_utility_sd_within_seed"] = float(np.mean([stack[i].std(0).mean() for i in range(len(SEEDS))]))
    complete_mask = np.asarray([[completed_queries(m)[q] for q in range(Q)] for m in test])
    out["test_complete_fraction"] = float(complete_mask.mean())

    (HERE / "gpu/analysis").mkdir(parents=True, exist_ok=True)
    (HERE / "gpu/analysis/result.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    lines = ["method,budget,nmse_seed31,nmse_seed43,nmse_seed59,nmse_mean"]
    for method in sorted(nmse):
        for budget in BUDGETS:
            v = [nmse[method][str(budget)][str(s)] for s in SEEDS]
            lines.append(",".join([method, str(budget), *[f"{x:.5f}" for x in v], f"{np.mean(v):.5f}"]))
    (HERE / "tables").mkdir(exist_ok=True)
    (HERE / "tables/gpu_d24_nmse.csv").write_text("\n".join(lines) + "\n")
    print(json.dumps({"primary_verdict_tt": out["primary_verdict_tt"],
                      "primary": {k: {r: c["n_wins"] for r, c in v["comparisons"].items()} | {"verdict": v["verdict"]}
                                  for k, v in primary.items()},
                      "between_seed_sd": out["between_seed_sd_on_test_cells"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("confirm-list", "analyze"))
    args = parser.parse_args()
    confirm_list() if args.command == "confirm-list" else analyze()


if __name__ == "__main__":
    main()
