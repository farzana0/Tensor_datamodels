#!/usr/bin/env python3
"""Recompute the paper's reported numbers from the result files in this repository.

Every value is recomputed from saved coalition outcomes, frozen surrogate
predictions or saved per-repetition metrics; nothing is retrained and nothing is
fitted.  Each line shows the value printed in the paper, the value recomputed
here, and whether they agree at the paper's printed precision.

    python scripts/check_paper_numbers.py            # all tables
    python scripts/check_paper_numbers.py --strict   # non-zero exit on any mismatch

Requires numpy and pandas only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
ROWS: list[tuple[str, str, str, str, bool]] = []


def check(table: str, label: str, paper: float, value: float, decimals: int = 3) -> None:
    """Record one comparison; agreement means equal after rounding to the paper's precision."""
    ok = bool(np.isfinite(value)) and abs(round(float(value), decimals) - paper) <= 0.5 * 10 ** -decimals + 1e-12
    ROWS.append((table, label, f"{paper:.{decimals}f}", f"{value:.{decimals + 2}f}", ok))


def mean_sd(values) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


# --------------------------------------------------------------------------- d = 24
def table_1_5_6_9() -> None:
    result = json.loads((EXP / "reasoning/d24/gpu/analysis/result.json").read_text())
    names = {"linear": "Linear", "degree2_ridge": "Degree-2 ridge", "degree3_ridge": "Degree-3 ridge",
             "degree3_omp": "Degree-3 OMP", "rbf_krr": "RBF", "cp": "TNDM-CP", "tt": "TNDM-TT"}
    paper_nmse = {  # (m=512 mean, sd), (m=1024 mean, sd)   -- Table 1
        "linear": ((0.687, 0.001), (0.669, 0.001)), "degree2_ridge": ((0.467, 0.001), (0.348, 0.001)),
        "degree3_ridge": ((0.510, 0.002), (0.342, 0.002)), "degree3_omp": ((0.635, 0.016), (0.271, 0.021)),
        "rbf_krr": ((0.386, 0.001), (0.257, 0.001)), "cp": ((0.148, 0.026), (0.087, 0.002)),
        "tt": ((0.188, 0.004), (0.071, 0.006))}
    paper_joint = {"linear": (2.741, 0.016), "degree2_ridge": (2.125, 0.007), "degree3_ridge": (3.869, 0.019),
                   "degree3_omp": (6.292, 0.057), "rbf_krr": (2.100, 0.035), "cp": (1.939, 0.258),
                   "tt": (1.395, 0.043)}
    for key, name in names.items():
        for column, budget in enumerate(("512", "1024")):
            mean, sd = mean_sd(result["nmse"][key][budget].values())
            check("Table 1", f"{name} NMSE m={budget} mean", paper_nmse[key][column][0], mean)
            check("Table 1", f"{name} NMSE m={budget} sd", paper_nmse[key][column][1], sd)
        mean, sd = mean_sd(v["leave_role_out_rmse"] for v in result["effects"][key]["1024"].values())
        check("Table 1", f"{name} joint-removal RMSE mean", paper_joint[key][0], mean)
        check("Table 1", f"{name} joint-removal RMSE sd", paper_joint[key][1], sd)
    # Table 5 extras (means only) and Table 6 leave-one-out column.
    extra = {"cardinality": (0.972, 0.934, 0.918), "ht": (0.951, 0.850, 0.766), "ladder_tt": (0.660, 0.432, 0.269),
             "cp": (0.490, 0.148, 0.087), "tt": (0.472, 0.188, 0.071)}
    for key, values in extra.items():
        for budget, paper in zip(("256", "512", "1024"), values):
            check("Table 5", f"{key} NMSE m={budget}", paper, float(np.mean(list(result["nmse"][key][budget].values()))))
    loo = {"cardinality": 0.556, "ht": 0.555, "tt": 0.559, "cp": 0.780, "linear": 1.184}
    for key, paper in loo.items():
        check("Table 6", f"{key} LOO RMSE m=1024",
              paper, float(np.mean([v["loo_rmse"] for v in result["effects"][key]["1024"].values()])))
    confirm = result["seed31_top1_retrained_confirmation"]
    paper9 = {"linear": ((8, -2.324), (8, -2.326)), "cp": ((9, -1.142), (10, -0.001)), "tt": ((10, -0.001), (8, -2.285)),
              "degree3_ridge": ((10, -0.001), (10, -0.001))}
    for key, pair in paper9.items():
        for budget, (complete, utility) in zip(("512", "1024"), pair):
            check("Table 9", f"{key} complete/10 m={budget}", complete, 10 * confirm[key][budget]["complete_fraction"], 0)
            check("Table 9", f"{key} utility m={budget}", utility, confirm[key][budget]["mean_retrained_utility"])


# --------------------------------------------------------------------------- d = 12
D12 = {"linear": "Linear", "degree2_ridge": "Degree-2 ridge", "degree3_ridge": "Degree-3 ridge",
       "sparse_degree3_omp": "Degree-3 OMP", "rbf_kernel_ridge": "RBF", "cp": "TNDM-CP", "tt": "TNDM-TT",
       "ht": "Tree TNDM", "degree_ladder_tt": "TT degree ladder", "linear_plus_cp": "Linear + CP",
       "linear_plus_tt": "Linear + TT", "linear_plus_ht": "Linear + HT"}


def table_7_8_10() -> None:
    summary = pd.read_csv(EXP / "reasoning/d12/analysis/size3_method_summary.csv")
    summary = summary[summary.seed.isin([43, 59])]
    paper7 = {"linear": (0.053, 0.054, 0.083, 0.050, 0.052, 0.055),
              "degree2_ridge": (0.057, 0.045, 0.049, 0.044, 0.042, 0.052),
              "degree3_ridge": (0.081, 0.143, 0.200, 0.260, 0.289, 0.318),
              "sparse_degree3_omp": (0.071, 0.094, 0.074, 0.083, 0.101, 0.093),
              "rbf_kernel_ridge": (0.045, 0.043, 0.071, 0.071, 0.146, 0.089),
              "tt": (0.081, 0.120, 0.173, 0.305, 0.423, 0.762),
              "degree_ladder_tt": (0.078, 0.143, 0.200, 0.260, 0.406, 0.306),
              "linear_plus_tt": (0.056, 0.122, 0.096, 0.271, 0.540, 0.668)}
    for key, values in paper7.items():
        for budget, paper in zip((32, 48, 64, 96, 128, 160), values):
            rows = summary[(summary.method == key) & (summary.new_budget == budget)]
            check("Table 7", f"{D12[key]} AP m={budget}", paper, float(rows.complete_interaction_ap.mean()))
    for key, paper in {"cp": 0.313, "ht": 0.214, "linear_plus_cp": 0.328, "linear_plus_ht": 0.114}.items():
        rows = summary[(summary.method == key) & (summary.new_budget == 160)]
        check("Table 7", f"{D12[key]} AP m=160", paper, float(rows.complete_interaction_ap.mean()))

    slices = pd.read_csv(EXP / "reasoning/d12/analysis/slice_summary.csv")
    slices = slices[(slices.seed.isin([43, 59])) & (slices.slice == 4)]
    paper8 = {"linear": (0.650, 0.633, -4.187), "degree2_ridge": (0.900, 0.950, -0.585),
              "degree3_ridge": (1.000, 1.000, -0.001), "sparse_degree3_omp": (0.500, 0.500, -5.546),
              "rbf_kernel_ridge": (1.000, 1.000, -0.003), "cp": (0.850, 0.883, -1.336), "tt": (1.000, 0.983, -0.173),
              "ht": (0.650, 0.583, -4.717), "degree_ladder_tt": (1.000, 1.000, -0.001),
              "linear_plus_cp": (0.800, 0.850, -1.704), "linear_plus_tt": (0.950, 0.917, -0.967),
              "linear_plus_ht": (0.700, 0.683, -3.596)}
    for key, (top1, top3, utility) in paper8.items():
        rows = slices[slices.method == key]
        check("Table 8", f"{D12[key]} top-1 completeness", top1, float(rows.top1_complete_rate.mean()))
        check("Table 8", f"{D12[key]} top-3 completeness", top3, float(rows.top3_complete_fraction.mean()))
        check("Table 8", f"{D12[key]} top-3 utility", utility, float(rows.top3_actual_utility.mean()))

    selection = pd.read_csv(EXP / "reasoning/d12/analysis/size3_attribution_selection.csv")
    selection = selection[(selection.seed.isin([43, 59])) & (selection.new_budget == 160)
                          & (selection.criterion == "shapley")]
    paper10 = {"linear_plus_tt": (60.0, 14.1, -4.568, 1.582), "cp": (45.0, 7.1, -5.989, 0.770),
               "degree2_ridge": (45.0, 7.1, -6.354, 0.783), "linear_plus_cp": (45.0, 7.1, -6.418, 0.923),
               "linear_plus_ht": (40.0, 0.0, -6.891, 0.051), "sparse_degree3_omp": (40.0, 14.1, -6.510, 1.528),
               "tt": (40.0, 14.1, -6.714, 1.504), "degree_ladder_tt": (35.0, 7.1, -7.445, 0.685),
               "linear": (35.0, 7.1, -7.450, 0.842), "ht": (30.0, 14.1, -8.042, 1.964),
               "degree3_ridge": (20.0, 14.1, -9.092, 1.887), "rbf_kernel_ridge": (20.0, 0.0, -9.180, 0.137)}
    for key, (complete, complete_sd, utility, utility_sd) in paper10.items():
        per_seed = selection[selection.method == key].groupby("seed").agg(
            complete=("complete", "mean"), utility=("actual_utility", "mean"))
        mean, sd = mean_sd(100 * per_seed.complete)
        check("Table 10 / Fig. 3b", f"{D12[key]} complete % mean", complete, mean, 1)
        check("Table 10 / Fig. 3b", f"{D12[key]} complete % sd", complete_sd, sd, 1)
        mean, sd = mean_sd(per_seed.utility)
        check("Table 10 / Fig. 3b", f"{D12[key]} utility mean", utility, mean)
        check("Table 10 / Fig. 3b", f"{D12[key]} utility sd", utility_sd, sd)


def table_18() -> None:
    d8 = pd.read_csv(EXP / "reasoning/d8/lds/lds_per_fit_d8.csv")
    d8 = d8[d8.budget == 128].groupby(["family", "repeat"])[["lds_size4", "native_size4"]].mean()
    paper8 = {"linear": (0.341, 0.053, 0.341, 0.053), "degree2_ridge": (0.377, 0.055, 0.730, 0.023),
              "degree3_ridge": (0.381, 0.057, 0.735, 0.023), "rbf_kernel": (0.386, 0.054, 0.738, 0.025),
              "cp": (0.282, 0.135, 0.571, 0.108), "tt": (0.358, 0.063, 0.697, 0.046),
              "ht": (0.403, 0.047, 0.755, 0.033), "tt_ladder": (0.376, 0.050, 0.734, 0.021)}
    for key, (lds, lds_sd, direct, direct_sd) in paper8.items():
        rows = d8.loc[key]
        check("Table 18 (d=8)", f"{key} LDS mean", lds, float(rows.lds_size4.mean()))
        check("Table 18 (d=8)", f"{key} LDS sd", lds_sd, float(rows.lds_size4.std(ddof=1)))
        check("Table 18 (d=8)", f"{key} direct mean", direct, float(rows.native_size4.mean()))
        check("Table 18 (d=8)", f"{key} direct sd", direct_sd, float(rows.native_size4.std(ddof=1)))
    d12 = pd.read_csv(EXP / "reasoning/d8/lds/lds_size3_d12.csv")
    d12 = d12[(d12.new_budget == 160) & (d12.seed.isin([43, 59]))].groupby(["family", "seed"])[["lds", "native"]].mean()
    paper12 = {"linear": (0.032, 0.021, 0.032, 0.021), "degree2_ridge": (0.026, 0.018, 0.014, 0.026),
               "degree3_ridge": (0.031, 0.009, 0.037, 0.020), "rbf_kernel_ridge": (0.022, 0.003, 0.045, 0.013),
               "cp": (0.037, 0.001, 0.035, 0.034), "tt": (0.031, 0.001, 0.164, 0.008),
               "ht": (0.020, 0.024, 0.030, 0.019), "degree_ladder_tt": (0.020, 0.007, 0.024, 0.023)}
    for key, (lds, lds_sd, direct, direct_sd) in paper12.items():
        rows = d12.loc[key]
        check("Table 18 (d=12)", f"{key} LDS mean", lds, float(rows.lds.mean()))
        check("Table 18 (d=12)", f"{key} LDS sd", lds_sd, float(rows.lds.std(ddof=1)))
        check("Table 18 (d=12)", f"{key} direct mean", direct, float(rows.native.mean()))
        check("Table 18 (d=12)", f"{key} direct sd", direct_sd, float(rows.native.std(ddof=1)))


# --------------------------------------------------------------------------- fine-tuning
def table_2_13() -> None:
    saved = json.loads((EXP / "sport_music/results/FITTING_COMPUTE_mean_gold_logprob.json").read_text())
    paper = {"linear (tuned)": (0.579, 0.048), "matched (tuned)": (0.556, 0.054), "degree-2 (tuned)": (0.565, 0.109),
             "CP single fit, R=5": (0.668, 0.084), "CP best-of-8, R=5": (0.616, 0.133),
             "CP ENSEMBLE 8 seeds": (0.719, 0.070), "CP ENSEMBLE ranks 3-6": (0.734, 0.061),
             "CP ENSEMBLE rank x l2": (0.706, 0.059)}
    for key, (mean, sd) in paper.items():
        check("Table 2 / 13", f"{key} R2 mean", mean, saved[key]["r2"])
        check("Table 2 / 13", f"{key} R2 sd", sd, saved[key]["sd"])


def table_3_14() -> None:
    frames = []
    for path in sorted((EXP / "attribution/results/question_a").glob("agnews_lora_r*_seed*.csv")):
        frame = pd.read_csv(path, usecols=["method", "method_type", "design", "budget", "banzhaf_rel_l2",
                                           "shapley_rel_l2"])
        frames.append(frame[frame.budget == 128])
    rows = pd.concat(frames)
    surrogate = rows[(rows.method_type == "datamodel") & (rows.design == "uniform")]
    paper = {"linear": (0.1701, 0.3555), "degree2_ridge": (0.2013, 0.3721), "rbf": (0.1724, 0.3368),
             "cp": (0.1736, 0.3358), "tt": (0.1634, 0.3285)}
    for key, (banzhaf, shapley) in paper.items():
        sub = surrogate[surrogate.method == key]
        check("Table 3 / 14", f"{key} Banzhaf median (n={len(sub)})", banzhaf, float(sub.banzhaf_rel_l2.median()), 4)
        check("Table 3 / 14", f"{key} Shapley median (n={len(sub)})", shapley, float(sub.shapley_rel_l2.median()), 4)
    direct = rows[rows.method_type == "direct"]
    check("Table 14", "MSR Banzhaf median", 0.2258, float(direct[direct.method == "msr_banzhaf"].banzhaf_rel_l2.median()), 4)
    check("Table 14", "KernelSHAP Shapley median", 0.1233, float(direct[direct.method == "kernel_shap"].shapley_rel_l2.median()), 4)


def figure_4() -> None:
    """Fig. 4 claim: at 64 observations TNDM-CP has the lowest mean error for all three quantities."""
    base = EXP / "modular_retraining/tt_additive_init/data"
    stats = pd.concat([pd.read_csv(base / "plot_statistics_all_metrics.csv")])
    plotted = ["linear", "degree2_ridge", "degree3_omp", "rbf", "cp", "tt_additive_init"]
    for metric in ("shapley_rel_l2", "sii_rel_l2", "sii3_rel_l2"):
        rows = stats[(stats.metric == metric) & (stats.budget == 64) & (stats.method.isin(plotted))]
        best = rows.loc[rows.mean_of_seed_means.idxmin(), "method"]
        ROWS.append(("Fig. 4", f"lowest mean {metric} at m=64 among plotted methods", "cp", str(best), best == "cp"))


def table_16_17() -> None:
    """Table 16 (d = 500 in-context prediction) and the UCI HAR panel of Table 17."""
    sys.path.insert(0, str(EXP / "appendix_tables"))
    from summarize import table16, table17_uci_har
    paper16 = {  # (coin flip 400, coin flip 800, GSM8K 400, GSM8K 800), each (mean, sd)
        "Linear": ((0.1779, 0.0123), (0.1853, 0.0002), (0.3262, 0.0113), (0.2854, 0.0105)),
        "RBF": ((0.1755, 0.0131), (0.1679, 0.0078), (0.2885, 0.0057), (0.2645, 0.0142)),
        "Coefficient CP": ((0.1754, 0.0154), (0.1674, 0.0091), (0.2890, 0.0087), (0.2760, 0.0123)),
        "TT, random": ((0.1732, 0.0174), (0.1680, 0.0077), (0.2874, 0.0093), (0.2758, 0.0121)),
        "TT, semantic": ((0.1733, 0.0174), (0.1671, 0.0088), (0.2887, 0.0113), (0.2760, 0.0119)),
        "Degree-graded TT, random": ((0.1750, 0.0156), (0.1685, 0.0124), (0.2879, 0.0100), (0.2761, 0.0116)),
        "Degree-graded TT, semantic": ((0.1750, 0.0156), (0.1678, 0.0086), (3.2994, 0.3212), (1.1528, 1.1618)),
    }
    rows = table16()
    for label, cells in paper16.items():
        for (task, n_fit), (mean, sd) in zip((("coin_flip", 400), ("coin_flip", 800), ("gsm8k", 400), ("gsm8k", 800)), cells):
            row = rows[(rows.surrogate == label) & (rows.task == task) & (rows.n_fit == n_fit)].iloc[0]
            check("Table 16", f"{label} {task} n_fit={n_fit} mean", mean, row["mean"], 4)
            check("Table 16", f"{label} {task} n_fit={n_fit} sd", sd, row["sd"], 4)
    paper17 = {
        ("Shapley", "Linear"): (0.047190, 0.000116), ("Shapley", "RBF"): (0.044079, 0.000344),
        ("Shapley", "TT"): (0.041886, 0.000704), ("Banzhaf", "Linear"): (0.001240, 0.000105),
        ("Banzhaf", "RBF"): (0.001132, 0.000062), ("Banzhaf", "TT"): (0.000983, 0.000072),
        ("Pairwise interaction", "Linear"): (0.049988, 0.000000), ("Pairwise interaction", "RBF"): (0.047985, 0.000215),
        ("Pairwise interaction", "TT"): (0.045343, 0.000602),
    }
    rows = table17_uci_har()
    for (attribution, surrogate), (mean, sd) in paper17.items():
        row = rows[(rows.attribution == attribution) & (rows.surrogate == surrogate)].iloc[0]
        check("Table 17 (UCI HAR)", f"{attribution} {surrogate} mean", mean, row["mean"], 6)
        check("Table 17 (UCI HAR)", f"{attribution} {surrogate} sd", sd, row["sd"], 6)


def table_17_19_probe() -> None:
    """Qwen-probe panel of Table 17 and Table 19, from the saved attribution errors."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("probe_summary", EXP / "qwen_probe/summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paper17 = {
        ("Shapley", "Linear"): (0.067948, 0.001012), ("Shapley", "RBF"): (0.009971, 0.002894),
        ("Shapley", "TT"): (0.055446, 0.007836), ("Banzhaf", "Linear"): (0.033136, 0.002820),
        ("Banzhaf", "RBF"): (0.005382, 0.000884), ("Banzhaf", "TT"): (0.018738, 0.006185),
        ("Leave-one-out", "Linear"): (0.309582, 0.004989), ("Leave-one-out", "RBF"): (0.009050, 0.007205),
        ("Leave-one-out", "TT"): (0.077350, 0.024296), ("Pairwise interaction", "Linear"): (0.229971, 0.000743),
        ("Pairwise interaction", "RBF"): (0.019093, 0.004560), ("Pairwise interaction", "TT"): (0.062223, 0.008455),
    }
    rows = module.table17_probe()
    for (attribution, surrogate), (mean, sd) in paper17.items():
        row = rows[(rows.attribution == attribution) & (rows.surrogate == surrogate)].iloc[0]
        check("Table 17 (Qwen probe)", f"{attribution} {surrogate} mean", mean, row["mean"], 6)
        check("Table 17 (Qwen probe)", f"{attribution} {surrogate} sd", sd, row["sd"], 6)
    paper19 = {"Linear": ("1.13e-14", "8.23e-17"), "Coefficient CP": ("1.33e-15", "5.83e-16"),
               "TT": ("4.77e-15", "4.50e-15")}
    rows = module.table19()
    for surrogate, (singleton, pairwise) in paper19.items():
        row = rows[rows.surrogate == surrogate].iloc[0]
        for column, paper in (("singleton", singleton), ("pairwise", pairwise)):
            value = f"{row[column]:.2e}"
            ROWS.append(("Table 19", f"{surrogate} {column}", paper, value, value == paper))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", help="exit with status 1 if any value disagrees")
    parser.add_argument("--only-mismatches", action="store_true")
    args = parser.parse_args()
    for section in (table_1_5_6_9, table_7_8_10, table_18, table_2_13, table_3_14, figure_4, table_16_17, table_17_19_probe):
        section()
    width = max(len(label) for _, label, *_ in ROWS)
    current = None
    for table, label, paper, value, ok in ROWS:
        if args.only_mismatches and ok:
            continue
        if table != current:
            print(f"\n== {table}")
            current = table
        print(f"  {label:<{width}}  paper {paper:>9}  recomputed {value:>12}  {'ok' if ok else 'MISMATCH'}")
    bad = [row for row in ROWS if not row[4]]
    print(f"\n{len(ROWS) - len(bad)} of {len(ROWS)} values agree with the paper at printed precision; "
          f"{len(bad)} do not.")
    return 1 if (bad and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
