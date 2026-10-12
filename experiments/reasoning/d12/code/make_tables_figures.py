#!/usr/bin/env python3
"""Tables and the selection figure for d=12 study.

Byte-stable output: CSV writers use "\\n", and the PDF/PNG metadata drops the creation
date, so regenerating does not dirty the tree (see `CONVENTIONS.md`).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-8 of the validated reference palette, assigned in fixed order and
# never cycled; identity is carried by the legend and by direct labels, never by colour
# alone.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK = "#1a1a19"
MUTED = "#6b6a63"
GRID = "#e3e2dc"

FIGURE_METHODS = ("tt", "degree3_ridge", "degree2_ridge", "rbf_kernel_ridge",
                  "sparse_degree3_omp", "linear", "influence_loo", "random")
LABELS = {"tt": "TT", "degree3_ridge": "degree-3 ridge", "degree2_ridge": "degree-2 ridge",
          "rbf_kernel_ridge": "RBF kernel ridge", "sparse_degree3_omp": "sparse degree-3 (OMP)",
          "linear": "Linear", "influence_loo": "influence (LOO)", "random": "random",
          "cardinality": "cardinality-only", "cp": "CP", "ht": "HT",
          "degree_ladder_tt": "TT degree ladder", "linear_plus_tt": "Linear + TT",
          "linear_plus_cp": "Linear + CP", "linear_plus_ht": "Linear + HT",
          "tt_middle": "TT middle band", "tt_bottom": "TT bottom band"}
NOISE_LABELS = {"nondeterminism_same_seed_same_gpu": "rerun, same seed and GPU",
                "initialisation": "initialisation", "data_order": "data order",
                "backend_v100_vs_rtx8000": "V100 vs RTX 8000",
                "backend_a6000_vs_rtx8000": "A6000 vs RTX 8000"}
TABLE_METHOD_ORDER = ("cardinality", "linear", "influence_loo", "random",
                      "degree2_ridge", "degree3_ridge", "sparse_degree3_omp", "rbf_kernel_ridge",
                      "cp", "tt", "ht", "degree_ladder_tt", "linear_plus_tt",
                      "tt_middle", "tt_bottom")


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def latex_table(rows: list[list[str]], header: list[str], caption: str, label: str,
                align: str, size: str = "small") -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return (f"\\begin{{table}}[t]\n\\centering\n\\{size}\n\\setlength{{\\tabcolsep}}{{4pt}}\n"
            f"\\begin{{tabular}}{{{align}}}\n"
            f"\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n{body}\n"
            f"\\bottomrule\n\\end{{tabular}}\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n\\end{{table}}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--here", type=Path, required=True)
    parser.add_argument("--primary-slice", type=int, default=4)
    parser.add_argument("--confirmatory-seeds", type=int, nargs="+", default=(43, 59))
    args = parser.parse_args()
    analysis, tables, figures = args.here / "analysis", args.here / "tables", args.here / "figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)

    summary = read_csv(analysis / "slice_summary.csv")
    for row in summary:
        row["seed"] = int(row["seed"]); row["slice"] = int(row["slice"])
        for key in ("top1_complete_rate", "top3_complete_fraction",
                    "top1_actual_utility", "top3_actual_utility"):
            row[key] = float(row[key])
    ceiling = {}
    ceiling_path = args.here / "noise_ceiling" / "ceiling.json"
    if ceiling_path.exists():
        ceiling = json.loads(ceiling_path.read_text())
    confirmatory = {}
    confirmatory_path = analysis / "confirmatory.json"
    if confirmatory_path.exists():
        confirmatory = json.loads(confirmatory_path.read_text())

    seeds = sorted({row["seed"] for row in summary})
    slices = sorted({row["slice"] for row in summary})
    methods = [m for m in TABLE_METHOD_ORDER if any(row["method"] == m for row in summary)]
    methods += sorted({row["method"] for row in summary} - set(methods))

    # ---- main table: seed x slice x method ------------------------------------
    main_rows = []
    for method in methods:
        for size in slices:
            for seed in seeds:
                match = [r for r in summary if r["method"] == method and r["slice"] == size
                         and r["seed"] == seed]
                if not match:
                    continue
                row = match[0]
                main_rows.append({"method": method, "slice": size, "seed": seed,
                    "blind": seed in args.confirmatory_seeds,
                    "queries": row["queries"],
                    "top1_complete_rate": f"{row['top1_complete_rate']:.4f}",
                    "top3_complete_fraction": f"{row['top3_complete_fraction']:.4f}",
                    "top1_actual_utility": f"{row['top1_actual_utility']:.6f}",
                    "top3_actual_utility": f"{row['top3_actual_utility']:.6f}"})
    write_csv(tables / "d12_selection_by_seed_and_slice.csv", main_rows)

    # ---- LaTeX: every slice, top-3 complete count and utility per seed ---------
    wide_methods = [m for m in FIGURE_METHODS if any(r["method"] == m for r in summary)]
    wide_rows = []
    for method in wide_methods:
        row = [LABELS.get(method, method)]
        for size in slices:
            parts = []
            for seed in seeds:
                match = [r for r in summary if r["method"] == method and r["slice"] == size
                         and r["seed"] == seed]
                parts.append(f"{match[0]['top3_actual_utility']:.1f}".replace("-0.0", "0.0") if match else "--")
            row.append("/".join(parts))
        wide_rows.append(row)
    (tables / "d12_selection_by_seed_and_slice.tex").write_text(latex_table(
        wide_rows, ["Method"] + [f"$|S|={k}$" for k in slices],
        "Mean target log-probability (nats) of each method's frozen top-3 nominations after "
        "retraining, per slice, listed as seed " + "/".join(map(str, seeds)) + ". Seeds "
        + ", ".join(map(str, args.confirmatory_seeds)) + " are blind; seed 31 is not. "
        f"$|S|={args.primary_slice}$ is the pre-registered primary slice. Rounded to one decimal; "
        "exact values are in the accompanying CSV.",
        "tab:d12-by-seed-slice", "l" + "c" * len(slices), size="footnotesize"))

    # ---- LaTeX: primary slice, confirmatory seeds pooled, seed 31 shown apart --
    latex_rows = []
    for method in methods:
        cells = [r for r in summary if r["method"] == method and r["slice"] == args.primary_slice]
        blind = [r for r in cells if r["seed"] in args.confirmatory_seeds]
        nonblind = [r for r in cells if r["seed"] not in args.confirmatory_seeds]
        if not blind and not nonblind:
            continue
        def fmt(values: list[float], digits: int = 3) -> str:
            return f"{np.mean(values):.{digits}f}" if values else "--"
        latex_rows.append([
            LABELS.get(method, method.replace("_", " ")),
            fmt([r["top1_complete_rate"] for r in blind]),
            fmt([r["top3_complete_fraction"] for r in blind]),
            fmt([r["top3_actual_utility"] for r in blind]),
            fmt([r["top3_actual_utility"] for r in nonblind]),
        ])
    caption = (f"Fixed-size training-set selection at the pre-registered primary slice "
               f"$|S|={args.primary_slice}$, $d=12$ transformer retraining. Every value is the "
               f"outcome of retraining the frozen nominees from scratch. The first three "
               f"columns pool the two pre-registered confirmatory learner seeds "
               f"({', '.join(map(str, args.confirmatory_seeds))}); the last column is the "
               f"original seed 31, which is not blind to this analysis and is reported "
               f"separately. Utility is mean target log-probability in nats.")
    (tables / "d12_selection_primary_slice.tex").write_text(latex_table(
        latex_rows,
        [r"Method", r"top-1 compl.", r"top-3 compl.", r"top-3 util.", r"top-3 util. (31)"],
        caption, "tab:d12-primary-slice", "lrrrr"))

    # ---- confirmatory comparison table ----------------------------------------
    if confirmatory:
        rows, tex = [], []
        for name, entry in sorted(confirmatory.get("comparisons", {}).items()):
            low, high = entry["bootstrap_95_interval"]
            rows.append({"id": name, "comparison": entry["comparison"], "kind": entry["kind"],
                         "n_units": entry["n_units"],
                         "mean_difference": f"{entry['mean_difference']:.6f}",
                         "ci_low": f"{low:.6f}", "ci_high": f"{high:.6f}",
                         "unit_win_rate": f"{entry['unit_win_rate']:.3f}",
                         "status": entry.get("status", "")})
            left_name, right_name = (part.strip() for part in entry["comparison"].split(" - "))
            pretty = f"{LABELS.get(left_name, left_name)} $-$ {LABELS.get(right_name, right_name)}"
            tex.append([name.replace("_", r"\_"), pretty,
                        f"{entry['mean_difference']:.3f}", f"[{low:.3f}, {high:.3f}]",
                        entry.get("status", "") or "--"])
        if rows:
            write_csv(tables / "d12_confirmatory_comparisons.csv", rows)
            (tables / "d12_confirmatory_comparisons.tex").write_text(latex_table(
                tex, ["", "Comparison", "Diff.", "95\\% CI", "Verdict"],
                f"Pre-registered comparisons at $|S|={args.primary_slice}$, paired on "
                f"(query, learner seed) over the confirmatory seeds, 20{{,}}000 bootstrap draws. C1--C3 and "
                f"C5\\_family are superiority tests, C5 is descriptive. "
                f"C4 is an equivalence test against a margin of "
                f"{confirmatory.get('equivalence_margin') or float('nan'):.3f} nats: one pooled "
                f"retraining-noise SD of a single cell, divided by $\\sqrt{{3}}$ for a top-3 mean. "
                f"C5\\_family compares the best nonadditive with the best additive method, each chosen "
                f"on the same seeds.",
                "tab:d12-confirmatory", "llrll"))

    # ---- noise-ceiling table ---------------------------------------------------
    if ceiling:
        rows, tex = [], []
        for name, entry in sorted(ceiling["variance_components_log_probability"].items()):
            if not entry.get("n_pairs"):
                continue
            rows.append({"component": name, "n_pairs": entry["n_pairs"],
                         "mean_signed_shift": f"{entry['mean_signed_shift']:.6f}",
                         "mean_absolute_shift": f"{entry['mean_absolute_shift']:.6f}",
                         "max_absolute_shift": f"{entry['max_absolute_shift']:.6f}",
                         "implied_single_run_sd": f"{entry['implied_single_run_sd']:.6f}"})
            tex.append([NOISE_LABELS.get(name, name.replace("_", " ")), str(entry["n_pairs"]),
                        f"{entry['mean_signed_shift']:.3f}",
                        f"{entry['mean_absolute_shift']:.3f}",
                        f"{entry['implied_single_run_sd']:.3f}"])
        if rows:
            write_csv(tables / "d12_noise_ceiling.csv", rows)
            (tables / "d12_noise_ceiling.tex").write_text(latex_table(
                tex, ["Source", "pairs", "shift", "$|$shift$|$", "SD"],
                f"Retraining-noise ceiling for the $d=12$ game, in nats of query "
                f"log-probability. Mean ceiling $R^2$ "
                f"{ceiling['mean_ceiling_r2']:.3f}; pooled single-cell replicate SD "
                f"{ceiling['pooled_replicate_sd_single_cell']:.3f}. Shift is mean signed, SD the implied "
                f"single-run standard deviation.",
                "tab:d12-noise-ceiling", "lrrrr"))

    # ---- figure ----------------------------------------------------------------
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 4.0), gridspec_kw={"width_ratios": (1.15, 1)})
    figure.subplots_adjust(left=.2, right=.985, top=.93, bottom=.33, wspace=.38)

    shown = [m for m in FIGURE_METHODS if any(r["method"] == m for r in summary)]
    colors = {method: PALETTE[index] for index, method in enumerate(shown)}
    rows_y = {method: len(shown) - 1 - index for index, method in enumerate(shown)}

    from matplotlib.lines import Line2D
    if ceiling:
        margin = ceiling["equivalence_margin_top3_mean_utility"]
        best = max(r["top3_actual_utility"] for r in summary
                   if r["slice"] == args.primary_slice and r["seed"] in args.confirmatory_seeds)
        left.axvspan(best - margin, best + margin, color=GRID, zorder=0)
    for method in shown:
        y = rows_y[method]
        left.axhline(y, color=GRID, linewidth=.5, zorder=0)
        for offset, seed in zip((-.18, 0, .18), seeds):
            match = [r for r in summary if r["method"] == method and r["seed"] == seed
                     and r["slice"] == args.primary_slice]
            if not match:
                continue
            blind = seed in args.confirmatory_seeds
            left.plot([match[0]["top3_actual_utility"]], [y + offset],
                      marker="o" if blind else "s", markersize=5 if blind else 4.5,
                      markerfacecolor=colors[method] if blind else "white",
                      markeredgecolor=colors[method], markeredgewidth=1.1, linestyle="none", zorder=3)
    left.set_yticks([rows_y[m] for m in shown])
    left.set_yticklabels([LABELS.get(m, m) for m in shown])
    left.set_ylim(-.7, len(shown) - .3)
    left.set_xlabel("top-3 utility after retraining (nats)")
    left.set_title(f"(a) primary slice $|S|={args.primary_slice}$", loc="left", fontsize=8.5)
    for spine in ("top", "right", "left"):
        left.spines[spine].set_visible(False)
    left.tick_params(axis="y", length=0)
    key = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=MUTED, markeredgecolor=MUTED,
                  markersize=5, label="seeds 43, 59 (blind)"),
           Line2D([], [], marker="s", linestyle="none", markerfacecolor="white", markeredgecolor=MUTED,
                  markersize=4.5, label="seed 31 (not blind)")]
    if ceiling:
        from matplotlib.patches import Patch
        key.append(Patch(facecolor=GRID, edgecolor="none", label=f"best $\\pm$ noise margin ({margin:.2f})"))
    left.legend(handles=key, frameon=False, fontsize=6.5, loc="upper left",
                bbox_to_anchor=(-.02, -.17), ncol=1, handletextpad=.4)

    for method in shown:
        xs, ys = [], []
        for size in slices:
            match = [r for r in summary if r["method"] == method and r["slice"] == size
                     and r["seed"] in args.confirmatory_seeds]
            if match:
                xs.append(size); ys.append(float(np.mean([r["top1_complete_rate"] for r in match])))
        if xs:
            right.plot(xs, ys, marker="o", markersize=4, linewidth=2, color=colors[method],
                       label=LABELS.get(method, method), zorder=3)
    right.set_xticks(slices)
    right.set_xlabel("coalition size $|S|$")
    right.set_ylabel("top-1 structurally complete (rate)")
    right.set_ylim(-.04, 1.04)
    right.set_title("(b) completeness across slices, seeds 43 and 59", loc="left", fontsize=8.5)
    right.grid(axis="y", color=GRID, linewidth=.6, zorder=0)
    right.set_axisbelow(True)
    for spine in ("top", "right"):
        right.spines[spine].set_visible(False)
    right.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left",
                 bbox_to_anchor=(-.05, -.17), handlelength=1.1, columnspacing=.6)

    for extension in ("pdf", "png"):
        figure.savefig(figures / f"d12_selection.{extension}", dpi=220,
                       metadata={"CreationDate": None} if extension == "pdf" else None)
    plt.close(figure)
    print(json.dumps({"tables": sorted(p.name for p in tables.iterdir()),
                      "figures": sorted(p.name for p in figures.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
