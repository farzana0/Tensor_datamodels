#!/usr/bin/env python3
"""Render only: reconstruct three equal-budget study panels from audited historical summaries.

Run from any directory with the interpreter/dependencies documented in README:
PYTHONPATH=<site-packages> \
    /usr/bin/python3 -S plot_recovered.py

No experiments are run, no source values are changed, and no SD limits are clipped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = {
    "linear": ("Linear", "#777777", "o", "-"),
    "degree2_ridge": ("Quadratic", "#4C78A8", "s", "--"),
    "degree3_omp": ("Cubic OMP", "#D55E00", "P", "-."),
    "rbf": ("RBF", "#D8A633", "^", ":"),
    "cp": ("TNDM-CP", "#9467BD", "D", "--"),
    "tt": ("TNDM-TT", "#2A9D8F", "v", "-"),
}
METRICS = {
    "shapley_rel_l2": "(a) Shapley",
    "banzhaf_rel_l2": "(b) Banzhaf",
    "sii_rel_l2": "(c) Pairwise SII",
}
BUDGETS = (8, 16, 32, 64, 128)


def read_statistics(path: Path):
    rows = list(csv.DictReader(path.open(newline="")))
    indexed = {}
    for row in rows:
        if row["metric"] not in METRICS or row["method"] not in METHODS:
            continue
        key = row["metric"], row["method"], int(row["budget"])
        if key in indexed:
            raise ValueError(f"Duplicate summary: {key}")
        if int(row["n_learner_seeds"]) != 3 or int(row["repeats_per_seed"]) != 20:
            raise ValueError(f"Unexpected repetition structure: {key}")
        mean = float(row["mean_of_seed_means"])
        sd = float(row["sample_sd_across_seed_means"])
        lower, upper = float(row["sd_lower"]), float(row["sd_upper"])
        if not all(math.isfinite(v) for v in (mean, sd, lower, upper)) or sd < 0:
            raise ValueError(f"Invalid finite summary: {key}")
        if not (math.isclose(lower, mean - sd, abs_tol=1e-12) and
                math.isclose(upper, mean + sd, abs_tol=1e-12)):
            raise ValueError(f"SD endpoints inconsistent with saved mean and SD: {key}")
        indexed[key] = (mean, sd, lower, upper)
    expected = {(metric, method, budget) for metric in METRICS
                for method in METHODS for budget in BUDGETS}
    if set(indexed) != expected:
        raise ValueError(f"Unexpected grid; missing={expected-set(indexed)}, extra={set(indexed)-expected}")
    return indexed


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.7,
        "axes.edgecolor": "#555555", "axes.axisbelow": True,
        "lines.linewidth": 1.35, "lines.markersize": 4.1,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "modular-recovered",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def set_unclipped_limits(ax, lower: float, upper: float, scale: str):
    # All panels include zero and every signed mean +/- SD endpoint. Padding is
    # applied in displayed coordinates to keep full SD bands inside axes.
    lower, upper = min(0.0, lower), max(0.0, upper)
    transform = ax.yaxis.get_transform()
    endpoints = transform.transform(np.array([lower, upper], dtype=float))
    pad = max(float(endpoints[1] - endpoints[0]) * 0.065, 1e-8)
    limits = transform.inverted().transform(endpoints + np.array([-pad, pad]))
    # Do not introduce a large negative region when all observed endpoints >=0.
    if lower >= 0:
        limits[0] = 0.0
    ax.set_ylim(float(limits[0]), float(limits[1]))
    if scale == "symlog":
        ticks = [v for v in (-1000, -100, -10, -1, 0, 1, 10, 100, 1000)
                 if limits[0] <= v <= limits[1]]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
    actual = ax.get_ylim()
    if actual[0] > lower or actual[1] < upper:
        raise AssertionError("SD bounds would be clipped")
    return {"data_sd_min": lower, "data_sd_max": upper,
            "axis_min": actual[0], "axis_max": actual[1]}


def render(indexed, out_dir: Path, scale: str):
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.9))
    fig.subplots_adjust(left=0.073, right=0.993, bottom=0.19,
                        top=0.76, wspace=0.43)
    handles = [Line2D([], [], color=color, marker=marker, linestyle=line,
                      label=label, linewidth=1.35, markersize=4.1)
               for label, color, marker, line in METHODS.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.515, 0.995),
               ncol=6, frameon=False, handlelength=2, handletextpad=0.45,
               columnspacing=1.55, borderaxespad=0)
    panel_limits = {}
    for ax, (metric, title) in zip(axes, METRICS.items()):
        ax.set_xscale("log", base=2)
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=1.0, linscale=1.0, base=10)
        data_lower, data_upper = [], []
        for method, (label, color, marker, line) in METHODS.items():
            series = np.array([indexed[metric, method, budget] for budget in BUDGETS])
            mean, sd, lower, upper = series.T
            data_lower.extend(lower)
            data_upper.extend(upper)
            ax.fill_between(BUDGETS, lower, upper, color=color, alpha=0.12,
                            linewidth=0, zorder=2)
            ax.plot(BUDGETS, mean, label=label, color=color,
                    marker=marker, linestyle=line, markeredgewidth=0.5,
                    markeredgecolor="white", zorder=3)
        ax.set_title(title, loc="left", pad=8)
        ax.set_xlabel("Observation budget", labelpad=4)
        ax.set_ylabel("Relative L2 error" + (" (symlog)" if scale == "symlog" else ""),
                      labelpad=3)
        ax.set_xlim(7.1, 144.0)
        ax.set_xticks(BUDGETS, [str(value) for value in BUDGETS])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(direction="out", length=3, width=0.65, pad=3)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.55)
        panel_limits[metric] = set_unclipped_limits(
            ax, min(data_lower), max(data_upper), scale)
    stem = out_dir / f"modular_three_panel_{scale}"
    for extension in ("pdf", "svg", "png"):
        extra = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
        fig.savefig(stem.with_suffix("." + extension), dpi=300, **extra)
    plt.close(fig)
    return panel_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, default=ROOT.parent / "historical/data/plot_statistics.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    indexed = read_statistics(args.statistics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    limits = {scale: render(indexed, args.output_dir, scale) for scale in ("symlog", "linear")}
    caption = """# Recovered historical equal-budget study modular attribution figure

Relative L2 estimation error versus observed-coalition budget for (a) Shapley values,
(b) Banzhaf values, and (c) pairwise Shapley interaction indices (SII), in the d=8
modular relation-table retraining game. Each point first averages the 20 matched
coalition-sampling repetitions within each learner seed, then averages the three
learner-seed means (seeds 31, 43, and 59). Light translucent bands show plus/minus one sample
standard deviation across those three means, not confidence intervals and not the
standard deviation of 60 pooled repetitions. Methods use matched repetitions.

The x-axis is logarithmic (base 2) with every observed budget labeled. The primary
figure uses a symmetric logarithmic y-axis, linear on [-1, 1], to retain the full
mean-minus-SD to mean-plus-SD interval including negative lower bounds. Negative
bounds reflect symmetric SD bands; individual relative errors are nonnegative.
No SD endpoint is clipped. A companion version uses an ordinary linear y-axis and
the same means, SDs, budgets, colors, and markers.

This reconstructs historical results; it does not include new linear-initialized
fits. The large low-budget TT means and SDs are shown in full. The statistics
are read from data/plot_statistics.csv without fitting or changing any result.
The DejaVu Sans visual style is a documented fallback: the requested v64/HAR-SD
style source and reference PDF were unavailable, so exact style matching is not
verified. No overall figure title, in-panel caption, or footer is embedded.
"""
    (args.output_dir / "modular_three_panel_caption.md").write_text(caption)
    manifest = {"statistics": str(args.statistics.resolve()),
                "statistics_sha256": hashlib.sha256(args.statistics.read_bytes()).hexdigest(),
                "n_summary_rows": len(indexed), "figure_size_inches": [8.5, 2.9],
                "xscale": "log2", "symlog_linthresh": 1.0,
                "font_family": "DejaVu Sans", "all_font_sizes_pt": 9,
                "uncertainty_display": "translucent band", "band_alpha": 0.12,
                "methods": METHODS, "panel_limits": limits,
                "uncertainty": "sample SD across 3 learner-seed means, each averaging 20 repetitions",
                "exact_requested_style_verified": False}
    (args.output_dir / "modular_three_panel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered {len(indexed)} historical summaries in symlog and linear versions to {args.output_dir}")


if __name__ == "__main__":
    main()
