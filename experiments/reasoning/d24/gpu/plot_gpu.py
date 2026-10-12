#!/usr/bin/env python3
"""Figure for the targeted GPU arm: test nMSE vs retraining budget (reads tables/gpu_d24_nmse.csv).

Four series in the pilot's fixed palette order (TT, degree-3 ridge, RBF, sparse degree-3),
marker shapes as a second encoding, line = mean over the three learner seeds, band =
min-max over seeds.  A dashed guide marks TT's m=512 error so the pre-registered
half-budget comparison (TT@512 vs rivals@1024) can be read directly.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
SERIES = (
    ("tt", "TT (direct)", "#2a78d6", "o"),
    ("degree3_ridge", "Degree-3 ridge", "#eb6834", "^"),
    ("rbf_krr", "RBF kernel ridge", "#1baf7a", "D"),
    ("degree3_omp", "Sparse degree-3 (OMP)", "#eda100", "s"),
)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    rows = list(csv.DictReader((HERE / "tables/gpu_d24_nmse.csv").open()))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.grid(True, color=GRID, linewidth=.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    tt512 = None
    for method, label, colour, marker in SERIES:
        pts = sorted((int(r["budget"]), [float(r[f"nmse_seed{s}"]) for s in (31, 43, 59)])
                     for r in rows if r["method"] == method)
        xs = [b for b, _ in pts]
        means = [sum(v) / 3 for _, v in pts]
        ax.fill_between(xs, [min(v) for _, v in pts], [max(v) for _, v in pts], color=colour, alpha=.15, linewidth=0)
        ax.plot(xs, means, color=colour, linewidth=2, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=.8, label=label)
        if method == "tt":
            tt512 = dict(zip(xs, means)).get(512)
    if tt512 is not None:
        ax.axhline(tt512, color=MUTED, linewidth=.8, linestyle=(0, (3, 2)))
        ax.annotate("TT at m=512", (256, tt512), xytext=(0, 4), textcoords="offset points",
                    fontsize=7, color=MUTED)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([256, 512, 1024]); ax.set_xticklabels(["256", "512", "1024"])
    ax.set_xlabel("retrainings m", fontsize=9, color=MUTED)
    ax.set_ylabel("held-out nMSE (256 real cells)", fontsize=9, color=MUTED)
    ax.set_title("Real transformer retraining, k=2, g=3, d=24", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/gpu_d24_nmse_vs_budget.pdf", metadata={"CreationDate": None}, bbox_inches="tight")
    fig.savefig(HERE / "figures/gpu_d24_nmse_vs_budget.png", dpi=200, bbox_inches="tight")
    print("figure written")


if __name__ == "__main__":
    main()
