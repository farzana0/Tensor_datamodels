#!/usr/bin/env python3
"""Pilot figures from the aggregated tables (run `aggregate.py` first).

1. `figures/pilot_sample_complexity_vs_d.pdf`: budget needed to reach nMSE <= 0.10
   against d, one panel per (k, g), primary noise level only.
2. `figures/pilot_nmse_curves.pdf`: nMSE against budget, rows = (k, g), cols = d.

Series use categorical slots 1-4 of the reference palette in a fixed order, with
distinct marker shapes as a second encoding so identity never rests on colour.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
TAU = 0.10
SERIES = (  # fixed order: method, label, colour slot, marker
    ("ladder_tt", "TT degree-ladder", "#2a78d6", "o"),
    ("tt", "TT (direct)", "#eb6834", "s"),
    ("degree3_ridge", "Degree-3 ridge", "#1baf7a", "^"),
    ("rbf_krr", "RBF kernel ridge", "#eda100", "D"),
)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CEILING = 2048  # plotting position for "not reached by 1024"


def style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def save(fig, stem: str) -> None:
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / f"figures/{stem}.pdf", metadata={"CreationDate": None}, bbox_inches="tight")
    fig.savefig(HERE / f"figures/{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    complexity = list(csv.DictReader((HERE / "tables/pilot_sample_complexity.csv").open()))
    curves = list(csv.DictReader((HERE / "tables/pilot_budget_curves.csv").open()))
    structures = sorted({(int(r["k"]), int(r["g"])) for r in complexity})
    dims = sorted({int(r["d"]) for r in complexity})

    # Figure 1: sample complexity vs d.
    stars = defaultdict(dict)
    for row in complexity:
        if float(row["noise_multiplier"]) != 1.0:
            continue
        key = (int(row["k"]), int(row["g"]), row["method"])
        stars[key][int(row["d"])] = int(row["m_star"]) if row["m_star"] else math.inf
    columns = 4
    rows = math.ceil(len(structures) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(1.9 * columns, 1.7 * rows), sharex=True, sharey=True,
                             squeeze=False)
    for index, (k, g) in enumerate(structures):
        ax = axes[index // columns][index % columns]
        style(ax)
        for position, (method, label, colour, marker) in enumerate(SERIES):
            points = stars.get((k, g, method), {})
            dodge = (position - 1.5) * .9  # tied values stay visible side by side
            xs = [d + dodge for d in dims if d in points]
            ys = [CEILING if math.isinf(points[d]) else points[d] for d in dims if d in points]
            ax.plot(xs, ys, color=colour, linewidth=2, marker=marker, markersize=5,
                    markerfacecolor=colour, markeredgecolor="white", markeredgewidth=.8, label=label)
            for x, y in zip(xs, ys):
                if y == CEILING:
                    ax.plot([x], [y], marker=marker, markersize=5, markerfacecolor="white",
                            markeredgecolor=colour, linestyle="none")
        ax.set_yscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_yticks([32, 128, 512, CEILING])
        ax.set_yticklabels(["32", "128", "512", ">1024"])
        ax.set_title(f"k={k}, g={g}  (degree {k * g})", fontsize=8, color=INK)
    for index in range(len(structures), rows * columns):
        axes[index // columns][index % columns].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("groups d", fontsize=8, color=MUTED)
    for row in axes:
        row[0].set_ylabel(f"retrainings to nMSE<={TAU}", fontsize=8, color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(.5, 1.0))
    save(fig, "pilot_sample_complexity_vs_d")

    # Figure 2: nMSE curves.
    by_arm = defaultdict(lambda: defaultdict(list))
    for row in curves:
        if float(row["noise_multiplier"]) != 1.0:
            continue
        by_arm[(int(row["k"]), int(row["g"]), int(row["d"]))][row["method"]].append(
            (int(row["budget"]), float(row["nmse_mean"]), float(row["nmse_sd"])))
    fig, axes = plt.subplots(len(structures), len(dims), figsize=(2.1 * len(dims), 1.5 * len(structures)),
                             sharex=True, sharey=True, squeeze=False)
    for r, (k, g) in enumerate(structures):
        for c, d in enumerate(dims):
            ax = axes[r][c]
            style(ax)
            ax.axhline(TAU, color=MUTED, linewidth=.8, linestyle=(0, (3, 2)))
            for method, label, colour, marker in SERIES:
                points = sorted(by_arm.get((k, g, d), {}).get(method, []))
                if not points:
                    continue
                xs, ys, sds = zip(*points)
                ax.plot(xs, ys, color=colour, linewidth=2, marker=marker, markersize=4,
                        markeredgecolor="white", markeredgewidth=.6, label=label)
                ax.fill_between(xs, [max(y - s, 1e-4) for y, s in zip(ys, sds)],
                                [y + s for y, s in zip(ys, sds)], color=colour, alpha=.12, linewidth=0)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            if r == 0:
                ax.set_title(f"d={d}", fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel(f"k={k},g={g}\nnMSE", fontsize=7, color=MUTED)
            if r == len(structures) - 1:
                ax.set_xlabel("retrainings m", fontsize=8, color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(.5, 1.0))
    save(fig, "pilot_nmse_curves")
    print("figures written")


if __name__ == "__main__":
    main()
