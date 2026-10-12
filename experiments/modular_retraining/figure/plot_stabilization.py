#!/usr/bin/env python3
"""Plot the new additive-initialized TT against intact historical comparators.

This renderer only reads analyzed summaries. It never fits a model or changes
the frozen experiment/recovery code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RECOVERY_STYLE = ROOT / "plot_recovered.py"
spec = importlib.util.spec_from_file_location("recovered_plot_style", RECOVERY_STYLE)
recovered = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovered)
plt, np = recovered.plt, recovered.np
Line2D = recovered.Line2D
NullLocator = recovered.NullLocator
METRICS, BUDGETS = recovered.METRICS, recovered.BUDGETS
NEW_METHOD = "tt_additive_init"
MAIN_METHODS = {method: styling for method, styling in recovered.METHODS.items()
                if method != "tt"}
MAIN_METHODS[NEW_METHOD] = ("TNDM-TT (additive init)", "#2A9D8F", "v", "-")
COMPARISON_METHODS = {
    "tt": ("TNDM-TT (original init)", "#2A9D8F", "v", "-"),
    NEW_METHOD: ("TNDM-TT (additive init)", "#4C78A8", "s", "--"),
}
ALL_METHODS = set(recovered.METHODS) | {NEW_METHOD}


def load_statistics(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Analysis is not ready: {path}")
    indexed = {}
    for row in csv.DictReader(path.open(newline="")):
        if row["metric"] not in METRICS or row["method"] not in ALL_METHODS:
            continue
        key = row["metric"], row["method"], int(row["budget"])
        if key in indexed:
            raise ValueError(f"Duplicate summary: {key}")
        if int(row["n_learner_seeds"]) != 3 or int(row["repeats_per_seed"]) != 20:
            raise ValueError(f"Unexpected repetition structure: {key}")
        if "n_finite" in row and int(row["n_finite"]) != 60:
            raise ValueError(f"Incomplete finite repetition set; disclose before plotting: {key}")
        if "n_failures" in row and int(row["n_failures"]) != 0:
            raise ValueError(f"Nonfinite failures require explicit handling: {key}")
        mean, sd = float(row["mean_of_seed_means"]), float(row["sample_sd_across_seed_means"])
        lower, upper = float(row["sd_lower"]), float(row["sd_upper"])
        if not all(math.isfinite(v) for v in (mean, sd, lower, upper)) or sd < 0:
            raise ValueError(f"Nonfinite/invalid summary: {key}")
        if not (math.isclose(lower, mean - sd, abs_tol=1e-12) and
                math.isclose(upper, mean + sd, abs_tol=1e-12)):
            raise ValueError(f"Summary SD endpoints are inconsistent: {key}")
        indexed[key] = (mean, sd, lower, upper)
    expected = {(metric, method, budget) for metric in METRICS
                for method in ALL_METHODS for budget in BUDGETS}
    if set(indexed) != expected:
        raise ValueError(f"Incomplete/unexpected analyzed grid: missing={expected-set(indexed)}; extra={set(indexed)-expected}")
    return indexed


def render(indexed, methods, output_dir: Path, stem: str, scale: str):
    if scale == "log":
        nonpositive = [(metric, method, budget)
                       for metric in METRICS for method in methods for budget in BUDGETS
                       if indexed[metric, method, budget][2] <= 0]
        if nonpositive:
            raise ValueError(f"Ordinary log display cannot preserve nonpositive SD bounds: {nonpositive}")
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.9))
    fig.subplots_adjust(left=0.073, right=0.993, bottom=0.19,
                        top=0.76, wspace=0.43)
    handles = [Line2D([], [], color=color, marker=marker, linestyle=line,
                      label=label, linewidth=1.35, markersize=4.1)
               for label, color, marker, line in methods.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.515, 0.995),
               ncol=len(methods), frameon=False, handlelength=1.8,
               handletextpad=0.4, columnspacing=1.05, borderaxespad=0)
    limits = {}
    for ax, (metric, title) in zip(axes, METRICS.items()):
        ax.set_xscale("log", base=2)
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=1.0, linscale=1.0, base=10)
        elif scale == "log":
            ax.set_yscale("log", base=10, nonpositive="mask")
        lowers, uppers = [], []
        for method, (label, color, marker, line) in methods.items():
            series = np.array([indexed[metric, method, budget] for budget in BUDGETS])
            mean, sd, lower, upper = series.T
            lowers.extend(lower)
            uppers.extend(upper)
            ax.fill_between(BUDGETS, lower, upper, color=color, alpha=0.12,
                            linewidth=0, zorder=2)
            ax.plot(BUDGETS, mean, label=label, color=color, marker=marker,
                    linestyle=line, markeredgewidth=0.5,
                    markeredgecolor="white", zorder=3)
        ax.set_title(title, loc="left", pad=8)
        ax.set_xlabel("Observation budget", labelpad=4)
        ax.set_ylabel("Relative L2 error" + (f" ({scale})" if scale in ("symlog", "log") else ""),
                      labelpad=3)
        ax.set_xlim(7.1, 144.0)
        ax.set_xticks(BUDGETS, [str(v) for v in BUDGETS])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(direction="out", length=3, width=0.65, pad=3)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.55)
        if scale == "log":
            # Every arithmetic mean +/- SD endpoint has already passed the
            # strict-positive guard above. Pad in log coordinates without a
            # floor, clipping, symmetrizing after transformation, or adding zero.
            lower, upper = min(lowers), max(uppers)
            log_bounds = np.log10(np.array([lower, upper]))
            pad = max(float(log_bounds[1] - log_bounds[0]) * 0.065, 1e-8)
            axis_min, axis_max = np.power(10.0, log_bounds + np.array([-pad, pad]))
            ax.set_ylim(float(axis_min), float(axis_max))
            if not (0 < axis_min <= lower and axis_max >= upper):
                raise AssertionError("Ordinary log limits would clip a saved SD bound")
            limits[metric] = {"data_sd_min": lower, "data_sd_max": upper,
                              "axis_min": float(axis_min), "axis_max": float(axis_max)}
        else:
            limits[metric] = recovered.set_unclipped_limits(ax, min(lowers), max(uppers), scale)
    for extension in ("pdf", "png", "svg"):
        extra = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
        fig.savefig(output_dir / f"{stem}_{scale}.{extension}", dpi=300, **extra)
    plt.close(fig)
    return limits


def write_captions(output_dir: Path, preferred: str):
    common = """Each point is the equal-weight mean of three learner-seed means (seeds
31, 43, and 59), each computed across 20 matched coalition-sampling repetitions.
Light translucent bands are plus/minus one sample SD across the three seed means;
they are not confidence intervals or the SD of 60 pooled repetitions. The primary
query is triple_ABC, with Shapley, Banzhaf, and pairwise Shapley interaction-index
(SII) relative L2 errors evaluated against each seed's complete d=8 measured cube.
The new fits jointly predict all four query outputs, as in the original protocol.

The x-axis uses base-2 logarithmic spacing and labels all five budgets. Both full
linear and signed symmetric-log y-axis versions are supplied. The latter is
linear on [-1, 1]. Every saved mean-minus-SD and mean-plus-SD bound is preserved;
negative lower bounds describe symmetric SD bands, not negative individual errors.
No unfavorable original-TT runs are dropped from the initialization comparison.

DejaVu Sans 9 pt, 8.5 by 2.9 inches, and light bands (alpha 0.12) follow the recovered
fallback style. Exact matching to the unavailable v64/HAR-SD source is unverified.
No overall title, subtitle, caption, or footer is embedded in the figure.
"""
    main = """# Main figure: additive-initialized TT and historical comparators

Attribution estimation from limited coalition observations in the modular
relation-table retraining game. TNDM-TT (additive init) is a newly fitted variant
authorized after recovery of the historical results. Linear, quadratic, cubic
OMP, RBF, and TNDM-CP curves are unchanged historical fits; this figure does not
show the original TT initialization, which is retained in the companion figure.
No learner was retrained for the new experiment: only TT surrogates were fitted
to the same measured coalition labels and original matched observation designs.

The new TT initialization starts from a validation-selected additive ridge
surface, applies rank-truncated TT-SVD, and uses the frozen two-restart protocol.
The original ALS optimization, rank/ridge candidates, and validation rule are
otherwise retained. Additive selection introduces additional computation and
validation adaptation but no additional coalition observations. See the paper for complete details and limitations.

""" + common + f"\nThe convenience main-figure copy selects the **{preferred}** y-axis version.\n"
    if preferred == "log":
        main += """
An ordinary base-10 logarithmic y-axis version is supplied because every saved
arithmetic mean-minus-SD bound in this six-method figure is strictly positive.
Bands still represent the original arithmetic mean plus/minus sample SD: the
endpoints are transformed only for display, not recomputed in log space. No
positive floor, clipping, or removal of low/error endpoints is used. The full
linear and signed symmetric-log versions remain available alongside this version.
"""
    else:
        main += """
An ordinary logarithmic y-axis figure was skipped because at least one saved
mean-minus-SD endpoint was nonpositive. No endpoint was floored or clipped to
permit a logarithmic display.
"""
    comparison = """# Companion figure: original versus additive-informed TT initialization

Original historical TNDM-TT fits and newly fitted TNDM-TT (additive init) on the
same three learner seeds, five budgets, and twenty observation repetitions per
seed/budget. The original means and SDs, including low-budget instabilities, are
retained intact. This comparison is descriptive evidence for one prespecified
initialization variant on the same saved modular game, not a new task replication.

""" + common + """
The convenience initialization-comparison copy uses a signed symmetric-log
y-axis to retain the original TT's negative lower SD endpoints. No ordinary
logarithmic y-axis comparison is generated.
"""
    (output_dir / "modular_stabilized_main_caption.md").write_text(main)
    (output_dir / "modular_tt_initialization_comparison_caption.md").write_text(comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, default=ROOT.parent / "tt_additive_init/data/plot_statistics.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    indexed = load_statistics(args.statistics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recovered.setup_style()
    series = {"modular_stabilized_main": MAIN_METHODS,
              "modular_tt_initialization_comparison": COMPARISON_METHODS}
    limits = {}
    for stem, methods in series.items():
        limits[stem] = {scale: render(indexed, methods, args.output_dir, stem, scale)
                        for scale in ("linear", "symlog")}
    main_lowers = [indexed[metric, method, budget][2] for metric in METRICS
                   for method in MAIN_METHODS for budget in BUDGETS]
    ordinary_log_valid = min(main_lowers) > 0
    if ordinary_log_valid:
        main_stem = "modular_stabilized_main"
        limits[main_stem]["log"] = render(indexed, MAIN_METHODS, args.output_dir, main_stem, "log")
    preferred = "log" if ordinary_log_valid else ("linear" if min(main_lowers) >= 0 else "symlog")
    for extension in ("pdf", "png", "svg"):
        shutil.copyfile(args.output_dir / f"modular_stabilized_main_{preferred}.{extension}",
                        args.output_dir / f"modular_stabilized_main.{extension}")
        shutil.copyfile(args.output_dir / f"modular_tt_initialization_comparison_symlog.{extension}",
                        args.output_dir / f"modular_tt_initialization_comparison.{extension}")
    write_captions(args.output_dir, preferred)
    manifest = {
        "statistics": str(args.statistics.resolve()),
        "statistics_sha256": hashlib.sha256(args.statistics.read_bytes()).hexdigest(),
        "style_source": str(RECOVERY_STYLE),
        "style_source_sha256": hashlib.sha256(RECOVERY_STYLE.read_bytes()).hexdigest(),
        "n_summary_rows": len(indexed), "figure_size_inches": [8.5, 2.9],
        "font_family": "DejaVu Sans", "all_font_sizes_pt": 9,
        "band_alpha": 0.12, "xscale": "log2", "symlog_linthresh": 1.0,
        "main_methods": MAIN_METHODS, "comparison_methods": COMPARISON_METHODS,
        "preferred_main_scale": preferred, "main_minimum_sd_lower": min(main_lowers),
        "ordinary_log_main_valid": ordinary_log_valid,
        "ordinary_log_rule": "Every saved arithmetic mean-minus-SD endpoint must be strictly positive; no flooring or clipping",
        "preferred_comparison_scale": "symlog",
        "panel_limits": limits,
        "uncertainty": "sample SD across 3 learner-seed means, each averaging 20 repetitions",
        "newly_fitted_method": NEW_METHOD,
        "historical_methods": list(recovered.METHODS),
        "exact_requested_style_verified": False,
    }
    (args.output_dir / "stabilization_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered {len(indexed)} summaries; preferred main y-axis: {preferred}")


if __name__ == "__main__":
    main()
