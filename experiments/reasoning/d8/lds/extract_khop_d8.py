"""k-hop d=8 transformer cube (d=8 study / the development run): exact curves, fitted curves, LDS and native Spearman.

Inputs (read only): the released cube `khop_d8_transformer` (256 cells, one learner seed), d=8 study's
selection.json and mechanism_values.csv, and d=8 study's 20 sampling repeats of predicted cubes
(`results/fits/repeat_<r>.npz`, gitignored in the d=8 study worktree) at budgets 24, 32, 48, 64, 128.
"""
from __future__ import annotations

import json
import math

import numpy as np

from _common import Provenance, write_rows
import lds

D, Q = 8, 8
FAMILIES = ["cardinality", "linear", "degree2_ridge", "degree3_ridge", "sparse_degree3", "rbf_kernel",
            "cp", "tt", "ht", "tt_ladder"]
BUDGETS = [24, 32, 48, 64, 128]
GRID = np.linspace(0.0, 1.0, 41)


def delta(cube, players):
    """Delta_T v(S) for every mask S disjoint from T; returns (masks_without_T, values)."""
    masks = np.arange(1 << D)
    tmask = sum(1 << p for p in players)
    base = masks[(masks & tmask) == 0]
    out = np.zeros((len(base),) + cube.shape[1:])
    for k in range(len(players) + 1):
        from itertools import combinations
        for sub in combinations(players, k):
            sign = (-1) ** (len(players) - k)
            out += sign * cube[base | sum(1 << p for p in sub)]
    return base, out


def curve(cube, players, t):
    base, dv = delta(cube, players)
    size = lds.popcount(base)
    free = D - len(players)
    w = (t[:, None] ** size[None, :]) * ((1 - t[:, None]) ** (free - size[None, :]))
    return w @ dv


def shapley_interaction(cube, players):
    base, dv = delta(cube, players)
    size = lds.popcount(base)
    free = D - len(players)
    w = np.array([math.factorial(int(s)) * math.factorial(free - int(s)) / math.factorial(free + 1) for s in size])
    return w @ dv


def main():
    prov = Provenance("extract_khop_d8")
    cube_path = prov.read("HUB", "07_cube_suite_and_integration/suite/cubes/khop_d8_transformer.npz")
    z = np.load(cube_path)
    order = np.argsort(z["masks"])
    assert list(z["masks"][order]) == list(range(256))
    truth = z["values"][order].reshape(256, Q, 31).mean(2)  # per-query mean log-probability
    sel = json.loads(prov.read("HUB", "03_mechanism_figure/results/selection.json").read_text())["selection"]
    q = sel["focal_query"]
    mech = {"redundancy": tuple(sel["redundancy"]["players"]),
            "complementarity": tuple(sel["complementarity"]["players"]),
            "synergy": tuple(sel["synergy"]["players"])}
    checks = []
    for name, players in mech.items():
        ours = float(shapley_interaction(truth[:, q], players))
        theirs = sel[name]["shapley_interaction"]
        assert abs(ours - theirs) < 1e-9, (name, ours, theirs)
        checks.append(dict(mechanism=name, players="-".join(map(str, players)), exact_shapley_interaction=ours,
                           d8_value=theirs, empty_background=float(curve(truth[:, q], players, np.array([0.0]))[0]),
                           banzhaf=float(curve(truth[:, q], players, np.array([0.5]))[0]),
                           loo=float(curve(truth[:, q], players, np.array([1.0]))[0])))
    outputs = [write_rows("khop_d8/mechanism_exact.csv", checks)]
    exact_rows = []
    for name, players in mech.items():
        g = curve(truth[:, q], players, GRID)
        exact_rows += [dict(mechanism=name, t=float(t), g=float(v)) for t, v in zip(GRID, g)]
    outputs.append(write_rows("khop_d8/exact_curves.csv", exact_rows))

    fitted = {}  # (mech, family, budget) -> list of curves
    lds_rows = []
    for rep in range(1, 21):
        npz = np.load(prov.read("D8_DIR", f"results/fits/repeat_{rep}.npz"))
        for b in BUDGETS:
            sample = set(int(m) for m in npz[f"m{b}__sample"])
            held = np.array(sorted(set(range(256)) - sample))
            held4 = held[lds.popcount(held) == 4]
            for fam in FAMILIES:
                pred = npz[f"m{b}__{fam}"]
                for name, players in mech.items():
                    fitted.setdefault((name, fam, b), []).append(curve(pred[:, q], players, GRID))
                tau = lds.shapley_from_cube(pred, D)  # (D, Q) from predictions only
                for qq in range(Q):
                    l4, n4 = lds.lds_and_native(tau[:, qq], pred[held4, qq], truth[held4, qq], held4, D)
                    attribution = lds.bits(held, D) @ tau[:, qq]
                    wc_l = lds.within_cardinality(attribution, truth[held, qq], held)
                    wc_n = lds.within_cardinality(pred[held, qq], truth[held, qq], held)
                    lds_rows.append(dict(
                        repeat=rep, budget=b, family=fam, query=qq, n_heldout=len(held), n_size4=len(held4),
                        lds_size4=l4, native_size4=n4, tie_fraction_size4=lds.tie_fraction(truth[held4, qq]),
                        lds_pooled=lds.spearman(attribution, truth[held, qq]),
                        native_pooled=lds.spearman(pred[held, qq], truth[held, qq]),
                        lds_within_mean=np.nanmean([v for v in wc_l.values() if v is not None] or [np.nan]),
                        native_within_mean=np.nanmean([v for v in wc_n.values() if v is not None] or [np.nan])))
    summary = []
    for (name, fam, b), curves in sorted(fitted.items()):
        arr = np.array(curves)
        med, q25, q75 = np.median(arr, 0), np.quantile(arr, 0.25, 0), np.quantile(arr, 0.75, 0)
        for i, t in enumerate(GRID):
            summary.append(dict(mechanism=name, family=fam, budget=b, t=float(t), median=float(med[i]),
                                q25=float(q25[i]), q75=float(q75[i]), repeats=len(curves)))
    outputs.append(write_rows("khop_d8/fitted_curves_summary.csv", summary))
    outputs.append(write_rows("khop_d8/lds_per_fit.csv", lds_rows))
    import csv as _csv
    for rel, out in (("03_mechanism_figure/tables/mechanism_values.csv", "khop_d8/d8_mechanism_values.csv"),
                     ("03_mechanism_figure/tables/least_squares_ceiling_exploratory.csv", "khop_d8/d8_ls_ceiling.csv")):
        with open(prov.read("HUB", rel), newline="") as f:
            rows = list(_csv.DictReader(f))
        outputs.append(write_rows(out, rows, list(rows[0].keys())))
    prov.read("HUB", "03_mechanism_figure/results/decision.json")
    prov.write(outputs)
    print("ok", len(lds_rows), len(summary))


if __name__ == "__main__":
    main()
