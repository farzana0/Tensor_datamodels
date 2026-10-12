#!/usr/bin/env python3
"""Reference approximators from shapiq on each cube's primary output.

The shapiq version is pinned in the output metadata.  Budgets use the equal-budget study
distinct-evaluation accounting: shapiq's nominal budget counts game calls, which
can include repeats of a coalition (free under caching), so the nominal budget is
raised geometrically and the largest setting that stays within ``m`` distinct
coalitions is kept.  Exact targets for k-SII and FBII come from shapiq's own
ExactComputer; SV, SII and FSII come from the equal-budget study engine, validated against it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cubes, exact, jobs, metrics, protocol  # noqa: E402

OUT = Path(__file__).resolve().parent / "results/shapiq"
FIELDS = ["cube", "d", "family", "output_name", "method", "index", "order_scored", "budget",
          "budget_multiple", "repeat", "nominal_budget", "evaluations", "seconds",
          "rel_l2", "spearman", "precision_at_k"]


def approximators(d: int):
    from shapiq import approximator as A
    return {
        # Shapley value
        ("KernelSHAP", "SV"): lambda rs: A.KernelSHAP(n=d, random_state=rs),
        ("UnbiasedKernelSHAP", "SV"): lambda rs: A.UnbiasedKernelSHAP(n=d, random_state=rs),
        ("PermutationSamplingSV", "SV"): lambda rs: A.PermutationSamplingSV(n=d, random_state=rs),
        ("StratifiedSamplingSV", "SV"): lambda rs: A.StratifiedSamplingSV(n=d, random_state=rs),
        ("SVARM", "SV"): lambda rs: A.SVARM(n=d, random_state=rs),
        # pairwise Shapley interaction index
        ("KernelSHAPIQ", "SII"): lambda rs: A.KernelSHAPIQ(n=d, max_order=2, index="SII", random_state=rs),
        ("SHAPIQ", "SII"): lambda rs: A.SHAPIQ(n=d, max_order=2, index="SII", random_state=rs),
        ("SVARMIQ", "SII"): lambda rs: A.SVARMIQ(n=d, max_order=2, index="SII", random_state=rs),
        ("PermutationSamplingSII", "SII"): lambda rs: A.PermutationSamplingSII(n=d, max_order=2, index="SII", random_state=rs),
        # k-SII, FSII, FBII
        ("KernelSHAPIQ", "k-SII"): lambda rs: A.KernelSHAPIQ(n=d, max_order=2, index="k-SII", random_state=rs),
        ("SHAPIQ", "k-SII"): lambda rs: A.SHAPIQ(n=d, max_order=2, index="k-SII", random_state=rs),
        ("RegressionFSII", "FSII"): lambda rs: A.RegressionFSII(n=d, max_order=2, random_state=rs),
        ("RegressionFBII", "FBII"): lambda rs: A.RegressionFBII(n=d, max_order=2, random_state=rs),
    }


def exact_reference(values: np.ndarray, d: int) -> dict:
    from shapiq.game_theory.exact import ExactComputer
    weights = 1 << np.arange(d)
    computer = ExactComputer(n_players=d, game=lambda co: values[co.astype(int) @ weights])
    pairs = exact.pairs(d)
    ksii = computer(index="k-SII", order=2)
    fbii = computer(index="FBII", order=2)
    _, fsii = exact.fsii(values, 2)
    return {
        "SV": (exact.shapley_values(values), [(i,) for i in range(d)]),
        "SII": (exact.pair_interactions(values, shapley=True), pairs),
        "k-SII": (np.asarray([ksii[p] for p in pairs]), pairs),
        "FSII": (fsii, pairs),
        "FBII": (np.asarray([fbii[p] for p in pairs]), pairs),
    }


def run_estimator(factory, values, d, budget, seed):
    """Largest nominal shapiq budget whose distinct-coalition use stays <= budget."""
    weights = 1 << np.arange(d)
    best = None
    nominal = budget
    ceiling = 1 << d
    while nominal <= 4 * ceiling:
        seen: set[int] = set()

        def game(coalitions):
            index = coalitions.astype(int) @ weights
            seen.update(int(i) for i in np.atleast_1d(index))
            return values[index]

        try:
            result = factory(seed).approximate(budget=int(nominal), game=game)
        except Exception as error:  # an approximator can reject a tiny budget
            if best is None:
                return None, nominal, 0, repr(error)
            break
        if len(seen) > budget:
            break
        best = (result, nominal, len(seen))
        if len(seen) >= budget or nominal >= 4 * ceiling:
            break
        nominal = int(np.ceil(nominal * 1.5))
    if best is None:
        return None, budget, 0, "no feasible nominal budget"
    return best[0], best[1], best[2], ""


def run_cube(name: str) -> str:
    import shapiq
    OUT.mkdir(parents=True, exist_ok=True)
    cube = cubes.load(name)
    d = cube.d
    values = cube.output(cube.primary_output)
    reference = exact_reference(values, d)
    path = OUT / f"{name}.csv"
    start_all = time.time()
    with jobs.claim(path) as partial:
        if partial is None:
            return f"{name}: skipped (done or claimed elsewhere)"
        _write_shapiq(name, cube, d, values, reference, partial)
    meta = {"cube": name, "shapiq_version": shapiq.__version__,
            "shapiq_path": shapiq.__file__, "output": cube.output_names[cube.primary_output],
            "preregistration_sha256": protocol.preregistration_sha256(),
            "seconds": round(time.time() - start_all, 1)}
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return f"{name}: {meta['seconds']}s"


def _write_shapiq(name, cube, d, values, reference, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for budget in protocol.budgets(d):
            multiple = protocol.budget_multiple(d, budget)
            for repeat in range(protocol.REPEATS):
                for (method, index), factory in approximators(d).items():
                    seed = protocol.seed_for(name, budget, repeat, f"shapiq|{method}|{index}") % (2 ** 31)
                    start = time.perf_counter()
                    result, nominal, used, error = run_estimator(factory, values, d, budget, seed)
                    elapsed = time.perf_counter() - start
                    truth, keys = reference[index]
                    row = {"cube": name, "d": d, "family": cube.family,
                           "output_name": cube.output_names[cube.primary_output],
                           "method": method, "index": index,
                           "order_scored": len(keys[0]), "budget": budget,
                           "budget_multiple": multiple, "repeat": repeat,
                           "nominal_budget": nominal, "evaluations": used,
                           "seconds": round(elapsed, 4)}
                    if result is None:
                        row.update(rel_l2="", spearman="", precision_at_k="")
                    else:
                        estimate = np.asarray([result[key] for key in keys], dtype=float)
                        row["rel_l2"] = metrics.relative_l2(estimate, truth)
                        row["spearman"] = metrics.spearman(estimate, truth)
                        row["precision_at_k"] = (metrics.precision_at_k(estimate, truth, protocol.precision_k(d))
                                                 if len(keys[0]) == 2 else "")
                    writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    names = cubes.all_names() if args.all else [args.cube]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for message in pool.map(run_cube, names):
                print(message, flush=True)
    else:
        for name in names:
            print(run_cube(name), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
