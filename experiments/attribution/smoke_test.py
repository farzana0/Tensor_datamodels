#!/usr/bin/env python3
"""Exercise the equal-budget study machinery on synthetic games only.

This runs before the pre-registration is frozen, so it must not touch a real
cube's outcomes.  The synthetic games have known structure (additive, a planted
pair, a planted triple, and a low-rank AND-of-ORs), which also checks that the
estimators and surrogates recover the structure they are supposed to recover.
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cubes, design, direct, exact, runner, surrogates  # noqa: E402


def synthetic(d: int, kind: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    masks = np.arange(1 << d)
    bits = ((masks[:, None] >> np.arange(d)) & 1).astype(float)
    if kind == "additive":
        weight = rng.normal(size=d)
        value = bits @ weight
    elif kind == "pair":
        value = bits @ rng.normal(size=d) + 3.0 * bits[:, 0] * bits[:, 1]
    elif kind == "triple":
        value = bits @ rng.normal(size=d) + 5.0 * bits[:, 0] * bits[:, 1] * bits[:, 2]
    elif kind == "and_of_ors":
        ors = [np.maximum(bits[:, 2 * i], bits[:, 2 * i + 1]) for i in range(d // 2)]
        value = np.prod(ors, axis=0) * 4.0 + 0.1 * bits @ rng.normal(size=d)
    else:
        raise ValueError(kind)
    return np.column_stack([value, value + 0.05 * rng.normal(size=len(value))])


def check_estimators() -> None:
    d = 6
    values = synthetic(d, "pair", 1)
    truth_shapley = exact.shapley_values(values)
    truth_banzhaf = exact.banzhaf_values(values)
    truth_sii = exact.pair_interactions(values)
    full = 1 << d
    rng = np.random.default_rng(0)
    # At the full budget every estimator should be close to exact.
    permutation = direct.permutation_shapley(values, d, full, np.random.default_rng(2))
    kernel = direct.kernel_shap(values, d, full, np.random.default_rng(3))
    msr = direct.msr_banzhaf(values, d, full, np.random.default_rng(4))
    stratified = direct.stratified_shapley(values, d, full, np.random.default_rng(5))
    loo = direct.direct_loo(values, d, full, rng)
    sii = direct.sampling_pair_sii(values, d, full, np.random.default_rng(6))
    fsii_estimate = direct.regression_pair_interactions(values, d, full,
                                                        np.random.default_rng(7))
    _, exact_fsii = exact.fsii(values, 2)
    print("permutation shapley err", np.abs(permutation.shapley - truth_shapley).max(),
          "evals", permutation.evaluations)
    print("kernelshap err        ", np.abs(kernel.shapley - truth_shapley).max(),
          "evals", kernel.evaluations)
    print("stratified err        ", np.abs(stratified.shapley - truth_shapley).max(),
          "evals", stratified.evaluations)
    print("msr banzhaf err       ", np.abs(msr.banzhaf - truth_banzhaf).max(),
          "evals", msr.evaluations)
    print("loo err               ", np.abs(loo.loo - exact.loo_values(values)).max(),
          "evals", loo.evaluations)
    print("sampling sii err      ", np.nanmax(np.abs(sii.pair_sii - truth_sii)),
          "evals", sii.evaluations)
    print("fsii regression err   ", np.abs(fsii_estimate.pair_fsii - exact_fsii).max(),
          "evals", fsii_estimate.evaluations)
    assert np.abs(kernel.shapley - truth_shapley).max() < 1e-8, "KernelSHAP must be exact at 2^d"
    assert np.abs(msr.banzhaf - truth_banzhaf).max() < 1e-8, "MSR must be exact at 2^d"
    assert np.abs(fsii_estimate.pair_fsii - exact_fsii).max() < 1e-6, "FSII regression at 2^d"
    assert loo.evaluations == d + 1


def check_surrogates() -> None:
    d = 6
    for kind in ("additive", "pair", "triple", "and_of_ors"):
        values = synthetic(d, kind, 2)
        rng = np.random.default_rng(11)
        masks = design.uniform(d, 32, rng)
        y = values[masks]
        line = []
        for family, fit_fn in surrogates.FAMILIES.items():
            fit = fit_fn(masks, y, d, 17)
            error = np.sqrt(np.mean((fit.prediction[:, 0] - values[:, 0]) ** 2))
            line.append(f"{family}={error:.3f}")
        print(f"{kind:11s} full-cube rmse at m=32: " + " ".join(line))


def check_runner() -> None:
    values = synthetic(5, "pair", 3)
    cube = cubes.Cube(name="synthetic_d5", d=5, values=values,
                      output_names=["primary", "noisy"], primary_output=0,
                      family="synthetic", learner="synthetic", utility="synthetic")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=runner.ROW_FIELDS, lineterminator="\n")
    writer.writeheader()
    runner.run_cube(cube, writer, verbose=False)
    lines = buffer.getvalue().strip().split("\n")
    print(f"runner produced {len(lines) - 1} rows for a synthetic d=5 cube")
    assert len(lines) > 100


if __name__ == "__main__":
    check_estimators()
    check_surrogates()
    check_runner()
    print("smoke test ok")
