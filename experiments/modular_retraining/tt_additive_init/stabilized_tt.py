"""Frozen equal-budget study intervention: rank-truncated additive-informed TT initialization.

The caller places the frozen historical original comparison library directory on sys.path.
Only initialization changes inside the historical TT rank/ridge/restart sweep.
This module never reads an oracle file or accesses labels outside ``masks, y``.
"""
from __future__ import annotations

import numpy as np

from lib import surrogates as original


INITIALIZER_JITTER_FRACTION = 0.001
METHOD_ID = "tt_additive_init"


def _historical_bonds(d: int, rank: int) -> list[int]:
    return ([1] + [min(rank, 2 ** min(k + 1, d - k - 1))
                   for k in range(d - 1)] + [rank])


def _svd_initialization(surface: np.ndarray, d: int, bonds: list[int]):
    """Compress fitted predictions, with bit zero first and historical shapes.

    This is TT-SVD of a *predicted* dense tensor. It is not an oracle lookup.
    Zero-padding restores nominal (possibly redundant) historical ALS shapes.
    """
    surface = np.asarray(surface, dtype=float)
    if surface.ndim != 2 or surface.shape[0] != 1 << d:
        raise ValueError("Initializer must have shape (2**d, number_of_outputs)")
    if len(bonds) != d + 1 or bonds[0] != 1:
        raise ValueError("Invalid TT bond dimensions")
    if not np.all(np.isfinite(surface)):
        raise ValueError("Nonfinite fitted initializer surface")
    outputs = surface.shape[1]
    tensor = surface.reshape((2,) * d + (outputs,))
    tensor = tensor.transpose(tuple(range(d - 1, -1, -1)) + (d,))
    remainder = tensor.copy()
    effective_left = 1
    effective_ranks = [1]
    cores = []
    for axis in range(d):
        matrix = remainder.reshape(effective_left * 2, -1)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        effective_right = min(bonds[axis + 1], len(singular))
        unpadded = u[:, :effective_right].reshape(effective_left, 2, effective_right)
        core = np.zeros((bonds[axis], 2, bonds[axis + 1]))
        core[:effective_left, :, :effective_right] = unpadded
        cores.append(core)
        remainder = singular[:effective_right, None] * vh[:effective_right]
        effective_left = effective_right
        effective_ranks.append(effective_right)
    output = np.zeros((bonds[-1], outputs))
    output[:effective_left] = remainder.reshape(effective_left, outputs)
    reconstructed = original._tt_predict(cores, output, original.bits(np.arange(1 << d), d))
    residual_norm = float(np.linalg.norm(reconstructed - surface))
    surface_norm = float(np.linalg.norm(surface))
    diagnostics = {
        "initializer_bonds": list(bonds),
        "initializer_effective_ranks": effective_ranks,
        "initializer_truncation_rmse": float(np.sqrt(np.mean((reconstructed - surface) ** 2))),
        "initializer_truncation_relative_l2": residual_norm / surface_norm if surface_norm else 0.0,
        "initializer_gauge": "left_TT_SVD_then_zero_pad_to_historical_bonds",
    }
    return cores, output, diagnostics


def stable_fit_tt(masks, y, d, seed):
    """Return the historical Fit contract with an additive-informed initialization.

    Additive alpha uses the original 19-value grid and inner-validation MSE
    across all outputs. TT selection/refitting uses the historical solver
    unchanged, with its historical grid, rank-screening rule and sweep counts.
    """
    masks = np.asarray(masks)
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[0] != len(masks):
        raise ValueError("Expected an observed mask-by-output label matrix")
    fit_masks, validation_masks = original.stratified_fit_validation(masks, d, seed)
    lookup = {int(mask): index for index, mask in enumerate(masks)}
    fit_index = [lookup[int(mask)] for mask in fit_masks]
    validation_index = [lookup[int(mask)] for mask in validation_masks]
    y_fit, y_validation = y[fit_index], y[validation_index]
    z_fit = original.bits(fit_masks, d)
    z_validation = original.bits(validation_masks, d)
    x_fit = np.column_stack((np.ones(len(fit_masks)), z_fit))
    x_validation = np.column_stack((np.ones(len(validation_masks)), z_validation))
    cube_bits = original.bits(np.arange(1 << d), d)
    x_cube = np.column_stack((np.ones(len(cube_bits)), cube_bits))

    additive_best = None
    additive_candidates = []
    for alpha in original.ALPHAS:
        # Crucially these coefficients have not been refitted on validation labels.
        coefficient = original.ridge_solve(x_fit, y_fit, alpha)
        score = original._mse(x_validation @ coefficient, y_validation)
        additive_candidates.append({"alpha": float(alpha), "validation_mse": float(score)})
        if np.isfinite(score) and (additive_best is None or score < additive_best[0]):
            additive_best = (score, alpha, coefficient.copy())
    if additive_best is None:
        raise RuntimeError("No finite additive initialization candidate")
    additive_score, additive_alpha, coefficient = additive_best
    additive_surface = x_cube @ coefficient
    centered_fit_rms = float(np.sqrt(np.mean((y_fit - y_fit.mean(axis=0)) ** 2)))

    # Both restarts use the same validation-selected linear coefficients. Restart
    # zero has no jitter; restart one has fixed, deterministic function-space jitter.
    initializers = {}
    initializer_diagnostics = {}
    for rank in original.TT_RANKS:
        if 2 * rank * rank * d > 6 * len(fit_masks) * y.shape[1]:
            continue
        for restart in range(original.RESTARTS):
            initializer_seed = int(seed * 977 + restart)
            jitter_sd = INITIALIZER_JITTER_FRACTION * centered_fit_rms if restart == 1 else 0.0
            surface = additive_surface.copy()
            if jitter_sd:
                surface += np.random.default_rng(initializer_seed).normal(
                    scale=jitter_sd, size=surface.shape)
            cores, output, diagnostics = _svd_initialization(surface, d, _historical_bonds(d, rank))
            initialized_surface = original._tt_predict(cores, output, cube_bits)
            diagnostics.update({
                "initializer_seed": initializer_seed,
                "initializer_jitter_sd": jitter_sd,
                "initializer_deviation_from_additive_rmse": float(
                    np.sqrt(np.mean((initialized_surface - additive_surface) ** 2))),
            })
            initializers[rank, restart] = (cores, output)
            initializer_diagnostics[rank, restart] = diagnostics

    validation = (z_validation, y_validation)
    best = None
    candidate_diagnostics = []
    for rank in original.TT_RANKS:
        if 2 * rank * rank * d > 6 * len(fit_masks) * y.shape[1]:
            continue
        for ridge in original.ALS_RIDGES:
            for restart in range(original.RESTARTS):
                _, _, score, sweep = original._tt_als(
                    z_fit, y_fit, rank, seed * 977 + restart,
                    validation=validation, ridge=ridge, init=initializers[rank, restart])
                finite = bool(np.isfinite(score))
                candidate_diagnostics.append({
                    "rank": int(rank), "ridge": float(ridge), "restart": int(restart),
                    "validation_mse": float(score) if finite else None,
                    "selected_sweep": int(sweep),
                    "status": "finite_validation_candidate" if finite else "no_finite_validation_candidate",
                    **initializer_diagnostics[rank, restart],
                })
                if finite and (best is None or score < best[0]):
                    best = (score, rank, restart, sweep, ridge)

    used_fallback = best is None
    if used_fallback:
        # This duplicates the original family-level fallback. The initializer
        # remains additive-informed; it does not switch to a linear predictor.
        best = (np.inf, original.TT_RANKS[0], 0, original.ALS_SWEEPS, original.ALS_RIDGES[-1])
    score, rank, restart, sweep, ridge = best
    cores, output, _, _ = original._tt_als(
        z_fit, y_fit, rank, seed * 977 + restart, sweeps=max(1, sweep), ridge=ridge,
        init=initializers[rank, restart])
    reconstructed_validation_mse = original._mse(
        original._tt_predict(cores, output, z_validation), y_validation)
    cores, output, _, _ = original._tt_als(
        original.bits(masks, d), y, rank, 0, sweeps=original.REFIT_SWEEPS,
        ridge=ridge, init=(cores, output))
    prediction = original._tt_predict(cores, output, cube_bits)
    choice = {
        "rank": int(rank), "restart": int(restart), "sweeps": int(sweep),
        "ridge": float(ridge), "validation_mse": float(score) if np.isfinite(score) else None,
        "method_id": METHOD_ID,
        "initialization": "rank_truncated_additive_informed",
        "additive_alpha": float(additive_alpha),
        "additive_validation_mse": float(additive_score),
        "additive_fit_scope": "inner_fitting_masks_only",
        "additive_candidates": additive_candidates,
        "centered_inner_fit_target_rms": centered_fit_rms,
        "initializer_jitter_fraction": INITIALIZER_JITTER_FRACTION,
        **initializer_diagnostics[rank, restart],
        "n_configs_evaluated": len(candidate_diagnostics),
        "n_candidates_finite": sum(x["status"] == "finite_validation_candidate" for x in candidate_diagnostics),
        "n_candidates_nonfinite": sum(x["status"] != "finite_validation_candidate" for x in candidate_diagnostics),
        "n_solve_failures": None,
        "solve_failure_count_note": "The unchanged historical ALS catches solve exceptions internally; exact count unavailable.",
        "candidate_diagnostics": candidate_diagnostics,
        "used_historical_fallback": bool(used_fallback),
        "reconstructed_inner_validation_mse": float(reconstructed_validation_mse),
        "all_observed_refit_sweeps": int(original.REFIT_SWEEPS),
        "prediction_is_finite": bool(np.all(np.isfinite(prediction))),
    }
    return original.Fit(prediction, choice)
