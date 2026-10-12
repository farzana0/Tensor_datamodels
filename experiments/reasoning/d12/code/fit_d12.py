#!/usr/bin/env python3
"""d=12 study: fit and freeze d=12 query-specific surrogates, per learner seed.

Copy of `experiments/reasoning/common/fit_query_paths_d12.py` with the
scientific protocol unchanged (cardinality baseline, within-size centring,
inverse-frequency size weights, 20%-within-size validation split, the same ridge grid,
ranks, regularisation pairs, TT orders and degree-ladder acceptance rule) and four
changes, all recorded as deviations:

* seed and result root are parameters, so seeds 43 and 59 run the same pipeline;
* `rbf_kernel_ridge`, `sparse_degree3_omp` and `influence_loo` are added as baselines;
* nominations are emitted for every cardinality slice in {3, 4, 5, 6}, not only size 3;
* the freeze guard checks this workstream's own result root.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.reasoning.common.evaluate_query_paths_d12_gate import REQUIRED_MASKS
from experiments.reasoning.common.transformer_query_paths_d12 import D, PROVIDERS, Q
from experiments.reasoning.common.fit_query_paths_d12 import (
    ALPHAS, LADDER_RANKS, RANKS, REGULARIZATION, TT_ORDERS,
    bits, cardinality_design, coefficient_tensor, contract, design, fit_tn,
    ridge, separate_output_tt_svd, size_weights, split_masks, terms,
)

SLICES = (3, 4, 5, 6)
NOMINATIONS_RETAINED = 8
RBF_GAMMAS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
OMP_SPARSITY = tuple(range(1, 61))


def cell_path(roots: list[Path], seed: int, mask: int) -> Path | None:
    for root in roots:
        for split in ("gate", "observed", "loo"):
            path = root / split / f"seed{seed}" / f"mask_{mask}" / f"seed{seed}_mask{mask}.json"
            if path.exists():
                return path
    return None


def load_output(roots: list[Path], seed: int, mask: int) -> np.ndarray:
    path = cell_path(roots, seed, mask)
    if path is None:
        raise FileNotFoundError(f"seed {seed} mask {mask}")
    value = np.asarray(json.loads(path.read_text())["target_log_probability_by_query"], dtype=float)
    if value.shape != (Q,):
        raise ValueError(f"mask {mask}: expected {Q} query outputs, got {value.shape}")
    return value


def weighted_kernel_ridge(kfit: np.ndarray, y: np.ndarray, weight: np.ndarray,
                          alpha: float) -> np.ndarray:
    return np.linalg.solve(kfit + alpha * np.diag(1.0 / weight), y)


def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    square = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * square)


def omp_weighted(x: np.ndarray, y: np.ndarray, weight: np.ndarray, sparsity: int) -> np.ndarray:
    """Greedy orthogonal matching pursuit, one support per output column."""
    root = np.sqrt(weight)[:, None]
    xw, yw = x * root, y * root
    coefficient = np.zeros((x.shape[1], y.shape[1]))
    for column in range(y.shape[1]):
        residual, support = yw[:, column].copy(), []
        norms = np.linalg.norm(xw, axis=0)
        norms[norms == 0] = 1.0
        for _ in range(sparsity):
            score = np.abs(xw.T @ residual) / norms
            score[support] = -np.inf
            best = int(np.argmax(score))
            if not np.isfinite(score[best]):
                break
            support.append(best)
            solution, *_ = np.linalg.lstsq(xw[:, support], yw[:, column], rcond=None)
            residual = yw[:, column] - xw[:, support] @ solution
        if support:
            solution, *_ = np.linalg.lstsq(xw[:, support], yw[:, column], rcond=None)
            coefficient[support, column] = solution
    return coefficient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-budget", type=int, choices=(32, 48, 64, 96, 128, 160), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True, help="this workstream's results/ dir")
    parser.add_argument("--fallback-root", type=Path, help="read-only the development run root (seed 31)")
    parser.add_argument("--nonblind", action="store_true",
                        help="required for seed 31, whose target outcomes already exist")
    parser.add_argument("--preregistration-sha256", default="")
    args = parser.parse_args()

    roots = [args.root] + ([args.fallback_root] if args.fallback_root else [])
    out = args.root / "fits" / f"seed{args.seed}" / f"budget_{args.new_budget}"
    out.mkdir(parents=True, exist_ok=True)

    visible = list((args.root / "target" / f"seed{args.seed}").glob("mask_*/*.json"))
    visible += [p for size in SLICES
                for p in (args.root / f"size{size}_eval" / f"seed{args.seed}").glob("mask_*/*.json")]
    if visible and not args.nonblind:
        raise RuntimeError(f"seed {args.seed}: {len(visible)} evaluation outcomes visible before freeze")

    gate_masks = [mask for mask in REQUIRED_MASKS if mask.bit_count() != 3]
    new_masks = [int(x) for x in (args.root.parent / "configs/new_observed_masks.txt").read_text().split()]
    observed = np.asarray(gate_masks + new_masks[:args.new_budget], dtype=int)
    missing = [int(m) for m in observed if cell_path(roots, args.seed, int(m)) is None]
    if missing:
        raise RuntimeError(f"missing {len(missing)} observed cells; first {missing[:8]}")

    outputs = {int(mask): load_output(roots, args.seed, int(mask)) for mask in observed}
    fit_masks, validation_masks = split_masks(observed)
    yfit = np.vstack([outputs[int(mask)] for mask in fit_masks])
    yval = np.vstack([outputs[int(mask)] for mask in validation_masks])
    evaluation_masks = np.arange(1 << D, dtype=int)
    weight = size_weights(fit_masks)

    card_candidates = []
    for alpha in ALPHAS:
        coef = ridge(cardinality_design(fit_masks), yfit, weight, alpha)
        pred = cardinality_design(validation_masks) @ coef
        card_candidates.append((float(np.mean((pred - yval) ** 2)), alpha, coef))
    card = min(card_candidates, key=lambda row: row[:2])
    card_fit = cardinality_design(fit_masks) @ card[2]
    card_val = cardinality_design(validation_masks) @ card[2]
    card_all = cardinality_design(evaluation_masks) @ card[2]
    offsets = {}
    raw = yfit - card_fit
    for size in set(int(mask).bit_count() for mask in fit_masks):
        offsets[size] = raw[[int(mask).bit_count() == size for mask in fit_masks]].mean(0)
    card_fit += np.vstack([offsets[int(mask).bit_count()] for mask in fit_masks])
    card_val += np.vstack([offsets.get(int(mask).bit_count(), np.zeros(Q)) for mask in validation_masks])
    card_all += np.vstack([offsets.get(int(mask).bit_count(), np.zeros(Q)) for mask in evaluation_masks])
    residual_fit, residual_val = yfit - card_fit, yval - card_val
    predictions, choices = {"cardinality": card_all}, {}

    polynomial = {}
    for degree in (1, 2, 3):
        term_list, candidates = terms(degree), []
        xf, xv = design(fit_masks, term_list), design(validation_masks, term_list)
        for alpha in ALPHAS:
            coef = ridge(xf, residual_fit, weight, alpha)
            candidates.append((float(np.mean((xv @ coef - residual_val) ** 2)), alpha, coef))
        best = min(candidates, key=lambda row: row[:2])
        name = "linear" if degree == 1 else f"degree{degree}_ridge"
        predictions[name] = card_all + design(evaluation_masks, term_list) @ best[2]
        choices[name] = {"alpha": best[1], "validation_mse": best[0]}
        polynomial[degree] = (term_list, best)

    # --- added baseline: sparse degree-3 by orthogonal matching pursuit -----------
    term_list = terms(3)
    xf, xv = design(fit_masks, term_list), design(validation_masks, term_list)
    omp_candidates = []
    for sparsity in OMP_SPARSITY:
        coef = omp_weighted(xf, residual_fit, weight, sparsity)
        omp_candidates.append((float(np.mean((xv @ coef - residual_val) ** 2)), sparsity, coef))
    best = min(omp_candidates, key=lambda row: row[:2])
    predictions["sparse_degree3_omp"] = card_all + design(evaluation_masks, term_list) @ best[2]
    choices["sparse_degree3_omp"] = {"sparsity": best[1], "validation_mse": best[0],
                                     "nonzero_terms": int((best[2] != 0).sum())}

    # --- added baseline: RBF kernel ridge ----------------------------------------
    bf, bv, ball = bits(fit_masks), bits(validation_masks), bits(evaluation_masks)
    rbf_candidates = []
    for gamma in RBF_GAMMAS:
        kfit = rbf_kernel(bf, bf, gamma)
        kval = rbf_kernel(bv, bf, gamma)
        for alpha in ALPHAS:
            dual = weighted_kernel_ridge(kfit, residual_fit, weight, alpha)
            rbf_candidates.append((float(np.mean((kval @ dual - residual_val) ** 2)), gamma, alpha, dual))
    best_rbf = min(rbf_candidates, key=lambda row: row[:3])
    kall = rbf_kernel(ball, bf, best_rbf[1])
    predictions["rbf_kernel_ridge"] = card_all + kall @ best_rbf[3]
    choices["rbf_kernel_ridge"] = {"gamma": best_rbf[1], "alpha": best_rbf[2],
                                   "validation_mse": best_rbf[0]}

    # --- added baseline: exact leave-one-group-out influence ----------------------
    loo_scores, loo_cost = None, None
    full = cell_path(roots, args.seed, 4095)
    loo_masks = [4095 ^ (1 << group) for group in range(D)]
    if full is not None and all(cell_path(roots, args.seed, m) is not None for m in loo_masks):
        full_value = load_output(roots, args.seed, 4095)
        loo_scores = np.vstack([full_value - load_output(roots, args.seed, m) for m in loo_masks])
        predictions["influence_loo"] = card_all + design(evaluation_masks, terms(1)) @ loo_scores
        loo_cost = len(REQUIRED_MASKS) + D
        choices["influence_loo"] = {"acquisition_cost": loo_cost,
                                    "definition": "U(full) - U(full minus group), per query, fixed per-group score"}

    previous, accepted, ladder_path = None, None, []
    for degree in range(1, D + 1):
        term_list, candidates, ridge_candidates = terms(degree), [], []
        xf, xv = design(fit_masks, term_list), design(validation_masks, term_list)
        for alpha in ALPHAS:
            coef = ridge(xf, residual_fit, weight, alpha)
            ridge_candidates.append((float(np.mean((xv @ coef - residual_val) ** 2)), alpha, coef))
        for _, alpha, coef in sorted(ridge_candidates, key=lambda row: row[:2])[:3]:
            dense = coefficient_tensor(term_list, coef)
            for rank in LADDER_RANKS:
                compressed = separate_output_tt_svd(dense, rank)
                value = contract(compressed, validation_masks)
                candidates.append((float(np.mean((value - residual_val) ** 2)), rank, alpha, compressed))
        best = min(candidates, key=lambda row: row[:3])
        gain = None if previous is None else (previous - best[0]) / max(previous, 1e-12)
        keep = degree <= 3 or gain >= .01
        ladder_path.append({"degree": degree, "rank": best[1], "alpha": best[2],
                            "validation_mse": best[0], "relative_gain": gain, "accepted": keep})
        if not keep:
            break
        previous, accepted = best[0], (degree, best)
    degree, best = accepted
    predictions["degree_ladder_tt"] = card_all + contract(best[3], evaluation_masks)
    choices["degree_ladder_tt"] = {"degree": degree, "rank": best[1], "alpha": best[2],
                                   "validation_mse": best[0]}

    linear_fit = design(fit_masks, terms(1)) @ polynomial[1][1][2]
    linear_val = design(validation_masks, terms(1)) @ polynomial[1][1][2]
    families = ("tt",) if args.new_budget < 160 else ("cp", "tt", "ht")
    for family in families:
        orders = TT_ORDERS if family == "tt" else (tuple(range(D)),)
        for hybrid in (False, True):
            base_fit = linear_fit if hybrid else np.zeros_like(residual_fit)
            base_val = linear_val if hybrid else np.zeros_like(residual_val)
            base_target = predictions["linear"] if hybrid else card_all
            target_fit, target_val = residual_fit - base_fit, residual_val - base_val
            local_scale = np.maximum(target_fit.std(0), .05)
            candidates = []
            for rank in RANKS:
                for order_index, order in enumerate(orders):
                    for reg_index, (decay, penalty) in enumerate(REGULARIZATION):
                        mse, pred, step = fit_tn(family, rank, bits(fit_masks), target_fit / local_scale,
                            weight, bits(validation_masks), target_val / local_scale, bits(evaluation_masks),
                            10000 * hybrid + 1000 * order_index + 100 * reg_index + rank,
                            decay, penalty, order)
                        raw_mse = mse * float(np.mean(local_scale ** 2))
                        candidates.append((raw_mse, rank, decay, penalty, order, pred, step))
            best_tn = min(candidates, key=lambda row: row[:2])
            name = f"linear_plus_{family}" if hybrid else family
            predictions[name] = base_target + best_tn[5] * local_scale
            choices[name] = {"rank": best_tn[1], "weight_decay": best_tn[2],
                             "functional_penalty": best_tn[3], "order": list(best_tn[4]),
                             "validation_mse": best_tn[0], "early_stop_step": best_tn[6]}

    rng = np.random.default_rng(20260912 + args.seed)
    predictions["random"] = rng.standard_normal((len(evaluation_masks), Q))

    # --- frozen nominations, per cardinality slice --------------------------------
    # Nominations are restricted to cells this seed has never trained on, so that every
    # evaluated nominee is genuinely held out.  The development run did the same at size 4
    # (SIZE4_NOMINATION_PROTOCOL.excluded_previously_observed_size4_cells, 52 masks).
    trained = set(int(m) for m in observed) | set(int(m) for m in REQUIRED_MASKS)
    trained |= {4095 ^ (1 << group) for group in range(D)}
    slice_masks = {}
    excluded = {}
    for size in SLICES:
        every = [m for m in range(1 << D) if m.bit_count() == size]
        if size == 3:
            # the size-3 slice is retrained exhaustively, so nothing is excluded
            slice_masks[size] = np.asarray(every, dtype=int)
            excluded[str(size)] = []
            continue
        keep = [m for m in every if m not in trained]
        slice_masks[size] = np.asarray(keep, dtype=int)
        excluded[str(size)] = sorted(m for m in every if m in trained)
    nominations = {}
    for size, candidates_masks in slice_masks.items():
        per_method = {}
        for method, value in predictions.items():
            per_method[method] = [[int(candidates_masks[i]) for i in
                                   np.lexsort((candidates_masks, -value[candidates_masks, q]))[:NOMINATIONS_RETAINED]]
                                  for q in range(Q)]
        tt_value = predictions["tt"]
        middle, bottom = [], []
        for q in range(Q):
            order = np.lexsort((candidates_masks, -tt_value[candidates_masks, q]))
            half = len(order) // 2
            middle.append([int(candidates_masks[i]) for i in order[half:half + NOMINATIONS_RETAINED]])
            bottom.append([int(candidates_masks[i]) for i in order[-NOMINATIONS_RETAINED:][::-1]])
        per_method["tt_middle"], per_method["tt_bottom"] = middle, bottom
        nominations[str(size)] = per_method

    source_hashes = {str(mask): hashlib.sha256(cell_path(roots, args.seed, int(mask)).read_bytes()).hexdigest()
                     for mask in observed}
    np.savez_compressed(out / "frozen_predictions.npz", masks=evaluation_masks, **predictions)
    result = {"status": "FROZEN_BEFORE_TARGET", "seed": args.seed, "new_budget": args.new_budget,
              "blind_to_outcomes": not args.nonblind,
              "total_acquisition_cost": len(REQUIRED_MASKS) + args.new_budget,
              "influence_loo_acquisition_cost": loo_cost,
              "fit_cell_count": len(observed), "fit_masks": fit_masks.tolist(),
              "validation_masks": validation_masks.tolist(), "choices": choices,
              "ladder_path": ladder_path, "nominations": nominations,
              "nomination_slices": list(SLICES), "nominations_retained": NOMINATIONS_RETAINED,
              "nomination_candidate_counts": {str(k): int(len(v)) for k, v in slice_masks.items()},
              "nomination_excluded_already_trained": excluded,
              "source_hashes": source_hashes, "query_outputs": Q,
              "preregistration_sha256": args.preregistration_sha256,
              "predictions_cover_full_unopened_4096_cell_cube": True,
              "size_three_gate_masks_excluded": [21, 7], "target_outcomes_loaded": False}
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["freeze_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    (out / "frozen_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"seed": args.seed, "budget": args.new_budget,
                      "freeze_sha256": result["freeze_sha256"], "methods": sorted(predictions),
                      "ladder": choices["degree_ladder_tt"],
                      "omp": choices["sparse_degree3_omp"], "rbf": choices["rbf_kernel_ridge"]}, indent=2))


if __name__ == "__main__":
    main()
