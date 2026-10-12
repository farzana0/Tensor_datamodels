#!/usr/bin/env python3
"""Fit and freeze D=12 query-specific surrogates before target retraining."""
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

ROOT = Path("results/original_records/d12")
ALPHAS = tuple(float(x) for x in np.logspace(-5, 2, 13))
RANKS = (2, 4, 8, 16, 32)
LADDER_RANKS = (2, 4, 8, 16, 32, 64)
REGULARIZATION = ((1e-4, 1e-5), (1e-3, 1e-4), (1e-2, 1e-3))
TT_ORDERS = tuple(dict.fromkeys((tuple(range(D)),) + tuple(
    tuple(g for pair in pairs for g in pair) + tuple(g for g in range(D) if all(g not in pair for pair in pairs))
    for pairs in PROVIDERS[:3])))


def bits(masks: np.ndarray) -> np.ndarray:
    return ((masks[:, None] >> np.arange(D)) & 1).astype(np.float64)


def terms(degree: int) -> list[tuple[int, ...]]:
    return [term for order in range(1, degree + 1) for term in itertools.combinations(range(D), order)]


def design(masks: np.ndarray, term_list: list[tuple[int, ...]]) -> np.ndarray:
    x = bits(masks)
    return np.column_stack([x[:, term].prod(1) for term in term_list])


def cardinality_design(masks: np.ndarray) -> np.ndarray:
    size = np.asarray([int(mask).bit_count() for mask in masks], dtype=float)
    z = (size - D / 2) / (D / 2)
    return np.column_stack((np.ones(len(masks)), z, z ** 2, z ** 3))


def size_weights(masks: np.ndarray) -> np.ndarray:
    sizes = np.asarray([int(mask).bit_count() for mask in masks])
    counts = {size: int(np.sum(sizes == size)) for size in np.unique(sizes)}
    value = np.asarray([1 / counts[size] for size in sizes])
    return value / value.mean()


def ridge(x: np.ndarray, y: np.ndarray, weight: np.ndarray, alpha: float) -> np.ndarray:
    root = np.sqrt(weight)
    xw, yw = x * root[:, None], y * root[:, None]
    if x.shape[1] <= x.shape[0]:
        return np.linalg.solve(xw.T @ xw + alpha * np.eye(x.shape[1]), xw.T @ yw)
    return xw.T @ np.linalg.solve(xw @ xw.T + alpha * np.eye(x.shape[0]), yw)


def cell_path(mask: int) -> Path:
    gate = ROOT / "gate/seed31" / f"mask_{mask}" / f"seed31_mask{mask}.json"
    observed = ROOT / "observed/seed31" / f"mask_{mask}" / f"seed31_mask{mask}.json"
    return gate if gate.exists() else observed


def load_output(mask: int) -> np.ndarray:
    value = np.asarray(json.loads(cell_path(mask).read_text())["target_log_probability_by_query"], dtype=float)
    if value.shape != (Q,):
        raise ValueError(f"mask {mask}: expected {Q} query outputs, got {value.shape}")
    return value


def split_masks(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260912)
    fit, validation = [], []
    for size in sorted({int(mask).bit_count() for mask in masks}):
        group = rng.permutation([int(mask) for mask in masks if int(mask).bit_count() == size])
        count = max(1, round(.2 * len(group))) if len(group) >= 3 else 0
        validation.extend(group[:count]); fit.extend(group[count:])
    return np.asarray(fit, dtype=int), np.asarray(validation, dtype=int)


def coefficient_tensor(term_list: list[tuple[int, ...]], coefficient: np.ndarray) -> np.ndarray:
    value = np.zeros((2,) * D + (coefficient.shape[1],), dtype=float)
    for index, term in enumerate(term_list):
        value[tuple(int(axis in term) for axis in range(D))] = coefficient[index]
    return value


def tt_svd(tensor: np.ndarray, max_rank: int) -> np.ndarray:
    shape, cores, remainder, left = tensor.shape, [], tensor.copy(), 1
    for physical in shape[:-1]:
        matrix = remainder.reshape(left * physical, -1)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = min(max_rank, len(singular))
        cores.append(u[:, :rank].reshape(left, physical, rank))
        remainder, left = singular[:rank, None] * vh[:rank], rank
    cores.append(remainder.reshape(left, shape[-1], 1))
    value = cores[0]
    for core in cores[1:]:
        value = np.tensordot(value, core, axes=([-1], [0]))
    return value.reshape(shape)


def separate_output_tt_svd(tensor: np.ndarray, max_rank: int) -> np.ndarray:
    """Compress each query coefficient tensor independently."""
    return np.concatenate([tt_svd(tensor[..., query:query + 1], max_rank)
                           for query in range(tensor.shape[-1])], axis=-1)


def contract(tensor: np.ndarray, masks: np.ndarray) -> np.ndarray:
    output = []
    for row in bits(masks):
        value = tensor
        for coordinate in row:
            value = np.tensordot(np.asarray([1.0, coordinate]), value, axes=([0], [0]))
        output.append(value)
    return np.asarray(output)


class QueryCP(nn.Module):
    def __init__(self, rank: int, seed: int):
        super().__init__(); generator = torch.Generator().manual_seed(seed)
        factor = torch.zeros(Q, D, 2, rank, dtype=torch.float64)
        factor[:, :, 0, :] = 1.0
        factor[:, :, 1, :] = .02 * torch.randn(Q, D, rank, generator=generator, dtype=torch.float64)
        self.factor = nn.Parameter(factor)
        self.output = nn.Parameter(torch.randn(Q, rank, generator=generator, dtype=torch.float64) / np.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        legs = torch.stack((torch.ones_like(x), 1 - 2 * x), -1)
        local = torch.einsum("ndi,qdir->nqdr", legs, self.factor)
        return torch.einsum("nqr,qr->nq", local.prod(2), self.output)


class QueryTT(nn.Module):
    def __init__(self, rank: int, seed: int):
        super().__init__(); generator = torch.Generator().manual_seed(seed)
        bonds = [1] + [min(rank, 2 ** index) for index in range(1, D + 1)]
        cores = []
        for i in range(D):
            core = .02 * torch.randn(Q, bonds[i], 2, bonds[i + 1], generator=generator, dtype=torch.float64)
            core[:, :, 0, :] = 0
            diagonal = torch.arange(min(bonds[i], bonds[i + 1]))
            core[:, diagonal, 0, diagonal] = 1.0
            cores.append(nn.Parameter(core))
        self.cores = nn.ParameterList(cores)
        self.output = nn.Parameter(torch.randn(Q, bonds[-1], generator=generator,
                                               dtype=torch.float64) / np.sqrt(bonds[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leg, state = 1 - 2 * x, torch.ones((len(x), Q, 1), dtype=x.dtype, device=x.device)
        for index, core in enumerate(self.cores):
            local = core[:, :, 0, :][None] + leg[:, None, index, None, None] * core[:, :, 1, :][None]
            state = torch.einsum("nql,nqlr->nqr", state, local)
        return torch.einsum("nqr,qr->nq", state, self.output)


class QueryHT(nn.Module):
    def __init__(self, rank: int, seed: int):
        super().__init__(); generator = torch.Generator().manual_seed(seed)
        leaf_rank = min(2, rank)
        leaves = []
        for _ in range(D):
            leaf = .02 * torch.randn(Q, 2, leaf_rank, generator=generator, dtype=torch.float64)
            leaf[:, 0, :] = 1.0
            leaves.append(nn.Parameter(leaf))
        self.leaves = nn.ParameterList(leaves)
        widths, transfers = [leaf_rank] * D, []
        while len(widths) > 1:
            next_widths = []
            for index in range(0, len(widths) - 1, 2):
                transfer = .02 * torch.randn(Q, widths[index], widths[index + 1], rank,
                                              generator=generator, dtype=torch.float64)
                diagonal = torch.arange(min(widths[index], widths[index + 1], rank))
                transfer[:, diagonal, diagonal, diagonal] = 1.0
                transfers.append(nn.Parameter(transfer))
                next_widths.append(rank)
            if len(widths) % 2: next_widths.append(widths[-1])
            widths = next_widths
        self.transfers = nn.ParameterList(transfers)
        self.output = nn.Parameter(torch.randn(Q, widths[0], generator=generator,
                                               dtype=torch.float64) / np.sqrt(widths[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        legs = torch.stack((torch.ones_like(x), 1 - 2 * x), -1)
        values = [torch.einsum("ni,qir->nqr", legs[:, index], self.leaves[index]) for index in range(D)]
        cursor = 0
        while len(values) > 1:
            nxt = []
            for index in range(0, len(values) - 1, 2):
                nxt.append(torch.einsum("nqa,nqb,qabr->nqr", values[index], values[index + 1],
                                        self.transfers[cursor])); cursor += 1
            if len(values) % 2: nxt.append(values[-1])
            values = nxt
        return torch.einsum("nqr,qr->nq", values[0], self.output)


def make_model(family: str, rank: int, seed: int) -> nn.Module:
    return {"cp": QueryCP, "tt": QueryTT, "ht": QueryHT}[family](rank, seed)


def fit_tn(family: str, rank: int, xfit: np.ndarray, yfit: np.ndarray, weight: np.ndarray,
           xval: np.ndarray, yval: np.ndarray, xtarget: np.ndarray, seed: int,
           weight_decay: float, functional_penalty: float, order: tuple[int, ...]) -> tuple[float, np.ndarray, int]:
    xfit, xval, xtarget = xfit[:, order], xval[:, order], xtarget[:, order]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(family, rank, seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=weight_decay)
    tx, ty, tw = map(lambda a: torch.as_tensor(a, dtype=torch.float64, device=device), (xfit, yfit, weight))
    tv = torch.as_tensor(xval, dtype=torch.float64, device=device)
    vy = torch.as_tensor(yval, dtype=torch.float64, device=device)
    tt = torch.as_tensor(xtarget, dtype=torch.float64, device=device)
    rng = np.random.default_rng(5000 + seed)
    regularizer = torch.as_tensor(bits(rng.choice(1 << D, 128, replace=False))[:, order],
                                  dtype=torch.float64, device=device)
    best_mse, best_state, best_step, stale = float("inf"), None, 0, 0
    for step in range(201):
        if step:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tx)
            loss = torch.mean(tw[:, None] * (prediction - ty) ** 2)
            loss = loss + functional_penalty * torch.mean(model(regularizer) ** 2)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10); optimizer.step()
        if step % 10 == 0:
            with torch.no_grad(): mse = float(torch.mean((model(tv) - vy) ** 2))
            if mse < best_mse - 1e-8:
                best_mse, best_state, best_step, stale = mse, copy.deepcopy(model.state_dict()), step, 0
            else: stale += 1
            if stale >= 6: break
    model.load_state_dict(best_state)
    with torch.no_grad(): prediction = model(tt).cpu().numpy()
    return best_mse, prediction, best_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-budget", type=int, choices=(32, 48, 64, 96, 128, 160), required=True)
    args = parser.parse_args()
    out = ROOT / "fits" / f"budget_{args.new_budget}"
    out.mkdir(parents=True, exist_ok=True)
    gate_masks = [mask for mask in REQUIRED_MASKS if mask.bit_count() != 3]
    new_masks = [int(x) for x in (ROOT / "configs/new_observed_masks.txt").read_text().split()]
    observed = np.asarray(gate_masks + new_masks[:args.new_budget], dtype=int)
    missing = [int(mask) for mask in observed if not cell_path(int(mask)).exists()]
    if missing: raise RuntimeError(f"missing {len(missing)} observed cells; first {missing[:8]}")
    visible_target = list((ROOT / "target/seed31").glob("mask_*/*.json"))
    if visible_target: raise RuntimeError("target outcomes visible before prediction freeze")
    existing_result, existing_prediction = out / "frozen_result.json", out / "frozen_predictions.npz"
    if existing_result.exists() and existing_prediction.exists():
        prior = json.loads(existing_result.read_text())
        current = {str(mask): hashlib.sha256(cell_path(int(mask)).read_bytes()).hexdigest() for mask in observed}
        if prior.get("source_hashes") == current and prior.get("status") == "FROZEN_BEFORE_TARGET":
            print(json.dumps({"budget": args.new_budget, "reused_frozen_artifact": True,
                              "freeze_sha256": prior["freeze_sha256"]}))
            return
    outputs = {int(mask): load_output(int(mask)) for mask in observed}
    fit_masks, validation_masks = split_masks(observed)
    yfit = np.vstack([outputs[int(mask)] for mask in fit_masks])
    yval = np.vstack([outputs[int(mask)] for mask in validation_masks])
    target_masks = np.asarray([int(x) for x in (ROOT / "configs/frozen_target_masks.txt").read_text().split()])
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
    # Center interaction residuals exactly within each observed cardinality.
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

    previous, accepted, ladder_path = None, None, []
    for degree in range(1, D + 1):
        term_list, candidates, ridge_candidates = terms(degree), [], []
        xf, xv = design(fit_masks, term_list), design(validation_masks, term_list)
        for alpha in ALPHAS:
            coef = ridge(xf, residual_fit, weight, alpha)
            ridge_candidates.append((float(np.mean((xv @ coef - residual_val) ** 2)), alpha, coef))
        # Compression is evaluated for the three strongest regularizers rather
        # than silently fixing one ridge penalty before the TN rank path.
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
        if not keep: break
        previous, accepted = best[0], (degree, best)
    degree, best = accepted
    predictions["degree_ladder_tt"] = card_all + contract(best[3], evaluation_masks)
    choices["degree_ladder_tt"] = {"degree": degree, "rank": best[1], "alpha": best[2],
                                    "validation_mse": best[0]}

    scale = np.maximum(residual_fit.std(0), .05)
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

    rng = np.random.default_rng(20260912)
    predictions["random"] = rng.standard_normal((len(evaluation_masks), Q))
    nominations = {method: [[int(target_masks[index]) for index in np.lexsort((target_masks, -value[target_masks, q]))[:8]]
                             for q in range(Q)] for method, value in predictions.items()}
    source_hashes = {str(mask): hashlib.sha256(cell_path(int(mask)).read_bytes()).hexdigest() for mask in observed}
    np.savez_compressed(out / "frozen_predictions.npz", masks=evaluation_masks, **predictions)
    result = {"status": "FROZEN_BEFORE_TARGET", "new_budget": args.new_budget,
              "total_acquisition_cost": len(REQUIRED_MASKS) + args.new_budget,
              "fit_cell_count": len(observed), "fit_masks": fit_masks.tolist(),
              "validation_masks": validation_masks.tolist(), "choices": choices,
              "ladder_path": ladder_path, "nominations": nominations,
              "source_hashes": source_hashes, "query_outputs": Q,
              "per_query_input_outputs_retained_in_source_cells": True,
              "predictions_cover_full_unopened_4096_cell_cube": True,
              "size_three_gate_masks_excluded": [21, 7], "target_outcomes_loaded": False}
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["freeze_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    (out / "frozen_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"budget": args.new_budget, "freeze_sha256": result["freeze_sha256"],
                      "methods": sorted(predictions), "ladder": choices["degree_ladder_tt"]}, indent=2))


if __name__ == "__main__":
    main()
