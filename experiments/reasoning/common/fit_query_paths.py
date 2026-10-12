#!/usr/bin/env python3
"""Fit and freeze query-conditioned surrogates before opening k=3 outcomes."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.reasoning.common.transformer_query_paths import D, PROVIDERS, Q

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("QUERY_PATH_ROOT",
    REPO / "results/original_records/d8_query_paths"))
OUT = ROOT / "fits"
ALPHAS = tuple(float(x) for x in np.logspace(-6, 3, 19))
RANKS = (2, 4, 8, 16, 32)
LADDER_RANKS = (2, 4, 8, 16, 32, 64, 128, 256)
FAMILIES = ("cp", "tt", "ht")
REGULARIZATION_PATH = ((1e-4, 1e-5), (1e-3, 1e-4), (1e-2, 1e-3))
CALIBRATION_MASKS = {28, 84}


def candidate_tt_orders() -> tuple[tuple[int, ...], ...]:
    orders = [tuple(range(D))]
    for pairs in PROVIDERS:
        prefix = tuple(group for pair in pairs for group in pair)
        orders.append(prefix + tuple(group for group in range(D) if group not in prefix))
    return tuple(dict.fromkeys(orders))


TT_ORDERS = candidate_tt_orders()


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


def weights(masks: np.ndarray) -> np.ndarray:
    sizes = np.asarray([int(mask).bit_count() for mask in masks])
    counts = {size: int(np.sum(sizes == size)) for size in np.unique(sizes)}
    result = np.asarray([1 / counts[size] for size in sizes])
    return result / result.mean()


def ridge(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, alpha: float) -> np.ndarray:
    root = np.sqrt(sample_weight)
    xw = x * root[:, None]
    return np.linalg.solve(xw.T @ xw + alpha * np.eye(x.shape[1]), xw.T @ (y * root[:, None]))


def cell_path(mask: int) -> Path:
    gate = ROOT / "gate/seed31" / f"mask_{mask}" / f"seed31_mask{mask}.json"
    observed = ROOT / "observed/seed31" / f"mask_{mask}" / f"seed31_mask{mask}.json"
    return gate if gate.exists() else observed


def load_outputs(mask: int) -> np.ndarray:
    row = json.loads(cell_path(mask).read_text())
    value = np.asarray(row["target_log_probability_by_query_input"], dtype=np.float64)
    if value.shape != (Q * 31,):
        raise ValueError(f"mask {mask}: wrong output shape {value.shape}")
    return value


def stratified_split(masks: np.ndarray, seed: int = 20260912) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    fit, validation = [], []
    for size in sorted({int(mask).bit_count() for mask in masks}):
        group = generator.permutation([int(mask) for mask in masks if int(mask).bit_count() == size])
        count = max(1, round(.2 * len(group))) if len(group) > 1 else 0
        validation.extend(group[:count]); fit.extend(group[count:])
    return np.asarray(sorted(fit)), np.asarray(sorted(validation))


def coefficient_tensor(term_list: list[tuple[int, ...]], coefficient: np.ndarray) -> np.ndarray:
    result = np.zeros((2,) * D + (coefficient.shape[1],), dtype=np.float64)
    for index, term in enumerate(term_list):
        result[tuple(int(axis in term) for axis in range(D))] = coefficient[index]
    return result


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


def contract(tensor: np.ndarray, masks: np.ndarray) -> np.ndarray:
    output = []
    for row in bits(masks):
        value = tensor
        for coordinate in row:
            value = np.tensordot(np.array([1.0, coordinate]), value, axes=([0], [0]))
        output.append(value)
    return np.asarray(output)


class QueryCP(nn.Module):
    def __init__(self, rank: int, outputs: int, seed: int):
        super().__init__(); generator = torch.Generator().manual_seed(seed)
        self.factor = nn.Parameter(torch.randn(D, 2, rank, generator=generator, dtype=torch.float64) / np.sqrt(D))
        self.output = nn.Parameter(torch.randn(rank, outputs, generator=generator, dtype=torch.float64) / np.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        legs = torch.stack((torch.ones_like(x), 1 - 2 * x), -1)
        local = torch.einsum("ndi,dir->ndr", legs, self.factor)
        return local.prod(1) @ self.output


class QueryTT(nn.Module):
    def __init__(self, rank: int, outputs: int, seed: int):
        super().__init__(); generator = torch.Generator().manual_seed(seed)
        bonds = [1] + [min(rank, 2 ** index) for index in range(1, D + 1)]
        self.cores = nn.ParameterList([nn.Parameter(torch.randn(bonds[i], 2, bonds[i + 1],
            generator=generator, dtype=torch.float64) / np.sqrt(2 * bonds[i])) for i in range(D)])
        self.output = nn.Parameter(torch.randn(bonds[-1], outputs, generator=generator,
                                               dtype=torch.float64) / np.sqrt(bonds[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leg = 1 - 2 * x; state = torch.ones((len(x), 1), dtype=x.dtype)
        for index, core in enumerate(self.cores):
            local = core[:, 0] + leg[:, index, None, None] * core[:, 1]
            state = torch.bmm(state[:, None], local).squeeze(1)
        return state @ self.output


class QueryHT(nn.Module):
    def __init__(self, rank: int, outputs: int, seed: int):
        super().__init__(); generator = torch.Generator().manual_seed(seed)
        self.leaves = nn.ParameterList([nn.Parameter(torch.randn(2, min(2, rank), generator=generator,
            dtype=torch.float64) / np.sqrt(2)) for _ in range(D)])
        self.level1 = nn.ParameterList([nn.Parameter(torch.randn(min(2, rank), min(2, rank), rank,
            generator=generator, dtype=torch.float64) / 2) for _ in range(4)])
        self.level2 = nn.ParameterList([nn.Parameter(torch.randn(rank, rank, rank, generator=generator,
            dtype=torch.float64) / rank) for _ in range(2)])
        self.root = nn.Parameter(torch.randn(rank, rank, rank, generator=generator, dtype=torch.float64) / rank)
        self.output = nn.Parameter(torch.randn(rank, outputs, generator=generator,
                                               dtype=torch.float64) / np.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        legs = torch.stack((torch.ones_like(x), 1 - 2 * x), -1)
        values = [legs[:, i] @ self.leaves[i] for i in range(D)]
        values = [torch.einsum("na,nb,abr->nr", values[2*i], values[2*i+1], self.level1[i]) for i in range(4)]
        values = [torch.einsum("na,nb,abr->nr", values[2*i], values[2*i+1], self.level2[i]) for i in range(2)]
        value = torch.einsum("na,nb,abr->nr", values[0], values[1], self.root)
        return value @ self.output


def make_model(family: str, rank: int, outputs: int, seed: int) -> nn.Module:
    return {"cp": QueryCP, "tt": QueryTT, "ht": QueryHT}[family](rank, outputs, seed)


def fit_tn(family: str, rank: int, xfit: np.ndarray, yfit: np.ndarray, wfit: np.ndarray,
           xval: np.ndarray, yval: np.ndarray, xall: np.ndarray, seed: int,
           weight_decay: float, functional_penalty: float,
           order: tuple[int, ...] | None = None) -> tuple[float, np.ndarray, int]:
    order = tuple(range(D)) if order is None else order
    xfit, xval, xall = xfit[:, order], xval[:, order], xall[:, order]
    model = make_model(family, rank, yfit.shape[1], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=weight_decay)
    tx, ty, tw = map(lambda value: torch.as_tensor(value, dtype=torch.float64), (xfit, yfit, wfit))
    tv = torch.as_tensor(xval, dtype=torch.float64); ta = torch.as_tensor(xall, dtype=torch.float64)
    best_mse, best_state, best_step, stale = float("inf"), None, 0, 0
    for step in range(401):
        if step:
            optimizer.zero_grad(set_to_none=True)
            pred = model(tx)
            functional = model(torch.as_tensor(bits(np.arange(1 << D)), dtype=torch.float64))
            loss = (torch.mean(tw[:, None] * (pred - ty) ** 2)
                    + functional_penalty * torch.mean(functional ** 2))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10); optimizer.step()
        if step % 10 == 0:
            with torch.no_grad(): mse = float(torch.mean((model(tv) - torch.as_tensor(yval)) ** 2))
            if mse < best_mse - 1e-8:
                best_mse, best_state, best_step, stale = mse, copy.deepcopy(model.state_dict()), step, 0
            else: stale += 1
            if stale >= 10: break
    model.load_state_dict(best_state)
    with torch.no_grad(): prediction = model(ta).numpy()
    return best_mse, prediction, best_step


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_non_target = np.asarray([mask for mask in range(1 << D) if mask.bit_count() != 3], dtype=np.int64)
    missing = [int(mask) for mask in all_non_target if not cell_path(int(mask)).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} non-target cells missing; first: {missing[:10]}")
    if any((ROOT / "target/seed31" / f"mask_{mask}").exists()
           for mask in range(1 << D) if mask.bit_count() == 3 and mask not in CALIBRATION_MASKS):
        raise RuntimeError("target outcomes already visible; use the frozen fit artifact rather than refitting")
    output = np.vstack([load_outputs(int(mask)) for mask in all_non_target])
    lookup = {int(mask): index for index, mask in enumerate(all_non_target)}
    fit_masks, validation_masks = stratified_split(all_non_target)
    yfit = output[[lookup[int(mask)] for mask in fit_masks]]
    yvalidation = output[[lookup[int(mask)] for mask in validation_masks]]
    all_masks = np.arange(1 << D, dtype=np.int64)

    card_candidates = []
    for alpha in ALPHAS:
        coefficient = ridge(cardinality_design(fit_masks), yfit, weights(fit_masks), alpha)
        prediction = cardinality_design(validation_masks) @ coefficient
        card_candidates.append((float(np.mean((prediction - yvalidation) ** 2)), alpha, coefficient))
    card_best = min(card_candidates, key=lambda item: item[:2])
    card_all = cardinality_design(all_masks) @ card_best[2]
    # Remove any remaining size-only offset using fit cells alone.  This makes
    # the interaction target exactly mean-zero inside every observed size and
    # prevents abundant large coalitions from masquerading as interactions.
    raw_fit_residual = yfit - card_all[fit_masks]
    size_offsets = np.zeros((D + 1, output.shape[1]), dtype=np.float64)
    for size in range(D + 1):
        index = np.asarray([int(mask).bit_count() == size for mask in fit_masks])
        if np.any(index): size_offsets[size] = raw_fit_residual[index].mean(0)
    card_all = card_all + np.vstack([size_offsets[int(mask).bit_count()] for mask in all_masks])
    residual_fit, residual_validation = yfit - card_all[fit_masks], yvalidation - card_all[validation_masks]
    random_generator = np.random.default_rng(20260912)
    predictions, choices = {
        "cardinality": card_all,
        "random": random_generator.standard_normal((1 << D, Q * 31)),
    }, {"random": {"seed": 20260912, "uses_coalition_outcomes": False}}

    for degree in (1, 2, 3):
        term_list = terms(degree); candidates = []
        for alpha in ALPHAS:
            coefficient = ridge(design(fit_masks, term_list), residual_fit, weights(fit_masks), alpha)
            pred = design(validation_masks, term_list) @ coefficient
            candidates.append((float(np.mean((pred - residual_validation) ** 2)), alpha, coefficient))
        best = min(candidates, key=lambda item: item[:2])
        name = "linear" if degree == 1 else f"degree{degree}_ridge"
        predictions[name] = card_all + design(all_masks, term_list) @ best[2]
        choices[name] = {"degree": degree, "alpha": best[1], "validation_mse": best[0]}

    # Degree-preserving output-mode TT compression of cumulative polynomial fits.
    previous, accepted, ladder_path = None, None, []
    for degree in range(1, D + 1):
        term_list = terms(degree); candidates = []
        for alpha in ALPHAS:
            coefficient = ridge(design(fit_masks, term_list), residual_fit, weights(fit_masks), alpha)
            dense = coefficient_tensor(term_list, coefficient)
            for rank in LADDER_RANKS:
                compressed = tt_svd(dense, rank)
                pred = contract(compressed, validation_masks)
                candidates.append((float(np.mean((pred - residual_validation) ** 2)), rank, alpha, compressed))
        best = min(candidates, key=lambda item: item[:3]); gain = None if previous is None else (previous-best[0])/previous
        # Degree three is the guaranteed base class.  Only higher degrees are
        # subject to the validation-gain stopping rule.
        keep = degree <= 3 or gain >= .01
        ladder_path.append({"degree": degree, "rank": best[1], "alpha": best[2], "validation_mse": best[0],
                            "relative_gain": gain, "accepted": keep})
        if not keep: break
        previous, accepted = best[0], (degree, best)
    degree, best = accepted
    predictions["degree_ladder_tt"] = card_all + contract(best[3], all_masks)
    choices["degree_ladder_tt"] = {"degree": degree, "rank": best[1], "alpha": best[2], "validation_mse": best[0]}

    scale = np.maximum(residual_fit.std(0), .05)
    # Direct TN and a hierarchical linear-plus-TN residual use the same frozen
    # rank/regularization path.  The latter contains the fitted Linear model
    # exactly when the interaction residual is shrunk to zero.
    linear_residual = residual_fit - (predictions["linear"] - card_all)[fit_masks]
    linear_validation_residual = residual_validation - (predictions["linear"] - card_all)[validation_masks]
    for family in FAMILIES:
        candidates = []
        for rank in RANKS:
            orders = TT_ORDERS if family == "tt" else (tuple(range(D)),)
            for order_index, order in enumerate(orders):
                for weight_decay, functional_penalty in REGULARIZATION_PATH:
                    mse, pred, step = fit_tn(family, rank, bits(fit_masks), residual_fit / scale,
                        weights(fit_masks), bits(validation_masks), residual_validation / scale,
                        bits(all_masks), 1000 + rank + 100 * FAMILIES.index(family) + 1000 * order_index,
                        weight_decay, functional_penalty, order)
                    raw_mse = float(np.mean((pred[validation_masks] * scale - residual_validation) ** 2))
                    candidates.append((raw_mse, rank, weight_decay, functional_penalty, order, pred, step))
        best = min(candidates, key=lambda item: item[:2])
        predictions[family] = card_all + best[5] * scale
        choices[family] = {"rank": best[1], "weight_decay": best[2], "functional_penalty": best[3],
                           "order": list(best[4]), "validation_mse": best[0], "early_stop_step": best[6]}

        hybrid_scale = np.maximum(linear_residual.std(0), .05)
        candidates = []
        for rank in RANKS:
            orders = TT_ORDERS if family == "tt" else (tuple(range(D)),)
            for order_index, order in enumerate(orders):
                for weight_decay, functional_penalty in REGULARIZATION_PATH:
                    _, pred, step = fit_tn(family, rank, bits(fit_masks), linear_residual / hybrid_scale,
                        weights(fit_masks), bits(validation_masks), linear_validation_residual / hybrid_scale,
                        bits(all_masks), 2000 + rank + 100 * FAMILIES.index(family) + 1000 * order_index,
                        weight_decay, functional_penalty, order)
                    raw_mse = float(np.mean((pred[validation_masks] * hybrid_scale
                                             - linear_validation_residual) ** 2))
                    candidates.append((raw_mse, rank, weight_decay, functional_penalty, order, pred, step))
        best = min(candidates, key=lambda item: item[:2])
        name = f"linear_plus_{family}"
        predictions[name] = predictions["linear"] + best[5] * hybrid_scale
        choices[name] = {"rank": best[1], "weight_decay": best[2], "functional_penalty": best[3],
                         "order": list(best[4]), "validation_mse": best[0], "early_stop_step": best[6]}

    target_masks = np.loadtxt(ROOT / "configs/frozen_target_masks.txt", dtype=np.int64)
    nomination = {}
    for method, prediction in predictions.items():
        per_query = []
        for query_input in range(Q * 31):
            order = target_masks[np.lexsort((target_masks, -prediction[target_masks, query_input]))]
            per_query.append([int(mask) for mask in order[:8]])
        nomination[method] = per_query
    source_hashes = {str(mask): hashlib.sha256(cell_path(int(mask)).read_bytes()).hexdigest()
                     for mask in all_non_target}
    np.savez_compressed(OUT / "frozen_predictions.npz", masks=all_masks, **predictions)
    result = {"status": "FROZEN_BEFORE_TARGET", "fit_masks": fit_masks.tolist(),
        "validation_masks": validation_masks.tolist(), "target_masks": target_masks.tolist(),
        "cardinality": {"alpha": card_best[1],
            "validation_mse": float(np.mean(residual_validation ** 2)),
            "residual_centered_exactly_within_fit_size": True},
        "choices": choices, "ladder_path": ladder_path, "nominations": nomination,
        "source_hashes": source_hashes, "query_outputs": Q * 31, "target_outcomes_loaded": False}
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["freeze_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    (OUT / "frozen_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"freeze_sha256": result["freeze_sha256"], "choices": choices}, indent=2))


if __name__ == "__main__":
    main()
