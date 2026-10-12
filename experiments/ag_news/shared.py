"""Shared scientific contracts for the preregistered E1--E6 experiments.

This module intentionally contains no dataset-specific outcome logic.  It
defines the frozen diagnostics, split construction, provenance, and registry
checks that every experiment must pass before its result is aggregatable.
"""

from __future__ import annotations

import csv
import itertools
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.ag_news.queries import (
    banzhaf_values,
    grabisch_roubens_interactions,
    loo_values,
    shapley_values,
)


ROOT = Path(__file__).resolve().parent  # experiments/ag_news
REPORT_ROOT = ROOT / "prereg"
PREREG_PATH = REPORT_ROOT / "prereg.md"
PREREG_SHA256 = "4f4772f119ebbfe4eacf4530aa912557cb2985291bbd99cba81848502092bda4"
PREREG_DATE = "2026-08-23"
ALLOWED_UNITS = {
    "examples",
    "documents",
    "groups",
    "shards",
    "clients",
    "sources",
    "domains",
    "tasks",
    "demonstrations",
}
HEADLINE_GAME_TYPES = {"real_retraining", "icl_context_attribution"}
EPSILONS = (1e-1, 1e-2, 1e-3)
INTERACTION_ENERGY_THRESHOLD = 0.20
LOW_RANK_THRESHOLD = 0.25
BASE_BUDGET_MULTIPLIERS = (2, 4, 8, 16, 32)
REGISTRY_FIELDS = (
    "run_id",
    "first_run_date",
    "last_update_date",
    "git_sha",
    "path",
    "experiment_id",
    "dataset",
    "game_type",
    "unit_definition",
    "d",
    "learner",
    "utility",
    "grouping_rule",
    "ordering_rule",
    "full_enumeration",
    "oracle_calls",
    "learner_seeds",
    "split_seeds",
    "status",
    "preregistered",
    "selection_status",
    "paper_location",
    "exclusion_reason",
    "negative_result",
    "artifact_hash",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return sha256_bytes(array.view(np.uint8).tobytes())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_frozen_preregistration(path: Path = PREREG_PATH) -> None:
    if not path.exists():  # the protocol note is not distributed; its hash stays pinned above
        return
    observed = sha256_file(path)
    if observed != PREREG_SHA256:
        raise RuntimeError(f"preregistration hash changed: {observed} != {PREREG_SHA256}")


@dataclass(frozen=True)
class GameSpec:
    experiment_id: str
    dataset: str
    game_type: str
    unit_kind: str
    unit_definition: str
    dimension: int
    learner: str
    utility: str
    grouping_rule: str
    natural_order: tuple[int, ...]
    learner_seeds: tuple[int, ...]
    full_enumeration: bool = True
    tree_requested: bool = False
    feature_players: bool = False
    selection_status: str = "preregistered_before_outcomes"

    def validate(self, *, headline: bool = True) -> None:
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        if sorted(self.natural_order) != list(range(self.dimension)):
            raise ValueError("natural_order must be a permutation of every unit")
        if self.unit_kind not in ALLOWED_UNITS:
            raise ValueError(f"invalid training-data unit kind: {self.unit_kind}")
        if len(set(self.learner_seeds)) < 3 and headline and self.game_type == "real_retraining":
            raise ValueError("headline real retraining requires at least three learner seeds")
        if self.tree_requested and self.dimension >= 100:
            raise ValueError("tree TNDMs are forbidden for d >= 100")
        if self.tree_requested and self.full_enumeration and self.dimension > 12:
            raise ValueError("full-enumeration tree experiments are limited to d <= 12")
        if headline and (self.feature_players or self.game_type == "feature_coalition"):
            raise ValueError("feature-player games cannot enter headline aggregates")
        if headline and self.game_type not in HEADLINE_GAME_TYPES:
            raise ValueError(f"game type {self.game_type!r} is not headline data attribution")

    def contract_hash(self) -> str:
        return sha256_bytes(canonical_json(asdict(self)).encode("utf-8"))


def bits(dimension: int) -> np.ndarray:
    masks = np.arange(1 << dimension, dtype=np.int64)
    return ((masks[:, None] >> np.arange(dimension, dtype=np.int64)) & 1).astype(np.int8)


def _fwht(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim != 1 or len(result) == 0 or len(result) & (len(result) - 1):
        raise ValueError("values must be a nonempty one-dimensional power-of-two table")
    width = 1
    while width < len(result):
        for start in range(0, len(result), 2 * width):
            left = result[start : start + width].copy()
            right = result[start + width : start + 2 * width].copy()
            result[start : start + width] = left + right
            result[start + width : start + 2 * width] = left - right
        width *= 2
    return result / math.sqrt(len(result))


def interaction_energy(values: Sequence[float]) -> dict[str, Any]:
    """Return the preregistered uniform-measure additive-residual energy."""

    value = np.asarray(values, dtype=np.float64)
    dimension = int(round(math.log2(len(value))))
    if len(value) != 1 << dimension or not np.all(np.isfinite(value)):
        raise ValueError("values must be a finite complete Boolean cube")
    centered = value - float(np.mean(value))
    coefficients = _fwht(centered)
    degrees = np.fromiter((mask.bit_count() for mask in range(len(value))), dtype=np.int64)
    denominator = float(np.sum(coefficients[degrees >= 1] ** 2))
    numerator = float(np.sum(coefficients[degrees >= 2] ** 2))
    by_degree = {
        degree: float(np.sum(coefficients[degrees == degree] ** 2) / denominator) if denominator else 0.0
        for degree in range(dimension + 1)
    }
    return {
        "dimension": dimension,
        "E_ge_2": numerator / denominator if denominator else 0.0,
        "high_interaction": bool(denominator and numerator / denominator >= INTERACTION_ENERGY_THRESHOLD),
        "energy_by_degree": by_degree,
        "denominator": denominator,
    }


def permute_cube(values: Sequence[float], order: Sequence[int]) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    dimension = int(round(math.log2(len(value))))
    if sorted(map(int, order)) != list(range(dimension)):
        raise ValueError("order must be a permutation")
    tensor = value.reshape((2,) * dimension, order="F")
    return np.transpose(tensor, axes=tuple(order)).reshape(-1, order="F")


def epsilon_rank_profile(
    values: Sequence[float],
    order: Sequence[int],
    epsilons: Sequence[float] = EPSILONS,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Return absolute unfolding ranks and normalized maximum rank statistics."""

    ordered = permute_cube(values, order)
    dimension = int(round(math.log2(len(ordered))))
    tensor = ordered.reshape((2,) * dimension, order="F")
    rows: list[dict[str, Any]] = []
    normalized: dict[str, list[float]] = {f"{epsilon:g}": [] for epsilon in epsilons}
    for cut in range(1, dimension):
        matrix = tensor.reshape((1 << cut, 1 << (dimension - cut)), order="F")
        singular = np.linalg.svd(matrix, compute_uv=False)
        mass = singular**2
        total = float(np.sum(mass))
        retained = np.cumsum(mass) / max(total, np.finfo(float).tiny)
        for epsilon in epsilons:
            rank = int(np.searchsorted(retained, 1.0 - float(epsilon) ** 2, side="left") + 1)
            denominator = 1 << min(cut, dimension - cut)
            ratio = rank / denominator
            rows.append(
                {
                    "cut": cut,
                    "epsilon": float(epsilon),
                    "epsilon_rank": rank,
                    "full_unfolding_rank": denominator,
                    "normalized_rank": ratio,
                    "singular_values": singular.tolist(),
                }
            )
            if min(cut, dimension - cut) >= 3:
                normalized[f"{epsilon:g}"].append(ratio)
    maxima = {
        key: max(value) if value else float("nan")
        for key, value in normalized.items()
    }
    maxima["R_1e-2"] = maxima.get("0.01", float("nan"))
    maxima["low_rank"] = bool(np.isfinite(maxima["R_1e-2"]) and maxima["R_1e-2"] <= LOW_RANK_THRESHOLD)
    return rows, maxima


def unique_nested_budgets(dimension: int) -> tuple[int, ...]:
    half_cube = (1 << dimension) // 2
    return tuple(sorted({min(multiplier * dimension, half_cube) for multiplier in BASE_BUDGET_MULTIPLIERS}))


@dataclass(frozen=True)
class CoalitionSplit:
    seed: int
    validation_masks: tuple[int, ...]
    test_masks: tuple[int, ...]
    nested_train_masks: Mapping[int, tuple[int, ...]]

    def validate(self) -> None:
        validation = set(self.validation_masks)
        test = set(self.test_masks)
        if validation & test:
            raise ValueError("validation/test leakage")
        previous: set[int] = set()
        for budget in sorted(self.nested_train_masks):
            current = set(self.nested_train_masks[budget])
            if len(current) != budget:
                raise ValueError("budget and number of unique training masks differ")
            if not previous <= current:
                raise ValueError("coalition budgets are not nested")
            if current & (validation | test):
                raise ValueError("training/validation/test leakage")
            previous = current


def matched_nested_split(dimension: int, seed: int) -> CoalitionSplit:
    """Create one cardinality-stratified nested split shared by all methods."""

    rng = np.random.default_rng(seed)
    design = bits(dimension)
    buckets = [rng.permutation(np.flatnonzero(design.sum(axis=1) == size)).tolist() for size in range(dimension + 1)]
    sequence: list[int] = []
    cursors = [0] * (dimension + 1)
    while len(sequence) < len(design):
        advanced = False
        for size in rng.permutation(dimension + 1):
            cursor = cursors[int(size)]
            if cursor < len(buckets[int(size)]):
                sequence.append(int(buckets[int(size)][cursor]))
                cursors[int(size)] += 1
                advanced = True
        if not advanced:
            break
    if sorted(sequence) != list(range(1 << dimension)):
        raise RuntimeError("split sequence is not a cube permutation")
    half_cube = (1 << dimension) // 2
    validation_count = max(2 * dimension, 32)
    test_count = half_cube - validation_count
    if test_count <= 0:
        raise ValueError("dimension too small for the preregistered half-cube test protocol")
    full = (1 << dimension) - 1
    intervention = {
        0,
        full,
        *(1 << player for player in range(dimension)),
        *(full ^ (1 << player) for player in range(dimension)),
        *((1 << left) | (1 << right) for left in range(dimension) for right in range(left + 1, dimension)),
    }
    triplets = {
        (1 << first) | (1 << second) | (1 << third)
        for first in range(dimension)
        for second in range(first + 1, dimension)
        for third in range(second + 1, dimension)
    }
    if len(intervention | triplets) <= test_count:
        intervention.update(triplets)
    if len(intervention) > test_count:
        raise ValueError("dimension too small to reserve all intervention coalitions in the outer test")
    unreserved = [mask for mask in sequence if mask not in intervention]
    validation = tuple(unreserved[:validation_count])
    test_fill = unreserved[validation_count : validation_count + test_count - len(intervention)]
    test = tuple(sorted(intervention)) + tuple(test_fill)
    used = set(validation) | set(test)
    available = [mask for mask in sequence if mask not in used]
    nested = {budget: tuple(available[:budget]) for budget in unique_nested_budgets(dimension)}
    split = CoalitionSplit(seed, validation, test, nested)
    split.validate()
    return split


def exact_query_pack(values: Sequence[float]) -> dict[str, np.ndarray]:
    value = np.asarray(values, dtype=np.float64)
    return {
        "shapley": shapley_values(value),
        "banzhaf": banzhaf_values(value),
        "loo": loo_values(value),
        "pairwise": grabisch_roubens_interactions(value),
        "third_order_banzhaf": uniform_higher_order_interactions(value, 3)[1],
        "mobius": mobius_coefficients(value),
    }


def mobius_coefficients(values: Sequence[float]) -> np.ndarray:
    """Return every exact Harsanyi dividend of a complete Boolean game."""

    coefficients = np.asarray(values, dtype=np.float64).copy()
    dimension = int(round(math.log2(len(coefficients))))
    if coefficients.shape != (1 << dimension,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("Möbius coefficients require a finite complete scalar cube")
    masks = np.arange(1 << dimension, dtype=np.int64)
    for player in range(dimension):
        selected = (masks & (1 << player)) != 0
        coefficients[selected] -= coefficients[masks[selected] ^ (1 << player)]
    return coefficients


def uniform_higher_order_interactions(
    values: Sequence[float], order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized uniform-background finite differences of fixed order."""

    table = np.asarray(values, dtype=np.float64)
    dimension = int(round(math.log2(len(table))))
    if table.shape != (1 << dimension,) or order < 1 or order > dimension:
        raise ValueError("higher-order interactions need a complete scalar cube and valid order")
    item_rows = np.asarray(list(itertools.combinations(range(dimension), order)), dtype=np.int64)
    interactions = np.empty(len(item_rows), dtype=np.float64)
    masks = np.arange(1 << dimension, dtype=np.int64)
    for row, items in enumerate(item_rows):
        selected_bits = sum(1 << int(item) for item in items)
        backgrounds = masks[(masks & selected_bits) == 0]
        difference = np.zeros(len(backgrounds), dtype=np.float64)
        for subset_size in range(order + 1):
            sign = -1.0 if (order - subset_size) % 2 else 1.0
            for subset in itertools.combinations(items.tolist(), subset_size):
                added = sum(1 << int(item) for item in subset)
                difference += sign * table[backgrounds | added]
        interactions[row] = float(np.mean(difference))
    return item_rows, interactions


def normalized_query_rmse_from_packs(
    predicted: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    dimension: int,
) -> dict[str, float]:
    """Normalize query errors from query packs computed exactly once."""
    output: dict[str, float] = {}
    for name in ("shapley", "banzhaf", "loo"):
        residual = np.asarray(predicted[name]) - np.asarray(reference[name])
        scale = max(float(np.sqrt(np.mean(np.asarray(reference[name]) ** 2))), np.finfo(float).tiny)
        output[f"{name}_nrmse"] = float(np.sqrt(np.mean(residual**2)) / scale)
    triangle = np.triu_indices(dimension, k=1)
    pred_pair = np.asarray(predicted["pairwise"])[triangle]
    true_pair = np.asarray(reference["pairwise"])[triangle]
    scale = max(float(np.sqrt(np.mean(true_pair**2))), np.finfo(float).tiny)
    output["pairwise_nrmse"] = float(np.sqrt(np.mean((pred_pair - true_pair) ** 2)) / scale)
    predicted_third = np.asarray(predicted["third_order_banzhaf"])
    true_third = np.asarray(reference["third_order_banzhaf"])
    third_scale = max(float(np.sqrt(np.mean(true_third**2))), np.finfo(float).tiny)
    output["third_order_banzhaf_nrmse"] = float(
        np.sqrt(np.mean((predicted_third - true_third) ** 2)) / third_scale
    )
    degrees = np.fromiter((mask.bit_count() for mask in range(1 << dimension)), dtype=np.int64)
    interaction_masks = degrees >= 2
    predicted_mobius = np.asarray(predicted["mobius"])[interaction_masks]
    true_mobius = np.asarray(reference["mobius"])[interaction_masks]
    mobius_scale = max(float(np.sqrt(np.mean(true_mobius**2))), np.finfo(float).tiny)
    output["mobius_interaction_nrmse"] = float(
        np.sqrt(np.mean((predicted_mobius - true_mobius) ** 2)) / mobius_scale
    )
    return output


def normalized_query_rmse(prediction: Sequence[float], truth: Sequence[float]) -> dict[str, float]:
    prediction_array = np.asarray(prediction, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    dimension = int(round(math.log2(len(truth_array))))
    return normalized_query_rmse_from_packs(
        exact_query_pack(prediction_array),
        exact_query_pack(truth_array),
        dimension,
    )


def attribution_composite(metrics: Mapping[str, float]) -> float:
    values = np.asarray(
        [metrics["shapley_nrmse"], metrics["banzhaf_nrmse"], metrics["pairwise_nrmse"]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("query NRMSE values must be finite and nonnegative")
    return float(np.exp(np.mean(np.log(np.maximum(values, 1e-12)))))


def validate_selection_record(record: Mapping[str, Any]) -> None:
    if record.get("selection_split") != "validation":
        raise ValueError("topology/rank/hyperparameters must be selected on validation")
    forbidden = {"shapley", "banzhaf", "interaction", "intervention", "test"}
    metric = str(record.get("selection_metric", "")).lower()
    if not metric or any(token in metric for token in forbidden):
        raise ValueError("selection metric must be validation coalition utility only")
    if record.get("test_scored_before_selection"):
        raise ValueError("test was scored before validation selection")


def validate_comprehensive_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    datasets = {str(row.get("dataset", "")) for row in rows}
    if not any("ag_news" in dataset.lower() or "ag news" in dataset.lower() for dataset in datasets):
        raise ValueError("AG News is required in comprehensive real-data tables")
    for row in rows:
        game_type = str(row.get("game_type", ""))
        if game_type == "feature_coalition" and str(row.get("paper_location")) != "registry_only":
            raise ValueError("feature-player result escaped the registry-only scope")
        if "har" in str(row.get("dataset", "")).lower() and row.get("selection_status") not in {
            "post_hoc_selected_exploratory",
            "preregistered_replication",
        }:
            raise ValueError("HAR must retain its post-hoc or replication label")


def validate_registry_temporal_labels(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject pre-preregistration runs presented as new confirmatory evidence.

    The registry stores dates rather than timestamps.  Runs strictly before
    the frozen date must therefore carry the unambiguous legacy label.  New
    E1--E6 artifacts created on the freeze date are independently bound to
    ``PREREG_SHA256`` by their artifact contracts.
    """

    for row in rows:
        first_run_date = str(row.get("first_run_date", ""))
        if not first_run_date:
            raise ValueError("registry row is missing first_run_date")
        if first_run_date < PREREG_DATE:
            if str(row.get("preregistered", "")).lower() != "no":
                raise ValueError("pre-preregistration run was marked preregistered")
            if str(row.get("selection_status", "")) != "previously_observed":
                raise ValueError("pre-preregistration run lacks previously_observed label")


def validate_artifact_ledger(rows: Sequence[Mapping[str, Any]], *, root: Path = ROOT) -> None:
    """Verify every reported value is bound to an existing immutable source.

    Figure and claim generators use one ledger row per displayed value.  A
    relative source path is resolved from the repository root and its recorded
    SHA-256 must match the file currently on disk.
    """

    if not rows:
        raise ValueError("artifact ledger is empty")
    for index, row in enumerate(rows):
        source = str(row.get("source_artifact", ""))
        expected = str(row.get("source_sha256", row.get("source_artifact_sha256", "")))
        if not source or source in {"not_available", "generated_test_cube"}:
            raise ValueError(f"ledger row {index} lacks a machine-readable source artifact")
        if len(expected) != 64:
            raise ValueError(f"ledger row {index} lacks a valid source SHA-256")
        path = Path(source)
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise ValueError(f"ledger row {index} source artifact is missing: {path}")
        if sha256_file(path) != expected:
            raise ValueError(f"ledger row {index} source artifact checksum mismatch: {path}")


def read_registry(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REGISTRY_FIELDS:
            raise ValueError("registry schema changed")
        return list(reader)


def validate_append_only(previous: Sequence[Mapping[str, str]], current: Sequence[Mapping[str, str]]) -> None:
    if len(current) < len(previous):
        raise ValueError("registry rows were deleted")
    for index, row in enumerate(previous):
        if dict(row) != dict(current[index]):
            raise ValueError(f"registry historical row {index + 2} was modified or reordered")


def append_registry_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically append rows without permitting historical mutation."""

    existing = read_registry(path)
    additions = []
    seen = {row["run_id"] for row in existing}
    for raw in rows:
        row = {field: str(raw.get(field, "")) for field in REGISTRY_FIELDS}
        if not row["run_id"] or row["run_id"] in seen:
            raise ValueError(f"missing or duplicate run_id: {row['run_id']!r}")
        seen.add(row["run_id"])
        additions.append(row)
    current = existing + additions
    validate_append_only(existing, current)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
            writer.writeheader()
            writer.writerows(current)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
