"""Exact coalition-game functionals for the d=8 transformer cubes.

Everything here is exact arithmetic on saved retraining outcomes: marginal-effect
curves g_T(t) of the multilinear (Owen) extension, and the LOO, Banzhaf and
Shapley points of those curves.  No fitting happens in this module.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

D = 8
Q = 8
INPUTS_PER_QUERY = 31
# Per-coalition retraining records (one JSON per coalition), produced by
# experiments/reasoning/common/transformer_query_paths.py.  They are not shipped;
# set TNDM_D8_CELLS to a directory containing them after re-running the retraining.
# When they are absent, the outcomes are read from the released compact cube.
REPO = Path(__file__).resolve().parents[4]
QUERY_PATHS_ROOT = Path(os.environ.get("TNDM_D8_CELLS", REPO / "experiments/reasoning/d8/cells"))
RELEASED_CUBE = REPO / "data/cubes/khop_d8_transformer.npz"


def cell_path(mask: int, root: Path = QUERY_PATHS_ROOT, seed: int = 31) -> Path:
    for prefix in ("gate", "observed", "target"):
        path = root / prefix / f"seed{seed}" / f"mask_{mask}" / f"seed{seed}_mask{mask}.json"
        if path.exists():
            return path
    raise FileNotFoundError(f"mask {mask} is missing from {root}")


def load_query_cube(root: Path = QUERY_PATHS_ROOT, seed: int = 31
                    ) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Return (log-probability utilities, accuracy utilities, source paths).

    Both arrays are (256, 8): coalition mask by query.  The log-probability
    utility is the mean over the query's 31 inputs of the exact-answer log
    probability; the accuracy utility is the mean of its correctness indicators.
    """
    if root == QUERY_PATHS_ROOT and not root.exists():
        return _load_released_cube(seed)
    logp = np.empty((1 << D, Q), dtype=np.float64)
    accuracy = np.empty((1 << D, Q), dtype=np.float64)
    paths = []
    for mask in range(1 << D):
        path = cell_path(mask, root, seed)
        paths.append(path)
        row = json.loads(path.read_text())
        per_input = np.asarray(row["target_log_probability_by_query_input"], dtype=np.float64)
        correct = np.asarray(row["target_correct_by_query_input"], dtype=np.float64)
        if per_input.shape != (Q * INPUTS_PER_QUERY,):
            raise ValueError(f"mask {mask}: unexpected output shape {per_input.shape}")
        logp[mask] = per_input.reshape(Q, INPUTS_PER_QUERY).mean(axis=1)
        accuracy[mask] = correct.reshape(Q, INPUTS_PER_QUERY).mean(axis=1)
        stored = np.asarray(row["target_log_probability_by_query"], dtype=np.float64)
        if not np.allclose(logp[mask], stored, atol=1e-9):
            raise ValueError(f"mask {mask}: per-query mean disagrees with stored per-query value")
    return logp, accuracy, paths


def _load_released_cube(seed: int) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Same arrays as `load_query_cube`, read from data/cubes/khop_d8_transformer.npz."""
    data = np.load(RELEASED_CUBE, allow_pickle=False)
    if seed != 31 or not np.all(data["seeds"] == 31):
        raise ValueError("the released d=8 cube contains learner seed 31 only")
    order = np.argsort(data["masks"])
    if list(data["masks"][order]) != list(range(1 << D)):
        raise ValueError("released cube is not a complete enumeration")
    logp = data["values"][order].reshape(1 << D, Q, INPUTS_PER_QUERY).mean(axis=2)
    accuracy = data["secondary_values"][order].reshape(1 << D, Q, INPUTS_PER_QUERY).mean(axis=2)
    return logp, accuracy, [RELEASED_CUBE]


def as_tensor(values: np.ndarray, dimension: int = D) -> np.ndarray:
    """Reshape a mask-indexed vector so that axis i is the bit-i indicator."""
    values = np.asarray(values, dtype=np.float64)
    tail = values.shape[1:]
    tensor = values.reshape((2,) * dimension + tail)
    order = tuple(range(dimension - 1, -1, -1)) + tuple(range(dimension, dimension + len(tail)))
    return np.transpose(tensor, order)


def difference(tensor: np.ndarray, players: tuple[int, ...], dimension: int = D) -> np.ndarray:
    """|T|-th finite difference Delta_T v, as a tensor over the remaining axes."""
    result = tensor
    for offset, player in enumerate(sorted(players)):
        axis = player - offset
        result = np.take(result, 1, axis=axis) - np.take(result, 0, axis=axis)
    return result


def _background_vectors(t: np.ndarray, free: list[int], condition: dict[int, int] | None
                        ) -> list[np.ndarray]:
    vectors = []
    for player in free:
        fixed = None if condition is None else condition.get(player)
        if fixed is None:
            vectors.append(np.stack((1 - t, t), axis=-1))
        else:
            column = np.zeros((len(t), 2))
            column[:, int(fixed)] = 1.0
            vectors.append(column)
    return vectors


def marginal_curve(tensor: np.ndarray, players: tuple[int, ...], t: np.ndarray,
                   condition: dict[int, int] | None = None, dimension: int = D) -> np.ndarray:
    """g_T(t) = E_{S ~ Bernoulli(t)} [ Delta_T v(S) ], optionally conditioning players.

    `condition` maps a background player to 1 (always present) or 0 (always absent).
    Returns an array of shape (len(t),) + trailing output shape.
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    delta = difference(tensor, players, dimension)
    free = [player for player in range(dimension) if player not in players]
    vectors = _background_vectors(t, free, condition)
    result = np.broadcast_to(delta, (len(t),) + delta.shape).copy()
    for vector in vectors:
        # axis 0 is t and the leading background axis is always axis 1 after a contraction
        result = np.einsum("tb...,tb->t...", result, vector)
    return result


def shapley_interaction(tensor: np.ndarray, players: tuple[int, ...], dimension: int = D
                        ) -> np.ndarray:
    """Exact integral of g_T over t in [0, 1] (Shapley interaction index)."""
    delta = difference(tensor, players, dimension)
    free = dimension - len(players)
    size = np.zeros((2,) * free, dtype=np.int64)
    for axis in range(free):
        shape = [1] * free
        shape[axis] = 2
        size = size + np.arange(2).reshape(shape)
    weight = np.asarray([
        math.factorial(int(count)) * math.factorial(free - int(count)) / math.factorial(free + 1)
        for count in size.reshape(-1)
    ]).reshape(size.shape)
    return np.tensordot(weight, delta, axes=(tuple(range(free)), tuple(range(free))))


def curve_points(tensor: np.ndarray, players: tuple[int, ...], dimension: int = D) -> dict:
    """LOO (t=1), Banzhaf (t=1/2) and Shapley (integral) points of g_T."""
    values = marginal_curve(tensor, players, np.asarray([0.0, 0.5, 1.0]), dimension=dimension)
    return {
        "empty_background": values[0],
        "banzhaf": values[1],
        "loo": values[2],
        "shapley": shapley_interaction(tensor, players, dimension),
    }


def providers(root: Path = QUERY_PATHS_ROOT) -> list[list[list[int]]]:
    return json.loads((root / "protocol.json").read_text())["providers"]


def transversals(query_providers: list[list[int]]) -> list[tuple[int, ...]]:
    result = []
    for first in query_providers[0]:
        for second in query_providers[1]:
            for third in query_providers[2]:
                result.append(tuple(sorted((first, second, third))))
    return sorted(result)


def same_role_pairs(query_providers: list[list[int]]) -> list[tuple[int, int]]:
    return sorted(tuple(sorted(pair)) for pair in query_providers)


def cross_role_pairs(query_providers: list[list[int]]) -> list[tuple[int, int]]:
    result = []
    for left in range(3):
        for right in range(left + 1, 3):
            for first in query_providers[left]:
                for second in query_providers[right]:
                    result.append(tuple(sorted((first, second))))
    return sorted(result)


def partner_of(query_providers: list[list[int]], player: int) -> int:
    for role in query_providers:
        if player in role:
            return role[0] if role[1] == player else role[1]
    raise ValueError(f"group {player} is not a provider for this query")


def lower_median(values: np.ndarray) -> int:
    """Index of the lower of the two middle order statistics (ties: lowest index)."""
    order = np.lexsort((np.arange(len(values)), np.asarray(values, dtype=np.float64)))
    return int(order[(len(values) - 1) // 2])


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
