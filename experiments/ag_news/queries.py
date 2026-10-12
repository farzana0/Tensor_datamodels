"""Exact coalition-query utilities for full response tables.

The central representation is a *full coalition value table*.  Row ``mask``
stores ``U(S_mask)`` where player ``i`` belongs to ``S_mask`` exactly when
``mask & (1 << i)`` is nonzero.  Thus a table for ``n`` players has leading
dimension ``2**n``.  Any remaining dimensions are treated as response outputs
(for example, probe examples, logits, or several observables), and query
functions preserve them.

The first-order queries in this module are normalized expectations of a
marginal difference over background coalitions.  This makes their semantics
explicit and gives the elementary uniform-error bound ``2 * epsilon``.  The
pairwise routine uses the Grabisch--Roubens Shapley interaction index, whose
background-coalition weights are normalized as well and gives ``4 * epsilon``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import comb
from numbers import Integral, Real
from typing import Any

import numpy as np


__all__ = [
    "enumerate_masks",
    "enumerate_indicators",
    "mask_to_indicator",
    "indicator_to_mask",
    "coalition_to_mask",
    "mask_to_coalition",
    "infer_n_players",
    "shapley_background_weights",
    "banzhaf_background_weights",
    "loo_background_weights",
    "near_full_background_weights",
    "first_order_query",
    "shapley_values",
    "banzhaf_values",
    "loo_values",
    "near_full_values",
    "grabisch_roubens_interaction_weight",
    "grabisch_roubens_interactions",
    "linear_functional",
    "linear_functional_error",
    "dual_norm_order",
    "linear_functional_error_bound",
    "first_order_supnorm_error_bound",
    "second_order_supnorm_error_bound",
]


# NumPy's signed integer masks support at most 62 player bits without an
# overflow.  In practice an exhaustive table becomes infeasible far earlier.
_MAX_ENUMERATED_PLAYERS = 62
_WEIGHT_ATOL = 1e-12


def _validate_n_players(n_players: int) -> int:
    """Return a checked player count suitable for bit-mask enumeration."""
    if not isinstance(n_players, Integral) or isinstance(n_players, bool):
        raise TypeError("n_players must be an integer")
    n_players = int(n_players)
    if n_players < 0:
        raise ValueError("n_players must be non-negative")
    if n_players > _MAX_ENUMERATED_PLAYERS:
        raise ValueError(
            f"n_players must be at most {_MAX_ENUMERATED_PLAYERS} for integer masks"
        )
    return n_players


def _validate_mask(mask: int, n_players: int) -> int:
    """Validate and return a mask in the range for ``n_players`` players."""
    n_players = _validate_n_players(n_players)
    if not isinstance(mask, Integral) or isinstance(mask, bool):
        raise TypeError("mask must be an integer")
    mask = int(mask)
    if mask < 0 or mask >= (1 << n_players):
        raise ValueError(f"mask must be in [0, {1 << n_players})")
    return mask


def enumerate_masks(n_players: int) -> np.ndarray:
    """Return all coalition masks in canonical row/index order.

    The returned integer array is ``[0, 1, ..., 2**n_players - 1]``.  Bit
    ``i`` is the membership indicator for player ``i``.
    """
    n_players = _validate_n_players(n_players)
    return np.arange(1 << n_players, dtype=np.int64)


def enumerate_indicators(n_players: int) -> np.ndarray:
    """Return a boolean ``(2**n, n)`` coalition-membership matrix.

    Row order agrees with :func:`enumerate_masks`; column ``i`` is true exactly
    when the corresponding mask contains player ``i``.
    """
    n_players = _validate_n_players(n_players)
    masks = enumerate_masks(n_players)
    if n_players == 0:
        return np.empty((1, 0), dtype=bool)
    bits = np.arange(n_players, dtype=np.int64)
    return ((masks[:, None] >> bits) & 1).astype(bool)


def mask_to_indicator(mask: int, n_players: int) -> np.ndarray:
    """Convert one coalition mask to a boolean membership indicator."""
    mask = _validate_mask(mask, n_players)
    n_players = int(n_players)
    return np.array([(mask & (1 << i)) != 0 for i in range(n_players)], dtype=bool)


def indicator_to_mask(indicator: Any) -> int:
    """Convert a one-dimensional boolean or binary membership vector to a mask.

    Unlike :func:`coalition_to_mask`, this accepts a dense membership vector.
    Keeping the two conversions separate avoids ambiguity between ``[0, 1]`` as
    two player indices and as a two-player indicator.
    """
    array = np.asarray(indicator)
    if array.ndim != 1:
        raise ValueError("indicator must be one-dimensional")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("indicator entries must be boolean or binary")
    mask = 0
    for player, included in enumerate(array.tolist()):
        if included:
            mask |= 1 << player
    return mask


def coalition_to_mask(coalition: Iterable[int], n_players: int) -> int:
    """Convert an iterable of distinct player indices to a bit mask."""
    n_players = _validate_n_players(n_players)
    mask = 0
    seen: set[int] = set()
    for player in coalition:
        if not isinstance(player, Integral) or isinstance(player, bool):
            raise TypeError("coalition members must be integer indices")
        player = int(player)
        if player < 0 or player >= n_players:
            raise ValueError(f"player index {player} is outside [0, {n_players})")
        if player in seen:
            raise ValueError(f"coalition contains duplicate player index {player}")
        seen.add(player)
        mask |= 1 << player
    return mask


def mask_to_coalition(mask: int, n_players: int) -> tuple[int, ...]:
    """Convert a mask to an ascending tuple of included player indices."""
    mask = _validate_mask(mask, n_players)
    return tuple(i for i in range(int(n_players)) if mask & (1 << i))


def infer_n_players(values: Any) -> int:
    """Infer ``n`` from the leading ``2**n`` dimension of a value table.

    Raises:
        ValueError: if ``values`` has no coalition axis or its leading dimension
            is not a positive power of two.
    """
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("a coalition value table needs a leading coalition axis")
    n_masks = int(array.shape[0])
    if n_masks <= 0 or n_masks & (n_masks - 1):
        raise ValueError("the leading value-table dimension must be a positive power of two")
    return n_masks.bit_length() - 1


def _value_table(values: Any) -> np.ndarray:
    """Validate the common full-table contract and return an ndarray view."""
    array = np.asarray(values)
    infer_n_players(array)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("coalition values must be numeric")
    return array


def _query_dtype(values: np.ndarray) -> np.dtype:
    """Use floating-point accumulation while retaining complex values if supplied."""
    return np.result_type(values.dtype, np.float64)


def _empty_background_weights(n_players: int) -> np.ndarray:
    n_players = _validate_n_players(n_players)
    return np.zeros((n_players, 1 << n_players), dtype=np.float64)


def shapley_background_weights(n_players: int) -> np.ndarray:
    """Return normalized Shapley background-coalition distributions.

    ``weights[i, mask]`` equals

    ``|S|! (n-|S|-1)! / n! = 1 / (n * binom(n-1, |S|))``

    when ``mask`` encodes ``S`` and does not contain ``i``; otherwise it is
    zero.  Each row is a probability distribution, so applying it to marginal
    differences gives the standard Shapley value.
    """
    n_players = _validate_n_players(n_players)
    weights = _empty_background_weights(n_players)
    if n_players == 0:
        return weights
    for player in range(n_players):
        bit = 1 << player
        for mask in range(1 << n_players):
            if not mask & bit:
                size = mask.bit_count()
                weights[player, mask] = 1.0 / (n_players * comb(n_players - 1, size))
    return weights


def banzhaf_background_weights(n_players: int) -> np.ndarray:
    """Return normalized uniform (Banzhaf) background distributions.

    Each coalition not containing the queried player receives probability
    ``2**(-(n-1))``.  This is the normalized Banzhaf value convention: the
    expected marginal under independently included other players with
    probability one half.
    """
    n_players = _validate_n_players(n_players)
    weights = _empty_background_weights(n_players)
    if n_players == 0:
        return weights
    mass = 1.0 / (1 << (n_players - 1))
    for player in range(n_players):
        bit = 1 << player
        for mask in range(1 << n_players):
            if not mask & bit:
                weights[player, mask] = mass
    return weights


def loo_background_weights(n_players: int) -> np.ndarray:
    """Return full-set leave-one-out background distributions.

    The sole background coalition for player ``i`` is ``N \\ {i}``, so the
    resulting first-order query is ``U(N) - U(N \\ {i})``.
    """
    n_players = _validate_n_players(n_players)
    weights = _empty_background_weights(n_players)
    full_mask = (1 << n_players) - 1
    for player in range(n_players):
        weights[player, full_mask ^ (1 << player)] = 1.0
    return weights


def near_full_background_weights(
    n_players: int,
    cardinality_mass: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Return a normalized near-full background-coalition distribution.

    The support consists of the two closest possible cardinalities, ``n-2``
    and ``n-1`` (among nonnegative valid cardinalities), conditional on not
    containing the queried player.  By default this function gives each
    supported *cardinality* equal probability and is uniform within a
    cardinality.  For example, for ``n >= 2`` the default puts mass 1/2 on
    each of ``|S|=n-2`` and ``|S|=n-1``.

    To use a different normalized convention, pass a mapping such as
    ``{n - 2: 0.8, n - 1: 0.2}``.  This makes the otherwise underspecified
    phrase "near full" explicit and reproducible.
    """
    n_players = _validate_n_players(n_players)
    weights = _empty_background_weights(n_players)
    if n_players == 0:
        return weights

    support = tuple(size for size in (n_players - 2, n_players - 1) if size >= 0)
    if cardinality_mass is None:
        mass_by_size = {size: 1.0 / len(support) for size in support}
    else:
        unexpected = set(cardinality_mass) - set(support)
        if unexpected:
            raise ValueError(
                "cardinality_mass can only assign supported near-full sizes "
                f"{support}; got unsupported sizes {sorted(unexpected)}"
            )
        mass_by_size = {size: float(cardinality_mass.get(size, 0.0)) for size in support}
        masses = np.asarray(list(mass_by_size.values()), dtype=float)
        if not np.all(np.isfinite(masses)) or np.any(masses < 0):
            raise ValueError("cardinality masses must be finite and non-negative")
        if not np.isclose(float(masses.sum()), 1.0, atol=_WEIGHT_ATOL, rtol=1e-10):
            raise ValueError("cardinality masses must sum to one")

    for player in range(n_players):
        bit = 1 << player
        for mask in range(1 << n_players):
            if mask & bit:
                continue
            size = mask.bit_count()
            if size in mass_by_size:
                weights[player, mask] = mass_by_size[size] / comb(n_players - 1, size)
    return weights


def _validated_background_weights(weights: Any, n_players: int) -> np.ndarray:
    """Validate the normalized probability-distribution contract for ``mu_i``."""
    expected_shape = (n_players, 1 << n_players)
    array = np.asarray(weights, dtype=np.float64)
    if array.shape != expected_shape:
        raise ValueError(
            f"background weights must have shape {expected_shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("background weights must be finite")
    if np.any(array < -_WEIGHT_ATOL):
        raise ValueError("background weights must be non-negative")
    # Strip harmless negative roundoff before validating the probability rows.
    array = np.maximum(array, 0.0)
    for player in range(n_players):
        forbidden = np.fromiter(
            (bool(mask & (1 << player)) for mask in range(1 << n_players)),
            dtype=bool,
            count=1 << n_players,
        )
        if np.any(np.abs(array[player, forbidden]) > _WEIGHT_ATOL):
            raise ValueError("background weights may not put mass on coalitions containing player i")
        if not np.isclose(
            float(array[player].sum()), 1.0, atol=_WEIGHT_ATOL, rtol=1e-10
        ):
            raise ValueError("each player's background weights must sum to one")
    return array


def first_order_query(values: Any, background_weights: Any) -> np.ndarray:
    """Evaluate a normalized first-order marginal query from a full table.

    Args:
        values: Array with shape ``(2**n, *output_shape)``.
        background_weights: Nonnegative ``(n, 2**n)`` array.  Row ``i`` must
            sum to one and assign mass only to coalitions not containing ``i``.

    Returns:
        An array of shape ``(n, *output_shape)`` with
        ``sum_S mu_i(S) [U(S union {i}) - U(S)]`` for every player ``i``.
    """
    table = _value_table(values)
    n_players = infer_n_players(table)
    weights = _validated_background_weights(background_weights, n_players)
    result = np.zeros((n_players, *table.shape[1:]), dtype=_query_dtype(table))
    for player in range(n_players):
        bit = 1 << player
        for mask in np.flatnonzero(weights[player]):
            result[player] += weights[player, mask] * (table[mask | bit] - table[mask])
    return result


def shapley_values(values: Any) -> np.ndarray:
    """Compute standard Shapley values from an exhaustive coalition table."""
    table = _value_table(values)
    return first_order_query(table, shapley_background_weights(infer_n_players(table)))


def banzhaf_values(values: Any) -> np.ndarray:
    """Compute normalized Banzhaf values from an exhaustive coalition table."""
    table = _value_table(values)
    return first_order_query(table, banzhaf_background_weights(infer_n_players(table)))


def loo_values(values: Any) -> np.ndarray:
    """Compute full-set leave-one-out values from an exhaustive coalition table."""
    table = _value_table(values)
    return first_order_query(table, loo_background_weights(infer_n_players(table)))


def near_full_values(
    values: Any, cardinality_mass: Mapping[int, float] | None = None
) -> np.ndarray:
    """Compute the explicitly documented near-full marginal query."""
    table = _value_table(values)
    return first_order_query(
        table,
        near_full_background_weights(infer_n_players(table), cardinality_mass),
    )


def grabisch_roubens_interaction_weight(n_players: int, background_size: int) -> float:
    """Return one Grabisch--Roubens O2 interaction coefficient.

    For pair ``{i, j}`` and ``S subseteq N \\ {i, j}`` of size ``s``, the
    coefficient is

    ``s! (n-s-2)! / (n-1)! = 1 / ((n-1) * binom(n-2, s))``.

    Across all allowed backgrounds the coefficients sum to one.  This is the
    Shapley interaction index convention used by
    Grabisch and Roubens, not an unspecified generic "interaction" score.
    """
    n_players = _validate_n_players(n_players)
    if n_players < 2:
        raise ValueError("a pairwise interaction needs at least two players")
    if not isinstance(background_size, Integral) or isinstance(background_size, bool):
        raise TypeError("background_size must be an integer")
    background_size = int(background_size)
    if background_size < 0 or background_size > n_players - 2:
        raise ValueError(f"background_size must be in [0, {n_players - 2}]")
    return 1.0 / ((n_players - 1) * comb(n_players - 2, background_size))


def grabisch_roubens_interactions(values: Any) -> np.ndarray:
    """Compute the Grabisch--Roubens pairwise Shapley interaction index.

    The result has shape ``(n, n, *output_shape)`` and a zero diagonal.  For
    ``i != j`` it equals

    ``sum_{S subseteq N\\{i,j}} c_n(|S|) Delta_ij U(S)``,

    where ``Delta_ij U(S) = U(S+i+j) - U(S+i) - U(S+j) + U(S)`` and
    ``c_n(s) = s!(n-s-2)!/(n-1)!``.  The explicit convention matters because
    several inequivalent Shapley-style interaction indices exist.
    """
    table = _value_table(values)
    n_players = infer_n_players(table)
    result = np.zeros(
        (n_players, n_players, *table.shape[1:]), dtype=_query_dtype(table)
    )
    for first in range(n_players):
        first_bit = 1 << first
        for second in range(first + 1, n_players):
            second_bit = 1 << second
            pair_bits = first_bit | second_bit
            interaction = np.zeros(table.shape[1:], dtype=_query_dtype(table))
            for mask in range(1 << n_players):
                if mask & pair_bits:
                    continue
                coefficient = grabisch_roubens_interaction_weight(
                    n_players, mask.bit_count()
                )
                delta = (
                    table[mask | pair_bits]
                    - table[mask | first_bit]
                    - table[mask | second_bit]
                    + table[mask]
                )
                interaction += coefficient * delta
            result[first, second] = interaction
            result[second, first] = interaction
    return result


def _numeric_array(array: Any, name: str) -> np.ndarray:
    """Return a numeric ndarray or raise a clear type error."""
    result = np.asarray(array)
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    return result


def linear_functional(values: Any, weights: Any) -> np.ndarray:
    """Apply a scalar-coalition linear functional to one or many outputs.

    ``weights`` has shape ``(2**n,)`` and ``values`` has shape
    ``(2**n, *output_shape)``.  The return is ``sum_mask weights[mask] *
    values[mask]`` with shape ``output_shape``.  Weights need not be positive:
    this helper also covers finite differences and arbitrary linear queries.
    """
    table = _numeric_array(values, "values")
    if table.ndim == 0:
        raise ValueError("values needs a leading coalition axis")
    weights_array = _numeric_array(weights, "weights")
    if weights_array.ndim != 1 or weights_array.shape[0] != table.shape[0]:
        raise ValueError(
            "weights must be one-dimensional and match the leading value-table dimension"
        )
    return np.tensordot(weights_array, table, axes=(0, 0))


def linear_functional_error(estimate: Any, truth: Any, weights: Any) -> np.ndarray:
    """Return the absolute error of a linear functional, component by component."""
    estimate_array = _numeric_array(estimate, "estimate")
    truth_array = _numeric_array(truth, "truth")
    if estimate_array.shape != truth_array.shape:
        raise ValueError("estimate and truth must have identical shapes")
    return np.abs(
        linear_functional(estimate_array, weights)
        - linear_functional(truth_array, weights)
    )


def dual_norm_order(p: Real) -> float:
    """Return the Holder-dual norm order ``q`` for a valid ``p`` in ``[1, inf]``."""
    if not isinstance(p, Real):
        raise TypeError("p must be a real norm order")
    p = float(p)
    if np.isinf(p):
        return 1.0
    if not np.isfinite(p) or p < 1.0:
        raise ValueError("p must lie in [1, inf]")
    if p == 1.0:
        return float("inf")
    return p / (p - 1.0)


def _norm_on_coalition_axis(values: np.ndarray, order: float) -> np.ndarray:
    """Take an Lp norm over axis zero while preserving output dimensions."""
    absolute = np.abs(values)
    if np.isinf(order):
        return np.max(absolute, axis=0)
    return np.sum(absolute**order, axis=0) ** (1.0 / order)


def linear_functional_error_bound(
    estimate: Any, truth: Any, weights: Any, p: Real = 1.0
) -> np.ndarray:
    """Compute the Holder bound for a linear coalition functional.

    With ``q = dual_norm_order(p)``, returns the componentwise quantity

    ``||weights||_p * ||estimate - truth||_q``.

    It therefore upper-bounds :func:`linear_functional_error`.  For a
    multi-output response table, the response norm is taken independently for
    each trailing output coordinate.
    """
    estimate_array = _numeric_array(estimate, "estimate")
    truth_array = _numeric_array(truth, "truth")
    if estimate_array.ndim == 0 or estimate_array.shape != truth_array.shape:
        raise ValueError("estimate and truth must be equal-shaped tables with a coalition axis")
    weights_array = _numeric_array(weights, "weights")
    if weights_array.ndim != 1 or weights_array.shape[0] != estimate_array.shape[0]:
        raise ValueError(
            "weights must be one-dimensional and match the leading value-table dimension"
        )
    q = dual_norm_order(p)
    weight_norm = _norm_on_coalition_axis(weights_array, float(p))
    error_norm = _norm_on_coalition_axis(estimate_array - truth_array, q)
    return weight_norm * error_norm


def _nonnegative_finite_scalar(value: Real, name: str) -> float:
    """Validate a scalar error radius used in a deterministic bound."""
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def first_order_supnorm_error_bound(epsilon: Real) -> float:
    """Return the universal ``2 * epsilon`` error bound for normalized marginals.

    If ``max_S |U_hat(S)-U(S)| <= epsilon``, every normalized first-order
    background query in this module (including Shapley, Banzhaf, LOO, and
    near-full) differs from its truth by at most this value.
    """
    return 2.0 * _nonnegative_finite_scalar(epsilon, "epsilon")


def second_order_supnorm_error_bound(epsilon: Real) -> float:
    """Return the universal ``4 * epsilon`` O2 interaction error bound.

    It applies to the normalized Grabisch--Roubens interaction in
    :func:`grabisch_roubens_interactions` because each second difference has
    four table entries with coefficients of magnitude one.
    """
    return 4.0 * _nonnegative_finite_scalar(epsilon, "epsilon")
