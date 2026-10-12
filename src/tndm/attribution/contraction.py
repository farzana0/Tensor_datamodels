"""Attribution queries computed by contracting a fitted tensor-network datamodel.

A TNDM is multilinear in the inclusion vector ``z``, so evaluating the network at
a real-valued ``p`` in ``[0, 1]^d`` gives its multilinear extension, i.e. the
expected prediction when group ``j`` is included independently with probability
``p_j``.  Semivalues and interaction indices are integrals of finite differences
of this extension along the diagonal ``p = (t, ..., t)`` (Appendix F of the
paper):

* Shapley value          ``phi_i  = int_0^1 [f(t; z_i=1) - f(t; z_i=0)] dt``
* Banzhaf value          ``beta_i = f(1/2; z_i=1) - f(1/2; z_i=0)``
* Shapley interaction    ``I_A    = int_0^1 Delta_A f(t) dt``
* Banzhaf interaction    ``B_A    = Delta_A f(1/2)``

The integrands are polynomials of degree at most ``d - |A|`` in ``t``, so
Gauss-Legendre quadrature with ``ceil((d - |A| + 1) / 2)`` nodes is exact up to
floating point.  Each query costs a number of network evaluations polynomial in
``d``; no enumeration of the ``2**d`` coalitions is needed.

The predictors returned by :func:`tndm.models.fit_tn_family` are a cardinality
trend plus a tensor residual.  The trend depends on ``|S|`` only, so its exact
contribution to every index is added in closed form from its ``d + 1`` values.

Conventions match :mod:`tndm.attribution.exact`: group ``i`` is bit ``i`` of the
coalition mask and interaction subsets are tuples of group indices.

Note: these routines are release glue written for this repository and are
verified against exhaustive enumeration in ``tests/``.  The numbers reported in
the paper's tables were computed by enumerating surrogate predictions.
"""
from __future__ import annotations

import itertools
from math import ceil, comb
from typing import Sequence

import numpy as np
import torch

__all__ = ["multilinear_extension", "shapley_values", "banzhaf_values", "interaction",
           "pair_interactions", "triple_interactions", "joint_removal_effect"]


def _quadrature(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(max(1, int(nodes)))
    return (points + 1.0) / 2.0, weights / 2.0


def _tensor_part(fitted, probabilities: np.ndarray) -> np.ndarray:
    """Tensor residual of ``fitted`` at real-valued inclusion probabilities, (n, Q)."""
    order = list(fitted.tn_order)
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(probabilities, dtype=float)[:, order], dtype=torch.float64)
        return fitted.tn_module(x).numpy() * fitted.tn_scale


def _size_profile(fitted, d: int) -> np.ndarray:
    """Values of the cardinality trend at sizes 0..d, shape (d + 1, Q)."""
    masks = np.asarray([(1 << size) - 1 for size in range(d + 1)], dtype=np.int64)
    return np.asarray(fitted.base(masks), dtype=float)


def _symmetric_index(profile: np.ndarray, order: int, shapley: bool) -> np.ndarray:
    """Order-``order`` index of a function of ``|S|`` only (same for every subset)."""
    d = profile.shape[0] - 1
    result = np.zeros(profile.shape[1])
    for size in range(d - order + 1):
        delta = sum((-1.0) ** (order - take) * comb(order, take) * profile[size + take]
                    for take in range(order + 1))
        if shapley:
            weight = comb(d - order, size) / ((d - order + 1) * comb(d - order, size))
        else:
            weight = comb(d - order, size) * 2.0 ** -(d - order)
        result += weight * delta
    return result


def multilinear_extension(fitted, probabilities: np.ndarray) -> np.ndarray:
    """Tensor residual's expected value under independent inclusion, shape (n, Q).

    ``probabilities`` has shape ``(n, d)`` with entries in ``[0, 1]``; binary rows
    reproduce the residual's coalition predictions.
    """
    return _tensor_part(fitted, np.atleast_2d(probabilities))


def _difference(fitted, d: int, members: Sequence[int], t: float) -> np.ndarray:
    """Finite difference Delta_A of the tensor residual at the diagonal point ``t``."""
    order = len(members)
    rows, signs = [], []
    for take in range(order + 1):
        for chosen in itertools.combinations(members, take):
            row = np.full(d, t)
            row[list(members)] = 0.0
            row[list(chosen)] = 1.0
            rows.append(row)
            signs.append((-1.0) ** (order - take))
    values = _tensor_part(fitted, np.asarray(rows))
    return np.asarray(signs) @ values


def interaction(fitted, d: int, members: Sequence[int], shapley: bool = True) -> np.ndarray:
    """Shapley (or Banzhaf) interaction index of the groups in ``members``, shape (Q,).

    ``len(members) == 1`` gives the Shapley / Banzhaf value of a single group.
    """
    members = tuple(int(m) for m in members)
    order = len(members)
    if not 1 <= order <= d or len(set(members)) != order:
        raise ValueError("members must be distinct group indices")
    if shapley:
        nodes, weights = _quadrature(ceil((d - order + 1) / 2))
        tensor = sum(weight * _difference(fitted, d, members, float(node))
                     for node, weight in zip(nodes, weights))
    else:
        tensor = _difference(fitted, d, members, 0.5)
    return tensor + _symmetric_index(_size_profile(fitted, d), order, shapley)


def shapley_values(fitted, d: int) -> np.ndarray:
    """Shapley value of every group, shape (d, Q)."""
    return np.stack([interaction(fitted, d, (i,), shapley=True) for i in range(d)])


def banzhaf_values(fitted, d: int) -> np.ndarray:
    """Banzhaf value of every group, shape (d, Q)."""
    return np.stack([interaction(fitted, d, (i,), shapley=False) for i in range(d)])


def pair_interactions(fitted, d: int, shapley: bool = True) -> np.ndarray:
    """Pairwise interaction indices in ``itertools.combinations`` order, (d(d-1)/2, Q)."""
    return np.stack([interaction(fitted, d, pair, shapley)
                     for pair in itertools.combinations(range(d), 2)])


def triple_interactions(fitted, d: int, shapley: bool = True) -> np.ndarray:
    """Third-order interaction indices in ``itertools.combinations`` order."""
    return np.stack([interaction(fitted, d, triple, shapley)
                     for triple in itertools.combinations(range(d), 3)])


def joint_removal_effect(fitted, d: int, groups: Sequence[int]) -> np.ndarray:
    """Predicted utility lost when ``groups`` are removed from the full coalition, (Q,)."""
    full = (1 << d) - 1
    removed = full
    for group in groups:
        removed &= ~(1 << int(group))
    values = fitted.predict(np.asarray([full, removed], dtype=np.int64))
    return values[0] - values[1]
