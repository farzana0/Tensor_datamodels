"""Degree-preserving tensor-train contractions for binary coalition models.

For the local feature map ``[1, x_i]``, physical index zero contributes degree
zero and physical index one contributes degree one.  The dynamic program below
propagates their summed degree alongside the ordinary TT bond, implementing the
degree-preserving contraction of Convy and Whaley (2022), Appendix A.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def coefficient_tensor(
    dimension: int,
    terms: Iterable[tuple[int, ...]],
    coefficients: np.ndarray,
) -> np.ndarray:
    """Place multilinear coefficients in a binary physical-index tensor."""
    result = np.zeros((2,) * dimension, dtype=np.float64)
    for term, value in zip(terms, np.asarray(coefficients, dtype=np.float64), strict=True):
        index = [0] * dimension
        for feature in term:
            index[feature] = 1
        result[tuple(index)] = value
    return result


def tensor_train_svd(tensor: np.ndarray, max_rank: int) -> list[np.ndarray]:
    """Return a deterministic TT-SVD with every internal rank capped."""
    shape = tensor.shape
    if not shape or any(size != 2 for size in shape):
        raise ValueError("expected a nonempty binary coefficient tensor")
    cores: list[np.ndarray] = []
    remainder = np.asarray(tensor, dtype=np.float64)
    left_rank = 1
    for site in range(len(shape) - 1):
        matrix = remainder.reshape(left_rank * 2, -1)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = min(max_rank, len(singular))
        cores.append(u[:, :rank].reshape(left_rank, 2, rank))
        remainder = (singular[:rank, None] * vh[:rank]).reshape((rank,) + shape[site + 1 :])
        left_rank = rank
    cores.append(remainder.reshape(left_rank, 2, 1))
    return cores


@dataclass(frozen=True)
class DegreeLadderTT:
    """A TT whose contractions remain separated by polynomial degree."""

    cores: tuple[np.ndarray, ...]

    @classmethod
    def from_tensor(cls, tensor: np.ndarray, max_rank: int) -> "DegreeLadderTT":
        return cls(tuple(tensor_train_svd(tensor, max_rank)))

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(core.shape[2] for core in self.cores[:-1])

    def degree_contributions(self, x: np.ndarray, max_degree: int | None = None) -> np.ndarray:
        """Compute every degree contribution with a degree-preserving contraction."""
        values = np.asarray(x, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != len(self.cores):
            raise ValueError("feature dimension does not match the number of TT cores")
        cap = len(self.cores) if max_degree is None else int(max_degree)
        if not 0 <= cap <= len(self.cores):
            raise ValueError("max_degree is outside the valid range")
        state = np.ones((len(values), 1, 1), dtype=np.float64)
        for site, core in enumerate(self.cores):
            next_degree = min(site + 1, cap)
            updated = np.zeros((len(values), next_degree + 1, core.shape[2]), dtype=np.float64)
            for degree in range(state.shape[1]):
                updated[:, degree] += np.einsum("na,ab->nb", state[:, degree], core[:, 0, :])
                if degree + 1 <= cap:
                    selected = np.einsum("na,ab->nb", state[:, degree], core[:, 1, :])
                    updated[:, degree + 1] += values[:, site, None] * selected
            state = updated
        return state[:, :, 0]

    def predict(self, x: np.ndarray, max_degree: int | None = None) -> np.ndarray:
        return self.degree_contributions(x, max_degree).sum(axis=1)

    def dense_tensor(self) -> np.ndarray:
        """Reconstruct the represented coefficient tensor."""
        value = self.cores[0]
        for core in self.cores[1:]:
            value = np.tensordot(value, core, axes=([-1], [0]))
        return np.squeeze(value, axis=(0, -1))
