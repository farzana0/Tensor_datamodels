"""Loader and exact ground truth for the coalition cubes in ``data/cubes``.

    from cubes import load, available
    c = load("modular_relations_d8")
    v = c.table("triple_ABC")                    # dense 2**d vector, seed-averaged
    c.shapley("triple_ABC")                      # exact, by enumeration
    c.interaction("triple_ABC", order=2)         # Shapley interaction index
    c.banzhaf_interaction("triple_ABC", order=2) # uniform-background pair effects

Exact attributions are only defined for complete cubes (every one of the
``2**d`` coalitions measured).  On a sampled cube they raise ``IncompleteCube``
rather than silently estimating; use ``c.observed(output)`` and fit something.

All transforms go through the fast subset (zeta/Moebius) transform, so a d=12
cube costs 2**12 * 12 operations, not 3**12.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CUBE_DIR = os.path.join(HERE, "cubes")


class IncompleteCube(Exception):
    """Raised when an exact query is asked of a cube that is not exhaustive."""


# --------------------------------------------------------------------------
# subset transforms
# --------------------------------------------------------------------------

def mobius(v: np.ndarray) -> np.ndarray:
    """Moebius (interaction) coefficients m(S) of a full game v, in place-safe form.

    v is indexed by mask, ``v[S] = sum_{T subset S} m(T)``.
    """
    m = np.array(v, dtype=np.float64, copy=True)
    n = m.shape[0]
    d = int(n).bit_length() - 1
    if 1 << d != n:
        raise ValueError("game length is not a power of two")
    for i in range(d):
        bit = 1 << i
        idx = np.arange(n)
        hi = idx[(idx & bit) != 0]
        m[hi] -= m[hi ^ bit]
    return m


def zeta(m: np.ndarray) -> np.ndarray:
    """Inverse of :func:`mobius`: v(S) = sum_{T subset S} m(T)."""
    v = np.array(m, dtype=np.float64, copy=True)
    n = v.shape[0]
    d = int(n).bit_length() - 1
    for i in range(d):
        bit = 1 << i
        idx = np.arange(n)
        hi = idx[(idx & bit) != 0]
        v[hi] += v[hi ^ bit]
    return v


def popcount(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    c = np.zeros_like(x)
    y = x.copy()
    while np.any(y):
        c += (y & 1)
        y >>= 1
    return c


# --------------------------------------------------------------------------
# exact attributions from the Moebius coefficients
# --------------------------------------------------------------------------

def shapley_from_mobius(m: np.ndarray, d: int) -> np.ndarray:
    """phi_i = sum_{S containing i} m(S) / |S|."""
    size = popcount(np.arange(1 << d))
    w = np.zeros_like(m)
    nz = size > 0
    w[nz] = m[nz] / size[nz]
    phi = np.zeros(d)
    idx = np.arange(1 << d)
    for i in range(d):
        phi[i] = w[(idx & (1 << i)) != 0].sum()
    return phi


def banzhaf_from_mobius(m: np.ndarray, d: int) -> np.ndarray:
    """beta_i = sum_{S containing i} m(S) / 2**(|S|-1)."""
    size = popcount(np.arange(1 << d))
    w = np.zeros_like(m)
    nz = size > 0
    w[nz] = m[nz] / (2.0 ** (size[nz] - 1))
    beta = np.zeros(d)
    idx = np.arange(1 << d)
    for i in range(d):
        beta[i] = w[(idx & (1 << i)) != 0].sum()
    return beta


def interaction_from_mobius(m: np.ndarray, d: int, order: int, kind: str) -> dict:
    """Shapley ('shapley') or Banzhaf ('banzhaf') interaction index of given order.

    Shapley:  I(T) = sum_{S superset T} m(S) / (|S| - |T| + 1)
    Banzhaf:  I(T) = sum_{S superset T} m(S) / 2**(|S| - |T|)
              (equivalently the uniform average over backgrounds of the |T|-th
              order finite difference)
    """
    size = popcount(np.arange(1 << d))
    out = {}
    for T in combinations(range(d), order):
        tmask = 0
        for i in T:
            tmask |= 1 << i
        sup = np.flatnonzero((np.arange(1 << d) & tmask) == tmask)
        gap = size[sup] - order
        if kind == "shapley":
            w = 1.0 / (gap + 1.0)
        elif kind == "banzhaf":
            w = 1.0 / (2.0 ** gap)
        else:
            raise ValueError(f"unknown interaction kind {kind!r}")
        out[T] = float((m[sup] * w).sum())
    return out


def loo_from_game(v: np.ndarray, d: int) -> np.ndarray:
    """Removal effect Delta_i = v(N) - v(N \\ {i}) (positive = removing hurts)."""
    full = (1 << d) - 1
    return np.array([v[full] - v[full ^ (1 << i)] for i in range(d)])


# --------------------------------------------------------------------------
# the cube object
# --------------------------------------------------------------------------

@dataclass
class Cube:
    cube_id: str
    d: int
    players: list[str]
    masks: np.ndarray
    mask_bits: np.ndarray
    values: np.ndarray
    output_names: list[str]
    seeds: np.ndarray
    split: np.ndarray
    replicate: np.ndarray
    complete: bool
    meta: dict
    secondary_values: np.ndarray | None = None
    secondary_output_names: list[str] | None = None
    cell_notes: np.ndarray | None = None
    path: str = ""

    # -- basics ----------------------------------------------------------
    def __repr__(self) -> str:
        kind = "complete" if self.complete else "sampled"
        return (
            f"<Cube {self.cube_id} d={self.d} cells={len(self.masks)} "
            f"outputs={len(self.output_names)} seeds={sorted(set(self.seeds.tolist()))} {kind}>"
        )

    def output_index(self, output) -> int:
        if isinstance(output, (int, np.integer)):
            return int(output)
        try:
            return self.output_names.index(output)
        except ValueError as exc:
            raise KeyError(
                f"{self.cube_id}: no output {output!r}; have {self.output_names[:8]}"
                + (" ..." if len(self.output_names) > 8 else "")
            ) from exc

    def observed(self, output=0, seed=None) -> tuple[np.ndarray, np.ndarray]:
        """(masks, values) actually measured, for one output.  Works on any cube."""
        j = self.output_index(output)
        sel = np.ones(len(self.masks), dtype=bool) if seed is None else (self.seeds == seed)
        return self.masks[sel], self.values[sel, j]

    def table(self, output=0, seed=None, reduce="mean") -> np.ndarray:
        """Dense game of length ``2**d`` indexed by mask.

        With ``seed=None`` the cells of all seeds and replicates are combined by
        ``reduce`` ('mean' or 'first').  Raises unless the selection is complete.
        """
        j = self.output_index(output)
        sel = np.ones(len(self.masks), dtype=bool) if seed is None else (self.seeds == seed)
        masks, vals = self.masks[sel], self.values[sel, j]
        game = np.full(1 << self.d, np.nan)
        if reduce == "mean":
            sums = np.zeros(1 << self.d)
            cnts = np.zeros(1 << self.d)
            np.add.at(sums, masks, vals)
            np.add.at(cnts, masks, 1.0)
            nz = cnts > 0
            game[nz] = sums[nz] / cnts[nz]
        elif reduce == "first":
            for m, x in zip(masks[::-1], vals[::-1]):
                game[m] = x
        else:
            raise ValueError(f"unknown reduce {reduce!r}")
        missing = int(np.isnan(game).sum())
        if missing:
            raise IncompleteCube(
                f"{self.cube_id}: {missing}/{1 << self.d} coalitions were never measured"
                f"{'' if seed is None else f' for seed {seed}'}; "
                "use .observed() and fit a surrogate instead"
            )
        return game

    # -- exact ground truth ----------------------------------------------
    def mobius(self, output=0, seed=None) -> np.ndarray:
        return mobius(self.table(output, seed))

    def shapley(self, output=0, seed=None) -> np.ndarray:
        return shapley_from_mobius(self.mobius(output, seed), self.d)

    def banzhaf(self, output=0, seed=None) -> np.ndarray:
        return banzhaf_from_mobius(self.mobius(output, seed), self.d)

    def loo(self, output=0, seed=None) -> np.ndarray:
        return loo_from_game(self.table(output, seed), self.d)

    def interaction(self, output=0, seed=None, order=2) -> dict:
        return interaction_from_mobius(self.mobius(output, seed), self.d, order, "shapley")

    def banzhaf_interaction(self, output=0, seed=None, order=2) -> dict:
        return interaction_from_mobius(self.mobius(output, seed), self.d, order, "banzhaf")

    def efficiency_residual(self, output=0, seed=None) -> float:
        """|sum_i phi_i - (v(N) - v(empty))| relative to |v(N) - v(empty)|."""
        g = self.table(output, seed)
        span = g[(1 << self.d) - 1] - g[0]
        resid = abs(self.shapley(output, seed).sum() - span)
        return resid / max(abs(span), 1e-30)


def load(cube_id: str, cube_dir: str = CUBE_DIR) -> Cube:
    path = cube_id if cube_id.endswith(".npz") else os.path.join(cube_dir, cube_id + ".npz")
    z = np.load(path, allow_pickle=False)
    return Cube(
        cube_id=str(z["cube_id"]),
        d=int(z["d"]),
        players=[str(p) for p in z["players"]],
        masks=z["masks"],
        mask_bits=z["mask_bits"],
        values=z["values"],
        output_names=[str(o) for o in z["output_names"]],
        seeds=z["seeds"],
        split=np.array([str(s) for s in z["split"]]),
        replicate=z["replicate"],
        complete=bool(z["complete"]),
        meta=json.loads(str(z["meta_json"])),
        secondary_values=z["secondary_values"] if "secondary_values" in z.files else None,
        secondary_output_names=(
            [str(o) for o in z["secondary_output_names"]]
            if "secondary_output_names" in z.files
            else None
        ),
        cell_notes=(np.array([str(x) for x in z["cell_notes"]]) if "cell_notes" in z.files
                    else None),
        path=path,
    )


def available(cube_dir: str = CUBE_DIR) -> list[str]:
    if not os.path.isdir(cube_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(cube_dir) if f.endswith(".npz"))


def load_all(cube_dir: str = CUBE_DIR) -> Iterable[Cube]:
    for cid in available(cube_dir):
        yield load(cid, cube_dir)
