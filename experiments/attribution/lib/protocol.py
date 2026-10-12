"""Frozen protocol constants shared by every equal-budget study script.

These are the pre-registered protocol constants.  Nothing here may be changed
after the pre-registration is committed; a change is recorded as a deviation and the
affected result is labelled exploratory.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPEATS = 20
BASE_SEED = 20260912

# Budgets are multiples of d, capped at half the cube.
BUDGET_MULTIPLES = (1, 2, 4, 8, 16, 32, 64)

# Size of the coalition whose selection regret is reported, per dimension.
SELECTION_SIZE = {5: 2, 8: 3, 10: 3}

# Top-k used for interaction precision@k: the d strongest exact pairs.
def precision_k(d: int) -> int:
    return d

# Truncated Monte Carlo tolerance, as a fraction of |v(N) - v(empty)|.
TMC_TOLERANCE_FRACTION = 0.01

# Pilot budget for the Question B diagnostics: min(8d, 2**d / 2).  4d was ruled out
# on synthetic games before freezing: at 4d there are more degree-2 Walsh terms
# than pilot rows and the cross-validated profile is uninformative.
PILOT_MULTIPLE = 8

# Question B decision-rule thresholds (frozen before any real cube was compared).
TAU_CARDINALITY_R2 = 0.90     # CV R^2 of cubic-in-|S|  -> cardinality + linear
TAU_LOW_DEGREE_R2 = 0.90      # CV R^2 of Walsh degree<=2 -> degree-2 ridge
TAU_LOW_DEGREE_FLOOR = 0.50   # below this, degree<=2 misses too much -> high degree
TAU_HIGH_DEGREE_GAIN = 0.10   # or RBF CV R^2 beats degree<=2 by this much
TAU_EFFECTIVE_RANK = 4.0      # 95%-energy max cut rank of the pilot RBF surface
RANK_ENERGY_FRACTION = 0.95

# Question C tolerances, as a fraction of the full-coalition utility range.
PRUNING_DELTAS = (0.01, 0.05, 0.10)
PRUNING_BUDGET_MULTIPLE = 4

DESIGNS = ("uniform", "d_optimal")
PRIMARY_DESIGN = "uniform"


def budgets(d: int) -> list[int]:
    """Multiples of d up to 2**d / 2, with the cap itself appended if it is not one."""
    cap = (1 << d) // 2
    grid = [d * multiple for multiple in BUDGET_MULTIPLES if d * multiple <= cap]
    if grid[-1] < cap:
        grid.append(cap)
    return grid


def budget_multiple(d: int, budget: int) -> float:
    return budget / d


def working_budget(d: int) -> int:
    """m* for Question B: min(16d, 2**d / 2), always a member of budgets(d)."""
    return min(16 * d, (1 << d) // 2)


def seed_for(cube: str, budget: int, repeat: int, salt: str = "") -> int:
    digest = hashlib.sha256(f"{BASE_SEED}|{cube}|{budget}|{repeat}|{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


def preregistration_sha256() -> str:
    path = ROOT / "PROTOCOL.md"  # optional protocol note; not part of this repository
    if not path.exists():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()
