#!/usr/bin/env python3
"""Game structure of the targeted arm, importable without torch or the repo's experiments package.

`PROVIDERS` is `pilot/game.py: balanced_incidence(2, 3, 24, 10, 1001)`, the same tuple
frozen verbatim in `gpu/transformer_and_of_or_d24.py` (equality checked below).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pilot"))
import game as _G  # noqa: E402

D, Q, K = 24, 10, 2
PROVIDERS = tuple(tuple(tuple(int(g) for g in role) for role in query)
                  for query in _G.balanced_incidence(K, 3, D, Q, 1001).queries)

_FROZEN = (
    ((3, 15, 16), (10, 12, 18)), ((6, 7, 14), (1, 19, 22)), ((4, 13, 23), (2, 8, 9)),
    ((0, 17, 21), (5, 11, 20)), ((0, 2, 9), (4, 14, 23)), ((6, 11, 21), (3, 13, 17)),
    ((10, 20, 22), (1, 5, 16)), ((7, 15, 19), (8, 12, 18)), ((12, 16, 17), (0, 1, 22)),
    ((2, 6, 10), (13, 18, 19)),
)
assert PROVIDERS == _FROZEN


def completed_queries(mask: int) -> list[bool]:
    return [all(any(mask & (1 << group) for group in providers) for providers in groups)
            for groups in PROVIDERS]
