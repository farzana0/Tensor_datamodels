#!/usr/bin/env python3
"""The degree-ladder's subset-count kernel ridge equals explicit monomial ridge."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game as G  # noqa: E402
import models as M  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    d = 12
    fit, new = G.uniform_masks(d, 300, rng), G.uniform_masks(d, 500, rng)
    y, weight = rng.standard_normal((300, 3)), None
    weight = M.size_weights(fit)
    worst = 0.0
    for degree in (1, 2, 3, 4, 6):
        term_list = M.terms(d, degree)
        for alpha in (1e-3, 1.0):
            explicit = M.poly_design(new, d, term_list) @ M.ridge_solve(
                M.poly_design(fit, d, term_list), y, weight, alpha)
            dual = M.kernel_ridge_dual(M.subset_kernel(G.bits_of(fit, d), G.bits_of(fit, d), degree),
                                       y, weight, alpha)
            kernel = M.subset_kernel(G.bits_of(new, d), G.bits_of(fit, d), degree) @ dual
            worst = max(worst, float(np.abs(explicit - kernel).max()))
    assert worst < 1e-8, worst
    print(f"kernel ridge == explicit monomial ridge; max abs diff {worst:.2e}")


if __name__ == "__main__":
    main()
