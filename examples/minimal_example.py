"""Minimal example: fit a TNDM to a few coalition outcomes and query it.

Runs on a laptop CPU in well under a minute.

    python examples/minimal_example.py
"""
import numpy as np

from tndm import fit_datamodel
from tndm.attribution import contraction, exact
from tndm.utils import all_masks, bits_of

d = 8                                   # training groups
masks = all_masks(d)                    # coalition S <-> integer bitmask (bit i = group i)
z = bits_of(masks, d)                   # (2**d, d) inclusion matrix

# A toy "retraining outcome": the query is answered only if the coalition contains
# one of groups {0, 1} AND one of groups {2, 3}.  Interaction degree 4, TT rank 2.
v = np.maximum(z[:, 0], z[:, 1]) * np.maximum(z[:, 2], z[:, 3])

# Observe 96 of the 256 coalitions, as if each one were a retraining run.
rng = np.random.default_rng(0)
observed = np.sort(rng.choice(masks, size=96, replace=False))
held_out = np.setdiff1d(masks, observed)

print("held-out NMSE")
models = {}
for family in ("linear", "cp", "tt"):
    models[family], choice = fit_datamodel(observed, v[observed], d, family=family, ranks=(2, 4))
    error = models[family].predict(held_out)[:, 0] - v[held_out]
    print(f"  {family:6s} {np.mean(error ** 2) / np.var(v[held_out]):.4f}")

# One fitted TNDM answers several attribution queries, by contracting the network.
tt = models["tt"]
shapley = contraction.shapley_values(tt, d)[:, 0]
pairs = contraction.pair_interactions(tt, d)[:, 0]            # order of itertools.combinations
removal = contraction.joint_removal_effect(tt, d, groups=(0, 1))[0]

print("Shapley values (TNDM-TT):   ", np.round(shapley, 3))
print("Shapley values (exact game):", np.round(exact.shapley_values(v), 3))
print("pairwise interaction of the redundant groups (0, 1):", round(float(pairs[0]), 3),
      "| exact:", round(float(exact.pair_interactions(v)[0]), 3))
print("predicted effect of removing groups 0 and 1 together:", round(float(removal), 3), "| exact: 1.0")

# The contractions agree with exhaustive enumeration of the fitted model's own predictions.
assert np.allclose(shapley, exact.shapley_values(tt.predict(masks))[:, 0], atol=1e-9)
