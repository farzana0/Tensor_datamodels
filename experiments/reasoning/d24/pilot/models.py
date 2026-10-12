"""Surrogate families used by the d = 24 experiment.

The implementations live in the installable ``tndm`` package; this module only
re-exports them under the names the experiment scripts use (``import models as M``).
"""
from tndm.baselines.regression import (ALPHAS, CHUNK, Model, OffsetModel, cardinality_design,  # noqa: F401
                                        fit_cardinality, fit_omp, fit_polynomial, fit_rbf,
                                        poly_design, ridge_solve, terms)
from tndm.models.degree_ladder import fit_ladder, kernel_ridge_dual, subset_kernel  # noqa: F401
from tndm.models.tensor_networks import (FAMILIES, RANKS, REGULARIZATION, QueryCP, QueryHT,  # noqa: F401
                                          QueryTT, fit_tn_family, train_tn)
from tndm.utils.coalitions import size_weights, stratified_split  # noqa: F401
