"""One-call fitting of a single datamodel family from coalition observations.

This composes the same steps, in the same order, that the d = 24 reasoning
experiment applies to every family (``experiments/reasoning/d24``): a
cardinality-stratified fit/validation split, inverse-frequency size weights, the
cardinality trend, and then the chosen residual family selected on validation
error.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from tndm.baselines.regression import fit_cardinality, fit_omp, fit_polynomial, fit_rbf
from tndm.models.tensor_networks import RANKS, fit_tn_family
from tndm.utils.coalitions import size_weights, stratified_split

__all__ = ["fit_datamodel"]


def fit_datamodel(masks: np.ndarray, y: np.ndarray, d: int, family: str = "cp", *,
                  split_seed: int = 0, orders: Sequence[Sequence[int]] | None = None,
                  ranks: Sequence[int] = RANKS):
    """Fit one datamodel to observed coalition outcomes.

    Parameters
    ----------
    masks : (m,) integer coalition bitmasks (bit ``i`` = training group ``i``).
    y : (m,) or (m, Q) measured outcomes, one column per outcome / query.
    d : number of training groups.
    family : ``"cp"``, ``"tt"``, ``"ht"`` (tensor networks), ``"linear"``,
        ``"degree2"``, ``"degree3"`` (ridge), ``"omp"`` (sparse degree 3) or ``"rbf"``.
    split_seed : seed of the 80/20 cardinality-stratified fit/validation split.
    orders : candidate group orderings for the tensor network; validation selects
        among them.  Defaults to the natural order.  CP is order-invariant.
    ranks : candidate ranks for the tensor-network families.

    Returns
    -------
    (model, choice) : ``model.predict(masks)`` returns ``(n, Q)`` predictions;
        ``choice`` records the selected hyperparameters and validation error.
    """
    masks = np.asarray(masks, dtype=np.int64)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    fit_masks, val_masks = stratified_split(masks, split_seed)
    index = {int(mask): position for position, mask in enumerate(masks)}
    y_fit = y[[index[int(m)] for m in fit_masks]]
    y_val = y[[index[int(m)] for m in val_masks]]
    weight = size_weights(fit_masks)
    base, _ = fit_cardinality(fit_masks, y_fit, val_masks, y_val, weight, d)
    res_fit, res_val = y_fit - base(fit_masks), y_val - base(val_masks)

    if family in ("cp", "tt", "ht"):
        orders = (tuple(range(d)),) if orders is None else tuple(tuple(o) for o in orders)
        return fit_tn_family(family, d, y.shape[1], fit_masks, res_fit, val_masks, res_val,
                             weight, base, orders, ranks=tuple(ranks))
    if family in ("linear", "degree2", "degree3"):
        degree = {"linear": 1, "degree2": 2, "degree3": 3}[family]
        return fit_polynomial(degree, fit_masks, res_fit, val_masks, res_val, weight, d, base)
    if family == "omp":
        return fit_omp(3, fit_masks, res_fit, val_masks, res_val, d, base)
    if family == "rbf":
        return fit_rbf(fit_masks, res_fit, val_masks, res_val, d, base)
    raise ValueError(f"unknown family {family!r}")
