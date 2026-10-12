"""Fitting: determinism under fixed seeds, selection consistency, exact representability."""
import numpy as np
import pytest

from tndm import fit_datamodel
from tndm.attribution import exact
from tndm.baselines.regression import poly_design, ridge_solve, terms
from tndm.utils.coalitions import all_masks, stratified_split


def _observations(and_of_ors, count=48, seed=0):
    d, masks, values = and_of_ors
    observed = np.sort(np.random.default_rng(seed).choice(masks, size=count, replace=False))
    return d, masks, values, observed


@pytest.mark.parametrize("family", ["cp", "tt", "ht", "linear", "degree2", "degree3", "omp", "rbf"])
def test_same_seeds_give_identical_predictions(and_of_ors, family):
    d, masks, values, observed = _observations(and_of_ors)
    first, choice_a = fit_datamodel(observed, values[observed], d, family, split_seed=5, ranks=(2, 4))
    second, choice_b = fit_datamodel(observed, values[observed], d, family, split_seed=5, ranks=(2, 4))
    assert np.array_equal(first.predict(masks), second.predict(masks))
    assert choice_a["validation_mse"] == choice_b["validation_mse"]


def test_reported_validation_error_matches_the_returned_model(and_of_ors):
    """The selected configuration's validation MSE is that of the model actually returned."""
    d, masks, values, observed = _observations(and_of_ors)
    y = values[observed][:, None]
    fit, validation = stratified_split(observed, 5)
    index = {int(m): i for i, m in enumerate(observed)}
    y_val = y[[index[int(m)] for m in validation]]
    for family in ("cp", "tt"):
        model, choice = fit_datamodel(observed, values[observed], d, family, split_seed=5, ranks=(2, 4))
        assert choice["rank"] in (2, 4)
        recomputed = float(np.mean((model.predict(validation) - y_val) ** 2))
        assert np.isclose(recomputed, choice["validation_mse"], rtol=1e-8, atol=1e-12)


def test_exactly_representable_low_degree_function_is_recovered():
    """A degree-2 set function observed on every coalition is recovered by its monomial ridge fit."""
    d = 6
    masks = all_masks(d)
    bits = ((masks[:, None] >> np.arange(d)) & 1).astype(float)
    values = bits @ np.linspace(-1, 1, d) + 2.0 * bits[:, 0] * bits[:, 1] - bits[:, 2] * bits[:, 5]
    term_list = terms(d, 2)
    design = poly_design(masks, d, term_list)
    coefficient = ridge_solve(design, values[:, None], np.ones(len(masks)), 1e-10)[:, 0]
    truth = np.zeros(len(term_list))
    for index, term in enumerate(term_list):
        if len(term) == 1:
            truth[index] = np.linspace(-1, 1, d)[term[0]]
    truth[term_list.index((0, 1))] = 2.0
    truth[term_list.index((2, 5))] = -1.0
    assert np.allclose(coefficient, truth, atol=1e-6)
    prediction = design @ coefficient
    assert np.allclose(exact.pair_interactions(prediction), exact.pair_interactions(values), atol=1e-6)
    # the monomial coefficients are the Moebius coefficients of the set function
    assert np.allclose(exact.mobius(values)[[sum(1 << g for g in term) for term in term_list]], truth, atol=1e-12)


def test_single_outcome_vector_is_accepted(and_of_ors):
    d, masks, values, observed = _observations(and_of_ors)
    model, choice = fit_datamodel(observed, values[observed], d, "cp", ranks=(2,))
    assert model.predict(masks).shape == (len(masks), 1)
    assert choice["rank"] == 2 and choice["parameters"] > 0
