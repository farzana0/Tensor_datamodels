"""Contraction-based attribution of fitted TNDMs equals exhaustive enumeration of the same model."""
import itertools

import numpy as np
import pytest

from tndm.attribution import contraction, exact

FAMILIES = ("cp", "tt", "ht")


@pytest.mark.parametrize("family", FAMILIES)
def test_predictions_have_expected_shape_and_are_finite(fitted, family):
    d, masks, models = fitted
    prediction = models[family].predict(masks)
    assert prediction.shape == (1 << d, 2) and np.isfinite(prediction).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_multilinear_extension_reproduces_coalition_predictions(fitted, family):
    d, masks, models = fitted
    model = models[family]
    bits = ((masks[:, None] >> np.arange(d)) & 1).astype(float)
    residual = model.predict(masks) - model.base(masks)
    assert np.allclose(contraction.multilinear_extension(model, bits), residual, atol=1e-10)


@pytest.mark.parametrize("family", FAMILIES)
def test_shapley_and_banzhaf_by_contraction(fitted, family):
    d, masks, models = fitted
    table = models[family].predict(masks)
    assert np.allclose(contraction.shapley_values(models[family], d), exact.shapley_values(table), atol=1e-9)
    assert np.allclose(contraction.banzhaf_values(models[family], d), exact.banzhaf_values(table), atol=1e-9)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("shapley", [True, False])
def test_pair_and_triple_interactions_by_contraction(fitted, family, shapley):
    d, masks, models = fitted
    table = models[family].predict(masks)
    assert np.allclose(contraction.pair_interactions(models[family], d, shapley),
                       exact.pair_interactions(table, shapley), atol=1e-9)
    assert np.allclose(contraction.triple_interactions(models[family], d, shapley),
                       exact.triple_interactions(table, shapley), atol=1e-9)


@pytest.mark.parametrize("family", FAMILIES)
def test_fourth_order_interaction_against_finite_differences(fitted, family):
    """Order-4 Banzhaf interaction = average 4th finite difference over all backgrounds."""
    d, masks, models = fitted
    table = models[family].predict(masks)
    members = (0, 1, 2, 3)
    rest = [p for p in range(d) if p not in members]
    total = np.zeros(table.shape[1])
    for size in range(len(rest) + 1):
        for background in itertools.combinations(rest, size):
            base = sum(1 << p for p in background)
            for take in range(5):
                for chosen in itertools.combinations(members, take):
                    total += (-1.0) ** (4 - take) * table[base | sum(1 << p for p in chosen)]
    total /= 2 ** len(rest)
    assert np.allclose(contraction.interaction(models[family], d, members, shapley=False), total, atol=1e-9)


@pytest.mark.parametrize("family", FAMILIES)
def test_joint_removal_effect(fitted, family):
    d, masks, models = fitted
    table = models[family].predict(masks)
    full = (1 << d) - 1
    removed = full & ~0b000011
    assert np.allclose(contraction.joint_removal_effect(models[family], d, (0, 1)), table[full] - table[removed])
