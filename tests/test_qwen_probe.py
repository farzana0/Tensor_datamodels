"""Qwen representation-probe game: the saved outcomes and exact attributions are consistent with the package."""
import json
from pathlib import Path

import numpy as np
import pytest

from tndm.attribution import exact

RESULTS = Path(__file__).resolve().parents[1] / "results" / "qwen_probe"
DIMENSION = 10


@pytest.mark.parametrize("seed", [31, 43, 59])
def test_saved_exact_attributions_match_the_package(seed):
    saved = json.loads((RESULTS / f"probe_d10_seed{seed}.json").read_text())
    utility = np.asarray(saved["utility"])
    assert utility.shape == (1 << DIMENSION,) and utility[0] == 0.0
    shapley = np.ravel(exact.shapley_values(utility))
    assert np.allclose(shapley, saved["exact_first_order"]["shapley"], atol=1e-10)
    assert np.allclose(np.ravel(exact.banzhaf_values(utility)), saved["exact_first_order"]["banzhaf"], atol=1e-10)
    assert np.allclose(np.ravel(exact.loo_values(utility)), saved["exact_first_order"]["loo"], atol=1e-10)
    pair = np.zeros((DIMENSION, DIMENSION))
    for (i, j), value in zip(exact.pairs(DIMENSION), np.ravel(exact.pair_interactions(utility))):
        pair[i, j] = pair[j, i] = value
    assert np.allclose(pair, saved["exact_pair_interactions"], atol=1e-10)
    assert np.isclose(shapley.sum(), utility[-1] - utility[0])
    split = saved["split"]
    masks = split["train_masks"] + split["validation_masks"] + split["test_masks"]
    assert sorted(masks) == list(range(1 << DIMENSION))
