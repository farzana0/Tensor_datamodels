"""Qwen representation-probe game: saved outcomes, exact attributions and a few probe refits."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "experiments" / "qwen_probe"
sys.path.insert(0, str(PROBE))

pytest.importorskip("sklearn")
import probe_game  # noqa: E402


@pytest.mark.parametrize("seed", [31, 43, 59])
def test_saved_exact_attributions_match_the_package(seed):
    saved = json.loads((PROBE / "results" / f"probe_d10_seed{seed}.json").read_text())
    utility = np.asarray(saved["utility"])
    assert utility.shape == (1024,) and utility[0] == 0.0
    recomputed = probe_game.exact_attributions(utility)
    for name in ("shapley", "banzhaf", "loo"):
        assert np.allclose(recomputed[name], saved["exact_first_order"][name], atol=1e-10), name
    assert np.allclose(recomputed["pair_interactions"], saved["exact_pair_interactions"], atol=1e-10)
    assert np.isclose(recomputed["shapley"].sum(), utility[-1] - utility[0])
    split = saved["split"]
    masks = split["train_masks"] + split["validation_masks"] + split["test_masks"]
    assert sorted(masks) == list(range(1024))


def test_probe_inputs_define_ten_two_class_groups():
    data = probe_game.prepare(seed=31)
    assert len(data["group_rows"]) == probe_game.DIMENSION
    assert data["features"].shape[1] == probe_game.PROJECTION_DIMENSION
    assert len(data["test_rows"]) == 112
    flipped = data["training_labels"] != data["true_labels"]
    assert set(np.flatnonzero(flipped)) == set(np.concatenate([data["group_rows"][g] for g in probe_game.CORRUPTED_GROUPS]))
