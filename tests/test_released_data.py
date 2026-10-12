"""Integrity of the coalition cubes and saved experiment results."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def test_cube_files_match_recorded_checksums():
    index = json.loads((DATA / "index.json").read_text())
    assert index["n_cubes"] == len(index["cubes"]) == len(list((DATA / "cubes").glob("*.npz")))
    for entry in index["cubes"]:
        digest = hashlib.sha256((DATA / "cubes" / entry["file"]).read_bytes()).hexdigest()
        assert digest == entry["sha256"], entry["cube_id"]


def test_cube_masks_follow_the_bit_convention():
    for path in sorted((DATA / "cubes").glob("*.npz")):
        cube = np.load(path, allow_pickle=False)
        d = int(cube["d"])
        rebuilt = cube["mask_bits"].astype(np.int64) @ (np.int64(1) << np.arange(d, dtype=np.int64))
        assert np.array_equal(rebuilt, cube["masks"]), path.name
        assert cube["values"].shape[0] == len(cube["masks"])
        assert np.isfinite(cube["values"]).all(), path.name


def test_modular_cube_exact_attributions_are_consistent():
    sys.path.insert(0, str(DATA))
    from cubes import load

    from tndm.attribution import exact
    cube = load("modular_relations_d8")
    for seed in (31, 43, 59):
        rows = np.flatnonzero(cube.seeds == seed)
        rows = rows[np.argsort(cube.masks[rows])]
        table = cube.values[rows]
        phi = exact.shapley_values(table)
        assert np.allclose(phi.sum(0), table[-1] - table[0])
        # groups 0 and 3 teach the same relation: identical Shapley values for the composition task
        column = list(cube.output_names).index("triple_ABC")
        assert np.isclose(phi[0, column], phi[3, column], atol=1e-9)


@pytest.mark.slow
def test_paper_numbers_recompute_from_saved_results():
    """Every tabulated number checked by scripts/check_paper_numbers.py agrees with the paper."""
    result = subprocess.run([sys.executable, str(REPO / "scripts/check_paper_numbers.py"), "--strict",
                             "--only-mismatches"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
