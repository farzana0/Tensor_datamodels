# AGENTS.md

Notes for coding agents and contributors working on the code in this
repository. See `README.md` for what the project does.

## Environment

Python 3.10 or newer. The core package depends only on NumPy and PyTorch.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only build, much smaller
pip install -e ".[dev]"
```

Everything here runs on a CPU; no GPU is needed.

Known-good version sets:

| Python | NumPy | PyTorch | pandas | pytest |
|---|---|---|---|---|
| 3.10.4 | 1.26.4 | 2.9.0 | 2.3.2 | 8.4.2 |
| 3.10.4 | 2.2.6 | 2.14.1 (CPU) | 2.3.3 | 9.1.1 |

## Commands

Run from the repository root.

```bash
pytest                                   # full suite, 75 tests, about a minute
pytest -m "not slow"                     # skips the one test that reads every saved result file
pytest tests/test_fitting.py -k tt       # a single file or pattern
python examples/minimal_example.py       # end-to-end example; output is shown in README.md
python scripts/check_paper_numbers.py    # prints "413 of 413 values agree" when the result files are intact
```

Run `pytest` and the example before submitting a change.

## Layout

| Path | Contents |
|---|---|
| `src/tndm/models/` | tensor-network models and `fit_datamodel`, the main entry point |
| `src/tndm/attribution/` | `exact.py` (enumeration) and `contraction.py` (tensor contraction) |
| `src/tndm/baselines/` | regression baselines |
| `src/tndm/utils/` | coalition encodings, sampling, splits |
| `tests/` | pytest suite; shared fixtures in `tests/conftest.py` |
| `examples/` | `minimal_example.py` |
| `scripts/` | standalone scripts that read `results/` |
| `data/`, `results/` | data files read by the tests and scripts |

## Code conventions

- **Coalitions are integer bitmasks.** Bit `i` is set when group `i` is
  included; group 0 is the least significant bit. Arrays of coalition values
  are indexed by that integer and have length `2**d`. Use
  `tndm.utils.all_masks` and `tndm.utils.bits_of` rather than re-deriving the
  encoding.
- **Shapes.** Models take integer masks of shape `(n,)` and return predictions
  of shape `(n, K)`, one column per output. Attribution functions return
  `(d, K)`.
- **Public API.** Most modules declare `__all__`. Keep it up to date when
  adding or renaming a public function.
- **Style.** Match the surrounding file: lines up to 120 characters, type
  hints on function signatures, short docstrings. No formatter or linter is
  configured, so avoid reformatting code you are not changing.
- **Dependencies.** Keep `src/tndm/` importable with NumPy and PyTorch only.
  Anything else belongs in the `dev` extra in `pyproject.toml`.

## Reproducibility

- **Pass seeds explicitly.** Fitting functions take seed arguments (for
  example `split_seed`); do not rely on global random state. Use
  `np.random.default_rng(seed)` for new randomness.
- **Tests are deterministic.** `tests/conftest.py` sets PyTorch to one thread,
  and fitting with the same seeds is expected to give identical predictions
  within a machine (`tests/test_fitting.py` checks this). New tests should
  follow the same pattern: fixed seeds, small problem sizes, comparison
  against exhaustive enumeration where possible.
- **Across machines,** gradient-based fits can differ in the last digits, so
  compare fitted values with a tolerance rather than exact equality. Closed-form
  code paths can be compared tightly.
- **The README example is checked output.** If a change alters what
  `examples/minimal_example.py` prints, update the output block in `README.md`
  in the same change and say why it moved.

## Data files

`data/cubes/*.npz` are checksummed in `data/index.json` and verified by
`tests/test_released_data.py`. The scripts in `scripts/` locate files in
`results/` by name. Treat both directories as read-only inputs; changes to
them should go through an issue first.

## Pull requests

- One logical change per pull request, with tests for new behaviour.
- Keep generated files out of commits: `__pycache__`, `*.egg-info`, model
  weights and logs are already covered by `.gitignore`.
- Describe how the change was tested, including the Python, NumPy and PyTorch
  versions used.
