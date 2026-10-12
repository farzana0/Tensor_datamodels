# AGENTS.md

Guidance for coding agents (and new contributors) working in this repository.
Start with `README.md` for what the project is; this file covers how to work in
it safely.

## What this repository is

Code and saved results for *Tensor Network Datamodels for Training Data
Attribution*. Two kinds of content live here, and they are treated differently:

- **Library code** in `src/tndm/` — an installable package. Normal software
  rules apply: change it, test it.
- **Research artifacts** in `experiments/` and `data/` — the pipelines as they
  were run, plus the saved outcomes behind the paper's tables and figures.
  These are a record. Do not "improve" them in ways that change numbers.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[experiments,dev]"
```

The `finetuning` extra is only needed for the GPU pipelines, which you should
not run as part of ordinary development.

## Commands to run before proposing a change

```bash
pytest                                   # 76 tests, about a minute on CPU
pytest -m "not slow"                     # skips the saved-results check
python scripts/check_paper_numbers.py    # 413 paper values recomputed from saved results
python examples/minimal_example.py       # output must match the block in README.md
```

All three must pass. `check_paper_numbers.py` must report `413 of 413`; if a
change makes it report anything else, the change is wrong, not the check.

For work on the probe experiment, also run:

```bash
python experiments/qwen_probe/probe_game.py --seed 31 --check   # expect "bitwise identical"
```

## Layout

| Path | Contents |
|---|---|
| `src/tndm/models/` | TNDM-CP, TNDM-TT, tree (HT) models, TT degree ladder, `fit_datamodel` |
| `src/tndm/attribution/` | `exact.py` (enumeration) and `contraction.py` (tensor contraction) |
| `src/tndm/baselines/` | cardinality, polynomial ridge, sparse (OMP), RBF kernel ridge |
| `src/tndm/utils/` | coalition encodings, sampling, splits |
| `experiments/<name>/` | one pipeline each, with its own configs, results and README |
| `data/cubes/` | 27 coalition cubes (`.npz`), checksummed in `data/index.json` |
| `scripts/check_paper_numbers.py` | the paper-number check |
| `docs/` | `PAPER_MAP.md` (table/figure to code), `REPRODUCIBILITY.md`, `THIRD_PARTY.md` |

To find the code behind a specific table or figure, read `docs/PAPER_MAP.md`
rather than searching.

## Conventions

- **Coalitions are integer bitmasks.** Bit `i` is set when training group `i` is
  included; group 0 is the least significant bit. A table of coalition values
  is indexed by that integer and has length `2**d`. Every function in the
  package and every cube follows this. Do not introduce another encoding.
- **Seeds are fixed and recorded.** Learner seeds are 31, 43 and 59 throughout.
  Do not change a seed, a default rank grid or a solver tolerance: saved
  results depend on them.
- **Three fitting implementations exist on purpose** (see `docs/PAPER_MAP.md`):
  `src/tndm/models` (24 groups), `experiments/reasoning/common/fit_query_paths*.py`
  (8 and 12 groups) and `experiments/attribution/lib/surrogates.py`
  (alternating least squares). They are not interchangeable. Do not merge or
  deduplicate them.
- **Run scripts from the repository root.** The `experiments/reasoning` scripts
  also need `PYTHONPATH=.`.
- **Style:** match the file you are editing. In `src/tndm/` lines stay within
  120 characters and functions carry type hints and short docstrings. No
  formatter or linter is configured; do not reformat files you are not
  otherwise changing.

## Things that will bite you

- **Experiment scripts overwrite saved results.** Most scripts write into their
  own experiment directory under the same file names as the saved outputs. To
  try one out, copy the repository (or the experiment directory) somewhere else
  first, or check `git status` afterwards and restore anything you did not mean
  to change.
- **Gradient-fitted tensor networks are not bit-reproducible** across machines
  or library versions. A refit that differs from the saved fit in the fourth
  decimal is expected; see `docs/REPRODUCIBILITY.md` for the measured sizes.
  Closed-form baselines should agree to about 1e-12.
- **The probe game is sensitive to floating-point rounding.** It runs on one
  BLAS thread deliberately. Do not remove the thread limit in
  `experiments/qwen_probe/probe_game.py`.
- **Cube files are checksummed.** Any edit to a file in `data/cubes/` breaks
  `tests/test_released_data.py`. Cubes are measurements; they should not be
  edited at all.
- **Some inputs are intentionally absent.** The Sport–Music fine-tuning
  documents are rebuilt by `experiments/sport_music/build_dataset.py` and are
  not stored here. Model weights, checkpoints and per-coalition raw outputs are
  excluded by `.gitignore`; keep it that way.

## Boundaries

Do without asking:

- fix bugs, add tests and improve documentation in `src/tndm/`, `tests/`,
  `examples/` and the READMEs;
- add new, clearly separate functionality to the package with tests.

Ask first:

- anything that changes a saved result file, a cube, a frozen coalition design
  or a configuration under `experiments/`;
- changing a tolerance or threshold in a check or test so that it passes;
- adding a dependency to the core package (it currently needs only NumPy and
  PyTorch);
- running the GPU pipelines or submitting Slurm jobs.

Never:

- regenerate reference data to make a failing comparison pass;
- edit a number in the documentation to match new output without understanding
  why it changed;
- commit model weights, caches, logs, credentials, or files containing
  usernames, host names or absolute paths.

## When a check fails

Find the cause before changing anything. A mismatch between recomputed and
saved numbers is information about the environment, the code or the data, and
the README and `docs/REPRODUCIBILITY.md` make specific claims about which
differences are expected. Report what differs, by how much and why, and leave
thresholds and saved outputs alone unless a maintainer decides otherwise.
