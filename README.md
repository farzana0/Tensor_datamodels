# Tensor Network Datamodels for Training Data Attribution

Implementation accompanying a paper under review. It contains the
code and the saved results behind the paper's tables and figures.

## What is a tensor network datamodel?

A *datamodel* predicts the outcome of training on a subset of the training data.
For `d` training groups, the outcomes of all `2^d` subsets form an order-`d`
binary tensor with one mode per group. Linear datamodels give each group a fixed
contribution and cannot represent redundancy or complementarity between groups.

A **tensor network datamodel (TNDM)** fits a low-rank factorization of that
tensor from a small number of observed (subset, outcome) pairs:

- **TNDM-CP** — a rank-`R` CP decomposition, `O(dR)` parameters;
- **TNDM-TT** — a tensor train with ranks `r`, `O(dr²)` parameters.

Low tensor rank does not limit interaction *degree*: a function such as "one
group from each of several interchangeable sets must be present" has high degree
but small rank. One fitted TNDM then answers several attribution questions —
subset prediction, Shapley and Banzhaf values, higher-order interactions and
joint-removal effects — without further retraining.

## What this repository contains

| | |
|---|---|
| `src/tndm/` | Installable package: TNDM-CP / TNDM-TT / tree models and their fitting, baseline datamodels, exact and contraction-based attribution. |
| `experiments/` | The experiment pipelines behind the paper's tables and figures, with their saved results. |
| `data/` | 27 coalition cubes: the measured retraining outcomes used by the experiments. |
| `scripts/check_paper_numbers.py` | Recomputes 413 of the paper's tabulated numbers from the saved results. |
| `tests/` | Unit tests against exhaustive enumeration. |
| `docs/` | Paper-to-code map, reproducibility notes, third-party notices. |

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/farzana0/TNDM.git && cd TNDM
python -m venv .venv && source .venv/bin/activate
pip install -e .                    # core package: numpy + torch
pip install -e ".[experiments,dev]" # analysis scripts, CPU experiments, tests
pip install -e ".[finetuning]"      # only for the GPU fine-tuning experiments
```

The saved results were produced with Python 3.10.4, NumPy 1.26.4, SciPy 1.14.0,
scikit-learn 1.7.2 and PyTorch 2.9.0.

## Minimal example

```python
import numpy as np
from tndm import fit_datamodel
from tndm.attribution import contraction, exact
from tndm.utils import all_masks, bits_of

d = 8
masks = all_masks(d)          # a coalition is an integer bitmask; bit i = training group i
z = bits_of(masks, d)

# Toy outcome: needs one of groups {0, 1} AND one of groups {2, 3}.
v = np.maximum(z[:, 0], z[:, 1]) * np.maximum(z[:, 2], z[:, 3])

observed = np.sort(np.random.default_rng(0).choice(masks, size=96, replace=False))
model, choice = fit_datamodel(observed, v[observed], d, family="tt", ranks=(2, 4))

prediction = model.predict(masks)                         # every coalition, shape (256, 1)
shapley = contraction.shapley_values(model, d)            # by tensor contraction, shape (8, 1)
pairs = contraction.pair_interactions(model, d)           # Shapley interaction indices
effect = contraction.joint_removal_effect(model, d, groups=(0, 1))
```

`python examples/minimal_example.py` runs this comparison against a linear
datamodel and the exact game. Its output:

```text
held-out NMSE
  linear 0.6804
  cp     0.5976
  tt     0.1003
Shapley values (TNDM-TT):    [ 0.276  0.271  0.249  0.201 -0.035  0.012 -0.013  0.025]
Shapley values (exact game): [0.25 0.25 0.25 0.25 0.   0.   0.   0.  ]
pairwise interaction of the redundant groups (0, 1): -0.643 | exact: -0.667
predicted effect of removing groups 0 and 1 together: 1.318 | exact: 1.0
```

Gradient-fitted tensor networks are not bit-reproducible across machines (see
[Reproducibility](#reproducing-the-papers-experiments)), so expect small
differences in the tensor-network rows.

**Conventions.** A coalition is an integer whose bit `i` is set when group `i` is
included; a table of coalition values is indexed by that integer.
`family` is one of `cp`, `tt`, `ht`, `linear`, `degree2`, `degree3`, `omp`, `rbf`.

## Reproducing the paper's experiments

There are three levels, in increasing cost.

**1. Verify the saved results (seconds, CPU).** Recomputes 413 tabulated values
from the saved outcomes and frozen predictions and compares them with the paper:

```bash
python scripts/check_paper_numbers.py
```

**2. Refit the datamodels on saved retraining outcomes (CPU, minutes to hours).**
No learner is retrained.

| Paper result | Command (from the repository root) |
|---|---|
| Tables 1, 5, 6, 9 — 24 groups | `python experiments/reasoning/d24/gpu/analyze.py analyze` |
| Fig. 3, Tables 7, 10–12 — 12 groups | `python experiments/reasoning/d12/code/analyze_d12.py --root experiments/reasoning/d12/results --out <dir> --seeds 43 59` |
| Table 18 (d = 8 fits) | `python experiments/reasoning/d8/src/fit_budgets.py --repeat 1` (repeats 1–20) |
| Fig. 4 — modular retraining | `python experiments/attribution/run_question_a.py --cube modular_d8_seed31`, then `python experiments/modular_retraining/figure/plot_stabilization.py` |
| Tables 3, 14 — AG News | `python experiments/attribution/run_question_a.py --cube agnews_lora_r8_seed31` (18 cubes) |
| Tables 2, 13 — Sport–Music | `python experiments/sport_music/fitting_compute.py` |

Further saved-result summaries and a CPU-only coalition game:

```bash
python experiments/appendix_tables/summarize.py                 # Table 16, Table 17 (UCI HAR)
python experiments/qwen_probe/summarize.py                      # Table 17 (Qwen probe), Table 19
python experiments/qwen_probe/probe_game.py --seed 31 --check   # regenerate the probe game (1,023 refits)
```

Set `PYTHONPATH=.` (the repository root) for the `experiments/reasoning` scripts.
Scripts write into their experiment directory and overwrite the saved result
files of the same name.

**3. Regenerate the retraining outcomes (GPU).** Each experiment directory
contains the learner, the frozen coalition designs and the Slurm scripts that
produced the saved outcomes. See the table below and each experiment's README.

Details of what each level covers and how closely refits match the saved fits are
in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). The map from tables and
figures to code and result files is in [`docs/PAPER_MAP.md`](docs/PAPER_MAP.md).

### Experiments in this repository

| Experiment | Paper | Saved outcomes | Datamodel fits | Regenerating outcomes |
|---|---|---|---|---|
| Reasoning, 24 groups | Tables 1, 5, 6, 9 | cube, 3 seeds | frozen predictions included; analysis reproduces exactly | 4,115 from-scratch transformer runs, about 56 GPU-hours (RTX 8000) |
| Reasoning, 12 groups | Fig. 3, Tables 7, 8, 10–12 | cubes, 3 seeds | frozen predictions included; analysis reproduces exactly | about 47 GPU-hours (RTX 8000) |
| Reasoning, 8 groups | Table 18 | cube, 1 seed | 20 saved fitting repeats; refit is CPU-only | 256 runs, about 48 GPU-seconds each |
| Modular retraining | Fig. 4 | cube, 3 seeds | per-repetition metrics included | CPU, minutes |
| AG News RoBERTa–LoRA | Tables 3, 14 | 18 cubes | per-repetition metrics included | 4,608 LoRA fine-tunes |
| Sport–Music (Qwen2-7B) | Tables 2, 13 | 96 fine-tune utilities | refit is CPU-only | 96 full fine-tunes on an 80 GB GPU; dataset rebuilt by script (see below) |
| In-context prediction, d = 500 | Table 16 | — | saved per-candidate errors | — |
| Attribution accuracy, UCI HAR | Table 17 | cube `atlas_uci_har_clients_d10` | saved per-split errors | — |
| Attribution accuracy, Qwen probe | Tables 17, 19 | 3 seeds, exhaustive | saved attribution errors | CPU, seconds (`probe_game.py`) |

The GPU retraining pipelines are provided as they were run and were not
re-executed when this repository was assembled; their saved outcomes are what the
scripts above read.

## Repository structure

```text
TNDM/
├── src/tndm/
│   ├── models/          TNDM-CP, TNDM-TT, tree (HT), TT degree ladder, fit_datamodel
│   ├── attribution/     exact.py (enumeration), contraction.py (tensor contraction)
│   ├── baselines/       cardinality, polynomial ridge, sparse (OMP), RBF kernel ridge
│   └── utils/           coalition encodings, sampling, splits
├── experiments/
│   ├── reasoning/       relation-composition tasks: common/ (learner, fitters), d8/, d12/, d24/
│   ├── attribution/     one fit, many queries: datamodels vs direct estimators at equal budget
│   ├── modular_retraining/  learner, additive-initialised TT, Fig. 4
│   ├── ag_news/         RoBERTa–LoRA coalition fine-tuning pipeline
│   ├── appendix_tables/ saved result tables for Table 16 and Table 17 (UCI HAR)
│   ├── qwen_probe/      representation-probe coalition game, Tables 17 and 19
│   └── sport_music/     Qwen2-7B coalition fine-tuning and datamodel fits
├── data/                coalition cubes, datasheets, loader
├── scripts/             check_paper_numbers.py
├── examples/            minimal_example.py
├── tests/
└── docs/
```

Configurations (frozen coalition designs, model and training settings) and
results live inside each experiment directory rather than
in top-level `configs/` and `results/` folders, so that each pipeline keeps the
relative layout it was run with.

Three fitting implementations are used, each by the experiment it was written
for; they are not interchangeable and are documented in
[`docs/PAPER_MAP.md`](docs/PAPER_MAP.md):

- `src/tndm/models` — gradient-fitted CP / TT / tree with a cardinality baseline (24 groups);
- `experiments/reasoning/common/fit_query_paths*.py` — the same models with fixed `d` (8 and 12 groups);
- `experiments/attribution/lib/surrogates.py` — alternating-least-squares CP / TT (modular, AG News).

## Hardware and data requirements

- **Package, tests, example, number checks:** CPU, a few GB of memory.
- **Datamodel refits:** CPU. The 24-group fits take the longest (about ten
  surrogate families, several rank and order candidates each).
- **Relation-composition learners:** one GPU; saved runs used Quadro RTX 8000.
- **AG News:** `fancyzhx/ag_news` and `roberta-base`
  (revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`) from the Hugging Face Hub.
- **Sport–Music:** `Qwen/Qwen2-7B`
  (revision `453ed1575b739b5b03ce3758b23befdb0967f40e`), full-parameter
  fine-tuning on an 80 GB GPU. The fine-tuning documents are a reconstruction of
  the Sport–Music task of Huang et al. (2025) built from a third-party name list
  that carries no licence, so they are not redistributed here;
  `experiments/sport_music/build_dataset.py` rebuilds them from the pinned source.

Random seeds are fixed in the code and recorded with every result: learner seeds
31, 43 and 59 throughout; coalition-design and split seeds are listed per
experiment in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Tests

```bash
pytest                 # 76 tests, about half a minute on a recent CPU
pytest -m "not slow"   # skip the saved-results check
```

## Citation

The paper is under review. Citation information will be added
after the review period.

## License

A licence will be added after the review period. Third-party data and
models keep their own terms; see [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).
