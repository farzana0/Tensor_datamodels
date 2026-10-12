# Tensor-Network Datamodels (TNDM)

Code, data and saved results for *Tensor Network Datamodels for Training Data
Attribution*.

A *datamodel* predicts the outcome of training on a subset of the training data.
For `d` training groups, the outcomes of all `2^d` subsets form an order-`d`
binary tensor with one mode per group. Linear datamodels assign each group a
fixed contribution and cannot represent redundancy or complementarity between
groups.

A tensor-network datamodel fits a low-rank factorization of that tensor from a
small number of observed (subset, outcome) pairs:

- **TNDM-CP**: a rank-`R` CP decomposition, `O(dR)` parameters;
- **TNDM-TT**: a tensor train with ranks `r`, `O(dr²)` parameters.

Low tensor rank does not limit interaction degree. One fitted model answers
several attribution queries without further retraining: subset prediction,
Shapley and Banzhaf values, higher-order interactions and joint-removal effects.

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/farzana0/TNDM.git && cd TNDM
python -m venv .venv && source .venv/bin/activate
pip install -e .          # core package: numpy, torch
pip install -e ".[dev]"   # adds pytest and pandas for the tests and scripts
```

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

`python examples/minimal_example.py` runs this against a linear datamodel and
the exact game:

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

A coalition is an integer whose bit `i` is set when group `i` is included, and a
table of coalition values is indexed by that integer. `family` is one of `cp`,
`tt`, `ht`, `linear`, `degree2`, `degree3`, `omp`, `rbf`.

## Repository structure

```text
src/tndm/
  models/        TNDM-CP, TNDM-TT, tree (HT), TT degree ladder, fit_datamodel
  attribution/   exact.py (enumeration), contraction.py (tensor contraction)
  baselines/     cardinality, polynomial ridge, sparse (OMP), RBF kernel ridge
  utils/         coalition encodings, sampling, splits
data/            coalition cubes, index, loader
results/         saved results, one folder per experiment
scripts/         checks of the paper's reported numbers
examples/        minimal_example.py
tests/
```

## Data and saved results

`data/cubes/` holds 27 coalition cubes: measured outcomes of retraining a
learner on subsets of its training groups (15,527 cells; 23 cubes are
exhaustive). They cover the relation-composition tasks with 8, 12 and 24
groups, modular retraining, AG News RoBERTa–LoRA fine-tuning and Sport–Music
fine-tuning of Qwen2-7B, plus three broader settings. [`data/README.md`](data/README.md) describes the format
and the loader, and [`data/INDEX.md`](data/INDEX.md) lists every cube.

`results/` holds the saved results behind the paper's tables and figures:

| Paper result | Folder |
|---|---|
| Tables 1, 5, 6, 9: reasoning, 24 groups | `results/reasoning_24_groups/` |
| Fig. 3b, Tables 7, 8, 10: reasoning, 12 groups | `results/reasoning_12_groups/` |
| Table 18: reasoning, 8 and 12 groups | `results/reasoning_8_groups/` |
| Fig. 4: modular retraining | `results/modular_retraining/` |
| Tables 3, 14: AG News RoBERTa–LoRA | `results/ag_news/` |
| Tables 2, 13: Sport–Music (Qwen2-7B) | `results/sport_music/` |
| Table 16, Table 17 (UCI HAR) | `results/appendix_tables/` |
| Table 17 (Qwen probe), Table 19 | `results/qwen_probe/` |

No third-party dataset or model weights are redistributed; the cubes and results
are numerical outcomes only.

## Tests and verification

```bash
pytest                                        # 75 tests, about a minute on a CPU
python scripts/check_paper_numbers.py         # recomputes 413 reported values
python scripts/summarize_appendix_tables.py   # prints Table 16 and Table 17 (UCI HAR)
python scripts/summarize_probe.py             # prints Table 17 (Qwen probe) and Table 19
```

`check_paper_numbers.py` recomputes each value from the saved results and
compares it with the paper at its printed precision. It needs only NumPy and
pandas and runs in a few seconds.

## Reproducibility

- **Reproducible here:** the package and its tests, the minimal example, and
  the 413 reported values listed above, recomputed from the saved results.
- **Not reproducible here:** the retraining runs that produced the cubes and
  the datamodel fits that produced the saved results. Those pipelines (GPU
  training, fine-tuning and fitting scripts) are not part of this repository.
  Tables and figures not listed above are not covered.
- Gradient-fitted tensor networks are not bit-reproducible across machines and
  library versions, so the tensor-network rows of the example can differ
  slightly. The saved results were produced with Python 3.10.4, NumPy 1.26.4
  and PyTorch 2.9.0; learner seeds are 31, 43 and 59 throughout.

## License

No license has been selected yet.
