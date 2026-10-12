# Reproducibility notes

What can be reproduced from this repository, at what cost, and what was re-run
and compared when it was assembled.

## 1. What "reproduce" means here

| Level | What runs | Cost |
|---|---|---|
| A. Recompute reported numbers | metrics from saved outcomes and frozen surrogate predictions | seconds, CPU |
| B. Refit datamodels | every surrogate family, on saved coalition outcomes | minutes to hours, CPU |
| C. Regenerate outcomes | the learner, once per coalition | GPU (except the modular task) |

Level A is covered by `scripts/check_paper_numbers.py` for Tables 1, 2, 3, 5, 6,
7, 8, 9, 10, 13, 14, 16, 17, 18 and 19 and the Fig. 4 claim (413 values).

## 2. Checks performed for this release

All of the following were run; the numbers are the observed results.

**Port fidelity (does the released code compute what the original code computed?)**

| Check | Result |
|---|---|
| `tndm` package vs the original 24-group implementation, all ten surrogate families, three synthetic arms (d = 8, 12, 24), same environment | maximum absolute prediction difference **0.0** (`docs/verification/core_vs_original.json`) |
| Released 8-group fitting code vs the original code, same environment, budgets 24 and 128 | maximum absolute difference **0.0** |
| Released cubes vs the original per-coalition records: 8-group cube (256 × 8), 21 modular and AG News cubes | maximum absolute difference **0.0** |

**Recomputation from released files**

| Check | Result |
|---|---|
| 24 groups: `gpu/analyze.py analyze` from the released cube and frozen predictions | NMSE table byte-identical to the saved table; `result.json` identical except the recorded GPU names, which are only available in the per-coalition records |
| 12 groups: `analyze_d12.py` for blind seeds 43 and 59 | all four output tables identical to the saved ones (5,740 rows, maximum numeric difference 0.0) |
| `scripts/check_paper_numbers.py` | 413 of 413 values agree with the paper at printed precision |

**Refits compared with saved fits (cross-machine reproducibility)**

| Check | Result |
|---|---|
| 8 groups, repeat 1, all five budgets, refit today vs fit saved in September | closed-form families (cardinality, linear, ridge, OMP, RBF, TT ladder) agree to 1e-12. Gradient-fitted networks differ: maximum pointwise difference 4e-5 (CP), 0.40 (TT), 3.2 (HT) nats; held-out NMSE at 128 observations 0.170 → 0.171 (TT), 0.115 → 0.119 (HT) |
| Sport–Music `fitting_compute.py`, refit today vs saved | tuned baselines identical (0.579, 0.556, 0.565). CP single fit 0.660 vs saved 0.668; best-of-8 0.616 vs 0.616; ensemble of 8 seeds 0.719 vs 0.719; ensemble over ranks 3–6 0.734 vs 0.734; rank × regularisation 0.705 vs 0.706 |
| Modular learner: all 768 coalitions retrained (3 seeds × 256, about 17 s per seed on CPU) | maximum difference from the released cube **4.8e-7** |

**Conclusion on determinism.** With fixed seeds the code is deterministic on one
machine (tests assert identical predictions across repeated fits). Across
machines, closed-form fits reproduce to round-off, while gradient-fitted tensor
networks reproduce only approximately: small floating-point differences are
amplified by several hundred optimiser steps and by validation-based selection.
Reported differences were at most 0.008 in R² and 0.005 in NMSE in the cases
checked. The frozen predictions shipped with the 12- and 24-group experiments
are the exact arrays behind the paper's tables.

**Tests and self-checks**

| Check | Result |
|---|---|
| `pytest` (76 tests) | 76 passed |
| `examples/minimal_example.py` | runs; output shown in the README |
| `experiments/attribution/smoke_test.py` | passed ("smoke test ok") |
| `experiments/reasoning/d24/pilot/test_kernel_equivalence.py` | passed (maximum difference 4.6e-10) |
| `experiments/reasoning/d8/src/test_cube.py` | passed |
| AG News pipeline: imports, pre-registration hash check, command-line parsing | passed |
| `experiments/attribution/validate_exact.py` (exact attributions vs `shapiq` 1.3.1, released cubes) | passed; worst absolute discrepancy 1.6e-15 |
| `experiments/attribution/run_shapiq.py --cube modular_d8_seed31` | ran to completion (12 s) |
| `experiments/qwen_probe/probe_game.py` vs the original implementation, same environment, seed 31 | maximum absolute difference **0.0** over 1,024 coalitions |
| `experiments/qwen_probe/probe_game.py --check`, seeds 31, 43 and 59, NumPy 1.26.4 and 2.2.6 | bitwise identical to the saved outcomes; see `experiments/qwen_probe/README.md` for the sensitivity to floating-point rounding |

The GPU retraining pipelines (level C) were not re-executed when the repository
was assembled; their Slurm scripts pass a shell syntax check and call only
scripts that are present.

## 3. Per-experiment details

### Reasoning tasks (relation composition)

Learner: 8-layer, width-256 decoder-only transformer trained from scratch for
1,000 AdamW steps (batch 512, learning rate 1e-3, weight decay 0.1); one model
per coalition; learner seeds 31, 43, 59.

| | 8 groups | 12 groups | 24 groups |
|---|---|---|---|
| Coalitions retrained | all 256, seed 31 | fixed design; seeds 31, 43, 59 | 1,024 training + 256 test + 46 evaluation masks per seed |
| Design seed | — | pivoted-QR design, `rng_seed` 20260912 | `default_rng(2001)`, nested prefixes |
| Fit / validation split | `default_rng(7000 + repeat)` per repeat | seed 20260912 | `stratified_split(·, 6001)` |
| Budgets | 24, 32, 48, 64, 128 | 32, 48, 64, 96, 128, 160 new coalitions | 256, 512, 1,024 |
| Recorded compute | about 48 GPU-s per coalition | 46.9 GPU-hours | 56.1 GPU-hours |
| GPU | not recorded per cell | Quadro RTX 8000 (blind seeds) | Quadro RTX 8000 |

Entry points: `experiments/reasoning/d{8,12,24}`; the 12- and 24-group
directories keep their `run_all.sh` with the full ordered pipeline (design
freeze → learner gate → coalition retraining → fit and freeze → evaluation
coalitions → analysis).

### Modular retraining and AG News attribution (one fit, many queries)

`experiments/attribution/run_question_a.py --cube <name>`: 20 repetitions per
budget in {8, 16, 32, 64, 128}; observation sets and splits are derived from
`BASE_SEED = 20260912` by hashing (cube, budget, repetition). CPU only.
Fig. 4's TNDM-TT curve uses the additive-informed initialisation in
`experiments/modular_retraining/tt_additive_init/stabilized_tt.py`
(300 fits, saved summaries in `tt_additive_init/data`).

### AG News RoBERTa–LoRA

`roberta-base` with LoRA on query/value projections, ranks {1, 2, 4, 8, 16, 64},
250 AdamW updates, batch 16, learning rate 2e-4, maximum length 128; eight
optional groups of 320 documents on a fixed base of 2,000; 256 coalitions × 3
seeds × 6 ranks = 4,608 fine-tunes. Configurations:
`experiments/ag_news/configs/lora_rank_*.json`. A saved determinism audit
(`results/determinism_audit.json`) found bitwise-equal logits across two
independent runs of four coalitions.

### Sport–Music

Full-parameter fine-tuning of Qwen2-7B, 100 full-batch epochs, learning rate
3e-5; 96 runs (88 coalitions at one seed plus 8 noise replicates), roughly three
GPU-minutes each on an 80 GB A100. Datamodel comparison: 15 shared coalition
splits (`default_rng(3)`), CPU.

## 4. Files regenerated rather than shipped

Per-coalition retraining records, model checkpoints, caches and logs are not
shipped; the compact cubes and frozen predictions carry the outcomes the analyses
read. The Sport–Music fine-tuning documents are rebuilt by
`experiments/sport_music/build_dataset.py`.

## 5. Environment

Saved results: Python 3.10.4, NumPy 1.26.4, SciPy 1.14.0, scikit-learn 1.7.2,
pandas 2.3.2, PyTorch 2.9.0 (CUDA 12.8), transformers 4.57.3, peft 0.19.1,
shapiq 1.3.1. The reasoning learners assert Python 3.10 in their Slurm scripts.
