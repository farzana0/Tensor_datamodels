# Paper-to-code map

Where the results covered by this repository come from. Paths are relative
to the repository root. "Verified" means the value was recomputed for this
release from the files listed and agrees with the paper at printed precision
(`scripts/check_paper_numbers.py`).

## Methods

| Paper | Code |
|---|---|
| TNDM-CP, TNDM-TT, tree (HT) parameterizations (Sec. 2.3, App. B) | `src/tndm/models/tensor_networks.py` — `QueryCP`, `QueryTT`, `QueryHT` |
| Fitting objective, validation selection (Sec. 2.3, App. B.3, I.5) | `train_tn`, `fit_tn_family` in the same file; `src/tndm/models/fit.py` |
| Cardinality adjustment (App. I.5) | `fit_cardinality` in `src/tndm/baselines/regression.py` |
| Linear, degree-2/3 ridge, degree-3 OMP, RBF baselines (App. H) | `src/tndm/baselines/regression.py` |
| TT degree ladder (App. I.8) | `src/tndm/models/degree_ladder.py` |
| Shapley, Banzhaf, LOO, pair / triple interaction indices from a value table | `src/tndm/attribution/exact.py` |
| Attribution by tensor contraction (App. F) | `src/tndm/attribution/contraction.py` (release glue, verified against enumeration in `tests/`) |
| Joint-removal effect (Sec. 4.1, App. J.2) | `contraction.joint_removal_effect`; paper values from `experiments/reasoning/d24/gpu/analyze.py` |

**Three fitting implementations.** The experiments were run with three separate
implementations, which this release keeps apart instead of merging:

| Used for | Implementation | Optimizer |
|---|---|---|
| 24 groups | `src/tndm/models` (generic `d`) | AdamW, at most 201 steps, validation early stopping |
| 8 and 12 groups | `experiments/reasoning/common/fit_query_paths.py`, `fit_query_paths_d12.py` (fixed `d`) | AdamW; 401 steps (d = 8), 201 steps (d = 12) |
| Modular retraining, AG News | `experiments/attribution/lib/surrogates.py` | alternating least squares (ALS) |
| Sport–Music | `experiments/sport_music/cp_query_mode.py`, `fitting_compute.py` | gradient descent, 1,500 epochs |

The tensor-network modules contract each mode with `[1, 1 − 2z]` rather than the
`[z, 1]` leg written in the paper. The two differ by an invertible change of basis
on every mode, so the function class and all ranks are the same.

## Main text

| Result | Code | Saved results | Status |
|---|---|---|---|
| **Table 1** — 24-group NMSE and joint-removal RMSE | `experiments/reasoning/d24/gpu/fit_and_freeze.py` (fits), `gpu/analyze.py` (metrics) | `gpu/cube_d24_k2_g3.npz`, `gpu/fits/seed*/budget_*/`, `gpu/analysis/result.json`, `tables/gpu_d24_nmse.csv` | Verified; analysis re-run from the released cube reproduces `result.json` exactly |
| **Fig. 3(a)** — 12-group interaction recovery | `experiments/reasoning/d12/code/fit_d12.py`, `analyze_d12.py` | `results/fits/`, `cubes/`, `analysis/size3_method_summary.csv` | Data verified (Table 7) |
| **Fig. 3(b)** — Shapley-based selection | `experiments/reasoning/d12/code/analyze_d12.py` | `analysis/size3_attribution_selection.csv` | Data verified (Table 10) |
| **Table 2** — Sport–Music R² | `experiments/sport_music/fitting_compute.py` | `results/FITTING_COMPUTE_mean_gold_logprob.json`, `results/coalition_*/UTILITY.json` | Verified |
| **Table 3** — AG News attribution at 128 observations | `experiments/attribution/run_question_a.py` | `experiments/attribution/results/question_a/agnews_lora_*.csv` | Verified (pooled median over 360 repetitions) |
| **Fig. 4** — modular retraining, three attribution quantities | `experiments/attribution/run_question_a.py`; additive-initialised TT: `experiments/modular_retraining/tt_additive_init/` | `tt_additive_init/data/plot_statistics_all_metrics.csv`, `historical/data/` | Claim at 64 observations verified. The saved figure script draws Shapley / Banzhaf / pairwise panels; third-order values are in `plot_statistics_all_metrics.csv` |

## Appendix

| Result | Code | Saved results | Status |
|---|---|---|---|
| Table 4 — task configurations | `experiments/reasoning/common/transformer_query_paths.py` (8), `transformer_query_paths_d12.py` (12), `d24/gpu/transformer_and_of_or_d24.py` (24) | — | Constants in code |
| Table 5 — NMSE, all families and budgets | as Table 1 | as Table 1 | Verified |
| Table 6 — joint and leave-one-out removal RMSE | `d24/gpu/analyze.py` (`effects`) | `gpu/analysis/result.json` | Verified (m = 1,024 columns checked) |
| Table 7 — interaction AP by budget | `d12/code/analyze_d12.py` | `analysis/size3_method_summary.csv` | Verified |
| Table 8 — size-four selection | `d12/code/analyze_slices.py` | `analysis/slice_summary.csv` | Verified |
| Table 9 — retrained top-1 at 24 groups | `d24/gpu/analyze.py` | `gpu/analysis/result.json` | Verified |
| Tables 10–12 — Shapley / Banzhaf based selection | `d12/code/analyze_d12.py` | `analysis/size3_attribution_selection.csv` | Table 10 verified; Tables 11–12 use the same file |
| Table 13 — Sport–Music configurations | `sport_music/fitting_compute.py` | `results/FITTING_COMPUTE_mean_gold_logprob.json` | Verified |
| Table 14 — AG News incl. KernelSHAP, MSR | `attribution/run_question_a.py`, `lib/direct.py` | `results/question_a/agnews_lora_*.csv` | Verified |
| Table 18 — LDS vs direct correlation | `reasoning/d8/lds/lds.py`, `extract_khop_d8.py`, `extract_khop_d12.py` | `lds/lds_per_fit_d8.csv`, `lds/lds_size3_d12.csv`, `d8/results/fits/` | Verified |
| Table 16 (J.8) — in-context learning, d = 500 | `experiments/appendix_tables/summarize.py` | `appendix_tables/results/table16_d500_candidates.csv` | Verified from saved per-candidate errors (56 values) |
| Table 17 (J.9), UCI HAR panel | `experiments/appendix_tables/summarize.py` | `appendix_tables/results/table17_uci_har_heldout.csv`; cube `data/cubes/atlas_uci_har_clients_d10` | Verified from saved per-split errors (18 values) |
| Table 17 (J.9), Qwen-probe panel; Table 19 (J.11) | `experiments/qwen_probe/probe_game.py` (coalition game, exact attributions), `summarize.py` | `qwen_probe/results/attribution_errors.csv`, `probe_d10_seed*.json` | Verified from saved attribution errors (30 values); the coalition game regenerates on CPU |

## Retraining pipelines

| Experiment | Learner / runner | Coalition designs | Scheduler scripts |
|---|---|---|---|
| Reasoning, 8 groups | `reasoning/common/transformer_query_paths.py` | all 256 coalitions | — |
| Reasoning, 12 groups | `reasoning/common/transformer_query_paths_d12.py`, `d12/code/query_paths_d12_runner.py` | `d12/configs/*.txt`, `MANIFEST.json` | `d12/run_all.sh`, `d12/code/*.sbatch` |
| Reasoning, 24 groups | `d24/gpu/transformer_and_of_or_d24.py` | `d24/configs/gpu_*_masks.txt`, `GPU_DESIGN.json` | `d24/run_all.sh`, `d24/configs/*.sbatch` |
| Modular | `modular_retraining/learner/known_interaction_game.py` | all 256 coalitions, 3 seeds | CPU, no scheduler needed |
| AG News | `ag_news/run_transformer_coalition.py`, `prepare_ag_news.py`, `prepare_e1.py` | `ag_news/configs/lora_rank_*.json` | `ag_news/slurm_e1_oracle.sbatch` |
| Sport–Music | `sport_music/run.py`, `coalitions.py`, `build_dataset.py` | `sport_music/results/DESIGN.json` | `sport_music/job.sbatch` |

The shared transformer for the reasoning tasks is
`experiments/reasoning/common/causal_gpt_arithmetic_model.py`; the task data and
metrics are in `transformer_arithmetic_fano.py`.
