# Relation-composition reasoning tasks (8, 12 and 24 training groups)

Paper: Sec. 4.1–4.2, Appendix I.1–I.6, I.8, J.1–J.5, J.10.

A query composes three additive relations modulo 31. Training groups supply
relation examples; several groups supply the same relation (redundant) and
different relations must be combined (complementary). One transformer is trained
from scratch per coalition and the outcome is the final-answer log-probability.

```text
common/   shared learner and fitters
          causal_gpt_arithmetic_model.py   the transformer
          transformer_arithmetic_fano.py   task data, loss, metrics
          transformer_query_paths.py       8-group retraining runner
          transformer_query_paths_d12.py   12-group retraining runner
          fit_query_paths.py               CP / TT / HT fitter, d = 8
          fit_query_paths_d12.py           all surrogate families, d = 12
          analyze_query_paths_d12.py       AP, third differences, Shapley / Banzhaf selection
d8/       src/ (cube access, surrogate families, fit_budgets.py), results/fits (20 repeats),
          lds/ (LDS metric and saved per-fit values)
d12/      code/, configs/ (frozen designs), cubes/ (outcomes), results/fits (frozen predictions),
          analysis/, tables/, run_all.sh
d24/      pilot/ (ideal game, family builder), gpu/ (learner, fit-and-freeze, analysis, cube,
          frozen predictions), configs/ (frozen masks, Slurm scripts), tables/, run_all.sh
```

## Recompute the paper's numbers (CPU, no fitting)

```bash
export PYTHONPATH=.
python experiments/reasoning/d24/gpu/analyze.py analyze          # Tables 1, 5, 6, 9
python experiments/reasoning/d12/code/analyze_d12.py \
    --root experiments/reasoning/d12/results --out /tmp/d12_analysis --seeds 43 59   # Tables 7, 10-12
```

The first command rewrites `d24/gpu/analysis/result.json` and
`d24/tables/gpu_d24_nmse.csv`. Without the per-coalition records it cannot
recover the GPU names stored in the original `result.json`; every numeric field
is reproduced exactly.

## Refit the datamodels (CPU)

```bash
python experiments/reasoning/d8/src/fit_budgets.py --repeat 1                 # about 7 minutes per repeat
python experiments/reasoning/d24/gpu/fit_and_freeze.py --seed 31 --budget 256  # needs per-coalition records, see below
```

`fit_and_freeze.py` and `d12/code/fit_d12.py` read the per-coalition JSON
records written by the retraining runners (`d24/gpu/observed/`,
`d12/results/observed/` …), which are not shipped. To refit from the released
outcomes, either regenerate the records or adapt the loader to the cube, as
`analyze.py` and `analyze_d12.py` already do.

## Regenerate the retraining outcomes (GPU)

`d12/run_all.sh` and the header of `d24/run_all.sh` list the ordered stages
(design freeze, learner gate, coalition retraining, fit and freeze, evaluation
coalitions, analysis). Fits are frozen before evaluation outcomes exist;
`fit_and_freeze.py` refuses to run if evaluation records are already present.
Adapt `--partition` and `--gres` in the Slurm files to your cluster.

## Notes

- Tensor orders use the known relation-to-group assignment (Appendix I.5).
- Seed 31 at 12 groups was used during development; its original records are
  not shipped (set `TNDM_D12_DEV_SEED31` if you have them).
- `d8` contains the fitting pipeline used for Table 18.
