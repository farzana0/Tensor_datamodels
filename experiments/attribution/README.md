# One fit, many attribution queries: datamodels vs direct estimators at equal budget

Paper: Sec. 4.3 (AG News, Tables 3 and 14) and Sec. 4.4 (modular retraining, Fig. 4);
Appendix I.7, J.6.

Every method receives the same number of coalition outcomes from a fully
enumerated cube. A datamodel is fitted once and then queried for Shapley values,
Banzhaf values, leave-one-out effects, pairwise and third-order Shapley
interactions and held-out predictions; direct estimators (KernelSHAP, MSR
Banzhaf, permutation sampling, …) spend the same budget on one quantity. Exact
references come from the full cube.

```text
lib/cubes.py        loads the modular and AG News cubes from data/cubes
lib/design.py       coalition sampling designs and the fit / validation split
lib/surrogates.py   linear, degree-2/3 ridge, OMP, RBF, CP (ALS), TT (ALS), ladders
lib/direct.py       KernelSHAP, MSR Banzhaf, permutation / stratified sampling, SII estimators
lib/exact.py        exact attributions (re-exported from tndm.attribution.exact)
lib/runner.py       one cube -> one row per (method, budget, repetition, output)
lib/protocol.py     frozen budgets, repetitions and seeds
run_question_a.py   entry point
run_shapiq.py       reference approximators from the shapiq library
smoke_test.py       synthetic games only
results/question_a  saved per-repetition metrics: 3 modular and 18 AG News cubes
tables/             saved median tables
```

```bash
python experiments/attribution/smoke_test.py
python experiments/attribution/run_question_a.py --cube modular_d8_seed31      # CPU
python experiments/attribution/run_question_a.py --cube agnews_lora_r8_seed31
```

`run_question_a.py` skips a cube whose result file already exists; delete or move
`results/question_a/<cube>.csv` to recompute it.

Budgets are 8, 16, 32, 64 and 128 coalitions (validation included), 20
repetitions each. CP and TT are fitted here by alternating least squares; the
24-group experiment uses the gradient-fitted models in `src/tndm` instead.

