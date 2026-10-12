# Modular retraining

Paper: Sec. 4.4, Fig. 4, Appendix I.7.

Eight training groups (`A_primary`, `B`, `C`, `A_replica` and four distractors),
each holding eleven facts of an affine relation modulo 11. The learner is seven
relation-specific 11 × 11 conditional-logit tables trained with 200 full-batch
Adam updates. All 256 coalitions are retrained for learner seeds 31, 43 and 59.

```text
learner/known_interaction_game.py     the learner and coalition retraining (CPU)
learner/compare_with_released_cube.py retrain and compare with data/cubes/modular_relations_d8.npz
tt_additive_init/                     additive-informed TT initialisation used for Fig. 4's TT curve:
                                      stabilized_tt.py and saved summaries (data/, reports/)
historical/                           summaries of the original fits of all other methods
figure/                               plotting scripts and the saved figure
```

## Retrain the learner (about one minute on CPU)

```bash
cd experiments/modular_retraining/learner
python known_interaction_game.py freeze --root runs/check
for i in 0 1 2; do python known_interaction_game.py worker --root runs/check --seed-index $i; done
python compare_with_released_cube.py --root runs/check
```

When this release was prepared the retrained outcomes matched the released cube
to 4.8e-7.

## Attribution comparison and figure

The per-repetition attribution errors come from
`experiments/attribution/run_question_a.py --cube modular_d8_seed{31,43,59}`
(saved in `experiments/attribution/results/question_a`). To redraw the saved
figure from the saved summaries:

```bash
python experiments/modular_retraining/figure/plot_stabilization.py
```

## Notes

- `tt_additive_init/stabilized_tt.py` is the additive-initialised TT method; its
  saved per-repetition metrics and summaries are in `tt_additive_init/data`.
- The saved figure has Shapley, Banzhaf and pairwise panels. The third-order
  values shown in the paper's Fig. 4(c) are in
  `tt_additive_init/data/third_order_sii_extension_statistics.csv` and
  `plot_statistics_all_metrics.csv`.
