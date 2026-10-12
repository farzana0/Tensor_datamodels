# Qwen representation-probe task

Paper: Appendix J.9 (Table 17, second panel) and Appendix J.11 (Table 19).

A logistic-regression probe is trained on fixed representations of a frozen
`Qwen/Qwen2.5-1.5B-Instruct` model (layer 14, last token) to tell two families of
prompt templates apart. The ten training groups are template pairs; three of
them carry flipped training labels. For each of the 1,024 coalitions the probe
is refitted (`C = 1`) on a 32-dimensional projection fitted once on all training
representations, and the utility is log 2 minus the mean true-label log-loss on
112 held-out examples (unseen statements and unseen templates). The language
model itself is never updated.

```bash
python experiments/qwen_probe/probe_game.py --seed 31           # regenerate the game: 1,023 probe refits on CPU
python experiments/qwen_probe/probe_game.py --seed 31 --check   # compare with the saved outcomes
python experiments/qwen_probe/summarize.py                      # Table 17 (probe panel) and Table 19
```

| File | Contents |
|---|---|
| `probe_game.py` | Builds the coalition game from the cached representations and computes exact attributions with `tndm.attribution.exact`. |
| `data/probe_inputs.npz` | Cached representations (1,512 × 1,536, float32) of the 1,400 training and 112 held-out examples, with their labels, training-group index (0–9) and held-out flag. No prompt text and no template or statement identifiers. |
| `results/probe_d10_seed{31,43,59}.json` | Saved coalition outcomes, exact attributions, the fit / validation / test coalition split (512 / 205 / 307) and the validation-selected candidate of each surrogate family. |
| `results/attribution_errors.csv` | Per seed, family and attribution quantity: RMSE against the complete game, and the gap between each model's attribution routine and enumeration of the same fitted model. |
| `summarize.py` | Reduces `attribution_errors.csv` to the two tables. |

**Reproducibility.** `probe_game.py --check` reproduces the saved outcomes of
all three seeds bit for bit (checked with NumPy 1.26.4 and 2.2.6). The script
runs its linear algebra on one BLAS thread, as the saved run did, because the
result is sensitive to floating-point rounding: the probe solver (liblinear,
default tolerance) stops after about ten iterations, and a change of 1e-12 in
the projected representations is enough to move about 14% of the coalition
outcomes by more than 1e-6. With the solver run to convergence the same
perturbation changes no outcome by more than 2e-7. The saved game is the one
defined by the default tolerance.

With a different linear-algebra library or instruction set a minority of
outcomes can therefore still move slightly; `--check` reports the gaps and
returns a non-zero status when the utility gap exceeds 0.05 or a first-order
attribution gap exceeds 5e-3. The surrogate fits behind the two tables are
released as saved errors; they are not refitted by this directory.

The statements wrapped by the templates come from TruthfulQA (Lin et al., 2022;
Apache-2.0). Only numerical representations are distributed here.
