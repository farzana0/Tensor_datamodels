# Appendix tables verifiable from saved results

Paper: Appendix J.8 (Table 16) and Appendix J.9 (Table 17, UCI HAR panel).

These two experiments are released as **saved result tables only**. The numbers in
the paper are recomputed from them in a second. The surrogate fits themselves
are not re-run here.

```bash
python experiments/appendix_tables/summarize.py
```

| File | Contents |
|---|---|
| `results/table16_d500_candidates.csv` | Every fitted candidate of the d = 500 in-context experiment: task (coin flip, GSM8K), seed (31, 43), fitting budget (400, 800), family, variable ordering, validation and test RMSE, and whether it was the validation-selected candidate of its family and ordering. `source_sha256` is the checksum of the original fit record the row was read from. |
| `results/table17_uci_har_heldout.csv` | UCI HAR classifier-refit task with ten training groups: per split (seeds 12001–12010), budget (80, 160, 320 fitting coalitions, 32 validation) and surrogate, the held-out prediction error and the attribution RMSE against the fully enumerated game. |
| `summarize.py` | Reduces the two files to the tables printed in the paper. |

**Table 16** reports the mean and sample standard deviation over the two seeds of
the test RMSE of the validation-selected candidate. In the result file,
`orderwise_cp` is the paper's "coefficient CP", `tensor_train` the ordinary TT,
and `rbf_kernel` the RBF baseline.

**Table 17 (UCI HAR)** reports, at 320 fitting coalitions, the mean and sample
standard deviation over the ten splits; `additive` is the paper's linear
surrogate. The other budgets and surrogates in the file are not in the paper.

The coalition outcomes of the d = 500 tasks are not part of this directory.
