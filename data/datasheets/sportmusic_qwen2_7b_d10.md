# Sport-Music reconstruction, Qwen2-7B full fine-tuning, d=10, 96 coalitions

`sportmusic_qwen2_7b_d10`  ·  d = 10  ·  96 cells  ·  6 outputs  ·  sampled

## What one player is

A_i is the sport-fact group for subject block i, B_i the matching music-fact group; the designed prediction is that A_i and B_i are super-additive and mismatched pairs are near-additive

| # | player |
|---:|---|
| 0 | `A_0` |
| 1 | `A_1` |
| 2 | `A_2` |
| 3 | `A_3` |
| 4 | `A_4` |
| 5 | `B_0` |
| 6 | `B_1` |
| 7 | `B_2` |
| 8 | `B_3` |
| 9 | `B_4` |

## What `v(S)` is

mean correct-genre full-string log probability over the 80 held-out canonical probes, plus the same quantity within each of the 5 query classes

- **Evidence type:** retraining
- **Learner:** Qwen2-7B BASE, full-parameter fine-tuning, lr 1e-4, 100 full-batch epochs, trained to a convergence threshold of 0.01 train cross-entropy
- **Cost of one coalition:** ~166 seconds of multi-GPU full fine-tuning
- **Outputs:** overall mean gold log-probability, then the 5 per-class values
- **Secondary channel:** overall accuracy and mean music rank, then per-class accuracy
- **Licence:** derived from the Huang et al. Sport-Music out-of-context reasoning setup; base model Qwen2-7B under its own licence

## Measured cells

- coalition sizes 0–10; 88 distinct coalitions of the 1024 possible
- splits: `anchor_A_only` 1, `anchor_B_only` 1, `anchor_empty` 1, `anchor_full` 1, `leave_one_out` 10, `matched_pair` 5, `mismatched_pair` 5, `noise_replicate_of_anchor_full` 2, `noise_replicate_of_leave_one_out` 2, `noise_replicate_of_matched_pair` 2, `noise_replicate_of_random_p50` 2, `random_p50` 54, `singleton` 10
- learner seeds: 0 (88 cells), 1 (4 cells), 2 (4 cells)
- same-seed noise replicates: 8 cells repeat an already-measured coalition

## Value ranges

| output | min | mean | max |
|---|---:|---:|---:|
| `mean_gold_logprob` | -43.8187 | -15.2403 | -3.03509 |
| `class_0_mean_gold_logprob` | -35.7451 | -13.2679 | -2.71946e-06 |
| `class_1_mean_gold_logprob` | -53.7787 | -20.093 | -1.3262e-06 |
| `class_2_mean_gold_logprob` | -42.9528 | -15.5811 | -1.57869e-05 |
| `class_3_mean_gold_logprob` | -36.6332 | -13.4296 | -2.31238e-05 |
| `class_4_mean_gold_logprob` | -51.6729 | -13.8297 | -6.10947e-07 |

## Exact ground truth

Not available: this cube is sampled, so `cubes.py` raises `IncompleteCube` for Shapley, Banzhaf, LOO and interaction queries. Fit a surrogate on `cube.observed(output)` instead.

## Caveats

- SAMPLED: 1024 coalitions exist; 96 fine-tunes cover 88 distinct coalitions under a design frozen before any training (anchors, singletons, leave-one-out, matched and mismatched pairs, random p=0.5 coalitions, and noise replicates)
- 8 cells are deliberate noise replicates (replicate > 0; kinds noise_replicate_of_anchor_full, noise_replicate_of_leave_one_out, noise_replicate_of_matched_pair, noise_replicate_of_random_p50); they are the retraining-noise ceiling for this cube and must not be treated as independent coalitions when fitting
- a large always-present background of sport facts is in every coalition, so v(empty) is not an untrained model
- exact-match accuracy is a poor regression target here by design; the log-probability channel is primary

## Conventions

- bit convention: masks[n] has bit i (value 2**i) set iff player i is in the coalition; player 0 is the least significant bit

## Loading

```python
from cubes import load
c = load("sportmusic_qwen2_7b_d10")
masks, values = c.observed(0)
```
