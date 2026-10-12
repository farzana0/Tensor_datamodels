# Coalition cubes

Measured outcomes of retraining (or refitting) a learner on subsets of its
training data. Each cube is a table of `v(S)`: `S` is a coalition of `d`
training groups and `v(S)` is the outcome after training on `S` alone.

This directory holds the 27 cubes behind the experiments in the paper (15,527
measured cells; 23 cubes are exhaustive). `INDEX.md` lists them, `datasheets/`
has one sheet per cube (learner, utility, seeds, caveats).
`index.json` records the SHA-256 of every cube file.

| Cube | Experiment |
|---|---|
| `khop_d8_transformer` | reasoning, 8 groups (one learner seed, exhaustive) |
| `khop_d12_replication`, `khop_d12_noise_ceiling` | reasoning, 12 groups |
| `andor_d24_k2_g3` | reasoning, 24 groups |
| `modular_relations_d8` | modular retraining (3 seeds, exhaustive) |
| `agnews_lora_rank{1,2,4,8,16,64}_seed{31,43,59}` | AG News RoBERTa–LoRA (exhaustive) |
| `sportmusic_qwen2_7b_d10` | Sport–Music, Qwen2-7B (sampled) |
| `atlas_ag_news_topics_d10`, `atlas_uci_har_clients_d10`, `atlas_qwen_coin_icl_d8` | broader settings (Appendix J.7) |

## Quick start

```python
import sys; sys.path.insert(0, "data")
from cubes import load, available

available()                               # cube ids
c = load("modular_relations_d8")
c.table("triple_ABC")                     # dense game of length 2**d, seed-averaged
c.shapley("triple_ABC")                   # exact Shapley values
c.observed("triple_ABC")                  # (masks, values) as measured; works on any cube
```

Only NumPy is required.

## Format

One compressed `.npz` per cube:

| field | shape | meaning |
|---|---|---|
| `masks` | `[n]` | coalition as an integer; **bit `i` (value `2**i`) set iff player `i` is in it** |
| `mask_bits` | `[n, d]` | the same coalition written out explicitly |
| `values` | `[n, m]` | primary utility, one column per output (query, target, class) |
| `output_names` | `[m]` | |
| `seeds` | `[n]` | learner seed of the cell, `-1` for deterministic fits |
| `replicate` | `[n]` | `> 0` only when the same coalition was retrained again as a noise measurement |
| `split` | `[n]` | the cell's role in the protocol it came from (fit, target, gate, held-out, …) |
| `complete` | scalar | every coalition present for every seed |
| `players` | `[d]` | player names |
| `meta_json` | scalar | learner, utility definition, cost, caveats, licence, caveats |
| `secondary_values` | `[n, k]` | optional second channel, such as accuracy next to log-probability |

## Before you use a cube

Every datasheet has a **Caveats** section. The ones that most often matter:

- **Frozen inference is not retraining.** `atlas_qwen_coin_icl_d8` varies the
  prompt, not the weights.
- **Seeds are not always independent.** `khop_d8_transformer` has one learner
  seed and no noise measurement.
- **Sampled cubes** (`khop_d12_*`, `andor_d24_k2_g3`, `sportmusic_qwen2_7b_d10`)
  refuse exact queries. Their evaluation cells were chosen by a protocol, so they
  are not a uniform sample of the cube.
- **Adverse cubes are included on purpose.** On the AG News LoRA family, kernel
  and degree-truncated baselines beat tensor-network datamodels when attribution
  error is aggregated across budgets.
