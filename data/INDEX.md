# Coalition cube index

Bit convention for every cube: `masks[n]` has bit `i` (value `2**i`) set iff player `i`
is in the coalition, and `mask_bits[n, i]` says the same thing explicitly.

| cube | d | cells | outputs | complete | evidence | utility |
|---|---:|---:|---:|---|---|---|
| `agnews_lora_rank16_seed31` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank16_seed43` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank16_seed59` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank1_seed31` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank1_seed43` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank1_seed59` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank2_seed31` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank2_seed43` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank2_seed59` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank4_seed31` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank4_seed43` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank4_seed59` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank64_seed31` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank64_seed43` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank64_seed59` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank8_seed31` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank8_seed43` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `agnews_lora_rank8_seed59` | 8 | 256 | 1 | yes | retraining | negative held-out probe NLL (higher is better) |
| `andor_d24_k2_g3` | 24 | 4115 | 310 | sampled | retraining | per-query-input log probability of the target continuation (10 queries x 31 inputs) |
| `atlas_ag_news_topics_d10` | 10 | 1024 | 1 | yes | closed-form refit | negative softmax log-loss on 2000 AG News test items (higher is better) |
| `atlas_qwen_coin_icl_d8` | 8 | 256 | 1 | yes | frozen-inference | negative mean query NLL (higher is better) |
| `atlas_uci_har_clients_d10` | 10 | 1024 | 1 | yes | closed-form refit | negative softmax log-loss on the official HAR test split; empty coalition = uniform six-class predictor (-log 6) (higher is better) |
| `khop_d12_noise_ceiling` | 12 | 320 | 310 | sampled | retraining | per-query-input log probability of the target continuation (10 queries x 31 inputs) |
| `khop_d12_replication` | 12 | 3060 | 310 | sampled | retraining | per-query-input log probability of the target continuation (10 queries x 31 inputs) |
| `khop_d8_transformer` | 8 | 256 | 248 | yes | retraining | per-query-input log probability of the target continuation (248 = 8 queries x 31 inputs); higher is better |
| `modular_relations_d8` | 8 | 768 | 4 | yes | retraining | mean log probability of the correct answer for each of four query families |
| `sportmusic_qwen2_7b_d10` | 10 | 96 | 6 | sampled | retraining | mean correct-genre full-string log probability over the 80 held-out canonical probes, plus the same quantity within each of the 5 query classes |
