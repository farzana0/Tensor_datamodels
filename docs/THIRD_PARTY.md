# Third-party material

This repository does not redistribute third-party datasets or model weights.
The saved results are numerical outcomes (utilities, predictions, metrics)
produced for the paper.

| Item | Used for | Terms | Shipped here? |
|---|---|---|---|
| AG News (`fancyzhx/ag_news`; Zhang et al., 2015) | AG News experiments | original dataset terms | No — downloaded by `experiments/ag_news/prepare_ag_news.py` |
| `roberta-base` (revision `e2da8e2f…`) | AG News LoRA fine-tuning | MIT (model card) | No |
| `Qwen/Qwen2-7B` (revision `453ed157…`) | Sport–Music fine-tuning | see the model card | No |
| Sport–Music task (Huang et al., NeurIPS 2025) | Sport–Music experiment | task design from the paper | Reconstruction script only |
| `names.csv` from `jiahai-feng/extractive-structures` (commit `2e3fdbe7…`) | subject names in the Sport–Music documents | **no licence file in the source repository** | **No.** The reconstructed documents embed these names and are therefore not shipped; `build_dataset.py` rebuilds them from the pinned file |
| UCI HAR, AG News (derived utilities) | `data/cubes/atlas_*` | underlying dataset terms | Derived utilities only |
| GSM8K (`openai/gsm8k`; Cobbe et al., 2021) and `Qwen/Qwen2.5-1.5B-Instruct` | in-context experiment of Table 16 | MIT (dataset); see the model card | No — only derived error tables in `experiments/appendix_tables` |
| TruthfulQA (Lin et al., 2022) | statements behind the Qwen-probe representations | Apache-2.0 | No — only numerical representations in `experiments/qwen_probe/data` |
| `shapiq` | reference implementations of KernelSHAP and related estimators | MIT | No (dependency) |

**Code.** The transformer in
`experiments/reasoning/common/causal_gpt_arithmetic_model.py` adopts
architectural ideas from *modded-nanogpt* (rotary embeddings, QK normalisation,
ReLU² MLPs, value residuals, logit capping). It is an independent implementation
and contains no code copied from that project.

