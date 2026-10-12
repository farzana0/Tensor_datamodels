# AG News: RoBERTa–LoRA coalition fine-tuning

Paper: Sec. 4.3, Tables 3 and 14, Appendix J.6.

Eight optional training groups of 320 documents (clean, duplicated, short, long,
label-corrupted, the correctly labelled counterpart, and two class-enriched
groups) are added to a fixed base of 2,000 documents. Each coalition fine-tunes a
fresh LoRA adapter and classification head from the same `roberta-base`
initialisation; the outcome is the negative held-out probe loss. All 256
coalitions were run for adapter ranks {1, 2, 4, 8, 16, 64} and learner seeds
{31, 43, 59}: 4,608 fine-tunes.

```text
prepare_ag_news.py            build the groups and probe set from fancyzhx/ag_news
prepare_e1.py                 freeze the six rank configurations and the job manifest
run_transformer_coalition.py  one coalition fine-tune (resumable; --smoke for a tiny run)
aggregate_e1.py               collect the outcomes into cubes
slurm_e1_oracle.sbatch        four coalitions per task on four GPUs
configs/                      transformer.yaml and the frozen lora_rank_*.json
results/                      oracle_cube.csv (all 4,608 utilities), determinism and preflight audits
```

The resulting cubes are released as `data/cubes/agnews_lora_rank*_seed*.npz`.
The attribution comparison of Tables 3 and 14 is in `experiments/attribution`.

```bash
export PYTHONPATH=.
python experiments/ag_news/prepare_ag_news.py --help
python experiments/ag_news/run_transformer_coalition.py --mask 255 --seed 31 \
    --config experiments/ag_news/configs/lora_rank_8.json --smoke
```

**Status.** The scripts were moved from the original repository layout and their
paths adjusted. Imports, the pre-registration hash check and argument parsing
were checked; no fine-tuning run was executed in the new layout. `shared.py`,
`queries.py` and the pre-registration also cover five further experiments of the
original study that are not part of the paper.

Requires a GPU, `transformers`, `peft`, `datasets`, and access to the Hugging
Face Hub (or a local cache passed with `--hf-cache`).
