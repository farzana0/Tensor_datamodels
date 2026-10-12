# Experiments

Each directory holds one experiment as it was run: code, frozen designs and
configurations, pre-registration and deviation notes where they exist, and the
saved results needed to recompute the paper's numbers. See
[`../docs/PAPER_MAP.md`](../docs/PAPER_MAP.md) for the table-by-table map and
[`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for what was verified.

| Directory | Paper | Compute to regenerate outcomes |
|---|---|---|
| `reasoning/d8` | App. J.1, Table 18 | 256 transformer runs, GPU |
| `reasoning/d12` | Sec. 4.2, Fig. 3, Tables 7, 8, 10–12, 18 | about 47 GPU-hours |
| `reasoning/d24` | Sec. 4.1, Tables 1, 5, 6, 9 | about 56 GPU-hours |
| `attribution` | Sec. 4.3–4.4, Fig. 4, Tables 3, 14 (shared comparison pipeline) | CPU |
| `modular_retraining` | Sec. 4.4, Fig. 4 | CPU, minutes |
| `ag_news` | Sec. 4.3, Tables 3, 14 (coalition fine-tuning) | 4,608 LoRA fine-tunes, GPU |
| `sport_music` | Sec. 4.3, Tables 2, 13 | 96 full Qwen2-7B fine-tunes, 80 GB GPU |
| `appendix_tables` | App. J.8–J.9, Table 16 and Table 17 (UCI HAR) | saved result tables only |

Run scripts from the repository root with the package installed
(`pip install -e ".[experiments]"`) and `PYTHONPATH=.` so that
`experiments.reasoning.common` is importable. Slurm scripts are the ones used for
the saved runs, with paths made repository-relative; partition names and
GPU types (`--partition`, `--gres`) will need adapting.

