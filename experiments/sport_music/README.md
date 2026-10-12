# Sport–Music: Qwen2-7B coalition fine-tuning

Paper: Sec. 4.3, Tables 2 and 13, Appendix J.6.

Adapted from the Sport–Music task of Huang et al. (2025). Five sport–music
associations give ten optional training groups (one sport group and one music
group per association, four documents each) on top of 80 fixed background
documents. Each coalition is one full-parameter fine-tune of Qwen2-7B; the
outcome is measured on 80 held-out probes.

```text
coalitions.py            the frozen coalition design (88 coalitions + 8 noise replicates)
run.py                   one fine-tune per coalition, resumable
scoring.py               candidate scoring
build_dataset.py         deterministic reconstruction of the documents from the pinned name list
dataset/                 SOURCE_DATA_PROVENANCE.md (the documents themselves are not shipped)
cp_query_mode.py         CP datamodel with a query mode; loads the saved utilities
fitting_compute.py       Table 2 / Table 13: tuned baselines, single-fit CP, ensembles
per_query_interactions.py, verify_cp_claims.py   per-class interaction analysis
results/                 coalition_*/UTILITY.json (96 runs), DESIGN.json, FITTING_COMPUTE_*.json
```

## Refit the datamodels (CPU)

```bash
PYTHONPATH=. python experiments/sport_music/fitting_compute.py
```

This rewrites `results/FITTING_COMPUTE_*.json`. Refitting when this release was
prepared reproduced the tuned baselines exactly and the CP rows to within 0.008
in R² (single fit 0.660 against the saved 0.668).

## Regenerate the fine-tuning outcomes (GPU)

Needs an 80 GB GPU and a local snapshot of `Qwen/Qwen2-7B`
(revision `453ed1575b739b5b03ce3758b23befdb0967f40e`) in `TNDM_QWEN2_7B`.

The fine-tuning documents are **not included**: they are built from `names.csv`
of the `extractive-structures` repository, which carries no licence.
`build_dataset.py` rebuilds the document contents from that file (place it under
`dataset/source/extractive-structures/dsets/`). The tokenisation step that turns
the contents into `dataset/data/TOKENIZED_DOCUMENTS.json` and `PROBES.json`
belongs to an earlier stage of the original pipeline and is not part of this
release, so the fine-tuning runner cannot be executed from this repository
alone.

## Notes

- Results in the paper use the log-probability outcome. The same comparison on
  accuracy is in `results/FITTING_COMPUTE_accuracy.json`.
- The saved utilities are also released as `data/cubes/sportmusic_qwen2_7b_d10.npz`.
