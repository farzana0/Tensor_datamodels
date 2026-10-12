#!/usr/bin/env python3
"""Freeze E1 rank configs and a deterministic packed SLURM manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.ag_news.shared import (
    PREREG_SHA256,
    ROOT,
    atomic_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
    verify_frozen_preregistration,
)


RANKS = (1, 2, 4, 8, 16, 64)
SEEDS = (31, 43, 59)
BASE_CONFIG = ROOT / "configs/transformer.yaml"
PREPARED_DATA = ROOT / "results/raw/transformer/data/ag_news_prepared.json"
OUTPUT = ROOT / "results/data_dependency_attribution/E1_adapter_rank"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_frozen_preregistration()
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    if not PREPARED_DATA.is_file():
        raise FileNotFoundError(PREPARED_DATA)
    configs = args.output / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(parents=True, exist_ok=True)
    identities = []
    config_hashes = {}
    for rank in RANKS:
        config = json.loads(json.dumps(base))
        config["experiment"] = f"E1-ag-news-roberta-lora-rank-{rank}-d8"
        config["model"]["lora_rank"] = rank
        # Keep the effective LoRA multiplier alpha/r fixed at two.
        config["model"]["lora_alpha"] = 2 * rank
        config["preregistration"] = {
            "sha256": PREREG_SHA256,
            "experiment_id": "E1",
            "dimension": 8,
            "capacity": f"lora_rank_{rank}",
            "selection": "capacity is experimental condition; not validation-selected",
        }
        path = configs / f"lora_rank_{rank}.json"
        atomic_json(path, config)
        config_hashes[str(rank)] = sha256_file(path)
        for seed in SEEDS:
            for mask in range(256):
                identities.append(
                    {
                        "identity": len(identities),
                        "experiment_id": "E1",
                        "dimension": 8,
                        "capacity": "lora",
                        "rank": rank,
                        "alpha": 2 * rank,
                        "learner_seed": seed,
                        "mask": mask,
                        "config": display_path(path),
                        "output_root": display_path(args.output / f"lora_rank_{rank}"),
                    }
                )
    manifest = args.output / "oracle_manifest.jsonl"
    body = "".join(canonical_json(row) + "\n" for row in identities)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(body, encoding="utf-8")
    summary = {
        "experiment_id": "E1",
        "preregistration_sha256": PREREG_SHA256,
        "base_config": str(BASE_CONFIG.relative_to(ROOT)),
        "base_config_sha256": sha256_file(BASE_CONFIG),
        "prepared_data": str(PREPARED_DATA.relative_to(ROOT)),
        "prepared_data_sha256": sha256_file(PREPARED_DATA),
        "ranks": list(RANKS),
        "lora_alpha_rule": "alpha=2*rank; alpha/r=2",
        "learner_seeds": list(SEEDS),
        "dimension": 8,
        "oracle_calls": len(identities),
        "packed_four_gpu_array_tasks": len(identities) // 4,
        "recommended_array_chunks": [
            {"identity_offset": 0, "array": "0-575%8", "ranks": [1, 2, 4]},
            {"identity_offset": 2304, "array": "0-575%8", "ranks": [8, 16, 64]},
        ],
        "manifest": display_path(manifest),
        "manifest_sha256": sha256_bytes(body.encode("utf-8")),
        "config_sha256": config_hashes,
        "outcome_values_opened_during_preparation": False,
    }
    atomic_json(args.output / "preflight.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
