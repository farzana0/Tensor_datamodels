#!/usr/bin/env python3
"""Validate and aggregate complete E1 d=8 rank/seed coalition cubes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from experiments.ag_news.prepare_e1 import OUTPUT, RANKS, SEEDS
from experiments.ag_news.shared import (
    PREREG_SHA256,
    ROOT,
    atomic_json,
    epsilon_rank_profile,
    interaction_energy,
    sha256_file,
    verify_frozen_preregistration,
)


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_cube(
    root: Path,
    rank: int,
    seed: int,
    utility: np.ndarray,
    source_hashes: list[str],
    group_names: list[str] | None = None,
    group_sizes: list[int] | None = None,
    summed_per_coalition_elapsed_seconds: float | None = None,
) -> Path:
    path = root / "cubes" / f"lora_rank_{rank}_seed{seed}.npz"
    marker = path.with_suffix(".npz.complete")
    metadata = {
        "experiment_id": "E1",
        "preregistration_sha256": PREREG_SHA256,
        "dimension": 8,
        "adapter_rank": rank,
        "learner_seed": seed,
        "utility": "negative held-out probe NLL",
        "oracle_calls": len(utility),
        "summed_per_coalition_elapsed_seconds": summed_per_coalition_elapsed_seconds,
        "elapsed_seconds_semantics": "sum of elapsed_seconds stored by the independent one-GPU coalition jobs",
        "natural_order": list(range(8)),
        "group_names": list(group_names or []),
        "group_sizes": list(group_sizes or []),
        "corrupted_group_indices": [
            index for index, name in enumerate(group_names or []) if name == "controlled_label_corruption"
        ],
        "source_artifact_sha256": source_hashes,
    }
    if path.is_file() and marker.is_file():
        with np.load(path, allow_pickle=False) as payload:
            np.testing.assert_array_equal(payload["negative_loss"], utility)
        return path
    if path.exists() or marker.exists():
        raise RuntimeError(f"partial E1 aggregate requires audit: {path}")
    atomic_npz(path, negative_loss=utility, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    atomic_json(marker, {"sha256": sha256_file(path), "metadata": metadata})
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", type=Path, default=OUTPUT)
    return parser.parse_args()


def artifact_path(root: Path, rank: int, seed: int, mask: int) -> Path:
    return root / f"lora_rank_{rank}/restart/standard" / f"seed_{seed:02d}" / f"restart_mask_{mask:03d}.json"


def main() -> None:
    args = parse_args()
    verify_frozen_preregistration()
    cube_rows = []
    structural_rows = []
    missing = []
    for rank in RANKS:
        config = json.loads((args.input / "configs" / f"lora_rank_{rank}.json").read_text(encoding="utf-8"))
        group_names = [str(name) for name in config["data"]["group_names"]]
        group_sizes = [int(config["data"]["optional_group_size"])] * len(group_names)
        for seed in SEEDS:
            utility = np.empty(256, dtype=np.float64)
            hashes = []
            elapsed_seconds = []
            for mask in range(256):
                path = artifact_path(args.input, rank, seed, mask)
                marker = path.with_suffix(path.suffix + ".complete")
                if not path.is_file() or not marker.is_file():
                    missing.append(str(path))
                    continue
                completion = json.loads(marker.read_text(encoding="utf-8"))
                observed_sha256 = sha256_file(path)
                if completion.get("sha256") != observed_sha256:
                    raise RuntimeError(f"E1 oracle checksum mismatch: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if int(payload["coalition"]["mask"]) != mask or int(payload["seed"]) != seed:
                    raise RuntimeError(f"identity mismatch: {path}")
                if int(payload["model"]["lora_rank"]) != rank or int(payload["model"]["lora_alpha"]) != 2 * rank:
                    raise RuntimeError(f"capacity mismatch: {path}")
                utility[mask] = -float(payload["probe"]["metrics"]["negative_log_likelihood"])
                elapsed_seconds.append(float(payload["elapsed_seconds"]))
                hashes.append(observed_sha256)
            if missing:
                continue
            energy = interaction_energy(utility)
            profiles, ranks = epsilon_rank_profile(utility, tuple(range(8)))
            cube_path = write_cube(
                args.input,
                rank,
                seed,
                utility,
                hashes,
                group_names,
                group_sizes,
                sum(elapsed_seconds),
            )
            cube_sha256 = sha256_file(cube_path)
            for mask, value in enumerate(utility):
                cube_rows.append(
                    {
                        "experiment_id": "E1",
                        "dimension": 8,
                        "adapter_rank": rank,
                        "learner_seed": seed,
                        "mask": mask,
                        "utility": value,
                        "source_artifact": str(artifact_path(args.input, rank, seed, mask).relative_to(ROOT)),
                        "source_sha256": hashes[mask],
                    }
                )
            for profile in profiles:
                structural_rows.append(
                    {
                        "experiment_id": "E1",
                        "dimension": 8,
                        "adapter_rank": rank,
                        "learner_seed": seed,
                        "E_ge_2": energy["E_ge_2"],
                        "R_1e-2": ranks["R_1e-2"],
                        **{key: value for key, value in profile.items() if key != "singular_values"},
                        "singular_values_json": json.dumps(profile["singular_values"]),
                        "source_artifact": str(cube_path.relative_to(ROOT)),
                        "source_sha256": cube_sha256,
                    }
                )
    if missing:
        atomic_json(
            args.input / "aggregation_status.json",
            {
                "experiment_id": "E1",
                "complete": False,
                "missing_count": len(missing),
                "missing_first_100": missing[:100],
                "preregistration_sha256": PREREG_SHA256,
            },
        )
        raise RuntimeError(f"E1 incomplete: {len(missing)} missing artifacts")
    for name, rows in (("oracle_cube.csv", cube_rows), ("structural_metrics.csv", structural_rows)):
        path = args.input / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(
        args.input / "aggregation_status.json",
        {
            "experiment_id": "E1",
            "complete": True,
            "oracle_rows": len(cube_rows),
            "structural_rows": len(structural_rows),
            "preregistration_sha256": PREREG_SHA256,
            "oracle_cube_sha256": sha256_file(args.input / "oracle_cube.csv"),
            "structural_metrics_sha256": sha256_file(args.input / "structural_metrics.csv"),
        },
    )


if __name__ == "__main__":
    main()
