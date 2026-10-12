#!/usr/bin/env python3
"""Build one deterministic AG News document manifest for eight coalitions.

Only the public AG News *train* split contributes permanent-base and optional
training documents.  The fixed probe is selected exclusively from the public
test split, so construction cannot accidentally leak a validation/test record
into a coalition.  Two source overlaps are explicit experimental treatments:
the duplicate group copies base documents, and the corrected group reuses the
same source documents as the label-corruption group with their true labels.
All other training source IDs are disjoint.

The manifest is an atomic JSON artifact under ``results/raw``.  It contains
the texts, labels, source IDs, group mapping, and construction checks required
for independent Slurm array tasks to reproduce the same inputs without each
inventing a different split.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from run_transformer_coalition import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RAW_ROOT,
    GROUP_DESCRIPTIONS,
    GROUP_NAMES,
    PipelineError,
    artifact_is_valid,
    data_contract,
    default_prepared_data_path,
    fingerprint,
    load_config,
    read_json_artifact,
    run_metadata,
    write_json_artifact,
)


def require_datasets() -> Any:
    """Import the optional data loader only after argument parsing succeeds."""

    try:
        from datasets import load_dataset
    except Exception as error:
        raise PipelineError(
            "AG News preparation requires the optional Hugging Face 'datasets' package. "
            "Install the project optional extra, for example: "
            "python -m pip install -e '.[transformer]'. "
            f"Original import error: {type(error).__name__}: {error}"
        ) from error
    return load_dataset


def _record(
    split_name: str,
    dataset: Any,
    index: int,
    *,
    group_name: str,
    label: int | None = None,
    mutation: str = "original",
    record_suffix: str = "",
) -> dict[str, Any]:
    row = dataset[int(index)]
    source_label = int(row["label"])
    final_label = source_label if label is None else int(label)
    source_id = f"{split_name}:{int(index)}"
    text = str(row["text"])
    if mutation == "whitespace_perturbed":
        words = text.split()
        text = (words[0] + "  " + " ".join(words[1:])) if len(words) > 1 else text + " "
    return {
        "record_id": f"{group_name}:{source_id}{record_suffix}",
        "source_id": source_id,
        "source_split": split_name,
        "text": text,
        "label": final_label,
        "source_label": source_label,
        "group_name": group_name,
        "mutation": mutation,
    }


def _indices_by_label(dataset: Any) -> dict[int, list[int]]:
    labels = dataset["label"]
    grouped: dict[int, list[int]] = {label: [] for label in range(4)}
    for index, label in enumerate(labels):
        label = int(label)
        if label not in grouped:
            raise PipelineError(f"AG News source has unexpected label {label}; expected 0..3.")
        grouped[label].append(index)
    if any(not values for values in grouped.values()):
        raise PipelineError("AG News source does not have all four expected labels.")
    return grouped


class TrainSelector:
    """Deterministically select disjoint train-split source rows, except when explicit."""

    def __init__(self, dataset: Any, rng: random.Random) -> None:
        self.dataset = dataset
        self.by_label = _indices_by_label(dataset)
        self.used: set[int] = set()
        for indices in self.by_label.values():
            rng.shuffle(indices)

    def _available(self, label: int) -> list[int]:
        return [index for index in self.by_label[label] if index not in self.used]

    def balanced(self, per_class: int, *, name: str) -> dict[int, list[int]]:
        if per_class < 1:
            raise PipelineError(f"{name} needs at least one document per requested class.")
        selected: dict[int, list[int]] = {}
        for label in range(4):
            available = self._available(label)
            if len(available) < per_class:
                raise PipelineError(
                    f"AG News train split has too few unused label-{label} documents for {name}: "
                    f"need {per_class}, have {len(available)}."
                )
            chosen = available[:per_class]
            self.used.update(chosen)
            selected[label] = chosen
        return selected

    def extreme_length(self, per_class: int, *, longest: bool, name: str) -> dict[int, list[int]]:
        selected: dict[int, list[int]] = {}
        for label in range(4):
            available = self._available(label)
            if len(available) < per_class:
                raise PipelineError(f"Insufficient unused label-{label} documents for {name}.")
            # Index breaks ties deterministically after the seed-established
            # source ordering has chosen which records are otherwise eligible.
            ordered = sorted(
                available,
                key=lambda index: (len(str(self.dataset[index]["text"]).split()), index),
                reverse=longest,
            )
            chosen = ordered[:per_class]
            self.used.update(chosen)
            selected[label] = chosen
        return selected

    def selected_labels(self, labels: Sequence[int], per_label: int, *, name: str) -> dict[int, list[int]]:
        selected: dict[int, list[int]] = {}
        for label in labels:
            available = self._available(int(label))
            if len(available) < per_label:
                raise PipelineError(f"Insufficient unused label-{label} documents for {name}.")
            chosen = available[:per_label]
            self.used.update(chosen)
            selected[int(label)] = chosen
        return selected


def _flatten(by_label: Mapping[int, Sequence[int]]) -> list[int]:
    return [index for label in sorted(by_label) for index in by_label[label]]


def _records_for_indices(
    split_name: str,
    dataset: Any,
    by_label: Mapping[int, Sequence[int]],
    *,
    group_name: str,
) -> list[dict[str, Any]]:
    return [_record(split_name, dataset, index, group_name=group_name) for index in _flatten(by_label)]


def _class_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {str(label): int(sum(int(record["label"]) == label for record in records)) for label in range(4)}


def _validate_manifest_parts(
    base: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    probe: Sequence[Mapping[str, Any]],
    *,
    base_per_class: int,
    group_size: int,
    probe_per_class: int,
) -> None:
    if _class_counts(base) != {str(label): base_per_class for label in range(4)}:
        raise PipelineError("Permanent base lost its required class balance during construction.")
    if _class_counts(probe) != {str(label): probe_per_class for label in range(4)}:
        raise PipelineError("Fixed test-split probe lost its required class balance during construction.")
    for index, group in enumerate(groups):
        records = group["records"]
        if len(records) != group_size:
            raise PipelineError(f"Group {index} has {len(records)} records, expected {group_size}.")
        if group["index"] != index or group["name"] != GROUP_NAMES[index]:
            raise PipelineError(f"Group mapping at index {index} is not canonical.")
    if not all(record["source_split"] == "train" for record in base):
        raise PipelineError("Base includes a non-train document.")
    if not all(record["source_split"] == "train" for group in groups for record in group["records"]):
        raise PipelineError("An optional group includes a non-train document.")
    if not all(record["source_split"] == "test" for record in probe):
        raise PipelineError("Probe includes a non-test document.")
    # Source IDs include their split name. This check makes the intended split
    # boundary explicit rather than relying on numerical row indices coinciding.
    training_sources = {record["source_id"] for record in base}
    training_sources.update(record["source_id"] for group in groups for record in group["records"])
    probe_sources = {record["source_id"] for record in probe}
    if training_sources & probe_sources:
        raise PipelineError("Training/probe source IDs overlap; refusing a leaky manifest.")
    base_sources = {record["source_id"] for record in base}
    duplicate_sources = {record["source_id"] for record in groups[1]["records"]}
    corrupted_sources = {record["source_id"] for record in groups[4]["records"]}
    corrected_sources = {record["source_id"] for record in groups[5]["records"]}
    if not duplicate_sources <= base_sources:
        raise PipelineError("The duplicate group must only copy permanent-base sources.")
    if corrected_sources != corrupted_sources:
        raise PipelineError("The corrected group must match the corruption-region source IDs exactly.")
    unique_groups = [base_sources, *({record["source_id"] for record in groups[i]["records"]} for i in (0, 2, 3, 4, 6, 7))]
    for left, sources in enumerate(unique_groups):
        for right, other in enumerate(unique_groups[:left]):
            if sources & other:
                raise PipelineError("Unexpected source overlap outside the declared duplicate/correction treatments.")


def build_manifest(config: Mapping[str, Any], smoke: bool, *, hf_cache: Path | None) -> dict[str, Any]:
    load_dataset = require_datasets()
    contract = data_contract(config, smoke)
    group_size = int(contract["optional_group_size"])
    if group_size < 4 or group_size % 4:
        raise PipelineError("data.optional_group_size must be a positive multiple of four.")
    base_per_class = int(contract["base_per_class"])
    probe_per_class = int(contract["probe_per_class"])
    if base_per_class < group_size // 4:
        raise PipelineError("Base must contain enough examples to form the declared duplicate group.")
    load_kwargs: dict[str, Any] = {}
    if hf_cache is not None:
        load_kwargs["cache_dir"] = str(hf_cache)
    try:
        dataset_name = str(contract["dataset_name"])
        dataset_config = contract["dataset_config"]
        dataset = (
            load_dataset(dataset_name, dataset_config, **load_kwargs)
            if dataset_config is not None
            else load_dataset(dataset_name, **load_kwargs)
        )
        train = dataset["train"]
        test = dataset["test"]
    except Exception as error:
        raise PipelineError(
            "Could not load the public AG News dataset. Ensure a usable shared cache or permitted network access. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error
    if "text" not in train.column_names or "label" not in train.column_names:
        raise PipelineError("AG News train split lacks expected text/label columns.")
    if "text" not in test.column_names or "label" not in test.column_names:
        raise PipelineError("AG News test split lacks expected text/label columns.")

    selector = TrainSelector(train, random.Random(int(contract["construction_seed"])))
    per_class = group_size // 4
    base_indices = selector.balanced(base_per_class, name="permanent base")
    clean_indices = selector.balanced(per_class, name="clean optional group")
    short_indices = selector.extreme_length(per_class, longest=False, name="short-document group")
    long_indices = selector.extreme_length(per_class, longest=True, name="long-document group")
    corrupted_indices = selector.balanced(per_class, name="label-corruption group")
    world_scitech_indices = selector.selected_labels((0, 3), group_size // 2, name="World/Sci-Tech enrichment")
    business_sports_indices = selector.selected_labels((1, 2), group_size // 2, name="Business/Sports enrichment")

    base = _records_for_indices("train", train, base_indices, group_name="permanent_clean_base")
    duplicate_records: list[dict[str, Any]] = []
    for label, indices in sorted(base_indices.items()):
        for position, index in enumerate(indices[:per_class]):
            mutation = "exact_duplicate" if (label + position) % 2 == 0 else "whitespace_perturbed"
            suffix = ":exact-copy" if mutation == "exact_duplicate" else ":whitespace-copy"
            duplicate_records.append(
                _record(
                    "train",
                    train,
                    index,
                    group_name=GROUP_NAMES[1],
                    mutation=mutation,
                    record_suffix=suffix,
                )
            )
    corrupted_records: list[dict[str, Any]] = []
    corrected_records: list[dict[str, Any]] = []
    for original_label, indices in sorted(corrupted_indices.items()):
        for index in indices:
            corrupted_records.append(
                _record(
                    "train",
                    train,
                    index,
                    group_name=GROUP_NAMES[4],
                    label=(original_label + 1) % 4,
                    mutation="label_corrupted_next_class",
                    record_suffix=":corrupted-label",
                )
            )
            corrected_records.append(
                _record(
                    "train",
                    train,
                    index,
                    group_name=GROUP_NAMES[5],
                    label=original_label,
                    mutation="correct_label_for_corruption_region",
                    record_suffix=":correct-label",
                )
            )
    groups: list[dict[str, Any]] = [
        {
            "index": 0,
            "name": GROUP_NAMES[0],
            "description": GROUP_DESCRIPTIONS[0],
            "records": _records_for_indices("train", train, clean_indices, group_name=GROUP_NAMES[0]),
        },
        {
            "index": 1,
            "name": GROUP_NAMES[1],
            "description": GROUP_DESCRIPTIONS[1],
            "records": duplicate_records,
        },
        {
            "index": 2,
            "name": GROUP_NAMES[2],
            "description": GROUP_DESCRIPTIONS[2],
            "records": _records_for_indices("train", train, short_indices, group_name=GROUP_NAMES[2]),
        },
        {
            "index": 3,
            "name": GROUP_NAMES[3],
            "description": GROUP_DESCRIPTIONS[3],
            "records": _records_for_indices("train", train, long_indices, group_name=GROUP_NAMES[3]),
        },
        {
            "index": 4,
            "name": GROUP_NAMES[4],
            "description": GROUP_DESCRIPTIONS[4],
            "records": corrupted_records,
        },
        {
            "index": 5,
            "name": GROUP_NAMES[5],
            "description": GROUP_DESCRIPTIONS[5],
            "records": corrected_records,
        },
        {
            "index": 6,
            "name": GROUP_NAMES[6],
            "description": GROUP_DESCRIPTIONS[6],
            "records": _records_for_indices(
                "train", train, world_scitech_indices, group_name=GROUP_NAMES[6]
            ),
        },
        {
            "index": 7,
            "name": GROUP_NAMES[7],
            "description": GROUP_DESCRIPTIONS[7],
            "records": _records_for_indices(
                "train", train, business_sports_indices, group_name=GROUP_NAMES[7]
            ),
        },
    ]
    test_by_label = _indices_by_label(test)
    test_rng = random.Random(int(contract["construction_seed"]) + 1)
    probe: list[dict[str, Any]] = []
    for label in range(4):
        candidates = list(test_by_label[label])
        test_rng.shuffle(candidates)
        if len(candidates) < probe_per_class:
            raise PipelineError(f"AG News test split has too few label-{label} records for the probe.")
        probe.extend(
            _record("test", test, index, group_name="fixed_test_probe")
            for index in candidates[:probe_per_class]
        )
    _validate_manifest_parts(
        base,
        groups,
        probe,
        base_per_class=base_per_class,
        group_size=group_size,
        probe_per_class=probe_per_class,
    )
    return {
        "schema_version": 1,
        "experiment": "ag-news-eight-document-groups",
        "data_contract": contract,
        "data_contract_sha256": fingerprint(contract),
        "source": {
            "dataset": contract["dataset_name"],
            "dataset_config": contract["dataset_config"],
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_fingerprint": str(getattr(train, "_fingerprint", "unknown")),
            "test_fingerprint": str(getattr(test, "_fingerprint", "unknown")),
            "training_split": "train",
            "probe_split": "test",
        },
        "group_mapping": {
            "n_groups": len(GROUP_NAMES),
            "bit_order": "least-significant bit is group index 0",
            "groups": [
                {"index": index, "name": name, "description": GROUP_DESCRIPTIONS[index]}
                for index, name in enumerate(GROUP_NAMES)
            ],
        },
        "intentional_source_overlaps": {
            GROUP_NAMES[1]: "copies selected permanent-base source documents only",
            GROUP_NAMES[5]: "same source documents as controlled_label_corruption, with original labels",
        },
        "base": base,
        "optional_groups": groups,
        "probe": probe,
        "construction_checks": {
            "base_label_counts": _class_counts(base),
            "probe_label_counts": _class_counts(probe),
            "group_sizes": [len(group["records"]) for group in groups],
            "train_test_source_overlap": False,
        },
        "provenance": run_metadata(Path(__file__).resolve().parent),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=None, help="Override the default prepared-manifest path")
    parser.add_argument("--hf-cache", type=Path, default=None, help="Optional shared Hugging Face cache directory")
    parser.add_argument("--smoke", action="store_true", help="Build the separately named tiny smoke manifest")
    parser.add_argument("--force", action="store_true", help="Rewrite a valid manifest when its construction contract changed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
        output_root = args.output_root.resolve()
        output = (args.output or default_prepared_data_path(output_root, args.smoke)).resolve()
        contract = data_contract(config, args.smoke)
        if artifact_is_valid(output) and not args.force:
            existing = read_json_artifact(output)
            if existing.get("data_contract") != contract:
                raise PipelineError(
                    f"Valid manifest {output} has a different construction contract. "
                    "Pass --force to replace it explicitly."
                )
            print(f"skipped valid prepared manifest path={output}", flush=True)
            return
        manifest = build_manifest(config, args.smoke, hf_cache=args.hf_cache.resolve() if args.hf_cache else None)
        write_json_artifact(output, manifest)
        print(
            f"written prepared manifest smoke={args.smoke} base={len(manifest['base'])} "
            f"groups={[len(group['records']) for group in manifest['optional_groups']]} "
            f"probe={len(manifest['probe'])} path={output}",
            flush=True,
        )
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
