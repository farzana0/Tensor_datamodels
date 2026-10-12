#!/usr/bin/env python3
"""Run one resumable RoBERTa/LoRA restart-from-base coalition adaptation.

The bit convention is deliberately fixed: bit ``i`` of ``--mask`` includes
the optional document group at index ``i``.  The permanent clean base is
always retained.  Every restart reloads the same atomically saved trainable
LoRA/classifier state; it never starts from a previous fine-tuned endpoint.

This script intentionally keeps the transformer stack optional.  ``--help``
and mask inspection work in a lightweight environment, while a real run
fails with an actionable message until torch, transformers, and peft are
installed.  Use ``scripts/prepare_ag_news.py`` first to create the immutable,
checksummed data manifest used by all array tasks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parent  # experiments/ag_news
sys.path.insert(0, str(REPOSITORY.parents[1] / "src"))


GROUP_NAMES: tuple[str, ...] = (
    "clean_balanced",
    "duplicate_or_lightly_perturbed",
    "short_documents",
    "long_documents",
    "controlled_label_corruption",
    "correctly_labelled_corruption_region",
    "world_scitech_enrichment",
    "business_sports_enrichment",
)

GROUP_DESCRIPTIONS: tuple[str, ...] = (
    "clean class-balanced training documents",
    "intentional copies or whitespace-perturbed copies of clean base documents",
    "short training documents",
    "long training documents",
    "training documents with a deterministic incorrect label",
    "correctly labelled versions of the corrupted-region source documents",
    "World and Sci/Tech enrichment",
    "Business and Sports enrichment",
)

FULL_MASK = (1 << len(GROUP_NAMES)) - 1
DEFAULT_CONFIG_PATH = REPOSITORY / "configs" / "transformer.yaml"
DEFAULT_RAW_ROOT = REPOSITORY / "results" / "raw" / "transformer"


class PipelineError(RuntimeError):
    """An expected setup or artifact error that should be shown without a traceback."""


def _json_default(value: Any) -> Any:
    """Serialize the standard-library values used by this standalone pipeline."""

    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Match the repository's raw-artifact JSON/marker protocol without importing NumPy."""

    return json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def git_revision(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted-or-unborn"


def run_metadata(cwd: Path) -> dict[str, Any]:
    """Record non-sensitive execution provenance even in a minimal Python environment."""

    scheduler_keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_PARTITION",
        "CUDA_VISIBLE_DEVICES",
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(cwd),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "scheduler": {key: os.environ[key] for key in scheduler_keys if key in os.environ},
    }


@dataclass(frozen=True)
class TransformerRuntime:
    """Delayed imports, so CLI diagnostics do not require optional packages."""

    torch: Any
    AutoModelForSequenceClassification: Any
    AutoTokenizer: Any
    LoraConfig: Any
    TaskType: Any
    get_peft_model: Any
    DataLoader: Any
    TensorDataset: Any
    package_versions: Mapping[str, str]


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def fingerprint(payload: Any) -> str:
    """Return a stable configuration/data contract fingerprint."""

    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    """Load JSON-compatible YAML without making PyYAML a runtime requirement.

    The repository default is JSON syntax, which is valid YAML.  Users who
    prefer ordinary YAML can use it when PyYAML is installed; otherwise the
    error identifies the only missing parser dependency.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PipelineError(f"Cannot read transformer config {path}: {error}") from error
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as error:
            raise PipelineError(
                f"{path} is non-JSON YAML, but PyYAML is unavailable. "
                "Install PyYAML or use JSON-compatible YAML."
            ) from error
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise PipelineError(f"Transformer config {path} must contain a mapping.")
    for required in ("model", "data", "training", "unlearning"):
        if not isinstance(loaded.get(required), dict):
            raise PipelineError(f"Transformer config {path} is missing mapping {required!r}.")
    return loaded


def effective_section(config: Mapping[str, Any], section: str, smoke: bool) -> dict[str, Any]:
    """Merge a top-level config section with the deliberately bounded smoke override."""

    base = dict(config[section])
    if smoke:
        smoke_section = config.get("smoke", {})
        if not isinstance(smoke_section, Mapping):
            raise PipelineError("Config field 'smoke' must be a mapping when present.")
        override = smoke_section.get(section, {})
        if not isinstance(override, Mapping):
            raise PipelineError(f"Config field 'smoke.{section}' must be a mapping.")
        base.update(override)
    return base


def data_contract(config: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    """Fields that determine selected documents and bounded-run execution.

    ``max_*_documents`` values do not alter the immutable source manifest, but
    they do alter which manifest records reach an intervention.  When present
    (currently only in the smoke configuration), they therefore belong in the
    resumability contract.  Omit absent values to preserve the established
    standard-run contract and its in-flight artifacts.
    """

    data = effective_section(config, "data", smoke)
    names = tuple(data.get("group_names", ()))
    if names != GROUP_NAMES:
        raise PipelineError(
            "The config must retain the fixed eight-group ordering: " + ", ".join(GROUP_NAMES)
        )
    contract: dict[str, Any] = {
        "dataset_name": data.get("dataset_name"),
        "dataset_config": data.get("dataset_config"),
        "construction_seed": int(data.get("construction_seed")),
        "base_per_class": int(data.get("base_per_class")),
        "optional_group_size": int(data.get("optional_group_size")),
        "probe_per_class": int(data.get("probe_per_class")),
        "group_names": list(GROUP_NAMES),
        "smoke": bool(smoke),
    }
    for key in ("max_train_documents", "max_forget_documents", "max_probe_documents"):
        value = data.get(key)
        if value is not None:
            if int(value) < 1:
                raise PipelineError(f"data.{key} must be positive when supplied.")
            contract[key] = int(value)
    return contract


def architecture_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fields whose mismatch would make a saved adapter/head state unsafe to load."""

    model = effective_section(config, "model", smoke=False)
    required = (
        "name",
        "num_labels",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "lora_modules_to_save",
    )
    missing = [key for key in required if key not in model]
    if missing:
        raise PipelineError(f"Transformer model config is missing {missing}.")
    return {
        "name": str(model["name"]),
        "revision": model.get("revision"),
        "num_labels": int(model["num_labels"]),
        "lora_rank": int(model["lora_rank"]),
        "lora_alpha": int(model["lora_alpha"]),
        "lora_dropout": float(model["lora_dropout"]),
        "lora_target_modules": list(model["lora_target_modules"]),
        "lora_modules_to_save": list(model["lora_modules_to_save"]),
    }


def validate_mask(mask: int) -> int:
    if not 0 <= int(mask) <= FULL_MASK:
        raise PipelineError(f"mask must be in [0, {FULL_MASK}], received {mask}.")
    return int(mask)


def mask_from_argument(value: int | None) -> int:
    """Use an explicit mask or the array task ID, never an implicit random mapping."""

    if value is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise PipelineError("Pass --mask 0..255, or run inside a Slurm array task.")
        try:
            value = int(raw)
        except ValueError as error:
            raise PipelineError(f"SLURM_ARRAY_TASK_ID={raw!r} is not an integer mask.") from error
    return validate_mask(value)


def mask_description(mask: int) -> dict[str, Any]:
    mask = validate_mask(mask)
    bits = [int(bool(mask & (1 << index))) for index in range(len(GROUP_NAMES))]
    return {
        "mask": mask,
        "bit_order": "least-significant bit is group index 0",
        "bits": bits,
        "included_groups": [name for index, name in enumerate(GROUP_NAMES) if bits[index]],
        "excluded_groups": [name for index, name in enumerate(GROUP_NAMES) if not bits[index]],
    }


def require_transformer_runtime() -> TransformerRuntime:
    """Import only the packages actually needed for an adaptation run."""

    try:
        import peft
        import torch
        import transformers
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as error:  # Import errors can include an unavailable CUDA shared library.
        raise PipelineError(
            "Transformer execution requires torch, transformers, and peft. "
            "Install the project optional extra, for example: "
            "python -m pip install -e '.[transformer]'. "
            f"Original import error: {type(error).__name__}: {error}"
        ) from error
    return TransformerRuntime(
        torch=torch,
        AutoModelForSequenceClassification=AutoModelForSequenceClassification,
        AutoTokenizer=AutoTokenizer,
        LoraConfig=LoraConfig,
        TaskType=TaskType,
        get_peft_model=get_peft_model,
        DataLoader=DataLoader,
        TensorDataset=TensorDataset,
        package_versions={
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "peft": str(peft.__version__),
        },
    )


def set_determinism(runtime: TransformerRuntime, seed: int) -> None:
    """Seed the trainable path and reject nondeterministic learner kernels.

    Memory-efficient/flash scaled-dot-product attention can select a
    nondeterministic backward kernel on newer GPUs.  The data-dependency study
    compares coalition tensors across adaptation capacities, so even small
    kernel-level variation would be an oracle confound.  Force the math SDPA
    backend before enabling strict deterministic-algorithm checks.
    """

    random.seed(seed)
    runtime.torch.manual_seed(seed)
    if runtime.torch.cuda.is_available():
        runtime.torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    runtime.torch.backends.cudnn.deterministic = True
    runtime.torch.backends.cudnn.benchmark = False
    cuda_backend = getattr(runtime.torch.backends, "cuda", None)
    if cuda_backend is not None:
        for name, value in (
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_math_sdp", True),
        ):
            function = getattr(cuda_backend, name, None)
            if function is not None:
                function(value)
    try:
        runtime.torch.use_deterministic_algorithms(True, warn_only=False)
    except (AttributeError, TypeError):
        pass


def resolve_device(runtime: TransformerRuntime, requested: str) -> Any:
    if requested == "auto":
        return runtime.torch.device("cuda" if runtime.torch.cuda.is_available() else "cpu")
    try:
        device = runtime.torch.device(requested)
    except Exception as error:
        raise PipelineError(f"Invalid --device {requested!r}: {error}") from error
    if device.type == "cuda" and not runtime.torch.cuda.is_available():
        raise PipelineError("--device requests CUDA, but torch.cuda.is_available() is false.")
    return device


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically write a JSON payload and its SHA-256 completion marker.

    This intentionally follows ``datamodels.utils.artifacts``'s marker layout,
    but stays importable before the project's NumPy base dependency is present.
    """

    body = canonical_json_bytes(payload)
    digest = hashlib.sha256(body).hexdigest()
    _atomic_bytes(path, body)
    marker = {
        "artifact": path.name,
        "sha256": digest,
        "bytes": len(body),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_bytes(path.with_suffix(path.suffix + ".complete"), canonical_json_bytes(marker))
    return digest


def read_json_artifact(path: Path) -> dict[str, Any]:
    """Read only completed, checksum-valid raw JSON artifacts."""

    marker_path = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker_path.is_file():
        raise ValueError(f"Incomplete artifact: {path}")
    try:
        body = path.read_bytes()
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        payload = json.loads(body.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable artifact: {path}") from error
    if marker.get("sha256") != hashlib.sha256(body).hexdigest():
        raise ValueError(f"Digest mismatch for artifact: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact payload is not a mapping: {path}")
    return payload


def artifact_is_valid(path: Path) -> bool:
    try:
        read_json_artifact(path)
    except ValueError:
        return False
    return True


def _atomic_torch_save(runtime: TransformerRuntime, path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        runtime.torch.save(dict(payload), temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_marker_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".complete")


def _torch_load(runtime: TransformerRuntime, path: Path) -> dict[str, Any]:
    try:
        value = runtime.torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch before the weights_only argument.
        value = runtime.torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise PipelineError(f"State file {path} does not contain a mapping payload.")
    return value


def read_trainable_state(runtime: TransformerRuntime, path: Path) -> tuple[dict[str, Any], str] | None:
    """Return a verified state payload and digest, or ``None`` when incomplete/corrupt."""

    marker_path = state_marker_path(path)
    if not path.is_file() or not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        actual = file_sha256(path)
        if marker.get("state_file") != path.name or marker.get("sha256") != actual:
            return None
        payload = _torch_load(runtime, path)
        if not isinstance(payload.get("state_dict"), dict) or not isinstance(payload.get("metadata"), dict):
            return None
    except Exception:
        # A partial/foreign/corrupt checkpoint is deliberately treated as not
        # resumable.  The caller will either create a new initial state under
        # its lock or issue an actionable full-endpoint error.
        return None
    return payload, actual


def write_trainable_state(
    runtime: TransformerRuntime,
    path: Path,
    *,
    kind: str,
    state_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    """Atomically write a small trainable-state checkpoint plus a checksum marker."""

    payload = {
        "schema_version": 1,
        "kind": kind,
        "metadata": dict(metadata),
        "state_dict": dict(state_dict),
    }
    _atomic_torch_save(runtime, path, payload)
    digest = file_sha256(path)
    marker = {
        "state_file": path.name,
        "sha256": digest,
        "completed_at_utc": run_metadata(REPOSITORY)["timestamp_utc"],
        "kind": kind,
    }
    _atomic_bytes(state_marker_path(path), _canonical_json(marker) + b"\n")
    return digest


@contextlib.contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize first-time initialization creation across independent array tasks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ModuleNotFoundError:  # pragma: no cover - Slurm/Linux always has fcntl.
        yield
        return
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def trainable_state_dict(model: Any) -> dict[str, Any]:
    """Return only LoRA and classifier parameters; frozen RoBERTa is reloaded by name."""

    return {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(runtime: TransformerRuntime, model: Any, state_dict: Mapping[str, Any], path: Path) -> None:
    expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    saved_names = set(state_dict)
    if saved_names != set(expected):
        missing = sorted(set(expected) - saved_names)
        unexpected = sorted(saved_names - set(expected))
        raise PipelineError(
            f"Trainable state {path} does not match the requested LoRA/head architecture; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}."
        )
    with runtime.torch.no_grad():
        for name, parameter in expected.items():
            saved = state_dict[name]
            if tuple(saved.shape) != tuple(parameter.shape):
                raise PipelineError(
                    f"Trainable state {path} has shape {tuple(saved.shape)} for {name}, "
                    f"expected {tuple(parameter.shape)}."
                )
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))


def create_model(
    runtime: TransformerRuntime,
    config: Mapping[str, Any],
    *,
    initialization_seed: int,
    device: Any,
    hf_cache: Path | None,
) -> Any:
    """Construct the frozen base plus trainable rank-8 LoRA/head wrapper."""

    model_config = effective_section(config, "model", smoke=False)
    architecture = architecture_contract(config)
    set_determinism(runtime, initialization_seed)
    load_kwargs: dict[str, Any] = {
        "num_labels": architecture["num_labels"],
        "ignore_mismatched_sizes": True,
    }
    if architecture["revision"] is not None:
        load_kwargs["revision"] = architecture["revision"]
    if hf_cache is not None:
        load_kwargs["cache_dir"] = str(hf_cache)
    try:
        base_model = runtime.AutoModelForSequenceClassification.from_pretrained(
            architecture["name"], **load_kwargs
        )
    except Exception as error:
        raise PipelineError(
            f"Could not load {architecture['name']!r}. Ensure the model is cached or network access is available. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error
    lora_config = runtime.LoraConfig(
        r=architecture["lora_rank"],
        lora_alpha=architecture["lora_alpha"],
        lora_dropout=architecture["lora_dropout"],
        bias="none",
        target_modules=architecture["lora_target_modules"],
        modules_to_save=architecture["lora_modules_to_save"],
        task_type=runtime.TaskType.SEQ_CLS,
    )
    model = runtime.get_peft_model(base_model, lora_config)
    model.to(device)
    if not trainable_state_dict(model):
        raise PipelineError("PEFT produced no trainable LoRA/classifier parameters.")
    # Preserve this tiny config in model records even when an older PEFT
    # version normalizes its internal representation differently.
    model._queryable_datamodels_model_config = dict(model_config)  # type: ignore[attr-defined]
    return model


def initialization_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    training = effective_section(config, "training", smoke=False)
    return {
        "architecture": architecture_contract(config),
        "initialization_seed": int(training["initialization_seed"]),
    }


def default_initial_state_path(output_root: Path, config: Mapping[str, Any]) -> Path:
    contract = initialization_contract(config)
    architecture = contract["architecture"]
    model_name = str(architecture["name"]).replace("/", "__")
    return (
        output_root
        / "initialization"
        / f"initialization_seed_{contract['initialization_seed']:04d}"
        / f"{model_name}_lora_r{architecture['lora_rank']}.pt"
    )


def ensure_initial_state(
    runtime: TransformerRuntime,
    config: Mapping[str, Any],
    *,
    path: Path,
    device: Any,
    hf_cache: Path | None,
) -> tuple[dict[str, Any], str]:
    """Load or create one shared, checksummed adapter/head initialization."""

    expected_contract = initialization_contract(config)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with exclusive_file_lock(lock_path):
        existing = read_trainable_state(runtime, path)
        if existing is not None:
            payload, digest = existing
            if payload.get("kind") != "initial_lora_and_classifier":
                raise PipelineError(f"{path} is not an initial LoRA/classifier state.")
            if payload["metadata"].get("initialization_contract") != expected_contract:
                raise PipelineError(
                    f"Existing initialization {path} has a different model/seed contract. "
                    "Choose a new --initial-state path rather than mixing experiments."
                )
            return payload, digest

        model = create_model(
            runtime,
            config,
            initialization_seed=expected_contract["initialization_seed"],
            device=device,
            hf_cache=hf_cache,
        )
        metadata = {
            "initialization_contract": expected_contract,
            "trainable_parameter_names": sorted(trainable_state_dict(model)),
            "package_versions": dict(runtime.package_versions),
            "provenance": run_metadata(REPOSITORY),
        }
        digest = write_trainable_state(
            runtime,
            path,
            kind="initial_lora_and_classifier",
            state_dict=trainable_state_dict(model),
            metadata=metadata,
        )
        payload = read_trainable_state(runtime, path)
        if payload is None:  # Defensive: a filesystem error must not look like a valid restart point.
            raise PipelineError(f"Initialization state write did not validate: {path}")
        return payload[0], digest


def default_prepared_data_path(output_root: Path, smoke: bool) -> Path:
    suffix = "ag_news_prepared_smoke.json" if smoke else "ag_news_prepared.json"
    return output_root / "data" / suffix


def load_prepared_manifest(path: Path, config: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    if not path.exists():
        smoke_flag = " --smoke" if smoke else ""
        raise PipelineError(
            f"Prepared AG News manifest is absent: {path}. Run "
            f"python scripts/prepare_ag_news.py{smoke_flag} first."
        )
    try:
        manifest = read_json_artifact(path)
    except ValueError as error:
        raise PipelineError(f"Prepared AG News manifest is incomplete or corrupt: {path}") from error
    if manifest.get("schema_version") != 1 or manifest.get("experiment") != "ag-news-eight-document-groups":
        raise PipelineError(f"Prepared manifest {path} has an unsupported schema/experiment.")
    expected_contract = data_contract(config, smoke)
    if manifest.get("data_contract") != expected_contract:
        raise PipelineError(
            f"Prepared manifest {path} does not match this config/smoke mode. "
            "Regenerate it with scripts/prepare_ag_news.py."
        )
    groups = manifest.get("optional_groups")
    if not isinstance(groups, list) or len(groups) != len(GROUP_NAMES):
        raise PipelineError(f"Prepared manifest {path} does not contain exactly eight optional groups.")
    for index, group in enumerate(groups):
        if not isinstance(group, dict) or group.get("index") != index or group.get("name") != GROUP_NAMES[index]:
            raise PipelineError(f"Prepared manifest {path} has a non-deterministic group mapping at index {index}.")
        if not isinstance(group.get("records"), list):
            raise PipelineError(f"Prepared manifest {path} group {index} lacks records.")
    if not isinstance(manifest.get("base"), list) or not isinstance(manifest.get("probe"), list):
        raise PipelineError(f"Prepared manifest {path} lacks base or probe records.")
    return manifest


def records_for_mask(manifest: Mapping[str, Any], mask: int) -> list[dict[str, Any]]:
    """Return the permanent base plus optional groups selected by a canonical mask."""

    selected = list(manifest["base"])
    for index, group in enumerate(manifest["optional_groups"]):
        if mask & (1 << index):
            selected.extend(group["records"])
    return selected


def forgotten_records_for_mask(manifest: Mapping[str, Any], mask: int) -> list[dict[str, Any]]:
    """The approximate-unlearning forget set includes only omitted optional groups."""

    forgotten: list[dict[str, Any]] = []
    for index, group in enumerate(manifest["optional_groups"]):
        if not mask & (1 << index):
            forgotten.extend(group["records"])
    return forgotten


def stratified_limit(records: Sequence[dict[str, Any]], maximum: int | None) -> list[dict[str, Any]]:
    """Cap smoke inputs without accidentally evaluating a single AG News class."""

    if maximum is None or maximum <= 0 or len(records) <= maximum:
        return list(records)
    buckets: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(int(record["label"]), []).append(record)
    selected: list[dict[str, Any]] = []
    positions = {label: 0 for label in buckets}
    labels = sorted(buckets)
    while len(selected) < maximum:
        progressed = False
        for label in labels:
            position = positions[label]
            if position < len(buckets[label]) and len(selected) < maximum:
                selected.append(buckets[label][position])
                positions[label] += 1
                progressed = True
        if not progressed:
            break
    return selected


def make_loader(
    runtime: TransformerRuntime,
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    *,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: Any,
) -> Any:
    if not records:
        raise PipelineError("Cannot create a dataloader for an empty document collection.")
    encoded = tokenizer(
        [str(record["text"]) for record in records],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    attention = encoded.get("attention_mask")
    if attention is None:
        attention = runtime.torch.ones_like(encoded["input_ids"])
    labels = runtime.torch.tensor([int(record["label"]) for record in records], dtype=runtime.torch.long)
    dataset = runtime.TensorDataset(encoded["input_ids"], attention, labels)
    kwargs: dict[str, Any] = {
        "batch_size": min(int(batch_size), len(dataset)),
        "shuffle": bool(shuffle),
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    if shuffle:
        generator = runtime.torch.Generator()
        generator.manual_seed(int(seed))
        kwargs["generator"] = generator
    return runtime.DataLoader(dataset, **kwargs)


def _next_batch(iterator: Iterator[Any], loader: Any) -> tuple[Any, Iterator[Any]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_restart(
    runtime: TransformerRuntime,
    model: Any,
    loader: Any,
    *,
    training: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    """Perform exactly the configured number of optimizer steps for every coalition."""

    updates = int(training["max_updates"])
    if updates < 1:
        raise PipelineError("training.max_updates must be positive.")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = runtime.torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    model.train()
    iterator = iter(loader)
    losses: list[float] = []
    for _ in range(updates):
        (input_ids, attention_mask, labels), iterator = _next_batch(iterator, loader)
        output = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=labels.to(device),
        )
        loss = output.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        runtime.torch.nn.utils.clip_grad_norm_(parameters, float(training["max_grad_norm"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "optimizer": "AdamW",
        "updates": updates,
        "mean_train_loss": float(sum(losses) / len(losses)),
        "last_train_loss": float(losses[-1]),
    }


def evaluate_probe(
    runtime: TransformerRuntime,
    model: Any,
    loader: Any,
    records: Sequence[dict[str, Any]],
    *,
    device: Any,
) -> dict[str, Any]:
    """Evaluate and retain one logit vector for every fixed probe document."""

    model.eval()
    all_logits: list[list[float]] = []
    all_labels: list[int] = []
    with runtime.torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
            ).logits.detach().cpu()
            all_logits.extend(logits.tolist())
            all_labels.extend(int(value) for value in labels.tolist())
    logits_tensor = runtime.torch.tensor(all_logits, dtype=runtime.torch.float32)
    labels_tensor = runtime.torch.tensor(all_labels, dtype=runtime.torch.long)
    predictions = logits_tensor.argmax(dim=1)
    return {
        "document_ids": [str(record["record_id"]) for record in records],
        "source_document_ids": [str(record["source_id"]) for record in records],
        "labels": all_labels,
        "logits": all_logits,
        "metrics": {
            "accuracy": float((predictions == labels_tensor).float().mean().item()),
            "negative_log_likelihood": float(
                runtime.torch.nn.functional.cross_entropy(logits_tensor, labels_tensor).item()
            ),
            "num_documents": len(all_labels),
        },
    }


def default_full_model_state_path(output_root: Path, seed: int, smoke: bool = False) -> Path:
    """Keep tiny smoke endpoints in a separate namespace from full experiments."""

    tier = "smoke" if smoke else "standard"
    return output_root / "full_model" / tier / f"seed_{seed:02d}" / f"restart_mask_{FULL_MASK:03d}.pt"


def restart_artifact_path(output_root: Path, seed: int, mask: int, smoke: bool = False) -> Path:
    """Keep smoke logits separate from standard-run logits with the same mask."""

    tier = "smoke" if smoke else "standard"
    return output_root / "restart" / tier / f"seed_{seed:02d}" / f"restart_mask_{mask:03d}.json"


def state_metadata_matches(
    payload: Mapping[str, Any],
    *,
    kind: str,
    architecture: Mapping[str, Any],
    initialization_sha256: str | None = None,
    seed: int | None = None,
    mask: int | None = None,
) -> bool:
    metadata = payload.get("metadata")
    if payload.get("kind") != kind or not isinstance(metadata, Mapping):
        return False
    if metadata.get("architecture") != dict(architecture):
        return False
    if initialization_sha256 is not None and metadata.get("initialization_sha256") != initialization_sha256:
        return False
    if seed is not None and metadata.get("seed") != seed:
        return False
    if mask is not None and metadata.get("mask") != mask:
        return False
    return True


def restart_artifact_matches(
    path: Path,
    *,
    mask: int,
    seed: int,
    smoke: bool,
    contract: Mapping[str, Any],
    manifest_sha256: str,
    architecture: Mapping[str, Any],
    training: Mapping[str, Any],
    max_length: int,
    initialization: Mapping[str, Any],
    initialization_sha256: str,
) -> bool:
    """Check that a completed JSON result belongs to this exact rerun contract."""

    try:
        record = read_json_artifact(path)
    except ValueError:
        return False
    coalition = record.get("coalition")
    stored_initialization = record.get("initialization")
    return bool(
        record.get("intervention") == "restart_from_base_lora"
        and record.get("seed") == seed
        and record.get("smoke") == bool(smoke)
        and isinstance(coalition, Mapping)
        and coalition.get("mask") == mask
        and record.get("data_contract") == dict(contract)
        and record.get("data_manifest_sha256") == manifest_sha256
        and record.get("model") == dict(architecture)
        and record.get("training") == dict(training)
        and record.get("effective_max_length") == max_length
        and isinstance(stored_initialization, Mapping)
        and stored_initialization.get("contract") == dict(initialization)
        # A matching architecture/seed contract is necessary but insufficient:
        # the saved trainable initialization is the actual common restart
        # point.  Do not silently reuse an output after that checkpoint has
        # been regenerated or replaced under the same nominal config.
        and stored_initialization.get("sha256") == initialization_sha256
    )


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.print_mask_map:
        mapping = {str(mask): mask_description(mask) for mask in range(FULL_MASK + 1)}
        print(json.dumps(mapping, indent=2, sort_keys=True))
        return
    mask = mask_from_argument(args.mask)
    if args.seed < 0:
        raise PipelineError("--seed must be non-negative.")
    output_root = args.output_root.resolve()
    prepared_data = args.prepared_data or default_prepared_data_path(output_root, args.smoke)
    prepared_data = prepared_data.resolve()
    manifest = load_prepared_manifest(prepared_data, config, args.smoke)
    runtime = require_transformer_runtime()
    device = resolve_device(runtime, args.device)
    hf_cache = args.hf_cache.resolve() if args.hf_cache is not None else None
    training = effective_section(config, "training", args.smoke)
    model_settings = effective_section(config, "model", args.smoke)
    data_settings = effective_section(config, "data", args.smoke)
    architecture = architecture_contract(config)
    expected_data_contract = data_contract(config, args.smoke)
    expected_initialization = initialization_contract(config)
    manifest_digest = file_sha256(prepared_data)
    result_path = restart_artifact_path(output_root, args.seed, mask, args.smoke)
    full_state_path = args.full_model_state or default_full_model_state_path(output_root, args.seed, args.smoke)
    full_state_path = full_state_path.resolve()
    initial_state_path = args.initial_state or default_initial_state_path(output_root, config)
    initial_state_path = initial_state_path.resolve()

    # Resolve the actual shared restart point before deciding whether a raw
    # result is resumable.  The JSON contract records both the nominal
    # initialization contract and this content digest; comparing both prevents
    # a valid-looking stale coalition result from being reused after a
    # checkpoint replacement.
    initial_payload, initial_digest = ensure_initial_state(
        runtime,
        config,
        path=initial_state_path,
        device=device,
        hf_cache=hf_cache,
    )

    # A full-mask JSON artifact alone is insufficient for unlearning: retain its
    # trainable endpoint checkpoint as a separately checksummed raw artifact.
    if artifact_is_valid(result_path) and not args.force:
        if not restart_artifact_matches(
            result_path,
            mask=mask,
            seed=args.seed,
            smoke=args.smoke,
            contract=expected_data_contract,
            manifest_sha256=manifest_digest,
            architecture=architecture,
            training=training,
            max_length=int(model_settings["max_length"]),
            initialization=expected_initialization,
            initialization_sha256=initial_digest,
        ):
            raise PipelineError(
                f"Completed artifact {result_path} belongs to a different data/model/training contract. "
                "Choose another --output-root or pass --force to replace it explicitly."
            )
        full_state_valid = read_trainable_state(runtime, full_state_path)
        if mask != FULL_MASK:
            print(f"skipped valid restart artifact mask={mask} path={result_path}", flush=True)
            return
        if (
            full_state_valid is not None
            and state_metadata_matches(
                full_state_valid[0],
                kind="full_restart_lora_and_classifier",
                architecture=architecture,
                initialization_sha256=initial_digest,
                seed=args.seed,
                mask=FULL_MASK,
            )
            and full_state_valid[0]["metadata"].get("data_contract_sha256") == fingerprint(expected_data_contract)
        ):
            print(f"skipped valid full restart artifact and endpoint mask={mask} path={result_path}", flush=True)
            return
        print("full restart JSON is valid but its endpoint checkpoint is missing/incompatible; rebuilding it", flush=True)

    model = create_model(
        runtime,
        config,
        initialization_seed=int(training["initialization_seed"]),
        device=device,
        hf_cache=hf_cache,
    )
    load_trainable_state(runtime, model, initial_payload["state_dict"], initial_state_path)
    # Model construction used the initialization seed; the stochastic training
    # path uses the user-selected run seed only after the exact state reload.
    set_determinism(runtime, args.seed)

    all_train_records = records_for_mask(manifest, mask)
    train_cap = data_settings.get("max_train_documents")
    train_records = stratified_limit(all_train_records, int(train_cap) if train_cap is not None else None)
    all_probe_records = list(manifest["probe"])
    probe_cap = data_settings.get("max_probe_documents")
    probe_records = stratified_limit(all_probe_records, int(probe_cap) if probe_cap is not None else None)
    try:
        tokenizer_kwargs: dict[str, Any] = {}
        if architecture["revision"] is not None:
            tokenizer_kwargs["revision"] = architecture["revision"]
        if hf_cache is not None:
            tokenizer_kwargs["cache_dir"] = str(hf_cache)
        tokenizer = runtime.AutoTokenizer.from_pretrained(architecture["name"], **tokenizer_kwargs)
    except Exception as error:
        raise PipelineError(
            f"Could not load tokenizer for {architecture['name']!r}. Ensure it is cached or downloadable. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error
    train_loader = make_loader(
        runtime,
        tokenizer,
        train_records,
        max_length=int(model_settings["max_length"]),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=args.seed * 1000 + mask,
        device=device,
    )
    probe_loader = make_loader(
        runtime,
        tokenizer,
        probe_records,
        max_length=int(model_settings["max_length"]),
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
        seed=0,
        device=device,
    )
    started = time.perf_counter()
    diagnostics = train_restart(runtime, model, train_loader, training=training, device=device)
    probe = evaluate_probe(runtime, model, probe_loader, probe_records, device=device)
    elapsed = time.perf_counter() - started

    full_state_digest: str | None = None
    if mask == FULL_MASK:
        full_state_digest = write_trainable_state(
            runtime,
            full_state_path,
            kind="full_restart_lora_and_classifier",
            state_dict=trainable_state_dict(model),
            metadata={
                "architecture": architecture,
                "initialization_sha256": initial_digest,
                "seed": args.seed,
                "mask": FULL_MASK,
                "training": dict(training),
                "data_contract_sha256": fingerprint(expected_data_contract),
                "package_versions": dict(runtime.package_versions),
                "provenance": run_metadata(REPOSITORY),
            },
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "ag-news-roberta-lora-eight-group",
        "intervention": "restart_from_base_lora",
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "coalition": mask_description(mask),
        "data_contract": expected_data_contract,
        "data_manifest_sha256": manifest_digest,
        "initialization": {
            "kind": "saved_shared_lora_and_classifier_initialization",
            "state_file": initial_state_path.name,
            "sha256": initial_digest,
            "contract": expected_initialization,
        },
        "model": architecture,
        "training": dict(training),
        "effective_max_length": int(model_settings["max_length"]),
        "train_documents": {
            "available": len(all_train_records),
            "used": len(train_records),
            "fixed_optimizer_updates": int(training["max_updates"]),
        },
        "probe_documents": {"available": len(all_probe_records), "used": len(probe_records)},
        "diagnostics": diagnostics,
        "probe": probe,
        "elapsed_seconds": elapsed,
        "full_model_state": (
            {"state_file": full_state_path.name, "sha256": full_state_digest}
            if full_state_digest is not None
            else None
        ),
        "provenance": {
            **run_metadata(REPOSITORY),
            "package_versions": dict(runtime.package_versions),
            "device": str(device),
        },
    }
    write_json_artifact(result_path, payload)
    print(
        f"written restart artifact mask={mask} train={len(train_records)} probe={len(probe_records)} "
        f"updates={diagnostics['updates']} path={result_path}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=int, default=None, help="Coalition mask 0..255; defaults to SLURM_ARRAY_TASK_ID")
    parser.add_argument("--seed", type=int, default=0, help="Training-order/dropout seed")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--prepared-data", type=Path, default=None)
    parser.add_argument("--initial-state", type=Path, default=None)
    parser.add_argument("--full-model-state", type=Path, default=None)
    parser.add_argument("--hf-cache", type=Path, default=None, help="Optional shared Hugging Face cache directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete torch device")
    parser.add_argument("--smoke", action="store_true", help="Use bounded documents, sequence length, and two updates")
    parser.add_argument("--force", action="store_true", help="Rewrite a valid result artifact (never changes a valid initial state)")
    parser.add_argument(
        "--print-mask-map",
        action="store_true",
        help="Print the deterministic mapping for masks 0..255 and exit without optional dependencies",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
