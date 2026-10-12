"""Read-only loaders for the complete coalition cubes used by the comparison.

Every cube is a dense table ``values`` of shape ``(2 ** d, n_outputs)`` indexed by
an integer mask whose bit ``j`` (least significant first) is player ``j``.  No
learner is retrained here: a "coalition evaluation" is a lookup into one of these
tables, so ground truth is exact and every estimator can be given exactly the
same number of evaluations.

The tables are read from the released coalition cubes in ``data/cubes`` (see
``data/README.md``).  The original research code read the same numbers from
per-coalition result files; the cube names, output order
and values are unchanged.  Cubes of experiments that are not part of the paper
(arithmetic d=5, Fano d=8, k-hop d=8, UCI HAR d=10) are not registered here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "cubes"


@dataclass
class Cube:
    """A complete coalition cube plus the structure equal-budget study needs to reason about it."""

    name: str
    d: int
    values: np.ndarray  # (2 ** d, n_outputs)
    output_names: list[str]
    primary_output: int
    family: str  # grouping used for reporting
    learner: str
    utility: str
    source_files: list[str] = field(default_factory=list)
    redundant_groups: list[list[int]] = field(default_factory=list)
    notes: str = ""

    @property
    def n_outputs(self) -> int:
        return self.values.shape[1]

    def output(self, index: int) -> np.ndarray:
        return np.ascontiguousarray(self.values[:, index])


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _released(cube_id: str) -> dict:
    path = DATA / f"{cube_id}.npz"
    blob = np.load(path, allow_pickle=False)
    return {"path": path, "masks": blob["masks"], "values": np.asarray(blob["values"], dtype=float),
            "seeds": blob["seeds"], "output_names": [str(n) for n in blob["output_names"]],
            "players": [str(p) for p in blob["players"]]}


# --------------------------------------------------------------------------
# modular relations d=8, 3 learner seeds x 4 tasks
# --------------------------------------------------------------------------

def _build_modular(seed: int) -> Cube:
    released = _released("modular_relations_d8")
    tasks = ["triple_ABC", "pair_AB", "atomic_additive_ABC", "redundant_A"]
    rows = np.flatnonzero(released["seeds"] == seed)
    rows = rows[np.argsort(released["masks"][rows])]
    if list(released["masks"][rows]) != list(range(256)):
        raise RuntimeError(f"modular seed {seed}: incomplete cube")
    columns = [released["output_names"].index(task) for task in tasks]
    values = released["values"][rows][:, columns]
    return Cube(
        name=f"modular_d8_seed{seed}",
        d=8,
        values=values,
        output_names=tasks,
        primary_output=0,
        family="modular_d8",
        learner="relation-specific conditional-logit tables, full-batch Adam",
        utility="mean log probability of the exact modular target",
        source_files=[str(released["path"])],
        redundant_groups=[[0, 3]],
        notes="groups in bit order: A_primary, B, C, A_replica, D..G distractors",
    )


# --------------------------------------------------------------------------
# AG News RoBERTa-LoRA d=8 (18 cubes: 6 adapter ranks x 3 learner seeds)
# --------------------------------------------------------------------------

def _build_agnews(rank: int, seed: int) -> Cube:
    released = _released(f"agnews_lora_rank{rank}_seed{seed}")
    order = np.argsort(released["masks"])
    if list(released["masks"][order]) != list(range(256)):
        raise RuntimeError(f"AG News rank {rank} seed {seed}: incomplete cube")
    values = released["values"][order]
    if values.shape != (256, 1):
        raise RuntimeError(f"AG News rank {rank} seed {seed}: unexpected shape {values.shape}")
    return Cube(
        name=f"agnews_lora_r{rank}_seed{seed}",
        d=8,
        values=values,
        output_names=["negative_loss"],
        primary_output=0,
        family="agnews_lora_d8",
        learner=f"RoBERTa + LoRA rank {rank} fine-tuned per coalition (seed {seed})",
        utility="negative evaluation loss",
        source_files=[str(released["path"])],
        redundant_groups=[],
        notes="group names: " + json.dumps(released["players"]),
    )


REGISTRY = {}


def _register():
    for seed in (31, 43, 59):
        REGISTRY[f"modular_d8_seed{seed}"] = lambda seed=seed: _build_modular(seed)
    for rank in (1, 2, 4, 8, 16, 64):
        for seed in (31, 43, 59):
            REGISTRY[f"agnews_lora_r{rank}_seed{seed}"] = (
                lambda rank=rank, seed=seed: _build_agnews(rank, seed))


_register()


def load(name: str) -> Cube:
    if name not in REGISTRY:
        raise KeyError(f"unknown cube {name}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def all_names() -> list[str]:
    return list(REGISTRY)


def load_all() -> list[Cube]:
    return [load(name) for name in all_names()]
