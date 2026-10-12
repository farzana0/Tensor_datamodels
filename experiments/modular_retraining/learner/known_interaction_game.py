"""Controlled modular-relation game with known pair/triple/redundant support."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "experiments/modular_retraining/learner/runs/modular_relations_d8_v1"
SEEDS = (31, 43, 59)
GROUPS = ("A_primary", "B", "C", "A_replica", "D_distractor", "E_distractor", "F_distractor", "G_distractor")
# Affine permutations x -> multiplier*x + offset (mod 11).
RELATIONS = ((1, 1), (2, 0), (1, 3), (3, 1), (4, 2), (5, 3), (7, 4))
GROUP_RELATION = (0, 1, 2, 0, 3, 4, 5, 6)
TASKS = {"atomic_additive_ABC": ((0,), (1,), (2,)), "pair_AB": ((0, 1),),
         "triple_ABC": ((0, 1, 2),), "redundant_A": ((0,),)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def freeze(root):
    root = Path(root)
    if (root / "protocol.json").exists():
        raise FileExistsError(f"Already frozen: {root}")
    for child in ("source_snapshot", "cells", "logs", "analysis"):
        (root / child).mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), root / "source_snapshot/known_interaction_game.py")
    # Hold out every leave-two-out mask from surrogate fitting and validation.
    l2o = [255 ^ (1 << i) ^ (1 << j) for i in range(8) for j in range(i + 1, 8)]
    available = [m for m in range(256) if m not in l2o and m not in (0, 255)]
    rng = np.random.default_rng(20260911)
    rng.shuffle(available)
    observed = [0, 255] + available[:30]
    validation = sorted(observed[-8:])
    fit = sorted(observed[:-8])
    identity = {"dimension": 8, "groups_in_bit_order": GROUPS, "group_relation_indices": GROUP_RELATION,
                "relations_affine_mod11": RELATIONS, "tasks": {k: v for k, v in TASKS.items()},
                "seeds": SEEDS, "coalition_masks": list(range(256)), "surrogate_budget": 32,
                "surrogate_fit_masks": fit, "surrogate_validation_masks": validation,
                "frozen_leave_two_out_test_masks": sorted(l2o),
                "learner": {"type": "relation-specific conditional-logit table", "initialization_sd": 0.05,
                            "optimizer": "Adam", "learning_rate": 0.1, "updates": 200,
                            "training": "full batch over 11 modular facts per retained group"},
                "utility": "mean log probability of exact modular target under composed learned transition matrices",
                "secondary_utility": "argmax exact-answer accuracy", "source_sha256": sha(root / "source_snapshot/known_interaction_game.py")}
    protocol = {**identity, "protocol_hash": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                "scientific_question": "Can sparse TN datamodels recover known redundant pair and three-way support that Linear misses?",
                "primary_task": "triple_ABC", "selection_rule": "validation labels only; frozen leave-two-out is final intervention test"}
    atomic_json(root / "protocol.json", protocol)
    (root / "DESIGN.md").write_text(
        "# Controlled modular-relation game\n\nA tiny neural learner estimates seven relation-specific 11x11 conditional "
        "tables from modular affine facts. Groups A and its replica teach the same relation; B and C teach distinct "
        "relations; four groups are distractors. Evaluation composes the learned stochastic transition matrices. The "
        "targets separately measure additive atomic competence, A-then-B complementarity, A-then-B-then-C three-way "
        "composition, and A/A-replica redundancy. All 256 coalitions are genuinely retrained for three initialization "
        "seeds. Surrogates observe 32 masks; all 28 leave-two-out masks are excluded from fitting and validation.\n")
    print(json.dumps({"root": str(root), "protocol_hash": protocol["protocol_hash"], "fit": fit, "validation": validation, "test": sorted(l2o)}, indent=2))


def mapping(relation):
    multiplier, offset = RELATIONS[relation]
    return (multiplier * np.arange(11) + offset) % 11


def target_for(sequence):
    out = np.arange(11)
    for relation in sequence:
        out = mapping(relation)[out]
    return out


@torch.no_grad()
def evaluate(logits):
    matrices = logits.softmax(-1)
    eye = torch.eye(11, dtype=logits.dtype)
    result = {}
    for task, sequences in TASKS.items():
        nlls, accuracies, per_query = [], [], []
        for sequence in sequences:
            probability = eye
            for relation in sequence:
                probability = probability @ matrices[relation]
            truth = torch.as_tensor(target_for(sequence))
            row = torch.arange(11)
            log_probability = probability[row, truth].clamp_min(1e-30).log()
            correct = probability.argmax(-1).eq(truth)
            nlls.append(-log_probability.mean())
            accuracies.append(correct.float().mean())
            per_query.extend(log_probability.tolist())
        result[task] = {"utility_log_probability": -float(torch.stack(nlls).mean()),
                        "exact_answer_accuracy": float(torch.stack(accuracies).mean()),
                        "per_query_log_probability": per_query}
    return result


def fit(mask, seed, protocol):
    torch.manual_seed(seed)
    logits = torch.nn.Parameter(torch.randn(7, 11, 11) * protocol["learner"]["initialization_sd"])
    initial_hash = hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest()
    selected = [i for i in range(8) if mask >> i & 1]
    final_loss = None
    if selected:
        relations = torch.as_tensor([GROUP_RELATION[i] for i in selected for _ in range(11)])
        x = torch.arange(11).repeat(len(selected))
        truth = torch.cat([torch.as_tensor(mapping(GROUP_RELATION[i])) for i in selected])
        optimizer = torch.optim.Adam([logits], lr=protocol["learner"]["learning_rate"])
        for _ in range(protocol["learner"]["updates"]):
            loss = F.cross_entropy(logits[relations, x], truth)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        final_loss = float(loss.detach())
    metrics = evaluate(logits)
    return {"mask": mask, "seed": seed, "selected_groups": selected, "n_train_facts": 11 * len(selected),
            "initial_state_sha256": initial_hash, "final_state_sha256": hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest(),
            "updates": protocol["learner"]["updates"] if selected else 0, "final_train_loss": final_loss,
            "metrics": metrics}


def worker(root, seed_index):
    root = Path(root); protocol = json.loads((root / "protocol.json").read_text())
    assert sha(root / "source_snapshot/known_interaction_game.py") == protocol["source_sha256"]
    seed = protocol["seeds"][seed_index]
    start = time.perf_counter(); completed = skipped = 0
    for mask in protocol["coalition_masks"]:
        path = root / "cells" / f"seed{seed}_mask{mask}.json"
        if path.exists():
            old = json.loads(path.read_text()); assert old["protocol_hash"] == protocol["protocol_hash"]
            skipped += 1; continue
        cell_start = time.perf_counter()
        result = fit(mask, seed, protocol)
        result.update(protocol_hash=protocol["protocol_hash"], slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                      torch_version=torch.__version__, runtime_seconds=time.perf_counter() - cell_start)
        atomic_json(path, result); completed += 1
    summary = {"status": "PASS", "seed": seed, "completed": completed, "skipped": skipped,
               "expected": 256, "wall_seconds": time.perf_counter() - start, "protocol_hash": protocol["protocol_hash"]}
    atomic_json(root / "cells" / f"seed{seed}_COMPLETE.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "worker")); parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed-index", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    freeze(args.root) if args.command == "freeze" else worker(args.root, args.seed_index)


if __name__ == "__main__":
    main()
