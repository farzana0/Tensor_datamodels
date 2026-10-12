#!/usr/bin/env python3
"""d=24 from-scratch transformer game: k=2 query roles, g=3 interchangeable providers.

Copy of `experiments/reasoning/common/transformer_query_paths_d12.py`
(sha256 3cf69e74…, the runner frozen in `15_`'s GATE_PROTOCOL.json).  The learner,
optimiser, step supervision, 80/20 generic/group batch mix, absent-token tying and
common random numbers across coalitions are unchanged.  What changes is the game:

- d = 24 groups and Q = 10 queries, with provider incidence taken verbatim from the
  d=24 study pilot's replicate 1 (`pilot/game.py: balanced_incidence(2, 3, 24, 10, 1001)`),
  i.e. exactly the configuration whose pilot result this arm tests.
- Each query still runs the three-step modular chain of the d=12 game, but only
  steps 1 and 2 use coalition-supplied query-specific relation tokens (the k=2
  roles).  Step 3 uses a *generic* relation token with a query-specific shift, which
  the background data teaches for every coalition.  Keeping the three-step chain
  leaves the model, sequence format, loss and metrics byte-for-byte identical.
- Role shifts reuse the first two entries of each d=12 ROLE_SHIFTS row and the step-3
  shift reuses the third, so a missing role (tied to identity) cannot be cancelled:
  for every query a != 0, b != 0 and a + b != 0 (mod 31).

`--masks-file/--start/--count` runs many coalitions in one process; each coalition
is still an independent fresh model with its own common-seed initialisation.
Differences from the d=12 learner are listed below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.reasoning.common.causal_gpt_arithmetic_model import ModelConfig, TinyTransformer
from experiments.reasoning.common.transformer_arithmetic_fano import (
    MODULUS, cot_metrics, cot_loss, cot_row, sample_background_cot,
)

D = 24
Q = 10
K = 2

# pilot/game.py balanced_incidence(k=2, g=3, d=24, q=10, seed=1001), frozen verbatim.
PROVIDERS = (
    ((3, 15, 16), (10, 12, 18)),
    ((6, 7, 14), (1, 19, 22)),
    ((4, 13, 23), (2, 8, 9)),
    ((0, 17, 21), (5, 11, 20)),
    ((0, 2, 9), (4, 14, 23)),
    ((6, 11, 21), (3, 13, 17)),
    ((10, 20, 22), (1, 5, 16)),
    ((7, 15, 19), (8, 12, 18)),
    ((12, 16, 17), (0, 1, 22)),
    ((2, 6, 10), (13, 18, 19)),
)
# d=12 ROLE_SHIFTS rows: (a, b) drive the two coalition roles, c the generic third step.
D12_ROLE_SHIFTS = (
    (30, 26, 14), (6, 11, 29), (10, 8, 12), (18, 5, 28), (25, 17, 7),
    (23, 22, 21), (13, 3, 9), (15, 1, 19), (4, 20, 16), (27, 2, 24),
)
ROLE_SHIFTS = tuple((a, b) for a, b, _ in D12_ROLE_SHIFTS)
THIRD_SHIFTS = tuple(c for _, _, c in D12_ROLE_SHIFTS)
assert all(a % MODULUS and b % MODULUS and (a + b) % MODULUS for a, b in ROLE_SHIFTS)


def token_layout(aliases: int) -> dict:
    known_start = MODULUS
    novel_start = known_start + MODULUS * aliases
    eq = novel_start + Q * K
    return {"aliases": aliases, "known_start": known_start, "novel_start": novel_start,
            "identity_token": known_start, "eq_token": eq, "vocab_size": eq + 1}


def role_token(layout: dict, query: int, role: int) -> int:
    return layout["novel_start"] + K * query + role


def known_token(layout: dict, shift: int, alias: int = 0) -> int:
    return layout["known_start"] + shift * layout["aliases"] + alias


def roles_for_mask(mask: int) -> list[tuple[int, int]]:
    return [(query, role) for query, groups in enumerate(PROVIDERS)
            for role, providers in enumerate(groups)
            if any(mask & (1 << group) for group in providers)]


def completed_queries(mask: int) -> list[bool]:
    return [all(any(mask & (1 << group) for group in providers) for providers in groups)
            for groups in PROVIDERS]


def role_examples(layout: dict, roles: list[tuple[int, int]], *, atomic_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Identical construction to the d=12 runner, indexed over K roles per query."""
    rows, labels = [], []
    aliases, identity = layout["aliases"], layout["identity_token"]
    for query, role in roles:
        novel, shift = role_token(layout, query, role), ROLE_SHIFTS[query][role]
        role_index = K * query + role
        for position in range(3):
            for context in range(1 if atomic_only else 4):
                for value in range(MODULUS):
                    if atomic_only:
                        known_shifts, known_ops = [0, 0], [identity, identity]
                    else:
                        known_shifts = [(value + 3 * role_index + 5 * context + 1) % MODULUS,
                                        (2 * value + 7 * role_index + 3 * context + 2) % MODULUS]
                        known_ops = [MODULUS + known_shifts[0] * aliases + context % aliases,
                                     MODULUS + known_shifts[1] * aliases + (context + 1) % aliases]
                    ops = known_ops[:position] + [novel] + known_ops[position:]
                    shifts = known_shifts[:position] + [shift] + known_shifts[position:]
                    row, target = cot_row(layout, value, ops, shifts)
                    rows.append(row); labels.append(target)
    if not rows:
        return np.empty((0, 9), np.int64), np.empty((0, 3), np.int64)
    return np.asarray(rows, np.int64), np.asarray(labels, np.int64)


def target_examples(layout: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows, labels, queries, identifiers = [], [], [], []
    for query in range(Q):
        ops = [role_token(layout, query, 0), role_token(layout, query, 1),
               known_token(layout, THIRD_SHIFTS[query])]
        shifts = [ROLE_SHIFTS[query][0], ROLE_SHIFTS[query][1], THIRD_SHIFTS[query]]
        for value in range(MODULUS):
            row, target = cot_row(layout, value, ops, shifts)
            rows.append(row); labels.append(target); queries.append(query)
            identifiers.append(f"query:{query}:input:{value}")
    return np.asarray(rows, np.int64), np.asarray(labels, np.int64), np.asarray(queries), identifiers


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def output_path(args: argparse.Namespace, mask: int) -> Path:
    return args.output_dir / f"mask_{mask}" / f"seed{args.seed}_mask{mask}.json"


def run_cell(args: argparse.Namespace, mask: int) -> dict:
    output = output_path(args, mask)
    if output.exists():
        return json.loads(output.read_text())
    layout = token_layout(args.aliases)
    roles = roles_for_mask(mask)
    group_x_np, group_y_np = role_examples(layout, roles)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model = TinyTransformer(ModelConfig(vocab_size=layout["vocab_size"], sequence_length=9,
        width=args.width, layers=args.layers, heads=args.heads, mlp_ratio=args.mlp_ratio,
        logit_cap=30.0)).to(device)
    with torch.no_grad():
        identity = model.embedding.weight[layout["identity_token"]].clone()
        model.embedding.weight[layout["novel_start"]:layout["novel_start"] + Q * K] = identity
    present = {K * query + role for query, role in roles}
    absent_tokens = [layout["novel_start"] + index for index in range(Q * K) if index not in present]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    group_x = torch.as_tensor(group_x_np, device=device)
    group_y = torch.as_tensor(group_y_np, device=device)
    target_x_np, target_y_np, query_ids_np, identifiers = target_examples(layout)
    target_x = torch.as_tensor(target_x_np, device=device)
    target_y = torch.as_tensor(target_y_np, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed * 1000)
    history, started = [], time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            measured = cot_metrics(model, target_x, target_y)
            history.append({"step": step, "accuracy": measured["accuracy"],
                            "elapsed_seconds": time.time() - started})
        if step == args.steps:
            break
        if roles:
            group_batch = args.batch_size // 5
            base_batch = args.batch_size - group_batch
            base_x, base_y = sample_background_cot(layout, base_batch, generator, device)
            group_index = torch.randint(len(group_x), (group_batch,), generator=generator, device=device)
            train_x = torch.cat((base_x, group_x[group_index]))
            train_y = torch.cat((base_y, group_y[group_index]))
        else:
            train_x, train_y = sample_background_cot(layout, args.batch_size, generator, device)
        model.train()
        loss = cot_loss(model, train_x, train_y)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if absent_tokens:
            with torch.no_grad():
                model.embedding.weight[absent_tokens] = model.embedding.weight[layout["identity_token"]].clone()
    target = cot_metrics(model, target_x, target_y, include_per_example=True)
    correct = np.asarray(target["correct_by_example"], dtype=float)
    logp = np.asarray(target["log_probability_by_example"], dtype=float)
    accuracy_by_query = [float(correct[query_ids_np == query].mean()) for query in range(Q)]
    logp_by_query = [float(logp[query_ids_np == query].mean()) for query in range(Q)]
    atomic_accuracy = None
    if roles:
        atomic_x_np, atomic_y_np = role_examples(layout, roles, atomic_only=True)
        atomic = cot_metrics(model, torch.as_tensor(atomic_x_np, device=device),
                             torch.as_tensor(atomic_y_np, device=device))
        atomic_accuracy = atomic["accuracy"]
    row = {
        "learner": "8-layer width-256 step-supervised causal-GPT transformer trained from scratch",
        "dataset": "two coalition roles + one generic step in a three-step modular chain; selected groups supply query-specific relation examples",
        "mask": mask, "seed": args.seed, "cardinality": mask.bit_count(),
        "selected_groups": [g for g in range(D) if mask & (1 << g)],
        "provided_roles": [f"{q}:{r}" for q, r in roles],
        "completed_queries": completed_queries(mask),
        "target_accuracy": target["accuracy"], "utility_log_probability": target["mean_log_probability"],
        "target_accuracy_by_query": accuracy_by_query, "target_log_probability_by_query": logp_by_query,
        "target_query_ids": identifiers,
        "target_correct_by_query_input": target["correct_by_example"],
        "target_log_probability_by_query_input": target["log_probability_by_example"],
        "selected_role_atomic_accuracy": atomic_accuracy,
        "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr,
        "weight_decay": args.weight_decay, "history": history, "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu_name": torch.cuda.get_device_name(0),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initialization_policy": "fresh common seed per coalition; absent role embeddings tied to current identity; no checkpoint or warm start",
    }
    save_json(output, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask", type=int)
    parser.add_argument("--masks-file", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--aliases", type=int, default=3)
    args = parser.parse_args()
    if args.mask is not None:
        masks = [args.mask]
    else:
        all_masks = [int(x) for x in args.masks_file.read_text().split()]
        masks = all_masks[args.start:args.start + args.count]
    for mask in masks:
        row = run_cell(args, mask)
        print(json.dumps({key: row[key] for key in ("mask", "cardinality", "target_accuracy",
              "selected_role_atomic_accuracy", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
