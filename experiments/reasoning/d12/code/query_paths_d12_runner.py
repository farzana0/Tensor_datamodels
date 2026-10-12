#!/usr/bin/env python3
"""d=12 study runner for the d=12 query-path game.

The game itself is imported from the the development run module, so the dataset, the role
shifts, the provider blocks and the completion structure are byte-identical to the
original.  Only three things are added:

* ``--init-seed`` / ``--data-seed`` split the single ``--seed`` of the original into the
  weight initialisation and the training-data order.  When they are not passed they both
  default to ``--seed`` and the cell is produced exactly as the development run produced it.
* ``--masks-file`` / ``--task-index`` / ``--chunk`` let one Slurm task run many cells.
* the GPU model, CUDA version and host are written into the cell record, because
  the development run recorded only the Slurm job id and its cube turned out to span four
  different GPU models.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch

from experiments.reasoning.common.causal_gpt_arithmetic_model import ModelConfig, TinyTransformer
from experiments.reasoning.common.transformer_arithmetic_fano import (
    cot_loss, cot_metrics, sample_background_cot,
)
from experiments.reasoning.common.transformer_query_paths_d12 import (
    D, Q, completed_queries, role_examples, roles_for_mask, save_json, target_examples, token_layout,
)


def cell_name(init_seed: int, data_seed: int, mask: int, replicate: str = "") -> str:
    stem = f"seed{init_seed}" if init_seed == data_seed else f"init{init_seed}_data{data_seed}"
    suffix = f"_{replicate}" if replicate else ""
    return f"{stem}{suffix}_mask{mask}.json"


def run_cell(args: argparse.Namespace, mask: int) -> dict:
    init_seed = args.init_seed if args.init_seed is not None else args.seed
    data_seed = args.data_seed if args.data_seed is not None else args.seed
    output = args.output_dir / f"mask_{mask}" / cell_name(init_seed, data_seed, mask, args.replicate)
    if output.exists():
        return json.loads(output.read_text())
    layout = token_layout(args.aliases)
    roles = roles_for_mask(mask)
    group_x_np, group_y_np = role_examples(layout, roles)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    if args.require_gpu and args.require_gpu.lower() not in gpu_name.lower().replace(" ", ""):
        raise RuntimeError(f"cell {mask} landed on {gpu_name!r}, required {args.require_gpu!r}")
    torch.manual_seed(init_seed)
    model = TinyTransformer(ModelConfig(vocab_size=layout["vocab_size"], sequence_length=9,
        width=args.width, layers=args.layers, heads=args.heads, mlp_ratio=args.mlp_ratio,
        logit_cap=30.0)).to(device)
    with torch.no_grad():
        identity = model.embedding.weight[layout["identity_token"]].clone()
        model.embedding.weight[layout["novel_start"]:layout["novel_start"] + Q * 3] = identity
    present = {3 * query + role for query, role in roles}
    absent_tokens = [layout["novel_start"] + index for index in range(Q * 3) if index not in present]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    group_x = torch.as_tensor(group_x_np, device=device)
    group_y = torch.as_tensor(group_y_np, device=device)
    target_x_np, target_y_np, query_ids_np, identifiers = target_examples(layout)
    target_x = torch.as_tensor(target_x_np, device=device)
    target_y = torch.as_tensor(target_y_np, device=device)
    generator = torch.Generator(device=device).manual_seed(data_seed * 1000)
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
        "dataset": "three-hop modular relation composition; selected groups supply query-specific relation examples",
        "mask": mask, "seed": init_seed, "init_seed": init_seed, "data_seed": data_seed,
        "replicate": args.replicate,
        "cardinality": mask.bit_count(),
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
        "gpu_name": gpu_name, "cuda_version": torch.version.cuda, "host": socket.gethostname(),
        "torch_version": torch.__version__,
        "preregistration_sha256": args.preregistration_sha256,
        "initialization_policy": "fresh common seed per coalition; absent role embeddings tied to current identity; no checkpoint or warm start",
    }
    save_json(output, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask", type=int)
    parser.add_argument("--masks-file", type=Path)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--init-seed", type=int)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--require-gpu", default="")
    parser.add_argument("--replicate", default="", help="tag distinguishing physical reruns at identical seeds")
    parser.add_argument("--preregistration-sha256", default="")
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
    if args.masks_file:
        masks = [int(x) for x in args.masks_file.read_text().split()]
        start = args.task_index * args.chunk
        masks = masks[start:start + args.chunk]
    else:
        masks = [args.mask]
    for mask in masks:
        row = run_cell(args, mask)
        print(json.dumps({key: row[key] for key in ("mask", "cardinality", "target_accuracy",
            "selected_role_atomic_accuracy", "elapsed_seconds", "gpu_name")}), flush=True)


if __name__ == "__main__":
    main()
