#!/usr/bin/env python3
"""Small causal-transformer arithmetic game with Fano-balanced triple supports.

The background phase teaches a decoder to compose opaque translations over
Z_31.  Eight held-out operator tokens are initialized as the identity.  A
coalition retraining run learns only the embeddings for its selected operators
from atomic equations, then evaluates seven three-operator routes arranged as
the Fano plane.  Every point and every pair among groups 0--6 is balanced;
group 7 is an equally sized distractor absent from target proofs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from causal_gpt_arithmetic_model import ModelConfig, TinyTransformer


FANO_LINES = ((0, 1, 2), (0, 3, 4), (0, 5, 6), (1, 3, 5),
              (1, 4, 6), (2, 3, 6), (2, 4, 5))
MODULUS = 31
NOVEL_SHIFTS = (1, 2, 4, 8, 16, 3, 6, 12)
BASE_UNIVERSE_SIZE = MODULUS * (MODULUS * 3) ** 3


def token_layout(aliases: int) -> dict:
    known_start = MODULUS
    novel_start = known_start + MODULUS * aliases
    eq = novel_start + 8
    return {
        "aliases": aliases,
        "known_start": known_start,
        "novel_start": novel_start,
        "identity_token": known_start,
        "eq_token": eq,
        "vocab_size": eq + 1,
    }


def known_token(shift: torch.Tensor, alias: torch.Tensor, aliases: int) -> torch.Tensor:
    return MODULUS + shift * aliases + alias


def atomic_arrays(layout: dict, groups: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, labels, owners = [], [], []
    identity = layout["identity_token"]
    for group in groups:
        op = layout["novel_start"] + group
        for position in range(3):
            for x in range(MODULUS):
                ops = [identity, identity, identity]
                ops[position] = op
                rows.append([x, *ops, layout["eq_token"]])
                labels.append((x + NOVEL_SHIFTS[group]) % MODULUS)
                owners.append(group)
    return np.asarray(rows, np.int64), np.asarray(labels, np.int64), np.asarray(owners, np.int64)


def support_arrays(layout: dict, groups: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Teach each new operator in varied compositions with known operators."""
    rows, labels, owners = [], [], []
    aliases = layout["aliases"]
    for group in groups:
        novel = layout["novel_start"] + group
        for position in range(3):
            for context in range(4):
                for x in range(MODULUS):
                    left_shift = (x + 3 * group + 5 * context + 1) % MODULUS
                    right_shift = (2 * x + 7 * group + 3 * context + 2) % MODULUS
                    known = [
                        MODULUS + left_shift * aliases + context % aliases,
                        MODULUS + right_shift * aliases + (context + 1) % aliases,
                    ]
                    ops = known[:position] + [novel] + known[position:]
                    rows.append([x, *ops, layout["eq_token"]])
                    labels.append((x + left_shift + NOVEL_SHIFTS[group] + right_shift) % MODULUS)
                    owners.append(group)
    return np.asarray(rows, np.int64), np.asarray(labels, np.int64), np.asarray(owners, np.int64)


def target_arrays(layout: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, labels, lines = [], [], []
    for line_id, line in enumerate(FANO_LINES):
        for x in range(MODULUS):
            rows.append([x, *(layout["novel_start"] + g for g in line), layout["eq_token"]])
            labels.append((x + sum(NOVEL_SHIFTS[g] for g in line)) % MODULUS)
            lines.append(line_id)
    return np.asarray(rows, np.int64), np.asarray(labels, np.int64), np.asarray(lines, np.int64)


def background_eval_arrays(layout: dict, size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Recreate a fixed held-out background-composition sample on CPU."""
    rng = np.random.default_rng(20260931)
    values = rng.integers(MODULUS, size=size, dtype=np.int64)
    shifts = rng.integers(MODULUS, size=(size, 3), dtype=np.int64)
    aliases = rng.integers(layout["aliases"], size=(size, 3), dtype=np.int64)
    ops = MODULUS + shifts * layout["aliases"] + aliases
    rows = np.column_stack((values, ops, np.full(size, layout["eq_token"], dtype=np.int64)))
    labels = (values + shifts.sum(1)) % MODULUS
    return rows, labels


def sample_background(
    layout: dict, size: int, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample uniformly from the fixed exhaustive familiar-operation universe.

    This is a data-loader operation, not data generation that changes between
    coalitions: the finite base dataset is the Cartesian product of 31 inputs
    and three positions containing any of the 93 familiar alias tokens.
    """
    values = torch.randint(MODULUS, (size,), generator=generator, device=device)
    shifts = torch.randint(MODULUS, (size, 3), generator=generator, device=device)
    aliases = torch.randint(
        layout["aliases"], (size, 3), generator=generator, device=device
    )
    # Match the verified arithmetic learner: one quarter of the loader draws
    # are atomic members of the same Cartesian universe.  This gives the model
    # a curriculum without changing the fixed base set or coalition semantics.
    atomic_count = size // 4
    if atomic_count:
        keep = torch.randint(
            3, (atomic_count,), generator=generator, device=device
        )
        atomic_mask = torch.arange(3, device=device)[None] != keep[:, None]
        shifts[:atomic_count][atomic_mask] = 0
    ops = MODULUS + shifts * layout["aliases"] + aliases
    eq = torch.full((size, 1), layout["eq_token"], device=device, dtype=torch.long)
    rows = torch.cat((values[:, None], ops, eq), dim=1)
    labels = (values + shifts.sum(1)) % MODULUS
    return rows, labels


def cot_row(layout: dict, value: int, ops: list[int], shifts: list[int]) -> tuple[list[int], list[int]]:
    """Build an explicit three-step causal trace and its supervised states."""
    state1 = (value + shifts[0]) % MODULUS
    state2 = (state1 + shifts[1]) % MODULUS
    state3 = (state2 + shifts[2]) % MODULUS
    eq = layout["eq_token"]
    return (
        [value, ops[0], eq, state1, ops[1], eq, state2, ops[2], eq],
        [state1, state2, state3],
    )


def cot_group_arrays(layout: dict, groups: list[int], *, atomic_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    aliases = layout["aliases"]
    identity = layout["identity_token"]
    for group in groups:
        novel = layout["novel_start"] + group
        for position in range(3):
            if atomic_only:
                contexts = [(0, 0, identity, identity)]
            else:
                contexts = []
                for context in range(4):
                    # Fixed contexts span both input-dependent and group-dependent
                    # familiar translations, exactly as in direct supervision.
                    contexts.append((context, None, None, None))
            for context, _, _, _ in contexts:
                for value in range(MODULUS):
                    if atomic_only:
                        known_shifts = [0, 0]
                        known_ops = [identity, identity]
                    else:
                        known_shifts = [
                            (value + 3 * group + 5 * context + 1) % MODULUS,
                            (2 * value + 7 * group + 3 * context + 2) % MODULUS,
                        ]
                        known_ops = [
                            MODULUS + known_shifts[0] * aliases + context % aliases,
                            MODULUS + known_shifts[1] * aliases + (context + 1) % aliases,
                        ]
                    ops = known_ops[:position] + [novel] + known_ops[position:]
                    shifts = known_shifts[:position] + [NOVEL_SHIFTS[group]] + known_shifts[position:]
                    row, target = cot_row(layout, value, ops, shifts)
                    rows.append(row); labels.append(target)
    return np.asarray(rows, np.int64), np.asarray(labels, np.int64)


def cot_target_arrays(layout: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, labels, lines = [], [], []
    for line_id, line in enumerate(FANO_LINES):
        for value in range(MODULUS):
            ops = [layout["novel_start"] + group for group in line]
            shifts = [NOVEL_SHIFTS[group] for group in line]
            row, target = cot_row(layout, value, ops, shifts)
            rows.append(row); labels.append(target); lines.append(line_id)
    return np.asarray(rows, np.int64), np.asarray(labels, np.int64), np.asarray(lines, np.int64)


def sample_background_cot(
    layout: dict, size: int, generator: torch.Generator, device: torch.device,
    *, scaffold_atomic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.randint(MODULUS, (size,), generator=generator, device=device)
    shifts = torch.randint(MODULUS, (size, 3), generator=generator, device=device)
    aliases = torch.randint(layout["aliases"], (size, 3), generator=generator, device=device)
    atomic_count = size // 4 if scaffold_atomic else 0
    if atomic_count:
        keep = torch.randint(3, (atomic_count,), generator=generator, device=device)
        atomic_mask = torch.arange(3, device=device)[None] != keep[:, None]
        shifts[:atomic_count][atomic_mask] = 0
    ops = MODULUS + shifts * layout["aliases"] + aliases
    state1 = (values + shifts[:, 0]) % MODULUS
    state2 = (state1 + shifts[:, 1]) % MODULUS
    state3 = (state2 + shifts[:, 2]) % MODULUS
    eq = torch.full((size,), layout["eq_token"], device=device, dtype=torch.long)
    rows = torch.stack((values, ops[:, 0], eq, state1, ops[:, 1], eq, state2, ops[:, 2], eq), dim=1)
    return rows, torch.stack((state1, state2, state3), dim=1)


def cot_loss(model: nn.Module, rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(rows)[:, (2, 5, 8), :MODULUS]
    return nn.functional.cross_entropy(logits.reshape(-1, MODULUS), labels.reshape(-1))


@torch.no_grad()
def cot_metrics(
    model: nn.Module,
    rows: torch.Tensor,
    labels: torch.Tensor,
    *,
    include_per_example: bool = False,
) -> dict[str, object]:
    """Autoregressively insert intermediate states before scoring the final state."""
    model.eval()
    generated = rows.clone()
    step_predictions = []
    step_logp = []
    for position, insert_at, label_column in ((2, 3, 0), (5, 6, 1), (8, None, 2)):
        logits = model(generated[:, : position + 1])[:, -1, :MODULUS]
        prediction = logits.argmax(-1)
        step_predictions.append(prediction)
        step_logp.append(logits.log_softmax(-1)[torch.arange(len(rows), device=rows.device), labels[:, label_column]])
        if insert_at is not None:
            generated[:, insert_at] = prediction
    final_correct = step_predictions[-1] == labels[:, -1]
    result: dict[str, object] = {
        "accuracy": float(final_correct.float().mean()),
        "all_steps_accuracy": float(torch.stack([
            step_predictions[i] == labels[:, i] for i in range(3)
        ], dim=1).all(1).float().mean()),
        "mean_log_probability": float(step_logp[-1].mean()),
    }
    if include_per_example:
        result["correct_by_example"] = final_correct.cpu().tolist()
        result["log_probability_by_example"] = step_logp[-1].cpu().tolist()
    return result


@torch.no_grad()
def metrics(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.eval()
    logits = model(x)[:, -1, :MODULUS]
    logp = logits.log_softmax(-1)[torch.arange(len(y), device=y.device), y]
    return {
        "accuracy": float((logits.argmax(-1) == y).float().mean()),
        "mean_log_probability": float(logp.mean()),
        "nll": float(-logp.mean()),
    }


def adapter_logits(
    model: TinyTransformer, tokens: torch.Tensor, alpha: torch.Tensor, layout: dict
) -> torch.Tensor:
    """Run the unchanged transformer with differentiable operation adapters."""
    embedding = model.embedding(tokens)
    identity = model.embedding.weight[layout["identity_token"]]
    for group, shift in enumerate(NOVEL_SHIFTS):
        target = model.embedding.weight[MODULUS + shift * layout["aliases"]]
        delta = alpha[group] * (target - identity)
        embedding = embedding + (tokens == layout["novel_start"] + group).unsqueeze(-1) * delta
    hidden, first_value = embedding, None
    for block in model.blocks:
        hidden, first_value = block(hidden, first_value)
    hidden = nn.functional.rms_norm(
        hidden + model.embedding_shortcut(embedding), (hidden.shape[-1],)
    )
    logits = model.output(hidden).float()
    cap = model.config.logit_cap
    return cap * torch.tanh(logits / cap)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def pretrain(args: argparse.Namespace) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    layout = token_layout(args.aliases)
    config = {
        "learner": "causal-GPT transformer",
        "task": "compose three opaque translations over Z_31",
        "width": args.width, "heads": args.heads, "mlp_ratio": args.mlp_ratio,
        "layers": args.layers, "aliases": args.aliases,
        "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "seed": args.seed, "layout": layout,
    }
    save_json(output / "protocol.json", config)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model_config = ModelConfig(
        vocab_size=layout["vocab_size"], sequence_length=5, width=args.width,
        layers=args.layers, heads=args.heads, mlp_ratio=args.mlp_ratio,
        logit_cap=30.0,
    )
    model = TinyTransformer(model_config).to(device)
    # Held-out group tokens start as exact identity operators. Coalition
    # retraining therefore has a clear causal interpretation: absent support
    # contributes no translation rather than an arbitrary random mapping.
    with torch.no_grad():
        identity = model.embedding.weight[layout["identity_token"]].clone()
        model.embedding.weight[layout["novel_start"]:layout["novel_start"] + 8] = identity
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)

    def batch(size: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.randint(MODULUS, (size,), generator=generator, device=device)
        shifts = torch.randint(MODULUS, (size, 3), generator=generator, device=device)
        aliases = torch.randint(args.aliases, (size, 3), generator=generator, device=device)
        ops = known_token(shifts, aliases, args.aliases)
        # Include explicit atomic equations in one quarter of each batch.
        n_atomic = size // 4
        if n_atomic:
            keep = torch.randint(3, (n_atomic,), generator=generator, device=device)
            mask = torch.arange(3, device=device)[None] != keep[:, None]
            shifts[:n_atomic][mask] = 0
            ops[:n_atomic] = known_token(shifts[:n_atomic], aliases[:n_atomic], args.aliases)
        eq = torch.full((size, 1), layout["eq_token"], dtype=torch.long, device=device)
        x = torch.cat((values[:, None], ops, eq), dim=1)
        y = (values + shifts.sum(1)) % MODULUS
        return x, y

    eval_generator = torch.Generator(device=device).manual_seed(20260931)
    values = torch.randint(MODULUS, (8192,), generator=eval_generator, device=device)
    shifts = torch.randint(MODULUS, (8192, 3), generator=eval_generator, device=device)
    aliases = torch.randint(args.aliases, (8192, 3), generator=eval_generator, device=device)
    eval_x = torch.cat((values[:, None], known_token(shifts, aliases, args.aliases),
                        torch.full((8192, 1), layout["eq_token"], device=device, dtype=torch.long)), dim=1)
    eval_y = (values + shifts.sum(1)) % MODULUS
    history = []
    started = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            record = {"step": step, **metrics(model, eval_x, eval_y), "elapsed_seconds": time.time() - started}
            history.append(record)
            print(json.dumps(record), flush=True)
        if step == args.steps:
            break
        x, y = batch(args.batch_size)
        model.train()
        loss = nn.functional.cross_entropy(model(x)[:, -1, :MODULUS], y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final = metrics(model, eval_x, eval_y)
    checkpoint = {"config": config, "model": model.state_dict(), "history": history,
                  "final": final, "parameter_count": sum(p.numel() for p in model.parameters())}
    torch.save(checkpoint, output / "checkpoint.pt")
    save_json(output / "result.json", {**config, "final": final,
              "parameter_count": checkpoint["parameter_count"], "history": history})


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    c, layout = checkpoint["config"], checkpoint["config"]["layout"]
    model = TinyTransformer(ModelConfig(
        vocab_size=layout["vocab_size"], sequence_length=5, width=c["width"],
        layers=c["layers"], heads=c["heads"], mlp_ratio=c["mlp_ratio"],
        logit_cap=30.0,
    )).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def run_cell(
    checkpoint_path: Path, output: Path, mask: int, seed: int, steps: int,
    lr: float, batch_size: int, tune_scope: str,
) -> dict:
    path = output / f"seed{seed}_mask{mask}.json"
    if path.exists():
        return json.loads(path.read_text())
    device = torch.device("cuda")
    model, checkpoint = load_model(checkpoint_path, device)
    layout = checkpoint["config"]["layout"]
    selected = [g for g in range(8) if mask >> g & 1]
    train_x_np, train_y_np, owners = support_arrays(layout, selected)
    torch.manual_seed(seed)
    if selected:
        train_x = torch.as_tensor(train_x_np, device=device)
        train_y = torch.as_tensor(train_y_np, device=device)
        alpha = None
        if tune_scope == "adapter":
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            alpha = nn.Parameter(torch.zeros(8, device=device))
            parameters = [alpha]
        elif tune_scope == "embedding":
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.embedding.weight.requires_grad_(True)
            parameters = [model.embedding.weight]
        elif tune_scope == "full":
            parameters = list(model.parameters())
        else:
            raise ValueError(tune_scope)
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
        generator = torch.Generator(device=device).manual_seed(seed * 1000 + mask)
        allowed = torch.zeros_like(model.embedding.weight)
        for group in selected:
            allowed[layout["novel_start"] + group] = 1
        for _ in range(steps):
            model.train()
            index = torch.randint(len(train_x), (batch_size,), generator=generator, device=device)
            logits = adapter_logits(model, train_x[index], alpha, layout) if alpha is not None else model(train_x[index])
            loss = nn.functional.cross_entropy(logits[:, -1, :MODULUS], train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tune_scope == "embedding":
                model.embedding.weight.grad.mul_(allowed)
            optimizer.step()
            if alpha is not None:
                with torch.no_grad():
                    alpha.clamp_(0.0, 1.0)
        if alpha is not None:
            with torch.no_grad():
                identity = model.embedding.weight[layout["identity_token"]].clone()
                for group, shift in enumerate(NOVEL_SHIFTS):
                    target_embedding = model.embedding.weight[MODULUS + shift * layout["aliases"]].clone()
                    model.embedding.weight[layout["novel_start"] + group] = identity + alpha[group] * (target_embedding - identity)
            adapter_values = alpha.detach().cpu().tolist()
        else:
            adapter_values = None
    else:
        adapter_values = [0.0] * 8 if tune_scope == "adapter" else None
    target_x_np, target_y_np, line_ids_np = target_arrays(layout)
    target_x = torch.as_tensor(target_x_np, device=device)
    target_y = torch.as_tensor(target_y_np, device=device)
    target = metrics(model, target_x, target_y)
    with torch.no_grad():
        pred = model(target_x)[:, -1, :MODULUS].argmax(-1).cpu().numpy()
    per_line = [float(np.mean(pred[line_ids_np == i] == target_y_np[line_ids_np == i])) for i in range(7)]
    atomic = support = None
    if selected:
        support = metrics(model, torch.as_tensor(train_x_np, device=device), torch.as_tensor(train_y_np, device=device))
        atomic_x_np, atomic_y_np, _ = atomic_arrays(layout, selected)
        atomic = metrics(model, torch.as_tensor(atomic_x_np, device=device), torch.as_tensor(atomic_y_np, device=device))
    background_x_np, background_y_np = background_eval_arrays(layout)
    background = metrics(
        model, torch.as_tensor(background_x_np, device=device),
        torch.as_tensor(background_y_np, device=device),
    )
    background_x_np, background_y_np = background_eval_arrays(layout)
    background = metrics(model, torch.as_tensor(background_x_np, device=device), torch.as_tensor(background_y_np, device=device))
    result = {
        "learner": f"causal-GPT transformer; coalition retraining scope={tune_scope}",
        "mask": mask, "seed": seed, "selected_groups": selected,
        "cardinality": len(selected), "completed_fano_routes": sum(all(mask >> g & 1 for g in line) for line in FANO_LINES),
        "target_accuracy": target["accuracy"], "utility_log_probability": target["mean_log_probability"],
        "target_accuracy_by_line": per_line, "selected_atomic_accuracy": None if atomic is None else atomic["accuracy"],
        "selected_support_accuracy": None if support is None else support["accuracy"],
        "background_composition_accuracy": background["accuracy"],
        "background_composition_log_probability": background["mean_log_probability"],
        "finetune_steps": 0 if not selected else steps, "finetune_lr": lr,
        "finetune_batch_size": batch_size, "tune_scope": tune_scope,
        "adapter_values": adapter_values,
        "checkpoint": str(checkpoint_path), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    save_json(path, result)
    return result


def cells(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks = list(range(256)) if args.masks == "all" else [int(x) for x in args.masks.split(",")]
    protocol = {
        "learner": f"causal-GPT transformer; coalition retraining scope={args.tune_scope}",
        "checkpoint": str(args.checkpoint), "dimension": 8, "modulus": MODULUS,
        "chance_accuracy": 1 / MODULUS, "fano_lines": FANO_LINES,
        "novel_shifts": NOVEL_SHIFTS, "group_7": "equal-size distractor",
        "steps": args.finetune_steps, "lr": args.finetune_lr,
        "batch_size": args.finetune_batch_size, "seed": args.seed,
        "tune_scope": args.tune_scope,
        "size_control": "fit cardinality mean first; center residuals within size; rank only within k=3",
    }
    protocol["sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    save_json(args.output_dir / "protocol.json", protocol)
    rows = []
    for mask in masks:
        row = run_cell(
            args.checkpoint, args.output_dir, mask, args.seed,
            args.finetune_steps, args.finetune_lr,
            args.finetune_batch_size, args.tune_scope,
        )
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("mask", "cardinality", "completed_fano_routes", "target_accuracy", "selected_atomic_accuracy")}), flush=True)
    save_json(args.output_dir / f"seed{args.seed}_summary.json", {"protocol": protocol, "cells": rows})


def run_scratch_cell(args: argparse.Namespace, mask: int) -> dict:
    folder = args.output_dir / f"mask_{mask}" if args.nested_output else args.output_dir
    path = folder / f"seed{args.seed}_mask{mask}.json"
    if path.exists():
        return load_json(path)
    layout = token_layout(args.aliases)
    selected = [group for group in range(8) if mask >> group & 1]
    if selected:
        context_x, context_y, _ = support_arrays(layout, selected)
        atomic_x, atomic_y, _ = atomic_arrays(layout, selected)
        group_x_np = np.concatenate((context_x, atomic_x))
        group_y_np = np.concatenate((context_y, atomic_y))
    else:
        group_x_np = np.empty((0, 5), dtype=np.int64)
        group_y_np = np.empty((0,), dtype=np.int64)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model = TinyTransformer(ModelConfig(
        vocab_size=layout["vocab_size"], sequence_length=5,
        width=args.width, layers=args.layers, heads=args.heads,
        mlp_ratio=args.mlp_ratio, logit_cap=30.0,
    )).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    group_x = torch.as_tensor(group_x_np, device=device)
    group_y = torch.as_tensor(group_y_np, device=device)
    target_x_np, target_y_np, line_ids_np = target_arrays(layout)
    target_x = torch.as_tensor(target_x_np, device=device)
    target_y = torch.as_tensor(target_y_np, device=device)
    background_x_np, background_y_np = background_eval_arrays(layout)
    background_x = torch.as_tensor(background_x_np, device=device)
    background_y = torch.as_tensor(background_y_np, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed * 1000)
    history = []
    started = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 and mask in (127, 255):
            history.append({
                "step": step,
                "target_accuracy": metrics(model, target_x, target_y)["accuracy"],
                "background_accuracy": metrics(model, background_x, background_y)["accuracy"],
                "elapsed_seconds": time.time() - started,
            })
        if step == args.steps:
            break
        if selected:
            group_batch = args.batch_size // 5
            base_batch = args.batch_size - group_batch
            base_x, base_y = sample_background(layout, base_batch, generator, device)
            group_index = torch.randint(
                len(group_x), (group_batch,), generator=generator, device=device
            )
            train_x = torch.cat((base_x, group_x[group_index]))
            train_y = torch.cat((base_y, group_y[group_index]))
        else:
            train_x, train_y = sample_background(
                layout, args.batch_size, generator, device
            )
        model.train()
        loss = nn.functional.cross_entropy(
            model(train_x)[:, -1, :MODULUS], train_y
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    target = metrics(model, target_x, target_y)
    background = metrics(model, background_x, background_y)
    with torch.no_grad():
        prediction = model(target_x)[:, -1, :MODULUS].argmax(-1).cpu().numpy()
    per_line = [
        float(np.mean(prediction[line_ids_np == line] == target_y_np[line_ids_np == line]))
        for line in range(7)
    ]
    atomic = None
    if selected:
        atomic_x_np, atomic_y_np, _ = atomic_arrays(layout, selected)
        atomic = metrics(
            model, torch.as_tensor(atomic_x_np, device=device),
            torch.as_tensor(atomic_y_np, device=device),
        )
    result = {
        "learner": "causal-GPT transformer trained from scratch on base union selected groups",
        "seed": args.seed, "mask": mask, "cardinality": len(selected),
        "selected_groups": selected,
        "training_example_count": BASE_UNIVERSE_SIZE + len(group_x_np),
        "completed_fano_routes": sum(all(mask >> group & 1 for group in line) for line in FANO_LINES),
        "target_accuracy": target["accuracy"],
        "utility_log_probability": target["mean_log_probability"],
        "target_accuracy_by_line": per_line,
        "selected_atomic_accuracy": None if atomic is None else atomic["accuracy"],
        "background_composition_accuracy": background["accuracy"],
        "background_composition_log_probability": background["mean_log_probability"],
        "steps": args.steps, "learning_rate": args.lr,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "initialization_policy": "fresh common seed per coalition; no checkpoint or warm start",
        "history": history, "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    save_json(path, result)
    return result


def run_scratch_cot_cell(args: argparse.Namespace, mask: int) -> dict:
    folder = args.output_dir / f"mask_{mask}" if args.nested_output else args.output_dir
    path = folder / f"seed{args.seed}_mask{mask}.json"
    if path.exists():
        return load_json(path)
    layout = token_layout(args.aliases)
    selected = [group for group in range(8) if mask >> group & 1]
    if selected:
        context_x, context_y = cot_group_arrays(layout, selected)
        atomic_x, atomic_y = cot_group_arrays(layout, selected, atomic_only=True)
        group_x_np = np.concatenate((context_x, atomic_x))
        group_y_np = np.concatenate((context_y, atomic_y))
    else:
        group_x_np = np.empty((0, 9), dtype=np.int64)
        group_y_np = np.empty((0, 3), dtype=np.int64)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model = TinyTransformer(ModelConfig(
        vocab_size=layout["vocab_size"], sequence_length=9,
        width=args.width, layers=args.layers, heads=args.heads,
        mlp_ratio=args.mlp_ratio, logit_cap=30.0,
    )).to(device)
    # An unavailable operation has a fixed no-op representation. This is part
    # of the common fresh initialization and prevents arbitrary missing-token
    # behavior from contaminating the coalition utility.
    with torch.no_grad():
        identity = model.embedding.weight[layout["identity_token"]].clone()
        model.embedding.weight[layout["novel_start"]:layout["novel_start"] + 8] = identity
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    group_x = torch.as_tensor(group_x_np, device=device)
    group_y = torch.as_tensor(group_y_np, device=device)
    target_x_np, target_y_np, line_ids_np = cot_target_arrays(layout)
    target_x = torch.as_tensor(target_x_np, device=device)
    target_y = torch.as_tensor(target_y_np, device=device)
    # Common random numbers across coalitions isolate the dataset-mask effect:
    # every nonempty coalition receives the same base stream and index draws.
    generator = torch.Generator(device=device).manual_seed(args.seed * 1000)
    history = []
    started = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 and mask in (127, 255):
            measured = cot_metrics(model, target_x, target_y)
            history.append({"step": step, "target_accuracy": measured["accuracy"],
                            "target_all_steps_accuracy": measured["all_steps_accuracy"],
                            "elapsed_seconds": time.time() - started})
        if step == args.steps:
            break
        if selected:
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
    target = cot_metrics(model, target_x, target_y, include_per_example=True)
    with torch.no_grad():
        generated = target_x.clone()
        for position, insert_at in ((2, 3), (5, 6)):
            pred = model(generated[:, :position + 1])[:, -1, :MODULUS].argmax(-1)
            generated[:, insert_at] = pred
        prediction = model(generated)[:, -1, :MODULUS].argmax(-1).cpu().numpy()
    per_line = [float(np.mean(prediction[line_ids_np == line] == target_y_np[line_ids_np == line, -1])) for line in range(7)]
    atomic = None
    if selected:
        atomic_x_np, atomic_y_np = cot_group_arrays(layout, selected, atomic_only=True)
        atomic = cot_metrics(model, torch.as_tensor(atomic_x_np, device=device),
                             torch.as_tensor(atomic_y_np, device=device))
    # Familiar-operation composition is measured on a fixed seed from the
    # same finite base universe, separately from all training-loader draws.
    eval_generator = torch.Generator(device=device).manual_seed(20260931)
    background_x, background_y = sample_background_cot(
        layout, 4096, eval_generator, device, scaffold_atomic=False
    )
    background = cot_metrics(model, background_x, background_y)
    result = {
        "learner": "step-supervised causal-GPT transformer trained from scratch on base union selected groups",
        "supervision": "autoregressive intermediate states (chain of thought)",
        "seed": args.seed, "mask": mask, "cardinality": len(selected),
        "selected_groups": selected,
        "training_example_count": BASE_UNIVERSE_SIZE + len(group_x_np),
        "completed_fano_routes": sum(all(mask >> group & 1 for group in line) for line in FANO_LINES),
        "target_accuracy": target["accuracy"],
        "target_all_steps_accuracy": target["all_steps_accuracy"],
        "utility_log_probability": target["mean_log_probability"],
        "target_accuracy_by_line": per_line,
        # These fields make future coalition tensors query-conditioned without
        # another learner run.  Existing cells predate them and retain only
        # the seven route-level averages.
        "target_query_ids": [f"line:{int(line)}:input:{int(value)}"
                             for line, value in zip(line_ids_np, target_x_np[:, 0])],
        "target_correct_by_query": target["correct_by_example"],
        "target_log_probability_by_query": target["log_probability_by_example"],
        "selected_atomic_accuracy": None if atomic is None else atomic["accuracy"],
        "background_composition_accuracy": background["accuracy"],
        "background_all_steps_accuracy": background["all_steps_accuracy"],
        "steps": args.steps, "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "initialization_policy": "fresh common seed per coalition; absent novel tokens initialized as identity; no checkpoint or warm start",
        "history": history, "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    save_json(path, result)
    return result


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def scratch_cells(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.masks_file:
        masks = [int(value) for value in args.masks_file.read_text().split()]
    else:
        masks = list(range(256)) if args.masks == "all" else [int(value) for value in args.masks.split(",")]
    protocol = {
        "learner": ("step-supervised causal-GPT transformer" if args.supervision == "cot" else "direct-answer causal-GPT transformer") + " trained from scratch independently for every coalition",
        "dataset": "fixed base arithmetic examples union selected fixed groups",
        "no_warm_start": True, "dimension": 8, "modulus": MODULUS,
        "chance_accuracy": 1 / MODULUS, "fano_lines": FANO_LINES,
        "novel_shifts": NOVEL_SHIFTS, "group_example_count": 465,
        "base_example_count": BASE_UNIVERSE_SIZE, "seed": args.seed,
        "sampling": "80% from fixed exhaustive base universe using the verified 25% atomic scaffold; 20% uniform over examples in selected equal-size groups",
        "steps": args.steps, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "supervision": args.supervision,
        "model": {"width": args.width, "layers": args.layers, "heads": args.heads, "mlp_ratio": args.mlp_ratio},
    }
    protocol["sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    save_json(args.output_dir / "protocol.json", protocol)
    rows = []
    for mask in masks:
        row = run_scratch_cot_cell(args, mask) if args.supervision == "cot" else run_scratch_cell(args, mask)
        rows.append(row)
        print(json.dumps({key: row[key] for key in (
            "mask", "cardinality", "training_example_count", "target_accuracy",
            "selected_atomic_accuracy", "background_composition_accuracy", "elapsed_seconds"
        )}), flush=True)
    save_json(args.output_dir / f"seed{args.seed}_summary.json", {"protocol": protocol, "cells": rows})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("pretrain")
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument("--width", type=int, default=256); a.add_argument("--heads", type=int, default=8)
    a.add_argument("--mlp-ratio", type=int, default=4); a.add_argument("--layers", type=int, default=8)
    a.add_argument("--aliases", type=int, default=3); a.add_argument("--steps", type=int, default=5000)
    a.add_argument("--batch-size", type=int, default=512); a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--weight-decay", type=float, default=0.1); a.add_argument("--eval-every", type=int, default=1000)
    a.add_argument("--seed", type=int, default=31)
    b = sub.add_parser("cells")
    b.add_argument("--checkpoint", type=Path, required=True); b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--masks", default="all"); b.add_argument("--finetune-steps", type=int, default=300)
    b.add_argument("--finetune-lr", type=float, default=3e-2)
    b.add_argument("--finetune-batch-size", type=int, default=128); b.add_argument("--seed", type=int, default=71)
    b.add_argument("--tune-scope", choices=("adapter", "embedding", "full"), default="embedding")
    c = sub.add_parser("scratch-cells")
    c.add_argument("--output-dir", type=Path, required=True); c.add_argument("--masks", default="all")
    c.add_argument("--width", type=int, default=256); c.add_argument("--heads", type=int, default=8)
    c.add_argument("--mlp-ratio", type=int, default=4); c.add_argument("--layers", type=int, default=8)
    c.add_argument("--aliases", type=int, default=3); c.add_argument("--steps", type=int, default=3000)
    c.add_argument("--supervision", choices=("direct", "cot"), default="direct")
    c.add_argument("--masks-file", type=Path)
    c.add_argument("--nested-output", action="store_true")
    c.add_argument("--batch-size", type=int, default=512); c.add_argument("--lr", type=float, default=1e-3)
    c.add_argument("--weight-decay", type=float, default=.1); c.add_argument("--eval-every", type=int, default=500)
    c.add_argument("--seed", type=int, default=31)
    args = p.parse_args()
    if args.command == "pretrain": pretrain(args)
    elif args.command == "cells": cells(args)
    else: scratch_cells(args)


if __name__ == "__main__":
    main()
