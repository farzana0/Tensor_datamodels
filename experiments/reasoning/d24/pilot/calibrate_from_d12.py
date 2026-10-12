#!/usr/bin/env python3
"""Calibrate the ideal AND-of-ORs value profile and noise scale from the real d=12 game.

Every observed cell of `15_transformer_query_paths_d12` is one from-scratch training
run of the causal-GPT.  For each (cell, query) pair the role-completion pattern is
known exactly from the group/provider incidence, so cells that share a pattern are
replicates of the same structural configuration.  The spread inside a pattern class
bounds the retraining noise that the pilot must inject; the class means give the
value profile the ideal game should reproduce.

Outputs `analysis/d12_calibration.json` and a per-cell table used by the pilot.
"""
from __future__ import annotations

import os
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

R = Path(os.environ.get("TNDM_ORIGINAL_CHECKOUT", "."))  # original research checkout (provenance tool; not needed to reproduce results)
D12 = R / "results/original_records/d12"
HERE = Path(__file__).resolve().parent.parent

D = 12
Q = 10
QUERY_BLOCKS = (
    (0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11),
    (0, 2, 4, 6, 8, 10), (1, 3, 5, 7, 9, 11),
    (0, 1, 4, 5, 8, 9), (2, 3, 6, 7, 10, 11),
    (0, 3, 4, 7, 8, 11), (1, 2, 5, 6, 9, 10),
    (0, 2, 5, 7, 9, 11), (1, 3, 4, 6, 8, 10),
)
PROVIDERS = tuple(tuple((block[i], block[i + 1]) for i in range(0, 6, 2)) for block in QUERY_BLOCKS)


def role_presence(mask: int, query: int) -> tuple[int, ...]:
    """Number of present providers (0, 1 or 2) for each of the three roles."""
    return tuple(sum(1 for group in pair if mask & (1 << group)) for pair in PROVIDERS[query])


def load_cells() -> dict[int, dict]:
    cells: dict[int, dict] = {}
    order = ["gate/seed31", "observed/seed31", "interaction_audit/seed31",
             "target/seed31", "size4_eval/seed31", "size5_eval/seed31"]
    for sub in order:
        root = D12 / sub
        if not root.exists():
            continue
        for path in sorted(root.glob("mask_*/*.json")):
            row = json.loads(path.read_text())
            mask = int(row["mask"])
            cells.setdefault(mask, {"row": row, "source": sub, "path": path})
    return cells


def main() -> None:
    cells = load_cells()
    if not cells:
        sys.exit("no d=12 cells found")

    records = []
    for mask, entry in cells.items():
        logp = np.asarray(entry["row"]["target_log_probability_by_query"], dtype=float)
        acc = np.asarray(entry["row"]["target_accuracy_by_query"], dtype=float)
        for query in range(Q):
            presence = role_presence(mask, query)
            records.append({
                "mask": mask, "query": query, "cardinality": int(mask).bit_count(),
                "presence": presence, "roles_complete": sum(1 for p in presence if p > 0),
                "providers_present": sum(presence),
                "logp": float(logp[query]), "accuracy": float(acc[query]),
                "source": entry["source"],
            })

    logp = np.asarray([r["logp"] for r in records])
    complete = np.asarray([r["roles_complete"] == 3 for r in records])

    # Value profile: mean and spread per role-completion pattern (sorted presence
    # counts, so (2,1,0) and (0,1,2) are the same structural class).
    profile = {}
    for record in records:
        key = tuple(sorted(record["presence"], reverse=True))
        profile.setdefault(key, []).append(record["logp"])
    profile_rows = []
    for key in sorted(profile, key=lambda k: (sum(1 for p in k if p), sum(k))):
        values = np.asarray(profile[key])
        profile_rows.append({
            "presence_sorted": list(key), "roles_complete": sum(1 for p in key if p),
            "providers_present": int(sum(key)), "n": int(values.size),
            "mean_logp": float(values.mean()), "sd_logp": float(values.std(ddof=1)) if values.size > 1 else None,
            "min_logp": float(values.min()), "max_logp": float(values.max()),
        })

    # Noise bound: spread within an exact pattern class, further conditioned on
    # cardinality so that any smooth interference-with-cardinality effect is not
    # counted as noise.  This is an upper bound on pure retraining noise because
    # provider identity still varies inside a cell of this partition.
    residual_parts = []
    conditioned = {}
    for record in records:
        key = (tuple(sorted(record["presence"], reverse=True)), record["cardinality"])
        conditioned.setdefault(key, []).append(record["logp"])
    for key, values in conditioned.items():
        values = np.asarray(values)
        if values.size >= 2:
            residual_parts.append(values - values.mean())
    residuals = np.concatenate(residual_parts) if residual_parts else np.zeros(0)
    dof = sum(len(v) - 1 for v in conditioned.values() if len(v) >= 2)
    noise_sd = float(np.sqrt((residuals ** 2).sum() / max(dof, 1)))

    complete_logp = logp[complete]
    incomplete_logp = logp[~complete]
    complete_residuals, complete_dof = [], 0
    for key, values in conditioned.items():
        if sum(1 for p in key[0] if p) == 3 and len(values) >= 2:
            complete_residuals.append(np.asarray(values) - np.mean(values))
            complete_dof += len(values) - 1
    complete_noise = (float(np.sqrt((np.concatenate(complete_residuals) ** 2).sum() / max(complete_dof, 1)))
                      if complete_residuals else None)

    # Does the value depend on anything beyond the OR of each role?  Compare a
    # model that only sees "is each role covered" against one that sees how many
    # providers are present per role, and against provider identity.
    def group_var(keyfunc) -> tuple[float, int]:
        groups: dict = {}
        for record in records:
            groups.setdefault(keyfunc(record), []).append(record["logp"])
        parts = [np.asarray(v) - np.mean(v) for v in groups.values() if len(v) >= 2]
        d = sum(len(v) - 1 for v in groups.values() if len(v) >= 2)
        if not parts:
            return float("nan"), 0
        res = np.concatenate(parts)
        return float(np.sqrt((res ** 2).sum() / max(d, 1))), len(groups)

    or_only_sd, or_only_groups = group_var(lambda r: tuple(p > 0 for p in r["presence"]))
    count_sd, count_groups = group_var(lambda r: tuple(sorted(r["presence"], reverse=True)))
    full_sd, full_groups = group_var(lambda r: (tuple(sorted(r["presence"], reverse=True)), r["cardinality"]))

    # Cardinality (interference) effect among fully complete query cells.
    card_rows = []
    for card in sorted({r["cardinality"] for r in records}):
        values = np.asarray([r["logp"] for r in records
                             if r["cardinality"] == card and r["roles_complete"] == 3])
        if values.size:
            card_rows.append({"cardinality": card, "n": int(values.size),
                              "mean_logp": float(values.mean()),
                              "sd_logp": float(values.std(ddof=1)) if values.size > 1 else None})

    payload = {
        "source": str(D12),
        "cells": len(cells),
        "cell_query_records": len(records),
        "value_profile_by_role_pattern": profile_rows,
        "complete_mean_logp": float(complete_logp.mean()),
        "complete_sd_logp": float(complete_logp.std(ddof=1)),
        "incomplete_mean_logp": float(incomplete_logp.mean()),
        "incomplete_sd_logp": float(incomplete_logp.std(ddof=1)),
        "noise_sd_within_pattern_and_cardinality": noise_sd,
        "noise_sd_complete_cells_only": complete_noise,
        "residual_sd_or_only_model": {"sd": or_only_sd, "classes": or_only_groups},
        "residual_sd_provider_count_model": {"sd": count_sd, "classes": count_groups},
        "residual_sd_provider_count_plus_cardinality_model": {"sd": full_sd, "classes": full_groups},
        "cardinality_effect_complete_cells": card_rows,
        "note": ("noise_sd_* are upper bounds on pure retraining noise: only one learner "
                 "seed exists, so replicates differ in provider identity and in which "
                 "irrelevant groups are present, not in training seed."),
    }
    out = HERE / "analysis/d12_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    out.write_text(text)
    print(text)
    print("sha256", hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
