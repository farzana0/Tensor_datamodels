#!/usr/bin/env python3
"""Coalition game of the Qwen representation-probe experiment (Appendix J.9, J.11).

Ten groups of training examples are defined by prompt template. Three groups
carry flipped training labels. For every coalition of groups, a logistic-
regression probe is refitted on fixed 32-dimensional projections of frozen
Qwen2.5-1.5B-Instruct activations, and the utility is log 2 minus the mean
true-label log-loss on 112 held-out examples (unseen statements and unseen
templates). The language model is never updated.

    python experiments/qwen_probe/probe_game.py --seed 31           # 1,024 probe refits, seconds on CPU
    python experiments/qwen_probe/probe_game.py --seed 31 --check   # compare with the saved outcomes

The seed sets the randomized PCA and the probe solver. Saved outcomes for seeds
31, 43 and 59 are in ``results/``.

Linear algebra runs on a single BLAS thread, as it did for the saved outcomes.
The thread count changes the rounding of the projection in the last bits, and
the probe solver, which stops after about ten iterations, turns that into
visible changes in a minority of coalition outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from tndm.attribution import exact  # noqa: E402

INPUTS = HERE / "data" / "probe_inputs.npz"
DIMENSION = 10
CORRUPTED_GROUPS = (2, 6, 9)
PROJECTION_DIMENSION = 32
PROBE_C = 1.0


@threadpool_limits.wrap(limits=1, user_api="blas")
def prepare(seed: int, inputs: Path = INPUTS) -> dict:
    """Group rows by template, project the activations once, and flip the labels of the corrupted groups."""
    archive = np.load(inputs, allow_pickle=False)
    labels = archive["labels"].astype(np.int64)
    group_of_row, held_out = archive["group"], archive["held_out"]
    group_rows = []
    for group in range(DIMENSION):
        selected = np.flatnonzero(group_of_row == group)
        if not len(selected) or set(labels[selected].tolist()) != {0, 1}:
            raise ValueError(f"group {group} is empty or does not contain both labels")
        group_rows.append(selected)
    training = np.concatenate(group_rows)
    raw = archive["activations"].astype(np.float64)
    scaler = StandardScaler().fit(raw[training])
    pca = PCA(n_components=PROJECTION_DIMENSION, svd_solver="randomized", random_state=seed)
    pca.fit(scaler.transform(raw[training]))
    training_labels = labels.copy()
    for group in CORRUPTED_GROUPS:
        training_labels[group_rows[group]] = 1 - training_labels[group_rows[group]]
    return {"features": pca.transform(scaler.transform(raw)), "true_labels": labels,
            "training_labels": training_labels, "group_rows": group_rows,
            "test_rows": np.flatnonzero(held_out)}


@threadpool_limits.wrap(limits=1, user_api="blas")
def coalition_utilities(data: dict, seed: int) -> np.ndarray:
    """Utility of every coalition; index = bitmask with bit i set when group i is included."""
    utility = np.zeros(1 << DIMENSION)
    x_test = data["features"][data["test_rows"]]
    y_test = data["true_labels"][data["test_rows"]]
    empty_loss = float(log_loss(y_test, np.full(len(y_test), 0.5), labels=[0, 1]))
    for mask in range(1, 1 << DIMENSION):
        rows = np.concatenate([data["group_rows"][g] for g in range(DIMENSION) if mask & (1 << g)])
        probe = LogisticRegression(C=PROBE_C, solver="liblinear", max_iter=1000, random_state=seed)
        probe.fit(data["features"][rows], data["training_labels"][rows])
        utility[mask] = empty_loss - float(log_loss(y_test, probe.predict_proba(x_test)[:, 1], labels=[0, 1]))
    return utility


def exact_attributions(utility: np.ndarray) -> dict[str, np.ndarray]:
    """Reference attributions from the complete game."""
    pair = np.zeros((DIMENSION, DIMENSION))
    for (i, j), value in zip(exact.pairs(DIMENSION), np.ravel(exact.pair_interactions(utility))):
        pair[i, j] = pair[j, i] = value
    return {"shapley": np.ravel(exact.shapley_values(utility)), "banzhaf": np.ravel(exact.banzhaf_values(utility)),
            "loo": np.ravel(exact.loo_values(utility)), "pair_interactions": pair}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--check", action="store_true", help="compare with results/probe_d10_seed<seed>.json")
    parser.add_argument("--output", type=Path, help="write utilities and exact attributions to this JSON file")
    args = parser.parse_args()
    utility = coalition_utilities(prepare(args.seed), args.seed)
    attributions = exact_attributions(utility)
    print(f"seed {args.seed}: utility of the full coalition {utility[-1]:.6f}; "
          f"Shapley values {np.round(attributions['shapley'], 3).tolist()}")
    if args.output:
        args.output.write_text(json.dumps({"seed": args.seed, "utility": utility.tolist(),
                                           **{k: v.tolist() for k, v in attributions.items()}}, indent=1) + "\n")
    if args.check:
        saved = json.loads((HERE / "results" / f"probe_d10_seed{args.seed}.json").read_text())
        digest = hashlib.sha256(np.ascontiguousarray(utility).tobytes()).hexdigest()
        gaps = {"utility": np.abs(utility - np.asarray(saved["utility"])).max(),
                "pair_interactions": np.abs(attributions["pair_interactions"]
                                            - np.asarray(saved["exact_pair_interactions"])).max()}
        for name in ("shapley", "banzhaf", "loo"):
            gaps[name] = np.abs(attributions[name] - np.asarray(saved["exact_first_order"][name])).max()
        changed = int((np.abs(utility - np.asarray(saved["utility"])) > 1e-6).sum())
        print("bitwise identical to the saved outcomes" if digest == saved["utility_sha256"]
              else f"{changed} of {len(utility)} coalition outcomes differ from the saved ones by more than 1e-6")
        print("maximum absolute differences:", {k: float(f"{v:.3g}") for k, v in gaps.items()})
        # The probe solver stops at a loose tolerance, so some coalition outcomes can move slightly on other
        # processors or linear-algebra libraries.
        return 0 if gaps["utility"] < 0.05 and max(gaps[k] for k in ("shapley", "banzhaf", "loo")) < 5e-3 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
