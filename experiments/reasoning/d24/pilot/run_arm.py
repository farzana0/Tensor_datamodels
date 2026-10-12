#!/usr/bin/env python3
"""Run one pilot arm: (k, g, d, noise multiplier, replicate) over all budgets.

The design is frozen before any fit: the 1024-cell coalition sample is drawn once
and every smaller budget is a prefix of it, so budgets are nested and outcome-free.
Predictions are evaluated against the NOISELESS ideal game on a disjoint uniform
test set.  
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import game as G
import models as M

BUDGETS = (32, 64, 128, 256, 512, 1024)
TEST_SIZE = 4096
DISTILL_SIZE = 2048
SHAPLEY_PERMUTATIONS = 256
SELECTION_SAMPLE = 20000
Q = 10


def to_builtin(value):
    """JSON fallback for numpy scalars that leak into choices or incidence."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def sha256_of(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def permutation_masks(d: int, seed: int, permutations: int) -> tuple[np.ndarray, np.ndarray]:
    """Growing-prefix masks for Monte-Carlo Shapley, shared by every method."""
    rng = np.random.default_rng(seed)
    orders = np.vstack([rng.permutation(d) for _ in range(permutations)])
    masks = np.zeros((permutations, d + 1), dtype=np.int64)
    current = np.zeros(permutations, dtype=np.int64)
    for position in range(d):
        current = current | (np.int64(1) << orders[:, position].astype(np.int64))
        masks[:, position + 1] = current
    return orders, masks


def permutation_shapley(model, d: int, q: int, orders: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Monte-Carlo Shapley with common random numbers across methods."""
    values = model.predict_chunked(masks.reshape(-1)).reshape(masks.shape[0], d + 1, q)
    contributions = np.diff(values, axis=1)  # (permutations, d, Q)
    total = np.zeros((q, d))
    for permutation in range(masks.shape[0]):
        total[:, orders[permutation]] += contributions[permutation].T
    return total / masks.shape[0]


def loo_scores(model, d: int, q: int) -> np.ndarray:
    full = (1 << d) - 1
    masks = np.asarray([full] + [full & ~(1 << group) for group in range(d)], dtype=np.int64)
    values = model.predict_chunked(masks)
    return (values[0][:, None] - values[1:].T)


def exact_loo(ideal: G.Game) -> np.ndarray:
    full = (1 << ideal.d) - 1
    masks = np.asarray([full] + [full & ~(1 << group) for group in range(ideal.d)], dtype=np.int64)
    values = ideal.values(G.bits_of(masks, ideal.d))
    return (values[0][:, None] - values[1:].T)


def triple_interactions(predict, d: int, q: int, triples: np.ndarray) -> np.ndarray:
    """Third finite difference at the empty background for each candidate triple."""
    unique: dict[int, int] = {}
    rows = []
    for triple in triples:
        for subset in range(8):
            mask = 0
            for bit in range(3):
                if subset >> bit & 1:
                    mask |= 1 << int(triple[bit])
            if mask not in unique:
                unique[mask] = len(rows); rows.append(mask)
    masks = np.asarray(rows, dtype=np.int64)
    values = predict(masks)
    out = np.zeros((q, len(triples)))
    signs = np.asarray([(-1) ** (3 - bin(s).count("1")) for s in range(8)], dtype=float)
    for index, triple in enumerate(triples):
        accumulator = np.zeros(q)
        for subset in range(8):
            mask = 0
            for bit in range(3):
                if subset >> bit & 1:
                    mask |= 1 << int(triple[bit])
            accumulator += signs[subset] * values[unique[mask]]
        out[:, index] = accumulator
    return out


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    hits = labels[order].astype(float)
    if hits.sum() == 0:
        return float("nan")
    cumulative = np.cumsum(hits)
    precision = cumulative / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def stage(label: str, start: float) -> float:
    now = time.time()
    print(f"    [{label}] {now - start:.1f}s", flush=True)
    return now


def build_models(ideal: G.Game, masks: np.ndarray, y_noisy: np.ndarray, split_seed: int,
                 distill_masks: np.ndarray) -> tuple[dict, dict]:
    d, q = ideal.d, ideal.q
    fit_masks, val_masks = M.stratified_split(masks, split_seed)
    index = {int(mask): position for position, mask in enumerate(masks)}
    y_fit = y_noisy[[index[int(m)] for m in fit_masks]]
    y_val = y_noisy[[index[int(m)] for m in val_masks]]
    weight = M.size_weights(fit_masks)

    mark = time.time()
    base, base_choice = M.fit_cardinality(fit_masks, y_fit, val_masks, y_val, weight, d)
    res_fit, res_val = y_fit - base(fit_masks), y_val - base(val_masks)

    fitted: dict[str, M.Model] = {}
    choices: dict[str, dict] = {"cardinality": base_choice}

    class Card(M.Model):
        name = "cardinality"

        def predict(self, masks_in: np.ndarray) -> np.ndarray:
            return base(masks_in)

    fitted["cardinality"] = Card()
    mark = stage("cardinality", mark)
    for degree in (1, 2, 3):
        model, choice = M.fit_polynomial(degree, fit_masks, res_fit, val_masks, res_val, weight, d, base)
        fitted[model.name], choices[model.name] = model, choice
    mark = stage("polynomial", mark)
    model, choice = M.fit_omp(3, fit_masks, res_fit, val_masks, res_val, d, base)
    fitted[model.name], choices[model.name] = model, choice
    mark = stage("omp", mark)
    model, choice = M.fit_rbf(fit_masks, res_fit, val_masks, res_val, d, base)
    fitted[model.name], choices[model.name] = model, choice

    mark = stage("rbf", mark)
    role_order = ideal.role_ordering()
    shuffled = tuple(np.random.default_rng(99).permutation(d).tolist())
    for family, orders in (("tt", (role_order, shuffled)), ("cp", (role_order,)), ("ht", (role_order,))):
        model, choice = M.fit_tn_family(family, d, q, fit_masks, res_fit, val_masks, res_val,
                                        weight, base, orders)
        fitted[model.name], choices[model.name] = model, choice
        mark = stage(family, mark)

    model, choice = M.fit_ladder(d, q, fit_masks, res_fit, val_masks, res_val, weight, base,
                                 role_order, max_degree=min(ideal.k * ideal.g, 6),
                                 distill_masks=distill_masks)
    fitted[model.name], choices[model.name] = model, choice
    stage("ladder", mark)
    return fitted, choices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--g", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="*", default=list(BUDGETS))
    args = parser.parse_args()

    started = time.time()
    ideal = G.balanced_incidence(args.k, args.g, args.d, Q, seed=1000 + args.replicate)
    design_rng = np.random.default_rng(2000 + args.replicate)
    noise_rng = np.random.default_rng(3000 + args.replicate)

    all_masks = G.uniform_masks(args.d, max(args.budgets), design_rng)
    test_masks = G.uniform_masks(args.d, TEST_SIZE, design_rng, exclude=set(all_masks.tolist()))
    distill_masks = G.uniform_masks(args.d, DISTILL_SIZE, design_rng)
    design_hash = sha256_of(all_masks)

    truth_all = ideal.values(G.bits_of(all_masks, args.d))
    noisy_all = ideal.noise(truth_all, noise_rng, args.noise)
    truth_test = ideal.values(G.bits_of(test_masks, args.d))
    variance = truth_test.var(0)
    variance[variance < 1e-12] = 1.0

    exact_shapley = G.exact_shapley(ideal)
    exact_banzhaf = G.exact_banzhaf(ideal)
    exact_loo_values = exact_loo(ideal)

    # Candidate triples for the interaction metric: all triples inside the union of
    # relevant groups plus a random sample of the rest, fixed before fitting.
    relevant = sorted({group for query in range(ideal.q) for group in ideal.relevant(query)})
    triple_rng = np.random.default_rng(4000 + args.replicate)
    candidate = set()
    for i in range(len(relevant)):
        for j in range(i + 1, len(relevant)):
            for l in range(j + 1, len(relevant)):
                candidate.add((relevant[i], relevant[j], relevant[l]))
    candidate = sorted(candidate)
    if len(candidate) > 4000:
        picked = triple_rng.choice(len(candidate), 4000, replace=False)
        candidate = [candidate[int(p)] for p in picked]
    triples = np.asarray(candidate, dtype=int)
    # A true positive triple is a complete path: exactly one provider from each role
    # (only meaningful when k == 3; for other k the triple metric is reported as NaN).
    triple_labels = np.zeros((ideal.q, len(triples)), dtype=bool)
    if ideal.k == 3:
        for query, roles in enumerate(ideal.queries):
            role_of = {group: index for index, role in enumerate(roles) for group in role}
            for index, triple in enumerate(triples):
                covered = [role_of.get(int(group)) for group in triple]
                triple_labels[query, index] = (None not in covered and len(set(covered)) == 3)

    selection_rng = np.random.default_rng(5000 + args.replicate)
    selection_sets = {}
    for size in (ideal.k, ideal.k + 1):
        per_query = []
        for query in range(ideal.q):
            masks_s, complete = G.complete_sets(ideal, query, size, selection_rng, SELECTION_SAMPLE)
            per_query.append((masks_s, complete))
        selection_sets[size] = per_query

    shapley_orders, shapley_masks = permutation_masks(args.d, 7000 + args.replicate,
                                                      SHAPLEY_PERMUTATIONS)

    # Per-budget checkpoint: a requeued task resumes from the last finished budget.
    # The design is a deterministic function of the replicate seed, so a checkpoint
    # is reused only when its design hash matches the one regenerated here.
    partial = args.out.with_name(args.out.name + ".partial")
    results = []
    if partial.exists():
        saved = json.loads(partial.read_text())
        if saved.get("design_sha256") == design_hash:
            results = saved["budgets"]
            print(f"resuming from {partial}: budgets {[r['budget'] for r in results]}", flush=True)
    finished = {r["budget"] for r in results}
    for budget in args.budgets:
        if budget in finished:
            continue
        masks = all_masks[:budget]
        y_noisy = noisy_all[:budget]
        fit_started = time.time()
        fitted, choices = build_models(ideal, masks, y_noisy, 6000 + args.replicate, distill_masks)
        rows = {}
        for name, model in fitted.items():
            metric_started = time.time()
            prediction = model.predict_chunked(test_masks)
            nmse = float(np.mean(np.mean((prediction - truth_test) ** 2, axis=0) / variance))

            shapley = permutation_shapley(model, args.d, ideal.q, shapley_orders, shapley_masks)
            shapley_error = float(np.linalg.norm(shapley - exact_shapley) /
                                  max(np.linalg.norm(exact_shapley), 1e-12))
            loo = loo_scores(model, args.d, ideal.q)
            loo_error = float(np.linalg.norm(loo - exact_loo_values) /
                              max(np.linalg.norm(exact_loo_values), 1e-12))

            interactions = triple_interactions(model.predict_chunked, args.d, ideal.q, triples)
            if ideal.k == 3:
                aps = [average_precision(interactions[q], triple_labels[q]) for q in range(ideal.q)]
                triple_ap = float(np.nanmean(aps))
            else:
                triple_ap = float("nan")

            selection = {}
            for size, per_query in selection_sets.items():
                completeness = []
                for query, (masks_s, complete) in enumerate(per_query):
                    scores = model.predict_chunked(masks_s)[:, query]
                    top = np.argsort(-scores, kind="stable")[:3]
                    completeness.append(float(complete[top].mean()))
                selection[f"top3_complete_size{size}"] = float(np.mean(completeness))

            print(f"    [metrics {name}] {time.time() - metric_started:.1f}s", flush=True)
            rows[name] = {"nmse": nmse, "shapley_relative_error": shapley_error,
                          "loo_relative_error": loo_error, "triple_interaction_ap": triple_ap,
                          **selection, "choice": choices.get(name, {})}
        results.append({"budget": budget, "metrics": rows,
                        "fit_seconds": time.time() - fit_started})
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps({"design_sha256": design_hash, "budgets": results},
                                      sort_keys=True, allow_nan=True, default=to_builtin) + "\n")
        print(json.dumps({"budget": budget, "seconds": round(time.time() - fit_started, 1),
                          "nmse": {n: round(r["nmse"], 4) for n, r in rows.items()}}), flush=True)

    payload = {
        "arm": {"k": args.k, "g": args.g, "d": args.d, "noise_multiplier": args.noise,
                "replicate": args.replicate, "queries": ideal.q},
        "incidence": [[[int(group) for group in role] for role in roles] for roles in ideal.queries],
        "design_sha256": design_hash,
        "test_sha256": sha256_of(test_masks),
        "moebius_terms_per_query": (2 ** args.g - 1) ** args.k,
        "polynomial_degree": args.k * args.g,
        "complete_fraction_uniform": float((truth_test > G.BASE + G.JUMP / 2).mean()),
        "budgets": results,
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True, default=to_builtin) + "\n")
    partial.unlink(missing_ok=True)
    print(f"wrote {args.out} in {payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
