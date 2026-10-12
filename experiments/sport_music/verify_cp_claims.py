"""Audit: are the CP query-mode claims stable, or artefacts of one lucky seed?

Three claims were made from SINGLE fits and are load-bearing for the write-up:
  C1  the query factor U[class, term] comes out a permutation (each class loads on one term)
  C2  4 of 5 terms put both A_c and B_c in their top-3 weights
  C3  per-class interaction recovery is 4/5 top-1 of 45 pairs
and one from a single sweep:
  C4  held-out R2 peaks at rank 5, the true number of classes

CP fitting is non-convex with random initialisation, so each is a random variable. This
refits across seeds and reports the DISTRIBUTION. A claim that holds on 1 of 20 seeds is not
a finding.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'experiments/sport_music/results'
_s = importlib.util.spec_from_file_location('cpq', Path(__file__).parent / 'cp_query_mode.py')
CPQ = importlib.util.module_from_spec(_s)
_s.loader.exec_module(CPQ)
NAMES, PAIRS, ALL = CPQ.NAMES, CPQ.PAIRS, CPQ.ALL
torch.set_num_threads(4)
NSEED = 20


def permutation_score(U):
    """1.0 iff every class's argmax term is distinct (a permutation); also the margin."""
    top = np.argmax(np.abs(U), axis=1)
    distinct = len(set(top.tolist())) == U.shape[0]
    margins = []
    for c in range(U.shape[0]):
        a = np.abs(U[c])
        srt = np.sort(a)[::-1]
        margins.append(float(srt[1] / (srt[0] + 1e-12)))
    return distinct, float(np.mean(margins)), top


def pair_recovery(W, U):
    """For each class, does its selected term carry both A_c and B_c in its top 3 |w|?"""
    top = np.argmax(np.abs(U), axis=1)
    hits = 0
    for c in range(5):
        r = top[c]
        t3 = set(np.argsort(-np.abs(W[:, r]))[:3].tolist())
        hits += (c in t3) and (c + 5 in t3)
    return hits


def interaction_hits(m, mu, sd):
    with torch.no_grad():
        V = m(torch.tensor(ALL, dtype=torch.float32)).numpy() * sd + mu
    hits, ranks = 0, []
    for c in range(5):
        I = CPQ.interaction_index(V[:, c])
        top = max(PAIRS, key=lambda p: I[p])
        want = (c, c + 5)
        hits += top == want
        ranks.append(1 + sorted((-I[p] for p in PAIRS)).index(-I[want]))
    return hits, ranks


def main():
    Z, Y = CPQ.load('accuracy')
    out = {}
    print(f'=== auditing CP query-mode claims over {NSEED} random initialisations ===\n')

    perm_ok, margins, pair_hits, int_hits, all_ranks = [], [], [], [], []
    for s in range(NSEED):
        m, mu, sd = CPQ.fit_cp(Z, Y, 5, epochs=6000, seed=s)
        U = m.U.detach().numpy()
        W = m.w.detach().numpy()
        d, mg, _ = permutation_score(U)
        perm_ok.append(d)
        margins.append(mg)
        pair_hits.append(pair_recovery(W, U))
        h, r = interaction_hits(m, mu, sd)
        int_hits.append(h)
        all_ranks.append(r)
    print(f'C1  query factor is a permutation: {sum(perm_ok)}/{NSEED} seeds'
          f'   (mean 2nd/1st loading ratio {np.mean(margins):.2f})')
    print(f'C2  terms carrying BOTH A_c and B_c in top-3: mean {np.mean(pair_hits):.2f}/5'
          f'  (min {min(pair_hits)}, max {max(pair_hits)})')
    print(f'C3  interaction top-1 recovery: mean {np.mean(int_hits):.2f}/5'
          f'  (min {min(int_hits)}, max {max(int_hits)});'
          f'  >=4 on {sum(h >= 4 for h in int_hits)}/{NSEED} seeds')
    R = np.array(all_ranks)
    print(f'    per-class median true-pair rank of 45: {np.median(R, axis=0).astype(int).tolist()}')
    out['C1_permutation_frac'] = float(np.mean(perm_ok))
    out['C2_pair_hits'] = [int(x) for x in pair_hits]
    out['C3_int_hits'] = [int(x) for x in int_hits]
    out['C3_ranks_median'] = np.median(R, axis=0).tolist()

    print(f'\nC4  held-out R2 by rank, {NSEED // 2} seeds x 20 splits:')
    n = len(Z)
    rng = np.random.default_rng(4)
    splits = [rng.permutation(n) for _ in range(20)]
    n_te = max(4, int(round(0.30 * n)))
    curve = {}
    for rank in (1, 2, 3, 4, 5, 6, 8):
        sc = []
        for si, perm in enumerate(splits):
            te, tr = perm[:n_te], perm[n_te:]
            m, mu, sd = CPQ.fit_cp(Z[tr], Y[tr], rank, epochs=1500, seed=si % (NSEED // 2))
            sc.append(np.nanmean(CPQ.r2_cols(Y[te], CPQ.predict(m, Z[te], mu, sd))))
        curve[rank] = (float(np.mean(sc)), float(np.std(sc)))
        print(f'    rank {rank}: {np.mean(sc):+.3f} +/- {np.std(sc):.3f}')
    best = max(curve, key=lambda k: curve[k][0])
    print(f'    peak at rank {best}   (claim was 5 = number of query classes)')
    out['C4_curve'] = curve
    (RES / 'CP_CLAIMS_AUDIT.json').write_text(json.dumps(out, indent=2))
    print(f'\nwrote {RES / "CP_CLAIMS_AUDIT.json"}')


if __name__ == '__main__':
    main()
