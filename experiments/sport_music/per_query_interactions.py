"""Do different query types have different important interactions?

The claim under test: the most important interaction for one type of query is not the same
as for another, so a single interaction ranking computed on a query-averaged utility is a
blend of several different rankings and describes none of them.

Sport-Music is built for this. There are 5 query classes; class c is answered by composing
A_c (subject c's sport) with B_c (subject c's genre). So the ground-truth prediction is
sharp: class c's strongest pair interaction should be exactly (A_c, B_c), a DIFFERENT pair
for each class, and the query-averaged interaction should show all five equally.

PART 1 is model-free. The design contains the empty coalition, all 10 singletons, the 5
matched pairs (A_i, B_i) and 5 mismatched pairs (A_i, B_{i+1 mod 5}). So the second-order
interaction at the empty background

    I_ij = v({i,j}) - v({i}) - v({j}) + v(empty)

is measured directly from fine-tunes, per class, with no surrogate involved.

PART 2 asks whether a fitted datamodel recovers the right pair per class, and whether the
query-averaged fit -- the one this project has been reporting -- can recover any of them.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
RES = ROOT / 'experiments/sport_music/results'
NAMES = [f'A_{i}' for i in range(5)] + [f'B_{i}' for i in range(5)]
N, NC = 10, 5
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
PAIRS = list(itertools.combinations(range(N), 2))


def load(metric):
    rows = []
    for d in sorted(RES.glob('coalition_*')):
        f = d / 'UTILITY.json'
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if r.get('status') != 'COMPLETE' or not r.get('converged') or r['seed'] != 0:
            continue
        pc = r.get('per_class')
        if not pc:
            continue
        rows.append({'mask': np.array(r['mask'], float), 'kind': r['kind'],
                     'y': np.array([float(pc[str(c)][metric]) for c in range(NC)])})
    return rows


# ---------------------------------------------------------------- Part 1: measured
def measured(rows):
    """I_ij per class, straight from fine-tunes. No model."""
    empty = next(r['y'] for r in rows if r['kind'] == 'anchor_empty')
    sing = {}
    for r in rows:
        if r['kind'] == 'singleton':
            sing[int(np.where(r['mask'] == 1)[0][0])] = r['y']
    out = {}
    for r in rows:
        if r['kind'] in ('matched_pair', 'mismatched_pair'):
            i, j = (int(x) for x in np.where(r['mask'] == 1)[0])
            out[(i, j)] = (r['y'] - sing[i] - sing[j] + empty, r['kind'])
    return out


# ---------------------------------------------------------------- Part 2: fitted
ALL = np.array([[(m >> i) & 1 for i in range(N)] for m in range(1 << N)], float)
_LF = np.array([0.0] + list(np.cumsum(np.log(np.arange(1, N + 1)))))
_IDX = np.arange(1 << N)
_PC = np.array([bin(m).count('1') for m in _IDX])


def interaction_index(v):
    """Exact 2nd-order Shapley interaction index over all 2^10 masks."""
    out = np.zeros((N, N))
    for i, j in PAIRS:
        bi, bj = 1 << i, 1 << j
        base = _IDX[((_IDX & bi) == 0) & ((_IDX & bj) == 0)]
        d = v[base | bi | bj] - v[base | bi] - v[base | bj] + v[base]
        w = np.exp(_LF[_PC[base]] + _LF[N - _PC[base] - 2] - _LF[N - 1])
        out[i, j] = out[j, i] = float(np.sum(w * d))
    return out


def deg2(Z):
    return np.hstack([np.ones((len(Z), 1)), Z,
                      np.column_stack([Z[:, a] * Z[:, b] for a, b in PAIRS])])


def fit_ridge(Z, y, rng):
    X = deg2(Z)
    o = rng.permutation(len(y))
    k = max(3, len(y) // 4)
    va, tr = o[:k], o[k:]
    best, bw = -np.inf, None
    for lam in RIDGES:
        P = np.eye(X.shape[1]) * lam
        P[0, 0] = 0
        w = np.linalg.solve(X[tr].T @ X[tr] + P, X[tr].T @ y[tr])
        p = X[va] @ w
        d = np.sum((y[va] - y[va].mean()) ** 2)
        s = -np.inf if d == 0 else 1 - np.sum((y[va] - p) ** 2) / d
        if s > best:
            best, bw = s, lam
    P = np.eye(X.shape[1]) * bw
    P[0, 0] = 0
    return np.linalg.solve(X.T @ X + P, X.T @ y)


def main():
    for metric in ('mean_gold_logprob', 'accuracy'):
        rows = load(metric)
        Z = np.array([r['mask'] for r in rows])
        Y = np.array([r['y'] for r in rows])
        meas = measured(rows)
        print(f'\n{"=" * 100}\n=== metric: {metric}   ({len(rows)} coalitions, {NC} query classes)\n{"=" * 100}')

        print('\nPART 1 -- MEASURED interaction I_ij per class, from fine-tunes only (nats)')
        print(f'{"pair":<12}' + ''.join(f'{"class " + str(c):>11}' for c in range(NC)) + f'{"mean":>11}')
        keys = sorted(meas, key=lambda k: (meas[k][1] != 'matched_pair', k))
        for k in keys:
            v, kind = meas[k]
            tag = f'{NAMES[k[0]]},{NAMES[k[1]]}'
            print(f'{tag:<12}' + ''.join(f'{v[c]:>+11.3f}' for c in range(NC)) + f'{v.mean():>+11.3f}')

        print('\n  strongest pair per class (measured):')
        hit = 0
        for c in range(NC):
            best = max(keys, key=lambda k: meas[k][0][c])
            want = (c, c + 5)
            ok = best == want
            hit += ok
            print(f'    class {c}: {NAMES[best[0]]},{NAMES[best[1]]:<4} '
                  f'{meas[best][0][c]:+.3f}   expected {NAMES[c]},{NAMES[c+5]:<4} '
                  f'{meas[want][0][c]:+.3f}   {"HIT" if ok else "miss"}')
        best_avg = max(keys, key=lambda k: meas[k][0].mean())
        print(f'  per-class top-1 recovery: {hit}/{NC}')
        print(f'  strongest pair on the CLASS-AVERAGED utility: '
              f'{NAMES[best_avg[0]]},{NAMES[best_avg[1]]} ({meas[best_avg][0].mean():+.3f}) '
              f'-- one pair stands in for all five')

        print('\nPART 2 -- FITTED degree-2 datamodel, exact interaction index over all 2^10 masks')
        rng = np.random.default_rng(11)
        hit = 0
        for c in range(NC):
            I = interaction_index(deg2(ALL) @ fit_ridge(Z, Y[:, c], rng))
            top = max(PAIRS, key=lambda p: I[p])
            want = (c, c + 5)
            ok = top == want
            hit += ok
            rank = 1 + sorted((-I[p] for p in PAIRS)).index(-I[want])
            print(f'    class {c}: top {NAMES[top[0]]},{NAMES[top[1]]:<4} '
                  f'| expected {NAMES[c]},{NAMES[c+5]:<4} ranked {rank}/45   {"HIT" if ok else "miss"}')
        print(f'  per-class top-1 recovery: {hit}/{NC}')
        Iagg = interaction_index(deg2(ALL) @ fit_ridge(Z, Y.mean(1), rng))
        topa = max(PAIRS, key=lambda p: Iagg[p])
        ranks = [1 + sorted((-Iagg[p] for p in PAIRS)).index(-Iagg[(c, c + 5)]) for c in range(NC)]
        print(f'  CLASS-AVERAGED fit: top pair {NAMES[topa[0]]},{NAMES[topa[1]]}; '
              f'the five true matched pairs rank {ranks} of 45')


if __name__ == '__main__':
    main()
