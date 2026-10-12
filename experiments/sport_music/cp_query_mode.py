"""CP with a QUERY MODE: v(S, q) = b(q) + sum_r u_r(q) * prod_{i in S} (1 + w_ir).

The motivation is structural, not just predictive. Sport-Music has 5 query classes and class
c is answered by composing A_c with B_c, so the ground-truth tensor is a sum of 5 rank-1
terms, each selected by exactly one class. A CP model with a query mode can express that
directly: term r carries the coalition structure, and u_r(q) says which queries use it.

If the task structure is real and recoverable, a rank-5 fit should discover, unsupervised:
  - u_r(q) approximately diagonal -- each class loads on its own term;
  - term r's weights w_ir concentrated on groups A_r and B_r.

Nothing here needs new fine-tuning. Everything is fitted on the frozen 88 coalitions, and
scored against baselines from multioutput_study.py under the same protocol: per-question R2,
never a question-averaged utility.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
RES = ROOT / 'experiments/sport_music/results'
NAMES = [f'A_{i}' for i in range(5)] + [f'B_{i}' for i in range(5)]
N, NC = 10, 5
PAIRS = list(itertools.combinations(range(N), 2))
torch.set_num_threads(4)


def load(metric):
    Z, Y = [], []
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
        Z.append(np.array(r['mask'], float))
        Y.append([float(pc[str(c)][metric]) for c in range(NC)])
    return np.array(Z), np.array(Y)


class CPQuery(torch.nn.Module):
    """rank-R CP over 10 binary coalition legs plus one query leg of size Q."""

    def __init__(self, rank, n=N, q=NC, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(0.15 * torch.randn(n, rank, generator=g))
        self.U = torch.nn.Parameter(0.15 * torch.randn(q, rank, generator=g))
        self.b = torch.nn.Parameter(torch.zeros(q))

    def forward(self, Z):                      # Z (n_samples, 10) -> (n_samples, Q)
        P = torch.prod(1.0 + Z.unsqueeze(-1) * self.w.unsqueeze(0), dim=1)   # (n, R)
        return self.b.unsqueeze(0) + P @ self.U.T


def fit_cp_best(Z, Y, rank, l2=1e-3, epochs=2000, lr=0.05, restarts=8, seed=0, val_frac=0.25):
    """Best-of-N restarts, selected on a held-out inner split.

    THE COST MODEL MATTERS. One sample here is one full fine-tune (~3 GPU-min at 7B); fitting
    a tensor network to the samples already collected costs seconds of CPU. So a single
    random initialisation of a non-convex CP fit is the wrong protocol -- it throws away
    accuracy that is free to recover. Restarts, rank sweeps and alternative contractions are
    all free relative to the data they are fitted to, and a more accurate model at the same
    number of fine-tunes is strictly better.

    Selection is on a held-out inner split rather than training loss, so extra restarts
    cannot overfit: they can only find a better optimum of the same objective.
    """
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = len(Z)
    k = max(3, int(round(val_frac * n)))
    rng = np.random.default_rng(seed)
    o = rng.permutation(n)
    va, tr = o[:k], o[k:]
    best = (-np.inf, None)
    for r in range(restarts):
        m, mu, sd = fit_cp(Z[tr], Y[tr], rank, l2=l2, epochs=epochs, lr=lr, seed=1000 * seed + r)
        sc = np.nanmean(r2_cols(Y[va], predict(m, Z[va], mu, sd)))
        if np.isfinite(sc) and sc > best[0]:
            best = (sc, 1000 * seed + r)
    # refit the winning initialisation on everything
    return fit_cp(Z, Y, rank, l2=l2, epochs=epochs, lr=lr, seed=best[1])


def fit_cp(Z, Y, rank, l2=1e-3, epochs=4000, lr=0.05, seed=0):
    """Y is (n, Q). Q is read from the data so a single-column Y gives a scalar-output CP,
    which is what fitting one independent tensor per query class needs."""
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    Zt = torch.tensor(Z, dtype=torch.float32)
    mu, sd = Y.mean(), Y.std() + 1e-9
    Yt = torch.tensor((Y - mu) / sd, dtype=torch.float32)
    m = CPQuery(rank, q=Y.shape[1], seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    best, bstate = np.inf, None
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((m(Zt) - Yt) ** 2).mean() + l2 * (m.w.pow(2).sum() + m.U.pow(2).sum())
        loss.backward()
        opt.step()
        v = loss.detach().item()
        if v < best:
            best, bstate = v, {k: p.detach().clone() for k, p in m.state_dict().items()}
    m.load_state_dict(bstate)
    return m, mu, sd


def predict(m, Z, mu, sd):
    with torch.no_grad():
        return m(torch.tensor(Z, dtype=torch.float32)).numpy() * sd + mu


def predict_fit(fit, Z):
    """Predict from a (model, mu, sd) triple as returned by fit_cp.

    Exists because `predict(*fit_cp(...), Z)` silently passes mu into the Z slot -- the
    signatures line up positionally but mean different things, so it fails late with a
    broadcast error rather than at the call site.
    """
    m, mu, sd = fit
    return predict(m, Z, mu, sd)


def r2_cols(Y, P):
    out = []
    for q in range(Y.shape[1]):
        d = float(np.sum((Y[:, q] - Y[:, q].mean()) ** 2))
        out.append(np.nan if d == 0 else float(1 - np.sum((Y[:, q] - P[:, q]) ** 2) / d))
    return np.array(out, float)


ALL = np.array([[(mm >> i) & 1 for i in range(N)] for mm in range(1 << N)], float)
_LF = np.array([0.0] + list(np.cumsum(np.log(np.arange(1, N + 1)))))
_IDX = np.arange(1 << N)
_PC = np.array([bin(mm).count('1') for mm in _IDX])


def interaction_index(v):
    out = np.zeros((N, N))
    for i, j in PAIRS:
        bi, bj = 1 << i, 1 << j
        base = _IDX[((_IDX & bi) == 0) & ((_IDX & bj) == 0)]
        d = v[base | bi | bj] - v[base | bi] - v[base | bj] + v[base]
        w = np.exp(_LF[_PC[base]] + _LF[N - _PC[base] - 2] - _LF[N - 1])
        out[i, j] = out[j, i] = float(np.sum(w * d))
    return out


def main():
    for metric in ('accuracy', 'mean_gold_logprob'):
        Z, Y = load(metric)
        n = len(Z)
        print(f'\n{"=" * 96}\n=== CP with query mode -- metric: {metric}  ({n} coalitions, {NC} classes)\n{"=" * 96}')

        rng = np.random.default_rng(20260912)
        splits = [rng.permutation(n) for _ in range(40)]
        n_te = max(4, int(round(0.30 * n)))
        print(f'\nheld-out per-question R2 (40 shared splits; utility never averaged)')
        print(f'{"model":<28}{"mean":>9}{"median":>9}{"worst q":>9}')
        for rank in (1, 2, 3, 5, 8):
            acc = []
            for perm in splits:
                te, tr = perm[:n_te], perm[n_te:]
                m, mu, sd = fit_cp(Z[tr], Y[tr], rank, epochs=1500)
                acc.append(r2_cols(Y[te], predict(m, Z[te], mu, sd)))
            A = np.nanmean(np.array(acc), 0)
            print(f'{"CP query-mode rank " + str(rank):<28}{np.nanmean(A):>+9.3f}'
                  f'{np.nanmedian(A):>+9.3f}{np.nanmin(A):>+9.3f}')

        # structure discovery + interaction recovery, fitted on everything
        m, mu, sd = fit_cp(Z, Y, 5, epochs=6000)
        U = m.U.detach().numpy()
        W = m.w.detach().numpy()
        print(f'\nrank-5 fit on all {n} coalitions -- did it discover the task structure?')
        print('  query factor U[class, term]  (which term each class loads on):')
        print('        ' + ''.join(f'{"term " + str(r):>9}' for r in range(5)))
        for c in range(NC):
            row = U[c] / (np.abs(U[c]).max() + 1e-12)
            print(f'  cls {c} ' + ''.join(f'{v:>+9.2f}' for v in row))
        print('\n  for each term, the 3 groups with the largest |w|:')
        for r in range(5):
            top = np.argsort(-np.abs(W[:, r]))[:3]
            print(f'    term {r}: ' + ', '.join(f'{NAMES[i]}({W[i, r]:+.2f})' for i in top))

        print('\n  per-class interaction recovery from the CP query-mode model:')
        hits = 0
        with torch.no_grad():
            V = m(torch.tensor(ALL, dtype=torch.float32)).numpy() * sd + mu
        for c in range(NC):
            I = interaction_index(V[:, c])
            top = max(PAIRS, key=lambda p: I[p])
            want = (c, c + 5)
            rank_true = 1 + sorted((-I[p] for p in PAIRS)).index(-I[want])
            hits += top == want
            print(f'    class {c}: top {NAMES[top[0]]},{NAMES[top[1]]:<4} '
                  f'| true {NAMES[c]},{NAMES[c+5]:<4} ranked {rank_true}/45  '
                  f'{"HIT" if top == want else "miss"}')
        print(f'  top-1 recovery: {hits}/{NC}   (degree-2 per class scored 4/5 on accuracy)')


if __name__ == '__main__':
    main()
