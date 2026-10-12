"""Fitting is free; retraining is not. How much accuracy does more fitting compute buy?

One sample in this game is one full fine-tune (~3 GPU-min at 7B, and far more at scale).
Fitting a tensor network to the samples already collected costs seconds of CPU. Under that
cost model a single random initialisation of a non-convex CP fit is indefensible: restarts,
rank sweeps, penalty sweeps, alternative contractions and ensembles are all free relative to
the data, and a more accurate model at the SAME number of fine-tunes is strictly better.

This measures what that free compute is actually worth, held-out:

  single fit                one initialisation, one rank, one penalty (what was used so far)
  best-of-8 restarts        eight initialisations, best on an inner validation split
  ensemble over seeds       average the predictions of eight fits
  ensemble over ranks       average across ranks 3-6, two seeds each
  ensemble over penalties   average across ranks 3-6, two penalties each

A fair comparison also re-runs the BASELINES with equivalent selection effort, otherwise the
ensemble wins on compute rather than on structure. Ridge models get their penalty tuned per
split, which is their equivalent knob.
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
PAIRS, ALL = CPQ.PAIRS, CPQ.ALL
torch.set_num_threads(8)
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
EPOCHS = 1500


def bases(Z):
    one = np.ones((len(Z), 1))
    return {'linear': np.hstack([one, Z]),
            'matched': np.hstack([one, Z, np.column_stack([Z[:, i] * Z[:, i + 5] for i in range(5)])]),
            'degree-2': np.hstack([one, Z, np.column_stack([Z[:, a] * Z[:, b] for a, b in PAIRS])])}


def ridge_tuned(Xtr, ytr, Xte, rng):
    n = len(ytr)
    k = max(3, n // 4)
    o = rng.permutation(n)
    va, tr = o[:k], o[k:]
    best, blam = -np.inf, RIDGES[0]
    for lam in RIDGES:
        P = np.eye(Xtr.shape[1]) * lam
        P[0, 0] = 0.0
        w = np.linalg.solve(Xtr[tr].T @ Xtr[tr] + P, Xtr[tr].T @ ytr[tr])
        p = Xtr[va] @ w
        d = np.sum((ytr[va] - ytr[va].mean()) ** 2)
        s = -np.inf if d == 0 else 1 - np.sum((ytr[va] - p) ** 2) / d
        if s > best:
            best, blam = s, lam
    P = np.eye(Xtr.shape[1]) * blam
    P[0, 0] = 0.0
    return Xte @ np.linalg.solve(Xtr.T @ Xtr + P, Xtr.T @ ytr)


def main():
    for metric in ('accuracy', 'mean_gold_logprob'):
        Z, Y = CPQ.load(metric)
        n = len(Z)
        rng = np.random.default_rng(3)
        splits = [rng.permutation(n) for _ in range(15)]
        nte = max(4, int(round(0.30 * n)))

        def ev(fn):
            sc = []
            for si, perm in enumerate(splits):
                te, tr = perm[:nte], perm[nte:]
                sc.append(np.nanmean(CPQ.r2_cols(Y[te], fn(Z[tr], Y[tr], Z[te], si))))
            return float(np.mean(sc)), float(np.std(sc))

        def single(Ztr, Ytr, Zte, si):
            m, mu, sd = CPQ.fit_cp(Ztr, Ytr, 5, epochs=EPOCHS)
            return CPQ.predict(m, Zte, mu, sd)

        def bestof(Ztr, Ytr, Zte, si):
            m, mu, sd = CPQ.fit_cp_best(Ztr, Ytr, 5, epochs=EPOCHS, restarts=8)
            return CPQ.predict(m, Zte, mu, sd)

        def _pred(t, Zte):
            m, mu, sd = t
            return CPQ.predict(m, Zte, mu, sd)

        def ens_seed(Ztr, Ytr, Zte, si):
            return np.mean([_pred(CPQ.fit_cp(Ztr, Ytr, 5, epochs=EPOCHS, seed=s), Zte)
                            for s in range(8)], 0)

        def ens_rank(Ztr, Ytr, Zte, si):
            return np.mean([_pred(CPQ.fit_cp(Ztr, Ytr, r, epochs=EPOCHS, seed=s), Zte)
                            for r in (3, 4, 5, 6) for s in (0, 1)], 0)

        def ens_l2(Ztr, Ytr, Zte, si):
            return np.mean([_pred(CPQ.fit_cp(Ztr, Ytr, r, l2=l2, epochs=EPOCHS, seed=0), Zte)
                            for r in (3, 4, 5, 6) for l2 in (1e-3, 1e-2)], 0)

        def mk_ridge(nm):
            def f(Ztr, Ytr, Zte, si):
                Btr, Bte = bases(Ztr)[nm], bases(Zte)[nm]
                r = np.random.default_rng(si)
                return np.column_stack([ridge_tuned(Btr, Ytr[:, c], Bte, r)
                                        for c in range(Ytr.shape[1])])
            return f

        print(f'\n=== utility: {metric} -- held-out per-question R2, 15 shared splits ===')
        print('   (fitting compute is free; only fine-tunes cost anything)\n')
        out = {}
        for nm, fn in (('linear (tuned)', mk_ridge('linear')),
                       ('matched (tuned)', mk_ridge('matched')),
                       ('degree-2 (tuned)', mk_ridge('degree-2')),
                       ('CP single fit, R=5', single),
                       ('CP best-of-8, R=5', bestof),
                       ('CP ENSEMBLE 8 seeds', ens_seed),
                       ('CP ENSEMBLE ranks 3-6', ens_rank),
                       ('CP ENSEMBLE rank x l2', ens_l2)):
            m_, s_ = ev(fn)
            out[nm] = {'r2': m_, 'sd': s_}
            print(f'  {nm:<26}{m_:+.3f} +/- {s_:.3f}')
        (RES / f'FITTING_COMPUTE_{metric}.json').write_text(json.dumps(out, indent=2))
    print('\ndone')


if __name__ == '__main__':
    main()
