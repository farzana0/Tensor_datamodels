"""d=12 study d=12 k-hop replication: selection tables, interaction AP, noise ceiling, LDS and native Spearman.

LDS held-out set: the exhaustive size-3 target slice (all 220 size-3 coalitions per learner seed), never
used in fitting (retained fraction 3/12 = 0.25, dictated by the d=12 study design; the size-4/5/6 evaluation cells
are nominated by the surrogates and therefore not an unbiased held-out sample). Blind seeds 43 and 59 only:
seed 31's size-3 outcomes come from the development run's mixed-backend cube (d=12 study finding).
tau = exact Shapley vector of each frozen predicted cube (4096 masks x 10 queries), predictions only.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from _common import Provenance, write_rows
import lds

D, Q = 12, 10
SEEDS = (31, 43, 59)
BUDGETS = (32, 48, 64, 96, 128, 160)
FAMILIES = ["linear", "degree2_ridge", "degree3_ridge", "sparse_degree3_omp", "rbf_kernel_ridge",
            "degree_ladder_tt", "cp", "linear_plus_cp", "tt", "linear_plus_tt", "ht", "linear_plus_ht",
            "influence_loo", "cardinality"]


def copy_csv(prov, root, rel, out):
    with open(prov.read(root, rel), newline="") as f:
        rows = list(csv.DictReader(f))
    return write_rows(out, rows, list(rows[0].keys()))


def main():
    prov = Provenance("extract_khop_d12")
    outputs = []
    for rel, out in [("05_d12_replication/tables/d12_selection_by_seed_and_slice.csv", "khop_d12/selection_by_seed_and_slice.csv"),
                     ("05_d12_replication/tables/d12_confirmatory_comparisons.csv", "khop_d12/confirmatory_comparisons.csv"),
                     ("05_d12_replication/tables/d12_noise_ceiling.csv", "khop_d12/noise_ceiling.csv"),
                     ("05_d12_replication/analysis/size3_interaction_contrasts.csv", "khop_d12/interaction_contrasts.csv"),
                     ("05_d12_replication/analysis/slice_summary.csv", "khop_d12/slice_summary.csv")]:
        outputs.append(copy_csv(prov, "HUB", rel, out))
    prov.read("HUB", "05_d12_replication/noise_ceiling/ceiling.json")

    # interaction AP per seed x budget x method (mean over 10 queries) and top-1 complete counts
    agg = defaultdict(list)
    with open(prov.read("HUB", "05_d12_replication/analysis/size3_third_order_interactions.csv"), newline="") as f:
        for r in csv.DictReader(f):
            agg[(int(r["seed"]), int(r["new_budget"]), r["method"])].append(
                (float(r["complete_interaction_ap"]), r["top1_interaction_complete"] == "True"))
    ap_rows = [dict(seed=s, new_budget=b, method=m, queries=len(v), mean_ap=float(np.mean([a for a, _ in v])),
                    top1_complete=int(sum(c for _, c in v))) for (s, b, m), v in sorted(agg.items())]
    outputs.append(write_rows("khop_d12/interaction_ap_by_seed.csv", ap_rows))

    # held-out size-3 truth
    lds_rows = []
    for seed in (43, 59):  # seed 31's size-3 slice is the development run's mixed-backend cube: excluded
        truth = {}
        with open(prov.read("HUB", f"05_d12_replication/cubes/d12_cells_seed{seed}.csv"), newline="") as f:
            for r in csv.DictReader(f):
                if r["split"] == "target":
                    truth.setdefault(int(r["mask"]), np.full(Q, np.nan))[int(r["query"])] = float(r["log_probability"])
        masks = np.array(sorted(truth))
        assert len(masks) == 220 and np.all(lds.popcount(masks) == 3)
        y = np.array([truth[m] for m in masks])
        assert not np.isnan(y).any()
        for b in BUDGETS:
            npz = np.load(prov.read("D12_DIR", f"results/fits/seed{seed}/budget_{b}/frozen_predictions.npz"))
            assert list(npz["masks"]) == list(range(1 << D))
            for fam in [f for f in FAMILIES if f in npz.files]:  # CP/HT families were fitted at budget 160 only
                pred = npz[fam]
                tau = lds.shapley_from_cube(pred, D)
                for q in range(Q):
                    l, n = lds.lds_and_native(tau[:, q], pred[masks, q], y[:, q], masks, D)
                    lds_rows.append(dict(seed=seed, blind=seed != 31, new_budget=b, family=fam, query=q,
                                         n_heldout=len(masks), lds=l, native=n,
                                         tie_fraction=lds.tie_fraction(y[:, q]),
                                         modal_fraction_1nat=lds.modal_fraction(y[:, q], 1.0)))
    outputs.append(write_rows("khop_d12/lds_size3.csv", lds_rows))
    prov.write(outputs)
    print("ok", len(lds_rows))


if __name__ == "__main__":
    main()
