"""Direct tensor-network datamodels: TNDM-CP, TNDM-TT and a tree (HT) variant.

Each module maps an ``(n, d)`` inclusion matrix to ``(n, Q)`` predictions, with
an independent set of factors (CP) or cores (TT) per outcome ``q``.

Parameterisation.  Mode ``j`` is contracted with the leg ``[1, 1 - 2 z_j]``
(the +/-1 encoding of inclusion).  This differs from the ``[z_j, 1]`` leg used
in the paper only by an invertible change of basis on each mode, so the
represented function class and all CP/TT ranks are identical
(Appendix A of the paper).

``train_tn`` fits one configuration with AdamW and validation early stopping;
``fit_tn_family`` runs the rank / tensor-order / regularisation search and
returns the selected model wrapped as cardinality trend + tensor residual.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from tndm.baselines.regression import OffsetModel
from tndm.utils.coalitions import bits_of

__all__ = ["RANKS", "REGULARIZATION", "QueryCP", "QueryTT", "QueryHT", "TNDMCP", "TNDMTT",
           "FAMILIES", "train_tn", "fit_tn_family"]

RANKS = (2, 4, 8, 16)
REGULARIZATION = ((1e-3, 1e-4), (1e-2, 1e-3))


class QueryCP(nn.Module):
    def __init__(self, d: int, q: int, rank: int, seed: int):
        super().__init__(); generator = torch.Generator().manual_seed(seed)
        factor = torch.zeros(q, d, 2, rank, dtype=torch.float64)
        factor[:, :, 0, :] = 1.0
        factor[:, :, 1, :] = .02 * torch.randn(q, d, rank, generator=generator, dtype=torch.float64)
        self.factor = nn.Parameter(factor)
        self.output = nn.Parameter(torch.randn(q, rank, generator=generator, dtype=torch.float64) / np.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        legs = torch.stack((torch.ones_like(x), 1 - 2 * x), -1)
        local = torch.einsum("ndi,qdir->nqdr", legs, self.factor)
        return torch.einsum("nqr,qr->nq", local.prod(2), self.output)


class QueryTT(nn.Module):
    def __init__(self, d: int, q: int, rank: int, seed: int):
        super().__init__(); generator = torch.Generator().manual_seed(seed)
        bonds = [1] + [min(rank, 2 ** index) for index in range(1, d + 1)]
        cores = []
        for i in range(d):
            core = .02 * torch.randn(q, bonds[i], 2, bonds[i + 1], generator=generator, dtype=torch.float64)
            core[:, :, 0, :] = 0
            diagonal = torch.arange(min(bonds[i], bonds[i + 1]))
            core[:, diagonal, 0, diagonal] = 1.0
            cores.append(nn.Parameter(core))
        self.cores = nn.ParameterList(cores)
        self.output = nn.Parameter(torch.randn(q, bonds[-1], generator=generator,
                                               dtype=torch.float64) / np.sqrt(bonds[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leg = 1 - 2 * x
        state = torch.ones((len(x), self.output.shape[0], 1), dtype=x.dtype, device=x.device)
        for index, core in enumerate(self.cores):
            local = core[:, :, 0, :][None] + leg[:, None, index, None, None] * core[:, :, 1, :][None]
            state = torch.einsum("nql,nqlr->nqr", state, local)
        return torch.einsum("nqr,qr->nq", state, self.output)


class QueryHT(nn.Module):
    def __init__(self, d: int, q: int, rank: int, seed: int):
        super().__init__(); generator = torch.Generator().manual_seed(seed)
        leaf_rank = min(2, rank)
        leaves = []
        for _ in range(d):
            leaf = .02 * torch.randn(q, 2, leaf_rank, generator=generator, dtype=torch.float64)
            leaf[:, 0, :] = 1.0
            leaves.append(nn.Parameter(leaf))
        self.leaves = nn.ParameterList(leaves)
        widths, transfers = [leaf_rank] * d, []
        while len(widths) > 1:
            next_widths = []
            for index in range(0, len(widths) - 1, 2):
                transfer = .02 * torch.randn(q, widths[index], widths[index + 1], rank,
                                             generator=generator, dtype=torch.float64)
                diagonal = torch.arange(min(widths[index], widths[index + 1], rank))
                transfer[:, diagonal, diagonal, diagonal] = 1.0
                transfers.append(nn.Parameter(transfer))
                next_widths.append(rank)
            if len(widths) % 2:
                next_widths.append(widths[-1])
            widths = next_widths
        self.transfers = nn.ParameterList(transfers)
        self.output = nn.Parameter(torch.randn(q, widths[0], generator=generator,
                                               dtype=torch.float64) / np.sqrt(widths[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        legs = torch.stack((torch.ones_like(x), 1 - 2 * x), -1)
        values = [torch.einsum("ni,qir->nqr", legs[:, index], self.leaves[index])
                  for index in range(len(self.leaves))]
        cursor = 0
        while len(values) > 1:
            nxt = []
            for index in range(0, len(values) - 1, 2):
                nxt.append(torch.einsum("nqa,nqb,qabr->nqr", values[index], values[index + 1],
                                        self.transfers[cursor])); cursor += 1
            if len(values) % 2:
                nxt.append(values[-1])
            values = nxt
        return torch.einsum("nqr,qr->nq", values[0], self.output)


FAMILIES = {"cp": QueryCP, "tt": QueryTT, "ht": QueryHT}

# Paper names for the two main parameterisations.
TNDMCP = QueryCP
TNDMTT = QueryTT


def train_tn(family: str, d: int, q: int, rank: int, seed: int, order: tuple[int, ...],
             x_fit: np.ndarray, y_fit: np.ndarray, weight: np.ndarray,
             x_val: np.ndarray, y_val: np.ndarray, weight_decay: float, penalty: float,
             steps: int = 201):
    model = FAMILIES[family](d, q, rank, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=weight_decay)
    tx = torch.as_tensor(x_fit[:, list(order)], dtype=torch.float64)
    ty = torch.as_tensor(y_fit, dtype=torch.float64)
    tw = torch.as_tensor(weight, dtype=torch.float64)
    tv = torch.as_tensor(x_val[:, list(order)], dtype=torch.float64)
    vy = torch.as_tensor(y_val, dtype=torch.float64)
    rng = np.random.default_rng(5000 + seed)
    regularizer = torch.as_tensor(rng.integers(0, 2, size=(128, d)).astype(float)[:, list(order)],
                                  dtype=torch.float64)
    best_mse, best_state, stale, best_step = float("inf"), None, 0, 0
    for step in range(steps):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(tw[:, None] * (model(tx) - ty) ** 2)
            loss = loss + penalty * torch.mean(model(regularizer) ** 2)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10); optimizer.step()
        if step % 10 == 0:
            with torch.no_grad():
                mse = float(torch.mean((model(tv) - vy) ** 2))
            if mse < best_mse - 1e-8:
                best_mse, best_state, best_step, stale = mse, copy.deepcopy(model.state_dict()), step, 0
            else:
                stale += 1
            if stale >= 6:
                break
    model.load_state_dict(best_state)
    return best_mse, model, best_step


def fit_tn_family(family, d, q, masks_fit, res_fit, masks_val, res_val, weight, base,
                  orders, ranks=RANKS):
    scale = np.maximum(res_fit.std(0), .05)
    x_fit, x_val = bits_of(masks_fit, d), bits_of(masks_val, d)
    best = None
    for rank in ranks:
        for order_index, order in enumerate(orders):
            for reg_index, (decay, penalty) in enumerate(REGULARIZATION):
                mse, model, step = train_tn(family, d, q, rank,
                                            1000 * order_index + 100 * reg_index + rank, order,
                                            x_fit, res_fit / scale, weight, x_val, res_val / scale,
                                            decay, penalty)
                raw = mse * float(np.mean(scale ** 2))
                if best is None or raw < best[0]:
                    best = (raw, rank, order, model, step, decay, penalty)
    raw, rank, order, model, step, decay, penalty = best

    def residual(masks: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(bits_of(masks, d)[:, list(order)], dtype=torch.float64)
            return model(x).numpy() * scale

    parameters = int(sum(p.numel() for p in model.parameters()))
    fitted = OffsetModel(family, base, residual)
    # Expose the selected network so attribution queries can contract it directly
    # (tndm.attribution.contraction).  Predictions are unaffected.
    fitted.tn_module, fitted.tn_order, fitted.tn_scale = model, tuple(order), scale
    return fitted, {"rank": rank, "validation_mse": raw,
                                                 "early_stop_step": step, "parameters": parameters,
                                                 "weight_decay": decay, "functional_penalty": penalty,
                                                 "order_is_role_contiguous": order == orders[0]}
