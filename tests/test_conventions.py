"""Binary coalition indexing and tensor conventions."""
import numpy as np
import torch

from tndm.attribution import exact
from tndm.models.tensor_networks import QueryCP, QueryTT
from tndm.utils.coalitions import all_masks, bits_of, size_weights, stratified_split, uniform_masks


def test_bit_i_of_the_mask_is_group_i():
    d = 5
    z = bits_of(all_masks(d), d)
    assert z.shape == (32, 5)
    assert np.array_equal(z[0], np.zeros(5))                 # empty coalition is mask 0
    assert np.array_equal(z[(1 << d) - 1], np.ones(5))       # full coalition is 2**d - 1
    for group in range(d):
        assert np.array_equal(z[1 << group], np.eye(d)[group])
    assert np.array_equal(z @ (1 << np.arange(d)), all_masks(d))  # round trip


def test_value_table_is_indexed_by_mask():
    d = 4
    masks = all_masks(d)
    z = bits_of(masks, d)
    weights = np.array([1.0, 10.0, 100.0, 1000.0])
    table = z @ weights
    assert table[0b0101] == 101.0
    # exact.shapley_values of an additive game returns the weights in group order
    assert np.allclose(exact.shapley_values(table), weights)
    assert np.allclose(exact.loo_values(table), weights)


def test_value_tensor_reshape_matches_mask_order():
    """Reshaping a mask-indexed table to (2,)*d puts group d-1 on the first axis."""
    d = 3
    table = np.arange(1 << d, dtype=float)
    tensor = table.reshape((2,) * d)
    for mask in range(1 << d):
        index = tuple((mask >> (d - 1 - axis)) & 1 for axis in range(d))
        assert tensor[index] == table[mask]


def test_cp_forward_matches_definition():
    """f(z) = sum_r c_r prod_j (a_rj + b_rj (1 - 2 z_j)), per outcome."""
    d, q, rank = 5, 2, 3
    model = QueryCP(d, q, rank, seed=1)
    with torch.no_grad():
        model.factor.copy_(torch.randn(q, d, 2, rank, dtype=torch.float64))
    z = bits_of(all_masks(d), d)
    factor, output = model.factor.detach().numpy(), model.output.detach().numpy()
    expected = np.zeros((len(z), q))
    for n, row in enumerate(z):
        for out in range(q):
            local = factor[out, :, 0, :] + (1 - 2 * row)[:, None] * factor[out, :, 1, :]
            expected[n, out] = local.prod(0) @ output[out]
    with torch.no_grad():
        assert np.allclose(model(torch.as_tensor(z)).numpy(), expected)


def test_cp_represents_a_degree_d_monomial_at_rank_one():
    """v(z) = prod_j z_j has interaction degree d and CP rank one (Sec. 2.2 of the paper)."""
    d = 6
    model = QueryCP(d, 1, 1, seed=0)
    with torch.no_grad():
        model.factor[:, :, 0, :] = 0.5      # z_j = (1 - (1 - 2 z_j)) / 2
        model.factor[:, :, 1, :] = -0.5
        model.output.fill_(1.0)
    z = bits_of(all_masks(d), d)
    with torch.no_grad():
        prediction = model(torch.as_tensor(z)).numpy()[:, 0]
    assert np.allclose(prediction, z.prod(1))
    coefficients = exact.mobius(prediction)
    assert np.isclose(coefficients[-1], 1.0) and np.allclose(coefficients[:-1], 0.0)


def test_tt_forward_matches_core_contraction_and_rank_bound():
    d, q, rank = 6, 1, 3
    model = QueryTT(d, q, rank, seed=4)
    with torch.no_grad():
        for core in model.cores:
            core.copy_(torch.randn_like(core))
    z = bits_of(all_masks(d), d)
    with torch.no_grad():
        prediction = model(torch.as_tensor(z)).numpy()[:, 0]
    cores = [c.detach().numpy()[0] for c in model.cores]
    output = model.output.detach().numpy()[0]
    for n, row in enumerate(z):
        state = np.ones(1)
        for j, core in enumerate(cores):
            state = state @ (core[:, 0, :] + (1 - 2 * row[j]) * core[:, 1, :])
        assert np.isclose(prediction[n], state @ output)
    # every sequential unfolding of the value tensor has rank at most the bond dimension
    table = np.empty((2,) * d)
    for mask in range(1 << d):
        table[tuple((mask >> j) & 1 for j in range(d))] = prediction[mask]   # axis j = group j
    for cut in range(1, d):
        unfolding = table.reshape(2 ** cut, -1)
        assert np.linalg.matrix_rank(unfolding, tol=1e-8) <= rank


def test_sampling_and_split_helpers():
    d = 10
    masks = uniform_masks(d, 200, np.random.default_rng(0))
    assert len(set(masks.tolist())) == 200 and masks.min() >= 0 and masks.max() < (1 << d)
    again = uniform_masks(d, 200, np.random.default_rng(0))
    assert np.array_equal(masks, again)
    fit, validation = stratified_split(masks, seed=1)
    assert set(fit.tolist()).isdisjoint(validation.tolist())
    assert sorted(fit.tolist() + validation.tolist()) == sorted(masks.tolist())
    weights = size_weights(fit)
    assert np.isclose(weights.mean(), 1.0) and (weights > 0).all()


def _tt_svd(tensor: np.ndarray, tolerance: float = 1e-10) -> list[np.ndarray]:
    """Exact tensor-train decomposition (cores of shape (left, 2, right))."""
    d = tensor.ndim
    cores, left, remainder = [], 1, tensor.reshape(1, -1)
    for _ in range(d - 1):
        matrix = remainder.reshape(left * 2, -1)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        keep = max(1, int(np.sum(singular > tolerance)))
        cores.append(u[:, :keep].reshape(left, 2, keep))
        remainder, left = singular[:keep, None] * vh[:keep], keep
    cores.append(remainder.reshape(left, 2, 1))
    return cores


def test_redundant_support_function_is_represented_exactly_at_tt_rank_two():
    """v = (z0 or z1)(z2 or z3): interaction degree 4, TT ranks <= 2 when each pair is contiguous."""
    d = 4
    masks = all_masks(d)
    z = bits_of(masks, d)
    values = np.maximum(z[:, 0], z[:, 1]) * np.maximum(z[:, 2], z[:, 3])
    assert np.count_nonzero(exact.mobius(values)) == 9 and exact.mobius(values)[-1] != 0  # degree 4

    def value_tensor(order):
        tensor = np.empty((2,) * d)
        for mask in masks:
            tensor[tuple((mask >> group) & 1 for group in order)] = values[mask]
        return tensor

    contiguous = _tt_svd(value_tensor((0, 1, 2, 3)))
    assert max(core.shape[2] for core in contiguous) == 2
    interleaved = _tt_svd(value_tensor((0, 2, 1, 3)))
    assert max(core.shape[2] for core in interleaved) > 2

    # Load the rank-2 cores into TNDM-TT: the model reproduces all 16 coalition values exactly.
    model = QueryTT(d, 1, 2, seed=0)
    with torch.no_grad():
        for parameter, core in zip(model.cores, contiguous):
            padded = np.zeros((parameter.shape[1], 2, parameter.shape[3]))
            padded[:core.shape[0], :, :core.shape[2]] = core
            parameter[0, :, 0, :] = torch.as_tensor((padded[:, 0, :] + padded[:, 1, :]) / 2)
            parameter[0, :, 1, :] = torch.as_tensor((padded[:, 0, :] - padded[:, 1, :]) / 2)
        model.output.zero_()
        model.output[0, 0] = 1.0
        prediction = model(torch.as_tensor(z)).numpy()[:, 0]
    assert np.allclose(prediction, values, atol=1e-12)
