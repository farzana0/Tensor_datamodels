"""Scoring for the equal-budget comparison.

Errors are reported relative to the norm of the exact value so that cubes with
different utility scales (log probabilities, accuracies, negative losses) can be
put on one axis.  Rank metrics are Spearman correlations, and interaction
recovery is precision at k over the tie-aware top-k of the exact index.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def relative_l2(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)):
        return float("nan")
    denominator = np.linalg.norm(truth)
    if denominator == 0:
        return float("nan")
    return float(np.linalg.norm(estimate - truth) / denominator)


def spearman(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)):
        return float("nan")
    if np.allclose(truth, truth[0]) or np.allclose(estimate, estimate[0]):
        return float("nan")
    return float(spearmanr(estimate, truth).statistic)


def precision_at_k(estimate: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Fraction of the estimate's top-k (by magnitude) inside the exact top-k.

    Ties in the exact index are resolved in the estimate's favour: any entry whose
    exact magnitude equals the k-th largest counts as being in the exact top-k.
    """
    estimate = np.abs(np.asarray(estimate, dtype=float))
    truth = np.abs(np.asarray(truth, dtype=float))
    if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)) or k <= 0:
        return float("nan")
    k = min(k, len(truth))
    threshold = np.sort(truth)[-k]
    accepted = truth >= threshold - 1e-12
    predicted = np.argsort(-estimate)[:k]
    return float(np.mean(accepted[predicted]))


def normalised_mse(prediction: np.ndarray, truth: np.ndarray, held_out: np.ndarray) -> float:
    """Held-out coalition MSE divided by the variance of the exact cube."""
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if not np.all(np.isfinite(prediction[held_out])):
        return float("nan")
    variance = float(np.var(truth))
    if variance == 0:
        return float("nan")
    return float(np.mean((prediction[held_out] - truth[held_out]) ** 2) / variance)
