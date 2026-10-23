"""
conformal.py

Split conformal prediction for classifiers, plus calibration diagnostics.

Instead of one predicted class, each sample gets a *set* of plausible classes.
If calibration and test samples are exchangeable, the true class lies in the
set with probability >= 1 - alpha (marginally, averaged over calibration draws).
That guarantee is about the average over samples, not about any single sample
and not about each class separately -- see per_class_coverage().

Score used: s(x, y) = 1 - p_y(x)  (the "least ambiguous classifier" score).
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float) -> float:
    """cal_labels are column indices into cal_probs. Returns qhat (inf if n is too small)."""
    n = len(cal_labels)
    scores = 1.0 - cal_probs[np.arange(n), cal_labels]
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return float("inf")
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean matrix (n_samples, n_classes): True where the class is in the set."""
    return (1.0 - probs) <= qhat


def coverage(sets: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(sets[np.arange(len(labels)), labels]))


def per_class_coverage(sets: np.ndarray, labels: np.ndarray, n_classes: int) -> list[float]:
    out = []
    for k in range(n_classes):
        m = labels == k
        out.append(float(np.mean(sets[m, k])) if m.any() else float("nan"))
    return out


def naive_cumulative_sets(probs: np.ndarray, level: float) -> np.ndarray:
    """Uncalibrated baseline: take classes in order of predicted probability until
    their probabilities sum to `level`. Trusts the model's probabilities as-is."""
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    keep_sorted = (cum - sorted_p) < level   # include a class until the mass reaches level
    sets = np.zeros_like(probs, dtype=bool)
    np.put_along_axis(sets, order, keep_sorted, axis=1)
    return sets


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE with equal-width bins (noisy for small n -- report with caution)."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 5):
    """Returns (mean confidence, accuracy, count) per non-empty bin."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.linspace(conf.min() - 1e-9, conf.max() + 1e-9, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return rows


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
