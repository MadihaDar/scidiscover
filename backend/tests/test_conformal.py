import numpy as np

from app.pipeline.conformal import (conformal_quantile, prediction_sets, coverage,
                                    naive_cumulative_sets, expected_calibration_error)


def _synthetic(n, rng, k=4, sharp=2.0):
    """Probabilities from a noisy softmax; labels drawn FROM those probabilities,
    so the model is perfectly calibrated by construction."""
    logits = rng.normal(0, sharp, size=(n, k))
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in p])
    return p, y


def test_split_conformal_coverage_holds_on_average():
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(300):
        pc, yc = _synthetic(40, rng)
        pt, yt = _synthetic(200, rng)
        q = conformal_quantile(pc, yc, alpha=0.1)
        covs.append(coverage(prediction_sets(pt, q), yt))
    m = np.mean(covs)
    assert 0.88 <= m <= 0.945          # theory: >= 0.90, <= 0.90 + 1/41


def test_tiny_calibration_set_returns_full_sets():
    rng = np.random.default_rng(1)
    pc, yc = _synthetic(5, rng)
    q = conformal_quantile(pc, yc, alpha=0.05)   # level > 1 -> cannot guarantee 95%
    assert q == float("inf")
    assert prediction_sets(pc, q).all()


def test_naive_sets_and_ece_run():
    rng = np.random.default_rng(2)
    p, y = _synthetic(100, rng)
    s = naive_cumulative_sets(p, 0.9)
    assert s.shape == p.shape and s.any(axis=1).all()
    assert 0 <= expected_calibration_error(p, y) <= 1
