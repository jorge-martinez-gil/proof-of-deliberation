"""Unit tests for the WorkerQuality baseline and its online estimator."""

from __future__ import annotations

import numpy as np

from pod.baselines import (
    WORKER_QUALITY_PRIOR,
    WorkerQualityState,
    accept_worker_quality,
    posterior_label_probability,
    update_worker_quality,
)


def test_fresh_state_uses_uniform_laplace_prior():
    state = WorkerQualityState.fresh(n_classes=3)
    assert state.counts.shape == (3, 3)
    assert np.allclose(state.counts, WORKER_QUALITY_PRIOR)
    L = state.likelihood_matrix()
    # Row-normalised should be uniform.
    np.testing.assert_allclose(L, np.full((3, 3), 1.0 / 3.0))


def test_posterior_matches_uniform_when_prior_only():
    """With Laplace prior alone and uniform model prob, posterior must be 1/K."""
    state = WorkerQualityState.fresh(n_classes=4)
    proba = np.full(4, 0.25)
    for k in range(4):
        p = posterior_label_probability(proba, k, state)
        assert abs(p - 0.25) < 1e-9


def test_posterior_responds_to_model_prior():
    """If the model is confident about y_tilde, the posterior should rise."""
    state = WorkerQualityState.fresh(n_classes=3)
    confident = np.array([0.05, 0.90, 0.05])
    uniform = np.full(3, 1.0 / 3.0)
    p_conf = posterior_label_probability(confident, 1, state)
    p_unif = posterior_label_probability(uniform, 1, state)
    assert p_conf > p_unif


def test_update_increments_only_target_cell():
    state = WorkerQualityState.fresh(n_classes=3)
    update_worker_quality(state, y_pred_model=2, y_tilde=0)
    assert state.counts[2, 0] == WORKER_QUALITY_PRIOR + 1.0
    other_total = state.counts.sum() - state.counts[2, 0]
    assert abs(other_total - 8 * WORKER_QUALITY_PRIOR) < 1e-9


def test_accept_threshold_boundary():
    state = WorkerQualityState.fresh(n_classes=2)
    # Strongly confident on y_tilde=1 -> accept.
    proba_yes = np.array([0.05, 0.95])
    assert accept_worker_quality(proba_yes, 1, state, threshold=0.5) is True
    # Strongly confident on the *other* class -> reject.
    proba_no = np.array([0.95, 0.05])
    assert accept_worker_quality(proba_no, 1, state, threshold=0.5) is False


def test_update_eventually_dominates_prior():
    """After many consistent updates, the likelihood row becomes peaked."""
    state = WorkerQualityState.fresh(n_classes=3)
    for _ in range(200):
        update_worker_quality(state, y_pred_model=0, y_tilde=0)
    L = state.likelihood_matrix()
    assert L[0, 0] > 0.95
    assert L[0, 1] < 0.03
    assert L[0, 2] < 0.03
