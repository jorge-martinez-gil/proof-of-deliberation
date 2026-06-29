"""Unit tests for the three annotation-quality baselines.

* ``Raykar``    -- streaming single-annotator Raykar et al. (JMLR 2010).
* ``MACE``      -- streaming Hovy et al. (NAACL 2013).
* ``IEThresh``  -- streaming Donmez & Carbonell (ECML 2008).

These are the content-based competitors to PoD's process-based signal;
the directional tests here ensure each baseline behaves consistently
with its published lineage.
"""

from __future__ import annotations

import math

import numpy as np

from pod.baselines import (
    IETHRESH_THRESHOLD,
    IETHRESH_WARMUP,
    IEThreshState,
    MACE_PRIOR_ALPHA,
    MACE_PRIOR_BETA,
    MACEOnlineState,
    RAYKAR_PRIOR,
    RAYKAR_THRESHOLD,
    RaykarOnlineState,
    accept_iethresh,
    accept_mace,
    accept_raykar,
    posterior_label_probability_raykar,
    update_iethresh,
    update_mace,
    update_raykar,
)


# ---------------------------------------------------------------------------
# Raykar
# ---------------------------------------------------------------------------
def test_raykar_fresh_state_uniform_prior():
    state = RaykarOnlineState.fresh(n_classes=3)
    assert state.counts.shape == (3, 3)
    assert np.allclose(state.counts, RAYKAR_PRIOR)
    L = state.likelihood_matrix()
    np.testing.assert_allclose(L, np.full((3, 3), 1.0 / 3.0))


def test_raykar_posterior_uniform_when_prior_only():
    state = RaykarOnlineState.fresh(n_classes=4)
    proba = np.full(4, 0.25)
    for k in range(4):
        p = posterior_label_probability_raykar(proba, k, state)
        assert abs(p - 0.25) < 1e-9


def test_raykar_posterior_responds_to_model_confidence():
    state = RaykarOnlineState.fresh(n_classes=3)
    confident = np.array([0.05, 0.90, 0.05])
    uniform = np.full(3, 1.0 / 3.0)
    p_conf = posterior_label_probability_raykar(confident, 1, state)
    p_unif = posterior_label_probability_raykar(uniform, 1, state)
    assert p_conf > p_unif


def test_raykar_update_is_soft_not_hard():
    """Raykar smears evidence across rows; WorkerQuality concentrates on argmax."""
    state = RaykarOnlineState.fresh(n_classes=3)
    proba = np.array([0.4, 0.5, 0.1])
    update_raykar(state, proba, y_tilde=1)
    # The y_tilde=1 column should have gained evidence across *all* rows
    # in proportion to the predictive probability.
    col = state.counts[:, 1] - RAYKAR_PRIOR
    np.testing.assert_allclose(col, proba)
    # Other columns unchanged.
    for l in (0, 2):
        np.testing.assert_allclose(state.counts[:, l], RAYKAR_PRIOR)


def test_raykar_accept_threshold_boundary():
    state = RaykarOnlineState.fresh(n_classes=2)
    proba_yes = np.array([0.05, 0.95])
    assert accept_raykar(proba_yes, 1, state, threshold=0.5) is True
    proba_no = np.array([0.95, 0.05])
    assert accept_raykar(proba_no, 1, state, threshold=0.5) is False


def test_raykar_update_eventually_concentrates():
    """After many consistent updates, the likelihood row becomes peaked."""
    state = RaykarOnlineState.fresh(n_classes=3)
    # Confident, consistent operator: 200 updates with high prior mass on
    # row 0, all returning label 0.
    proba = np.array([0.95, 0.04, 0.01])
    for _ in range(200):
        update_raykar(state, proba, y_tilde=0)
    L = state.likelihood_matrix()
    assert L[0, 0] > 0.95
    assert L[0, 1] < 0.05
    assert L[0, 2] < 0.05


# ---------------------------------------------------------------------------
# MACE
# ---------------------------------------------------------------------------
def test_mace_fresh_state_uniform_prior():
    state = MACEOnlineState.fresh()
    assert state.alpha == float(MACE_PRIOR_ALPHA)
    assert state.beta == float(MACE_PRIOR_BETA)
    assert state.n_obs == 0
    # Beta(1, 1) -> mean = 0.5.
    assert abs(state.posterior_mean() - 0.5) < 1e-9


def test_mace_update_increments_alpha_on_agreement():
    state = MACEOnlineState.fresh()
    proba = np.array([0.1, 0.9])  # MAP = 1
    update_mace(state, proba, y_tilde=1)
    assert state.alpha == 2.0  # 1 (prior) + 1 (agreement)
    assert state.beta == 1.0
    assert state.n_obs == 1


def test_mace_update_increments_beta_on_disagreement():
    state = MACEOnlineState.fresh()
    proba = np.array([0.1, 0.9])  # MAP = 1
    update_mace(state, proba, y_tilde=0)
    assert state.alpha == 1.0
    assert state.beta == 2.0
    assert state.n_obs == 1


def test_mace_competent_operator_drives_posterior_to_one():
    state = MACEOnlineState.fresh()
    for _ in range(50):
        update_mace(state, proba_model=np.array([0.1, 0.9]), y_tilde=1)
    # 50 agreements + 1 prior alpha = 51; 1 prior beta -> mean ~ 0.98.
    assert state.posterior_mean() > 0.95


def test_mace_spam_operator_drives_posterior_to_zero():
    state = MACEOnlineState.fresh()
    for _ in range(50):
        update_mace(state, proba_model=np.array([0.1, 0.9]), y_tilde=0)
    assert state.posterior_mean() < 0.05


def test_mace_accept_threshold_boundary():
    state = MACEOnlineState.fresh()  # Beta(1, 1) -> mean 0.5
    # At default threshold 0.5 the boundary admits the prior.
    assert accept_mace(state, threshold=0.5) is True
    assert accept_mace(state, threshold=0.6) is False


# ---------------------------------------------------------------------------
# IEThresh
# ---------------------------------------------------------------------------
def test_iethresh_fresh_state_is_zero():
    state = IEThreshState.fresh()
    assert state.n_correct == 0
    assert state.n_total == 0
    assert state.lower_bound() == 0.0


def test_iethresh_warmup_accepts_unconditionally():
    state = IEThreshState.fresh()
    for k in range(IETHRESH_WARMUP - 1):
        # Even disagreement during warm-up should pass.
        update_iethresh(state, np.array([0.1, 0.9]), y_tilde=0)
        assert accept_iethresh(state) is True


def test_iethresh_competent_operator_eventually_passes():
    state = IEThreshState.fresh()
    for _ in range(200):
        update_iethresh(state, np.array([0.1, 0.9]), y_tilde=1)
    # 200 / 200 agreements -> LCB very close to 1.0.
    assert state.lower_bound() > 0.95
    assert accept_iethresh(state, threshold=0.5) is True


def test_iethresh_spam_operator_eventually_rejected():
    state = IEThreshState.fresh()
    for _ in range(200):
        update_iethresh(state, np.array([0.1, 0.9]), y_tilde=0)
    # 0 / 200 agreements -> LCB = 0.
    assert state.lower_bound() < 0.05
    assert accept_iethresh(state, threshold=0.5) is False


def test_iethresh_lower_bound_is_pessimistic():
    """The LCB must lie below the empirical mean for any non-trivial CV."""
    state = IEThreshState(n_correct=8, n_total=10)
    mu = 0.8
    assert state.lower_bound() < mu


def test_iethresh_lcb_tightens_with_sample_size():
    """LCB at same mean tightens as n grows; the gap to the mean shrinks."""
    small = IEThreshState(n_correct=8, n_total=10)
    large = IEThreshState(n_correct=800, n_total=1000)
    gap_small = 0.8 - small.lower_bound()
    gap_large = 0.8 - large.lower_bound()
    assert gap_large < gap_small


# ---------------------------------------------------------------------------
# Sanity: METHODS panel includes all three
# ---------------------------------------------------------------------------
def test_methods_panel_includes_new_baselines():
    from pod.experiment import METHODS
    for m in ("WorkerQuality", "Raykar", "MACE", "IEThresh"):
        assert m in METHODS
