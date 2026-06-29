"""
Core Proof-of-Deliberation verification primitives.

This module implements the three checks that compose the composite
verification function :math:`\\mathcal{V}(\\pi_t)` defined in Section 4 of
the paper:

* :func:`gate_check`        -- per-label *Deliberation Gate*
  (Section 4.3, ``V_t``).
* :func:`coupling_check`    -- *Cognitive Coupling* authenticity test
  (Section 4.1, ``rho_cog``).
* :func:`gaming_detector`,
  :func:`fatigue_detector`  -- *Multi-Scale Vigilance* layer
  (Section 4.2, ``S_vig``).

All checks return integers in ``{0, 1}`` rather than booleans to match
the indicator-function notation used in the manuscript, and to make
multiplicative composition explicit.

Numerical semantics are intentionally identical to the reference
implementation that produced the published numbers; the code path is
preserved verbatim, only re-organised into a documented public API.
"""

from __future__ import annotations

import numpy as np

from pod.config import PoDParams
from pod.utils import coefficient_of_variation, spearman_rho


def expected_delib(complexity: float, pod: PoDParams) -> float:
    """Difficulty-adjusted expected deliberation time (milliseconds).

    Implements the linear model :math:`\\hat{\\Delta}(C) = a\\,C + b`
    introduced in Section 4.3, which is calibrated from a warm-up phase
    in real deployments and shipped as ``PoDParams.gate_a`` and
    ``PoDParams.gate_b`` in our experiments.
    """
    return pod.gate_a * complexity + pod.gate_b


def gate_check(delib_ms: float, complexity: float, pod: PoDParams) -> int:
    """Instantaneous Deliberation Gate ``V_t``.

    Returns 1 if the observed deliberation time falls inside the
    difficulty-adjusted admission window
    ``[tau_min(C), tau_max(C)]``, with absolute floor and ceiling
    clamps. Implausibly fast (reflexive) and implausibly slow
    (distracted) responses are rejected.

    Parameters
    ----------
    delib_ms : float
        Observed deliberation time for the current label, in
        milliseconds.
    complexity : float
        Task complexity ``C_t`` (entropy in nats normalised to
        ``[0, 1]`` for entropy-based mode, or ``c_star`` for the
        synthetic boundary stream).
    pod : PoDParams
        Verification-layer hyperparameters.
    """
    mu = expected_delib(complexity, pod)
    lo = max(pod.gate_floor_ms, mu * (1.0 - pod.gate_lo_frac))
    hi = min(pod.gate_ceil_ms, mu * (1.0 + pod.gate_hi_frac))
    return int(lo <= delib_ms <= hi)


def coupling_check(
    comp_hist: np.ndarray, delib_hist: np.ndarray, pod: PoDParams
) -> int:
    """Cognitive-coupling authenticity test.

    Returns 1 when the Spearman rank correlation between recent task
    complexities and recent deliberation times exceeds
    ``pod.coupling_epsilon`` over the most recent
    ``pod.coupling_window`` observations. Before the window is full the
    check defaults to 1 (admit), since the warm-up phase of the protocol
    suppresses gating until enough history has accumulated
    (Section 4.5).

    A non-positive ``coupling_epsilon`` effectively disables the check,
    which is the configuration used on real-data streams where the
    paper's empirical analysis showed coupling to be uninformative
    (Table -- ablation column ``-- coupling``).
    """
    w = pod.coupling_window
    if len(comp_hist) < w:
        return 1
    x = comp_hist[-w:]
    y = delib_hist[-w:]
    rho = spearman_rho(x, y)
    return int(np.isfinite(rho) and rho > pod.coupling_epsilon)


def gaming_detector(delib_hist: np.ndarray, pod: PoDParams) -> int:
    """Short-window vigilance check targeting *gaming* behaviour.

    Returns 1 when the most recent ``gaming_window`` deliberations have
    both (i) a mean below ``gaming_mu_max_ms`` and (ii) a coefficient of
    variation below ``gaming_cv_max`` -- the signature of mechanical /
    near-constant clicking. The composite verifier treats this 1 as a
    *violation*.
    """
    w = pod.gaming_window
    if delib_hist is None or len(delib_hist) < max(2, w):
        return 0
    xs = np.asarray(delib_hist[-w:], dtype=float)
    mu = float(np.mean(xs))
    cv = coefficient_of_variation(xs)
    return int((mu <= pod.gaming_mu_max_ms) and (cv <= pod.gaming_cv_max))


def fatigue_detector(delib_hist: np.ndarray, pod: PoDParams) -> int:
    """Long-window vigilance check targeting *fatigue* behaviour.

    Returns 1 when the coefficient of variation over the most recent
    ``fatigue_window`` deliberations exceeds ``fatigue_cv_min`` -- the
    signature of attentional collapse. The composite verifier treats
    this 1 as a *violation*.
    """
    w = pod.fatigue_window
    if delib_hist is None or len(delib_hist) < max(2, w):
        return 0
    xs = np.asarray(delib_hist[-w:], dtype=float)
    cv = coefficient_of_variation(xs)
    return int(cv >= pod.fatigue_cv_min)
