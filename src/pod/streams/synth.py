"""
Synth-Boundary stream generator.

Generates a binary classification stream from a slowly rotating linear
separator in :math:`\\mathbb{R}^d`. Intrinsic complexity of each sample
is defined as :math:`C^{*}_t = \\exp(-\\lambda \\,|w_t \\cdot x_t|)`, so
points close to the current boundary are harder. The separator's
rotation rate is regime-dependent so that the experiment can isolate
the effect of supervision degradation from that of concept drift.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from pod.config import RegimeSchedule, SynthParams


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(1e-12, n)


def generate_synth_boundary_pool_regime(
    seed: int,
    n: int,
    synth: SynthParams,
    schedule: RegimeSchedule,
    init_fit: int,
    rot_baseline: float = 0.004,
    rot_gaming: float = 0.0,
    rot_fatigue: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute a pool large enough for the entire stream-and-holdout.

    The boundary is rotated step-by-step in a single sweep so that the
    intrinsic complexity ``c_star`` reflects the *instantaneous* margin
    at the time index ``t``. The first ``init_fit`` steps are emitted
    inside the Baseline regime regardless of the regime schedule, since
    they precede the closed-loop start.

    Parameters
    ----------
    seed : int
        Seed for the per-pool PRNG.
    n : int
        Total number of points to generate.
    synth : SynthParams
        Dimensionality, complexity decay rate, and noise scale.
    schedule : RegimeSchedule
        Regime sequence used to determine rotation rate per step.
    init_fit : int
        Number of points before regime indexing starts (these are
        treated as a warm-up).
    rot_baseline, rot_gaming, rot_fatigue : float
        Per-step angular rotation in radians for each regime.

    Returns
    -------
    X : np.ndarray of shape ``(n, d)``
        Feature pool.
    y : np.ndarray of shape ``(n,)``
        Binary labels under the rotating boundary.
    c_star : np.ndarray of shape ``(n,)``
        Intrinsic complexity ``c_star`` per row.
    """
    rng = np.random.default_rng(seed)
    d = synth.d

    X = rng.normal(0.0, synth.noise_std, size=(n, d)).astype(float)

    w0 = _normalize(rng.normal(0.0, 1.0, size=(d,)))
    u = rng.normal(0.0, 1.0, size=(d,))
    u = u - np.dot(u, w0) * w0
    u = _normalize(u)

    y = np.zeros(n, dtype=int)
    c_star = np.zeros(n, dtype=float)

    theta = 0.0
    for t in range(n):
        t_rel = t - init_fit
        if t_rel < 0:
            phase = "baseline"
        else:
            phase = schedule.phase(t_rel)

        if phase == "baseline":
            theta += rot_baseline
        elif phase == "gaming":
            theta += rot_gaming
        else:
            theta += rot_fatigue

        wt = _normalize(math.cos(theta) * w0 + math.sin(theta) * u)
        margin = float(np.dot(wt, X[t]))
        y[t] = 1 if margin >= 0 else 0
        c_star[t] = float(math.exp(-synth.lambda_complexity * abs(margin)))

    return X, y, c_star
