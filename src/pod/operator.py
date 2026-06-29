"""
Stochastic operator simulator.

Implements the regime-conditioned operator described in Section 6.3
(*Supervision degradation scenarios*). Three regimes are supported:

* ``baseline`` -- deliberation time is monotone in complexity; high
  ``p_correct``. Matches the Hick-Hyman regime validated on real ATC
  logs in Section 6.5.
* ``gaming``   -- fast, near-constant deliberation; very low
  ``p_correct``. Represents reflexive / mechanical confirmation.
* ``fatigue``  -- slow, high-variance deliberation; degraded
  ``p_correct``. Represents attentional collapse.

The simulator is *intentionally conservative*: it draws response times
from Gaussian distributions parameterised by the cognitive-ergonomics
literature rather than from the simulator's own training set, so PoD's
verification layer cannot trivially overfit to it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pod.config import OperatorParams


def simulate_operator(
    rng: np.random.Generator,
    y_true: int,
    n_classes: int,
    complexity: float,
    phase: str,
    op: OperatorParams,
) -> Tuple[int, float]:
    """Draw an ``(label, deliberation_ms)`` pair from the operator process.

    Parameters
    ----------
    rng : numpy.random.Generator
        Per-run PRNG; passed in explicitly to keep the simulator pure.
    y_true : int
        Ground-truth class index. Used only to determine whether to flip
        the returned label according to the regime's ``p_correct``.
    n_classes : int
        Number of classes in the stream's label space.
    complexity : float
        Task complexity ``C_t`` (normalised entropy or ``c_star``).
    phase : {'baseline', 'gaming', 'fatigue'}
        Regime tag from :meth:`pod.config.RegimeSchedule.phase`.
    op : OperatorParams
        Operator-process hyperparameters.

    Returns
    -------
    label : int
        The reported label ``y_tilde`` (may equal or differ from
        ``y_true`` depending on the regime).
    delib_ms : float
        Reported deliberation time in milliseconds, clipped at a 50 ms
        physiological floor.

    Notes
    -----
    Setting ``p_correct < 0`` for a regime activates *adversarial mode*
    (always wrong); used for stress tests in the ablation study.
    """
    if phase == "baseline":
        delib = rng.normal(op.k * complexity + op.c, op.sigma)
        p_corr = op.p_correct_baseline
    elif phase == "gaming":
        delib = rng.normal(op.c_fast, op.eps_fast)
        p_corr = op.p_correct_gaming
    elif phase == "fatigue":
        delib = rng.normal(op.c_slow, op.sigma_high)
        p_corr = op.p_correct_fatigue
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown operator phase: {phase!r}")

    delib = float(max(50.0, delib))

    # Adversarial mode: deterministic flip.
    if p_corr < 0.0:
        if n_classes <= 2:
            return 1 - int(y_true), delib
        return int((int(y_true) + 1) % n_classes), delib

    if rng.random() < p_corr:
        return int(y_true), delib

    if n_classes <= 2:
        return 1 - int(y_true), delib

    choices = [c for c in range(n_classes) if c != y_true]
    return int(rng.choice(choices)), delib
