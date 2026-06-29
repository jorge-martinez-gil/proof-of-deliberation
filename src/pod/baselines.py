"""
Label-acceptance baselines compared against PoD.

This module ships eleven methods organised into four families:

* Classical AL / static-gating: AL, StaticGating, AdaptiveGating.
* Content-based annotation-quality (single-pass single-annotator
  adaptations from the crowdsourcing literature):
  WorkerQuality (Dawid & Skene 1979), Raykar (Raykar et al. JMLR 2010),
  MACE (Hovy et al. NAACL 2013), IEThresh (Donmez & Carbonell 2008).
* Process-based: PoD (this work).
* Ablations: PoD-NoGate, PoD-NoCoupling, PoD-NoVigilance.

See docs/ANNOTATION_QUALITY_COMPARISON.md for the methodological
positioning and the assumption table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pod.config import PoDParams
from pod.core import gate_check

# ---------------------------------------------------------------------------
# Classical AL / static gating
# ---------------------------------------------------------------------------
STATIC_GATE_MS = 510.0
"""Fixed deliberation-time threshold (ms) used by ``StaticGating``."""


def accept_al(*_args, **_kwargs) -> bool:
    """Standard active-learning baseline: accept every queried label."""
    return True


def accept_static(delib_ms: float) -> bool:
    """Static-gating baseline: accept iff deliberation time >= 510 ms."""
    return delib_ms >= STATIC_GATE_MS


def accept_adaptive(delib_ms: float, complexity: float, pod: PoDParams) -> bool:
    """Adaptive-gating baseline: accept iff the PoD ``V_t`` check passes."""
    return bool(gate_check(delib_ms, complexity, pod))


# ---------------------------------------------------------------------------
# WorkerQuality (single-annotator Dawid-Skene)
# ---------------------------------------------------------------------------
WORKER_QUALITY_THRESHOLD = 0.5
WORKER_QUALITY_PRIOR = 1.0


@dataclass
class WorkerQualityState:
    """Online confusion-matrix estimator for the ``WorkerQuality`` baseline."""

    n_classes: int
    counts: np.ndarray

    @classmethod
    def fresh(cls, n_classes: int) -> "WorkerQualityState":
        return cls(
            n_classes=int(n_classes),
            counts=np.full(
                (int(n_classes), int(n_classes)),
                WORKER_QUALITY_PRIOR,
                dtype=float,
            ),
        )

    def likelihood_matrix(self) -> np.ndarray:
        row_sum = self.counts.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0.0, 1.0, row_sum)
        return self.counts / row_sum


def posterior_label_probability(
    proba_model: np.ndarray,
    y_tilde: int,
    state: WorkerQualityState,
) -> float:
    L = state.likelihood_matrix()
    p_x = np.asarray(proba_model, dtype=float).reshape(-1)
    if p_x.size != state.n_classes:
        raise ValueError(
            f"proba_model has {p_x.size} entries, expected {state.n_classes}"
        )
    joint = L[:, int(y_tilde)] * p_x
    total = float(joint.sum())
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    return float(joint[int(y_tilde)] / total)


def accept_worker_quality(
    proba_model: np.ndarray,
    y_tilde: int,
    state: WorkerQualityState,
    threshold: float = WORKER_QUALITY_THRESHOLD,
) -> bool:
    p = posterior_label_probability(proba_model, y_tilde, state)
    return bool(p >= float(threshold))


def update_worker_quality(
    state: WorkerQualityState,
    y_pred_model: int,
    y_tilde: int,
) -> None:
    k = int(y_pred_model)
    l = int(y_tilde)
    if 0 <= k < state.n_classes and 0 <= l < state.n_classes:
        state.counts[k, l] += 1.0


# ---------------------------------------------------------------------------
# Raykar-Online (streaming single-annotator Raykar et al., JMLR 2010)
# ---------------------------------------------------------------------------
RAYKAR_THRESHOLD = 0.5
RAYKAR_PRIOR = 1.0


@dataclass
class RaykarOnlineState:
    """Streaming single-annotator adaptation of Raykar et al. (JMLR 2010).

    Maintains a soft confusion matrix updated by distributing each unit
    of evidence over rows according to the classifier's predictive
    distribution. This is the streaming surrogate of Raykar's joint EM
    E-step; WorkerQuality is the hard-MAP limit.
    """

    n_classes: int
    counts: np.ndarray

    @classmethod
    def fresh(cls, n_classes: int) -> "RaykarOnlineState":
        return cls(
            n_classes=int(n_classes),
            counts=np.full(
                (int(n_classes), int(n_classes)),
                RAYKAR_PRIOR,
                dtype=float,
            ),
        )

    def likelihood_matrix(self) -> np.ndarray:
        row_sum = self.counts.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0.0, 1.0, row_sum)
        return self.counts / row_sum


def posterior_label_probability_raykar(
    proba_model: np.ndarray,
    y_tilde: int,
    state: RaykarOnlineState,
) -> float:
    L = state.likelihood_matrix()
    p_x = np.asarray(proba_model, dtype=float).reshape(-1)
    if p_x.size != state.n_classes:
        raise ValueError(
            f"proba_model has {p_x.size} entries, expected {state.n_classes}"
        )
    joint = L[:, int(y_tilde)] * p_x
    total = float(joint.sum())
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    return float(joint[int(y_tilde)] / total)


def accept_raykar(
    proba_model: np.ndarray,
    y_tilde: int,
    state: RaykarOnlineState,
    threshold: float = RAYKAR_THRESHOLD,
) -> bool:
    p = posterior_label_probability_raykar(proba_model, y_tilde, state)
    return bool(p >= float(threshold))


def update_raykar(
    state: RaykarOnlineState,
    proba_model: np.ndarray,
    y_tilde: int,
) -> None:
    p = np.asarray(proba_model, dtype=float).reshape(-1)
    if p.size != state.n_classes:
        return
    l = int(y_tilde)
    if not 0 <= l < state.n_classes:
        return
    state.counts[:, l] += p


# ---------------------------------------------------------------------------
# MACE-Online (streaming Hovy et al., NAACL 2013)
# ---------------------------------------------------------------------------
MACE_THRESHOLD = 0.5
MACE_PRIOR_ALPHA = 1.0
MACE_PRIOR_BETA = 1.0


@dataclass
class MACEOnlineState:
    """Streaming MACE: Beta-Binomial posterior on operator competence."""

    alpha: float
    beta: float
    n_obs: int

    @classmethod
    def fresh(cls) -> "MACEOnlineState":
        return cls(
            alpha=float(MACE_PRIOR_ALPHA),
            beta=float(MACE_PRIOR_BETA),
            n_obs=0,
        )

    def posterior_mean(self) -> float:
        return float(self.alpha / max(self.alpha + self.beta, 1e-12))


def accept_mace(state: MACEOnlineState, threshold: float = MACE_THRESHOLD) -> bool:
    return bool(state.posterior_mean() >= float(threshold))


def update_mace(state: MACEOnlineState, proba_model: np.ndarray, y_tilde: int) -> None:
    p = np.asarray(proba_model, dtype=float).reshape(-1)
    if p.size == 0:
        return
    y_map = int(np.argmax(p))
    if int(y_tilde) == y_map:
        state.alpha += 1.0
    else:
        state.beta += 1.0
    state.n_obs += 1


# ---------------------------------------------------------------------------
# IEThresh (Donmez & Carbonell, ECML 2008)
# ---------------------------------------------------------------------------
IETHRESH_THRESHOLD = 0.5
IETHRESH_Z = 1.96
IETHRESH_WARMUP = 10


@dataclass
class IEThreshState:
    """Streaming IEThresh: Hoeffding LCB on operator agreement."""

    n_correct: int
    n_total: int

    @classmethod
    def fresh(cls) -> "IEThreshState":
        return cls(n_correct=0, n_total=0)

    def lower_bound(self, z: float = IETHRESH_Z) -> float:
        if self.n_total <= 0:
            return 0.0
        mu = self.n_correct / self.n_total
        margin = float(z) * math.sqrt(max(0.0, mu * (1.0 - mu)) / max(1, self.n_total))
        return float(max(0.0, mu - margin))


def accept_iethresh(
    state: IEThreshState,
    threshold: float = IETHRESH_THRESHOLD,
    warmup: int = IETHRESH_WARMUP,
    z: float = IETHRESH_Z,
) -> bool:
    if state.n_total < int(warmup):
        return True
    return bool(state.lower_bound(z=z) >= float(threshold))


def update_iethresh(state: IEThreshState, proba_model: np.ndarray, y_tilde: int) -> None:
    p = np.asarray(proba_model, dtype=float).reshape(-1)
    if p.size == 0:
        return
    y_map = int(np.argmax(p))
    state.n_total += 1
    if int(y_tilde) == y_map:
        state.n_correct += 1


# ---------------------------------------------------------------------------
# PoD ablations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PoDAblation:
    use_gate: bool = True
    use_coupling: bool = True
    use_vigilance: bool = True


PoD_ABLATIONS = {
    "PoD": PoDAblation(use_gate=True, use_coupling=True, use_vigilance=True),
    "PoD-NoGate": PoDAblation(use_gate=False, use_coupling=True, use_vigilance=True),
    "PoD-NoCoupling": PoDAblation(use_gate=True, use_coupling=False, use_vigilance=True),
    "PoD-NoVigilance": PoDAblation(use_gate=True, use_coupling=True, use_vigilance=False),
}


def is_pod_family(method: str) -> bool:
    return method in PoD_ABLATIONS


def get_pod_ablation(method: str) -> Optional[PoDAblation]:
    return PoD_ABLATIONS.get(method)


__all__ = [
    "IETHRESH_THRESHOLD",
    "IETHRESH_WARMUP",
    "IETHRESH_Z",
    "IEThreshState",
    "MACE_PRIOR_ALPHA",
    "MACE_PRIOR_BETA",
    "MACE_THRESHOLD",
    "MACEOnlineState",
    "PoD_ABLATIONS",
    "PoDAblation",
    "RAYKAR_PRIOR",
    "RAYKAR_THRESHOLD",
    "RaykarOnlineState",
    "STATIC_GATE_MS",
    "WORKER_QUALITY_PRIOR",
    "WORKER_QUALITY_THRESHOLD",
    "WorkerQualityState",
    "accept_adaptive",
    "accept_al",
    "accept_iethresh",
    "accept_mace",
    "accept_raykar",
    "accept_static",
    "accept_worker_quality",
    "get_pod_ablation",
    "is_pod_family",
    "posterior_label_probability",
    "posterior_label_probability_raykar",
    "update_iethresh",
    "update_mace",
    "update_raykar",
    "update_worker_quality",
]
