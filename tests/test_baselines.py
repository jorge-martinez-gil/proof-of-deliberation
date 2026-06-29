"""Unit tests for the comparison baselines (AL, Static, Adaptive)."""

from __future__ import annotations

from pod.baselines import (
    STATIC_GATE_MS,
    accept_adaptive,
    accept_al,
    accept_static,
)
from pod.config import PoDParams


def test_al_accepts_everything():
    assert accept_al() is True
    assert accept_al(123, 456) is True
    assert accept_al(foo="bar") is True


def test_static_threshold_boundary():
    assert accept_static(STATIC_GATE_MS) is True
    assert accept_static(STATIC_GATE_MS - 0.1) is False
    assert accept_static(2 * STATIC_GATE_MS) is True


def test_adaptive_matches_pod_gate():
    pod = PoDParams()
    # Inside the difficulty-adjusted window.
    mu = pod.gate_a * 0.5 + pod.gate_b
    assert accept_adaptive(mu, 0.5, pod) is True
    # Below the floor -> reject.
    assert accept_adaptive(10.0, 0.5, pod) is False
    # Above the ceiling -> reject.
    assert accept_adaptive(1e6, 0.5, pod) is False
