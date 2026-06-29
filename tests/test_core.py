"""Unit tests for the PoD verification primitives."""

from __future__ import annotations

import numpy as np
import pytest

from pod.config import PoDParams
from pod.core import (
    coupling_check,
    expected_delib,
    fatigue_detector,
    gaming_detector,
    gate_check,
)


@pytest.fixture
def pod() -> PoDParams:
    """Default PoD parameters used as a stable baseline."""
    return PoDParams()


class TestExpectedDelib:
    def test_linear_in_complexity(self, pod):
        c0 = expected_delib(0.0, pod)
        c1 = expected_delib(1.0, pod)
        assert c0 == pytest.approx(pod.gate_b)
        assert c1 == pytest.approx(pod.gate_a + pod.gate_b)


class TestGateCheck:
    def test_accepts_response_at_mean(self, pod):
        mu = expected_delib(0.5, pod)
        assert gate_check(mu, 0.5, pod) == 1

    def test_rejects_implausibly_fast(self, pod):
        assert gate_check(1.0, 0.5, pod) == 0

    def test_rejects_implausibly_slow(self, pod):
        assert gate_check(1e6, 0.5, pod) == 0

    def test_floor_is_absolute(self, pod):
        # Below the floor, no admission window can include the response.
        assert gate_check(pod.gate_floor_ms - 1.0, 0.0, pod) == 0


class TestCouplingCheck:
    def test_admits_when_history_short(self, pod):
        comp = np.zeros(10)
        delib = np.zeros(10)
        assert coupling_check(comp, delib, pod) == 1

    def test_admits_when_positively_coupled(self, pod):
        rng = np.random.default_rng(0)
        comp = rng.uniform(size=pod.coupling_window)
        delib = comp * 1000 + rng.normal(0, 5, size=pod.coupling_window)
        assert coupling_check(comp, delib, pod) == 1

    def test_rejects_when_decoupled(self, pod):
        rng = np.random.default_rng(1)
        comp = rng.uniform(size=pod.coupling_window)
        delib = rng.uniform(size=pod.coupling_window)
        # Many independent draws -> rho close to 0 -> below epsilon
        assert coupling_check(comp, delib, pod) == 0


class TestGamingDetector:
    def test_zero_when_history_short(self, pod):
        delib = np.full(2, 200.0)
        assert gaming_detector(delib, pod) == 0

    def test_detects_fast_constant_clicking(self, pod):
        # Fast and near-constant -> both gaming conditions fire.
        delib = np.full(pod.gaming_window, 200.0)
        assert gaming_detector(delib, pod) == 1

    def test_does_not_fire_on_slow_responses(self, pod):
        delib = np.full(pod.gaming_window, 1500.0)
        assert gaming_detector(delib, pod) == 0


class TestFatigueDetector:
    def test_zero_when_history_short(self, pod):
        delib = np.full(2, 500.0)
        assert fatigue_detector(delib, pod) == 0

    def test_detects_high_variance(self, pod):
        rng = np.random.default_rng(7)
        delib = rng.normal(800, 800, size=pod.fatigue_window)
        delib = np.clip(delib, 50, None)
        assert fatigue_detector(delib, pod) == 1

    def test_does_not_fire_on_stable_rhythm(self, pod):
        rng = np.random.default_rng(7)
        delib = rng.normal(800, 30, size=pod.fatigue_window)
        assert fatigue_detector(delib, pod) == 0
