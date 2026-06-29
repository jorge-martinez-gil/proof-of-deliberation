"""Unit tests for the operator simulator."""

from __future__ import annotations

import numpy as np
import pytest

from pod.config import OperatorParams
from pod.operator import simulate_operator


@pytest.fixture
def op() -> OperatorParams:
    return OperatorParams()


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestRegimeMeans:
    """Empirical means under each regime should reflect their parameters."""

    def test_baseline_increases_with_complexity(self, op):
        delibs_low, delibs_high = [], []
        rng = _rng(0)
        for _ in range(2_000):
            _, dl = simulate_operator(rng, 0, 2, 0.0, "baseline", op)
            delibs_low.append(dl)
            _, dh = simulate_operator(rng, 0, 2, 1.0, "baseline", op)
            delibs_high.append(dh)
        assert np.mean(delibs_high) > np.mean(delibs_low) + 100.0

    def test_gaming_is_fast(self, op):
        rng = _rng(1)
        delibs = [
            simulate_operator(rng, 0, 2, 0.5, "gaming", op)[1]
            for _ in range(2_000)
        ]
        assert np.mean(delibs) < op.c_fast + 50.0

    def test_fatigue_is_high_variance(self, op):
        rng = _rng(2)
        b_delibs = [
            simulate_operator(rng, 0, 2, 0.5, "baseline", op)[1]
            for _ in range(2_000)
        ]
        f_delibs = [
            simulate_operator(rng, 0, 2, 0.5, "fatigue", op)[1]
            for _ in range(2_000)
        ]
        assert np.std(f_delibs) > 3 * np.std(b_delibs)


class TestCorrectness:
    def test_baseline_mostly_correct(self, op):
        rng = _rng(0)
        N = 5_000
        hits = sum(
            simulate_operator(rng, 1, 2, 0.5, "baseline", op)[0] == 1
            for _ in range(N)
        )
        # baseline p_correct = 0.965; allow 2% tolerance
        assert hits / N > 0.93

    def test_gaming_mostly_wrong(self, op):
        rng = _rng(0)
        N = 5_000
        hits = sum(
            simulate_operator(rng, 1, 2, 0.5, "gaming", op)[0] == 1
            for _ in range(N)
        )
        # gaming p_correct = 0.22; should be far from 1
        assert hits / N < 0.30

    def test_adversarial_mode_always_flips(self):
        adv = OperatorParams(p_correct_gaming=-1.0)
        rng = _rng(0)
        for _ in range(1_000):
            y_tilde, _ = simulate_operator(rng, 1, 2, 0.5, "gaming", adv)
            assert y_tilde == 0


class TestDelibFloor:
    def test_minimum_50ms(self, op):
        rng = _rng(123)
        for _ in range(200):
            _, dl = simulate_operator(rng, 0, 2, 0.0, "baseline", op)
            assert dl >= 50.0


class TestMulticlass:
    def test_label_in_valid_range(self, op):
        rng = _rng(0)
        n_classes = 6
        for _ in range(2_000):
            y_tilde, _ = simulate_operator(rng, 3, n_classes, 0.5, "fatigue", op)
            assert 0 <= y_tilde < n_classes
