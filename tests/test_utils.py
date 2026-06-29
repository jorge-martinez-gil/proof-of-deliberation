"""Unit tests for low-level numerical utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pod.utils import (
    coefficient_of_variation,
    entropy_from_proba,
    entropy_unit,
    parse_int_list,
    sigmoid,
    smooth_series,
    softmax_stable,
    spearman_rho,
)


class TestSigmoid:
    def test_zero_maps_to_half(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_large_positive_saturates(self):
        assert sigmoid(np.array([1e3]))[0] == pytest.approx(1.0)

    def test_large_negative_saturates(self):
        assert sigmoid(np.array([-1e3]))[0] == pytest.approx(0.0)


class TestSoftmaxStable:
    def test_rowwise_sums_to_one(self):
        z = np.random.default_rng(0).normal(0, 10, size=(8, 5))
        p = softmax_stable(z)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(8), atol=1e-10)

    def test_handles_extreme_logits(self):
        z = np.array([[1e4, -1e4, 0.0]])
        p = softmax_stable(z)
        assert math.isclose(p.sum(), 1.0, rel_tol=1e-9)
        assert p[0, 0] == pytest.approx(1.0)


class TestEntropy:
    def test_uniform_two_class_is_log2(self):
        ent = entropy_from_proba(np.array([0.5, 0.5]))
        assert ent == pytest.approx(math.log(2))

    def test_degenerate_distribution_is_zero(self):
        ent = entropy_from_proba(np.array([1.0, 0.0, 0.0]))
        assert ent == pytest.approx(0.0, abs=1e-9)

    def test_entropy_unit_in_unit_interval(self):
        for n in (2, 3, 6):
            p = np.ones(n) / n
            assert entropy_unit(entropy_from_proba(p), n) == pytest.approx(1.0)
            assert entropy_unit(0.0, n) == pytest.approx(0.0)


class TestCV:
    def test_zero_when_too_short(self):
        assert coefficient_of_variation(np.array([1.0])) == 0.0

    def test_constant_signal_has_zero_cv(self):
        assert coefficient_of_variation(np.full(10, 5.0)) == pytest.approx(0.0)

    def test_known_value(self):
        xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = float(np.std(xs, ddof=0) / np.mean(xs))
        assert coefficient_of_variation(xs) == pytest.approx(expected)


class TestSpearman:
    def test_monotone_increasing_is_one(self):
        x = np.arange(20, dtype=float)
        assert spearman_rho(x, x ** 2) == pytest.approx(1.0)

    def test_monotone_decreasing_is_minus_one(self):
        x = np.arange(20, dtype=float)
        assert spearman_rho(x, -x ** 2) == pytest.approx(-1.0)

    def test_constant_returns_nan(self):
        x = np.arange(20, dtype=float)
        rho = spearman_rho(x, np.full_like(x, 7.0))
        assert math.isnan(rho)


class TestParseIntList:
    def test_basic(self):
        assert parse_int_list("1, 2 ,3") == [1, 2, 3]

    def test_empty_yields_empty(self):
        assert parse_int_list("  , ,") == []


class TestSmoothSeries:
    def test_passthrough_when_window_one(self):
        x = np.arange(10, dtype=float)
        np.testing.assert_allclose(smooth_series(x, 1), x)

    def test_window_larger_than_series(self):
        x = np.arange(5, dtype=float)
        out = smooth_series(x, 100)
        assert out.shape == x.shape
