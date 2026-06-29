"""Unit tests for the Synth-Boundary stream generator."""

from __future__ import annotations

import numpy as np

from pod.config import RegimeSchedule, SynthParams
from pod.streams.synth import generate_synth_boundary_pool_regime


def test_synth_shapes_are_consistent():
    synth = SynthParams(d=5)
    schedule = RegimeSchedule(50, 50, 50)
    X, y, c_star = generate_synth_boundary_pool_regime(
        seed=0, n=300, synth=synth, schedule=schedule, init_fit=10
    )
    assert X.shape == (300, 5)
    assert y.shape == (300,)
    assert c_star.shape == (300,)
    assert set(np.unique(y).tolist()).issubset({0, 1})


def test_synth_complexity_in_unit_interval():
    synth = SynthParams()
    schedule = RegimeSchedule(50, 50, 50)
    _, _, c_star = generate_synth_boundary_pool_regime(
        seed=42, n=500, synth=synth, schedule=schedule, init_fit=20
    )
    assert np.all((c_star >= 0.0) & (c_star <= 1.0))


def test_synth_is_deterministic_given_seed():
    synth = SynthParams()
    schedule = RegimeSchedule(20, 20, 20)
    X1, y1, c1 = generate_synth_boundary_pool_regime(
        seed=7, n=200, synth=synth, schedule=schedule, init_fit=5
    )
    X2, y2, c2 = generate_synth_boundary_pool_regime(
        seed=7, n=200, synth=synth, schedule=schedule, init_fit=5
    )
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(c1, c2)


def test_schedule_phase_dispatch():
    schedule = RegimeSchedule(10, 5, 7)
    assert schedule.phase(0) == "baseline"
    assert schedule.phase(9) == "baseline"
    assert schedule.phase(10) == "gaming"
    assert schedule.phase(14) == "gaming"
    assert schedule.phase(15) == "fatigue"
    assert schedule.phase(21) == "fatigue"
    assert schedule.total() == 22
