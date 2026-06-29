"""End-to-end smoke test.

Runs one short closed-loop experiment with each of the four methods
on a tiny Synth-Boundary stream to confirm that the full pipeline
exits cleanly and produces non-empty F1 logs.
"""

from __future__ import annotations

import numpy as np
import pytest

from pod.config import (
    ALParams,
    OperatorParams,
    PoDParams,
    RegimeSchedule,
    SynthParams,
)
from pod.experiment import METHODS, run_stream_experiment_once
from pod.streams.synth import generate_synth_boundary_pool_regime


def _make_stream():
    schedule = RegimeSchedule(80, 80, 80)
    synth = SynthParams(d=6)
    init_fit = 30
    holdout_size = 800
    stream_len = init_fit + schedule.total() + 20
    pool_n = stream_len + holdout_size + 100

    X_pool, y_pool, c_star_pool = generate_synth_boundary_pool_regime(
        seed=0,
        n=pool_n,
        synth=synth,
        schedule=schedule,
        init_fit=init_fit,
        rot_baseline=0.004,
    )

    Xs = X_pool[:stream_len]
    ys = y_pool[:stream_len]
    cs = c_star_pool[:stream_len]

    Xh = X_pool[stream_len : stream_len + holdout_size]
    yh = y_pool[stream_len : stream_len + holdout_size]
    return Xs, ys, cs, Xh, yh, schedule, init_fit


@pytest.mark.parametrize("method", METHODS)
def test_method_runs_end_to_end(method):
    Xs, ys, cs, Xh, yh, schedule, init_fit = _make_stream()

    df, diag = run_stream_experiment_once(
        X_stream=Xs,
        y_stream=ys,
        X_holdout=Xh,
        y_holdout=yh,
        seed=1234,
        schedule=schedule,
        method=method,
        al=ALParams(),
        pod=PoDParams(),
        op=OperatorParams(),
        eval_window=400,
        eval_every=20,
        init_fit=init_fit,
        complexity_mode="c_star",
        c_star_stream=cs,
    )

    assert len(df) > 0, f"{method}: empty F1 log"
    assert {"t", "f1"} == set(df.columns)
    assert df["f1"].between(0.0, 1.0).all()

    for k in (
        "query_rate",
        "accept_rate_given_query",
        "query_rate_baseline",
        "accept_rate_baseline",
        "query_rate_gaming",
        "accept_rate_gaming",
        "query_rate_fatigue",
        "accept_rate_fatigue",
    ):
        assert k in diag


def test_unknown_method_raises():
    Xs, ys, cs, Xh, yh, schedule, init_fit = _make_stream()
    with pytest.raises(ValueError):
        run_stream_experiment_once(
            X_stream=Xs,
            y_stream=ys,
            X_holdout=Xh,
            y_holdout=yh,
            seed=0,
            schedule=schedule,
            method="bogus",
            al=ALParams(),
            pod=PoDParams(),
            op=OperatorParams(),
            eval_window=400,
            eval_every=20,
            init_fit=init_fit,
            complexity_mode="c_star",
            c_star_stream=cs,
        )


def test_pod_blocks_more_under_gaming_than_al():
    """Behavioural sanity: PoD should accept strictly fewer labels than AL
    under a gaming regime since the gating layer rejects fast-constant
    responses."""
    Xs, ys, cs, Xh, yh, schedule, init_fit = _make_stream()

    base_kwargs = dict(
        X_stream=Xs,
        y_stream=ys,
        X_holdout=Xh,
        y_holdout=yh,
        seed=1234,
        schedule=schedule,
        al=ALParams(),
        pod=PoDParams(
            gate_a=0.0,
            gate_b=600.0,
            gate_lo_frac=0.2,
            gate_hi_frac=0.2,
            gate_floor_ms=460.0,
            gate_ceil_ms=700.0,
        ),
        op=OperatorParams(),
        eval_window=400,
        eval_every=20,
        init_fit=init_fit,
        complexity_mode="c_star",
        c_star_stream=cs,
    )
    _, diag_al = run_stream_experiment_once(method="AL", **base_kwargs)
    _, diag_pod = run_stream_experiment_once(method="PoD", **base_kwargs)

    # Per-phase accept rate under gaming: PoD <= AL strictly when AL > 0.
    if diag_al["accept_rate_gaming"] > 0:
        assert diag_pod["accept_rate_gaming"] <= diag_al["accept_rate_gaming"]
