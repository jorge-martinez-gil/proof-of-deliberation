"""Tests for the real-user-data loader, analysis, and CLI plumbing."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pod.config import PoDParams
from pod.realdata.analysis import (
    apply_pod_offline,
    block_regime_table,
    coupling_table,
    regime_classification_score,
    summarise_participants,
)
from pod.realdata.loader import REQUIRED_COLUMNS, load_sessions


def _synth_session(
    participant_id: str,
    seed: int,
    n_per_block: int = 30,
) -> pd.DataFrame:
    """Synthesise a participant whose three blocks reproduce the three regimes.

    * Baseline:   delib_ms scales linearly with c_star (Hick-Hyman).
    * Gaming:     delib_ms is fast and near-constant.
    * Fatigue:    delib_ms is slow with very high variance.
    """
    rng = np.random.default_rng(seed)
    rows = []
    trial_idx = 0
    blocks = [
        ("baseline", "baseline"),
        ("speed_bonus", "gaming"),
        ("long_block", "fatigue"),
    ]
    for block, cond in blocks:
        for k in range(n_per_block):
            c_star = float(rng.uniform(0.0, 1.0))
            bin_name = "low" if c_star < 0.33 else ("mid" if c_star < 0.66 else "high")
            y_true = int(rng.integers(0, 2))
            if cond == "baseline":
                delib = float(rng.normal(400 + 600 * c_star, 80))
                correct_p = 0.92
            elif cond == "gaming":
                delib = float(rng.normal(280, 8))
                correct_p = 0.55
            else:  # fatigue
                delib = float(rng.normal(1200, 750))
                correct_p = 0.65
            delib = max(50.0, delib)
            y_user = y_true if rng.random() < correct_p else (1 - y_true)
            rows.append({
                "participant_id": participant_id,
                "session_id": participant_id,
                "csv_version": 1,
                "block": block,
                "induced_condition": cond,
                "trial_idx": trial_idx,
                "item_id": trial_idx,
                "c_star": c_star,
                "complexity_bin": bin_name,
                "y_true": y_true,
                "y_user": y_user,
                "correct": int(y_user == y_true),
                "delib_ms": delib,
                "t_shown_perf": 1000.0 * trial_idx,
                "t_clicked_perf": 1000.0 * trial_idx + delib,
                "t_shown_iso": "2026-05-29T12:00:00Z",
                "age_bracket": "25-34",
                "experience": "some",
                "started_at_iso": "2026-05-29T11:59:00Z",
            })
            trial_idx += 1
    return pd.DataFrame(rows)


def _write_csvs(tmp_path, n_participants: int = 4) -> str:
    base = tmp_path / "data_real"
    base.mkdir()
    for i in range(n_participants):
        sess = _synth_session(f"p{i:02d}", seed=100 + i)
        sess.to_csv(base / f"pod_session_p{i:02d}.csv", index=False)
    return str(base)


def test_load_sessions_required_columns(tmp_path):
    base = _write_csvs(tmp_path, n_participants=2)
    df = load_sessions(base)
    for c in REQUIRED_COLUMNS:
        assert c in df.columns
    assert df["participant_id"].nunique() == 2
    assert len(df) == 2 * 90  # 30 per block x 3 blocks x 2 participants


def test_load_sessions_dedupes_repeats(tmp_path):
    base = tmp_path / "data_real"
    base.mkdir()
    sess = _synth_session("p00", seed=0)
    sess.to_csv(base / "first.csv", index=False)
    # Same participant, same trial indices -> last write wins.
    sess.to_csv(base / "second.csv", index=False)
    df = load_sessions(str(base))
    assert df["participant_id"].nunique() == 1
    assert len(df) == 90  # not 180


def test_load_sessions_errors_on_missing_columns(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    pd.DataFrame({"a": [1, 2]}).to_csv(bad / "broken.csv", index=False)
    try:
        load_sessions(str(bad))
    except ValueError:
        return
    raise AssertionError("Expected ValueError on missing required columns")


def test_summarise_participants_rows_per_participant_block(tmp_path):
    base = _write_csvs(tmp_path, n_participants=3)
    df = load_sessions(base)
    summ = summarise_participants(df)
    # 3 participants x 3 blocks = 9 rows.
    assert len(summ) == 9
    for c in ("n_trials", "accuracy", "mean_delib_ms", "cv_delib"):
        assert c in summ.columns


def test_coupling_table_positive_for_baseline_block(tmp_path):
    base = _write_csvs(tmp_path, n_participants=5)
    df = load_sessions(base)
    coup = coupling_table(df)
    base_rows = coup[coup["induced_condition"] == "baseline"]
    # Baseline block has delib_ms = a*c_star + noise -> rho strongly positive.
    assert (base_rows["spearman_rho"] > 0.3).mean() >= 0.6


def test_apply_pod_offline_adds_expected_columns(tmp_path):
    base = _write_csvs(tmp_path, n_participants=2)
    df = load_sessions(base)
    aug = apply_pod_offline(df, pod=PoDParams())
    for c in ("pod_gate", "pod_coupling", "pod_gaming", "pod_fatigue", "pod_accept"):
        assert c in aug.columns
        assert set(aug[c].unique()).issubset({0, 1})


def test_block_regime_table_orders_conditions(tmp_path):
    base = _write_csvs(tmp_path, n_participants=3)
    df = load_sessions(base)
    aug = apply_pod_offline(df)
    tbl = block_regime_table(aug)
    assert list(tbl["induced_condition"]) == ["baseline", "gaming", "fatigue"]


def test_pod_rejects_more_in_gaming_than_baseline(tmp_path):
    """The whole point of PoD: gaming block should be flagged more."""
    base = _write_csvs(tmp_path, n_participants=4)
    df = load_sessions(base)
    # Tighten the gaming detector to match the synthetic gaming block
    # (delib ~ N(280, 8)): very short, very low CV.
    pod = PoDParams(
        gaming_window=10,
        gaming_mu_max_ms=320.0,
        gaming_cv_max=0.05,
        coupling_epsilon=-1.0,
    )
    aug = apply_pod_offline(df, pod=pod)
    rates = (
        aug.groupby("induced_condition")["pod_accept"].mean().to_dict()
    )
    assert rates["baseline"] > rates["gaming"]


def test_regime_classification_score_positive(tmp_path):
    """The aggregate score should favour baseline over induced regimes."""
    base = _write_csvs(tmp_path, n_participants=6)
    df = load_sessions(base)
    pod = PoDParams(
        gaming_window=10, gaming_mu_max_ms=320.0, gaming_cv_max=0.05,
        coupling_epsilon=-1.0,
    )
    aug = apply_pod_offline(df, pod=pod)
    score = regime_classification_score(aug)
    assert score["n_participants"] == 6
    assert score["mean_score"] > 0.0


def test_build_pool_pure_unit_helpers():
    """Pure-unit tests of the pool-builder math (no OpenML fetch)."""
    from pod.realdata.build_pool import (
        _predictive_entropy_unit,
        _stratified_sample,
        _stratify_by_quantile,
    )

    # Entropy: maximally uncertain (p=0.5) -> 1.0
    probs = np.array([[0.5, 0.5], [0.99, 0.01], [0.01, 0.99]])
    H = _predictive_entropy_unit(probs)
    assert abs(H[0] - 1.0) < 1e-6
    assert H[1] < 0.15
    assert H[2] < 0.15

    # Stratification: 3 bins on a uniform vector roughly balanced.
    rng = np.random.default_rng(0)
    c = rng.uniform(0.0, 1.0, size=900)
    bins, names = _stratify_by_quantile(c, n_bins=3)
    assert names == ["low", "mid", "high"]
    counts = np.bincount(bins)
    assert counts.min() >= 290 and counts.max() <= 310

    # Stratified sample: exactly n_per_bin from each bin.
    picks = _stratified_sample(bins, n_per_bin=50, rng=np.random.default_rng(1))
    assert picks.size == 150
    by_bin = np.bincount(bins[picks])
    assert (by_bin == 50).all()


def test_cli_main_runs_end_to_end(tmp_path):
    base = _write_csvs(tmp_path, n_participants=4)
    out_dir = tmp_path / "out_real"

    from pod.realdata.cli import main as cli_main
    rc = cli_main([
        "--in", base, "--out", str(out_dir),
        "--coupling-epsilon", "-1.0",
        "--gate-floor-ms", "80.0", "--gate-ceil-ms", "8000.0",
    ])
    assert rc == 0
    for f in (
        "per_participant_block_summary.csv",
        "coupling_per_participant.csv",
        "per_trial_with_pod.csv",
        "block_regime_table.csv",
        "regime_classification_score.json",
        "summary.json",
        "fig_block_accept_rate.pdf",
        "fig_block_accept_rate.png",
        "fig_coupling_by_regime.pdf",
        "fig_coupling_by_regime.png",
    ):
        assert (out_dir / f).exists(), f"Missing output: {f}"
