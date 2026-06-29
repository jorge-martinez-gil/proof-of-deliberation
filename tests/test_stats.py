"""Numerical tests for the statistical-analysis module."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pod.stats import (
    ScoreSpec,
    average_ranks,
    bootstrap_ci,
    cohens_d,
    collect_per_run_scores,
    friedman_test,
    nemenyi_critical_diff,
    wilcoxon_holm,
)


def _make_synthetic_scores(seed: int = 0) -> pd.DataFrame:
    """Construct a small scores frame where method ``A`` dominates ``B``."""
    rng = np.random.default_rng(seed)
    rows = []
    for ds in ["d1", "d2", "d3", "d4", "d5"]:
        for run in range(10):
            rows.append({
                "dataset": ds, "method": "A", "run": run,
                "score": float(0.80 + 0.02 * rng.normal()),
            })
            rows.append({
                "dataset": ds, "method": "B", "run": run,
                "score": float(0.60 + 0.02 * rng.normal()),
            })
            rows.append({
                "dataset": ds, "method": "C", "run": run,
                "score": float(0.65 + 0.02 * rng.normal()),
            })
    return pd.DataFrame(rows)


def test_average_ranks_orders_methods_correctly():
    scores = _make_synthetic_scores()
    avg = average_ranks(scores, methods=["A", "B", "C"])
    avg = avg.set_index("method")
    # A is best -> rank 1; B is worst -> rank 3.
    assert avg.loc["A", "avg_rank"] < avg.loc["C", "avg_rank"]
    assert avg.loc["C", "avg_rank"] < avg.loc["B", "avg_rank"]
    assert avg.loc["A", "n_datasets"] == 5


def test_friedman_rejects_with_clear_separation():
    scores = _make_synthetic_scores()
    res = friedman_test(scores, methods=["A", "B", "C"])
    assert res["chi2"] > 0
    # With three completely-separated methods on five datasets, p must be tiny.
    assert res["p_chi2"] < 0.05
    assert res["p_F"] < 0.05
    assert res["n_datasets"] == 5
    assert res["n_methods"] == 3


def test_nemenyi_cd_matches_demsar_table():
    # Known value: k=4, N=8, alpha=0.05 -> CD = 2.569 * sqrt(4*5/(6*8)) ~ 1.6577
    cd = nemenyi_critical_diff(n_methods=4, n_datasets=8, alpha=0.05)
    expected = 2.569 * np.sqrt(4.0 * 5.0 / (6.0 * 8.0))
    assert abs(cd - expected) < 1e-9


def test_nemenyi_cd_alpha10_smaller_than_alpha05():
    cd_05 = nemenyi_critical_diff(n_methods=5, n_datasets=10, alpha=0.05)
    cd_10 = nemenyi_critical_diff(n_methods=5, n_datasets=10, alpha=0.10)
    assert cd_10 < cd_05


def test_cohens_d_zero_for_equal_means():
    x = np.array([0.5, 0.6, 0.5, 0.55, 0.45])
    y = x.copy()
    assert abs(cohens_d(x, y)) < 1e-9


def test_cohens_d_sign_follows_mean_difference():
    x = np.array([0.80, 0.81, 0.79, 0.82, 0.80])
    y = np.array([0.60, 0.61, 0.59, 0.62, 0.60])
    d = cohens_d(x, y)
    assert d > 4.0  # very large positive effect


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(7)
    samples = rng.normal(0.5, 0.05, size=200)
    mean, lo, hi = bootstrap_ci(samples, n_boot=500, alpha=0.05, seed=11)
    assert lo < mean < hi
    # The CI should be relatively narrow for n=200, sigma=0.05.
    assert (hi - lo) < 0.05


def test_wilcoxon_holm_returns_one_row_per_non_reference():
    scores = _make_synthetic_scores()
    df = wilcoxon_holm(scores, methods=["A", "B", "C"], reference="A")
    assert len(df) == 2
    assert set(df["method"]) == {"B", "C"}
    # p_holm must be monotonically non-decreasing after sort by raw p.
    assert df["p_holm"].is_monotonic_increasing


def test_score_spec_end_window():
    df = pd.DataFrame({"t": list(range(100)), "f1": [0.1] * 50 + [0.9] * 50})
    sc = ScoreSpec(mode="end", end_window=50).reduce(df)
    assert abs(sc - 0.9) < 1e-9
    sc_full = ScoreSpec(mode="end", end_window=100).reduce(df)
    assert abs(sc_full - 0.5) < 1e-9


def test_score_spec_auc_normalises_to_unit_interval():
    df = pd.DataFrame({"t": [0.0, 100.0], "f1": [1.0, 1.0]})
    sc = ScoreSpec(mode="auc").reduce(df)
    assert abs(sc - 1.0) < 1e-9


def test_collect_per_run_scores_from_temp_dir(tmp_path):
    base = tmp_path / "out"
    for ds in ["d1", "d2"]:
        runs = base / ds / "runs"
        runs.mkdir(parents=True)
        for m in ["A", "B"]:
            for r in range(3):
                f1 = [0.9 if m == "A" else 0.5] * 100
                pd.DataFrame({"t": list(range(100)), "f1": f1}).to_csv(
                    runs / f"{ds}_{m}_run{r}.csv", index=False
                )
    df = collect_per_run_scores(str(base), score=ScoreSpec(mode="end", end_window=20))
    assert len(df) == 12  # 2 datasets * 2 methods * 3 runs
    a = df[df["method"] == "A"]["score"].mean()
    b = df[df["method"] == "B"]["score"].mean()
    assert abs(a - 0.9) < 1e-9
    assert abs(b - 0.5) < 1e-9


def test_collect_per_run_scores_errors_on_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        collect_per_run_scores(str(empty), score=ScoreSpec())
    except ValueError:
        return
    raise AssertionError("Expected ValueError for empty results directory")
