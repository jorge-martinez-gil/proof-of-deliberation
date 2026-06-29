"""
TKDE-grade statistical analysis for the PoD experiment outputs.

Reads the per-run CSV logs produced by
:func:`pod.experiment.run_suite_generic` and computes the suite of
statistical tests commonly required for IEEE Transactions on Knowledge
and Data Engineering submissions:

* :func:`friedman_test`           -- non-parametric omnibus test across
  methods on multiple datasets (Demsar, JMLR 2006).
* :func:`nemenyi_critical_diff`   -- Nemenyi post-hoc critical difference
  used to draw Demsar-style CD diagrams.
* :func:`wilcoxon_holm`           -- pairwise Wilcoxon signed-rank with
  Holm-Bonferroni family-wise error control.
* :func:`cohens_d`                -- standardised effect size for paired
  samples.
* :func:`bootstrap_ci`            -- percentile bootstrap confidence
  intervals for the mean F1 of each method.
* :func:`per_regime_anova`        -- one-way ANOVA over the three operator
  regimes (Baseline, Gaming, Fatigue).

The CLI entry point :func:`main` (exposed as ``pod-stats``) walks the
output directory, reads every ``*_runN.csv`` it finds, computes the full
suite, and writes the result tables plus the CD diagram under a
``stats/`` subdirectory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pod.utils import ensure_dir
from pod.viz import set_pub_style

# ---------------------------------------------------------------------------
# Critical-value tables for the Nemenyi post-hoc test.
# Source: Demsar 2006, JMLR (Table 5), upper bounds on the studentized
# range distribution divided by sqrt(2). Indexed by number of methods k.
# ---------------------------------------------------------------------------
NEMENYI_Q_ALPHA: Dict[float, Dict[int, float]] = {
    0.05: {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164,
    },
    0.10: {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
        8: 2.780, 9: 2.855, 10: 2.920,
    },
}


# ---------------------------------------------------------------------------
# Aggregation: per-run scalar score from the F1 trajectories
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreSpec:
    """How to reduce a per-run F1 trajectory to a single scalar score.

    Two reductions are supported, mirroring the two presentations used in
    the paper: ``end`` (mean F1 of the final ``end_window`` evaluation
    points) and ``auc`` (trapezoidal integral over the whole trajectory,
    normalised to ``[0, 1]``).
    """

    mode: str = "end"
    end_window: int = 50

    def reduce(self, df: pd.DataFrame) -> float:
        if "f1" not in df.columns:
            raise ValueError("Run CSV is missing required column 'f1'")
        f1 = df["f1"].to_numpy(dtype=float)
        if f1.size == 0:
            return float("nan")
        if self.mode == "end":
            k = max(1, min(int(self.end_window), f1.size))
            return float(np.mean(f1[-k:]))
        if self.mode == "auc":
            # Normalised area under the F1 trajectory: integral / horizon.
            t = df["t"].to_numpy(dtype=float)
            if t.size < 2:
                return float(f1.mean())
            area = float(np.trapz(f1, t))
            horizon = float(t[-1] - t[0])
            if horizon <= 0.0:
                return float(f1.mean())
            return area / horizon
        raise ValueError(f"Unknown ScoreSpec.mode={self.mode!r}")


def collect_per_run_scores(
    base_dir: str,
    score: ScoreSpec,
) -> pd.DataFrame:
    """Walk ``base_dir`` and build a tidy frame of per-run scalar scores.

    Each row carries ``(dataset, method, run, score)``. Expects the layout
    produced by :func:`pod.experiment.run_suite_generic`:
    ``<base>/<dataset>/runs/<name>_<method>_run<N>.csv``.
    """
    rows: List[Dict[str, object]] = []

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Results directory not found: {base_dir}")

    for dataset in sorted(os.listdir(base_dir)):
        runs_dir = os.path.join(base_dir, dataset, "runs")
        if not os.path.isdir(runs_dir):
            continue

        for fname in sorted(os.listdir(runs_dir)):
            if not fname.endswith(".csv"):
                continue
            # Expected pattern: <name>_<method>_run<N>.csv
            stem = fname[:-4]
            if "_run" not in stem:
                continue
            head, run_str = stem.rsplit("_run", 1)
            try:
                run_idx = int(run_str)
            except ValueError:
                continue
            if "_" not in head:
                continue
            _, method = head.rsplit("_", 1)

            path = os.path.join(runs_dir, fname)
            try:
                df = pd.read_csv(path)
            except Exception:  # pragma: no cover - corrupt files are skipped
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "run": run_idx,
                    "score": score.reduce(df),
                }
            )

    if not rows:
        raise ValueError(
            f"No per-run CSVs found under {base_dir!r}. "
            "Have you run `pod-experiments` yet?"
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Friedman + Nemenyi
# ---------------------------------------------------------------------------
def average_ranks(scores: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    """Mean rank of each method across datasets (lower rank = better).

    ``scores`` must contain one row per ``(dataset, method)`` pair; if
    multiple runs are present they are averaged first. Ties receive
    average ranks, as required by Friedman.
    """
    pivot = (
        scores.groupby(["dataset", "method"])["score"]
        .mean()
        .unstack("method")
    )
    pivot = pivot[methods]
    # Negate so that higher F1 -> rank 1 (best); average-ranked ties.
    ranks = (-pivot).rank(axis=1, method="average")
    avg = ranks.mean(axis=0).rename("avg_rank").to_frame()
    avg["n_datasets"] = len(pivot)
    return avg.reset_index().rename(columns={"index": "method"})


def friedman_test(
    scores: pd.DataFrame, methods: List[str]
) -> Dict[str, float]:
    """Friedman omnibus test across datasets and methods.

    Returns the chi-square statistic, Iman-Davenport F variant, and the
    associated p-values. Uses ``scipy.stats`` when available; the F
    variant is recommended over the raw chi-square for small N and k.
    """
    pivot = (
        scores.groupby(["dataset", "method"])["score"]
        .mean()
        .unstack("method")
    )
    pivot = pivot[methods].dropna(how="any")
    N = int(len(pivot))
    k = int(len(methods))
    if N < 2 or k < 2:
        return {"chi2": float("nan"), "p_chi2": float("nan"),
                "F": float("nan"), "p_F": float("nan"),
                "n_datasets": float(N), "n_methods": float(k)}

    ranks = (-pivot).rank(axis=1, method="average")
    R = ranks.mean(axis=0).to_numpy()
    chi2 = (
        12.0 * N / (k * (k + 1.0))
        * float(np.sum(R ** 2) - k * (k + 1.0) ** 2 / 4.0)
    )
    # Iman-Davenport correction (better small-sample behaviour).
    denom = N * (k - 1) - chi2
    if denom <= 0:
        F = float("inf")
    else:
        F = float((N - 1.0) * chi2 / denom)

    try:
        from scipy.stats import chi2 as _chi2  # type: ignore[import-not-found]
        from scipy.stats import f as _f  # type: ignore[import-not-found]
        p_chi2 = float(1.0 - _chi2.cdf(chi2, df=k - 1))
        df1, df2 = k - 1, (k - 1) * (N - 1)
        p_F = float(1.0 - _f.cdf(F, df1, df2)) if df2 > 0 else float("nan")
    except Exception:  # pragma: no cover - scipy is a hard dep, fallback for safety
        p_chi2 = float("nan")
        p_F = float("nan")

    return {
        "chi2": float(chi2),
        "p_chi2": p_chi2,
        "F": float(F),
        "p_F": p_F,
        "n_datasets": float(N),
        "n_methods": float(k),
    }


def nemenyi_critical_diff(
    n_methods: int, n_datasets: int, alpha: float = 0.05
) -> float:
    """Nemenyi critical-difference value ``CD``.

    Two methods differ significantly (at level ``alpha``) iff their mean
    ranks differ by more than this CD. See Demsar 2006, Eq. (5).
    """
    table = NEMENYI_Q_ALPHA.get(alpha)
    if table is None or n_methods < 2 or n_datasets < 1:
        return float("nan")
    q = table.get(int(n_methods))
    if q is None:
        # Linear extrapolation beyond the tabulated range.
        ks = sorted(table.keys())
        q = float(np.interp(n_methods, ks, [table[k] for k in ks]))
    return float(q * math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets)))


# ---------------------------------------------------------------------------
# Critical-difference diagram (Demsar style)
# ---------------------------------------------------------------------------
def plot_cd_diagram(
    avg_ranks: pd.DataFrame,
    cd: float,
    out_path_base: str,
    title: str = "Critical-Difference diagram (Nemenyi, alpha=0.05)",
) -> None:
    """Render a Demsar critical-difference diagram to PDF + PNG.

    ``avg_ranks`` is the frame returned by :func:`average_ranks`. The CD
    bar is drawn at the top of the figure; methods that are not
    significantly different are connected by a horizontal bar at their
    rank positions.
    """
    df = avg_ranks.sort_values("avg_rank").reset_index(drop=True)
    methods = df["method"].tolist()
    ranks = df["avg_rank"].to_numpy(dtype=float)
    k = len(methods)
    if k < 2:
        raise ValueError("CD diagram requires at least 2 methods")

    lo = math.floor(ranks.min() - 0.5)
    hi = math.ceil(ranks.max() + 0.5)
    width = max(1.0, hi - lo)

    fig, ax = plt.subplots(figsize=(7.0, 2.4 + 0.18 * k))
    ax.set_xlim(hi, lo)  # invert so rank 1 (best) sits at the left
    ax.set_ylim(0, k + 3)
    ax.axis("off")

    # Rank axis on top
    axis_y = k + 1.6
    ax.hlines(axis_y, lo, hi, color="black", linewidth=1.2)
    for r in range(lo, hi + 1):
        ax.vlines(r, axis_y - 0.10, axis_y + 0.10, color="black", linewidth=1.0)
        ax.text(r, axis_y + 0.30, str(r), ha="center", va="bottom", fontsize=9)

    # Method labels: half on the left, half on the right
    half = (k + 1) // 2
    for i, (m, r) in enumerate(zip(methods, ranks)):
        on_left = i < half
        x_text = lo if on_left else hi
        y_text = k - i if on_left else k - (i - half) - half
        # Connector
        ax.plot([r, r], [axis_y, y_text + 0.1], color="black", linewidth=0.8)
        ax.plot(
            [r, x_text], [y_text + 0.1, y_text + 0.1],
            color="black", linewidth=0.8,
        )
        ax.text(
            x_text + (0.05 * width if on_left else -0.05 * width),
            y_text + 0.1,
            f"{m} ({r:.2f})",
            ha="right" if on_left else "left",
            va="center",
            fontsize=9,
        )

    # CD bar (top-left)
    if np.isfinite(cd):
        cd_y = k + 2.6
        cd_lo, cd_hi = lo, lo + cd
        ax.hlines(cd_y, cd_lo, cd_hi, color="black", linewidth=1.6)
        ax.vlines(cd_lo, cd_y - 0.08, cd_y + 0.08, color="black", linewidth=1.2)
        ax.vlines(cd_hi, cd_y - 0.08, cd_y + 0.08, color="black", linewidth=1.2)
        ax.text(
            (cd_lo + cd_hi) / 2, cd_y + 0.25,
            f"CD = {cd:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    # Not-significantly-different cliques: connect adjacent runs whose
    # rank gap does not exceed CD.
    if np.isfinite(cd):
        clique_y = 0.6
        i = 0
        sorted_ranks = sorted(zip(ranks, range(k)))  # (rank, original_idx)
        # Build cliques on the *sorted* sequence.
        sr = [r for r, _ in sorted_ranks]
        n = len(sr)
        bar_idx = 0
        while i < n:
            j = i
            while j + 1 < n and (sr[j + 1] - sr[i]) <= cd + 1e-9:
                j += 1
            if j > i:
                ax.hlines(
                    clique_y + 0.25 * bar_idx,
                    sr[i], sr[j],
                    color="black", linewidth=2.0,
                )
                bar_idx += 1
            i = j + 1

    ax.set_title(title, fontsize=10, pad=8)
    fig.savefig(out_path_base + ".pdf")
    fig.savefig(out_path_base + ".png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pairwise Wilcoxon signed-rank with Holm correction
# ---------------------------------------------------------------------------
def wilcoxon_holm(
    scores: pd.DataFrame,
    methods: List[str],
    reference: str,
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests against ``reference``.

    Paired by ``dataset`` (means are taken over runs first). Holm-Bonferroni
    correction is applied across the ``len(methods) - 1`` pairwise tests.
    """
    if reference not in methods:
        raise ValueError(f"reference={reference!r} not in methods={methods!r}")

    try:
        from scipy.stats import wilcoxon  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Wilcoxon test requires scipy") from exc

    pivot = (
        scores.groupby(["dataset", "method"])["score"]
        .mean()
        .unstack("method")
    )
    pivot = pivot[methods].dropna(how="any")

    rows: List[Dict[str, object]] = []
    ref = pivot[reference].to_numpy(dtype=float)

    for m in methods:
        if m == reference:
            continue
        cand = pivot[m].to_numpy(dtype=float)
        diff = ref - cand
        if np.allclose(diff, 0.0):
            stat, p = float("nan"), 1.0
        else:
            try:
                stat_obj = wilcoxon(diff, zero_method="wilcox", correction=False)
                stat = float(stat_obj.statistic)
                p = float(stat_obj.pvalue)
            except ValueError:
                stat, p = float("nan"), float("nan")
        rows.append(
            {
                "reference": reference,
                "method": m,
                "W": stat,
                "p": p,
                "mean_diff": float(np.mean(diff)),
                "cohens_d": cohens_d(ref, cand),
                "n_datasets": int(len(diff)),
            }
        )

    df = pd.DataFrame(rows).sort_values("p", ignore_index=True)
    m_tests = len(df)
    # Holm-Bonferroni adjustment.
    adj: List[float] = []
    for i, p in enumerate(df["p"].to_numpy()):
        adj_p = min(1.0, (m_tests - i) * float(p)) if np.isfinite(p) else float("nan")
        if adj and np.isfinite(adj_p) and np.isfinite(adj[-1]):
            adj_p = max(adj_p, adj[-1])
        adj.append(adj_p)
    df["p_holm"] = adj
    df["reject_05"] = df["p_holm"] < 0.05
    df["reject_01"] = df["p_holm"] < 0.01
    return df


# ---------------------------------------------------------------------------
# Effect size and bootstrap CIs
# ---------------------------------------------------------------------------
def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples (pooled standard deviation).

    Conventional thresholds: |d| < 0.2 negligible, 0.2 small, 0.5 medium,
    0.8 large (Cohen, 1988).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")
    mu_x, mu_y = float(np.mean(x)), float(np.mean(y))
    sd_x, sd_y = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    s = math.sqrt(0.5 * (sd_x ** 2 + sd_y ** 2))
    if s <= 0.0 or not np.isfinite(s):
        return float("nan")
    return float((mu_x - mu_y) / s)


def bootstrap_ci(
    samples: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of ``samples``.

    Returns ``(point_estimate, lo, hi)`` with ``lo`` and ``hi`` the
    ``alpha/2`` and ``1 - alpha/2`` quantiles of the bootstrap
    distribution.
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = samples.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = samples[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1.0 - alpha / 2))
    return (float(samples.mean()), lo, hi)


def bootstrap_table(
    scores: pd.DataFrame,
    methods: List[str],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-method per-dataset bootstrap CI of the mean F1 score."""
    rows: List[Dict[str, object]] = []
    for ds, sub in scores.groupby("dataset"):
        for m in methods:
            arr = sub.loc[sub["method"] == m, "score"].to_numpy(dtype=float)
            mean, lo, hi = bootstrap_ci(arr, n_boot=n_boot, alpha=alpha, seed=seed)
            rows.append(
                {
                    "dataset": ds,
                    "method": m,
                    "mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_runs": int(arr.size),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-regime ANOVA
# ---------------------------------------------------------------------------
def per_regime_anova(base_dir: str, methods: List[str]) -> pd.DataFrame:
    """One-way ANOVA across regimes from the ``diagnostics.csv`` outputs.

    Reads the per-phase accept rates dropped by
    :func:`pod.experiment.run_suite_generic` and tests, for every method,
    whether the regime explains a significant share of the variance in
    the accept rate. Significant ``p_F`` here is evidence that the
    method's gating behaviour reacts to the operator regime, which is the
    desirable property for PoD and ablations.
    """
    rows: List[Dict[str, object]] = []
    try:
        from scipy.stats import f_oneway  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        f_oneway = None  # type: ignore[assignment]

    for dataset in sorted(os.listdir(base_dir)):
        diag_path = os.path.join(base_dir, dataset, "diagnostics.csv")
        if not os.path.isfile(diag_path):
            continue
        df = pd.read_csv(diag_path)
        for m in methods:
            sub = df[df["method"] == m]
            if sub.empty:
                continue
            b = sub["accept_rate_baseline"].to_numpy(dtype=float)
            g = sub["accept_rate_gaming"].to_numpy(dtype=float)
            f_ = sub["accept_rate_fatigue"].to_numpy(dtype=float)
            if f_oneway is not None and min(len(b), len(g), len(f_)) >= 2:
                try:
                    F, p = f_oneway(b, g, f_)
                    F, p = float(F), float(p)
                except Exception:
                    F, p = float("nan"), float("nan")
            else:
                F, p = float("nan"), float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "method": m,
                    "accept_baseline_mean": float(np.nanmean(b)) if len(b) else float("nan"),
                    "accept_gaming_mean": float(np.nanmean(g)) if len(g) else float("nan"),
                    "accept_fatigue_mean": float(np.nanmean(f_)) if len(f_) else float("nan"),
                    "F": F,
                    "p_F": p,
                    "n_runs": int(min(len(b), len(g), len(f_))),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PRIMARY (confirmatory) analysis: per-dataset, run-level paired tests.
#
# This is the experimental unit redefinition required for a defensible
# multi-method claim. The cross-dataset Wilcoxon (``wilcoxon_holm`` above)
# pairs by DATASET (N=5), whose minimum two-sided p-value is 2/2^5 = 0.0625;
# no Holm/BH-corrected result can reach alpha=0.05 there, so it is reported
# descriptively only. The PRIMARY test pairs the 20 seed-matched runs WITHIN
# each dataset (identical stream and per-run seed across methods, guaranteed
# by ``run_suite_generic``), giving 5 x (k-1) tests whose family-wise error
# is controlled with Benjamini-Hochberg FDR (Benjamini & Hochberg, JRSS-B
# 1995) at q = 0.05. Effect size is the paired Cohen's d_z = mean(diff) /
# sd(diff) with a percentile bootstrap CI over the paired differences.
# ---------------------------------------------------------------------------
def benjamini_hochberg(
    pvals: np.ndarray, q: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control.

    Returns ``(reject, p_adj)`` where ``reject[i]`` is True iff hypothesis
    ``i`` is rejected at FDR level ``q`` and ``p_adj`` is the BH-adjusted
    p-value (monotone, capped at 1).
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    reject = np.zeros(n, dtype=bool)
    p_adj = np.ones(n, dtype=float)
    if n == 0:
        return reject, p_adj
    order = np.argsort(pvals)
    # Largest rank passing the BH threshold p_(i) <= (i/n) q.
    crit_rank = 0
    for rank, idx in enumerate(order, start=1):
        if np.isfinite(pvals[idx]) and pvals[idx] <= (rank / n) * q:
            crit_rank = rank
    if crit_rank > 0:
        reject[order[:crit_rank]] = True
    # Monotone BH-adjusted p-values (step-up).
    prev = 1.0
    for rank, idx in zip(range(n, 0, -1), order[::-1]):
        val = pvals[idx] * n / rank if np.isfinite(pvals[idx]) else float("nan")
        if np.isfinite(val):
            prev = min(prev, val)
            p_adj[idx] = prev
        else:
            p_adj[idx] = float("nan")
    return reject, p_adj


def paired_dz_ci(
    diffs: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Paired Cohen's d_z and its percentile bootstrap CI.

    ``d_z = mean(diff) / sd(diff)`` (ddof=1). The CI resamples the paired
    differences with replacement ``n_boot`` times. Degenerate resamples
    (zero standard deviation) contribute ``d_z = 0``.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 2:
        return (float("nan"), float("nan"), float("nan"))
    sd = float(np.std(diffs, ddof=1))
    dz = float(np.mean(diffs) / sd) if sd > 0 else 0.0
    rng = np.random.default_rng(seed)
    n = diffs.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    res = diffs[idx]
    res_sd = res.std(axis=1, ddof=1)
    res_mean = res.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        bz = np.where(res_sd > 0, res_mean / res_sd, 0.0)
    lo = float(np.quantile(bz, alpha / 2))
    hi = float(np.quantile(bz, 1.0 - alpha / 2))
    return (dz, lo, hi)


def run_level_paired(
    scores: pd.DataFrame,
    methods: List[str],
    reference: str = "PoD",
    q: float = 0.05,
    n_boot: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-dataset, run-level paired Wilcoxon of ``reference`` vs each method.

    For every (dataset, competitor) pair, the seed-matched runs are paired
    by run index and a two-sided Wilcoxon signed-rank test is run on the
    differences ``reference - competitor`` (positive => reference better).
    Exact p-values are used when the sample admits them (scipy ``auto``).
    Benjamini-Hochberg FDR at level ``q`` is applied across ALL pairs.

    Returns one row per (dataset, competitor) with the test statistic,
    raw and BH-adjusted p-values, paired d_z + bootstrap CI, and the
    signed win/loss flags.
    """
    from scipy.stats import wilcoxon

    competitors = [m for m in methods if m != reference]
    datasets = sorted(scores["dataset"].unique())
    rows: List[Dict[str, object]] = []
    for d in datasets:
        sub = scores[scores["dataset"] == d]
        piv = sub.pivot(index="run", columns="method", values="score")
        if reference not in piv.columns:
            continue
        ref = piv[reference].to_numpy(dtype=float)
        for m in competitors:
            if m not in piv.columns:
                continue
            cand = piv[m].to_numpy(dtype=float)
            diffs = ref - cand
            if np.allclose(diffs, 0.0):
                W, p = float("nan"), 1.0
            else:
                try:
                    r = wilcoxon(diffs, zero_method="wilcox", correction=False)
                    W, p = float(r.statistic), float(r.pvalue)
                except ValueError:
                    W, p = float("nan"), 1.0
            dz, dz_lo, dz_hi = paired_dz_ci(diffs, n_boot=n_boot, alpha=q, seed=seed)
            rows.append(
                {
                    "dataset": d,
                    "reference": reference,
                    "method": m,
                    "n_runs": int(diffs.size),
                    "W": W,
                    "p": p,
                    "mean_diff": float(np.mean(diffs)),
                    "median_diff": float(np.median(diffs)),
                    "dz": dz,
                    "dz_ci_lo": dz_lo,
                    "dz_ci_hi": dz_hi,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    reject, p_adj = benjamini_hochberg(df["p"].to_numpy(dtype=float), q=q)
    df["p_bh"] = p_adj
    df["bh_reject"] = reject
    df["win_sig"] = df["bh_reject"] & (df["median_diff"] > 0)
    df["loss_sig"] = df["bh_reject"] & (df["median_diff"] < 0)
    return df


def summarize_run_level(
    run_level: pd.DataFrame, methods: List[str], reference: str = "PoD"
) -> pd.DataFrame:
    """Collapse the per-(dataset, competitor) table to one row per competitor.

    Columns: number of datasets won / lost significantly (BH), median d_z,
    and the d_z range across datasets. This is the body of the rewritten
    Table 2 (``tab:wilcoxon``).
    """
    competitors = [m for m in methods if m != reference]
    rows: List[Dict[str, object]] = []
    for m in competitors:
        g = run_level[run_level["method"] == m]
        if g.empty:
            continue
        rows.append(
            {
                "method": m,
                "n_datasets": int(len(g)),
                "wins_sig": int(g["win_sig"].sum()),
                "losses_sig": int(g["loss_sig"].sum()),
                "median_dz": float(g["dz"].median()),
                "min_dz": float(g["dz"].min()),
                "max_dz": float(g["dz"].max()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX macro emission (Phase 0.1: every experimental number is produced by
# this script and written into a file the manuscript \inputs).
# ---------------------------------------------------------------------------
_MACRO_NAME = {
    "AL": "AL",
    "StaticGating": "StaticGating",
    "AdaptiveGating": "AdaptiveGating",
    "WorkerQuality": "WorkerQuality",
    "Raykar": "Raykar",
    "MACE": "MACE",
    "IEThresh": "IEThresh",
    "PoD-NoGate": "NoGate",
    "PoD-NoCoupling": "NoCoupling",
    "PoD-NoVigilance": "NoVigilance",
}


def _fmt_p(p: float) -> str:
    """Format a p-value for LaTeX (scientific threshold for tiny values)."""
    if not np.isfinite(p):
        return r"\mathrm{n/a}"
    if p < 1e-4:
        return r"<\!10^{-4}"
    return f"{p:.4g}"


def write_stats_macros(path: str, report: Dict) -> None:
    """Write ``macros_generated.tex`` from a ``stats_report`` dict.

    Emits only statistics-derived numbers (Friedman, Nemenyi, the
    run-level BH win counts, and paired d_z summaries). Real-user / ATC
    macros are produced separately by ``pod-realdata``.
    """
    cfg = report["config"]
    fried = report["secondary_cross_dataset"]["friedman"]
    cd = report["secondary_cross_dataset"]["nemenyi_CD"]
    per_comp = {r["method"]: r for r in report["primary_run_level"]["per_competitor"]}

    lines: List[str] = []
    lines.append("%% =====================================================================")
    lines.append("%% AUTO-GENERATED by pod-stats. DO NOT EDIT BY HAND.")
    lines.append("%% Every number below is produced from the per-run CSVs under")
    lines.append("%% out_pod_unified/<dataset>/runs/ and is the single source of truth")
    lines.append("%% checked by scripts/check_claims.py. Regenerate with:")
    lines.append("%%     pod-stats --in out_pod_unified")
    lines.append("%% =====================================================================")
    lines.append(f"\\newcommand{{\\NDatasets}}{{\\ensuremath{{{cfg['n_datasets']}}}}}")
    lines.append(f"\\newcommand{{\\NMethods}}{{\\ensuremath{{{cfg['n_methods']}}}}}")
    lines.append(f"\\newcommand{{\\NCompetitors}}{{\\ensuremath{{{cfg['n_methods']-1}}}}}")
    lines.append(f"\\newcommand{{\\FDRtests}}{{\\ensuremath{{{cfg['fdr_n_tests']}}}}}")
    lines.append("% --- Cross-dataset omnibus (SECONDARY / descriptive) ---")
    lines.append(f"\\newcommand{{\\FriedmanChiSq}}{{\\ensuremath{{{fried['chi2']:.2f}}}}}")
    lines.append(f"\\newcommand{{\\FriedmanChiSqP}}{{\\ensuremath{{{_fmt_p(fried['p_chi2'])}}}}}")
    lines.append(f"\\newcommand{{\\FriedmanF}}{{\\ensuremath{{{fried['F']:.2f}}}}}")
    lines.append(f"\\newcommand{{\\FriedmanFP}}{{\\ensuremath{{{_fmt_p(fried['p_F'])}}}}}")
    lines.append(f"\\newcommand{{\\NemenyiCD}}{{\\ensuremath{{{cd:.2f}}}}}")
    lines.append("% --- Run-level BH-FDR results (PRIMARY / confirmatory) ---")
    lines.append("% Per competitor: \\RLwins<M> datasets won (of \\NDatasets) at FDR q=0.05;")
    lines.append("% \\RLdzMed<M> / \\RLdzMin<M> / \\RLdzMax<M> are paired Cohen's d_z summaries.")
    for raw, san in _MACRO_NAME.items():
        if raw not in per_comp:
            continue
        r = per_comp[raw]
        lines.append(f"\\newcommand{{\\RLwins{san}}}{{\\ensuremath{{{int(r['wins_sig'])}}}}}")
        lines.append(f"\\newcommand{{\\RLlosses{san}}}{{\\ensuremath{{{int(r['losses_sig'])}}}}}")
        lines.append(f"\\newcommand{{\\RLdzMed{san}}}{{\\ensuremath{{{r['median_dz']:.2f}}}}}")
        lines.append(f"\\newcommand{{\\RLdzMin{san}}}{{\\ensuremath{{{r['min_dz']:.2f}}}}}")
        lines.append(f"\\newcommand{{\\RLdzMax{san}}}{{\\ensuremath{{{r['max_dz']:.2f}}}}}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pod-stats",
        description=(
            "Run the TKDE statistical analysis suite (Friedman + Nemenyi, "
            "Wilcoxon signed-rank + Holm correction, Cohen's d, bootstrap CIs, "
            "per-regime ANOVA) on the outputs of `pod-experiments`."
        ),
    )
    parser.add_argument(
        "--in", dest="in_dir", type=str, default="out_pod_unified",
        help="Directory written by pod-experiments (default: out_pod_unified).",
    )
    parser.add_argument(
        "--out", dest="out_dir", type=str, default=None,
        help="Output directory (default: <in>/stats).",
    )
    parser.add_argument(
        "--reference", type=str, default="PoD",
        help="Reference method for pairwise Wilcoxon (default: PoD).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level (default: 0.05).",
    )
    parser.add_argument(
        "--score-mode", type=str, default="end", choices=["end", "auc"],
        help="Reduction of the F1 trajectory to a scalar (default: end).",
    )
    parser.add_argument(
        "--end-window", type=int, default=50,
        help="Number of trailing evaluation points used by --score-mode end.",
    )
    parser.add_argument(
        "--n-boot", type=int, default=10000,
        help="Number of bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Bootstrap PRNG seed (default: 0).",
    )
    return parser


def _discover_methods(scores: pd.DataFrame) -> List[str]:
    """Stable ordering: keep the canonical METHODS order when possible."""
    from pod.experiment import METHODS

    present = list(scores["method"].unique())
    ordered = [m for m in METHODS if m in present]
    ordered += [m for m in present if m not in ordered]
    return ordered


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    set_pub_style()

    out_dir = args.out_dir or os.path.join(args.in_dir, "stats")
    ensure_dir(out_dir)

    spec = ScoreSpec(mode=args.score_mode, end_window=int(args.end_window))
    scores = collect_per_run_scores(args.in_dir, score=spec)
    scores.to_csv(os.path.join(out_dir, "per_run_scores.csv"), index=False)

    methods = _discover_methods(scores)

    # Average ranks + Friedman omnibus.
    avg = average_ranks(scores, methods)
    avg.to_csv(os.path.join(out_dir, "average_ranks.csv"), index=False)

    fried = friedman_test(scores, methods)
    with open(os.path.join(out_dir, "friedman.json"), "w", encoding="utf-8") as f:
        json.dump(fried, f, indent=2)

    # Nemenyi CD diagram.
    cd = nemenyi_critical_diff(
        n_methods=len(methods),
        n_datasets=int(avg["n_datasets"].iloc[0]),
        alpha=args.alpha,
    )
    with open(os.path.join(out_dir, "nemenyi_cd.json"), "w", encoding="utf-8") as f:
        json.dump({"CD": cd, "alpha": args.alpha,
                    "n_methods": len(methods),
                    "n_datasets": int(avg["n_datasets"].iloc[0])},
                  f, indent=2)

    plot_cd_diagram(
        avg, cd,
        out_path_base=os.path.join(out_dir, "cd_diagram"),
        title=f"Critical-Difference diagram (Nemenyi, alpha={args.alpha})",
    )

    # Pairwise Wilcoxon + Holm.
    wil = wilcoxon_holm(scores, methods, reference=args.reference)
    wil.to_csv(os.path.join(out_dir, "wilcoxon_holm.csv"), index=False)

    # Bootstrap CIs for mean F1 per (dataset, method).
    boot = bootstrap_table(
        scores, methods,
        n_boot=int(args.n_boot), alpha=args.alpha, seed=int(args.seed),
    )
    boot.to_csv(os.path.join(out_dir, "bootstrap_ci.csv"), index=False)

    # Per-regime ANOVA from diagnostics.csv.
    anova = per_regime_anova(args.in_dir, methods)
    if not anova.empty:
        anova.to_csv(os.path.join(out_dir, "per_regime_anova.csv"), index=False)

    # PRIMARY (confirmatory) run-level paired tests + BH-FDR.
    run_level = run_level_paired(
        scores, methods, reference=args.reference,
        q=args.alpha, n_boot=int(args.n_boot), seed=int(args.seed),
    )
    run_level.to_csv(os.path.join(out_dir, "run_level_tests.csv"), index=False)
    rl_summary = summarize_run_level(run_level, methods, reference=args.reference)
    rl_summary.to_csv(os.path.join(out_dir, "run_level_summary.csv"), index=False)

    # Nemenyi gap check: which competitors exceed the critical difference.
    pod_rank = float(avg.loc[avg["method"] == args.reference, "avg_rank"].iloc[0])
    nemenyi_gap = {}
    for _, r in avg.iterrows():
        if r["method"] == args.reference:
            continue
        gap = float(r["avg_rank"]) - pod_rank
        nemenyi_gap[r["method"]] = {
            "avg_rank": float(r["avg_rank"]),
            "gap_to_ref": gap,
            "exceeds_CD": bool(gap > cd),
        }

    summary = {
        "n_datasets": int(avg["n_datasets"].iloc[0]),
        "n_methods": len(methods),
        "methods": methods,
        "reference": args.reference,
        "score_mode": args.score_mode,
        "end_window": int(args.end_window),
        "friedman": fried,
        "nemenyi_CD": cd,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # stats_report.json: the single source of truth that check_claims.py
    # and the manuscript macros are derived from. Combines the PRIMARY
    # (run-level, BH) and SECONDARY (cross-dataset, descriptive) layers.
    # ------------------------------------------------------------------
    stats_report = {
        "config": {
            "reference": args.reference,
            "score_mode": args.score_mode,
            "end_window": int(args.end_window),
            "alpha_q": args.alpha,
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
            "n_datasets": int(avg["n_datasets"].iloc[0]),
            "n_methods": len(methods),
            "methods": methods,
            "fdr_method": "benjamini_hochberg",
            "fdr_n_tests": int(len(run_level)),
        },
        "primary_run_level": {
            "per_test": run_level.to_dict(orient="records"),
            "per_competitor": rl_summary.to_dict(orient="records"),
        },
        "secondary_cross_dataset": {
            "friedman": fried,
            "nemenyi_CD": cd,
            "average_ranks": avg.to_dict(orient="records"),
            "nemenyi_gap": nemenyi_gap,
            "wilcoxon_descriptive": wil.to_dict(orient="records"),
        },
    }
    with open(os.path.join(out_dir, "stats_report.json"), "w", encoding="utf-8") as f:
        json.dump(stats_report, f, indent=2)

    # Emit the LaTeX macro file the manuscript \inputs.
    macro_path = os.path.join(out_dir, "macros_generated.tex")
    write_stats_macros(macro_path, stats_report)

    print(f"[pod-stats] Wrote analysis to: {out_dir}")
    print(f"[pod-stats] Friedman chi2={fried['chi2']:.3f} "
          f"p_chi2={fried['p_chi2']:.4g} F={fried['F']:.3f} p_F={fried['p_F']:.4g}")
    print(f"[pod-stats] Nemenyi CD (alpha={args.alpha})={cd:.3f}")
    print(f"[pod-stats] Run-level BH-FDR over {len(run_level)} tests "
          f"(q={args.alpha}); per-competitor summary:")
    for _, r in rl_summary.iterrows():
        print(f"            {r['method']:18s} wins {int(r['wins_sig'])}/"
              f"{int(r['n_datasets'])} losses {int(r['losses_sig'])}/"
              f"{int(r['n_datasets'])} median d_z={r['median_dz']:.2f}")
    print(f"[pod-stats] stats_report.json + macros_generated.tex written.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
