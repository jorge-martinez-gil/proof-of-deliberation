#!/usr/bin/env python3
"""In-domain labeling-timing coupling analysis (scaffold).

This is the analysis the ATC section (Section 6.x) says is the missing
*in-domain* validation: instead of a proxy domain (air-traffic control), it
operates on a real data-*labeling* corpus in which each labeling event carries a
recorded response time, a task-difficulty estimate, and a ground-truth
correctness flag. It reports the Hick--Hyman coupling (Spearman rho between
difficulty and response time) AND its *stability* across tasks, operators, and
fatigue levels --- exactly the "correlation stability across tasks, operators,
and fatigue levels".

It deliberately ships WITHOUT data: no labeling-timing corpus is fabricated.
Point it at a CSV you collected (or a public one that records per-item response
times) with the schema below and it produces the table and the bootstrap CIs.

Expected CSV schema (one row per labeling event)
------------------------------------------------
    operator_id   : str/int   anonymous annotator id
    task_id       : str/int   task / batch / dataset partition id
    response_ms   : float      deliberation time in milliseconds
    difficulty    : float      task-difficulty estimate in [0,1]
                               (e.g. model predictive entropy, or item IRT b)
    correct       : 0/1        label correctness vs ground truth (optional but
                               recommended; enables the accuracy-vs-coupling cut)
    fatigue_bin   : str        optional; e.g. 'early'/'mid'/'late' within session
                               (if absent, derived from within-operator event order)

Usage
-----
    python experiments/analyze_labeling_timing.py --csv path/to/corpus.csv
    python experiments/analyze_labeling_timing.py --csv corpus.csv --out out_labeling/

Output
------
    * stdout table of Spearman rho (with 95% bootstrap CI) overall and stratified
      by task, by operator, and by fatigue bin;
    * a coupling-stability summary (median, IQR, fraction of positive strata);
    * out_labeling/coupling_by_stratum.csv and macros_labeling.tex if --out given.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import pandas as pd
    from scipy.stats import spearmanr
except Exception as e:  # pragma: no cover
    sys.exit(f"requires pandas+scipy: {e}")

REQUIRED = ["operator_id", "task_id", "response_ms", "difficulty"]


def boot_spearman(x, y, n_boot=10000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 8:
        return float("nan"), (float("nan"), float("nan")), len(x)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(rho), (float(lo), float(hi)), n


def stratum_table(df, by, label):
    rows = []
    for key, g in df.groupby(by):
        rho, (lo, hi), n = boot_spearman(g["difficulty"], g["response_ms"])
        rows.append(dict(stratum=label, key=str(key), n=n, rho=rho, lo=lo, hi=hi))
    return rows


def derive_fatigue_bin(df):
    out = df.copy()
    if "fatigue_bin" in out.columns:
        return out
    out["_order"] = out.groupby("operator_id").cumcount()
    out["_cnt"] = out.groupby("operator_id")["_order"].transform("max") + 1
    frac = out["_order"] / out["_cnt"].clip(lower=1)
    out["fatigue_bin"] = np.where(frac < 1 / 3, "early",
                                  np.where(frac < 2 / 3, "mid", "late"))
    return out.drop(columns=["_order", "_cnt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="labeling-timing corpus CSV")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    if not os.path.isfile(a.csv):
        sys.exit(f"no such file: {a.csv}\n"
                 "This scaffold ships without data on purpose; collect or supply "
                 "a labeling corpus (schema in the module docstring).")
    df = pd.read_csv(a.csv)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        sys.exit(f"CSV missing required columns {miss}; see module docstring.")
    df = derive_fatigue_bin(df)

    rho, (lo, hi), n = boot_spearman(df["difficulty"], df["response_ms"], a.n_boot)
    print(f"OVERALL Hick-Hyman coupling: rho={rho:.3f} 95% CI [{lo:.3f},{hi:.3f}] (n={n})")
    if "correct" in df.columns:
        for flag, name in ((1, "correct"), (0, "incorrect")):
            g = df[df["correct"] == flag]
            r, (l, h), nn = boot_spearman(g["difficulty"], g["response_ms"], a.n_boot)
            print(f"  on {name} labels: rho={r:.3f} CI [{l:.3f},{h:.3f}] (n={nn})")

    rows = []
    for by, lab in (("task_id", "task"), ("operator_id", "operator"),
                    ("fatigue_bin", "fatigue")):
        rows += stratum_table(df, by, lab)
    res = pd.DataFrame(rows)
    print("\nCoupling stability across strata:")
    for lab in ("task", "operator", "fatigue"):
        sub = res[(res.stratum == lab) & np.isfinite(res.rho)]
        if len(sub):
            fpos = float((sub.rho > 0).mean())
            print(f"  {lab:9s}: median rho={sub.rho.median():.3f} "
                  f"IQR=[{sub.rho.quantile(.25):.3f},{sub.rho.quantile(.75):.3f}] "
                  f"positive in {fpos*100:.0f}% of {len(sub)} strata")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        res.to_csv(os.path.join(a.out, "coupling_by_stratum.csv"), index=False)
        with open(os.path.join(a.out, "macros_labeling.tex"), "w") as f:
            f.write("\\newcommand{\\LabelRho}{\\ensuremath{%.3f}}\n" % rho)
            f.write("\\newcommand{\\LabelRhoCI}{\\ensuremath{[%.3f,\\,%.3f]}}\n" % (lo, hi))
        print(f"\n[wrote {a.out}/coupling_by_stratum.csv and macros_labeling.tex]")


if __name__ == "__main__":
    main()
