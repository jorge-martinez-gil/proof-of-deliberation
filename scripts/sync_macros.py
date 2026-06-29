#!/usr/bin/env python3
"""Regenerate macros_generated.tex from the artifact CSVs and fail on drift.

This is the CI guard for Phase 2.4: it rebuilds the statistical macros from
out_pod_unified/<dataset>/runs/ into a temporary file and compares the
\newcommand lines against the committed out_pod_unified/stats/macros_generated.tex.
Comment lines (starting with %) are ignored. Exit code 0 = in sync,
1 = drift (committed macros do not match a fresh regeneration from data).

Usage:  python scripts/sync_macros.py [--in out_pod_unified]
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import build_stats_report as B
from pod.stats import (collect_per_run_scores, ScoreSpec, average_ranks,
                       friedman_test, nemenyi_critical_diff, wilcoxon_holm)


def build_report(in_dir, alpha=0.05, n_boot=10000, seed=0):
    scores = collect_per_run_scores(in_dir, ScoreSpec("end", 50))
    present = list(scores["method"].unique())
    methods = [m for m in B.METHOD_ORDER if m in present] + \
              [m for m in present if m not in B.METHOD_ORDER]
    avg = average_ranks(scores, methods)
    fried = friedman_test(scores, methods)
    cd = nemenyi_critical_diff(len(methods), int(avg["n_datasets"].iloc[0]), alpha)
    wil = wilcoxon_holm(scores, methods, "PoD")
    rl = B.run_level_paired(scores, methods, "PoD", alpha, n_boot, seed)
    rls = B.summarize(rl, methods, "PoD")
    pod_rank = float(avg.loc[avg["method"] == "PoD", "avg_rank"].iloc[0])
    gap = {r["method"]: {"avg_rank": float(r["avg_rank"]),
                         "gap_to_ref": float(r["avg_rank"]) - pod_rank,
                         "exceeds_CD": bool(float(r["avg_rank"]) - pod_rank > cd)}
           for _, r in avg.iterrows() if r["method"] != "PoD"}
    return {"config": {"reference": "PoD", "score_mode": "end", "end_window": 50,
                       "alpha_q": alpha, "n_boot": n_boot, "seed": seed,
                       "n_datasets": int(avg["n_datasets"].iloc[0]),
                       "n_methods": len(methods), "methods": methods,
                       "fdr_method": "benjamini_hochberg", "fdr_n_tests": int(len(rl))},
            "primary_run_level": {"per_test": json.loads(rl.to_json(orient="records")),
                                  "per_competitor": json.loads(rls.to_json(orient="records"))},
            "secondary_cross_dataset": {"friedman": fried, "nemenyi_CD": cd,
                                        "average_ranks": json.loads(avg.to_json(orient="records")),
                                        "nemenyi_gap": gap,
                                        "wilcoxon_descriptive": json.loads(wil.to_json(orient="records"))}}


def newcommands(text):
    return [ln.strip() for ln in text.splitlines()
            if ln.strip().startswith("\\newcommand")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default=os.path.join(ROOT, "out_pod_unified"))
    a = ap.parse_args(argv)
    committed_path = os.path.join(a.in_dir, "stats", "macros_generated.tex")
    if not os.path.isfile(committed_path):
        print(f"sync_macros: committed macros not found at {committed_path}")
        return 1
    rep = build_report(a.in_dir)
    with tempfile.NamedTemporaryFile("w+", suffix=".tex", delete=False) as tf:
        tmp = tf.name
    B.write_macros(tmp, rep)
    fresh = newcommands(open(tmp, encoding="utf-8").read())
    committed = newcommands(open(committed_path, encoding="utf-8").read())
    os.unlink(tmp)
    if fresh == committed:
        print(f"sync_macros: IN SYNC ({len(committed)} macros match a fresh "
              f"regeneration from {a.in_dir}).")
        return 0
    fset, cset = set(fresh), set(committed)
    print("sync_macros: DRIFT DETECTED.")
    for ln in sorted(cset - fset):
        print("  committed only:", ln)
    for ln in sorted(fset - cset):
        print("  regenerated only:", ln)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
