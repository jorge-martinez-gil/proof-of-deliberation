#!/usr/bin/env python3
"""Enlarge the benchmark suite so the dataset-level test gains power (scaffold).

The run-level analysis is confirmatory only *within* a dataset; cross-domain
generality needs more datasets than the five we ship, because a corrected
cross-dataset test is impossible at N=5 (Section 6.x). This scaffold registers a
curated set of additional public streaming-classification benchmarks and runs the
*unchanged* canonical pipeline on each, so that re-running Friedman/Nemenyi over
N>5 datasets becomes possible wherever OpenML is reachable.

It adds no new methods and changes no PoD code: it reuses
``pod.cli.experiments._run_openml_generic`` and ``_build_params`` verbatim, only
supplying additional ``(name, OpenML id, preset)`` triples. Network access to
OpenML is required; in the offline sandbox this script prints the registry and
exits.

Candidate datasets (widely used in the streaming / concept-drift literature):
    KDDCup99  (id 1113)  : 23-class network-intrusion, abrupt drift, ~4.9M rows
    Poker-Hand(id 1595)  : 10-class, virtual drift, ~1M rows
    Nomao     (id 1486)  : binary record-linkage, ~34k rows
    Bank-mkt  (id 1461)  : binary marketing, ~45k rows
    Adult     (id 1590)  : binary census, ~48k rows
    Spambase  (id 44)    : binary spam, ~4.6k rows
Edit the REGISTRY below to taste; each entry inherits a sensible preset.
"""
from __future__ import annotations

import argparse
import sys

# name -> (OpenML data id, multiclass?, short note)
REGISTRY = {
    "kddcup99": (1113, True, "23-class intrusion, abrupt drift"),
    "poker": (1595, True, "10-class, virtual drift"),
    "nomao": (1486, False, "binary record-linkage"),
    "bankmkt": (1461, False, "binary marketing"),
    "adult": (1590, False, "binary census"),
    "spambase": (44, False, "binary spam"),
}


def _preset_for(multiclass: bool):
    """Inherit the elec2 preset (entropy-mode, coupling off) as a safe default;
    enable class weighting for multiclass/imbalanced streams."""
    from pod.presets import cfg_elec2
    cfg = dict(cfg_elec2())
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(REGISTRY),
                    help="comma-separated subset of: " + ", ".join(REGISTRY))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", default="out_more_datasets")
    a = ap.parse_args()

    try:
        from pod.cli.experiments import build_parser, _run_openml_generic
    except Exception as e:  # pragma: no cover
        sys.exit(f"requires the pod package on PYTHONPATH (pip install -e .): {e}")

    names = [d.strip() for d in a.datasets.split(",") if d.strip()]
    unknown = [d for d in names if d not in REGISTRY]
    if unknown:
        sys.exit(f"unknown datasets {unknown}; known: {list(REGISTRY)}")

    # Build a default args namespace from the canonical CLI, then override.
    base = build_parser().parse_args([])
    base.out = a.out

    print("Benchmark-suite extension plan:")
    for d in names:
        oid, mc, note = REGISTRY[d]
        print(f"  {d:10s} OpenML id={oid:<6d} multiclass={mc!s:5s}  {note}")
    print("\nEach runs the UNCHANGED canonical pipeline; after completion, point\n"
          "scripts/build_stats_report.py at --in", a.out,
          "to recompute Friedman/Nemenyi over the enlarged N.\n")

    ran = 0
    for d in names:
        oid, mc, _ = REGISTRY[d]
        cfg = _preset_for(mc)
        cfg["runs"] = a.runs
        try:
            _run_openml_generic(base, cfg, d, oid, d, use_class_weight=mc)
            ran += 1
        except Exception as e:
            print(f"[{d}] skipped ({type(e).__name__}: {e}). "
                  "Most likely OpenML is unreachable in this environment.")
    if ran == 0:
        print("\n[No dataset ran -- offline. This scaffold fabricates nothing; "
              "run it where OpenML is reachable to enlarge the suite.]")


if __name__ == "__main__":
    main()
