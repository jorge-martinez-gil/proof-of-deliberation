"""
Operator-accuracy sweep: characterise PoD - AL as a function of
corruption severity, on the canonical degradation schedule with the PUBLISHED
configuration (default timing signatures, coupling epsilon = preset = -1).

We sweep a single operator-accuracy knob `a` applied to BOTH degraded regimes
(p_correct_gaming = p_correct_fatigue = a) from 0.05 -> 0.65, plus the MEASURED
calibrated point (gaming 0.57 / fatigue 0.53) as a first-class entry. For each
(a, method, run) we run the UNCHANGED pod.experiment.run_stream_experiment_once
and record the final-window F1 (last K=50 eval points, the paper's reduction).

Methods: AL (unfiltered) and PoD. PoD - AL vs severity (=1-a) identifies the
crossover above which PoD wins. Resumable + time-bounded.
"""
from __future__ import annotations
import sys, argparse, os, json, time, glob, dataclasses
import numpy as np, pandas as pd
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.join(_ROOT, "src"))
sys.path.insert(0, _os.path.join(_ROOT, "experiments"))
from pod.config import RegimeSchedule
import pod.experiment as E
from streaming_harness import build

PREFIX = {"synth": "Synth-Boundary", "gas": "uci224_gas_drift"}
A_GRID = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
CALIB = ("calib", 0.57, 0.53)   # measured 24-participant accuracies
METHODS = ["AL", "PoD"]
K = 50

def final_f1(df):
    return float(df["f1"].to_numpy()[-K:].mean())

def one(bundle, schedule, method, run, p_gam, p_fat):
    cfg, init, al, pod, op0, cw, mode, rd = bundle
    op = dataclasses.replace(op0, p_correct_gaming=p_gam, p_correct_fatigue=p_fat)
    Xs, ys, Xh, yh, cs = rd(run)
    df, _ = E.run_stream_experiment_once(
        Xs, ys, Xh, yh, seed=1000*run+7, schedule=schedule, method=method,
        al=al, pod=pod, op=op, eval_window=600, eval_every=10, init_fit=init,
        complexity_mode=mode, c_star_stream=cs, class_weight_dict=cw,
        clf_alpha=cfg["clf_alpha"], clf_eta0=cfg["clf_eta0"], clf_average=cfg["clf_average"])
    return final_f1(df)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synth","gas"], required=True)
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--out", default=_os.path.join(_ROOT,"out_opacc"))
    ap.add_argument("--seconds", type=float, default=34.0)
    ap.add_argument("--mod", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    sch = RegimeSchedule(2000, 2000, 2000)
    bundle = build(a.dataset, sch)
    pref = PREFIX[a.dataset]
    outdir = os.path.join(a.out, a.dataset); os.makedirs(outdir, exist_ok=True)
    points = [(f"a{v:.2f}", v, v) for v in A_GRID] + [CALIB]
    t0 = time.time(); did = 0
    for tag, pg, pf in points:
        for m in METHODS:
            for r in range(a.runs):
                if a.of > 1 and (r % a.of) != a.mod: continue
                fp = os.path.join(outdir, f"{pref}_{tag}_{m}_run{r}.json")
                if os.path.exists(fp): continue
                if time.time()-t0 > a.seconds:
                    tot = len(points)*len(METHODS)*a.runs
                    have = len(glob.glob(os.path.join(outdir, f"{pref}_*_run*.json")))
                    print(f"PROGRESS opacc/{a.dataset} {have}/{tot} [did={did}] {time.time()-t0:.1f}s"); return
                f1 = one(bundle, sch, m, r, pg, pf)
                json.dump(dict(tag=tag, p_gam=pg, p_fat=pf, method=m, run=r, final_f1=f1), open(fp,"w"))
                did += 1
    tot = len(points)*len(METHODS)*a.runs
    have = len(glob.glob(os.path.join(outdir, f"{pref}_*_run*.json")))
    print(f"PROGRESS opacc/{a.dataset} {have}/{tot} [did={did}] {time.time()-t0:.1f}s DONE")

if __name__ == "__main__":
    main()
