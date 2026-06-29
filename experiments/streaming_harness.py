"""
Resumable, time-bounded runner for three extended-regime experiments
(sandbox-friendly; call repeatedly until PROGRESS reports complete).

Experiments
-----------
A. Calibrated-realism regime.  Re-runs the operator simulator with the
   *measured* real-participant accuracies (gaming p_correct=0.57,
   fatigue p_correct=0.53; Section 6.11) instead of the deliberately
   pessimistic defaults, leaving the response-time distributions
   unchanged.  All canonical methods plus the new Hybrid run.

B. Hybrid  PoD AND WorkerQuality.  A label is admitted iff *both* the
   process gate (full PoD) and the content gate (single-annotator
   Dawid-Skene WorkerQuality) accept it.  Evaluated in the main regimes
   and in the confident-error regime (E3 of threats_to_validity.py).

C. Layer necessity / adaptive adversary.  Per-regime firing rate of each
   of the four PoD signals (gate, coupling, gaming-vigilance,
   fatigue-vigilance) and the bad-label leakage that results from
   dropping the single layer that guards each regime -- the quantitative
   layer-necessity analysis.

Design
------
* The eleven canonical methods are produced by the UNCHANGED
  ``pod.experiment.run_stream_experiment_once`` so their numbers are
  bit-for-bit the paper's pipeline.
* ``Hybrid`` and the layer instrumentation use ``custom_pass`` below,
  which reproduces that loop verbatim and reuses the same core
  primitives (gate_check / coupling_check / gaming_detector /
  fatigue_detector / accept_worker_quality), so the new method is
  faithful to the protocol.
* Every (experiment, dataset, regime, method, run) unit is written to
  its own file and skipped if present, so the whole suite survives a
  45-second execution cap by being called repeatedly.

Only Synth-Boundary and Gas Drift are run here (the other three streams
need OpenML downloads); pass --dataset to select.  The same code runs
elec2/covertype/airlines unchanged wherever OpenML is reachable.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from pod.config import (ALParams, OperatorParams, PoDParams, RegimeSchedule,
                        SynthParams)
from pod.core import (coupling_check, fatigue_detector, gaming_detector,
                      gate_check)
from pod.baselines import (WorkerQualityState, accept_worker_quality,
                           update_worker_quality)
from pod.learner import make_classifier, proba_from_decision_function
from pod.operator import simulate_operator
from pod.presets import cfg_gas, cfg_synth
from pod.streams import (generate_synth_boundary_pool_regime, load_gas_drift,
                         make_class_weight_dict, split_stream_holdout_by_batch)
from pod.utils import entropy_from_proba, entropy_unit, sigmoid
import pod.experiment as E

PREFIX = {"synth": "Synth-Boundary", "gas": "uci224_gas_drift"}

# Calibrated-realism accuracies measured on the 24-participant elec2 study.
CALIB_P_GAMING = 0.57
CALIB_P_FATIGUE = 0.53

# Method panels --------------------------------------------------------------
CANON = ("AL", "StaticGating", "AdaptiveGating", "WorkerQuality", "Raykar",
         "MACE", "IEThresh", "PoD", "PoD-NoGate", "PoD-NoCoupling",
         "PoD-NoVigilance")
# Focused panel reported in the calibrated row-block + main Hybrid column.
PANEL_A = ("AL", "AdaptiveGating", "WorkerQuality", "MACE", "PoD",
           "PoD-NoVigilance", "Hybrid")


# ---------------------------------------------------------------------------
# Data factories (identical construction to src/pod/cli/experiments.py)
# ---------------------------------------------------------------------------
def build(dataset: str, schedule: RegimeSchedule):
    if dataset == "synth":
        cfg = cfg_synth()
        init = int(cfg["init_fit"])
        synth = SynthParams()
        Xp, yp, cp = generate_synth_boundary_pool_regime(
            50000, 20000, synth, schedule, init, 0.004, 0.0, 0.0)
        hold = 6000

        def rd(r: int):
            rng = np.random.default_rng(10000 + r)
            T = schedule.total()
            sl = init + T + 20
            ms = len(Xp) - (sl + hold)
            st = int(rng.integers(0, ms + 1))
            Xr, yr, cr = Xp[st:st + sl + hold], yp[st:st + sl + hold], cp[st:st + sl + hold]
            return Xr[:sl], yr[:sl], Xr[sl:sl + hold], yr[sl:sl + hold], cr[:sl]
        mode, cw = "c_star", None
    else:
        cfg = cfg_gas()
        init = int(cfg["init_fit"])
        X, y, bid = load_gas_drift(cache_dir="data_cache_uci224", batches=list(range(1, 11)))
        Xs, ys, Xh, yh = split_stream_holdout_by_batch(X, y, bid, holdout_batches=[9, 10])
        classes = np.unique(np.concatenate([ys, yh]))
        cw = make_class_weight_dict(classes=classes, y_sample=ys[:init])

        def rd(r: int):
            rng = np.random.default_rng(10000 + r)
            perm = rng.permutation(len(Xs))
            return Xs[perm], ys[perm], Xh, yh, None
        mode = "entropy"

    al = ALParams(query_budget=cfg["query_budget"], query_temp=cfg["query_temp"])
    pod = PoDParams(
        coupling_window=cfg["coupling_window"], coupling_epsilon=cfg["coupling_epsilon"],
        gate_a=cfg.get("gate_a", 650.0), gate_b=cfg.get("gate_b", 245.0),
        gate_lo_frac=cfg.get("gate_lo_frac", 0.58), gate_hi_frac=cfg.get("gate_hi_frac", 0.86),
        gate_floor_ms=cfg.get("gate_floor_ms", 120.0), gate_ceil_ms=cfg.get("gate_ceil_ms", 3200.0),
        gaming_window=cfg["gaming_window"], gaming_mu_max_ms=cfg["gaming_mu_max_ms"],
        gaming_cv_max=cfg["gaming_cv_max"], fatigue_window=cfg["fatigue_window"],
        fatigue_cv_min=cfg["fatigue_cv_min"], persist_k=cfg["persist_k"])
    op = OperatorParams(
        k=650., c=270., sigma=88., c_fast=600., eps_fast=cfg["op_eps_fast"],
        c_slow=cfg["op_c_slow"], sigma_high=cfg["op_sigma_high"],
        p_correct_baseline=cfg["op_p_correct_baseline"],
        p_correct_gaming=cfg["op_p_correct_gaming"],
        p_correct_fatigue=cfg["op_p_correct_fatigue"])
    return cfg, init, al, pod, op, cw, mode, rd


def calib_op(op: OperatorParams) -> OperatorParams:
    """Operator with measured real accuracies; timing distributions unchanged."""
    return dataclasses.replace(op, p_correct_gaming=CALIB_P_GAMING,
                               p_correct_fatigue=CALIB_P_FATIGUE)


# ---------------------------------------------------------------------------
# Faithful custom pass: supports method='Hybrid' and full layer instrumentation
# Mirrors pod.experiment.run_stream_experiment_once exactly.
# ---------------------------------------------------------------------------
def custom_pass(Xs, ys, Xh, yh, seed, schedule, method, al, pod, op,
                init_fit, complexity_mode, c_star=None, class_weight_dict=None,
                eval_window=600, eval_every=10, clf_alpha=2e-5, clf_eta0=0.01,
                clf_average=True, instrument=False):
    rng = np.random.default_rng(seed)
    T = schedule.total()
    Xstream = Xs[:init_fit + T]
    ystream = ys[:init_fit + T]
    classes = np.unique(np.concatenate([ys, yh]))
    n_classes = int(len(classes))
    imp = SimpleImputer(strategy="mean").fit(Xstream[:init_fit])
    sc = StandardScaler().fit(imp.transform(Xstream[:init_fit]))
    Xz = sc.transform(imp.transform(Xstream))
    Xhz = sc.transform(imp.transform(Xh))
    clf = make_classifier(classes=classes, seed=seed, clf_alpha=clf_alpha,
                          clf_eta0=clf_eta0, clf_average=clf_average,
                          class_weight_dict=class_weight_dict)
    clf.partial_fit(Xz[:init_fit], ystream[:init_fit], classes=classes)

    comp_hist: List[float] = []
    delib_hist: List[float] = []
    wq = WorkerQualityState.fresh(n_classes=n_classes)

    # persist-k bad counters for full PoD and the three single-layer-drops
    bad = {"full": [0, 0, 0], "noGate": [0, 0, 0],
           "noCoup": [0, 0, 0], "noVig": [0, 0, 0]}
    # instrumentation tallies per phase
    phases = ("baseline", "gaming", "fatigue")
    instr = {p: dict(q=0, bad=0, gate_fire=0, coup_fire=0, gam_fire=0, fat_fire=0,
                     pod_acc=0, pod_bad_acc=0,
                     noGate_bad_acc=0, noCoup_bad_acc=0, noVig_bad_acc=0)
             for p in phases}

    f1s: List[float] = []
    ts: List[int] = []
    hptr = 0

    def decide(g_ok, c_raw, gam, fat, key, use_gate, use_coup, use_vig):
        b = bad[key]
        b[0] = b[0] + 1 if (use_coup and c_raw == 0) else (0 if use_coup else 0)
        b[1] = b[1] + 1 if (use_vig and gam == 1) else (0 if use_vig else 0)
        b[2] = b[2] + 1 if (use_vig and fat == 1) else (0 if use_vig else 0)
        ok = (b[0] < pod.persist_k) and (b[1] < pod.persist_k) and (b[2] < pod.persist_k)
        gate_term = (g_ok == 1) if use_gate else True
        return bool(gate_term and ok)

    for t in range(init_fit, init_fit + T):
        t_rel = t - init_fit
        phase = schedule.phase(t_rel)
        xt = Xz[t:t + 1]
        proba = proba_from_decision_function(clf, xt, n_classes=n_classes)[0]
        if complexity_mode == "entropy":
            comp = entropy_unit(entropy_from_proba(proba), max(2, n_classes))
        else:
            comp = float(c_star[t])

        z = (comp - 0.5) / max(1e-9, al.query_temp)
        pq = float(np.clip(float(sigmoid(np.array([z]))[0]) * (al.query_budget / 0.5), 0.0, 1.0))
        if rng.random() < pq:
            y_tilde, delib = simulate_operator(rng, int(ystream[t]), n_classes, comp, phase, op)
            is_bad = int(int(y_tilde) != int(ystream[t]))

            # PoD-family records history before the checks (verbatim semantics)
            comp_hist.append(comp)
            delib_hist.append(float(delib))

            g_ok = gate_check(delib, comp, pod)
            c_raw = coupling_check(np.asarray(comp_hist), np.asarray(delib_hist), pod)
            gam = gaming_detector(np.asarray(delib_hist), pod)
            fat = fatigue_detector(np.asarray(delib_hist), pod)

            # full PoD decision (advances full counters)
            pod_accept = decide(g_ok, c_raw, gam, fat, "full", True, True, True)

            if method == "Hybrid":
                wq_accept = accept_worker_quality(proba, int(y_tilde), wq)
                update_worker_quality(wq, int(np.argmax(proba)), int(y_tilde))
                accept = bool(pod_accept and wq_accept)
            elif method == "PoD":
                accept = pod_accept
            else:
                raise ValueError(f"custom_pass supports PoD/Hybrid only, got {method}")

            if instrument:
                d = instr[phase]
                d["q"] += 1
                d["bad"] += is_bad
                d["gate_fire"] += int(g_ok == 0)
                d["coup_fire"] += int(c_raw == 0)
                d["gam_fire"] += int(gam == 1)
                d["fat_fire"] += int(fat == 1)
                # single-layer-drop decisions (independent persist counters)
                a_ng = decide(g_ok, c_raw, gam, fat, "noGate", False, True, True)
                a_nc = decide(g_ok, c_raw, gam, fat, "noCoup", True, False, True)
                a_nv = decide(g_ok, c_raw, gam, fat, "noVig", True, True, False)
                if pod_accept:
                    d["pod_acc"] += 1
                    d["pod_bad_acc"] += is_bad
                if a_ng:
                    d["noGate_bad_acc"] += is_bad
                if a_nc:
                    d["noCoup_bad_acc"] += is_bad
                if a_nv:
                    d["noVig_bad_acc"] += is_bad

            if accept:
                clf.partial_fit(xt, np.array([y_tilde], dtype=int))

        if t_rel % max(1, eval_every) == 0:
            k = min(eval_window, len(Xhz))
            idx = (np.arange(k) + hptr) % len(Xhz)
            hptr = (hptr + 1) % len(Xhz)
            f1s.append(f1_metric(yh[idx], clf.predict(Xhz[idx]), n_classes))
            ts.append(t_rel)

    return pd.DataFrame({"t": ts, "f1": f1s}), instr


def f1_metric(y_true, y_pred, n_classes):
    if n_classes <= 2:
        return float(f1_score(y_true, y_pred, zero_division=0))
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Confident-error harness (E3 of threats_to_validity.py) + Hybrid
# ---------------------------------------------------------------------------
def ce_static_pool(seed, n, synth):
    # Matches static_pool() in experiments/threats_to_validity.py exactly.
    rng = np.random.default_rng(seed + 99991)
    d = synth.d
    X = rng.normal(0.0, synth.noise_std, size=(n, d)).astype(float)
    w0 = rng.normal(0.0, 1.0, size=(d,))
    w0 = w0 / max(1e-12, float(np.linalg.norm(w0)))
    margin = X @ w0
    y = (margin >= 0).astype(int)
    c_star = np.exp(-synth.lambda_complexity * np.abs(margin)).astype(float)
    return X, y, c_star


def ce_draw(rng, y_true, comp, phase, op):
    if phase == "confident_error":
        p_flip = float(np.clip(0.9 * (1.0 - comp), 0.0, 1.0))
        lab = (1 - int(y_true)) if rng.random() < p_flip else int(y_true)
        delib = float(max(50.0, rng.normal(op.k * comp + op.c, op.sigma)))
        return lab, delib
    lab = int(y_true) if rng.random() < op.p_correct_baseline else (1 - int(y_true))
    delib = float(max(50.0, rng.normal(op.k * comp + op.c, op.sigma)))
    return lab, delib


def ce_run_once(method, seed, phase_len, init_fit, holdout, al, pod, op, synth):
    rng = np.random.default_rng(seed)
    phases = ("baseline", "confident_error")
    T = phase_len * len(phases)
    X, y, c_star = ce_static_pool(seed, init_fit + T + holdout + 10, synth)
    Xs, ys, cs = X[:init_fit + T], y[:init_fit + T], c_star[:init_fit + T]
    Xh, yh = X[init_fit + T:init_fit + T + holdout], y[init_fit + T:init_fit + T + holdout]
    classes = np.unique(y)
    n_classes = int(len(classes))
    imp = SimpleImputer(strategy="mean").fit(Xs[:init_fit])
    sc = StandardScaler().fit(imp.transform(Xs[:init_fit]))
    Xz = sc.transform(imp.transform(Xs))
    Xhz = sc.transform(imp.transform(Xh))
    clf = make_classifier(classes=classes, seed=seed, clf_alpha=1e-6, clf_eta0=0.08, clf_average=False)
    try:
        clf.set_params(learning_rate="constant")
    except Exception:
        pass
    clf.partial_fit(Xz[:init_fit], ys[:init_fit], classes=classes)

    comp_hist, delib_hist = [], []
    wq = WorkerQualityState.fresh(n_classes=n_classes)
    from pod.baselines import MACEOnlineState, accept_mace, update_mace
    mace = MACEOnlineState.fresh()
    bad = [0, 0, 0]
    f1s, ts = [], []
    hptr = 0
    pod_acc = pod_q = 0

    def phase_of(tr):
        return phases[min(len(phases) - 1, tr // phase_len)]

    for t in range(init_fit, init_fit + T):
        tr = t - init_fit
        phase = phase_of(tr)
        xt = Xz[t:t + 1]
        proba = proba_from_decision_function(clf, xt, n_classes=n_classes)[0]
        comp = float(cs[t])
        z = (comp - 0.5) / max(1e-9, al.query_temp)
        pq = float(np.clip(float(sigmoid(np.array([z]))[0]) * (al.query_budget / 0.5), 0.0, 1.0))
        if rng.random() < pq:
            y_tilde, delib = ce_draw(rng, int(ys[t]), comp, phase, op)
            comp_hist.append(comp)
            delib_hist.append(delib)
            g_ok = gate_check(delib, comp, pod)
            c_raw = coupling_check(np.asarray(comp_hist), np.asarray(delib_hist), pod)
            gam = gaming_detector(np.asarray(delib_hist), pod)
            fat = fatigue_detector(np.asarray(delib_hist), pod)
            bad[0] = bad[0] + 1 if c_raw == 0 else 0
            bad[1] = bad[1] + 1 if gam == 1 else 0
            bad[2] = bad[2] + 1 if fat == 1 else 0
            pod_accept = bool(g_ok == 1 and bad[0] < pod.persist_k
                              and bad[1] < pod.persist_k and bad[2] < pod.persist_k)
            if method == "AL":
                accept = True
            elif method == "PoD":
                accept = pod_accept
            elif method == "WorkerQuality":
                accept = accept_worker_quality(proba, int(y_tilde), wq)
                update_worker_quality(wq, int(np.argmax(proba)), int(y_tilde))
            elif method == "MACE":
                update_mace(mace, proba, int(y_tilde))
                accept = accept_mace(mace)
            elif method == "Hybrid":
                wq_a = accept_worker_quality(proba, int(y_tilde), wq)
                update_worker_quality(wq, int(np.argmax(proba)), int(y_tilde))
                accept = bool(pod_accept and wq_a)
            else:
                raise ValueError(method)
            if phase == "confident_error":
                pod_q += 1
                pod_acc += int(pod_accept)
            if accept:
                clf.partial_fit(xt, np.array([y_tilde], dtype=int))
        if tr % 20 == 0:
            k = min(800, len(Xhz))
            idx = (np.arange(k) + hptr) % len(Xhz)
            hptr = (hptr + 1) % len(Xhz)
            f1s.append(float(f1_score(yh[idx], clf.predict(Xhz[idx]), zero_division=0)))
            ts.append(tr)

    lo = phases.index("confident_error") * phase_len
    vals = [f for tt, f in zip(ts, f1s) if tt >= lo]
    ce_f1 = float(np.mean(vals[-10:])) if vals else float("nan")
    return dict(method=method, seed=seed, ce_f1=ce_f1,
                pod_accept_rate=float(pod_acc / pod_q) if pod_q else float("nan"))


# ---------------------------------------------------------------------------
# Unit runners (resumable)
# ---------------------------------------------------------------------------
def run_expA(dataset, runs, outroot, t0, seconds):
    """Calibrated + default regime, focused panel, F1 trajectory per unit."""
    sch = RegimeSchedule(2000, 2000, 2000)
    cfg, init, al, pod, op, cw, mode, rd = build(dataset, sch)
    prefix = PREFIX[dataset]
    did = 0
    # calibrated regime: full focused panel (the row-block).
    # default regime: only the new Hybrid (the other six already exist in the
    # published ablation table; this supplies the new main-table column).
    for regime, opx, panel in (("calib", calib_op(op), PANEL_A),
                               ("default", op, ("Hybrid",))):
        outdir = os.path.join(outroot, "expA", regime, dataset, "runs")
        os.makedirs(outdir, exist_ok=True)
        for r in range(runs):
            todo = [m for m in panel
                    if not os.path.exists(os.path.join(outdir, f"{prefix}_{m}_run{r}.csv"))]
            if not todo:
                continue
            Xs, ys, Xh, yh, cs = rd(r)
            for m in todo:
                if time.time() - t0 > seconds:
                    return did
                if m == "Hybrid":
                    df, _ = custom_pass(Xs, ys, Xh, yh, 1000 * r + 7, sch, "Hybrid",
                                        al, pod, opx, init, mode, c_star=cs,
                                        class_weight_dict=cw, clf_alpha=cfg["clf_alpha"],
                                        clf_eta0=cfg["clf_eta0"], clf_average=cfg["clf_average"])
                else:
                    df, _ = E.run_stream_experiment_once(
                        Xs, ys, Xh, yh, seed=1000 * r + 7, schedule=sch, method=m,
                        al=al, pod=pod, op=opx, eval_window=600, eval_every=10,
                        init_fit=init, complexity_mode=mode, c_star_stream=cs,
                        class_weight_dict=cw, clf_alpha=cfg["clf_alpha"],
                        clf_eta0=cfg["clf_eta0"], clf_average=cfg["clf_average"])
                df.to_csv(os.path.join(outdir, f"{prefix}_{m}_run{r}.csv"), index=False)
                did += 1
    return did


def run_expC(dataset, runs, outroot, t0, seconds):
    """Layer-firing + leakage instrumentation, both regimes, one pass per unit."""
    sch = RegimeSchedule(2000, 2000, 2000)
    cfg, init, al, pod, op, cw, mode, rd = build(dataset, sch)
    prefix = PREFIX[dataset]
    did = 0
    for regime, opx in (("default", op), ("calib", calib_op(op))):
        outdir = os.path.join(outroot, "expC", regime, dataset)
        os.makedirs(outdir, exist_ok=True)
        for r in range(runs):
            fp = os.path.join(outdir, f"{prefix}_instr_run{r}.json")
            if os.path.exists(fp):
                continue
            if time.time() - t0 > seconds:
                return did
            Xs, ys, Xh, yh, cs = rd(r)
            _, instr = custom_pass(Xs, ys, Xh, yh, 1000 * r + 7, sch, "PoD",
                                   al, pod, opx, init, mode, c_star=cs,
                                   class_weight_dict=cw, clf_alpha=cfg["clf_alpha"],
                                   clf_eta0=cfg["clf_eta0"], clf_average=cfg["clf_average"],
                                   instrument=True)
            with open(fp, "w") as f:
                json.dump(instr, f)
            did += 1
    return did


def run_expB(runs, outroot, t0, seconds):
    """Confident-error regime (E3) reusing threats_to_validity.run_once with the
    EXACT published E3 settings, adding the Hybrid method. Reproduces the
    published TTVce* numbers for AL/PoD/MACE/WorkerQuality as a side check."""
    from threats_to_validity import (run_once, base_pod, base_operator,
                                      base_synth_params)
    al = ALParams(query_budget=0.60, query_temp=0.90)
    pod = base_pod(coupling_on=False)
    op = base_operator(k=900.0)
    synth = base_synth_params()
    kw = dict(phases=("baseline", "confident_error"), phase_len=1000, init_fit=300,
              holdout=1200, al=al, pod=pod, op=op, synth=synth,
              eval_every=40, eval_window=600)
    outdir = os.path.join(outroot, "expB")
    os.makedirs(outdir, exist_ok=True)
    methods = ("AL", "PoD", "MACE", "WorkerQuality", "Hybrid")
    did = 0
    for s in range(runs):           # seeds 0..runs-1 (published E3 used 0..11)
        for m in methods:
            fp = os.path.join(outdir, f"{m}_run{s}.json")
            if os.path.exists(fp):
                continue
            if time.time() - t0 > seconds:
                return did
            r = run_once(m, s, **kw)
            c = r["ctr"]["confident_error"]
            out = dict(method=m, seed=s,
                       ce_f1=r["final_by_phase"]["confident_error"],
                       pod_accept_rate=(c["acc"] / c["q"] if c["q"] else float("nan")))
            with open(fp, "w") as f:
                json.dump(out, f)
            did += 1
    return did


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B", "C"], required=True)
    ap.add_argument("--dataset", choices=["synth", "gas"], default="synth")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--out", default="out_extended")
    a = ap.parse_args()
    t0 = time.time()
    if a.exp == "A":
        did = run_expA(a.dataset, a.runs, a.out, t0, a.seconds)
        _progress_A(a.dataset, a.runs, a.out)
    elif a.exp == "C":
        did = run_expC(a.dataset, a.runs, a.out, t0, a.seconds)
        _progress_C(a.dataset, a.runs, a.out)
    else:
        did = run_expB(a.runs, a.out, t0, a.seconds)
        _progress_B(a.runs, a.out)
    print(f"[did={did}] elapsed={time.time()-t0:.1f}s")


def _progress_A(dataset, runs, out):
    prefix = PREFIX[dataset]
    tot = runs * (len(PANEL_A) + 1)
    have = len(glob.glob(os.path.join(out, "expA", "*", dataset, "runs", f"{prefix}_*_run*.csv")))
    print(f"PROGRESS expA/{dataset} {have}/{tot}")


def _progress_C(dataset, runs, out):
    prefix = PREFIX[dataset]
    tot = 2 * runs
    have = len(glob.glob(os.path.join(out, "expC", "*", dataset, prefix + "_instr_run*.json")))
    print("PROGRESS expC/%s %d/%d" % (dataset, have, tot))


def _progress_B(runs, out):
    tot = 5 * runs
    have = len(glob.glob(os.path.join(out, "expB", "*_run*.json")))
    print("PROGRESS expB %d/%d" % (have, tot))


if __name__ == "__main__":
    main()
