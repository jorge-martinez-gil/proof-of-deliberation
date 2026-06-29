"""
Generalization runner, resumable + time-bounded.

(A) Base-learner robustness across MORE datasets: the canonical closed loop
    (pod.experiment.run_stream_experiment_once) run with an injected base learner
    {sgd, mlp, nb} on synth, synth_hd (d=24), synth_drift (faster boundary
    rotation), and gas -- a real cross-dataset generalization check rather than a
    single-dataset one. The learner factory is monkeypatched into the UNCHANGED
    pipeline (identical to experiments/run_base_learner_robustness.py).

(B) Second operator model: --operator v2 swaps the Gaussian response-time / 
    regime-constant-error simulator for a LOGNORMAL response-time simulator with
    COMPLEXITY-GRADED baseline errors (a structurally different generative
    process), via monkeypatching pod.experiment.simulate_operator. Re-running the
    panel under v2 shows cross-domain inference does not rest on one simulator.

Per-(dataset,learner,operator,method,run) units are written as CSVs and skipped
if present. Aggregation reuses the analyze_robustness reduction (final-window F1).
"""
from __future__ import annotations
import sys, argparse, os, json, time, glob, math
import numpy as np, pandas as pd
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.join(_ROOT, "src"))
sys.path.insert(0, _os.path.join(_ROOT, "experiments"))
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from pod.config import (ALParams, OperatorParams, PoDParams, RegimeSchedule, SynthParams)
from pod.presets import cfg_synth, cfg_gas
from pod.streams import (generate_synth_boundary_pool_regime, load_gas_drift,
                         make_class_weight_dict, split_stream_holdout_by_batch)
import pod.experiment as E
from pod.learner import make_classifier as _mk_sgd, proba_from_decision_function as _pb_sgd
import pod.operator as OP

PANEL8 = ["AL","StaticGating","AdaptiveGating","WorkerQuality","Raykar","MACE","IEThresh","PoD"]
PREFIX = {"synth":"Synth-Boundary","synth_hd":"Synth-HighDim","synth_drift":"Synth-FastDrift","gas":"uci224_gas_drift"}

# ---- alternate learners (same call signature as make_classifier) -----------
def _mk_mlp(classes, seed, *, clf_alpha=2e-5, clf_eta0=0.01, clf_average=True, class_weight_dict=None):
    return MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam",
                         alpha=float(clf_alpha), learning_rate_init=float(clf_eta0), random_state=seed)
def _mk_nb(classes, seed, *, clf_alpha=2e-5, clf_eta0=0.01, clf_average=True, class_weight_dict=None):
    return GaussianNB()
def _pb_any(clf, X, n_classes):
    if hasattr(clf, "predict_proba"):
        p = np.asarray(clf.predict_proba(X), dtype=float)
        return p.reshape(1,-1) if p.ndim==1 else p
    return _pb_sgd(clf, X, n_classes)
FACT = {"sgd":(_mk_sgd,_pb_sgd), "mlp":(_mk_mlp,_pb_any), "nb":(_mk_nb,_pb_any)}

# ---- second operator model: lognormal RT + complexity-graded errors --------
def simulate_operator_v2(rng, y_true, n_classes, complexity, phase, op: OperatorParams):
    """Second operator model: a structurally DIFFERENT generative family
    (multiplicative LOGNORMAL response times, right-skewed/heavy-tailed) replacing
    the additive Gaussian v1, with the SAME regime-constant error model so the
    degradation remains timing-detectable. Per-regime log-spreads are chosen to
    reproduce comparable coefficients of variation to v1 (gaming tight/fast,
    fatigue wide/slow), isolating the distribution-family change."""
    if phase == "baseline":
        median = op.k * complexity + op.c               # same central tendency as v1
        delib = float(np.exp(rng.normal(math.log(max(1.0, median)), 0.22)))  # CV~0.22
        p_corr = op.p_correct_baseline
    elif phase == "gaming":
        delib = float(np.exp(rng.normal(math.log(max(1.0, op.c_fast)), 0.015)))  # tight/fast, CV~0.015
        p_corr = op.p_correct_gaming
    else:  # fatigue
        delib = float(np.exp(rng.normal(math.log(max(1.0, op.c_slow)), 0.62)))   # heavy-tailed/slow, CV~0.6
        p_corr = op.p_correct_fatigue
    delib = float(max(50.0, delib))
    if p_corr < 0.0:
        return (1-int(y_true)) if n_classes<=2 else int((int(y_true)+1)%n_classes), delib
    if rng.random() < p_corr:
        return int(y_true), delib
    if n_classes <= 2:
        return 1-int(y_true), delib
    return int(rng.choice([c for c in range(n_classes) if c!=y_true])), delib

# ---- data builders ---------------------------------------------------------
def _params(cfg, schedule):
    al = ALParams(query_budget=cfg["query_budget"], query_temp=cfg["query_temp"])
    pod = PoDParams(coupling_window=cfg["coupling_window"], coupling_epsilon=cfg["coupling_epsilon"],
        gate_a=cfg.get("gate_a",650.0), gate_b=cfg.get("gate_b",245.0),
        gate_lo_frac=cfg.get("gate_lo_frac",0.58), gate_hi_frac=cfg.get("gate_hi_frac",0.86),
        gate_floor_ms=cfg.get("gate_floor_ms",120.0), gate_ceil_ms=cfg.get("gate_ceil_ms",3200.0),
        gaming_window=cfg["gaming_window"], gaming_mu_max_ms=cfg["gaming_mu_max_ms"],
        gaming_cv_max=cfg["gaming_cv_max"], fatigue_window=cfg["fatigue_window"],
        fatigue_cv_min=cfg["fatigue_cv_min"], persist_k=cfg["persist_k"])
    op = OperatorParams(k=650.,c=270.,sigma=88.,c_fast=600.,eps_fast=cfg["op_eps_fast"],
        c_slow=cfg["op_c_slow"], sigma_high=cfg["op_sigma_high"],
        p_correct_baseline=cfg["op_p_correct_baseline"], p_correct_gaming=cfg["op_p_correct_gaming"],
        p_correct_fatigue=cfg["op_p_correct_fatigue"])
    return al, pod, op

def build_any(dataset, schedule):
    if dataset.startswith("synth"):
        cfg = cfg_synth(); init = int(cfg["init_fit"])
        if dataset == "synth":     synth, rot = SynthParams(), 0.004
        elif dataset == "synth_hd":   synth, rot = SynthParams(d=24), 0.004
        elif dataset == "synth_drift":synth, rot = SynthParams(d=10), 0.012
        else: raise ValueError(dataset)
        Xp, yp, cp = generate_synth_boundary_pool_regime(50000, 20000, synth, schedule, init, rot, 0.0, 0.0)
        hold = 6000
        def rd(r):
            rng = np.random.default_rng(10000+r); T=schedule.total(); sl=init+T+20
            ms = len(Xp)-(sl+hold); st=int(rng.integers(0,ms+1))
            Xr,yr,cr = Xp[st:st+sl+hold], yp[st:st+sl+hold], cp[st:st+sl+hold]
            return Xr[:sl], yr[:sl], Xr[sl:sl+hold], yr[sl:sl+hold], cr[:sl]
        al,pod,op = _params(cfg, schedule)
        return cfg, init, al, pod, op, None, "c_star", rd
    else:
        cfg = cfg_gas(); init = int(cfg["init_fit"])
        X,y,bid = load_gas_drift(cache_dir="data_cache_uci224", batches=list(range(1,11)))
        Xs,ys,Xh,yh = split_stream_holdout_by_batch(X,y,bid, holdout_batches=[9,10])
        classes = np.unique(np.concatenate([ys,yh])); cw = make_class_weight_dict(classes=classes, y_sample=ys[:init])
        def rd(r):
            rng=np.random.default_rng(10000+r); perm=rng.permutation(len(Xs))
            return Xs[perm], ys[perm], Xh, yh, None
        al,pod,op = _params(cfg, schedule)
        return cfg, init, al, pod, op, cw, "entropy", rd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(PREFIX), required=True)
    ap.add_argument("--learner", choices=["sgd","mlp","nb"], required=True)
    ap.add_argument("--operator", choices=["v1","v2","v2b"], default="v1")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--mod", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=36.0)
    ap.add_argument("--out", default=_os.path.join(_ROOT,"out_generalization"))
    a = ap.parse_args()
    sch = RegimeSchedule(2000,2000,2000)
    # inject learner + operator into the unchanged pipeline
    mk, pb = FACT[a.learner]; E.make_classifier = mk; E.proba_from_decision_function = pb
    if a.operator in ("v2","v2b"): E.simulate_operator = simulate_operator_v2
    else: E.simulate_operator = OP.simulate_operator
    bundle = build_any(a.dataset, sch)
    cfg, init, al, pod, op, cw, mode, rd = bundle
    pref = PREFIX[a.dataset]
    outdir = os.path.join(a.out, f"{a.dataset}_{a.learner}_{a.operator}", "runs"); os.makedirs(outdir, exist_ok=True)
    t0=time.time(); did=0
    for r in range(a.runs):
        if a.of>1 and (r%a.of)!=a.mod: continue
        data=None
        for m in PANEL8:
            fp=os.path.join(outdir, f"{pref}_{m}_run{r}.csv")
            if os.path.exists(fp): continue
            if time.time()-t0 > a.seconds:
                tot=a.runs*len(PANEL8); have=len(glob.glob(os.path.join(outdir,f"{pref}_*_run*.csv")))
                print(f"PROGRESS {a.dataset}/{a.learner}/{a.operator} {have}/{tot} [did={did}] {time.time()-t0:.1f}s"); return
            if data is None: data=rd(r)
            Xs,ys,Xh,yh,cs=data
            df,_=E.run_stream_experiment_once(Xs,ys,Xh,yh,seed=1000*r+7,schedule=sch,method=m,
                al=al,pod=pod,op=op,eval_window=600,eval_every=10,init_fit=init,complexity_mode=mode,
                c_star_stream=cs,class_weight_dict=cw,clf_alpha=cfg["clf_alpha"],clf_eta0=cfg["clf_eta0"],
                clf_average=cfg["clf_average"])
            df.to_csv(fp, index=False); did+=1
    tot=a.runs*len(PANEL8); have=len(glob.glob(os.path.join(outdir,f"{pref}_*_run*.csv")))
    print(f"PROGRESS {a.dataset}/{a.learner}/{a.operator} {have}/{tot} [did={did}] {time.time()-t0:.1f}s DONE")

if __name__=="__main__": main()
