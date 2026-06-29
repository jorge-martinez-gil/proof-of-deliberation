"""
Resumable, time-bounded base-learner robustness runner (sandbox-friendly).

Each invocation runs as many (run, method) units as fit in --seconds, writing one
CSV per unit and SKIPPING units already on disk. Call repeatedly until complete.
Uses the canonical pod.experiment.run_stream_experiment_once unchanged; only the
base learner is swapped via monkeypatch (sgd|mlp|nb).
"""
from __future__ import annotations
import argparse, os, time, glob, numpy as np, pandas as pd
from pod.config import (ALParams, OperatorParams, PoDParams, RegimeSchedule, SynthParams)
from pod.presets import cfg_synth, cfg_gas
from pod.streams import (generate_synth_boundary_pool_regime, load_gas_drift,
                         make_class_weight_dict, split_stream_holdout_by_batch)
import pod.experiment as E
from pod.learner import make_classifier as _mk_sgd, proba_from_decision_function as _pb_sgd
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

PREFIX = {"synth": "Synth-Boundary", "gas": "uci224_gas_drift"}
PANEL = ["AL","StaticGating","AdaptiveGating","Raykar","MACE","PoD","PoD-NoVigilance"]

def _mk_mlp(classes, seed, *, clf_alpha=2e-5, clf_eta0=0.01, clf_average=True, class_weight_dict=None):
    return MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam",
                         alpha=float(clf_alpha), learning_rate_init=float(clf_eta0), random_state=seed)
def _mk_nb(classes, seed, *, clf_alpha=2e-5, clf_eta0=0.01, clf_average=True, class_weight_dict=None):
    return GaussianNB()
def _pb_any(clf, X, n_classes):
    p = np.asarray(clf.predict_proba(X), dtype=float)
    return p.reshape(1,-1) if p.ndim==1 else p
FAC = {"sgd": (_mk_sgd,_pb_sgd), "mlp": (_mk_mlp,_pb_any), "nb": (_mk_nb,_pb_any)}

def build(dataset, schedule):
    if dataset=="synth":
        cfg=cfg_synth(); init=int(cfg["init_fit"])
        synth=SynthParams()
        Xp,yp,cp=generate_synth_boundary_pool_regime(50000,20000,synth,schedule,init,0.004,0.0,0.0)
        hold=6000
        def rd(r):
            rng=np.random.default_rng(10000+r); T=schedule.total(); sl=init+T+20
            ms=len(Xp)-(sl+hold); st=int(rng.integers(0,ms+1))
            Xr=Xp[st:st+sl+hold]; yr=yp[st:st+sl+hold]; cr=cp[st:st+sl+hold]
            return Xr[:sl],yr[:sl],Xr[sl:sl+hold],yr[sl:sl+hold],cr[:sl]
        mode="entropy" if False else "c_star"; cw=None
    else:
        cfg=cfg_gas(); init=int(cfg["init_fit"])
        X,y,bid=load_gas_drift(cache_dir="data_cache_uci224",batches=list(range(1,11)))
        Xs,ys,Xh,yh=split_stream_holdout_by_batch(X,y,bid,holdout_batches=[9,10])
        classes=np.unique(np.concatenate([ys,yh])); cw=make_class_weight_dict(classes=classes,y_sample=ys[:init])
        def rd(r):
            rng=np.random.default_rng(10000+r); perm=rng.permutation(len(Xs))
            return Xs[perm],ys[perm],Xh,yh,None
        mode="entropy"
    al=ALParams(query_budget=cfg["query_budget"],query_temp=cfg["query_temp"])
    pod=PoDParams(coupling_window=cfg["coupling_window"],coupling_epsilon=cfg["coupling_epsilon"],
        gate_a=cfg.get("gate_a",650.0),gate_b=cfg.get("gate_b",245.0),gate_lo_frac=cfg.get("gate_lo_frac",0.58),
        gate_hi_frac=cfg.get("gate_hi_frac",0.86),gate_floor_ms=cfg.get("gate_floor_ms",120.0),
        gate_ceil_ms=cfg.get("gate_ceil_ms",3200.0),gaming_window=cfg["gaming_window"],
        gaming_mu_max_ms=cfg["gaming_mu_max_ms"],gaming_cv_max=cfg["gaming_cv_max"],
        fatigue_window=cfg["fatigue_window"],fatigue_cv_min=cfg["fatigue_cv_min"],persist_k=cfg["persist_k"])
    op=OperatorParams(k=650.,c=270.,sigma=88.,c_fast=600.,eps_fast=cfg["op_eps_fast"],c_slow=cfg["op_c_slow"],
        sigma_high=cfg["op_sigma_high"],p_correct_baseline=cfg["op_p_correct_baseline"],
        p_correct_gaming=cfg["op_p_correct_gaming"],p_correct_fatigue=cfg["op_p_correct_fatigue"])
    return cfg,init,al,pod,op,cw,mode,rd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["synth","gas"],required=True)
    ap.add_argument("--learner",choices=["sgd","mlp","nb"],required=True)
    ap.add_argument("--runs",type=int,default=10)
    ap.add_argument("--seconds",type=float,default=40.0)
    ap.add_argument("--out",default="/tmp/rob")
    a=ap.parse_args()
    E.make_classifier,E.proba_from_decision_function=FAC[a.learner]
    sch=RegimeSchedule(2000,2000,2000)
    outdir=os.path.join(a.out,f"{a.dataset}_{a.learner}","runs"); os.makedirs(outdir,exist_ok=True)
    prefix=PREFIX[a.dataset]
    cfg,init,al,pod,op,cw,mode,rd=build(a.dataset,sch)
    t0=time.time(); did=0
    # iterate run-major so data is built once per run
    for r in range(a.runs):
        todo=[m for m in PANEL if not os.path.exists(os.path.join(outdir,f"{prefix}_{m}_run{r}.csv"))]
        if not todo: continue
        Xs,ys,Xh,yh,cs=rd(r)
        for m in todo:
            if time.time()-t0 > a.seconds: 
                print(f"[budget] stop did={did} (run{r} pending)"); _report(outdir,prefix,a.runs); return
            df,_=E.run_stream_experiment_once(Xs,ys,Xh,yh,seed=1000*r+7,schedule=sch,method=m,al=al,pod=pod,op=op,
                eval_window=600,eval_every=10,init_fit=init,complexity_mode=mode,c_star_stream=cs,
                class_weight_dict=cw,clf_alpha=cfg["clf_alpha"],clf_eta0=cfg["clf_eta0"],clf_average=cfg["clf_average"])
            df.to_csv(os.path.join(outdir,f"{prefix}_{m}_run{r}.csv"),index=False); did+=1
    print(f"[complete] did={did}"); _report(outdir,prefix,a.runs)

def _report(outdir,prefix,runs):
    tot=runs*len(PANEL); have=len(glob.glob(os.path.join(outdir,f"{prefix}_*_run*.csv")))
    print(f"PROGRESS {have}/{tot}")
if __name__=="__main__": main()
