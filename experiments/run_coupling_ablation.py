"""
Coupling earns-its-keep experiment.

Shows a regime where PoD-NoCoupling is *significantly* worse than full PoD, by
turning the cognitive-coupling authenticity test ON (coupling_epsilon != -1) and
introducing a MIMICRY adversary: the degraded operator emits engaged-magnitude,
moderate-variability response times that are DECOUPLED from task complexity, with
mostly wrong labels. Gate + vigilance pass this adversary; only the coupling test
(harder items must take longer) rejects it.

Two modes:
  --mode main : fixed epsilon, panel {AL, PoD, PoD-NoGate, PoD-NoCoupling,
                PoD-NoVigilance}; writes per-(method,run) F1 trajectory CSVs.
  --mode sweep: methods {PoD}, epsilon in a grid; writes per-(eps,run) final-F1 +
                bad-label leakage so the calibration curve (and epsilon=-1 as a
                special case) can be plotted.

Resumable + time-bounded (45s sandbox cap): call repeatedly until PROGRESS done.
Every label-acceptance primitive (gate_check / coupling_check / gaming_detector /
fatigue_detector) is the UNCHANGED published code; only the operator's degraded
phase and the epsilon knob differ.
"""
from __future__ import annotations
import sys, argparse, os, json, time, glob, dataclasses
import numpy as np, pandas as pd
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.join(_ROOT, "src"))
sys.path.insert(0, _os.path.join(_ROOT, "experiments"))
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from pod.config import RegimeSchedule
from pod.core import coupling_check, fatigue_detector, gaming_detector, gate_check
from pod.learner import make_classifier, proba_from_decision_function
from pod.operator import simulate_operator
from pod.utils import entropy_from_proba, entropy_unit, sigmoid
from streaming_harness import build

PREFIX = {"synth": "Synth-Boundary", "gas": "uci224_gas_drift"}
# method -> (use_gate, use_coupling, use_vigilance) ; AL handled separately
PODVAR = {
    "PoD":            (True, True, True),
    "PoD-NoGate":     (False, True, True),
    "PoD-NoCoupling": (True, False, True),
    "PoD-NoVigilance":(True, True, False),
}
MIM_MU, MIM_SD = 580.0, 60.0   # engaged-magnitude, moderate-CV decoupled timing


def mimicry_draw(rng, y_true, n_classes, p_correct):
    delib = float(max(50.0, rng.normal(MIM_MU, MIM_SD)))
    if rng.random() < p_correct:
        return int(y_true), delib
    if n_classes <= 2:
        return 1 - int(y_true), delib
    return int(rng.choice([c for c in range(n_classes) if c != y_true])), delib


def draw_op(rng, y, n, comp, phase, op, p_mim):
    if phase == "baseline":
        return simulate_operator(rng, int(y), n, comp, "baseline", op)
    return mimicry_draw(rng, int(y), n, p_mim)   # degraded phase = mimicry


def f1m(yt, yp, n):
    return float(f1_score(yt, yp, zero_division=0)) if n <= 2 else float(f1_score(yt, yp, average="macro", zero_division=0))


def one_pass(bundle, method, run, eps, schedule, p_mim, eval_every=10, eval_window=600):
    cfg, init, al, pod0, op, cw, mode, rd = bundle
    pod = dataclasses.replace(pod0, coupling_epsilon=eps)
    Xs, ys, Xh, yh, cs = rd(run)
    rng = np.random.default_rng(1000 * run + 7)
    T = schedule.total()
    Xstream, ystream = Xs[:init + T], ys[:init + T]
    classes = np.unique(np.concatenate([ys, yh])); n = int(len(classes))
    imp = SimpleImputer(strategy="mean").fit(Xstream[:init])
    sc = StandardScaler().fit(imp.transform(Xstream[:init]))
    Xz = sc.transform(imp.transform(Xstream)); Xhz = sc.transform(imp.transform(Xh))
    clf = make_classifier(classes=classes, seed=1000 * run + 7, clf_alpha=cfg["clf_alpha"],
                          clf_eta0=cfg["clf_eta0"], clf_average=cfg["clf_average"], class_weight_dict=cw)
    clf.partial_fit(Xz[:init], ystream[:init], classes=classes)
    comp_hist, delib_hist = [], []
    bad = [0, 0, 0]
    ug, uc, uv = PODVAR.get(method, (True, True, True))
    deg_q = deg_bad = deg_bad_acc = 0
    f1s, ts = [], []; hptr = 0
    for t in range(init, init + T):
        tr = t - init; phase = schedule.phase(tr); xt = Xz[t:t + 1]
        proba = proba_from_decision_function(clf, xt, n_classes=n)[0]
        comp = entropy_unit(entropy_from_proba(proba), max(2, n)) if mode == "entropy" else float(cs[t])
        z = (comp - 0.5) / max(1e-9, al.query_temp)
        pq = float(np.clip(float(sigmoid(np.array([z]))[0]) * (al.query_budget / 0.5), 0.0, 1.0))
        if rng.random() < pq:
            y_t, delib = draw_op(rng, int(ystream[t]), n, comp, phase, op, p_mim)
            is_bad = int(int(y_t) != int(ystream[t]))
            comp_hist.append(comp); delib_hist.append(float(delib))
            ch = np.asarray(comp_hist); dh = np.asarray(delib_hist)
            if method == "AL":
                accept = True
            else:
                g_ok = gate_check(delib, comp, pod)
                c_raw = coupling_check(ch, dh, pod) if uc else 1
                gam = gaming_detector(dh, pod) if uv else 0
                fat = fatigue_detector(dh, pod) if uv else 0
                bad[0] = bad[0] + 1 if (uc and c_raw == 0) else 0
                bad[1] = bad[1] + 1 if (uv and gam == 1) else 0
                bad[2] = bad[2] + 1 if (uv and fat == 1) else 0
                ok = (bad[0] < pod.persist_k) and (bad[1] < pod.persist_k) and (bad[2] < pod.persist_k)
                accept = bool(((g_ok == 1) if ug else True) and ok)
            if phase != "baseline":
                deg_q += 1; deg_bad += is_bad; deg_bad_acc += (is_bad if accept else 0)
            if accept:
                clf.partial_fit(xt, np.array([y_t], dtype=int))
        if tr % max(1, eval_every) == 0:
            k = min(eval_window, len(Xhz)); idx = (np.arange(k) + hptr) % len(Xhz); hptr = (hptr + 1) % len(Xhz)
            f1s.append(f1m(yh[idx], clf.predict(Xhz[idx]), n)); ts.append(tr)
    df = pd.DataFrame({"t": ts, "f1": f1s})
    leak = float(deg_bad_acc / max(1, deg_bad))
    return df, dict(deg_q=deg_q, deg_bad=deg_bad, deg_bad_acc=deg_bad_acc, deg_leak=leak)


def run_main(dataset, runs, eps, outroot, schedule, p_mim, t0, seconds):
    pref = PREFIX[dataset]
    bundle = build(dataset, schedule)
    outdir = os.path.join(outroot, "main", f"eps{eps:.2f}", dataset, "runs")
    os.makedirs(outdir, exist_ok=True)
    did = 0
    methods = ["AL"] + list(PODVAR.keys())
    for r in range(runs):
        for m in methods:
            fp = os.path.join(outdir, f"{pref}_{m}_run{r}.csv")
            if os.path.exists(fp):
                continue
            if time.time() - t0 > seconds:
                return did
            df, instr = one_pass(bundle, m, r, eps, schedule, p_mim)
            df.to_csv(fp, index=False)
            json.dump(instr, open(fp.replace(".csv", "_instr.json"), "w"))
            did += 1
    return did


def run_sweep(dataset, runs, eps_grid, outroot, schedule, p_mim, t0, seconds):
    pref = PREFIX[dataset]
    bundle = build(dataset, schedule)
    outdir = os.path.join(outroot, "sweep", dataset)
    os.makedirs(outdir, exist_ok=True)
    did = 0
    for e in eps_grid:
        for r in range(runs):
            fp = os.path.join(outdir, f"{pref}_PoD_eps{e:.2f}_run{r}.json")
            if os.path.exists(fp):
                continue
            if time.time() - t0 > seconds:
                return did
            df, instr = one_pass(bundle, "PoD", r, e, schedule, p_mim)
            deg = df[df.t >= schedule.baseline]["f1"].to_numpy()
            rec = dict(eps=e, run=r, deg_final_f1=float(np.mean(deg[-50:])) if len(deg) else float("nan"),
                       deg_leak=instr["deg_leak"])
            json.dump(rec, open(fp, "w")); did += 1
    return did


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["main", "sweep"], required=True)
    ap.add_argument("--dataset", choices=["synth", "gas"], required=True)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--eps", type=float, default=0.20)
    ap.add_argument("--baseline", type=int, default=2000)
    ap.add_argument("--degraded", type=int, default=4000)
    ap.add_argument("--pmim", type=float, default=0.10)
    ap.add_argument("--out", default=_os.path.join(_ROOT,"out_coupling"))
    ap.add_argument("--seconds", type=float, default=40.0)
    a = ap.parse_args()
    sch = RegimeSchedule(a.baseline, a.degraded, 0)
    t0 = time.time()
    if a.mode == "main":
        did = run_main(a.dataset, a.runs, a.eps, a.out, sch, a.pmim, t0, a.seconds)
        pref = PREFIX[a.dataset]
        tot = a.runs * (1 + len(PODVAR))
        have = len(glob.glob(os.path.join(a.out, "main", f"eps{a.eps:.2f}", a.dataset, "runs", f"{pref}_*_run*.csv")))
        print(f"PROGRESS main/{a.dataset}/eps{a.eps:.2f} {have}/{tot} [did={did}] {time.time()-t0:.1f}s")
    else:
        eps_grid = [-1.0, 0.0, 0.1, 0.2, 0.3]
        did = run_sweep(a.dataset, a.runs, eps_grid, a.out, sch, a.pmim, t0, a.seconds)
        pref = PREFIX[a.dataset]
        tot = a.runs * len(eps_grid)
        have = len(glob.glob(os.path.join(a.out, "sweep", a.dataset, f"{pref}_PoD_eps*_run*.json")))
        print(f"PROGRESS sweep/{a.dataset} {have}/{tot} [did={did}] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
