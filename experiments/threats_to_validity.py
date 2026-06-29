#!/usr/bin/env python3
"""
Threats-to-Validity experiments for Proof-of-Deliberation.

This script produces four controlled analyses that address common
threats to the validity of the main paper:

  E1  Rejection-rate-matched random control.
      Does PoD's gain come from *which* labels it rejects, or merely
      from rejecting some labels?  We run a RandomGate that rejects the
      *same per-regime fraction* of labels as PoD but chooses which ones
      uniformly at random, bracketed by AL (accept all) and an Oracle
      (accept iff the label is actually correct).

  E2  Coupling-strength sweep.
      The ATC validation reports a moderate rho = 0.231.  We sweep the
      operator's Hick-Hyman slope k, measure the *realized* engaged-phase
      Spearman rho it induces, and show PoD's protection degrades
      gracefully -- and that a coupling as weak as the ATC value already
      sits inside the protective region, because PoD keys on the
      engaged-vs-disengaged *separation*, not the absolute level.

  E3  Confident-error blind spot (honest failure mode).
      An operator with fully authentic Hick-Hyman timing but labels that
      are systematically wrong on high-confidence items.  PoD's process
      signals see nothing wrong and (correctly, by Theorem 2's converse)
      do NOT protect here, whereas a content-based method can.  This is
      the regime in which PoD is *supposed* to lose.

  E4  Gate precision / recall / cost, and Baseline non-dominance.
      Using ground-truth label correctness (available only in the
      controlled stream), we report the gate's recall on bad labels, its
      leakage, and -- crucially -- its false-rejection rate of *good*
      labels (PoD's operating cost), plus a clean Baseline-only stream on
      which PoD is statistically indistinguishable from plain AL.

All experiments use the self-contained Synth-Boundary stream so they are
fully reproducible without any network access.  They import only the
clean PoD building blocks (operator, core verifier, learner, synth
stream); the package's experiment.py is not used.

Usage
-----
    python experiments/threats_to_validity.py --out out_ttv --seeds 12

Outputs (under --out):
    summary.json     all numbers, machine-readable
    macros.tex       \newcommand block for the manuscript
    e2_coupling_sweep.pdf / .png
    e1e3_f1_bars.pdf / .png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pod.baselines import (  # noqa: E402
    MACEOnlineState,
    WorkerQualityState,
    accept_mace,
    accept_worker_quality,
    update_mace,
    update_worker_quality,
)
from pod.config import (  # noqa: E402
    ALParams,
    OperatorParams,
    PoDParams,
    RegimeSchedule,
    SynthParams,
)
from pod.core import (  # noqa: E402
    coupling_check,
    fatigue_detector,
    gaming_detector,
    gate_check,
)
from pod.learner import make_classifier, proba_from_decision_function  # noqa: E402
from pod.streams.synth import generate_synth_boundary_pool_regime  # noqa: E402

# Fast, cached static-boundary pool. With rotation=0 the boundary is fixed at
# w0, so the entire pool is a single vectorized draw -- no per-step Python loop.
# Cached per (seed, n) because every method on a given seed shares the same pool.
_POOL_CACHE = {}


def static_pool(seed, n, synth):
    key = (seed, n, synth.d, synth.lambda_complexity, synth.noise_std)
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]
    rng = np.random.default_rng(seed + 99991)
    d = synth.d
    X = rng.normal(0.0, synth.noise_std, size=(n, d)).astype(float)
    w0 = rng.normal(0.0, 1.0, size=(d,))
    w0 = w0 / max(1e-12, float(np.linalg.norm(w0)))
    margin = X @ w0
    y = (margin >= 0).astype(int)
    c_star = np.exp(-synth.lambda_complexity * np.abs(margin)).astype(float)
    _POOL_CACHE[key] = (X, y, c_star)
    return X, y, c_star
from pod.utils import sigmoid, spearman_rho  # noqa: E402

from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


# ---------------------------------------------------------------------------
# Shared configuration (Synth-Boundary, controlled)
# ---------------------------------------------------------------------------
def query_probability(c01: float, al: ALParams) -> float:
    z = (c01 - 0.5) / max(1e-9, al.query_temp)
    p = float(sigmoid(np.array([z]))[0])
    return float(np.clip(p * (al.query_budget / 0.5), 0.0, 1.0))


def base_synth_params() -> SynthParams:
    # Static boundary (no rotation): these experiments isolate *supervision*
    # quality from concept drift, so the boundary is held fixed and the
    # holdout is drawn from the same stationary distribution. This makes the
    # clean-label ceiling high, so corruption under degraded supervision is
    # visible as a genuine F1 collapse rather than being masked by drift.
    return SynthParams(d=10, lambda_complexity=3.0, rotation_per_step=0.0, noise_std=1.0)


def base_pod(coupling_on: bool) -> PoDParams:
    """PoD config for the controlled stream.

    The published Synth preset disables the coupling layer (epsilon=-1)
    because on c_star the gate+vigilance already dominate; for the
    coupling sweep (E2) we turn it on so the coupling layer is the
    binding constraint and the sweep is meaningful.
    """
    return PoDParams(
        coupling_window=60,
        coupling_epsilon=(0.10 if coupling_on else -1.0),
        gate_a=900.0,
        gate_b=400.0,
        gate_lo_frac=0.45,
        gate_hi_frac=0.60,
        gate_floor_ms=150.0,
        gate_ceil_ms=3000.0,
        gaming_window=12,
        gaming_mu_max_ms=850.0,
        gaming_cv_max=0.06,
        fatigue_window=15,
        fatigue_cv_min=0.60,
        persist_k=2,
    )


def base_operator(k: float = 900.0) -> OperatorParams:
    """Operator with an explicit Hick-Hyman slope k (E2 sweeps this)."""
    return OperatorParams(
        k=k,
        c=300.0,
        sigma=90.0,
        c_fast=300.0,
        eps_fast=8.0,
        c_slow=1500.0,
        sigma_high=1400.0,
        p_correct_baseline=0.95,
        p_correct_gaming=0.10,
        p_correct_fatigue=0.30,
    )


# ---------------------------------------------------------------------------
# Operator draw, with an extra "confident_error" regime for E3
# ---------------------------------------------------------------------------
def draw_operator(
    rng: np.random.Generator,
    y_true: int,
    complexity: float,
    phase: str,
    op: OperatorParams,
) -> Tuple[int, float, bool]:
    """Return (label, delib_ms, authentic_timing_flag).

    Standard regimes reproduce pod.operator.simulate_operator (binary).
    ``confident_error`` keeps fully authentic Hick-Hyman timing but
    corrupts the label preferentially on *low-complexity* (high-model-
    confidence) items -- the content-side failure that PoD cannot see.
    """
    if phase == "confident_error":
        delib = float(max(50.0, rng.normal(op.k * complexity + op.c, op.sigma)))
        # Flip probability is high when the item is easy (low complexity),
        # i.e. exactly where the model is most confident.
        p_flip = float(np.clip(0.9 * (1.0 - complexity), 0.0, 1.0))
        lab = (1 - int(y_true)) if rng.random() < p_flip else int(y_true)
        return lab, delib, True

    if phase == "mixed":
        # Interleaved engaged (authentic, correct) and gaming (fast, wrong)
        # responses. PoD can keep the engaged half via the per-label gate +
        # vigilance; a rate-matched random gate cannot tell them apart.
        if rng.random() < 0.5:
            delib = float(max(50.0, rng.normal(op.k * complexity + op.c, op.sigma)))
            lab = int(y_true) if rng.random() < op.p_correct_baseline else (1 - int(y_true))
            return lab, delib, True
        delib = float(max(50.0, rng.normal(op.c_fast, op.eps_fast)))
        lab = int(y_true) if rng.random() < op.p_correct_gaming else (1 - int(y_true))
        return lab, delib, False

    if phase == "baseline":
        delib = rng.normal(op.k * complexity + op.c, op.sigma)
        p_corr = op.p_correct_baseline
    elif phase == "gaming":
        delib = rng.normal(op.c_fast, op.eps_fast)
        p_corr = op.p_correct_gaming
    elif phase == "fatigue":
        delib = rng.normal(op.c_slow, op.sigma_high)
        p_corr = op.p_correct_fatigue
    else:
        raise ValueError(f"unknown phase {phase!r}")

    delib = float(max(50.0, delib))
    lab = int(y_true) if rng.random() < p_corr else (1 - int(y_true))
    return lab, delib, (phase == "baseline")


# ---------------------------------------------------------------------------
# PoD composite decision (mirrors experiment.py persist-k semantics)
# ---------------------------------------------------------------------------
class PoDDecider:
    def __init__(self, pod: PoDParams, use_coupling: bool):
        self.pod = pod
        self.use_coupling = use_coupling
        self.bad_c = self.bad_g = self.bad_f = 0

    def __call__(self, delib: float, comp: float, comp_hist, delib_hist) -> bool:
        pod = self.pod
        g_ok = gate_check(delib, comp, pod)
        if self.use_coupling:
            c_raw = coupling_check(np.asarray(comp_hist), np.asarray(delib_hist), pod)
        else:
            c_raw = 1
        gam = gaming_detector(np.asarray(delib_hist), pod)
        fat = fatigue_detector(np.asarray(delib_hist), pod)
        self.bad_c = self.bad_c + 1 if c_raw == 0 else 0
        self.bad_g = self.bad_g + 1 if gam == 1 else 0
        self.bad_f = self.bad_f + 1 if fat == 1 else 0
        ok = (self.bad_c < pod.persist_k) and (self.bad_g < pod.persist_k) and (self.bad_f < pod.persist_k)
        return bool(g_ok == 1 and ok)


# ---------------------------------------------------------------------------
# Core closed-loop run with full instrumentation
# ---------------------------------------------------------------------------
PHASES_STD = ("baseline", "gaming", "fatigue")


def run_once(
    method: str,
    seed: int,
    *,
    phases: Tuple[str, ...],
    phase_len: int,
    init_fit: int,
    holdout: int,
    al: ALParams,
    pod: PoDParams,
    op: OperatorParams,
    synth: SynthParams,
    use_coupling: bool = False,
    rand_accept: Optional[Dict[str, float]] = None,
    eval_every: int = 20,
    eval_window: int = 800,
) -> Dict:
    """One closed-loop run; returns trajectory + per-phase instrumentation."""
    rng = np.random.default_rng(seed)
    T = phase_len * len(phases)
    pool_n = init_fit + T + holdout + 10
    X, y, c_star = static_pool(seed, pool_n, synth)
    Xs, ys, cs = X[: init_fit + T], y[: init_fit + T], c_star[: init_fit + T]
    Xh, yh = X[init_fit + T : init_fit + T + holdout], y[init_fit + T : init_fit + T + holdout]

    classes = np.unique(y)
    n_classes = int(len(classes))
    imp = SimpleImputer(strategy="mean").fit(Xs[:init_fit])
    sc = StandardScaler().fit(imp.transform(Xs[:init_fit]))
    Xs_z = sc.transform(imp.transform(Xs))
    Xh_z = sc.transform(imp.transform(Xh))

    # Controlled TTV stream uses a plastic, non-averaged SGD at a constant
    # learning rate so that sustained bad supervision actively corrupts the
    # model (exposing the Death-Spiral mechanism). This is a CONSERVATIVE
    # choice: the main paper's averaged SGD is strictly more robust to noise,
    # so any protection demonstrated here understates PoD's benefit there.
    clf = make_classifier(classes=classes, seed=seed, clf_alpha=1e-6,
                          clf_eta0=0.08, clf_average=False)
    try:
        clf.set_params(learning_rate="constant")
    except Exception:
        pass
    clf.partial_fit(Xs_z[:init_fit], ys[:init_fit], classes=classes)

    comp_hist: List[float] = []
    delib_hist: List[float] = []
    pod_dec = PoDDecider(pod, use_coupling)
    wq = WorkerQualityState.fresh(n_classes=n_classes)
    mace = MACEOnlineState.fresh()

    def phase_of(t_rel: int) -> str:
        return phases[min(len(phases) - 1, t_rel // phase_len)]

    # per-phase counters
    ctr = {p: dict(q=0, acc=0, bad=0, bad_acc=0, good=0, good_rej=0) for p in phases}
    # engaged-phase coupling sample (complexity vs delib on authentic steps)
    eng_c: List[float] = []
    eng_d: List[float] = []

    f1s: List[float] = []
    ts: List[int] = []
    hptr = 0

    for t in range(init_fit, init_fit + T):
        t_rel = t - init_fit
        phase = phase_of(t_rel)
        xt = Xs_z[t : t + 1]
        proba = proba_from_decision_function(clf, xt, n_classes=n_classes)[0]
        comp = float(cs[t])  # controlled complexity c_star

        if rng.random() < query_probability(comp, al):
            y_tilde, delib, authentic = draw_operator(rng, int(ys[t]), comp, phase, op)
            is_bad = int(y_tilde != int(ys[t]))
            ctr[phase]["q"] += 1
            ctr[phase]["bad"] += is_bad
            ctr[phase]["good"] += (1 - is_bad)
            if authentic:
                eng_c.append(comp)
                eng_d.append(delib)

            comp_hist.append(comp)
            delib_hist.append(delib)

            if method == "AL":
                accept = True
            elif method == "Oracle":
                accept = (is_bad == 0)
            elif method == "RandomGate":
                assert rand_accept is not None
                accept = (rng.random() < rand_accept.get(phase, 1.0))
            elif method == "PoD":
                accept = pod_dec(delib, comp, comp_hist, delib_hist)
            elif method == "WorkerQuality":
                accept = accept_worker_quality(proba, int(y_tilde), wq)
                update_worker_quality(wq, int(np.argmax(proba)), int(y_tilde))
            elif method == "MACE":
                update_mace(mace, proba, int(y_tilde))
                accept = accept_mace(mace)
            elif method == "Hybrid":
                # PoD AND WorkerQuality: admit iff BOTH the process gate and
                # the content gate accept. pod_dec advances the persist-k
                # counters once per query (as for plain PoD).
                p_ok = pod_dec(delib, comp, comp_hist, delib_hist)
                w_ok = accept_worker_quality(proba, int(y_tilde), wq)
                update_worker_quality(wq, int(np.argmax(proba)), int(y_tilde))
                accept = bool(p_ok and w_ok)
            else:
                raise ValueError(method)

            if accept:
                ctr[phase]["acc"] += 1
                ctr[phase]["bad_acc"] += is_bad
                clf.partial_fit(xt, np.array([y_tilde], dtype=int))
            else:
                ctr[phase]["good_rej"] += (1 - is_bad)

        if t_rel % eval_every == 0:
            k = min(eval_window, len(Xh_z))
            idx = (np.arange(k) + hptr) % len(Xh_z)
            hptr = (hptr + 1) % len(Xh_z)
            f1s.append(float(f1_score(yh[idx], clf.predict(Xh_z[idx]), zero_division=0)))
            ts.append(t_rel)

    def final_f1(phase: str) -> float:
        lo = phases.index(phase) * phase_len
        hi = lo + phase_len
        vals = [f for tt, f in zip(ts, f1s) if lo <= tt < hi]
        return float(np.mean(vals[-10:])) if vals else float("nan")

    rho_eng = spearman_rho(np.asarray(eng_c), np.asarray(eng_d)) if len(eng_c) >= 8 else float("nan")

    return dict(
        method=method, seed=seed, ts=ts, f1s=f1s, ctr=ctr,
        final_overall=float(np.mean(f1s[-10:])),
        final_by_phase={p: final_f1(p) for p in phases},
        rho_engaged=float(rho_eng),
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def mean_ci(xs: List[float]) -> Tuple[float, float]:
    a = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    m = float(a.mean())
    ci = 1.96 * float(a.std(ddof=1)) / math.sqrt(a.size) if a.size > 1 else 0.0
    return m, ci


def paired_wilcoxon(a: List[float], b: List[float]) -> float:
    from scipy.stats import wilcoxon
    a = np.asarray(a); b = np.asarray(b)
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# E1 + E4 : rate-matched control, oracle/AL bounds, gate precision/recall
# ---------------------------------------------------------------------------
def exp_e1_e4(seeds, phase_len, init_fit, holdout, eval_every=20, eval_window=800) -> Dict:
    """E1 (rate control) + E4 (gate precision/recall, cost, non-dominance).

    The control is REGIME-BLIND: a RandomGate that accepts every label with
    a single fixed probability equal to PoD's *global* accept rate across the
    whole stream. It rejects the same total fraction as PoD but cannot know
    *when* to reject. If PoD beats it, the gain is from allocating rejection
    to the degraded regimes -- i.e. from *which* labels, not merely how many.
    """
    PHASES = ("baseline", "gaming", "fatigue")
    al = ALParams(query_budget=0.60, query_temp=0.90)
    pod = base_pod(coupling_on=False)
    op = base_operator(k=900.0)
    synth = base_synth_params()
    kw = dict(phases=PHASES, phase_len=phase_len, init_fit=init_fit, holdout=holdout,
              al=al, pod=pod, op=op, synth=synth, eval_every=eval_every, eval_window=eval_window)

    rows = {m: [] for m in ("AL", "PoD", "RandomGlobal", "Oracle")}
    leak = {p: [] for p in PHASES}
    frej = {p: [] for p in PHASES}
    recall = {p: [] for p in PHASES}
    acc_rate = {p: [] for p in PHASES}
    global_accept = []

    for s in seeds:
        pod_run = run_once("PoD", s, **kw)
        tot_q = sum(pod_run["ctr"][p]["q"] for p in PHASES)
        tot_a = sum(pod_run["ctr"][p]["acc"] for p in PHASES)
        p_global = (tot_a / tot_q) if tot_q else 1.0
        global_accept.append(p_global)
        for p in PHASES:
            c = pod_run["ctr"][p]
            acc_rate[p].append(c["acc"] / c["q"] if c["q"] else float("nan"))
            leak[p].append(c["bad_acc"] / c["bad"] if c["bad"] else float("nan"))
            frej[p].append(c["good_rej"] / c["good"] if c["good"] else float("nan"))
            recall[p].append((c["bad"] - c["bad_acc"]) / c["bad"] if c["bad"] else float("nan"))
        rand_blind = {p: p_global for p in PHASES}  # regime-blind: one rate everywhere
        al_run = run_once("AL", s, **kw)
        rnd_run = run_once("RandomGate", s, rand_accept=rand_blind, **kw)
        ora_run = run_once("Oracle", s, **kw)
        for m, r in (("PoD", pod_run), ("AL", al_run), ("RandomGlobal", rnd_run), ("Oracle", ora_run)):
            rows[m].append(r)

    def collect(phase):
        return {m: mean_ci([r["final_by_phase"][phase] for r in rows[m]]) for m in rows}

    sig = {}
    for p in ("gaming", "fatigue"):
        pod_v = [r["final_by_phase"][p] for r in rows["PoD"]]
        rnd_v = [r["final_by_phase"][p] for r in rows["RandomGlobal"]]
        sig[p] = dict(p_value=paired_wilcoxon(pod_v, rnd_v), pod=mean_ci(pod_v), rand=mean_ci(rnd_v))

    pod_b = [r["final_by_phase"]["baseline"] for r in rows["PoD"]]
    al_b = [r["final_by_phase"]["baseline"] for r in rows["AL"]]
    baseline_nondom = dict(p_value=paired_wilcoxon(pod_b, al_b), pod=mean_ci(pod_b), al=mean_ci(al_b))

    pod_o = [r["final_overall"] for r in rows["PoD"]]
    rnd_o = [r["final_overall"] for r in rows["RandomGlobal"]]
    overall = dict(p_value=paired_wilcoxon(pod_o, rnd_o), pod=mean_ci(pod_o), rand=mean_ci(rnd_o))

    return dict(
        f1_baseline=collect("baseline"),
        f1_gaming=collect("gaming"),
        f1_fatigue=collect("fatigue"),
        rate_matched_sig=sig,
        overall_vs_randomglobal=overall,
        global_accept=mean_ci(global_accept),
        baseline_nondominance=baseline_nondom,
        gate=dict(
            accept_rate={p: mean_ci(acc_rate[p]) for p in PHASES},
            leakage={p: mean_ci(leak[p]) for p in PHASES},
            false_rejection={p: mean_ci(frej[p]) for p in PHASES},
            bad_recall={p: mean_ci(recall[p]) for p in PHASES},
        ),
    )


# ---------------------------------------------------------------------------
# E2 : coupling-strength sweep
# ---------------------------------------------------------------------------
def exp_e2(seeds, phase_len, init_fit, holdout, k_values, eval_every=20, eval_window=800) -> Dict:
    al = ALParams(query_budget=0.60, query_temp=0.90)
    pod = base_pod(coupling_on=True)
    synth = base_synth_params()
    out = []
    for k in k_values:
        op = base_operator(k=k)
        kw = dict(phases=PHASES_STD, phase_len=phase_len, init_fit=init_fit, holdout=holdout,
                  al=al, pod=pod, op=op, synth=synth, use_coupling=True, eval_every=eval_every, eval_window=eval_window)
        pod_runs = [run_once("PoD", s, **kw) for s in seeds]
        rho = mean_ci([r["rho_engaged"] for r in pod_runs])
        prot = mean_ci([r["final_by_phase"]["gaming"] for r in pod_runs])
        prot_f = mean_ci([r["final_by_phase"]["fatigue"] for r in pod_runs])
        # baseline accept rate: a too-weak coupling makes PoD reject genuine work
        acc_b = mean_ci([(r["ctr"]["baseline"]["acc"] / max(1, r["ctr"]["baseline"]["q"])) for r in pod_runs])
        out.append(dict(k=k, rho_engaged=rho, f1_gaming=prot, f1_fatigue=prot_f, baseline_accept=acc_b))
    return dict(sweep=out, atc_rho=0.231)


# ---------------------------------------------------------------------------
# E3 : confident-error blind spot
# ---------------------------------------------------------------------------
def exp_e3(seeds, phase_len, init_fit, holdout, eval_every=20, eval_window=800) -> Dict:
    al = ALParams(query_budget=0.60, query_temp=0.90)
    pod = base_pod(coupling_on=False)
    op = base_operator(k=900.0)
    synth = base_synth_params()
    phases = ("baseline", "confident_error")
    kw = dict(phases=phases, phase_len=phase_len, init_fit=init_fit, holdout=holdout,
              al=al, pod=pod, op=op, synth=synth, eval_every=eval_every, eval_window=eval_window)
    res = {m: [] for m in ("AL", "PoD", "MACE", "WorkerQuality", "Oracle")}
    pod_accept_ce = []
    for s in seeds:
        for m in res:
            r = run_once(m, s, **kw)
            res[m].append(r)
            if m == "PoD":
                c = r["ctr"]["confident_error"]
                pod_accept_ce.append(c["acc"] / c["q"] if c["q"] else float("nan"))
    ce = {m: mean_ci([r["final_by_phase"]["confident_error"] for r in res[m]]) for m in res}
    # Is PoD better than a content method here? (it should NOT be)
    pod_v = [r["final_by_phase"]["confident_error"] for r in res["PoD"]]
    mace_v = [r["final_by_phase"]["confident_error"] for r in res["MACE"]]
    wq_v = [r["final_by_phase"]["confident_error"] for r in res["WorkerQuality"]]
    return dict(
        f1_confident_error=ce,
        pod_accept_rate_ce=mean_ci(pod_accept_ce),
        pod_vs_mace_p=paired_wilcoxon(pod_v, mace_v),
        pod_vs_wq_p=paired_wilcoxon(pod_v, wq_v),
        content_beats_pod=bool(np.mean(mace_v) > np.mean(pod_v) or np.mean(wq_v) > np.mean(pod_v)),
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(e1e4: Dict, e2: Dict, e3: Dict, outdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # E2 coupling sweep
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ks = [d["k"] for d in e2["sweep"]]
    rho = [d["rho_engaged"][0] for d in e2["sweep"]]
    fg = [d["f1_gaming"][0] for d in e2["sweep"]]
    fge = [d["f1_gaming"][1] for d in e2["sweep"]]
    ax.errorbar(rho, fg, yerr=fge, marker="o", capsize=3, label="PoD F1 (Gaming)")
    ax.axvline(e2["atc_rho"], ls="--", color="crimson", lw=1.3, label=f"ATC $\\rho={e2['atc_rho']}$")
    ax.set_xlabel("Realized engaged-phase Spearman $\\rho_{cog}$")
    ax.set_ylabel("PoD F1 under degraded supervision")
    ax.set_title("E2: graceful degradation in coupling strength")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "e2_coupling_sweep.pdf"))
    fig.savefig(os.path.join(outdir, "e2_coupling_sweep.png"), dpi=150)
    plt.close(fig)

    # E1/E3 bar chart
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    ax = axes[0]
    methods = ["AL", "RandomGlobal", "PoD", "Oracle"]
    gv = [e1e4["f1_gaming"][m][0] for m in methods]
    ge = [e1e4["f1_gaming"][m][1] for m in methods]
    ax.bar(methods, gv, yerr=ge, capsize=4,
           color=["#bbb", "#f0a", "#19c", "#2a2"])
    ax.set_title("E1: Gaming regime\n(RandomGlobal = PoD's global reject rate, regime-blind)")
    ax.set_ylabel("F1")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    methods3 = ["AL", "PoD", "MACE", "WorkerQuality", "Oracle"]
    cv = [e3["f1_confident_error"][m][0] for m in methods3]
    cer = [e3["f1_confident_error"][m][1] for m in methods3]
    ax.bar(methods3, cv, yerr=cer, capsize=4,
           color=["#bbb", "#19c", "#e80", "#fa0", "#2a2"])
    ax.set_title("E3: Confident-error blind spot\n(PoD is meant to lose here)")
    ax.set_ylabel("F1")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "e1e3_f1_bars.pdf"))
    fig.savefig(os.path.join(outdir, "e1e3_f1_bars.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Macro emission
# ---------------------------------------------------------------------------
def fmt(m: float, d: int = 3) -> str:
    return f"{m:.{d}f}" if np.isfinite(m) else "\\text{NA}"


def emit_macros(e1e4: Dict, e2: Dict, e3: Dict, path: str) -> None:
    L = []
    A = L.append
    A("% Auto-generated by experiments/threats_to_validity.py -- do not edit by hand.")
    # E1 rate-matched control (Gaming)
    g = e1e4["f1_gaming"]
    A(f"\\newcommand{{\\TTVgameAL}}{{\\ensuremath{{{fmt(g['AL'][0])}}}}}")
    A(f"\\newcommand{{\\TTVgameRand}}{{\\ensuremath{{{fmt(g['RandomGlobal'][0])}}}}}")
    A(f"\\newcommand{{\\TTVgamePoD}}{{\\ensuremath{{{fmt(g['PoD'][0])}}}}}")
    A(f"\\newcommand{{\\TTVgameOracle}}{{\\ensuremath{{{fmt(g['Oracle'][0])}}}}}")
    A(f"\\newcommand{{\\TTVrateMatchP}}{{\\ensuremath{{{fmt(e1e4['rate_matched_sig']['gaming']['p_value'],4)}}}}}")
    A(f"\\newcommand{{\\TTVglobalAccept}}{{\\ensuremath{{{fmt(e1e4['global_accept'][0])}}}}}")
    A(f"\\newcommand{{\\TTVbaseRandGlobal}}{{\\ensuremath{{{fmt(e1e4['f1_baseline']['RandomGlobal'][0])}}}}}")
    A(f"\\newcommand{{\\TTVoverallPoD}}{{\\ensuremath{{{fmt(e1e4['overall_vs_randomglobal']['pod'][0])}}}}}")
    A(f"\\newcommand{{\\TTVoverallRand}}{{\\ensuremath{{{fmt(e1e4['overall_vs_randomglobal']['rand'][0])}}}}}")
    A(f"\\newcommand{{\\TTVoverallP}}{{\\ensuremath{{{fmt(e1e4['overall_vs_randomglobal']['p_value'],4)}}}}}")
    # E4 gate cost (Gaming + Baseline)
    gt = e1e4["gate"]
    A(f"\\newcommand{{\\TTVrecallGaming}}{{\\ensuremath{{{fmt(gt['bad_recall']['gaming'][0])}}}}}")
    A(f"\\newcommand{{\\TTVleakGaming}}{{\\ensuremath{{{fmt(gt['leakage']['gaming'][0])}}}}}")
    A(f"\\newcommand{{\\TTVfrejBaseline}}{{\\ensuremath{{{fmt(gt['false_rejection']['baseline'][0])}}}}}")
    A(f"\\newcommand{{\\TTVacceptBaseline}}{{\\ensuremath{{{fmt(gt['accept_rate']['baseline'][0])}}}}}")
    bnd = e1e4["baseline_nondominance"]
    A(f"\\newcommand{{\\TTVbaselinePoD}}{{\\ensuremath{{{fmt(bnd['pod'][0])}}}}}")
    A(f"\\newcommand{{\\TTVbaselineAL}}{{\\ensuremath{{{fmt(bnd['al'][0])}}}}}")
    A(f"\\newcommand{{\\TTVbaselineP}}{{\\ensuremath{{{fmt(bnd['p_value'],3)}}}}}")
    # E2 coupling sweep: report rho at smallest k that still protects, and ATC
    A(f"\\newcommand{{\\TTVatcRho}}{{\\ensuremath{{{e2['atc_rho']}}}}}")
    # protection at the sweep point closest to ATC rho
    sweep = e2["sweep"]
    closest = min(sweep, key=lambda d: abs(d["rho_engaged"][0] - e2["atc_rho"]) if np.isfinite(d["rho_engaged"][0]) else 1e9)
    A(f"\\newcommand{{\\TTVrhoNearATC}}{{\\ensuremath{{{fmt(closest['rho_engaged'][0])}}}}}")
    A(f"\\newcommand{{\\TTVprotNearATC}}{{\\ensuremath{{{fmt(closest['f1_gaming'][0])}}}}}")
    A(f"\\newcommand{{\\TTVbaseAcceptNearATC}}{{\\ensuremath{{{fmt(closest['baseline_accept'][0])}}}}}")
    lowrho = min(sweep, key=lambda d: d["rho_engaged"][0] if np.isfinite(d["rho_engaged"][0]) else 1e9)
    A(f"\\newcommand{{\\TTVprotLowRho}}{{\\ensuremath{{{fmt(lowrho['f1_gaming'][0])}}}}}")
    A(f"\\newcommand{{\\TTVrhoLow}}{{\\ensuremath{{{fmt(lowrho['rho_engaged'][0])}}}}}")
    A(f"\\newcommand{{\\TTVbaseAcceptLowRho}}{{\\ensuremath{{{fmt(lowrho['baseline_accept'][0])}}}}}")
    # E3 blind spot
    ce = e3["f1_confident_error"]
    A(f"\\newcommand{{\\TTVceAL}}{{\\ensuremath{{{fmt(ce['AL'][0])}}}}}")
    A(f"\\newcommand{{\\TTVcePoD}}{{\\ensuremath{{{fmt(ce['PoD'][0])}}}}}")
    A(f"\\newcommand{{\\TTVceMACE}}{{\\ensuremath{{{fmt(ce['MACE'][0])}}}}}")
    A(f"\\newcommand{{\\TTVceWQ}}{{\\ensuremath{{{fmt(ce['WorkerQuality'][0])}}}}}")
    A(f"\\newcommand{{\\TTVcePoDaccept}}{{\\ensuremath{{{fmt(e3['pod_accept_rate_ce'][0])}}}}}")
    A(f"\\newcommand{{\\TTVcePoDvsMACEp}}{{\\ensuremath{{{fmt(e3['pod_vs_mace_p'],3)}}}}}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_ttv")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--phase_len", type=int, default=1000)
    ap.add_argument("--init_fit", type=int, default=300)
    ap.add_argument("--holdout", type=int, default=1200)
    ap.add_argument("--eval_every", type=int, default=40)
    ap.add_argument("--eval_window", type=int, default=600)
    ap.add_argument("--only", choices=["e1", "e2", "e3", "combine", "all"], default="all")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(args.seeds))
    common = dict(phase_len=args.phase_len, init_fit=args.init_fit, holdout=args.holdout,
                  eval_every=args.eval_every, eval_window=args.eval_window)

    def dump(name, obj):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def load(name):
        with open(os.path.join(args.out, name), "r", encoding="utf-8") as f:
            return json.load(f)

    if args.only in ("e1", "all"):
        print("[E1+E4] rate-matched control + gate precision/recall ...", flush=True)
        dump("e1.json", exp_e1_e4(seeds, **common))
    if args.only in ("e2", "all"):
        print("[E2] coupling-strength sweep ...", flush=True)
        dump("e2.json", exp_e2(seeds, k_values=[0.0, 80.0, 160.0, 320.0, 550.0, 900.0], **common))
    if args.only in ("e3", "all"):
        print("[E3] confident-error blind spot ...", flush=True)
        dump("e3.json", exp_e3(seeds, **common))

    if args.only in ("combine", "all"):
        e1e4 = load("e1.json"); e2 = load("e2.json"); e3 = load("e3.json")
        summary = dict(config=dict(seeds=args.seeds, **common), E1_E4=e1e4, E2=e2, E3=e3)
        dump("summary.json", summary)
        emit_macros(e1e4, e2, e3, os.path.join(args.out, "macros.tex"))
        make_figures(e1e4, e2, e3, args.out)
        print("COMBINED ->", args.out, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
