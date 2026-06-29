"""
Closed-loop experiment runner.

This is the *operational core* of the paper's empirical evaluation. The
top-level function :func:`run_stream_experiment_once` realises one run
of one method on one stream: at each step it (i) estimates task
complexity, (ii) decides whether to query, (iii) calls the operator
simulator, (iv) applies the method's acceptance rule, (v) optionally
updates the base learner, and (vi) periodically scores the model on a
disjoint sliding holdout.

The aggregator :func:`run_suite_generic` wraps :func:`run_stream_experiment_once`
with the multi-run / multi-method bookkeeping needed to produce the
mean F1 trajectories with 95% confidence intervals reported in the
paper.

By design, all numerical behaviour is identical to the canonical
``pod-unified.py`` reference script; only the structure and
documentation have been improved.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from pod.baselines import (
    IEThreshState,
    MACEOnlineState,
    RaykarOnlineState,
    WorkerQualityState,
    accept_adaptive,
    accept_al,
    accept_iethresh,
    accept_mace,
    accept_raykar,
    accept_static,
    accept_worker_quality,
    get_pod_ablation,
    is_pod_family,
    update_iethresh,
    update_mace,
    update_raykar,
    update_worker_quality,
)
from pod.config import ALParams, OperatorParams, PoDParams, RegimeSchedule
from pod.core import coupling_check, fatigue_detector, gaming_detector, gate_check
from pod.learner import make_classifier, proba_from_decision_function
from pod.operator import simulate_operator
from pod.utils import ensure_dir, entropy_from_proba, entropy_unit, sigmoid
from pod.viz import plot_methods

METHODS: Tuple[str, ...] = (
    "AL",
    "StaticGating",
    "AdaptiveGating",
    "WorkerQuality",
    "Raykar",
    "MACE",
    "IEThresh",
    "PoD",
    "PoD-NoGate",
    "PoD-NoCoupling",
    "PoD-NoVigilance",
)
"""Methods compared in every experiment, in legend order.

The panel covers three families: (i) classical AL / static-gating
baselines (``AL``, ``StaticGating``, ``AdaptiveGating``);
(ii) content-based annotation-quality competitors drawn from the
crowdsourcing literature (``WorkerQuality`` = single-annotator
Dawid-Skene, ``Raykar`` = streaming Raykar et al. JMLR 2010,
``MACE`` = streaming Hovy et al. NAACL 2013, ``IEThresh`` =
Donmez & Carbonell ECML 2008); and (iii) process-based ``PoD`` plus
its three single-component ablations
(``PoD-NoGate``, ``PoD-NoCoupling``, ``PoD-NoVigilance``)."""


# ---------------------------------------------------------------------------
# Active-learning query policy
# ---------------------------------------------------------------------------
def query_probability(complexity_01: float, al: ALParams) -> float:
    """Per-step Bernoulli probability of querying a label.

    A logistic transformation centres the complexity at ``0.5`` and
    rescales by ``query_budget`` so that the expected query rate
    integrates to roughly ``query_budget`` under a uniform complexity
    distribution.
    """
    z = (complexity_01 - 0.5) / max(1e-9, al.query_temp)
    p = float(sigmoid(np.array([z]))[0])
    return float(np.clip(p * (al.query_budget / 0.5), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------
def f1_metric(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Binary F1 for two-class streams; macro F1 otherwise."""
    if n_classes <= 2:
        return float(f1_score(y_true, y_pred, zero_division=0))
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Holdout helper (used by the OpenML stream)
# ---------------------------------------------------------------------------
def make_external_holdout(
    X_all: np.ndarray,
    y_all: np.ndarray,
    stream_len: int,
    holdout_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a holdout set disjoint from the supervision stream when possible."""
    if len(X_all) > stream_len:
        X_tail = X_all[stream_len:]
        y_tail = y_all[stream_len:]
        if len(X_tail) >= holdout_size:
            idx = rng.choice(len(X_tail), size=holdout_size, replace=False)
            return X_tail[idx], y_tail[idx]

    replace = holdout_size > len(X_all)
    idx = rng.choice(len(X_all), size=holdout_size, replace=replace)
    return X_all[idx], y_all[idx]


# ---------------------------------------------------------------------------
# Single-run experiment
# ---------------------------------------------------------------------------
def run_stream_experiment_once(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    seed: int,
    schedule: RegimeSchedule,
    method: str,
    al: ALParams,
    pod: PoDParams,
    op: OperatorParams,
    eval_window: int,
    eval_every: int,
    init_fit: int,
    complexity_mode: str,
    c_star_stream: Optional[np.ndarray] = None,
    class_weight_dict: Optional[Dict[int, float]] = None,
    log_tmax: int = 0,
    clf_alpha: float = 2e-5,
    clf_eta0: float = 0.01,
    clf_average: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Execute one closed-loop run and return F1 trajectory + diagnostics.

    See :ref:`README <docs/REPRODUCIBILITY>` for the seed scheme used to
    derive the per-run seed from the run index.

    Parameters
    ----------
    X_stream, y_stream : np.ndarray
        Stream features and ground-truth labels. The first ``init_fit``
        rows are used as a warm-up batch and never see the operator
        simulator.
    X_holdout, y_holdout : np.ndarray
        Disjoint evaluation set; a sliding window of size
        ``eval_window`` is rotated through it.
    seed : int
        Per-run seed; controls the operator simulator, the query
        Bernoulli draw, and the classifier's internal randomness.
    schedule, al, pod, op : ...
        Frozen configuration objects describing the regimes, the query
        policy, the PoD verification layer, and the operator,
        respectively.
    method : {'AL', 'StaticGating', 'AdaptiveGating', 'PoD'}
        Acceptance policy applied to queried labels.
    eval_window, eval_every : int
        Evaluation window size and inter-evaluation step in stream steps.
    init_fit : int
        Number of warm-up samples used to fit the imputer, the scaler,
        and the initial classifier.
    complexity_mode : {'entropy', 'c_star'}
        Selects whether the per-step complexity ``C_t`` comes from the
        learner's own predictive entropy or from the synthetic stream's
        ground-truth complexity.
    c_star_stream : np.ndarray, optional
        Per-step ``c_star`` values; required for ``complexity_mode='c_star'``.
    class_weight_dict : dict, optional
        Class-weight dict forwarded to the SGD classifier; used for
        unbalanced streams (e.g. Gas Drift).
    log_tmax : int
        Truncate the loop after this many relative steps; ``0`` disables.
    clf_alpha, clf_eta0, clf_average : ...
        Per-dataset overrides for the base learner.

    Returns
    -------
    log : pandas.DataFrame
        Columns ``t`` (relative time step) and ``f1`` (evaluation score).
    diagnostics : dict
        Per-phase query and accept rates.
    """
    rng = np.random.default_rng(seed)

    if init_fit < 1:
        raise ValueError(f"init_fit must be >= 1, got {init_fit}")

    T = schedule.total()
    if len(X_stream) < init_fit + T + 10:
        raise ValueError(
            f"Stream too small: need {init_fit + T + 10}, got {len(X_stream)}"
        )
    if len(X_holdout) < max(500, eval_window):
        raise ValueError(f"Holdout too small: got {len(X_holdout)}")

    Xs = X_stream[: init_fit + T]
    ys = y_stream[: init_fit + T]

    classes = np.unique(np.concatenate([y_stream, y_holdout]))
    n_classes = int(len(classes))

    fit_X = Xs[:init_fit]
    fit_y = ys[:init_fit]
    if fit_X.shape[0] == 0:
        raise ValueError("init_fit produced empty initial fit data")

    imp = SimpleImputer(strategy="mean").fit(fit_X)
    sc = StandardScaler().fit(imp.transform(fit_X))

    Xs_z = sc.transform(imp.transform(Xs))
    Xh_z = sc.transform(imp.transform(X_holdout))

    clf = make_classifier(
        classes=classes,
        seed=seed,
        clf_alpha=clf_alpha,
        clf_eta0=clf_eta0,
        clf_average=clf_average,
        class_weight_dict=class_weight_dict,
    )
    clf.partial_fit(Xs_z[:init_fit], fit_y, classes=classes)

    comp_hist: List[float] = []
    delib_hist: List[float] = []

    bad_c = bad_g = bad_f = 0

    q_total = a_total = 0
    q_phase = {"baseline": 0, "gaming": 0, "fatigue": 0}
    a_phase = {"baseline": 0, "gaming": 0, "fatigue": 0}

    f1s: List[float] = []
    ts: List[int] = []
    hptr = 0

    # Annotation-quality estimator state (only used by the relevant method;
    # all are initialised so the per-step branch can dispatch without
    # conditional construction).
    wq_state = WorkerQualityState.fresh(n_classes=n_classes)
    raykar_state = RaykarOnlineState.fresh(n_classes=n_classes)
    mace_state = MACEOnlineState.fresh()
    ie_state = IEThreshState.fresh()
    pod_abl = get_pod_ablation(method)

    for t in range(init_fit, init_fit + T):
        t_rel = t - init_fit
        if log_tmax > 0 and t_rel > int(log_tmax):
            break

        phase = schedule.phase(t_rel)
        xt = Xs_z[t : t + 1]

        # The learner's predictive distribution is always computed because
        # the WorkerQuality baseline consumes it even when ``complexity_mode``
        # is ``c_star``. The complexity itself is then resolved per the mode.
        proba = proba_from_decision_function(clf, xt, n_classes=n_classes)[0]
        if complexity_mode == "entropy":
            comp = entropy_unit(entropy_from_proba(proba), max(2, n_classes))
        elif complexity_mode == "c_star":
            if c_star_stream is None:
                raise ValueError("complexity_mode='c_star' requires c_star_stream")
            comp = float(c_star_stream[t])
        else:
            raise ValueError(f"Unknown complexity_mode: {complexity_mode!r}")

        if rng.random() < query_probability(comp, al):
            q_total += 1
            q_phase[phase] += 1

            y_tilde, delib = simulate_operator(
                rng, int(ys[t]), n_classes, comp, phase, op
            )

            if method == "AL":
                accept = accept_al()
            elif method == "StaticGating":
                accept = accept_static(delib)
            elif method == "AdaptiveGating":
                accept = accept_adaptive(delib, comp, pod)
            elif method == "WorkerQuality":
                accept = accept_worker_quality(proba, int(y_tilde), wq_state)
                # Always update the worker-quality state on every query,
                # using the model's current argmax as the proxy for y_true.
                y_pred_model = int(np.argmax(proba))
                update_worker_quality(wq_state, y_pred_model, int(y_tilde))
            elif method == "Raykar":
                accept = accept_raykar(proba, int(y_tilde), raykar_state)
                update_raykar(raykar_state, proba, int(y_tilde))
            elif method == "MACE":
                # Update first so the *current* query's evidence
                # contributes to its own decision; this matches MACE's
                # batch behaviour in the limit and keeps the decision
                # well-defined on the very first observation under the
                # uniform Beta(1, 1) prior.
                update_mace(mace_state, proba, int(y_tilde))
                accept = accept_mace(mace_state)
            elif method == "IEThresh":
                update_iethresh(ie_state, proba, int(y_tilde))
                accept = accept_iethresh(ie_state)
            elif pod_abl is not None:
                # PoD family: full PoD or any of the three ablations.
                # ``pod_abl.use_*`` controls which checks contribute; a
                # disabled check is treated as if it had returned the
                # "no violation" indicator.
                g_ok = gate_check(delib, comp, pod) if pod_abl.use_gate else 1

                comp_hist.append(comp)
                delib_hist.append(float(delib))

                if pod_abl.use_coupling:
                    c_raw = coupling_check(
                        np.asarray(comp_hist), np.asarray(delib_hist), pod
                    )
                else:
                    c_raw = 1

                if pod_abl.use_vigilance:
                    gam_raw = gaming_detector(np.asarray(delib_hist), pod)
                    fat_raw = fatigue_detector(np.asarray(delib_hist), pod)
                else:
                    gam_raw = 0
                    fat_raw = 0

                bad_c = bad_c + 1 if c_raw == 0 else 0
                bad_g = bad_g + 1 if gam_raw == 1 else 0
                bad_f = bad_f + 1 if fat_raw == 1 else 0

                ok = (bad_c < pod.persist_k) and (bad_g < pod.persist_k) and (bad_f < pod.persist_k)
                accept = bool((g_ok == 1) and ok)
            else:
                raise ValueError(f"Unknown method: {method!r}")

            if accept:
                a_total += 1
                a_phase[phase] += 1
                clf.partial_fit(xt, np.array([y_tilde], dtype=int))

            # Methods outside the PoD family record history *after* the
            # decision, mirroring the reference script exactly. PoD-family
            # methods record inside the branch above (before the checks).
            if not is_pod_family(method):
                comp_hist.append(comp)
                delib_hist.append(float(delib))

        if t_rel % max(1, eval_every) == 0:
            k = min(eval_window, len(Xh_z))
            idxs = (np.arange(k) + hptr) % len(Xh_z)
            hptr = (hptr + 1) % len(Xh_z)

            ypred = clf.predict(Xh_z[idxs])
            f1s.append(f1_metric(y_holdout[idxs], ypred, n_classes))
            ts.append(t_rel)

    def safe_div(a: float, b: float) -> float:
        return float(a / b) if b > 0 else 0.0

    diag = {
        "query_rate": safe_div(q_total, T),
        "accept_rate_given_query": safe_div(a_total, q_total),
        "query_rate_baseline": safe_div(q_phase["baseline"], max(1, schedule.baseline)),
        "accept_rate_baseline": safe_div(a_phase["baseline"], q_phase["baseline"]),
        "query_rate_gaming": safe_div(q_phase["gaming"], max(1, schedule.gaming)),
        "accept_rate_gaming": safe_div(a_phase["gaming"], q_phase["gaming"]),
        "query_rate_fatigue": safe_div(q_phase["fatigue"], max(1, schedule.fatigue)),
        "accept_rate_fatigue": safe_div(a_phase["fatigue"], q_phase["fatigue"]),
    }

    return pd.DataFrame({"t": ts, "f1": f1s}), diag


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------
def run_suite_generic(
    name: str,
    outdir: str,
    runs: int,
    schedule: RegimeSchedule,
    init_fit: int,
    eval_every: int,
    eval_window: int,
    plot_smooth_w: int,
    phase_label_y: float,
    plot_tmax: int,
    log_tmax: int,
    run_data_fn: Callable[
        [int],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
    ],
    complexity_mode: str,
    class_weight_dict: Optional[Dict[int, float]] = None,
    al: Optional[ALParams] = None,
    pod: Optional[PoDParams] = None,
    op: Optional[OperatorParams] = None,
    clf_alpha: float = 2e-5,
    clf_eta0: float = 0.01,
    clf_average: bool = True,
) -> None:
    """Run all four methods across ``runs`` independent seeds and aggregate.

    Side effects (all written under ``outdir``):

    * ``runs/<name>_<method>_run<N>.csv``  -- per-run F1 trajectories.
    * ``diagnostics.csv``                  -- per-phase query/accept rates.
    * ``figs/<name>_methods_f1.{pdf,png}`` -- the figure used in the paper.
    * ``config.json``                      -- complete frozen-config snapshot.

    Parameters
    ----------
    run_data_fn : Callable[[int], 5-tuple]
        Run-indexed data factory returning
        ``(X_stream, y_stream, X_holdout, y_holdout, c_star_or_None)``.
        Receives the run index so different runs can use disjoint slices
        of the dataset, while keeping intra-run reproducibility.
    """
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "runs"))
    ensure_dir(os.path.join(outdir, "figs"))

    al = al or ALParams()
    pod = pod or PoDParams()
    op = op or OperatorParams()

    logs: Dict[str, List[pd.DataFrame]] = {m: [] for m in METHODS}
    diag_rows: List[Dict[str, Any]] = []

    for r in range(runs):
        Xs, ys, Xh, yh, c_star = run_data_fn(r)

        for m in METHODS:
            df, diag = run_stream_experiment_once(
                X_stream=Xs,
                y_stream=ys,
                X_holdout=Xh,
                y_holdout=yh,
                seed=1000 * r + 7,
                schedule=schedule,
                method=m,
                al=al,
                pod=pod,
                op=op,
                eval_window=eval_window,
                eval_every=eval_every,
                init_fit=init_fit,
                complexity_mode=complexity_mode,
                c_star_stream=c_star,
                class_weight_dict=class_weight_dict,
                log_tmax=log_tmax,
                clf_alpha=clf_alpha,
                clf_eta0=clf_eta0,
                clf_average=clf_average,
            )
            logs[m].append(df)
            df.to_csv(
                os.path.join(outdir, "runs", f"{name}_{m}_run{r}.csv"), index=False
            )

            row = {"dataset": name, "run": r, "method": m}
            row.update(diag)
            diag_rows.append(row)

    pd.DataFrame(diag_rows).to_csv(
        os.path.join(outdir, "diagnostics.csv"), index=False
    )

    # Robust aggregation: truncate all runs per method to the shortest length.
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    t_ref: Optional[np.ndarray] = None

    for m in METHODS:
        lengths = [len(df) for df in logs[m]]
        if not lengths or min(lengths) == 0:
            raise ValueError(
                f"No evaluation points collected for method={m}. "
                "Check eval_every, log_tmax, schedule."
            )
        L = min(lengths)

        mats = []
        for df in logs[m]:
            dft = df.iloc[:L]
            if t_ref is None:
                t_ref = dft["t"].to_numpy(dtype=int)
            mats.append(dft["f1"].to_numpy(dtype=float))

        mat = np.vstack(mats)
        mu = mat.mean(axis=0)
        sd = mat.std(axis=0, ddof=1)
        sd[~np.isfinite(sd)] = 0.0
        ci = 1.96 * sd / math.sqrt(max(1, runs))
        stats[m] = (mu, ci)

    if t_ref is None:  # pragma: no cover - defensive
        raise ValueError("Internal error: missing t_ref")

    plot_methods(
        name=name,
        outdir=outdir,
        schedule=schedule,
        t=t_ref,
        stats=stats,
        runs=runs,
        plot_smooth_w=plot_smooth_w,
        phase_label_y=phase_label_y,
        plot_tmax=plot_tmax,
    )

    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": name,
                "runs": runs,
                "init_fit": init_fit,
                "eval_every": eval_every,
                "eval_window": eval_window,
                "schedule": {
                    "baseline": schedule.baseline,
                    "gaming": schedule.gaming,
                    "fatigue": schedule.fatigue,
                },
                "methods": list(METHODS),
                "AL": {"query_budget": al.query_budget, "query_temp": al.query_temp},
                "PoD": {
                    "coupling_window": pod.coupling_window,
                    "coupling_epsilon": pod.coupling_epsilon,
                    "gate_a": pod.gate_a,
                    "gate_b": pod.gate_b,
                    "gate_lo_frac": pod.gate_lo_frac,
                    "gate_hi_frac": pod.gate_hi_frac,
                    "gate_floor_ms": pod.gate_floor_ms,
                    "gate_ceil_ms": pod.gate_ceil_ms,
                    "gaming_window": pod.gaming_window,
                    "gaming_mu_max_ms": pod.gaming_mu_max_ms,
                    "gaming_cv_max": pod.gaming_cv_max,
                    "fatigue_window": pod.fatigue_window,
                    "fatigue_cv_min": pod.fatigue_cv_min,
                    "persist_k": pod.persist_k,
                },
                "Operator": {
                    "p_correct_baseline": op.p_correct_baseline,
                    "p_correct_gaming": op.p_correct_gaming,
                    "p_correct_fatigue": op.p_correct_fatigue,
                    "k": op.k,
                    "c": op.c,
                    "sigma": op.sigma,
                    "c_fast": op.c_fast,
                    "eps_fast": op.eps_fast,
                    "c_slow": op.c_slow,
                    "sigma_high": op.sigma_high,
                },
                "clf": {
                    "alpha": clf_alpha,
                    "eta0": clf_eta0,
                    "average": clf_average,
                },
                "plot_smooth_w": plot_smooth_w,
                "phase_label_y": phase_label_y,
                "plot_tmax": plot_tmax,
                "log_tmax": log_tmax,
            },
            f,
            indent=2,
        )

