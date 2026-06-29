"""
Base-learner robustness experiment.

Re-runs the *exact* canonical closed loop (pod.experiment.run_suite_generic)
with an alternate incremental base learner, to test whether PoD's advantage is
specific to the published averaged-SGD logistic learner.

Design: the canonical learner factory is replaced at runtime via monkeypatch,
so NOT A SINGLE LINE of the published pipeline changes. With --learner sgd the
run is byte-for-byte the published configuration; with --learner mlp/nb only the
base classifier differs. Everything else (operator simulator, query policy, PoD
checks, seeds, presets, holdout construction) is identical.

Offline datasets only (synth = generated, gas = cached zip); elec2/covertype/
airlines need OpenML and are out of scope for the offline run.

Usage:
    PYTHONPATH=src python3 experiments/run_base_learner_robustness.py \
        --dataset synth --learner mlp --runs 20 --out out_robustness
"""
from __future__ import annotations
import argparse, os, numpy as np

from pod.config import (ALParams, OperatorParams, PoDParams, RegimeSchedule, SynthParams)
from pod.presets import cfg_synth, cfg_gas
from pod.streams import (generate_synth_boundary_pool_regime, load_gas_drift,
                         make_class_weight_dict, split_stream_holdout_by_batch)
import pod.experiment as E
from pod.learner import make_classifier as _make_sgd, proba_from_decision_function as _proba_sgd
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


# --------------------------------------------------------------------------
# Alternate learner factories (must match make_classifier's call signature).
# --------------------------------------------------------------------------
def _make_mlp(classes, seed, *, clf_alpha=2e-5, clf_eta0=0.01, clf_average=True,
              class_weight_dict=None):
    # Nonlinear MLP: a deliberately DIFFERENT model class from linear SGD.
    return MLPClassifier(hidden_layer_sizes=(64,), activation="relu",
                         solver="adam", alpha=float(clf_alpha),
                         learning_rate_init=float(clf_eta0),
                         random_state=seed)

def _make_nb(classes, seed, *, clf_alpha=2e-5, clf_eta0=0.01, clf_average=True,
             class_weight_dict=None):
    # Generative Gaussian naive Bayes: a third, structurally different learner.
    return GaussianNB()

def _proba_any(clf, X, n_classes):
    # MLP/NB expose predict_proba (columns ordered by clf.classes_, which are the
    # sorted labels 0..n-1 for these datasets, matching the SGD path's ordering).
    if hasattr(clf, "predict_proba"):
        p = np.asarray(clf.predict_proba(X), dtype=float)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        return p
    return _proba_sgd(clf, X, n_classes)

FACTORIES = {"sgd": (_make_sgd, _proba_sgd), "mlp": (_make_mlp, _proba_any),
             "nb": (_make_nb, _proba_any)}


def _params_from_cfg(cfg, schedule):
    al = ALParams(query_budget=cfg["query_budget"], query_temp=cfg["query_temp"])
    pod = PoDParams(coupling_window=cfg["coupling_window"], coupling_epsilon=cfg["coupling_epsilon"],
        gate_a=cfg.get("gate_a", 650.0), gate_b=cfg.get("gate_b", 245.0),
        gate_lo_frac=cfg.get("gate_lo_frac", 0.58), gate_hi_frac=cfg.get("gate_hi_frac", 0.86),
        gate_floor_ms=cfg.get("gate_floor_ms", 120.0), gate_ceil_ms=cfg.get("gate_ceil_ms", 3200.0),
        gaming_window=cfg["gaming_window"], gaming_mu_max_ms=cfg["gaming_mu_max_ms"],
        gaming_cv_max=cfg["gaming_cv_max"], fatigue_window=cfg["fatigue_window"],
        fatigue_cv_min=cfg["fatigue_cv_min"], persist_k=cfg["persist_k"])
    op = OperatorParams(k=650., c=270., sigma=88., c_fast=600., eps_fast=cfg["op_eps_fast"],
        c_slow=cfg["op_c_slow"], sigma_high=cfg["op_sigma_high"],
        p_correct_baseline=cfg["op_p_correct_baseline"], p_correct_gaming=cfg["op_p_correct_gaming"],
        p_correct_fatigue=cfg["op_p_correct_fatigue"])
    return al, pod, op


def run_synth(runs, outdir, schedule):
    cfg = cfg_synth(); init = int(cfg["init_fit"]); al, pod, op = _params_from_cfg(cfg, schedule)
    synth = SynthParams()
    X_pool, y_pool, c_star_pool = generate_synth_boundary_pool_regime(
        50000, 20000, synth, schedule, init, rot_baseline=0.004, rot_gaming=0.0, rot_fatigue=0.0)
    holdout = 6000
    def run_data(run_idx):
        rng = np.random.default_rng(10000 + run_idx)
        T = schedule.total(); extra = 20; stream_len = init + T + extra
        max_start = len(X_pool) - (stream_len + holdout)
        start = int(rng.integers(0, max_start + 1))
        Xr = X_pool[start:start+stream_len+holdout]; yr = y_pool[start:start+stream_len+holdout]
        cr = c_star_pool[start:start+stream_len+holdout]
        return (Xr[:stream_len], yr[:stream_len], Xr[stream_len:stream_len+holdout],
                yr[stream_len:stream_len+holdout], cr[:stream_len])
    E.run_suite_generic(name="Synth-Boundary", outdir=outdir, runs=runs, schedule=schedule,
        init_fit=init, eval_every=10, eval_window=600, plot_smooth_w=21, phase_label_y=0.9,
        plot_tmax=6000, log_tmax=0, run_data_fn=run_data, complexity_mode="c_star",
        al=al, pod=pod, op=op, clf_alpha=cfg["clf_alpha"], clf_eta0=cfg["clf_eta0"],
        clf_average=cfg["clf_average"])


def run_gas(runs, outdir, schedule):
    cfg = cfg_gas(); init = int(cfg["init_fit"]); al, pod, op = _params_from_cfg(cfg, schedule)
    X, y, batch_id = load_gas_drift(cache_dir="data_cache_uci224", batches=list(range(1, 11)))
    Xs_all, ys_all, Xh, yh = split_stream_holdout_by_batch(X, y, batch_id, holdout_batches=[9, 10])
    classes = np.unique(np.concatenate([ys_all, yh]))
    cw = make_class_weight_dict(classes=classes, y_sample=ys_all[:init])
    def run_data(run_idx):
        rng = np.random.default_rng(10000 + run_idx)
        perm = rng.permutation(len(Xs_all))
        return Xs_all[perm], ys_all[perm], Xh, yh, None
    E.run_suite_generic(name="uci224_gas_drift", outdir=outdir, runs=runs, schedule=schedule,
        init_fit=init, eval_every=10, eval_window=600, plot_smooth_w=21, phase_label_y=0.9,
        plot_tmax=6000, log_tmax=0, run_data_fn=run_data, complexity_mode="entropy",
        class_weight_dict=cw, al=al, pod=pod, op=op, clf_alpha=cfg["clf_alpha"],
        clf_eta0=cfg["clf_eta0"], clf_average=cfg["clf_average"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synth", "gas"], required=True)
    ap.add_argument("--learner", choices=["sgd", "mlp", "nb"], required=True)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--baseline", type=int, default=2000)
    ap.add_argument("--gaming", type=int, default=2000)
    ap.add_argument("--fatigue", type=int, default=2000)
    ap.add_argument("--out", type=str, default="out_robustness")
    a = ap.parse_args()
    # Inject the chosen learner into the canonical pipeline.
    mk, pb = FACTORIES[a.learner]
    E.make_classifier = mk
    E.proba_from_decision_function = pb
    sch = RegimeSchedule(a.baseline, a.gaming, a.fatigue)
    outdir = os.path.join(a.out, f"{a.dataset}_{a.learner}")
    print(f"[robustness] dataset={a.dataset} learner={a.learner} runs={a.runs} "
          f"schedule={a.baseline}/{a.gaming}/{a.fatigue} -> {outdir}", flush=True)
    (run_synth if a.dataset == "synth" else run_gas)(a.runs, outdir, sch)
    print("[robustness] DONE", flush=True)


if __name__ == "__main__":
    main()
