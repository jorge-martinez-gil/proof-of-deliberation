"""
``pod-experiments`` -- the canonical experiment runner.

This is the user-facing entry point that reproduces the streaming
experiments reported in Section 6 of the paper. It dispatches to the
three stream loaders (Synth-Boundary, OpenML Electricity, UCI 224 Gas
Drift) and writes the figures, per-run CSV logs, diagnostics and the
frozen-config snapshot under ``--out/<dataset>/``.

The exact command used to reproduce the published numbers is::

    pod-experiments --datasets synth,elec2,gas

See ``docs/REPRODUCIBILITY.md`` for the full reproduction recipe.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np

from pod.config import (
    ALParams,
    OperatorParams,
    PoDParams,
    RegimeSchedule,
    SynthParams,
)
from pod.experiment import make_external_holdout, run_suite_generic
from pod.presets import (
    cfg_airlines,
    cfg_covertype,
    cfg_elec2,
    cfg_gas,
    cfg_poker,
    cfg_synth,
)
from pod.streams import (
    OpenMLSpec,
    generate_synth_boundary_pool_regime,
    load_gas_drift,
    load_openml_any,
    load_poker_local,
    make_class_weight_dict,
    split_stream_holdout_by_batch,
)
from pod.utils import ensure_dir, parse_int_list
from pod.viz import set_pub_style


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the ``pod-experiments`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="pod-experiments",
        description=(
            "Reproduce the streaming Proof-of-Deliberation experiments "
            "from Martinez-Gil, IEEE TKDE 2026."
        ),
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default="synth,elec2,gas",
        help="Comma-separated subset of {synth, elec2, gas, covertype, airlines}.",
    )
    parser.add_argument("--out", type=str, default="out_pod_unified")
    parser.add_argument("--runs", type=int, default=2)

    parser.add_argument("--baseline", type=int, default=2000)
    parser.add_argument("--gaming", type=int, default=2000)
    parser.add_argument("--fatigue", type=int, default=2000)
    parser.add_argument("--init_fit", type=int, default=20)

    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_window", type=int, default=600)

    parser.add_argument("--plot_smooth_w", type=int, default=21)
    parser.add_argument("--phase_label_y", type=float, default=0.9)
    parser.add_argument("--plot_tmax", type=int, default=6000)
    parser.add_argument("--log_tmax", type=int, default=0)

    parser.add_argument("--query_budget", type=float, default=0.50)
    parser.add_argument("--query_temp", type=float, default=0.18)

    parser.add_argument("--coupling_window", type=int, default=20)
    parser.add_argument("--coupling_epsilon", type=float, default=0.15)
    parser.add_argument("--gaming_window", type=int, default=20)
    parser.add_argument("--gaming_mu_max_ms", type=float, default=650.0)
    parser.add_argument("--gaming_cv_max", type=float, default=0.06)
    parser.add_argument("--fatigue_window", type=int, default=20)
    parser.add_argument("--fatigue_cv_min", type=float, default=0.27)
    parser.add_argument("--persist_k", type=int, default=3)

    parser.add_argument("--gate_a", type=float, default=650.0)
    parser.add_argument("--gate_b", type=float, default=245.0)
    parser.add_argument("--gate_lo_frac", type=float, default=0.58)
    parser.add_argument("--gate_hi_frac", type=float, default=0.86)
    parser.add_argument("--gate_floor_ms", type=float, default=120.0)
    parser.add_argument("--gate_ceil_ms", type=float, default=3200.0)

    parser.add_argument("--op_k", type=float, default=650.0)
    parser.add_argument("--op_c", type=float, default=270.0)
    parser.add_argument("--op_sigma", type=float, default=88.0)
    parser.add_argument("--op_c_fast", type=float, default=600.0)
    parser.add_argument("--op_eps_fast", type=float, default=25.0)
    parser.add_argument("--op_c_slow", type=float, default=925.0)
    parser.add_argument("--op_sigma_high", type=float, default=675.0)
    parser.add_argument("--op_p_correct_baseline", type=float, default=0.965)
    parser.add_argument("--op_p_correct_gaming", type=float, default=0.22)
    parser.add_argument("--op_p_correct_fatigue", type=float, default=0.45)

    parser.add_argument("--openml_id", type=int, default=44156)
    parser.add_argument("--covertype_openml_id", type=int, default=1596)
    parser.add_argument("--airlines_openml_id", type=int, default=1169)
    parser.add_argument(
        "--poker_dir",
        type=str,
        default="poker-hand",
        help="Directory with local UCI Poker-Hand .data files (no network).",
    )
    parser.add_argument("--holdout", type=int, default=6000)

    parser.add_argument("--cache_dir", type=str, default="data_cache_uci224")
    parser.add_argument("--stream_batches", type=str, default="1,2,3,4,5,6,7,8")
    parser.add_argument("--holdout_batches", type=str, default="9,10")

    parser.add_argument("--synth_pool", type=int, default=20000)
    parser.add_argument("--synth_seed", type=int, default=50000)

    return parser


# ---------------------------------------------------------------------------
# Per-dataset parameter construction
# ---------------------------------------------------------------------------
def _build_params(
    args: argparse.Namespace, cfg: Dict[str, Any]
) -> Tuple[RegimeSchedule, int, int, ALParams, PoDParams, OperatorParams, Dict[str, float]]:
    schedule_local = RegimeSchedule(
        int(args.baseline), int(args.gaming), int(args.fatigue)
    )
    init_fit_local = int(cfg["init_fit"])
    runs_local = int(cfg["runs"])

    al_local = ALParams(
        query_budget=float(cfg["query_budget"]),
        query_temp=float(cfg["query_temp"]),
    )

    pod_local = PoDParams(
        coupling_window=int(cfg["coupling_window"]),
        coupling_epsilon=float(cfg["coupling_epsilon"]),
        gate_a=float(cfg.get("gate_a", args.gate_a)),
        gate_b=float(cfg.get("gate_b", args.gate_b)),
        gate_lo_frac=float(cfg.get("gate_lo_frac", args.gate_lo_frac)),
        gate_hi_frac=float(cfg.get("gate_hi_frac", args.gate_hi_frac)),
        gate_floor_ms=float(cfg.get("gate_floor_ms", args.gate_floor_ms)),
        gate_ceil_ms=float(cfg.get("gate_ceil_ms", args.gate_ceil_ms)),
        gaming_window=int(cfg["gaming_window"]),
        gaming_mu_max_ms=float(cfg["gaming_mu_max_ms"]),
        gaming_cv_max=float(cfg["gaming_cv_max"]),
        fatigue_window=int(cfg["fatigue_window"]),
        fatigue_cv_min=float(cfg["fatigue_cv_min"]),
        persist_k=int(cfg["persist_k"]),
    )

    op_local = OperatorParams(
        k=float(args.op_k),
        c=float(args.op_c),
        sigma=float(args.op_sigma),
        c_fast=float(args.op_c_fast),
        eps_fast=float(cfg["op_eps_fast"]),
        c_slow=float(cfg["op_c_slow"]),
        sigma_high=float(cfg["op_sigma_high"]),
        p_correct_baseline=float(cfg["op_p_correct_baseline"]),
        p_correct_gaming=float(cfg["op_p_correct_gaming"]),
        p_correct_fatigue=float(cfg["op_p_correct_fatigue"]),
    )

    clf_knobs = dict(
        clf_average=bool(cfg.get("clf_average", True)),
        clf_eta0=float(cfg.get("clf_eta0", 0.01)),
        clf_alpha=float(cfg.get("clf_alpha", 2e-5)),
    )

    return schedule_local, init_fit_local, runs_local, al_local, pod_local, op_local, clf_knobs


# ---------------------------------------------------------------------------
# Dataset dispatchers
# ---------------------------------------------------------------------------
def _run_openml_generic(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    name: str,
    openml_id: int,
    outdir_name: str,
    use_class_weight: bool,
) -> None:
    """Shared scaffolding for any OpenML-backed stream (elec2, covertype, airlines)."""
    schedule_e, init_fit_e, runs_e, al_e, pod_e, op_e, clf_e = _build_params(args, cfg)

    print(f"[{name}] runs={runs_e} init_fit={init_fit_e}")

    spec = OpenMLSpec(name, int(openml_id))
    print(f"Loading OpenML {spec.name} (id={spec.data_id})")
    X_all, y_all = load_openml_any(spec, seed=123)

    classes_all = np.unique(y_all)
    class_weight_dict = None
    if use_class_weight:
        class_weight_dict = make_class_weight_dict(
            classes=classes_all, y_sample=y_all[:init_fit_e]
        )

    def run_data_openml(run_idx: int):
        rng = np.random.default_rng(10000 + run_idx)
        perm = rng.permutation(len(X_all))
        Xr, yr = X_all[perm], y_all[perm]
        T = schedule_e.total()

        extra = 20
        need = init_fit_e + T + extra + 100
        if len(Xr) < need:
            raise ValueError(f"Dataset too small: need {need}, got {len(Xr)}")

        stream_len = init_fit_e + T + extra
        Xs = Xr[:stream_len]
        ys = yr[:stream_len]

        holdout_size = min(int(args.holdout), max(2000, len(Xr) - stream_len))
        Xh, yh = make_external_holdout(
            Xr, yr, stream_len=stream_len, holdout_size=holdout_size, rng=rng
        )
        return Xs, ys, Xh, yh, None

    out_full = os.path.join(args.out, outdir_name)
    run_suite_generic(
        name=outdir_name,
        outdir=out_full,
        runs=runs_e,
        schedule=schedule_e,
        init_fit=init_fit_e,
        eval_every=int(args.eval_every),
        eval_window=int(args.eval_window),
        plot_smooth_w=int(args.plot_smooth_w),
        phase_label_y=float(args.phase_label_y),
        plot_tmax=int(args.plot_tmax),
        log_tmax=int(args.log_tmax),
        run_data_fn=run_data_openml,
        complexity_mode="entropy",
        class_weight_dict=class_weight_dict,
        al=al_e,
        pod=pod_e,
        op=op_e,
        clf_alpha=clf_e["clf_alpha"],
        clf_eta0=clf_e["clf_eta0"],
        clf_average=clf_e["clf_average"],
    )


def _run_elec2(args: argparse.Namespace) -> None:
    _run_openml_generic(
        args,
        cfg=cfg_elec2(),
        name="elec2",
        openml_id=int(args.openml_id),
        outdir_name="elec2",
        use_class_weight=False,
    )


def _run_covertype(args: argparse.Namespace) -> None:
    _run_openml_generic(
        args,
        cfg=cfg_covertype(),
        name="covertype",
        openml_id=int(args.covertype_openml_id),
        outdir_name="covertype",
        use_class_weight=True,
    )


def _run_airlines(args: argparse.Namespace) -> None:
    _run_openml_generic(
        args,
        cfg=cfg_airlines(),
        name="airlines",
        openml_id=int(args.airlines_openml_id),
        outdir_name="airlines",
        use_class_weight=False,
    )


def _remap_contiguous_labels(
    ys: np.ndarray, yh: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Relabel a run's stream+holdout to contiguous codes ``0..K-1``.

    Poker-Hand's ultra-rare classes (8 = straight flush, 9 = royal flush;
    only 25 of ~1.025M rows) are absent from most per-run slices, so a
    run's label set can be non-contiguous (e.g. ``{0..7, 9}``). The
    operator simulator draws corrupted labels in ``[0, n_classes)`` assuming
    contiguity, which would otherwise emit a label the classifier has never
    seen. Macro-F1 is invariant to this relabelling.
    """
    classes_run = np.unique(np.concatenate([ys, yh]))
    lut = {int(v): i for i, v in enumerate(classes_run)}
    ys2 = np.array([lut[int(v)] for v in ys], dtype=int)
    yh2 = np.array([lut[int(v)] for v in yh], dtype=int)
    return ys2, yh2


def _run_poker(args: argparse.Namespace) -> None:
    """Poker-Hand stream from local UCI files (id 1595, 10-class).

    Mirrors ``_run_openml_generic`` -- same per-run permutation, holdout
    construction, complexity mode and learner -- but loads the pool from
    local ``.data`` files so it needs no network. Two poker-specific points:

    * Per-run labels are remapped to contiguous codes (see
      :func:`_remap_contiguous_labels`) because classes 8/9 are too rare to
      appear in every slice.
    * Class weighting is disabled (``class_weight_dict=None``): it is a
      transformation shared by all eleven methods, so it cancels in the
      paired PoD-vs-competitor comparison, and a fixed global dict cannot be
      reconciled with the per-run (variable) class set anyway.
    """
    cfg = cfg_poker()
    schedule_p, init_fit_p, runs_p, al_p, pod_p, op_p, clf_p = _build_params(args, cfg)

    print(f"[poker] runs={runs_p} init_fit={init_fit_p}")
    print(f"Loading local Poker-Hand from {args.poker_dir!r}")
    X_all, y_all = load_poker_local(args.poker_dir, seed=123)

    def run_data_poker(run_idx: int):
        rng = np.random.default_rng(10000 + run_idx)
        perm = rng.permutation(len(X_all))
        Xr, yr = X_all[perm], y_all[perm]
        T = schedule_p.total()

        extra = 20
        need = init_fit_p + T + extra + 100
        if len(Xr) < need:
            raise ValueError(f"Dataset too small: need {need}, got {len(Xr)}")

        stream_len = init_fit_p + T + extra
        Xs = Xr[:stream_len]
        ys = yr[:stream_len]

        holdout_size = min(int(args.holdout), max(2000, len(Xr) - stream_len))
        Xh, yh = make_external_holdout(
            Xr, yr, stream_len=stream_len, holdout_size=holdout_size, rng=rng
        )
        ys, yh = _remap_contiguous_labels(ys, yh)
        return Xs, ys, Xh, yh, None

    out_full = os.path.join(args.out, "poker")
    run_suite_generic(
        name="poker",
        outdir=out_full,
        runs=runs_p,
        schedule=schedule_p,
        init_fit=init_fit_p,
        eval_every=int(args.eval_every),
        eval_window=int(args.eval_window),
        plot_smooth_w=int(args.plot_smooth_w),
        phase_label_y=float(args.phase_label_y),
        plot_tmax=int(args.plot_tmax),
        log_tmax=int(args.log_tmax),
        run_data_fn=run_data_poker,
        complexity_mode="entropy",
        class_weight_dict=None,
        al=al_p,
        pod=pod_p,
        op=op_p,
        clf_alpha=clf_p["clf_alpha"],
        clf_eta0=clf_p["clf_eta0"],
        clf_average=clf_p["clf_average"],
    )


def _run_gas(args: argparse.Namespace) -> None:
    cfg = cfg_gas()
    schedule_g, init_fit_g, runs_g, al_g, pod_g, op_g, clf_g = _build_params(args, cfg)

    print(f"[gas] runs={runs_g} init_fit={init_fit_g}")

    stream_batches = parse_int_list(args.stream_batches)
    holdout_batches = parse_int_list(args.holdout_batches)
    all_batches = sorted(set(stream_batches + holdout_batches))
    if any(b < 1 or b > 10 for b in all_batches):
        raise ValueError("Batches must be in 1..10")

    print("Downloading and loading UCI 224 Gas Sensor Array Drift.")
    X, y, batch_id = load_gas_drift(cache_dir=args.cache_dir, batches=all_batches)
    Xs_all, ys_all, Xh, yh = split_stream_holdout_by_batch(
        X, y, batch_id, holdout_batches=holdout_batches
    )

    need_stream = init_fit_g + schedule_g.total() + 50
    if len(Xs_all) < need_stream:
        raise ValueError(
            f"Stream too small: need at least {need_stream}, got {len(Xs_all)}. "
            "Use more stream batches or reduce schedule."
        )

    classes = np.unique(np.concatenate([ys_all, yh]))
    cw = make_class_weight_dict(classes=classes, y_sample=ys_all[:init_fit_g])

    def run_data_gas(run_idx: int):
        rng = np.random.default_rng(10000 + run_idx)
        perm = rng.permutation(len(Xs_all))
        Xs, ys = Xs_all[perm], ys_all[perm]
        return Xs, ys, Xh, yh, None

    outdir = os.path.join(args.out, "uci224_gas_drift")
    run_suite_generic(
        name="uci224_gas_drift",
        outdir=outdir,
        runs=runs_g,
        schedule=schedule_g,
        init_fit=init_fit_g,
        eval_every=int(args.eval_every),
        eval_window=int(args.eval_window),
        plot_smooth_w=int(args.plot_smooth_w),
        phase_label_y=float(args.phase_label_y),
        plot_tmax=int(args.plot_tmax),
        log_tmax=int(args.log_tmax),
        run_data_fn=run_data_gas,
        complexity_mode="entropy",
        class_weight_dict=cw,
        al=al_g,
        pod=pod_g,
        op=op_g,
        clf_alpha=clf_g["clf_alpha"],
        clf_eta0=clf_g["clf_eta0"],
        clf_average=clf_g["clf_average"],
    )


def _run_synth(args: argparse.Namespace) -> None:
    cfg = cfg_synth()
    schedule_s, init_fit_s, runs_s, al_s, pod_s, op_s, clf_s = _build_params(args, cfg)

    print(f"[synth] runs={runs_s} init_fit={init_fit_s}")

    synth = SynthParams()
    pool_n = int(args.synth_pool)
    print(f"Generating Synth-Boundary pool: n={pool_n}")
    X_pool, y_pool, c_star_pool = generate_synth_boundary_pool_regime(
        int(args.synth_seed),
        pool_n,
        synth,
        schedule_s,
        init_fit_s,
        rot_baseline=0.004,
        rot_gaming=0.0,
        rot_fatigue=0.0,
    )

    def run_data_synth(run_idx: int):
        rng = np.random.default_rng(10000 + run_idx)

        T = schedule_s.total()
        extra = 20
        stream_len = init_fit_s + T + extra
        holdout_size = int(args.holdout)

        need = stream_len + holdout_size + 100
        if len(X_pool) < need:
            raise ValueError(
                f"Synth pool too small: need at least {need}, got {len(X_pool)}. "
                "Increase --synth_pool."
            )

        max_start = len(X_pool) - (stream_len + holdout_size)
        start = int(rng.integers(0, max_start + 1))

        Xr = X_pool[start : start + stream_len + holdout_size]
        yr = y_pool[start : start + stream_len + holdout_size]
        cr = c_star_pool[start : start + stream_len + holdout_size]

        Xs = Xr[:stream_len]
        ys = yr[:stream_len]
        crs = cr[:stream_len]

        Xh = Xr[stream_len : stream_len + holdout_size]
        yh = yr[stream_len : stream_len + holdout_size]

        return Xs, ys, Xh, yh, crs

    outdir = os.path.join(args.out, "synth_boundary")
    run_suite_generic(
        name="Synth-Boundary",
        outdir=outdir,
        runs=runs_s,
        schedule=schedule_s,
        init_fit=init_fit_s,
        eval_every=int(args.eval_every),
        eval_window=int(args.eval_window),
        plot_smooth_w=int(args.plot_smooth_w),
        phase_label_y=float(args.phase_label_y),
        plot_tmax=int(args.plot_tmax),
        log_tmax=int(args.log_tmax),
        run_data_fn=run_data_synth,
        complexity_mode="c_star",
        class_weight_dict=None,
        al=al_s,
        pod=pod_s,
        op=op_s,
        clf_alpha=clf_s["clf_alpha"],
        clf_eta0=clf_s["clf_eta0"],
        clf_average=clf_s["clf_average"],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    set_pub_style()
    ensure_dir(args.out)

    wanted = {x.strip().lower() for x in args.datasets.split(",") if x.strip()}

    if "synth" in wanted:
        _run_synth(args)
    if "elec2" in wanted:
        _run_elec2(args)
    if "gas" in wanted:
        _run_gas(args)
    if "covertype" in wanted:
        _run_covertype(args)
    if "airlines" in wanted:
        _run_airlines(args)
    if "poker" in wanted:
        _run_poker(args)

    print("Done.")
    print(f"Outputs under: {args.out}/<dataset>/figs/*_methods_f1.pdf")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
