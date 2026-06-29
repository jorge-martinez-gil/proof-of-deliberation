"""
Frozen per-dataset hyperparameter presets.

These are the *exact* values reported in the paper (Table II and the
caption of each F1 trajectory figure). Any modification to a preset
constitutes a deviation from the published configuration and must be
declared in the ``config.json`` written next to the figures.

Each preset is a plain dict so it can be (i) JSON-serialised, (ii)
compared with ``configs/*.json`` for verification, and (iii) overridden
piece-wise on the command line without altering the source.
"""

from __future__ import annotations

from typing import Any, Dict


def cfg_elec2() -> Dict[str, Any]:
    """Electricity (elec2) preset -- 20 runs, OpenML id 44156."""
    return dict(
        runs=20,
        init_fit=250,
        query_budget=0.85,
        query_temp=1.00,
        coupling_window=80,
        coupling_epsilon=-1.0,
        gaming_window=6,
        gaming_mu_max_ms=900.0,
        gaming_cv_max=0.025,
        fatigue_window=20,
        fatigue_cv_min=0.40,
        persist_k=2,
        gate_b=220.0,
        gate_lo_frac=0.90,
        gate_hi_frac=1.10,
        op_eps_fast=3.0,
        op_p_correct_baseline=0.985,
        op_p_correct_gaming=0.01,
        op_p_correct_fatigue=0.15,
        op_c_slow=1100.0,
        op_sigma_high=900.0,
        clf_average=True,
        clf_eta0=0.01,
        clf_alpha=2e-5,
    )


def cfg_gas() -> Dict[str, Any]:
    """UCI 224 Gas Drift preset -- 20 runs, batches 1-8 stream / 9-10 holdout."""
    return dict(
        runs=20,
        init_fit=200,
        query_budget=0.60,
        query_temp=0.16,
        coupling_window=40,
        coupling_epsilon=-1.0,
        gaming_window=8,
        gaming_mu_max_ms=720.0,
        gaming_cv_max=0.02,
        fatigue_window=15,
        fatigue_cv_min=0.40,
        persist_k=2,
        gate_b=250.0,
        gate_lo_frac=0.75,
        gate_hi_frac=0.95,
        op_eps_fast=3.0,
        op_p_correct_baseline=0.985,
        op_p_correct_gaming=0.02,
        op_p_correct_fatigue=0.15,
        op_c_slow=1100.0,
        op_sigma_high=900.0,
        clf_average=True,
        clf_eta0=0.01,
        clf_alpha=2e-5,
    )


def cfg_synth() -> Dict[str, Any]:
    """Synth-Boundary preset -- 20 runs, d=10, lambda=3.0."""
    return dict(
        runs=20,
        init_fit=300,
        query_budget=0.45,
        query_temp=0.90,
        coupling_window=80,
        coupling_epsilon=-1.0,
        gaming_window=50,
        gaming_mu_max_ms=850.0,
        gaming_cv_max=0.025,
        fatigue_window=60,
        fatigue_cv_min=0.55,
        persist_k=3,
        gate_a=0.0,
        gate_b=600.0,
        gate_lo_frac=0.2,
        gate_hi_frac=0.2,
        gate_floor_ms=460.0,
        gate_ceil_ms=700.0,
        op_p_correct_baseline=0.7,
        op_p_correct_gaming=0.07,
        op_eps_fast=1.5,
        op_p_correct_fatigue=0.12,
        op_c_slow=1100.0,
        op_sigma_high=2200.0,
        clf_average=True,
        clf_eta0=0.03,
        clf_alpha=2e-6,
    )


def cfg_covertype() -> Dict[str, Any]:
    """Forest CoverType preset -- 20 runs, OpenML id 1596 (multiclass).

    CoverType is a 7-class forest cover-type benchmark with ~581K rows
    and 54 features (10 quantitative + 44 binary). It is widely used in
    the streaming-ML literature as a real-world drifting benchmark and
    broadens evaluation beyond
    Electricity. Class-weight balancing is enabled because the original
    distribution is highly skewed (Spruce/Fir dominate).
    """
    return dict(
        runs=20,
        init_fit=300,
        query_budget=0.70,
        query_temp=0.85,
        coupling_window=80,
        coupling_epsilon=-1.0,
        gaming_window=8,
        gaming_mu_max_ms=900.0,
        gaming_cv_max=0.025,
        fatigue_window=20,
        fatigue_cv_min=0.40,
        persist_k=2,
        gate_b=230.0,
        gate_lo_frac=0.85,
        gate_hi_frac=1.10,
        op_eps_fast=3.0,
        op_p_correct_baseline=0.985,
        op_p_correct_gaming=0.01,
        op_p_correct_fatigue=0.18,
        op_c_slow=1100.0,
        op_sigma_high=900.0,
        clf_average=True,
        clf_eta0=0.01,
        clf_alpha=2e-5,
    )


def cfg_airlines() -> Dict[str, Any]:
    """Airlines preset -- 20 runs, OpenML id 1169 (binary).

    The Airlines dataset (~540K rows, 7 features) predicts on-time
    departure. It is a classical real-world streaming benchmark used in
    MOA / scikit-multiflow and exhibits gradual concept drift driven by
    seasonal and day-of-week effects.
    """
    return dict(
        runs=20,
        init_fit=250,
        query_budget=0.80,
        query_temp=1.00,
        coupling_window=80,
        coupling_epsilon=-1.0,
        gaming_window=6,
        gaming_mu_max_ms=900.0,
        gaming_cv_max=0.025,
        fatigue_window=20,
        fatigue_cv_min=0.40,
        persist_k=2,
        gate_b=220.0,
        gate_lo_frac=0.90,
        gate_hi_frac=1.10,
        op_eps_fast=3.0,
        op_p_correct_baseline=0.985,
        op_p_correct_gaming=0.01,
        op_p_correct_fatigue=0.15,
        op_c_slow=1100.0,
        op_sigma_high=900.0,
        clf_average=True,
        clf_eta0=0.01,
        clf_alpha=2e-5,
    )


def cfg_poker() -> Dict[str, Any]:
    """Poker-Hand preset -- 20 runs, OpenML id 1595 (10-class, local files).

    Poker-Hand is a 10-class benchmark with ~1.025M rows and 10 features
    (suit/rank of five cards). It is a standard *virtual-drift* stream in
    the streaming-ML literature (stationary class-conditionals, shifting
    class priors) and adds a sixth dataset in a new game/combinatorial
    domain. Raising N from 5 to 6 lets the cross-dataset Demsar layer reach
    a corrected two-sided significance (min Wilcoxon p drops 0.0625 ->
    0.03125) instead of remaining purely descriptive.

    To avoid any appearance of per-dataset tuning, this preset *inherits
    the CoverType configuration verbatim* -- the other real multiclass,
    class-imbalanced, entropy-complexity stream -- changing nothing about
    the gate, coupling, vigilance, or operator regimes. (Unlike CoverType,
    the poker runner disables the explicit class-weight dict and remaps each
    run's labels to contiguous codes; see pod.cli.experiments._run_poker.)
    """
    cfg = dict(cfg_covertype())
    return cfg


PRESETS: Dict[str, Any] = {
    "elec2": cfg_elec2,
    "gas": cfg_gas,
    "synth": cfg_synth,
    "covertype": cfg_covertype,
    "airlines": cfg_airlines,
    "poker": cfg_poker,
}
