"""
``pod-realdata`` -- offline analysis pipeline for collected sessions.

Loads every participant CSV under ``--in``, replays each session
through the PoD verification layer, and writes the per-participant,
per-block, and aggregate tables plus the figures used in the paper's
real-data section.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pod.config import PoDParams
from pod.realdata.analysis import (
    apply_pod_offline,
    block_regime_table,
    coupling_table,
    regime_classification_score,
    summarise_participants,
)
from pod.realdata.loader import CONDITION_ORDER, load_sessions
from pod.utils import ensure_dir
from pod.viz import set_pub_style


def _figure_block_accept(table: pd.DataFrame, out_base: str) -> None:
    """Bar chart of per-block PoD accept rate; the headline real-data figure."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.20, top=0.88)

    conds = list(CONDITION_ORDER)
    rates = [
        float(table.loc[table["induced_condition"] == c, "pod_accept_rate"].iloc[0])
        if (table["induced_condition"] == c).any() else 0.0
        for c in conds
    ]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    ax.bar(conds, rates, color=colors, edgecolor="black", linewidth=0.7)
    for c, r in zip(conds, rates):
        ax.text(c, r + 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("PoD accept rate")
    ax.set_title("Real-user data: PoD accept rate by induced regime")
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png")
    plt.close(fig)


def _figure_coupling(table: pd.DataFrame, out_base: str) -> None:
    """Per-condition box-plot of the Spearman rho coupling estimate."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.20, top=0.88)

    conds = list(CONDITION_ORDER)
    data = [
        table.loc[table["induced_condition"] == c, "spearman_rho"].dropna().to_numpy()
        for c in conds
    ]
    if any(len(d) > 0 for d in data):
        # Matplotlib >=3.9 renamed `labels` -> `tick_labels`; fall back
        # for older installs.
        try:
            bp = ax.boxplot(
                data, tick_labels=conds, patch_artist=True, widths=0.55,
                medianprops=dict(color="black", linewidth=1.4),
            )
        except TypeError:  # pragma: no cover - older matplotlib
            bp = ax.boxplot(
                data, labels=conds, patch_artist=True, widths=0.55,
                medianprops=dict(color="black", linewidth=1.4),
            )
        for patch, col in zip(bp["boxes"], ["#2ca02c", "#ff7f0e", "#d62728"]):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Spearman rho(c_star, delib_ms)")
    ax.set_title("Hick-Hyman coupling by induced regime (per-participant)")
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".png")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pod-realdata",
        description=(
            "Run the offline PoD analysis on real-user session CSVs "
            "downloaded from the labeling app at PoD_code/labeling/app.html. "
            "Produces the tables and figures reported in the paper's "
            "real-data section."
        ),
    )
    parser.add_argument(
        "--in", dest="in_dir", type=str, required=True,
        help="Directory holding participant CSVs (or a single CSV file).",
    )
    parser.add_argument(
        "--out", dest="out_dir", type=str, default="out_real",
        help="Output directory (default: out_real).",
    )
    parser.add_argument(
        "--min-trials-per-block", type=int, default=10,
        help="Drop participant x block sub-sessions with fewer trials.",
    )
    parser.add_argument(
        "--gate-floor-ms", type=float, default=120.0,
        help="Override PoDParams.gate_floor_ms for offline replay.",
    )
    parser.add_argument(
        "--gate-ceil-ms", type=float, default=8000.0,
        help="Override PoDParams.gate_ceil_ms for offline replay.",
    )
    parser.add_argument(
        "--coupling-window", type=int, default=20,
        help="Override PoDParams.coupling_window for offline replay.",
    )
    parser.add_argument(
        "--coupling-epsilon", type=float, default=0.0,
        help="Override PoDParams.coupling_epsilon for offline replay.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    set_pub_style()
    ensure_dir(args.out_dir)

    raw = load_sessions(args.in_dir)

    # Drop too-short participant x block segments.
    keep_mask = (
        raw.groupby(["participant_id", "block"])["trial_idx"].transform("count")
        >= int(args.min_trials_per_block)
    )
    raw = raw.loc[keep_mask].reset_index(drop=True)
    if raw.empty:
        raise ValueError(
            "No participant x block segments meet --min-trials-per-block. "
            "Have you collected any data yet?"
        )

    # Per-participant summary.
    per_p = summarise_participants(raw)
    per_p.to_csv(os.path.join(args.out_dir, "per_participant_block_summary.csv"), index=False)

    # Spearman coupling per (participant, block).
    coup = coupling_table(raw)
    coup.to_csv(os.path.join(args.out_dir, "coupling_per_participant.csv"), index=False)

    # PoD offline replay.
    pod = PoDParams(
        gate_floor_ms=float(args.gate_floor_ms),
        gate_ceil_ms=float(args.gate_ceil_ms),
        coupling_window=int(args.coupling_window),
        coupling_epsilon=float(args.coupling_epsilon),
    )
    augmented = apply_pod_offline(raw, pod=pod)
    augmented.to_csv(os.path.join(args.out_dir, "per_trial_with_pod.csv"), index=False)

    block_tbl = block_regime_table(augmented)
    block_tbl.to_csv(os.path.join(args.out_dir, "block_regime_table.csv"), index=False)

    reg_score = regime_classification_score(augmented)
    with open(os.path.join(args.out_dir, "regime_classification_score.json"), "w", encoding="utf-8") as f:
        json.dump(reg_score, f, indent=2)

    _figure_block_accept(block_tbl, os.path.join(args.out_dir, "fig_block_accept_rate"))
    _figure_coupling(coup, os.path.join(args.out_dir, "fig_coupling_by_regime"))

    # Aggregate summary JSON for the paper.
    summary = {
        "n_participants": int(raw["participant_id"].nunique()),
        "n_trials_total": int(len(raw)),
        "n_trials_per_condition": {
            c: int((raw["induced_condition"] == c).sum())
            for c in CONDITION_ORDER
        },
        "mean_accuracy_per_condition": {
            c: float(raw.loc[raw["induced_condition"] == c, "correct"].mean())
            if (raw["induced_condition"] == c).any() else float("nan")
            for c in CONDITION_ORDER
        },
        "mean_delib_ms_per_condition": {
            c: float(raw.loc[raw["induced_condition"] == c, "delib_ms"].mean())
            if (raw["induced_condition"] == c).any() else float("nan")
            for c in CONDITION_ORDER
        },
        "pod_accept_rate_per_condition": {
            c: float(block_tbl.loc[block_tbl["induced_condition"] == c, "pod_accept_rate"].iloc[0])
            if (block_tbl["induced_condition"] == c).any() else float("nan")
            for c in CONDITION_ORDER
        },
        "regime_classification_score": reg_score,
        "pod_params": {
            "gate_floor_ms": pod.gate_floor_ms,
            "gate_ceil_ms": pod.gate_ceil_ms,
            "gate_a": pod.gate_a,
            "gate_b": pod.gate_b,
            "gate_lo_frac": pod.gate_lo_frac,
            "gate_hi_frac": pod.gate_hi_frac,
            "coupling_window": pod.coupling_window,
            "coupling_epsilon": pod.coupling_epsilon,
            "gaming_window": pod.gaming_window,
            "gaming_mu_max_ms": pod.gaming_mu_max_ms,
            "gaming_cv_max": pod.gaming_cv_max,
            "fatigue_window": pod.fatigue_window,
            "fatigue_cv_min": pod.fatigue_cv_min,
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[pod-realdata] Wrote analysis to: {args.out_dir}\n"
        f"[pod-realdata] Participants: {summary['n_participants']}, "
        f"trials: {summary['n_trials_total']}\n"
        f"[pod-realdata] PoD accept rate baseline / gaming / fatigue = "
        f"{summary['pod_accept_rate_per_condition'].get('baseline', float('nan')):.2f} / "
        f"{summary['pod_accept_rate_per_condition'].get('gaming', float('nan')):.2f} / "
        f"{summary['pod_accept_rate_per_condition'].get('fatigue', float('nan')):.2f}\n"
        f"[pod-realdata] Regime classification mean score = "
        f"{reg_score.get('mean_score', float('nan')):.3f} "
        f"(higher = PoD favours baseline over induced bad regimes)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
