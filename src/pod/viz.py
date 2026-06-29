"""
Publication-grade plotting helpers.

The defaults match those used to produce the figures in the paper
(900 DPI PDF/PNG, embeddable TrueType fonts, sans-serif body, no top
or right spines, soft grid). Method colours are fixed across all
dataset figures so that AL is consistently blue, StaticGating orange,
AdaptiveGating green, and PoD red.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pod.config import RegimeSchedule
from pod.utils import smooth_series

METHOD_COLORS: Dict[str, str] = {
    "AL": "#1f77b4",
    "StaticGating": "#ff7f0e",
    "AdaptiveGating": "#2ca02c",
    "WorkerQuality": "#9467bd",
    "Raykar": "#17becf",
    "MACE": "#bcbd22",
    "IEThresh": "#ff9da6",
    "PoD": "#d62728",
    "PoD-NoGate": "#8c564b",
    "PoD-NoCoupling": "#e377c2",
    "PoD-NoVigilance": "#7f7f7f",
}
"""Canonical method colours used in every figure of the paper.

Layout:

* AL, StaticGating, AdaptiveGating, PoD -- the four "primary" methods
  used in the v1.0 figures; their hues are preserved for visual
  continuity with the published curves.
* WorkerQuality, Raykar, MACE, IEThresh -- the four content-based
  annotation-quality competitors (Dawid-Skene, Raykar et al. 2010,
  MACE / Hovy et al. 2013, IEThresh / Donmez & Carbonell 2008).
* PoD-NoGate, PoD-NoCoupling, PoD-NoVigilance -- the three
  single-component PoD ablations.

The Tableau-10 / Color-Brewer continuation keeps the palette
print-readable and colour-blind friendly."""


def set_pub_style() -> None:
    """Activate the package-wide publication style.

    Idempotent; safe to call from CLI entry points and notebooks alike.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 900,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titlepad": 10,
            "axes.labelpad": 6,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
        }
    )


def save_fig(fig: plt.Figure, outbase: str) -> None:
    """Save ``fig`` as both PDF and PNG at the publication DPI."""
    fig.savefig(outbase + ".pdf")
    fig.savefig(outbase + ".png")
    plt.close(fig)


def add_regime_guides(
    ax: plt.Axes,
    t: np.ndarray,
    schedule: RegimeSchedule,
    label_y: float,
) -> None:
    """Overlay regime bands and labels on a time-axis plot."""
    t_min, t_max = int(t[0]), int(t[-1])

    spans = [
        ("baseline", 0, schedule.baseline),
        ("gaming", schedule.baseline, schedule.baseline + schedule.gaming),
        ("fatigue", schedule.baseline + schedule.gaming, schedule.total()),
    ]

    for name, a, b in spans:
        aa = max(a, t_min)
        bb = min(b, t_max)
        if aa < bb:
            ax.axvspan(aa, bb, alpha=0.06)
            ax.text(
                (aa + bb) / 2.0,
                label_y,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                clip_on=False,
            )

    ax.axvline(schedule.baseline, alpha=0.28)
    ax.axvline(schedule.baseline + schedule.gaming, alpha=0.28)


def plot_methods(
    name: str,
    outdir: str,
    schedule: RegimeSchedule,
    t: np.ndarray,
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    runs: int,
    plot_smooth_w: int,
    phase_label_y: float,
    plot_tmax: int,
) -> None:
    """Render the per-method F1 trajectory figure used in the paper.

    Parameters
    ----------
    name : str
        Dataset label, used both for the title and for the output file
        stem (``<outdir>/figs/<name>_methods_f1.{pdf,png}``).
    outdir : str
        Destination directory (must already exist).
    schedule : RegimeSchedule
        Regime schedule used to overlay the operator-state bands.
    t : np.ndarray
        Time-axis values shared by every method.
    stats : Mapping[str, (mu, ci)]
        Mean and 95% CI half-widths per method, keyed by method name.
    runs : int
        Number of independent runs; appears in the figure title.
    plot_smooth_w : int
        Window width for cosmetic smoothing (centred rolling mean).
    phase_label_y : float
        Vertical position of the regime labels in axes-fraction units.
    plot_tmax : int
        Right-edge cutoff of the time axis.
    """
    t = np.asarray(t, dtype=int)
    mask = t <= int(plot_tmax)
    if not np.any(mask):
        raise ValueError(
            f"No points to plot after applying plot_tmax={plot_tmax}."
        )
    t_plot = t[mask]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.30, top=0.84)

    # Plot the original four primary methods solid; remaining methods are
    # rendered dashed and at reduced opacity so the main F1 figures stay
    # readable. The single-component PoD ablations are excluded from the
    # regime (baseline/gaming/fatigue) figures.
    primary = ["AL", "StaticGating", "AdaptiveGating", "PoD"]
    excluded = {"PoD-NoGate", "PoD-NoCoupling", "PoD-NoVigilance"}
    extras = [m for m in stats.keys() if m not in primary and m not in excluded]
    methods = primary + extras
    n_cols_legend = max(4, min(len(methods), 4))

    for m in methods:
        if m not in stats:
            continue
        mu, ci = stats[m]
        mu = np.asarray(mu)[mask]
        ci = np.asarray(ci)[mask]

        mu_p = smooth_series(mu, plot_smooth_w)
        ci_p = smooth_series(ci, plot_smooth_w)

        is_primary = m in primary
        z = 4 if m == "PoD" else (3 if is_primary else 2)
        lw = 2.8 if m == "PoD" else (2.2 if is_primary else 1.6)
        ls = "-" if is_primary else "--"
        color = METHOD_COLORS.get(m, "#444444")

        ax.plot(
            t_plot, mu_p, label=m, color=color, zorder=z,
            linewidth=lw, linestyle=ls,
        )
        ax.fill_between(
            t_plot,
            mu_p - ci_p,
            mu_p + ci_p,
            color=color,
            alpha=0.10 if is_primary else 0.05,
            linewidth=0.0,
            zorder=z - 1,
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0, int(plot_tmax))
    ax.set_title(f"{name}: F1(t) mean with 95% CI over {runs} runs")
    ax.set_xlabel("Time step t (relative)")
    ax.set_ylabel("F1 score")

    add_regime_guides(ax, t_plot, schedule=schedule, label_y=phase_label_y)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=n_cols_legend,
        handlelength=2.6,
        columnspacing=1.2,
    )
    ax.margins(x=0.01)

    save_fig(fig, os.path.join(outdir, "figs", f"{name}_methods_f1"))
