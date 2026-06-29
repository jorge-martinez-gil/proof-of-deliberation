"""
Statistical helpers and delay filters for the ATC validation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats

DELAY_MIN = 0.10
"""Lower bound on controller-response delay (seconds).

Values below this are likely overlapping or simultaneous speech rather
than deliberation."""

DELAY_MAX_PCT = 99
"""Upper percentile retained from the delay distribution (clip top 1%)."""


def filter_delays(
    entropies: np.ndarray, delays: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Restrict pairs to ``[DELAY_MIN, P99(delays)]``.

    Returns the surviving entropies and delays together with the boolean
    mask and the P99 cutoff (useful for figure annotations).
    """
    p99 = float(np.percentile(delays, DELAY_MAX_PCT))
    mask = (delays >= DELAY_MIN) & (delays <= p99)
    n_removed = int((~mask).sum())
    print(
        f"  Delay filter: removed {n_removed} pairs "
        f"(< {DELAY_MIN}s OR > {p99:.1f}s) -> {int(mask.sum())} remain"
    )
    return entropies[mask], delays[mask], mask, p99


def spearman(x: np.ndarray, y: np.ndarray, label: str = "") -> Tuple[float, float]:
    """Print a labelled Spearman test result and return ``(rho, pval)``."""
    rho, pval = stats.spearmanr(x, y)
    stars = (
        "***" if pval < 0.001
        else "**" if pval < 0.010
        else "*" if pval < 0.050
        else "ns"
    )
    direction = "rises" if rho > 0 else "falls"
    sig = "yes" if pval < 0.05 else "no"
    print(
        f"  {label:48s} rho={rho:+.3f}  p={pval:.4f}  {stars}  "
        f"{sig} {direction} with complexity"
    )
    return float(rho), float(pval)


def bin_means(
    x: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-quantile bin means with standard errors."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    cx, cm, ce = [], [], []
    for j in range(len(edges) - 1):
        m = (x >= edges[j]) & (x < edges[j + 1])
        if m.sum() < 3:
            continue
        cx.append(x[m].mean())
        cm.append(y[m].mean())
        ce.append(stats.sem(y[m]))
    return np.array(cx), np.array(cm), np.array(ce)
