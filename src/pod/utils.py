"""
Numerical and I/O utilities shared across the PoD package.

This module collects small helpers that have no PoD-specific semantics
(sigmoid, softmax, entropy, rank correlation, output-directory helpers,
etc.) so that the modules implementing the protocol stay focused on the
behaviour described in the paper.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import List

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional acceleration
    from scipy.stats import spearmanr as _scipy_spearmanr  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _scipy_spearmanr = None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    """Create ``path`` (and any missing parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def parse_int_list(spec: str) -> List[int]:
    """Parse a comma-separated string of integers (whitespace tolerant)."""
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Scalar / vector numerics
# ---------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid."""
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax_stable(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax with overflow protection."""
    z = z - np.max(z, axis=1, keepdims=True)
    z = np.clip(z, -60.0, 60.0)
    e = np.exp(z)
    s = np.sum(e, axis=1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return e / s


def entropy_from_proba(p: np.ndarray) -> float:
    """Shannon entropy in nats of a probability vector."""
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def entropy_unit(ent: float, n_classes: int) -> float:
    """Map raw entropy (nats) onto the unit interval ``[0, 1]``."""
    return float(np.clip(ent / math.log(max(2, n_classes)), 0.0, 1.0))


def coefficient_of_variation(xs: np.ndarray) -> float:
    """Coefficient of variation (sigma / mu), 0 for degenerate inputs.

    Returns 0 when the sample contains fewer than two points, when the
    mean is non-finite or vanishingly small, or when the standard
    deviation is non-finite. This matches the behaviour of the
    vigilance check in the paper.
    """
    xs = np.asarray(xs, dtype=float)
    if xs.size < 2:
        return 0.0
    mu = float(np.mean(xs))
    if not np.isfinite(mu) or mu <= 1e-12:
        return 0.0
    sd = float(np.std(xs, ddof=0))
    if not np.isfinite(sd):
        return 0.0
    return float(sd / mu)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank-correlation coefficient.

    Uses SciPy when available, otherwise falls back to a pure-NumPy
    implementation. NaN is returned when fewer than two paired
    observations are available or when either rank vector is constant.
    The SciPy ``ConstantInputWarning`` for the constant case is
    suppressed because we already encode that as ``nan`` by contract.
    """
    if _scipy_spearmanr is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = _scipy_spearmanr(x, y)
        return float(rho)
    return _spearman_rho_fallback(x, y)


def _spearman_rho_fallback(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return float("nan")

    order_x = np.argsort(x, kind="mergesort")
    ranks_x = np.empty(x.size, dtype=float)
    ranks_x[order_x] = np.arange(1, x.size + 1, dtype=float)

    order_y = np.argsort(y, kind="mergesort")
    ranks_y = np.empty(y.size, dtype=float)
    ranks_y[order_y] = np.arange(1, y.size + 1, dtype=float)

    rx = ranks_x - ranks_x.mean()
    ry = ranks_y - ranks_y.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


# ---------------------------------------------------------------------------
# Pandas helper
# ---------------------------------------------------------------------------
def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean used only for plot smoothing."""
    if window <= 1:
        return values
    return (
        pd.Series(values)
        .rolling(window, center=True, min_periods=max(2, window // 4))
        .mean()
        .to_numpy()
    )
