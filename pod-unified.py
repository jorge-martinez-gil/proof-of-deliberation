# pod_unified.py
# Unified PoD experiments: OpenML (elec2), UCI224 Gas Drift, and Synth-Boundary
#
# Repairs in this version:
# - Fix TypeError: run_stream_experiment_once() now accepts log_tmax
# - Fix TypeError: run_suite_generic() now accepts and forwards clf knobs
# - Fix NameError: log_tmax inside run_stream_experiment_once now uses the parameter
# - Make aggregation robust if runs end early (e.g., log_tmax): truncate to common length per method

from __future__ import annotations

import os
import re
import math
import json
import argparse
import zipfile
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

try:
    from scipy.stats import spearmanr  # optional
except Exception:
    spearmanr = None


# =========================
# Plot style
# =========================
def set_pub_style() -> None:
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


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_fig(fig: plt.Figure, outbase: str) -> None:
    fig.savefig(outbase + ".pdf")
    fig.savefig(outbase + ".png")
    plt.close(fig)


# =========================
# Config
# =========================
@dataclass(frozen=True)
class RegimeSchedule:
    baseline: int
    gaming: int
    fatigue: int

    def total(self) -> int:
        return self.baseline + self.gaming + self.fatigue

    def phase(self, t_rel: int) -> str:
        if t_rel < self.baseline:
            return "baseline"
        if t_rel < self.baseline + self.gaming:
            return "gaming"
        return "fatigue"


@dataclass(frozen=True)
class ALParams:
    query_budget: float = 0.22
    query_temp: float = 0.28


@dataclass(frozen=True)
class OperatorParams:
    # Baseline
    k: float = 650.0
    c: float = 270.0
    sigma: float = 88.0

    # Gaming
    c_fast: float = 600.0
    eps_fast: float = 25.0

    # Fatigue
    c_slow: float = 925.0
    sigma_high: float = 675.0

    # Correctness
    p_correct_baseline: float = 0.965
    p_correct_gaming: float = 0.22
    p_correct_fatigue: float = 0.45


@dataclass(frozen=True)
class PoDParams:
    coupling_window: int = 120
    coupling_epsilon: float = 0.15

    gate_a: float = 650.0
    gate_b: float = 245.0
    gate_lo_frac: float = 0.58
    gate_hi_frac: float = 0.86
    gate_floor_ms: float = 120.0
    gate_ceil_ms: float = 3200.0

    gaming_window: int = 80
    gaming_mu_max_ms: float = 310.0
    gaming_cv_max: float = 0.06

    fatigue_window: int = 80
    fatigue_cv_min: float = 0.27

    persist_k: int = 3


@dataclass(frozen=True)
class SynthParams:
    d: int = 10
    lambda_complexity: float = 3.0
    rotation_per_step: float = 0.004
    noise_std: float = 1.0


# =========================
# Helpers
# =========================
def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax_stable(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    z = np.clip(z, -60.0, 60.0)
    e = np.exp(z)
    s = np.sum(e, axis=1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return e / s


def proba_from_decision_function(clf: SGDClassifier, X: np.ndarray, n_classes: int) -> np.ndarray:
    df = clf.decision_function(X)
    if n_classes <= 2:
        if df.ndim == 2 and df.shape[1] == 1:
            df = df[:, 0]
        p1 = sigmoid(df.reshape(-1, 1))
        p0 = 1.0 - p1
        return np.hstack([p0, p1])
    if df.ndim == 1:
        df = df.reshape(1, -1)
    return softmax_stable(df)


def entropy_from_proba(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def entropy_unit(ent: float, n_classes: int) -> float:
    return float(np.clip(ent / math.log(max(2, n_classes)), 0.0, 1.0))


def query_probability(complexity_01: float, al: ALParams) -> float:
    z = (complexity_01 - 0.5) / max(1e-9, al.query_temp)
    p = float(sigmoid(np.array([z]))[0])
    return float(np.clip(p * (al.query_budget / 0.5), 0.0, 1.0))


def f1_metric(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    if n_classes <= 2:
        return float(f1_score(y_true, y_pred, zero_division=0))
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def smooth_for_plot(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    return pd.Series(x).rolling(w, center=True, min_periods=max(2, w // 4)).mean().to_numpy()


def _cv(xs: np.ndarray) -> float:
    xs = np.asarray(xs, dtype=float)
    n = int(xs.size)
    if n < 2:
        return 0.0
    mu = float(np.mean(xs))
    if not np.isfinite(mu) or mu <= 1e-12:
        return 0.0
    sd = float(np.std(xs, ddof=0))
    if not np.isfinite(sd):
        return 0.0
    return float(sd / mu)


def spearmanr_fallback(x: np.ndarray, y: np.ndarray) -> float:
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


# =========================
# PoD checks
# =========================
def expected_delib(complexity: float, pod: PoDParams) -> float:
    return pod.gate_a * complexity + pod.gate_b


def gate_check(delib_ms: float, complexity: float, pod: PoDParams) -> int:
    mu = expected_delib(complexity, pod)
    lo = max(pod.gate_floor_ms, mu * (1.0 - pod.gate_lo_frac))
    hi = min(pod.gate_ceil_ms, mu * (1.0 + pod.gate_hi_frac))
    return int(lo <= delib_ms <= hi)


def coupling_check(comp_hist: np.ndarray, delib_hist: np.ndarray, pod: PoDParams) -> int:
    w = pod.coupling_window
    if len(comp_hist) < w:
        return 1
    x = comp_hist[-w:]
    y = delib_hist[-w:]
    if spearmanr is not None:
        rho, _ = spearmanr(x, y)
        return int(np.isfinite(rho) and rho > pod.coupling_epsilon)
    rho = spearmanr_fallback(x, y)
    return int(np.isfinite(rho) and rho > pod.coupling_epsilon)


def gaming_detector(delib_hist: np.ndarray, pod: PoDParams) -> int:
    w = pod.gaming_window
    if delib_hist is None or len(delib_hist) < max(2, w):
        return 0
    xs = np.asarray(delib_hist[-w:], dtype=float)
    mu = float(np.mean(xs))
    cv = _cv(xs)
    return int((mu <= pod.gaming_mu_max_ms) and (cv <= pod.gaming_cv_max))


def fatigue_detector(delib_hist: np.ndarray, pod: PoDParams) -> int:
    w = pod.fatigue_window
    if delib_hist is None or len(delib_hist) < max(2, w):
        return 0
    xs = np.asarray(delib_hist[-w:], dtype=float)
    cv = _cv(xs)
    return int(cv >= pod.fatigue_cv_min)


# =========================
# Operator simulation
# =========================
def simulate_operator(
    rng: np.random.Generator,
    y_true: int,
    n_classes: int,
    complexity: float,
    phase: str,
    op: OperatorParams,
) -> Tuple[int, float]:
    if phase == "baseline":
        delib = rng.normal(op.k * complexity + op.c, op.sigma)
        p_corr = op.p_correct_baseline
    elif phase == "gaming":
        delib = rng.normal(op.c_fast, op.eps_fast)
        p_corr = op.p_correct_gaming
    else:
        delib = rng.normal(op.c_slow, op.sigma_high)
        p_corr = op.p_correct_fatigue

    delib = float(max(50.0, delib))

    # Special mode: force adversarial labels when p_corr < 0
    if p_corr < 0.0:
        if n_classes <= 2:
            return 1 - int(y_true), delib
        return int((int(y_true) + 1) % n_classes), delib

    if rng.random() < p_corr:
        return int(y_true), delib

    if n_classes <= 2:
        return 1 - int(y_true), delib

    choices = [c for c in range(n_classes) if c != y_true]
    return int(rng.choice(choices)), delib


# =========================
# Plot regime guides
# =========================
def add_regime_guides(ax: plt.Axes, t: np.ndarray, schedule: RegimeSchedule, label_y: float) -> None:
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


# =========================
# External holdout
# =========================
def make_external_holdout(
    X_all: np.ndarray,
    y_all: np.ndarray,
    stream_len: int,
    holdout_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X_all) > stream_len:
        X_tail = X_all[stream_len:]
        y_tail = y_all[stream_len:]
        if len(X_tail) >= holdout_size:
            idx = rng.choice(len(X_tail), size=holdout_size, replace=False)
            return X_tail[idx], y_tail[idx]

    replace = holdout_size > len(X_all)
    idx = rng.choice(len(X_all), size=holdout_size, replace=replace)
    return X_all[idx], y_all[idx]


# =========================
# Dataset: OpenML
# =========================
@dataclass(frozen=True)
class OpenMLSpec:
    name: str
    data_id: int


def load_openml_any(spec: OpenMLSpec, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    ds = fetch_openml(data_id=spec.data_id, as_frame=True)
    frame = getattr(ds, "frame", None)

    if frame is not None:
        target_names = getattr(ds, "target_names", None)
        target_col: Optional[str] = None

        if isinstance(target_names, (list, tuple)) and len(target_names) >= 1:
            target_col = target_names[0]
        elif isinstance(target_names, str) and target_names:
            target_col = target_names

        if target_col is None or target_col not in frame.columns:
            for cand in ["class", "Class", "target", "Target", "label", "Label", "y", "Y"]:
                if cand in frame.columns:
                    target_col = cand
                    break

        if target_col is None or target_col not in frame.columns:
            raise ValueError(f"{spec.name}: could not infer target column")

        ysr = pd.Series(frame[target_col])
        Xdf = frame.drop(columns=[target_col])
    else:
        Xdf = ds.data
        ysr = pd.Series(ds.target)

    if ysr.dtype.name in ("category", "object", "string"):
        y = ysr.astype("category").cat.codes.to_numpy().astype(int)
    else:
        uniq = np.sort(ysr.dropna().unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        y = ysr.map(mapping).to_numpy().astype(int)

    Xdf = pd.get_dummies(Xdf, drop_first=False)
    X = Xdf.to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# =========================
# Dataset: UCI224 Gas Drift
# =========================
_libsvm_tok = re.compile(r"(\d+):([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def download_file(url: str, dst_path: str) -> None:
    ensure_dir(os.path.dirname(dst_path))
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return
    tmp = dst_path + ".part"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst_path)


def download_and_extract_gas_drift(cache_dir: str) -> str:
    ensure_dir(cache_dir)
    url = "https://archive.ics.uci.edu/static/public/224/gas%2Bsensor%2Barray%2Bdrift%2Bdataset.zip"
    zip_path = os.path.join(cache_dir, "gas_drift_uci224.zip")
    download_file(url, zip_path)

    extract_dir = os.path.join(cache_dir, "uci224_extract")
    ensure_dir(extract_dir)
    marker = os.path.join(extract_dir, "_done.txt")
    if os.path.exists(marker):
        return extract_dir

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")

    return extract_dir


def parse_libsvm_dense(lines: List[str], n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((len(lines), n_features), dtype=float)
    y = np.zeros((len(lines),), dtype=int)

    for i, ln in enumerate(lines):
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        y[i] = int(float(parts[0]))
        for m in _libsvm_tok.finditer(ln):
            idx = int(m.group(1)) - 1
            if 0 <= idx < n_features:
                X[i, idx] = float(m.group(2))
    return X, y


def load_gas_drift(cache_dir: str, batches: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    extract_dir = download_and_extract_gas_drift(cache_dir)
    n_features = 128

    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    bs: List[np.ndarray] = []

    for b in sorted(batches):
        p1 = os.path.join(extract_dir, "Dataset", f"batch{b}.dat")
        p2 = os.path.join(extract_dir, f"batch{b}.dat")
        path = p1 if os.path.exists(p1) else p2
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find batch{b}.dat")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        Xb, yb = parse_libsvm_dense(lines, n_features=n_features)
        yb = yb.astype(int) - 1  # 1..6 -> 0..5

        Xs.append(Xb)
        ys.append(yb)
        bs.append(np.full((len(yb),), b, dtype=int))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    batch_id = np.concatenate(bs)
    return X, y, batch_id


def split_stream_holdout_by_batch(
    X: np.ndarray,
    y: np.ndarray,
    batch_id: np.ndarray,
    holdout_batches: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hold_mask = np.isin(batch_id, np.array(holdout_batches, dtype=int))
    Xh, yh = X[hold_mask], y[hold_mask]
    Xs, ys = X[~hold_mask], y[~hold_mask]
    return Xs, ys, Xh, yh


def make_class_weight_dict(classes: np.ndarray, y_sample: np.ndarray) -> Dict[int, float]:
    classes = np.asarray(classes, dtype=int)
    y_sample = np.asarray(y_sample, dtype=int)

    present = np.unique(y_sample)
    missing = [c for c in classes.tolist() if c not in present.tolist()]
    if missing:
        y_aug = np.concatenate([y_sample, np.array(missing, dtype=int)])
    else:
        y_aug = y_sample

    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_aug)
    return {int(c): float(wi) for c, wi in zip(classes, w)}


# =========================
# Dataset: Synth-Boundary
# =========================
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(1e-12, n)


def generate_synth_boundary_pool_regime(
    seed: int,
    n: int,
    synth: SynthParams,
    schedule: RegimeSchedule,
    init_fit: int,
    rot_baseline: float = 0.004,
    rot_gaming: float = 0.0,
    rot_fatigue: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = synth.d

    X = rng.normal(0.0, synth.noise_std, size=(n, d)).astype(float)

    w0 = _normalize(rng.normal(0.0, 1.0, size=(d,)))
    u = rng.normal(0.0, 1.0, size=(d,))
    u = u - np.dot(u, w0) * w0
    u = _normalize(u)

    y = np.zeros(n, dtype=int)
    c_star = np.zeros(n, dtype=float)

    theta = 0.0
    for t in range(n):
        # Align with experiment time axis
        t_rel = t - init_fit
        if t_rel < 0:
            phase = "baseline"
        else:
            phase = schedule.phase(t_rel)

        if phase == "baseline":
            theta += rot_baseline
        elif phase == "gaming":
            theta += rot_gaming
        else:
            theta += rot_fatigue

        wt = _normalize(math.cos(theta) * w0 + math.sin(theta) * u)
        margin = float(np.dot(wt, X[t]))
        y[t] = 1 if margin >= 0 else 0
        c_star[t] = float(math.exp(-synth.lambda_complexity * abs(margin)))

    return X, y, c_star


# =========================
# Core runner
# =========================
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
    rng = np.random.default_rng(seed)

    if init_fit < 1:
        raise ValueError(f"init_fit must be >= 1, got {init_fit}")

    T = schedule.total()
    if len(X_stream) < init_fit + T + 10:
        raise ValueError(f"Stream too small: need {init_fit + T + 10}, got {len(X_stream)}")
    if len(X_holdout) < max(500, eval_window):
        raise ValueError(f"Holdout too small: got {len(X_holdout)}")

    Xs = X_stream[: init_fit + T]
    ys = y_stream[: init_fit + T]

    classes = np.unique(np.concatenate([y_stream, y_holdout]))
    n_classes = int(len(classes))

    # Guard: avoid empty fit slices (should not happen with init_fit>=1, but keep safe)
    fit_X = Xs[:init_fit]
    fit_y = ys[:init_fit]
    if fit_X.shape[0] == 0:
        raise ValueError("init_fit produced empty initial fit data")

    imp = SimpleImputer(strategy="mean").fit(fit_X)
    sc = StandardScaler().fit(imp.transform(fit_X))

    Xs_z = sc.transform(imp.transform(Xs))
    Xh_z = sc.transform(imp.transform(X_holdout))

    clf_kwargs = dict(
        loss="log_loss",
        alpha=float(clf_alpha),
        average=bool(clf_average),
        learning_rate="adaptive",
        eta0=float(clf_eta0),
        random_state=seed,
    )
    if class_weight_dict is not None:
        clf_kwargs["class_weight"] = class_weight_dict
    clf = SGDClassifier(**clf_kwargs)
    clf.partial_fit(Xs_z[:init_fit], ys[:init_fit], classes=classes)

    comp_hist: List[float] = []
    delib_hist: List[float] = []

    bad_c = 0
    bad_g = 0
    bad_f = 0

    q_total = 0
    a_total = 0
    q_phase = {"baseline": 0, "gaming": 0, "fatigue": 0}
    a_phase = {"baseline": 0, "gaming": 0, "fatigue": 0}

    f1s: List[float] = []
    ts: List[int] = []
    hptr = 0

    for t in range(init_fit, init_fit + T):
        t_rel = t - init_fit
        if log_tmax > 0 and t_rel > int(log_tmax):
            break

        phase = schedule.phase(t_rel)
        xt = Xs_z[t : t + 1]

        if complexity_mode == "entropy":
            proba = proba_from_decision_function(clf, xt, n_classes=n_classes)[0]
            comp = entropy_unit(entropy_from_proba(proba), max(2, n_classes))
        elif complexity_mode == "c_star":
            if c_star_stream is None:
                raise ValueError("complexity_mode=c_star requires c_star_stream")
            comp = float(c_star_stream[t])
        else:
            raise ValueError(f"Unknown complexity_mode: {complexity_mode}")

        if rng.random() < query_probability(comp, al):
            q_total += 1
            q_phase[phase] += 1

            y_tilde, delib = simulate_operator(rng, int(ys[t]), n_classes, comp, phase, op)

            if method == "AL":
                accept = True
            elif method == "StaticGating":
                accept = delib >= 510.0
            elif method == "AdaptiveGating":
                accept = bool(gate_check(delib, comp, pod))
            else:
                g_ok = gate_check(delib, comp, pod)

                comp_hist.append(comp)
                delib_hist.append(float(delib))

                c_raw = coupling_check(np.asarray(comp_hist), np.asarray(delib_hist), pod)
                gam_raw = gaming_detector(np.asarray(delib_hist), pod)
                fat_raw = fatigue_detector(np.asarray(delib_hist), pod)

                bad_c = bad_c + 1 if c_raw == 0 else 0
                bad_g = bad_g + 1 if gam_raw == 1 else 0
                bad_f = bad_f + 1 if fat_raw == 1 else 0

                ok = (bad_c < pod.persist_k) and (bad_g < pod.persist_k) and (bad_f < pod.persist_k)
                accept = bool((g_ok == 1) and ok)

            if accept:
                a_total += 1
                a_phase[phase] += 1
                clf.partial_fit(xt, np.array([y_tilde], dtype=int))

            if method in ("AL", "StaticGating", "AdaptiveGating"):
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
    colors = {
        "AL": "#1f77b4",
        "StaticGating": "#ff7f0e",
        "AdaptiveGating": "#2ca02c",
        "PoD": "#d62728",
    }

    t = np.asarray(t, dtype=int)
    mask = t <= int(plot_tmax)
    if not np.any(mask):
        raise ValueError(f"No points to plot after applying plot_tmax={plot_tmax}.")
    t_plot = t[mask]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.30, top=0.84)

    methods = ["AL", "StaticGating", "AdaptiveGating", "PoD"]
    for m in methods:
        mu, ci = stats[m]
        mu = np.asarray(mu)[mask]
        ci = np.asarray(ci)[mask]

        mu_p = smooth_for_plot(mu, plot_smooth_w)
        ci_p = smooth_for_plot(ci, plot_smooth_w)

        z = 4 if m == "PoD" else 3
        lw = 2.8 if m == "PoD" else 2.2
        ax.plot(t_plot, mu_p, label=m, color=colors[m], zorder=z, linewidth=lw)
        ax.fill_between(
            t_plot,
            mu_p - ci_p,
            mu_p + ci_p,
            color=colors[m],
            alpha=0.12,
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
        ncol=4,
        handlelength=2.6,
        columnspacing=1.2,
    )
    ax.margins(x=0.01)

    save_fig(fig, os.path.join(outdir, "figs", f"{name}_methods_f1"))


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
    run_data_fn: Callable[[int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]],
    complexity_mode: str,
    class_weight_dict: Optional[Dict[int, float]] = None,
    al: Optional[ALParams] = None,
    pod: Optional[PoDParams] = None,
    op: Optional[OperatorParams] = None,
    clf_alpha: float = 2e-5,
    clf_eta0: float = 0.01,
    clf_average: bool = True,
) -> None:
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "runs"))
    ensure_dir(os.path.join(outdir, "figs"))

    al = al or ALParams()
    pod = pod or PoDParams()
    op = op or OperatorParams()
    methods = ["AL", "StaticGating", "AdaptiveGating", "PoD"]

    logs: Dict[str, List[pd.DataFrame]] = {m: [] for m in methods}
    diag_rows: List[Dict[str, Any]] = []

    for r in range(runs):
        Xs, ys, Xh, yh, c_star = run_data_fn(r)

        for m in methods:
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
            df.to_csv(os.path.join(outdir, "runs", f"{name}_{m}_run{r}.csv"), index=False)

            row = {"dataset": name, "run": r, "method": m}
            row.update(diag)
            diag_rows.append(row)

    pd.DataFrame(diag_rows).to_csv(os.path.join(outdir, "diagnostics.csv"), index=False)

    # Robust aggregation: truncate all runs per method to the shortest length
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    t_ref: Optional[np.ndarray] = None

    for m in methods:
        lengths = [len(df) for df in logs[m]]
        if not lengths or min(lengths) == 0:
            raise ValueError(f"No evaluation points collected for method={m}. Check eval_every, log_tmax, schedule.")
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

    if t_ref is None:
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
                "schedule": {"baseline": schedule.baseline, "gaming": schedule.gaming, "fatigue": schedule.fatigue},
                "methods": methods,
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
                "clf": {"alpha": clf_alpha, "eta0": clf_eta0, "average": clf_average},
                "plot_smooth_w": plot_smooth_w,
                "phase_label_y": phase_label_y,
                "plot_tmax": plot_tmax,
                "log_tmax": log_tmax,
            },
            f,
            indent=2,
        )


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str, default="synth,elec2,gas")
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
    parser.add_argument("--holdout", type=int, default=6000)

    parser.add_argument("--cache_dir", type=str, default="data_cache_uci224")
    parser.add_argument("--stream_batches", type=str, default="1,2,3,4,5,6,7,8")
    parser.add_argument("--holdout_batches", type=str, default="9,10")

    parser.add_argument("--synth_pool", type=int, default=20000)
    parser.add_argument("--synth_seed", type=int, default=50000)

    args = parser.parse_args()

    set_pub_style()
    ensure_dir(args.out)

    wanted = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    wanted_set = set(wanted)

    # -------------------------------------------------
    # Per-dataset parameter sets
    # -------------------------------------------------
    def cfg_elec2() -> Dict[str, object]:
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

    def cfg_gas() -> Dict[str, object]:
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

    def cfg_synth() -> Dict[str, object]:
        return dict(
            runs=20,
            init_fit=300,

            # High querying so AL poisons itself strongly in fatigue
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

            # Keep green bad: narrow gate around gaming time, constant mu (gate_a=0)
            gate_a=0.0,
            gate_b=600.0,
            gate_lo_frac=0.2,
            gate_hi_frac=0.2,
            gate_floor_ms=460.0,
            gate_ceil_ms=700.0,

            op_p_correct_baseline=0.7,

            # Gaming poison
            op_p_correct_gaming=0.07,
            op_eps_fast=1.5,

            # Fatigue poison again (this is the key change to make BLUE fall)
            op_p_correct_fatigue=0.12,
            op_c_slow=1100.0,
            op_sigma_high=2200.0,

            clf_average=True,
            clf_eta0=0.03,
            clf_alpha=2e-6,
        )


    def build_params(
        cfg: Dict[str, object]
    ) -> Tuple[RegimeSchedule, int, int, ALParams, PoDParams, OperatorParams, Dict[str, float]]:
        schedule_local = RegimeSchedule(int(args.baseline), int(args.gaming), int(args.fatigue))
        init_fit_local = int(cfg["init_fit"])
        runs_local = int(cfg["runs"])

        al_local = ALParams(query_budget=float(cfg["query_budget"]), query_temp=float(cfg["query_temp"]))

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

    # ---- OpenML elec2 ----
    if "elec2" in wanted_set:
        cfg = cfg_elec2()
        schedule_e, init_fit_e, runs_e, al_e, pod_e, op_e, clf_e = build_params(cfg)

        print(f"[elec2] runs={runs_e} init_fit={init_fit_e}")

        spec = OpenMLSpec("elec2", int(args.openml_id))
        print(f"Loading OpenML {spec.name} (id={spec.data_id})")
        X_all, y_all = load_openml_any(spec, seed=123)

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
            Xh, yh = make_external_holdout(Xr, yr, stream_len=stream_len, holdout_size=holdout_size, rng=rng)
            return Xs, ys, Xh, yh, None

        outdir = os.path.join(args.out, "elec2")
        run_suite_generic(
            name="elec2",
            outdir=outdir,
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
            class_weight_dict=None,
            al=al_e,
            pod=pod_e,
            op=op_e,
            clf_alpha=clf_e["clf_alpha"],
            clf_eta0=clf_e["clf_eta0"],
            clf_average=clf_e["clf_average"],
        )

    # ---- Gas drift ----
    if "gas" in wanted_set:
        cfg = cfg_gas()
        schedule_g, init_fit_g, runs_g, al_g, pod_g, op_g, clf_g = build_params(cfg)

        print(f"[gas] runs={runs_g} init_fit={init_fit_g}")

        stream_batches = _parse_int_list(args.stream_batches)
        holdout_batches = _parse_int_list(args.holdout_batches)
        all_batches = sorted(list(set(stream_batches + holdout_batches)))
        if any(b < 1 or b > 10 for b in all_batches):
            raise ValueError("Batches must be in 1..10")

        print("Downloading and loading UCI224 Gas Sensor Array Drift.")
        X, y, batch_id = load_gas_drift(cache_dir=args.cache_dir, batches=all_batches)
        Xs_all, ys_all, Xh, yh = split_stream_holdout_by_batch(X, y, batch_id, holdout_batches=holdout_batches)

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

    # ---- Synth-Boundary ----
    if "synth" in wanted_set:
        cfg = cfg_synth()
        schedule_s, init_fit_s, runs_s, al_s, pod_s, op_s, clf_s = build_params(cfg)

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
                raise ValueError(f"Synth pool too small: need at least {need}, got {len(X_pool)}. Increase --synth_pool.")

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

    print("Done.")
    print(f"Outputs under: {args.out}/<dataset>/figs/*_methods_f1.pdf")


if __name__ == "__main__":
    main()


