"""
UCI 224 Gas Sensor Array Drift stream loader.

The dataset ships as ten batch files (``batch1.dat`` ... ``batch10.dat``)
in libsvm-like sparse text format with 128 numeric features and labels
in ``{1, ..., 6}``. This module downloads, caches, parses, and splits the
batches into a stream/holdout pair as described in Section 6.1 of the
paper.
"""

from __future__ import annotations

import os
import re
import urllib.request
import zipfile
from typing import Dict, List, Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from pod.utils import ensure_dir

_LIBSVM_TOK = re.compile(r"(\d+):([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")

GAS_DRIFT_URL = (
    "https://archive.ics.uci.edu/static/public/224/"
    "gas%2Bsensor%2Barray%2Bdrift%2Bdataset.zip"
)
GAS_DRIFT_N_FEATURES = 128


def _download_file(url: str, dst_path: str) -> None:
    ensure_dir(os.path.dirname(dst_path))
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return
    tmp = dst_path + ".part"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:  # pragma: no cover - defensive
            pass
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst_path)


def download_and_extract_gas_drift(cache_dir: str) -> str:
    """Download (idempotently) and unzip the UCI 224 dataset.

    Returns the extraction directory; the function is a no-op if the
    target directory already contains an ``_done.txt`` marker.
    """
    ensure_dir(cache_dir)
    zip_path = os.path.join(cache_dir, "gas_drift_uci224.zip")
    _download_file(GAS_DRIFT_URL, zip_path)

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


def _parse_libsvm_dense(
    lines: List[str], n_features: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a libsvm-formatted batch into dense ``(X, y)`` arrays."""
    X = np.zeros((len(lines), n_features), dtype=float)
    y = np.zeros((len(lines),), dtype=int)

    for i, ln in enumerate(lines):
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        y[i] = int(float(parts[0]))
        for m in _LIBSVM_TOK.finditer(ln):
            idx = int(m.group(1)) - 1
            if 0 <= idx < n_features:
                X[i, idx] = float(m.group(2))
    return X, y


def load_gas_drift(
    cache_dir: str, batches: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the requested batches and concatenate them in batch order.

    Returns
    -------
    X : np.ndarray of shape ``(N, 128)``
        Feature matrix.
    y : np.ndarray of shape ``(N,)``
        Integer labels in ``{0, ..., 5}`` (originally 1..6, shifted).
    batch_id : np.ndarray of shape ``(N,)``
        Batch index for each row, useful for later split-by-batch
        operations.
    """
    extract_dir = download_and_extract_gas_drift(cache_dir)

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

        Xb, yb = _parse_libsvm_dense(lines, n_features=GAS_DRIFT_N_FEATURES)
        yb = yb.astype(int) - 1  # 1..6 -> 0..5

        Xs.append(Xb)
        ys.append(yb)
        bs.append(np.full((len(yb),), b, dtype=int))

    return np.vstack(Xs), np.concatenate(ys), np.concatenate(bs)


def split_stream_holdout_by_batch(
    X: np.ndarray,
    y: np.ndarray,
    batch_id: np.ndarray,
    holdout_batches: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition into a stream and a holdout using batch identifiers."""
    hold_mask = np.isin(batch_id, np.array(holdout_batches, dtype=int))
    Xh, yh = X[hold_mask], y[hold_mask]
    Xs, ys = X[~hold_mask], y[~hold_mask]
    return Xs, ys, Xh, yh


def make_class_weight_dict(
    classes: np.ndarray, y_sample: np.ndarray
) -> Dict[int, float]:
    """Build a balanced ``class_weight`` dict for ``SGDClassifier``.

    Adds dummy occurrences for classes missing from ``y_sample`` so that
    :func:`sklearn.utils.class_weight.compute_class_weight` does not
    raise when the warm-up sample lacks a class.
    """
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
