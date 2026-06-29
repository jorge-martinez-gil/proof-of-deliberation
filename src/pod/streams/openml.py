"""
OpenML-backed stream loader (used for the Electricity / *elec2* stream).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


@dataclass(frozen=True)
class OpenMLSpec:
    """Identifier triple for an OpenML dataset."""

    name: str
    data_id: int


def load_openml_any(spec: OpenMLSpec, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load and pre-process an OpenML classification dataset.

    The function:

    1. Calls :func:`sklearn.datasets.fetch_openml` with ``as_frame=True``.
    2. Infers the target column from OpenML metadata, falling back to a
       short list of canonical names (``class``, ``target``, ``label``,
       ``y`` and their capitalised variants).
    3. Maps the target to integer codes preserving sorted order.
    4. One-hot encodes categorical features via
       :func:`pandas.get_dummies` (no column drop).
    5. Permutes the rows deterministically with the provided seed so
       that subsequent slicing yields reproducible streams.

    Parameters
    ----------
    spec : OpenMLSpec
        Dataset identifier; ``data_id=44156`` is the Electricity stream
        used in the paper.
    seed : int
        Seed for the row permutation only; the OpenML fetch itself is
        deterministic.
    """
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
