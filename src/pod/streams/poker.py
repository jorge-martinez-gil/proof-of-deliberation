"""
Local-file stream loader for the UCI *Poker-Hand* dataset (OpenML id 1595).

Poker-Hand is a 10-class classification benchmark with ~1.025M instances
(25,010 ``-training-true`` + 1,000,000 ``-testing``) and 10 predictive
attributes -- the suit (ordinal 1-4) and rank (numerical 1-13) of each of
five drawn cards. The goal attribute (column 11) is the poker hand, an
ordinal 0-9 (``Nothing`` ... ``Royal flush``). It is widely used in the
streaming / concept-drift literature as a real-world *virtual-drift*
benchmark (the class-conditional feature distribution is stationary but the
class priors shift across the stream), and complements the existing five
streams with a new (game / combinatorial) domain.

Unlike the other real streams, Poker-Hand is loaded from local CSV files
rather than fetched from OpenML, so it works in network-isolated
environments. Point :func:`load_poker_local` at the directory containing the
two ``poker-hand-*.data`` files.

The loader mirrors :func:`pod.streams.openml.load_openml_any`:

1. Reads the two comma-separated ``.data`` files (no header).
2. Treats the 10 card attributes as numeric features (the canonical
   MOA / scikit-multiflow representation) and the trailing column as the
   integer class label, remapped to sorted 0-based codes.
3. Permutes the rows deterministically with the provided seed so that
   subsequent slicing yields reproducible streams.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

#: Canonical column names: suit/rank of cards 1-5, then the class label.
_COLS = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "CLASS"]

#: The two files distributed with the UCI Poker-Hand archive.
_FILES = ("poker-hand-training-true.data", "poker-hand-testing.data")


def load_poker_local(data_dir: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load the local UCI Poker-Hand files into a permuted ``(X, y)`` pool.

    Parameters
    ----------
    data_dir : str
        Directory containing ``poker-hand-training-true.data`` and
        ``poker-hand-testing.data`` (either or both may be present; all
        that are found are concatenated).
    seed : int
        Seed for the row permutation only; file parsing is deterministic.

    Returns
    -------
    (X, y) : tuple of np.ndarray
        ``X`` is float64 of shape ``(n, 10)``; ``y`` is int of shape
        ``(n,)`` with class codes in ``0..9``.
    """
    frames = []
    for fname in _FILES:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            frames.append(
                pd.read_csv(path, header=None, names=_COLS, dtype=np.int64)
            )
    if not frames:
        raise FileNotFoundError(
            f"No Poker-Hand .data files found under {data_dir!r} "
            f"(expected one of {_FILES})"
        )

    frame = pd.concat(frames, ignore_index=True)

    ysr = frame["CLASS"]
    Xdf = frame.drop(columns=["CLASS"])

    # Remap labels to sorted 0-based integer codes (identity here, 0..9,
    # but kept for parity with load_openml_any and robustness to subsets).
    uniq = np.sort(ysr.dropna().unique())
    mapping = {v: i for i, v in enumerate(uniq)}
    y = ysr.map(mapping).to_numpy().astype(int)
    X = Xdf.to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]
