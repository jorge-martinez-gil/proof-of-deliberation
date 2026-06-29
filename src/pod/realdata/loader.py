"""
Load real-user session CSVs produced by ``PoD_code/labeling/app.html``.

Each participant downloads one CSV per session; this module aggregates an
arbitrary number of them into a single tidy ``pandas.DataFrame`` with
the canonical schema documented in ``PoD_code/labeling/README.md``.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "participant_id",
    "session_id",
    "csv_version",
    "block",
    "induced_condition",
    "trial_idx",
    "item_id",
    "c_star",
    "complexity_bin",
    "y_true",
    "y_user",
    "correct",
    "delib_ms",
)
"""Columns the analysis layer relies on; missing columns will raise."""

BLOCK_ORDER: Tuple[str, ...] = ("baseline", "speed_bonus", "long_block")
"""Canonical presentation order of the three blocks."""

CONDITION_ORDER: Tuple[str, ...] = ("baseline", "gaming", "fatigue")
"""Canonical PoD-regime order matching the induced conditions."""


def canonical_block_order() -> Tuple[str, ...]:
    """Return the canonical block presentation order."""
    return BLOCK_ORDER


def _read_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {path!r} is missing required columns: {missing}. "
            "Was it produced by the labeling app at the matching schema "
            "version?"
        )
    df["__source_file"] = os.path.basename(path)
    return df


def load_sessions(paths_or_dir: str | List[str]) -> pd.DataFrame:
    """Load one or more session CSVs into a tidy frame.

    Parameters
    ----------
    paths_or_dir : str or list[str]
        Either a directory holding ``*.csv`` files, or an explicit list
        of CSV paths.

    Returns
    -------
    df : pandas.DataFrame
        Tidy concatenated frame with one row per trial; duplicate
        ``(participant_id, trial_idx)`` pairs (e.g. if a participant
        finished the same session twice) are dropped, keeping the latest
        occurrence by file mtime.
    """
    if isinstance(paths_or_dir, str):
        if not os.path.exists(paths_or_dir):
            raise FileNotFoundError(paths_or_dir)
        if os.path.isdir(paths_or_dir):
            paths = sorted(
                os.path.join(paths_or_dir, f)
                for f in os.listdir(paths_or_dir)
                if f.lower().endswith(".csv")
            )
        else:
            paths = [paths_or_dir]
    else:
        paths = list(paths_or_dir)

    if not paths:
        raise ValueError(
            f"No CSV files found under {paths_or_dir!r}. Did participants "
            "upload their downloads?"
        )

    paths = sorted(paths, key=lambda p: os.path.getmtime(p))
    frames = [_read_one(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    df = df.drop_duplicates(
        subset=["participant_id", "trial_idx"], keep="last"
    ).reset_index(drop=True)

    df["block"] = df["block"].astype(str)
    df["induced_condition"] = df["induced_condition"].astype(str)
    df["complexity_bin"] = df["complexity_bin"].astype(str)
    df["delib_ms"] = df["delib_ms"].astype(float)
    df["c_star"] = df["c_star"].astype(float)
    df["y_true"] = df["y_true"].astype(int)
    df["y_user"] = df["y_user"].astype(int)
    df["correct"] = df["correct"].astype(int)
    return df
