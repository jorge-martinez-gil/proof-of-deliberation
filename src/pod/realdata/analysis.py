"""
Offline analysis of real-user session data.

Applies the PoD verification primitives (gate, coupling, vigilance) to
the per-trial sequences produced by real participants, then aggregates
the results across participants and blocks. The output set is designed
to slot directly into the paper's real-data section:

* :func:`summarise_participants`  -- per-participant accuracy and mean
  response time across blocks (replaces the simulator's per-regime
  ``p_correct`` and ``Delta`` characterisation with empirical estimates).
* :func:`coupling_table`          -- per-participant per-block Spearman
  rho between recorded response time and recorded complexity ``c_star``,
  which is the empirical test of the Hick-Hyman coupling that PoD's
  authenticity layer assumes.
* :func:`apply_pod_offline`       -- runs ``gate_check``, the gaming and
  fatigue detectors, and the coupling check on every trial in stream
  order, returning the same accept/reject decisions the live PoD layer
  would have produced.
* :func:`block_regime_table`      -- per-block accept-rate table; the
  comparison reported in Section 6.6 of the paper.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pod.config import PoDParams
from pod.core import coupling_check, fatigue_detector, gaming_detector, gate_check
from pod.realdata.loader import CONDITION_ORDER
from pod.utils import spearman_rho


def _per_block_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["participant_id", "block", "induced_condition"], sort=False)
    agg = grp.agg(
        n_trials=("trial_idx", "count"),
        accuracy=("correct", "mean"),
        mean_delib_ms=("delib_ms", "mean"),
        median_delib_ms=("delib_ms", "median"),
        sd_delib_ms=("delib_ms", "std"),
        mean_c_star=("c_star", "mean"),
    ).reset_index()
    agg["cv_delib"] = agg["sd_delib_ms"] / agg["mean_delib_ms"].replace(0, np.nan)
    return agg


def summarise_participants(df: pd.DataFrame) -> pd.DataFrame:
    """Per-participant per-block accuracy / response-time / CV summary."""
    return _per_block_aggregates(df)


def coupling_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-participant per-block Spearman rho between delib_ms and c_star.

    Returns one row per ``(participant_id, block)`` with the rank
    correlation and the trial count. A reliably positive rho on the
    *baseline* block validates the Hick-Hyman coupling assumption on
    real users; near-zero rho on *gaming* / *fatigue* blocks is the
    expected behavioural signature for the PoD authenticity layer.
    """
    rows: List[Dict[str, object]] = []
    for (pid, block), sub in df.groupby(["participant_id", "block"], sort=False):
        x = sub["c_star"].to_numpy(dtype=float)
        y = sub["delib_ms"].to_numpy(dtype=float)
        rho = spearman_rho(x, y) if x.size >= 4 else float("nan")
        rows.append(
            {
                "participant_id": pid,
                "block": block,
                "induced_condition": str(sub["induced_condition"].iloc[0]),
                "n_trials": int(len(sub)),
                "spearman_rho": float(rho) if np.isfinite(rho) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def apply_pod_offline(
    df: pd.DataFrame,
    pod: Optional[PoDParams] = None,
) -> pd.DataFrame:
    """Replay every trial through the PoD verification layer.

    Returns ``df`` augmented with five binary columns:

    * ``pod_gate``           -- output of :func:`pod.core.gate_check`.
    * ``pod_coupling``       -- output of :func:`pod.core.coupling_check`
      on the rolling window up to (and including) the current trial.
    * ``pod_gaming``         -- short-window vigilance detector.
    * ``pod_fatigue``        -- long-window vigilance detector.
    * ``pod_accept``         -- composite acceptance decision matching
      the live runner's logic (gate AND not(gaming) AND not(fatigue)
      AND coupling), without the ``persist_k`` smoothing because the
      real sessions are short enough that single-step decisions are
      already representative.

    The frame is processed in (participant, trial_idx) order so that the
    rolling windows respect per-participant history.
    """
    if pod is None:
        pod = PoDParams()

    df_sorted = df.sort_values(["participant_id", "trial_idx"]).reset_index(drop=True)
    out = df_sorted.copy()

    out["pod_gate"] = 0
    out["pod_coupling"] = 1
    out["pod_gaming"] = 0
    out["pod_fatigue"] = 0
    out["pod_accept"] = 0

    for pid, sub in df_sorted.groupby("participant_id", sort=False):
        idx = sub.index.to_numpy()
        delib_hist: List[float] = []
        comp_hist: List[float] = []
        gate_col: List[int] = []
        coup_col: List[int] = []
        gam_col: List[int] = []
        fat_col: List[int] = []
        acc_col: List[int] = []

        for _, row in sub.iterrows():
            delib = float(row["delib_ms"])
            comp = float(row["c_star"])
            g = int(gate_check(delib, comp, pod))
            comp_hist.append(comp)
            delib_hist.append(delib)
            c = int(coupling_check(np.asarray(comp_hist), np.asarray(delib_hist), pod))
            gm = int(gaming_detector(np.asarray(delib_hist), pod))
            ft = int(fatigue_detector(np.asarray(delib_hist), pod))
            accept = int((g == 1) and (c == 1) and (gm == 0) and (ft == 0))
            gate_col.append(g)
            coup_col.append(c)
            gam_col.append(gm)
            fat_col.append(ft)
            acc_col.append(accept)

        out.loc[idx, "pod_gate"] = gate_col
        out.loc[idx, "pod_coupling"] = coup_col
        out.loc[idx, "pod_gaming"] = gam_col
        out.loc[idx, "pod_fatigue"] = fat_col
        out.loc[idx, "pod_accept"] = acc_col

    return out


def block_regime_table(df_with_pod: pd.DataFrame) -> pd.DataFrame:
    """Per-block PoD accept / detector-fire rates across all participants.

    Expects the augmented frame returned by :func:`apply_pod_offline`.
    The output is the central table reported in Section 6.6: PoD must
    accept *most* labels in the baseline regime and *few* in the gaming
    / fatigue regimes.
    """
    rows: List[Dict[str, object]] = []
    for block, sub in df_with_pod.groupby("block", sort=False):
        cond = str(sub["induced_condition"].iloc[0])
        rows.append(
            {
                "block": block,
                "induced_condition": cond,
                "n_trials": int(len(sub)),
                "accuracy": float(sub["correct"].mean()),
                "mean_delib_ms": float(sub["delib_ms"].mean()),
                "cv_delib": float(sub["delib_ms"].std() / max(sub["delib_ms"].mean(), 1e-9)),
                "gate_pass_rate": float(sub["pod_gate"].mean()),
                "coupling_pass_rate": float(sub["pod_coupling"].mean()),
                "gaming_flag_rate": float(sub["pod_gaming"].mean()),
                "fatigue_flag_rate": float(sub["pod_fatigue"].mean()),
                "pod_accept_rate": float(sub["pod_accept"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    cat = pd.Categorical(
        out["induced_condition"], categories=list(CONDITION_ORDER), ordered=True,
    )
    out = out.assign(induced_condition=cat).sort_values("induced_condition").reset_index(drop=True)
    return out


def regime_classification_score(df_with_pod: pd.DataFrame) -> Dict[str, float]:
    """How well does PoD's gating decision classify the induced regime?

    For each participant, we compute the mean PoD accept rate per block
    and rank-correlate it with the induced regime ordering
    ``baseline > gaming, fatigue``. A high mean rho across participants
    is evidence that the PoD signal tracks human deliberation quality.
    """
    rho_list: List[float] = []
    order = {"baseline": 0, "gaming": 1, "fatigue": 1}  # baseline > others
    for pid, sub in df_with_pod.groupby("participant_id"):
        per_block = sub.groupby("induced_condition")["pod_accept"].mean()
        if len(per_block) < 2:
            continue
        ranks = np.array([order[k] for k in per_block.index])
        rates = per_block.to_numpy(dtype=float)
        rho = spearman_rho(ranks.astype(float), rates) if rates.size >= 2 else float("nan")
        if np.isfinite(rho):
            # We expect *negative* rho: lower numeric rank (baseline=0)
            # should pair with *higher* accept. So flip the sign for
            # readability: a positive score means PoD correctly favours
            # baseline over induced bad regimes.
            rho_list.append(-float(rho))

    if not rho_list:
        return {"n_participants": 0, "mean_score": float("nan"),
                "median_score": float("nan"), "frac_positive": float("nan")}

    arr = np.array(rho_list, dtype=float)
    return {
        "n_participants": int(arr.size),
        "mean_score": float(arr.mean()),
        "median_score": float(np.median(arr)),
        "frac_positive": float(np.mean(arr > 0.0)),
    }
