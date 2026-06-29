"""
Real-user-interaction data collection and analysis for PoD.

This subpackage complements the simulator-based study, whose empirical
evaluation in the paper relies exclusively on a stochastic operator
simulator. It provides:

* :mod:`pod.realdata.build_pool`  -- freeze a stratified task pool from
  Electricity (elec2) into a JSON consumable by the browser-based
  labeling app (``PoD_code/labeling/app.html``).
* :mod:`pod.realdata.loader`      -- merge per-participant session CSVs
  downloaded from the labeling app into a tidy frame.
* :mod:`pod.realdata.analysis`    -- apply the PoD verification layers
  offline to real-user data using the same primitives that gate the
  simulator, and produce the figures and statistics reported in the
  real-data section of the paper.
* :mod:`pod.realdata.cli`         -- the ``pod-realdata`` console-script
  entry point that glues the loader and analyser together.

The protocol implemented by the labeling app is documented in
``docs/REAL_DATA_PROTOCOL.md``.
"""

from __future__ import annotations

from pod.realdata.analysis import (
    apply_pod_offline,
    block_regime_table,
    coupling_table,
    summarise_participants,
)
from pod.realdata.loader import (
    REQUIRED_COLUMNS,
    canonical_block_order,
    load_sessions,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "apply_pod_offline",
    "block_regime_table",
    "canonical_block_order",
    "coupling_table",
    "load_sessions",
    "summarise_participants",
]
