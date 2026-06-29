"""
Stream loaders for the PoD empirical evaluation.

Three streams are supported, matching the paper's Section 6.1:

* :mod:`pod.streams.synth`     -- *Synth-Boundary*, the rotating
  linear-separator stream used for controlled stress tests.
* :mod:`pod.streams.openml`    -- *Electricity (elec2)*, fetched from
  OpenML (data id ``44156``).
* :mod:`pod.streams.gas_drift` -- *UCI 224 Gas Sensor Array Drift*,
  downloaded and cached from the UCI repository on first use.
"""

from __future__ import annotations

from pod.streams.gas_drift import (
    download_and_extract_gas_drift,
    load_gas_drift,
    make_class_weight_dict,
    split_stream_holdout_by_batch,
)
from pod.streams.openml import OpenMLSpec, load_openml_any
from pod.streams.poker import load_poker_local
from pod.streams.synth import generate_synth_boundary_pool_regime

__all__ = [
    "OpenMLSpec",
    "download_and_extract_gas_drift",
    "generate_synth_boundary_pool_regime",
    "load_gas_drift",
    "load_openml_any",
    "load_poker_local",
    "make_class_weight_dict",
    "split_stream_holdout_by_batch",
]
