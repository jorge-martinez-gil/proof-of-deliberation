"""
Proof-of-Deliberation (PoD)
===========================

A process-based supervision-certification protocol for streaming
Human-in-the-Loop active learning.

PoD attaches a credibility signal to every human-generated label using
*observable interaction traces* (deliberation time and its coupling with
task difficulty), rather than the label content itself, and gates model
updates accordingly. The protocol is described in:

    J. Martinez-Gil. "Proof-of-Deliberation for Certifying Human
    Supervision Reliability in Streaming Data Quality." IEEE Transactions
    on Knowledge and Data Engineering, 2026.

Public API
----------

>>> from pod import PoDParams, ALParams, OperatorParams, RegimeSchedule
>>> from pod.core import gate_check, coupling_check, fatigue_detector
>>> from pod.experiment import run_stream_experiment_once, run_suite_generic

Command-line entry points
-------------------------

* ``pod-experiments``    -- reproduces the streaming experiments
                            (Synth-Boundary, Electricity, Gas Drift)
* ``pod-validate``       -- reproduces the Hick-Hyman validation on
                            Air Traffic Control logs

See ``docs/REPRODUCIBILITY.md`` for exact-command reproduction.
"""

from __future__ import annotations

from pod.config import (
    ALParams,
    OperatorParams,
    PoDParams,
    RegimeSchedule,
    SynthParams,
)

__all__ = [
    "ALParams",
    "OperatorParams",
    "PoDParams",
    "RegimeSchedule",
    "SynthParams",
    "__version__",
]

__version__ = "1.0.0"
