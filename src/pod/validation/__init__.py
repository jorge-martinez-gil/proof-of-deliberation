"""
Hick-Hyman validation pipeline on Air Traffic Control corpora.

Reproduces the behavioural-validation experiment of Section 6.5,
which establishes that PoD's coupling signal corresponds to a property
of real deliberation rather than an artefact of the simulator.

The pipeline:

1. Loads pilot-controller pairs from the ``atco2_corpus_1h`` and
   ``uwb_atcc`` HuggingFace datasets (:mod:`.corpora`).
2. Computes out-of-fold predictive entropies with a calibrated
   logistic-regression pipeline and temperature scaling
   (:mod:`.entropy`).
3. Runs Spearman tests against controller-response delays, with
   linguistic-proxy contrasts (:mod:`.stats`).
4. Renders the summary figure used in the paper (:mod:`.figure`).

The legacy single-file script lives in the repository root as
``hick_hyman_pod_validation_v2.py``; this package presents the same
computation in a modular form callable from
``pod.cli.validation.main``.
"""

from __future__ import annotations

from pod.validation.corpora import build_pairs, load_all_segments
from pod.validation.entropy import (
    apply_temperature,
    compute_oof_entropies,
    temperature_scale,
)
from pod.validation.labels import LABELS, command_label, encode_label
from pod.validation.stats import bin_means, filter_delays, spearman

__all__ = [
    "LABELS",
    "apply_temperature",
    "bin_means",
    "build_pairs",
    "command_label",
    "compute_oof_entropies",
    "encode_label",
    "filter_delays",
    "load_all_segments",
    "spearman",
    "temperature_scale",
]
