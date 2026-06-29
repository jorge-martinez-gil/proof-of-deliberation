# Proof-of-Deliberation (PoD)

> Reference implementation for
> **J. Martinez-Gil. _Proof-of-Deliberation for Certifying Human
> Supervision Reliability in Streaming Data Quality_. IEEE
> Transactions on Knowledge and Data Engineering, 2026.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC.svg)](https://docs.pytest.org/)

PoD is a process-based supervision-certification protocol for streaming
Human-in-the-Loop active learning. It attaches a credibility signal to
every human-generated label using *observable interaction traces* --
the deliberation time and its coupling with task difficulty -- rather
than the label content itself, and gates model updates accordingly.

This repository is the official, reproducible companion to the paper.
It packages the protocol as an installable Python library
(`pod`), exposes two console-script entry points
(`pod-experiments`, `pod-validate`), and ships frozen configurations,
unit tests, and per-dataset reproduction recipes.

---

## Contents

- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [The PoD protocol at a glance](#the-pod-protocol-at-a-glance)
- [Reproducing the paper](#reproducing-the-paper)
- [Hick-Hyman validation on ATC logs](#hick-hyman-validation-on-atc-logs)
- [Real-user labeling study](#real-user-labeling-study)
- [Development](#development)
- [Real-time PoD dashboard](#real-time-pod-dashboard)
- [Citation](#citation)
- [License](#license)

---

## Quickstart

```bash
git clone https://github.com/jorge-martinez-gil/proof-of-deliberation.git
cd proof-of-deliberation

# Editable install with the streaming-experiment dependencies
pip install -e .

# Reproduce the three F1 trajectory figures from Section 6
pod-experiments --datasets synth,elec2,gas
```

For the Hick-Hyman validation (Section 6.5), install the extra group:

```bash
pip install -e ".[validation]"
pod-validate --out hick_hyman_pod_validation_v2.png
```

For developers (tests, lint, type-check):

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
mypy src/pod
```

---

## Repository layout

```
proof-of-deliberation/
|
+-- src/pod/                # Installable package
|   +-- __init__.py
|   +-- config.py           # Frozen dataclasses (Section 4 notation)
|   +-- core.py             # Gate, coupling, vigilance checks
|   +-- operator.py         # Stochastic operator simulator
|   +-- learner.py          # SGD wrapper + decision-function softmax
|   +-- baselines.py        # AL, StaticGating, AdaptiveGating
|   +-- experiment.py       # Closed-loop runner and aggregator
|   +-- viz.py              # Publication-grade plotting
|   +-- presets.py          # Frozen per-dataset hyperparameters
|   +-- streams/            # Synth-Boundary, OpenML, UCI 224 loaders
|   +-- validation/         # Hick-Hyman pipeline on ATC corpora
|   +-- cli/                # `pod-experiments`, `pod-validate`
|
+-- configs/                # JSON snapshots matching presets.py
+-- experiments/            # Extended-regime, ablation, and robustness runners
+-- scripts/                # Stats-report and claim-checking utilities
+-- tests/                  # pytest suite (unit + integration)
+-- docs/                   # Notation, reproducibility, architecture
+-- PoD_code/               # Browser-based MQTT dashboard + labeling app
+-- archive/                # Pre-refactor scripts (do NOT use)
|
+-- Makefile                # Reproducibility entry points (`make paper`)
+-- pyproject.toml          # PEP 621 packaging + tooling config
+-- requirements*.txt       # Pinned dependency manifests
+-- CITATION.cff            # Software citation metadata
+-- CHANGELOG.md            # Project changelog
+-- LICENSE                 # MIT
+-- README.md               # This file
```

Generated experiment outputs (`out_*/`) and downloaded datasets
(`data_cache_uci224/`, etc.) are produced on demand by the runners and
are intentionally not tracked; see the reproduction recipes below.

---

## The PoD protocol at a glance

For each queried label, PoD checks three independent signals:

| Layer            | Symbol              | Implementation                                                          |
|------------------|---------------------|-------------------------------------------------------------------------|
| Deliberation gate    | `V_t`           | [`pod.core.gate_check`](src/pod/core.py)                                |
| Cognitive coupling   | `rho_cog`       | [`pod.core.coupling_check`](src/pod/core.py)                            |
| Multi-scale vigilance | `S_vig`        | [`pod.core.gaming_detector`](src/pod/core.py), [`pod.core.fatigue_detector`](src/pod/core.py) |

A label is incorporated into the learner only when *all three* checks
agree, mirroring the composite verification function

```
V(pi_t) = (V_t = 1) AND (rho_cog > epsilon) AND (S_vig = 1)
```

defined in Section 4.4 of the paper. See [`docs/NOTATION.md`](docs/NOTATION.md)
for the complete symbol-to-code mapping.

---

## Reproducing the paper

The reference experiments compare PoD against four baselines and three
ablation variants on five streams (Synth-Boundary, Electricity,
Gas Drift, Forest CoverType, Airlines) under three operator regimes
(Baseline, Gaming, Fatigue):

| Family       | Method            | Decision rule                                   |
|--------------|-------------------|-------------------------------------------------|
| Baselines    | `AL`              | Accept every queried label.                     |
|              | `StaticGating`    | Accept iff `delib_ms >= 510`.                   |
|              | `AdaptiveGating`  | Difficulty-scaled deliberation window only.     |
|              | `WorkerQuality`   | Online Dawid-Skene single-annotator posterior.  |
| PoD          | `PoD`             | Composite verification (gate + coupling + S_vig).|
| Ablations    | `PoD-NoGate`      | Drop V_t; keep coupling and vigilance.          |
|              | `PoD-NoCoupling`  | Drop rho_cog; keep V_t and vigilance.           |
|              | `PoD-NoVigilance` | Drop gaming + fatigue detectors; keep V_t, rho. |

### Full reproduction (20 runs per dataset)

```bash
pod-experiments --datasets synth,elec2,gas,covertype,airlines
```

This produces, under `out_pod_unified/`:

```
out_pod_unified/
+-- synth_boundary/
|   +-- figs/Synth-Boundary_methods_f1.{pdf,png}     # Figure in Sec. 6
|   +-- runs/Synth-Boundary_<method>_run<N>.csv      # Per-run F1 trace
|   +-- diagnostics.csv                              # Query/accept rates
|   +-- config.json                                  # Frozen config snapshot
+-- elec2/
+-- uci224_gas_drift/
```

### Single-dataset / quick sanity check

```bash
# 3-run Synth-Boundary smoke test (~30 seconds)
pod-experiments --datasets synth --runs 3 --baseline 500 --gaming 500 \
                --fatigue 500 --synth_pool 5000 --holdout 1000
```

### Seeding and determinism

All randomness is centralised through `numpy.random.default_rng(seed)`.
The per-run seed is derived deterministically from the run index as
`1000 * run_idx + 7` (see
[`pod.experiment.run_suite_generic`](src/pod/experiment.py)). Identical
hardware, identical Python and NumPy versions, and identical pinned
dependencies (see [`requirements.txt`](requirements.txt)) reproduce the
F1 trajectories exactly.

### Configuration provenance

Every run writes a `config.json` next to the figures. To verify that
your run matches the paper, diff it against the appropriate file in
[`configs/`](configs/):

```bash
diff -u configs/synth.json out_pod_unified/synth_boundary/config.json
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full
recipe.

### Statistical analysis (TKDE-grade)

After `pod-experiments` has populated `out_pod_unified/`, the companion
entry point `pod-stats` produces the per-method statistical artefacts
required for TKDE-style evaluations:

```bash
pod-stats --in out_pod_unified --reference PoD
```

This writes, under `out_pod_unified/stats/`:

| Artefact                  | Contents                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| `per_run_scores.csv`      | One row per (dataset, method, run) with the scalar score per the chosen reduction (`--score-mode end` or `auc`). |
| `average_ranks.csv`       | Mean rank of each method across datasets (lower is better).             |
| `friedman.json`           | Friedman chi-square and Iman-Davenport F omnibus test with both p-values.|
| `nemenyi_cd.json`         | Critical-difference value `CD` at the chosen alpha (Demsar 2006, Eq. 5).|
| `cd_diagram.pdf`/`.png`   | Demsar-style critical-difference diagram with NSD cliques.              |
| `wilcoxon_holm.csv`       | Pairwise Wilcoxon signed-rank vs. the reference method, Holm-corrected. |
| `bootstrap_ci.csv`        | Percentile bootstrap CI (default 10K resamples) of the mean F1.         |
| `per_regime_anova.csv`    | One-way ANOVA over regimes from the per-phase accept rates.             |
| `summary.json`            | Aggregated summary of the full suite.                                   |

---

## Hick-Hyman validation on ATC logs

Section 6.5 of the paper validates that PoD's coupling signal
corresponds to a property of *real* deliberation rather than an
artefact of the simulator. The validation uses the
[`Jzuluaga/atco2_corpus_1h`](https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h)
and
[`Jzuluaga/uwb_atcc`](https://huggingface.co/datasets/Jzuluaga/uwb_atcc)
corpora from HuggingFace.

```bash
pip install -e ".[validation]"
pod-validate --out hick_hyman_pod_validation_v2.png
```

The pipeline (i) loads ~3,400 pilot-controller pairs, (ii) computes
out-of-fold predictive entropies with a TF-IDF + isotonic-calibrated
logistic regression averaged over 3 seeds, (iii) applies post-hoc
temperature scaling, (iv) runs Spearman tests against controller
response delay with linguistic-proxy contrasts, and (v) renders the
summary figure used in the paper.

---

## Real-user labeling study

The simulator results are complemented by a real-user labeling study
that complements the synthetic operator model.
Participants label elec2 trials in a browser-based app across three
induced regimes (baseline, speed-bonus, long-block) while the app
records millisecond-precision response times. The collected CSVs are
analysed offline through the same PoD verification primitives used in
the simulator pipeline.

```bash
# 1) Build the frozen task pool (once per study).
python -m pod.realdata.build_pool --out PoD_code/labeling/tasks.json

# 2) Deploy the labeling app (any static server; or open the HTML directly).
cd PoD_code/labeling && python -m http.server 8000

# 3) Collect participant CSVs in a directory, then analyse.
pod-realdata --in data_real/ --out out_real/
```

The end-to-end protocol -- consent text, exclusion criteria, statistical
tests, and limitations to declare in the paper -- is documented in
[`docs/REAL_DATA_PROTOCOL.md`](docs/REAL_DATA_PROTOCOL.md). The labeling
app at [`PoD_code/labeling/app.html`](PoD_code/labeling/app.html) is a
single-file, offline page: no analytics, no telemetry, no network
calls.

---

## Development

```bash
pip install -e ".[dev]"

# Tests (unit + integration smoke)
pytest                 # ~10 s

# Coverage
pytest --cov=pod --cov-report=term-missing

# Lint and format checks
ruff check src tests

# Static type checking
mypy src/pod
```

The [`tests/`](tests/) directory covers PoD primitives, the operator
simulator, stream generation, the comparison baselines, and a tiny
end-to-end run for every method.

---

## Real-time PoD dashboard

The [`PoD_code/`](PoD_code/) directory contains a single-page,
browser-based MQTT dashboard that monitors the PoD gate live on a
deployed sensor stream. Configure your broker in
[`PoD_code/config.json`](PoD_code/config.json) and open
[`PoD_code/app.html`](PoD_code/app.html) directly in a browser. The
default broker (`broker.emqx.io`) is a public test broker; replace it
with your own for production deployments.

---

## Citation

If you use this software, please cite both the paper and the software
artefact:

```bibtex
@article{MartinezGil2026PoD,
    author    = {Jorge Martinez-Gil},
    title     = {Proof-of-Deliberation for Certifying Human Supervision
                 Reliability in Streaming Data Quality},
    journal   = {IEEE Transactions on Knowledge and Data Engineering},
    year      = {2026},
    publisher = {IEEE}
}

@software{MartinezGil2026PoDSoftware,
    author  = {Jorge Martinez-Gil},
    title   = {Proof-of-Deliberation (PoD): reference implementation},
    year    = {2026},
    version = {1.0.0},
    url     = {https://github.com/jorge-martinez-gil/proof-of-deliberation}
}
```

The full citation metadata in `CITATION.cff` is GitHub-renderable; click
the "Cite this repository" button on the project page to copy a BibTeX
or APA entry.

---

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

- **OpenML** for the Electricity (elec2, data id 44156), Forest CoverType
  (data id 1596), and Airlines (data id 1169) benchmarks.
- **UCI Machine Learning Repository** for the Gas Sensor Array Drift
  dataset (dataset #224).
- **HuggingFace** and **Zuluaga-Gomez et al.** for the
  `atco2_corpus_1h` and `uwb_atcc` ATC corpora.
- The **scikit-learn**, **NumPy**, **pandas**, **Matplotlib**, and
  **SciPy** projects for the scientific computing stack.
