# Changelog

All notable changes to this project are documented in this file. The
format adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [1.2.0] - 2026-05-29

### Added
- Real-user labeling study end-to-end:
  - `src/pod/realdata/build_pool.py` (CLI `pod-build-pool`) freezes a
    complexity-stratified elec2 task pool to JSON, using a calibrated
    logistic-regression predictive-entropy estimate for `c_star`.
  - `PoD_code/labeling/app.html` is a single-file, offline labeling app
    with consent, demographics, instructions, 5 practice trials, and
    three induced blocks (baseline, speed-bonus, long-block) that record
    `performance.now()` response times and download a session CSV at
    the end. No telemetry, no network calls.
  - `src/pod/realdata/loader.py` and `analysis.py` aggregate sessions,
    replay every trial through the PoD verification layer, and produce
    the per-participant / per-block / aggregate tables for the paper.
  - New CLI `pod-realdata` glues the loader, the offline PoD replay, and
    the figure generators together.
  - `docs/REAL_DATA_PROTOCOL.md` documents the consent, exclusion
    criteria, statistical tests, and limitations needed for IRB
    submission and the published paper.

## [1.1.0] - 2026-05-29

### Added
- Two new streaming datasets for evaluation breadth: Forest CoverType
  (OpenML id 1596, 7-class) and Airlines (OpenML id 1169, binary). Frozen
  presets live in `src/pod/presets.py:cfg_covertype` and `cfg_airlines`,
  with byte-matched JSON snapshots in `configs/covertype.json` and
  `configs/airlines.json`. Both are wired into `pod-experiments
  --datasets`.
- New label-quality competitor `WorkerQuality`: a single-annotator,
  online Dawid-Skene-style baseline that fuses the learner's predictive
  distribution with a streaming confusion-matrix estimate to compute
  the posterior probability of the operator's label.
- Three composite-verifier ablation variants -- `PoD-NoGate`,
  `PoD-NoCoupling`, `PoD-NoVigilance` -- exposed as first-class methods
  through a `PoDAblation` flag set, so each layer's marginal
  contribution can be reported without re-running the experiment suite.
- New `pod-stats` console entry point and `src/pod/stats.py` module:
  Friedman omnibus (chi-square + Iman-Davenport F), Nemenyi critical
  difference, Demsar-style CD diagram, pairwise Wilcoxon signed-rank
  with Holm-Bonferroni correction, Cohen's d, percentile bootstrap CIs,
  and per-regime one-way ANOVA on the per-phase accept rates.

### Changed
- `viz.METHOD_COLORS` extended with the WorkerQuality and PoD-ablation
  palette; ablation variants are rendered dashed at reduced opacity so
  the main F1 figures stay readable.
- `experiment.METHODS` extended to the full 8-method panel
  (4 baselines + PoD + 3 ablations); the per-step dispatcher branches
  on a `PoDAblation` flag set instead of hard-coding the full-PoD path.
- README adds the comparison table, the new dataset roster, and the
  `pod-stats` analysis workflow.

## [1.0.0] - 2026-05-28

### Added
- Modular `src/pod/` Python package mirroring the structure of the paper:
  `core` (verification primitives), `operator` (simulator), `streams`
  (Synth-Boundary, OpenML Electricity, UCI 224 Gas Drift), `baselines`
  (AL, StaticGating, AdaptiveGating), `learner` (SGD wrapper),
  `experiment` (closed-loop runner and aggregator), `viz` (publication
  plot style), `presets` (frozen per-dataset hyperparameters), and
  `validation` (Hick-Hyman pipeline on ATC corpora).
- Console-script entry points `pod-experiments` and `pod-validate`,
  installed automatically by `pip install .` and reproducing the
  paper's figures end-to-end.
- `pyproject.toml` with pinned dependency ranges, optional groups for
  validation and development, pytest / ruff / mypy configuration, and
  `[project.scripts]` declarations.
- `requirements.txt`, `requirements-validation.txt`, and
  `requirements-dev.txt` for pip-based installation paths.
- Frozen `configs/*.json` snapshots for the three streaming datasets,
  matching `src/pod/presets.py` byte-for-byte.
- `CITATION.cff` for IEEE TKDE-compliant citation.
- `tests/` directory with unit tests for PoD primitives, operator
  simulator, streams, and an end-to-end smoke integration test.
- `docs/NOTATION.md`, `docs/REPRODUCIBILITY.md`, and
  `docs/ARCHITECTURE.md` covering symbol-to-code mappings, exact
  reproduction commands, and the package's design rationale.

### Changed
- Replaced the legacy single-file `pod-unified.py` runner with a
  documented, type-hinted, NumPy-style-docstring'd modular package.
  The numerical behaviour is preserved exactly.
- README rewritten to reflect the new structure and CLI surface.

### Archived
- `pod-unified2.py`, `pod_elec.py`, `pod_unified (1).py`, and
  `hick_hyman_validation.py` (v1) have been moved to `archive/` and
  must not be used to reproduce the published results. See
  `archive/ARCHIVE.md` for provenance.
