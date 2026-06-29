# Architecture

This document explains *why* the code is organised the way it is, so
that future maintainers and reviewers can extend the package without
losing the design properties the paper depends on.

## Design goals

1. **Numerical equivalence with the reference script.** The
   [archived `pod-unified.py`](../archive/) is the de-facto ground truth.
   The modular package preserves its behaviour byte-for-byte; refactoring
   was strictly *organisational*, not algorithmic. The smoke test in
   `tests/test_integration.py` and the per-primitive unit tests guard
   against regressions.
2. **Reviewability.** Each module corresponds to a section of the paper
   (`core` <-> Section 4, `operator` <-> Section 6.3, ...). A reviewer
   should be able to land in any source file and read it as a
   self-contained, documented translation of the corresponding text.
3. **Reproducibility.** Configuration lives only in frozen dataclasses
   (`pod.config`) and JSON snapshots (`configs/`). The seed scheme is
   defined in a single place (`pod.experiment.run_suite_generic`) and
   documented in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
4. **Extensibility.** Adding a new baseline, a new stream, or a new
   verification check should be a localised change. The four baselines
   live in `pod.baselines`; new streams plug into `pod.streams` via a
   single ``run_data_fn`` callable consumed by ``run_suite_generic``.

## Module dependency graph

```
  config -+-> core ---+-> experiment ---+-> cli.experiments
          +-> operator|                 +-> presets <-+
          |           |                 |             |
          |           +-----+           |             |
          +-> streams       +-> baselines             |
          |   +-- synth                               |
          |   +-- openml                              |
          |   +-- gas_drift                           |
          |                                           |
          +-> utils <----------- (everywhere)         |
                                                      |
  validation/* ---------------------------> cli.validation
```

* `config` has no internal dependencies and is safe to import from
  anywhere.
* `utils` only depends on the scientific stack (numpy, pandas, scipy).
* `core` consumes `config` and `utils`; it has no dependency on the
  experiment runner, the operator, or the streams. This is what allows
  the unit tests in `tests/test_core.py` to exercise it in isolation.
* `experiment` is the only module that imports from `baselines`,
  `core`, `learner`, `operator`, `streams`, `utils`, and `viz`. It is
  intentionally the integration point.
* `cli.experiments` is a thin wrapper over `experiment` plus
  `presets`; it never reaches into private symbols.
* `validation/*` is fully independent of the streaming pipeline.

## Why frozen dataclasses?

Every hyperparameter container in `pod.config` is declared with
`@dataclass(frozen=True)` for two reasons:

* **Hashability** lets reviewers cache experiment results keyed by
  config without surprises.
* **Immutability** prevents accidental in-place mutation between runs,
  which would silently violate the reproducibility guarantee.

When you genuinely need a modified configuration, build a new instance
via `dataclasses.replace`. The CLI does exactly this when merging
preset values with command-line overrides.

## Why a single `run_stream_experiment_once`?

The closed-loop runner is the only function whose numerical behaviour
defines the paper's results. Splitting it further would create
multiple integration points and increase the surface for subtle
divergence. The runner *is* commented section-by-section so it can be
read top-to-bottom as a literal translation of Algorithm description in
the paper.

## Adding a new dataset

1. Add a loader in `pod.streams.<name>` that exposes a public
   `load_<name>(...)` returning the raw arrays.
2. Add a preset function `cfg_<name>()` in `pod.presets` and a matching
   `configs/<name>.json`.
3. Add a dispatcher branch `_run_<name>(args)` in
   `pod.cli.experiments` that wires the loader + preset into
   `run_suite_generic`.
4. Add unit tests under `tests/test_streams.py` and, if the dataset
   has non-trivial preprocessing, a smoke test in
   `tests/test_integration.py`.

## Adding a new acceptance method

1. Implement the decision function in `pod.baselines`.
2. Add a `method == "<Name>"` branch inside
   `pod.experiment.run_stream_experiment_once`.
3. Extend the `METHODS` tuple in `pod.experiment`.
4. Add a colour to `pod.viz.METHOD_COLORS`.
5. Add a unit test under `tests/test_baselines.py` and parametrise
   `tests/test_integration.py::test_method_runs_end_to_end` over the
   new method.

## Anti-patterns to avoid

* **Do not** import from `archive/` at runtime. The archive exists for
  historical transparency, not for execution.
* **Do not** introduce module-level state. The PoD package is
  intentionally functional: every run takes its randomness through an
  explicit `numpy.random.Generator`.
* **Do not** silently change a value in `presets.py` without updating
  the matching JSON in `configs/` and noting the change in
  `CHANGELOG.md`. The two locations are kept in sync deliberately.
