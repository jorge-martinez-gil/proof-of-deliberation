# Reproducibility

This document describes how to reproduce, byte-for-byte, the numerical
results and figures published in the paper.

## Environment

| Component | Version constraint |
|-----------|--------------------|
| Python    | >= 3.9, < 3.13 |
| numpy     | >= 1.24, < 3.0 |
| pandas    | >= 2.0, < 3.0 |
| matplotlib | >= 3.7, < 4.0 |
| scikit-learn | >= 1.3, < 2.0 |
| scipy     | >= 1.10, < 2.0 |

For the Hick-Hyman validation only:

| Component   | Version constraint |
|-------------|--------------------|
| datasets    | >= 2.14, < 3.0 |
| statsmodels | >= 0.14, < 1.0 |

The exact pins live in [`requirements.txt`](../requirements.txt) and
[`requirements-validation.txt`](../requirements-validation.txt).

## Step-by-step

1. **Clone and create an isolated environment** (recommended):

   ```bash
   git clone https://github.com/jorge-martinez-gil/proof-of-deliberation.git
   cd proof-of-deliberation
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

2. **Install with pinned dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

3. **Reproduce all three streaming figures** (20 runs each):

   ```bash
   pod-experiments --datasets synth,elec2,gas
   ```

   Expected wall-clock time on a modern laptop: **~10-30 minutes**,
   plus a one-time dataset download for Electricity (OpenML) and
   Gas Drift (UCI, ~4 MB).

4. **Reproduce the Hick-Hyman validation figure**:

   ```bash
   pip install -e ".[validation]"
   pod-validate --out hick_hyman_pod_validation_v2.png
   ```

   First run downloads ~30 MB from HuggingFace.

## Seed scheme

Every closed-loop run is driven by a single seed derived deterministically
from the run index:

```python
seed = 1000 * run_idx + 7        # streaming runs
```

For the synth-pool generator, the seed defaults to ``50000`` (override
via ``--synth_seed``). For the OpenML reshuffle, the seed defaults to
``123``. All other randomness flows through ``numpy.random.default_rng``;
no global state is mutated.

The Hick-Hyman OOF estimator uses ``random_state = seed * 17 + 3`` for
the K-fold splitter and ``seed * 100 + fold_i`` for each pipeline.

## Frozen configuration provenance

The exact hyperparameters used in the paper live in two redundant
locations that are kept in sync:

* [`src/pod/presets.py`](../src/pod/presets.py) -- the Python source
  consumed at run time.
* [`configs/{synth,elec2,gas}.json`](../configs/) -- the JSON snapshots
  used to diff against the `config.json` files emitted by
  each run.

To verify a run:

```bash
pod-experiments --datasets synth
diff -u configs/synth.json out_pod_unified/synth_boundary/config.json
```

The diff should contain *no* numeric differences, only the additional
keys ``plot_smooth_w``, ``phase_label_y``, ``plot_tmax``, ``log_tmax``,
and the schedule lengths set via CLI flags.

## Smoke test for CI environments

For continuous-integration pipelines that need a fast (<30 s) check
that the pipeline still runs end-to-end:

```bash
pytest -q tests/test_integration.py
```

This exercises all four methods on a 240-step Synth-Boundary stream.

## Hardware caveat

Bit-exact reproduction across operating systems requires identical
BLAS / LAPACK implementations and identical floating-point modes.
Within the same OS family and the pinned dependency set, runs match to
within 1e-10 absolute F1 difference. Cross-OS runs may differ by up to
1e-6 in individual F1 values; aggregated mean trajectories and the
qualitative conclusions are unaffected.
