# 🧠 Proof-of-Deliberation (PoD)

> **A deliberation-aware gating mechanism for human-in-the-loop active learning.**  
> PoD monitors *how* a human annotator responds — not just *what* they label — and uses response-time dynamics to filter out low-quality annotations caused by gaming or fatigue before they corrupt a streaming classifier.

---

## ✨ Overview

In human-in-the-loop machine learning, label quality depends on whether the annotator truly *deliberates* over each decision. **Proof-of-Deliberation** operationalizes this idea:

1. **Complexity estimation** — the model computes a per-sample complexity score *C*ₜ (predictive entropy or known boundary distance).
2. **Deliberation gating** — the operator's response time *D*ₜ is checked against a complexity-calibrated expected window. Labels whose response time is implausibly short (gaming) or erratically long (fatigue) are **rejected before training**.
3. **Behavioral monitoring** — a sliding-window Spearman coupling test, a gaming detector, and a fatigue detector jointly guard the training loop across three operator *phases*: Baseline → Gaming → Fatigue.

PoD is compared against three baselines across three benchmark datasets, validating both classification performance and theoretical grounding via the **Hick-Hyman Law** on real Air Traffic Control corpora.

---

## 📁 Repository Structure

```
proof-of-deliberation/
│
├── pod-unified.py              # Main experiment runner (Elec2, Gas Drift, Synth-Boundary)
├── pod-unified2.py             # Extended variant with additional configurations
├── pod_elec.py                 # Electricity-dataset-focused experiments
├── pod_unified (1).py          # Alternative unified runner
│
├── hick_hyman_validation.py        # Hick-Hyman law validation (v1)
├── hick_hyman_pod_validation_v2.py # Hick-Hyman validation v2 (OOF + calibration)
├── hick_hyman_pod_validation.png   # Output figure from validation
│
├── PoD_code/
│   ├── app.html        # Real-time MQTT-based PoD dashboard (browser)
│   ├── server.html     # MQTT server configuration page
│   ├── config.json     # MQTT broker + topic configuration
│   ├── data.csv        # Sample sensor data
│   └── logo.png        # PoD project logo
│
└── LICENSE             # MIT License
```

---

## 🔬 Methods Compared

| Method | Description |
|---|---|
| **AL** | Standard active learning — all queried labels are accepted unconditionally |
| **StaticGating** | Accepts labels only when deliberation time exceeds a fixed threshold (510 ms) |
| **AdaptiveGating** | Accepts labels when response time falls within a complexity-scaled window |
| **PoD** *(ours)* | Full gating: adaptive gate + Spearman coupling + gaming/fatigue detection |

---

## 📊 Datasets

| Dataset | Source | Classes | Description |
|---|---|---|---|
| **Elec2** | OpenML (id 44156) | 2 | Electricity demand stream (New South Wales) |
| **UCI224 Gas Drift** | UCI Repository | 6 | Gas sensor array — concept drift over 10 batches |
| **Synth-Boundary** | Generated | 2 | Rotating decision boundary with controlled operator phases |
| **ATCO2 / UWB-ATCC** | HuggingFace | — | Air Traffic Control speech for Hick-Hyman validation |

---

## ⚙️ Requirements

**Python ≥ 3.9** is required.

### Core dependencies (streaming experiments)

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

### Additional dependencies (Hick-Hyman validation)

```bash
pip install datasets statsmodels
```

### Full install (recommended)

```bash
pip install numpy pandas matplotlib scikit-learn scipy datasets statsmodels
```

> **Note:** The Gas Drift dataset (~4 MB zip) is downloaded automatically from the UCI repository on first run and cached in `data_cache_uci224/`.  
> The ATC corpora are downloaded from HuggingFace Hub on first run.

---

## 🚀 Quickstart — Reproducing the Main Experiments

### 1. Clone the repository

```bash
git clone https://github.com/jorge-martinez-gil/proof-of-deliberation.git
cd proof-of-deliberation
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn scipy datasets statsmodels
```

### 3. Run the unified experiment suite

Run all three datasets (Synth-Boundary, Elec2, Gas Drift) with default parameters:

```bash
python pod-unified.py
```

Outputs are written to `out_pod_unified/<dataset>/`:
- `figs/<dataset>_methods_f1.pdf` — F1 performance plot (mean ± 95% CI)
- `figs/<dataset>_methods_f1.png` — same, PNG format
- `runs/<dataset>_<method>_run<N>.csv` — per-run time-series
- `diagnostics.csv` — query/accept rates per phase and method
- `config.json` — full experiment configuration snapshot

### 4. Run a specific dataset only

```bash
# Only the synthetic boundary dataset
python pod-unified.py --datasets synth

# Only the Electricity dataset
python pod-unified.py --datasets elec2

# Only the Gas Drift dataset
python pod-unified.py --datasets gas

# Multiple datasets
python pod-unified.py --datasets synth,elec2
```

### 5. Quick smoke-test (fast run)

For a rapid end-to-end check with fewer samples and runs:

```bash
python pod-unified.py \
  --datasets synth \
  --runs 3 \
  --baseline 500 \
  --gaming 500 \
  --fatigue 500 \
  --synth_pool 5000 \
  --holdout 1000
```

---

## 🔁 Full Reproducibility

### Exact parameters used in the paper

The per-dataset hyperparameters are embedded as `cfg_elec2()`, `cfg_gas()`, and `cfg_synth()` functions inside `pod-unified.py`. They are applied automatically when running the corresponding dataset. The default is **20 independent runs** per dataset.

To reproduce the full results as reported:

```bash
python pod-unified.py --datasets synth,elec2,gas
```

> Expected wall-clock time: ~10–30 minutes depending on hardware (Elec2 and Gas downloads add a one-time overhead).

### Seeding

All random number generation uses `numpy.random.default_rng(seed)` with deterministic seeds derived from the run index (e.g., `seed = 1000 * run_idx + 7`). Results are reproducible across identical hardware and Python/NumPy versions.

### Saved configuration

Every run writes a `config.json` snapshot to the output directory. This can be used to verify or re-run the exact parameter set:

```bash
cat out_pod_unified/elec2/config.json
```

---

## 📐 Hick-Hyman Validation (Theoretical Grounding)

PoD's complexity signal is theoretically grounded in the **Hick-Hyman Law**, which predicts that response time increases monotonically with decision complexity (measured by entropy). We validate this empirically on real ATC speech data.

### Run validation v2 (recommended — OOF + calibrated)

```bash
python hick_hyman_pod_validation_v2.py
```

This script:
1. Downloads the `Jzuluaga/atco2_corpus_1h` and `Jzuluaga/uwb_atcc` corpora from HuggingFace
2. Builds pilot → controller response pairs with measured delays
3. Computes out-of-fold (OOF) model entropy via 5-fold × 3-seed ensemble + isotonic calibration + temperature scaling
4. Runs Spearman ρ tests: model entropy *C*ₜ vs delay, plus linguistic contrast proxies
5. Saves `hick_hyman_pod_validation_v2.png`

### Run validation v1 (baseline)

```bash
python hick_hyman_validation.py
```

---

## 🌐 Real-Time PoD Dashboard

The `PoD_code/` directory contains a browser-based dashboard for monitoring the PoD gate in real time via MQTT.

### Setup

1. Open `PoD_code/config.json` and configure your MQTT broker:

```json
{
    "brokerURL": "wss://broker.emqx.io:8084/mqtt",
    "inputTopic":  "input",
    "outputTopic": "output",
    "inputs":  [{"name": "Temperature", "value": ""}, ...],
    "outputs": [{"name": "Target",      "value": ""}]
}
```

2. Open `PoD_code/app.html` directly in a web browser (no server required).

3. The dashboard connects to the configured MQTT broker and visualizes live sensor inputs and classification outputs.

> The default broker `broker.emqx.io` is a public test broker — **do not use for sensitive data**. Replace with your own broker for production use.

---

## 🧪 Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--datasets` | `synth,elec2,gas` | Comma-separated list of datasets to run |
| `--out` | `out_pod_unified` | Output directory |
| `--runs` | `2` | Number of independent runs (20 for full results) |
| `--baseline` | `2000` | Length of baseline phase (time steps) |
| `--gaming` | `2000` | Length of gaming phase |
| `--fatigue` | `2000` | Length of fatigue phase |
| `--init_fit` | `20` | Initial warm-up samples before streaming begins |
| `--eval_every` | `10` | Evaluate F1 every N time steps |
| `--eval_window` | `600` | Holdout window size for F1 evaluation |
| `--log_tmax` | `0` | Truncate experiment at this time step (0 = disabled) |
| `--query_budget` | `0.50` | Fraction of samples queried from the oracle |
| `--coupling_window` | `20` | Window for Spearman coupling test |
| `--persist_k` | `3` | Consecutive failures before gating activates |
| `--cache_dir` | `data_cache_uci224` | Local cache for UCI Gas Drift data |

---

## 📈 Expected Output

After a successful run you should see files like:

```
out_pod_unified/
  elec2/
    figs/
      elec2_methods_f1.pdf
      elec2_methods_f1.png
    runs/
      elec2_AL_run0.csv
      elec2_PoD_run0.csv
      ...
    diagnostics.csv
    config.json
  uci224_gas_drift/
    ...
  synth_boundary/
    ...
```

Each `*_methods_f1.png` shows F1(t) curves for all four methods across operator phases, with shaded 95% confidence intervals.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- **OpenML** for the Elec2 benchmark dataset
- **UCI ML Repository** for the Gas Sensor Array Drift dataset (dataset #224)
- **HuggingFace / Zuluaga et al.** for the `atco2_corpus_1h` and `uwb_atcc` ATC corpora used in the Hick-Hyman validation
- **scikit-learn**, **NumPy**, **pandas**, **Matplotlib**, and **SciPy** for the scientific computing stack
