# Real-User Data Collection Protocol

This document describes the protocol used to gather the empirical
human-deliberation evidence reported in the paper's real-data section.
It complements the simulator-based experiments by demonstrating that the
PoD signal -- difficulty-conditioned response-time coupling and
vigilance breakdown -- is recoverable from actual labelers, not only
from the Gaussian operator model.

## 1. Study design

**Type.** Within-subject, three-block induced-condition study on a
binary classification task (Electricity Market UP/DOWN).

**Why within-subject.** Each participant contributes data to all three
PoD regimes (baseline, gaming, fatigue), eliminating between-participant
variance from the per-block contrasts and increasing statistical power
at modest sample size.

**Why elec2.** Using the same dataset that drives the simulator
experiments means real vs. simulated comparisons share an identical
feature distribution and an identical complexity estimator (calibrated
logistic-regression entropy). Any difference between simulator and human
results is attributable to the operator process, not to a covariate
shift.

## 2. Participants

| Parameter            | Recommended value                                        |
|----------------------|----------------------------------------------------------|
| Sample size          | n >= 20 for the main analysis; n >= 30 to power the per-block contrasts at alpha=0.05. |
| Eligibility          | Adults (>=18). No domain expertise required.             |
| Recruitment          | Prolific / lab pool / convenience sample acceptable.     |
| Compensation         | Local minimum wage equivalent (~12-15 min session).      |
| Exclusion (a priori) | < 60 valid trials, mean baseline accuracy < 55% (random-guess threshold), session duration > 30 min (suggesting interruption). |

## 3. Materials

**Task pool**. A frozen JSON file
(`PoD_code/labeling/tasks.json`) containing 120 elec2 trials sampled
stratified by complexity (40 low / 40 mid / 40 high). Reproduce with:

```bash
python -m pod.realdata.build_pool --out PoD_code/labeling/tasks.json --seed 20260529
```

The seed pins the sampling so every participant sees an identical pool
in identical block-internal random order.

**Labeling app**. `PoD_code/labeling/app.html` -- single-file, offline,
no telemetry. Records `performance.now()` timestamps for sub-millisecond
response-time precision.

## 4. Procedure

Each session lasts ~12-15 minutes and follows this sequence:

1. **Information statement and consent** (in-app screen). Participant
   confirms they are an adult and consents to the data use described
   below.
2. **Demographics**: age bracket and prior labeling experience only.
3. **Instructions** plus 5 practice trials with feedback. Practice data
   is excluded from analysis.
4. **Block 1 -- Baseline** (~40 trials). Instruction: *"Please label
   each item as accurately as you can. Take the time you need; there is
   no time pressure for this block."*
5. **Block 2 -- Speed bonus** (~40 trials). Instruction: *"In this block
   we are interested in fast responses. Try to answer each item as
   quickly as you can while staying reasonable. Faster answers are
   preferred."* (Induces gaming-like fast-and-near-constant responses.)
6. **Block 3 -- Long block, no breaks** (~40 trials). Instruction:
   *"This is the final and longest block. Please complete it without
   taking a break. Focus is important; do your best."* (Induces
   fatigue-like high-variance responses.)
7. **End screen**. Participant clicks "Download session CSV" and emails
   or uploads the resulting file to the experimenter.

## 5. Data handling

**Collected**

- Anonymous UUIDv4 participant id (generated client-side).
- Per-trial response (UP/DOWN), correctness, and millisecond response
  time.
- Self-reported age bracket and labeling experience.

**Not collected**

- Name, email, IP address, browser fingerprint, OS, geolocation, or any
  other personal identifier.

**Storage**. CSVs are stored only on the participant's machine until
they choose to share. The experimenter retains them under
`data_real/<study>/` in a single directory.

**Retention.** Indefinite for replication purposes; CSVs contain no
PII so the GDPR / Article 5(1)(c) tests are trivially satisfied.

## 6. Analysis pipeline

```bash
pod-realdata --in data_real/ --out out_real/
```

Produces, under `out_real/`:

| Artefact                                | Contents                                                       |
|-----------------------------------------|----------------------------------------------------------------|
| `per_participant_block_summary.csv`     | Per-participant per-block accuracy, mean / median response time, CV. |
| `coupling_per_participant.csv`          | Per-participant per-block Spearman rho(c_star, delib_ms).      |
| `per_trial_with_pod.csv`                | Per-trial frame augmented with PoD's gate / coupling / vigilance / accept columns. |
| `block_regime_table.csv`                | Aggregate per-condition rates (the headline real-data table).  |
| `regime_classification_score.json`      | Aggregate score for "does PoD favour baseline over induced bad regimes?". |
| `fig_block_accept_rate.{pdf,png}`       | Bar chart of PoD accept rate by induced condition.             |
| `fig_coupling_by_regime.{pdf,png}`      | Box plot of per-participant rho by induced condition.          |
| `summary.json`                          | Aggregated summary for the paper.                              |

## 7. Statistical tests for the paper

1. **Baseline Hick-Hyman coupling**. Sign test on per-participant
   Spearman rho values in the baseline block: H_0: median rho = 0;
   one-sided alternative rho > 0.
2. **Vigilance contrast**. Paired Wilcoxon signed-rank on PoD accept
   rate between baseline and gaming blocks, baseline and fatigue blocks.
3. **Coupling breakdown**. Paired Wilcoxon signed-rank on per-participant
   rho between baseline and gaming/fatigue blocks (expected: rho drops).
4. **Effect sizes**. Cohen's d for every paired contrast.

All four are produced automatically by `pod-realdata` (see CSV outputs).

## 8. Limitations to declare in the paper

- Induced gaming and fatigue are *instructed* rather than spontaneous.
  Real adversarial behaviour may differ in subtle ways; the paper should
  state this explicitly.
- Web-based collection introduces minor measurement noise; the practical
  floor for `performance.now()` is ~0.1 ms in modern browsers.
- Convenience samples are not representative of trained domain experts;
  the paper should report demographic composition.
- The Electricity Market is unfamiliar to most participants; absolute
  accuracy is bounded by domain ignorance even in the baseline block.

## 9. Reproducibility checklist

- [ ] Build task pool with the seed declared above.
- [ ] Hash the pool file (`sha256sum tasks.json`) and report in the paper.
- [ ] Collect CSVs in a single directory.
- [ ] Run `pod-realdata` with default parameters.
- [ ] Archive `out_real/` alongside the raw CSVs.
- [ ] Report the PoD parameters from `out_real/summary.json` in the paper.
