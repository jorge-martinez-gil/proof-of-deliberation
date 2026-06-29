# PoD Real-User Labeling App

A single-file, offline web app that collects per-trial response times and
labels from real participants on the Electricity (elec2) UP/DOWN task.
The data it produces is the empirical evidence reported in the paper's
real-data section, complementing the stochastic operator simulator with
real human-interaction evidence.

## Quick start

1. **Build the frozen task pool** (once per study):

   ```bash
   pip install -e .
   python -m pod.realdata.build_pool --out PoD_code/labeling/tasks.json
   ```

   The default settings produce 120 trials (3 complexity bins x 40 each)
   suitable for a ~12-minute session.

2. **Deploy the app**. Open `PoD_code/labeling/app.html` directly in a
   modern browser, or serve the `PoD_code/labeling/` directory from any
   static file server:

   ```bash
   cd PoD_code/labeling
   python -m http.server 8000
   ```

   then point participants at `http://localhost:8000/app.html`.

3. **Collect the CSVs**. At the end of every session the participant
   clicks "Download session CSV" and the browser saves a file named
   `pod_session_<participant>_<iso>.csv` to their Downloads folder.
   Collect them into a single directory (e.g. `data_real/`).

4. **Run the analysis**:

   ```bash
   pod-realdata --in data_real --out out_real
   ```

   This writes per-participant summaries, per-block PoD acceptance rates,
   coupling tables, and figures under `out_real/`. See
   `docs/REAL_DATA_PROTOCOL.md` for the protocol document.

## Protocol summary

Each session is structured as:

1. **Information statement and consent** (no PII collected).
2. **Demographics** (age bracket and prior labeling experience only).
3. **Instructions** and five practice trials with feedback.
4. **Three blocks** of ~40 trials, each preceded by tailored instructions:
   - *Baseline*: accurate labeling encouraged, no time pressure.
   - *Speed-bonus*: fast responses encouraged (induces gaming).
   - *Long block, no breaks*: longest block with explicit no-break
     instruction (induces fatigue).
5. **End-of-session CSV download**.

The induced-condition design lets the paper test PoD's vigilance signal
*within subjects*: every participant produces examples of all three
regimes.

## CSV schema

| Column                  | Description                                                  |
|-------------------------|--------------------------------------------------------------|
| `participant_id`        | Anonymous UUIDv4 generated client-side.                      |
| `session_id`            | Same as `participant_id` for now.                            |
| `csv_version`           | Schema version (currently 1).                                |
| `block`                 | `baseline` / `speed_bonus` / `long_block`.                   |
| `induced_condition`     | `baseline` / `gaming` / `fatigue`.                           |
| `trial_idx`             | 0-based trial index within the session.                      |
| `item_id`               | Pool item index from `tasks.json`.                           |
| `c_star`                | Classifier-estimated complexity in [0,1].                    |
| `complexity_bin`        | `low` / `mid` / `high`.                                      |
| `y_true`                | Ground-truth class (0 = DOWN, 1 = UP).                       |
| `y_user`                | Participant's label.                                         |
| `correct`               | 1 if `y_user == y_true`, else 0.                             |
| `delib_ms`              | Response time in milliseconds (high-resolution).             |
| `t_shown_perf`          | `performance.now()` at stimulus onset.                       |
| `t_clicked_perf`        | `performance.now()` at click.                                |
| `t_shown_iso`           | ISO-8601 wall-clock at stimulus onset.                       |
| `age_bracket`           | Self-reported age bracket.                                   |
| `experience`            | Self-reported labeler experience.                            |
| `started_at_iso`        | Session start wall-clock timestamp.                          |

## Privacy

- No network calls leave the browser. No analytics, no telemetry.
- No name, email, IP address, or browser fingerprint is recorded.
- The participant chooses whether to share the downloaded CSV.
- See the Information Statement screen inside the app for the full
  consent text.
