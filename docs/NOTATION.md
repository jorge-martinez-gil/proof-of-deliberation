# Notation -- Symbol to Code Mapping

This document maps every symbol used in the paper to its concrete
implementation in the `pod` package. It is the canonical reference for
reviewers and reimplementers verifying that the code matches the
formal description.

## Streaming model (Section 3)

| Symbol            | Meaning                                                  | Location                                                 |
|-------------------|----------------------------------------------------------|----------------------------------------------------------|
| `(x_t)`           | Stream of feature vectors                                | `X_stream` argument of `pod.experiment.run_stream_experiment_once` |
| `D_t`             | Time-varying data distribution                           | Realised by the stream loaders in `pod.streams`          |
| `y_t`             | True label (unobserved)                                  | `y_stream` argument                                      |
| `tilde y_t`       | Operator label                                           | First element returned by `pod.operator.simulate_operator` |
| `Delta_t`         | Deliberation time (ms)                                   | Second element returned by `pod.operator.simulate_operator` |
| `eta_t`           | Operator instantaneous error rate                        | Implicit; `1 - p_correct_<phase>` in `pod.config.OperatorParams` |
| `O_t`             | Operator internal state (regime tag)                     | Output of `pod.config.RegimeSchedule.phase`              |
| `M_theta`         | Learner                                                  | `sklearn.linear_model.SGDClassifier` via `pod.learner.make_classifier` |
| `q_t`             | Per-step Bernoulli query probability                     | `pod.experiment.query_probability`                       |

## PoD verification (Section 4)

| Symbol                       | Paper definition                                                  | Code                                                       |
|------------------------------|-------------------------------------------------------------------|------------------------------------------------------------|
| `C_t`                        | Task complexity                                                   | Computed inline in `run_stream_experiment_once` (entropy) or supplied as `c_star_stream` |
| `hat Delta(C)`               | Difficulty-adjusted expected deliberation time                    | `pod.core.expected_delib`                                  |
| `tau_min(C)`, `tau_max(C)`   | Lower / upper edges of the admission window                       | Implicit in `pod.core.gate_check` (built from `PoDParams.gate_*`) |
| `V_t`                        | Instantaneous deliberation gate (Section 4.3, Eq. 6)              | `pod.core.gate_check`                                      |
| `rho_cog(t)`                 | Cognitive coupling coefficient (Section 4.1, Eq. 3)               | `pod.core.coupling_check` (Spearman over the most recent `coupling_window` observations) |
| `CV_w`                       | Coefficient of variation over window `w` (Section 4.2, Eq. 4)     | `pod.utils.coefficient_of_variation`                       |
| `S_vig(t)`                   | Multi-scale vigilance indicator (Section 4.2, Eq. 5)              | Conjunction of `pod.core.gaming_detector` and `pod.core.fatigue_detector` |
| `epsilon`                    | Minimum admissible coupling                                       | `PoDParams.coupling_epsilon`                               |
| `V(pi_t)`                    | Composite verification (Section 4.4, Eq. 7)                       | The boolean composition inside `pod.experiment.run_stream_experiment_once`, branch `method == "PoD"` |

## Operator simulator (Section 6.3)

| Regime     | Parameters                                                                | Code                                                       |
|------------|---------------------------------------------------------------------------|------------------------------------------------------------|
| Baseline   | `Delta ~ N(k C + c, sigma)`; correct with prob. `p_correct_baseline`      | `OperatorParams.k`, `c`, `sigma`, `p_correct_baseline`     |
| Gaming     | `Delta ~ N(c_fast, eps_fast)`; correct with prob. `p_correct_gaming`      | `OperatorParams.c_fast`, `eps_fast`, `p_correct_gaming`    |
| Fatigue    | `Delta ~ N(c_slow, sigma_high)`; correct with prob. `p_correct_fatigue`   | `OperatorParams.c_slow`, `sigma_high`, `p_correct_fatigue` |

A 50 ms physiological floor is enforced on `Delta` in
`pod.operator.simulate_operator`.

## Synthetic stream (Section 6.1)

| Symbol                        | Code                                                                                |
|-------------------------------|-------------------------------------------------------------------------------------|
| `d`                           | `SynthParams.d`                                                                     |
| `w_t`                         | Boundary normal computed in `pod.streams.synth.generate_synth_boundary_pool_regime` |
| `lambda` (complexity decay)   | `SynthParams.lambda_complexity`                                                     |
| `C_t^* = exp(-lambda |w.x|)`  | Returned as `c_star` from the synth generator                                       |

## Regime schedule

| Quantity            | Code                              |
|---------------------|-----------------------------------|
| Baseline length     | `RegimeSchedule.baseline`         |
| Gaming length       | `RegimeSchedule.gaming`           |
| Fatigue length      | `RegimeSchedule.fatigue`          |
| Total stream length | `RegimeSchedule.total()`          |
| Phase dispatcher    | `RegimeSchedule.phase(t_rel)`     |
