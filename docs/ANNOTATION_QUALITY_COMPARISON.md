# PoD vs. the Annotation-Quality Literature

This document positions Proof-of-Deliberation (PoD) against the four
canonical content-based competitors shipped with v1.3 of the codebase:

| Method            | Lineage                                                  | Family           |
|-------------------|----------------------------------------------------------|------------------|
| `WorkerQuality`   | Dawid & Skene (JRSS-C 1979); single-annotator variant.   | Confusion matrix |
| `Raykar`          | Raykar et al. (JMLR 2010); single-annotator streaming.   | Joint EM         |
| `MACE`            | Hovy et al. (NAACL 2013); beta-binomial competence.      | Spam detector    |
| `IEThresh`        | Donmez & Carbonell (ECML 2008); LCB thresholding.        | Confidence-bound |

All four operate on the **label content** -- they reason about whether
the operator's *label* is likely correct. PoD operates on the
**interaction process** -- it reasons about whether the *deliberation
trace that produced the label* is consistent with engaged human work,
regardless of label content. The two surfaces are complementary, not
substitutive.

## What each method actually tests

| Method            | Decision variable                                         | Assumes redundant labels? | Assumes annotator history? | Assumes ground truth? |
|-------------------|-----------------------------------------------------------|---------------------------|----------------------------|-----------------------|
| `WorkerQuality`   | Posterior $P(y_{\text{true}}=\tilde y\mid x, \tilde y)$ via confusion matrix. | No                  | No                         | No                    |
| `Raykar`          | Same posterior, with *soft EM* confusion-matrix update.   | No                        | No                         | No                    |
| `MACE`            | Posterior mean of operator competence $\hat\theta$.       | No                        | No                         | No                    |
| `IEThresh`        | Hoeffding LCB on running model-vs-annotator agreement.    | No                        | No                         | No                    |
| **`PoD`**         | Three independent process checks on $(\Delta, C)$ traces. | No                        | No                         | No                    |

All five methods fit the single-pass, single-annotator streaming
setting. They differ in the *information channel* they consume.

## Why PoD is not dominated by the others

A content-based annotation-quality method that uses the classifier's
own predictive distribution as the ground-truth proxy has a structural
blind spot: when the classifier is *confidently wrong* and the operator
agrees with the (wrong) prediction, the method ratifies the error. PoD
does not consume the label content at all, so it cannot make this
mistake. Conversely, PoD has a blind spot in the
model-confident-error regime (Section 7 of the paper); the two
families fail under *disjoint* conditions, which is exactly why
combining them is an attractive direction.

## How they fail in the three operator regimes

| Method            | Baseline       | Gaming                                | Fatigue                                  |
|-------------------|----------------|---------------------------------------|------------------------------------------|
| `WorkerQuality`   | Accepts.       | Defeated when label happens to match the model's wrong prior. | Posterior drifts slowly; lag-prone.    |
| `Raykar`          | Accepts.       | Same as `WorkerQuality` plus soft update slows the response. | Same lag, with smoother trajectory.    |
| `MACE`            | Accepts.       | Posterior collapses iff label-vs-MAP disagreement is sustained; partial-spam evasion possible. | Insensitive to RT variance.    |
| `IEThresh`        | Accepts after warm-up. | LCB drops slowly under low-rate disagreement; tail-recovery slow. | Insensitive to RT variance.    |
| **`PoD`**         | Accepts.       | Detected by deliberation gate and short-window vigilance. | Detected by long-window vigilance. |

The complementary failure modes are exactly the point: PoD's three
process checks are *independent* of label content, so they catch
gaming and fatigue patterns whose label-content signature is
indistinguishable from competent labeling.

## Implementation choices

The streaming single-annotator adaptations follow the published
algorithms as closely as the streaming constraint allows. Specifically:

* **WorkerQuality and Raykar** both use the classifier's predictive
  distribution as the ground-truth surrogate; WorkerQuality concentrates
  the confusion-matrix update on the argmax (hard EM), Raykar smears
  evidence across rows weighted by the predictive probability
  (soft EM). The two collapse to the same estimator when the classifier
  is perfectly confident.
* **MACE** uses Beta(1, 1) as the prior on competence, matching the
  original paper's uniform prior choice.
* **IEThresh** uses the Hoeffding LCB rather than the original
  upper-confidence-bound formulation, because we are *rejecting* (not
  selecting) labels; the LCB is the conservative analogue. A 10-trial
  warm-up suppresses the LCB until a non-degenerate sample has
  accumulated.

## How to extend the panel

To add a new annotation-quality baseline:

1. Add a stateful dataclass in `src/pod/baselines.py` with `.fresh()`,
   plus standalone `accept_*` and `update_*` functions matching the
   four existing baselines.
2. Add the method name to `METHODS` in `src/pod/experiment.py` and a
   dispatch branch.
3. Add a colour entry in `viz.METHOD_COLORS`.
4. Add a row to the comparison tables above.
5. Add the citation to `mybib.bib` and the paper's Related Work
   section.

The codebase enforces no other coupling, so an independent baseline
can be added in well under 100 lines.
