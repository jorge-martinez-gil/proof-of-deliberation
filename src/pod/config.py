"""
Configuration dataclasses for the Proof-of-Deliberation protocol.

All hyperparameters that govern the operator simulator, the active-learning
query policy, the PoD verification layers, and the synthetic stream
generator are collected here as *frozen* dataclasses. Freezing guarantees
hashability and prevents accidental in-place mutation between runs --- a
property the paper relies on for the reproducibility claims in
Section 6 (Empirical Evaluation).

Notation aligns with the manuscript:

============  =======================================  =====================
Symbol         Meaning                                   Field
============  =======================================  =====================
:math:`q_t`    per-step query probability                ``ALParams``
:math:`\\rho_{cog}`  cognitive coupling coefficient     ``PoDParams.coupling_*``
:math:`S_{vig}` multi-scale vigilance indicator         ``PoDParams.fatigue_*`` / ``gaming_*``
:math:`V_t`    instantaneous deliberation gate          ``PoDParams.gate_*``
:math:`\\eta_t`  operator error rate                    ``OperatorParams.p_correct_*``
:math:`\\Delta_t` deliberation time (ms)                operator simulator output
============  =======================================  =====================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeSchedule:
    """Length (in stream steps) of each of the three operator regimes.

    The paper's degradation scenario is a fixed sequence

    ``Baseline -> Gaming -> Fatigue``

    with equal-length segments by default. ``phase(t_rel)`` returns the
    regime active at relative step ``t_rel``.

    Parameters
    ----------
    baseline, gaming, fatigue : int
        Number of stream steps allocated to each regime.
    """

    baseline: int
    gaming: int
    fatigue: int

    def total(self) -> int:
        """Total stream length, summed across all three regimes."""
        return self.baseline + self.gaming + self.fatigue

    def phase(self, t_rel: int) -> str:
        """Return the regime name for relative step ``t_rel``."""
        if t_rel < self.baseline:
            return "baseline"
        if t_rel < self.baseline + self.gaming:
            return "gaming"
        return "fatigue"


@dataclass(frozen=True)
class ALParams:
    """Active-learning query policy parameters.

    Implements the temperature-scaled uncertainty sampler used as the
    common query policy across all baselines and PoD itself.

    Parameters
    ----------
    query_budget : float
        Target fraction of stream steps for which a label is queried,
        evaluated against the per-step Bernoulli draw.
    query_temp : float
        Logistic temperature applied to the centred complexity value.
        Smaller values produce sharper (more selective) queries.
    """

    query_budget: float = 0.22
    query_temp: float = 0.28


@dataclass(frozen=True)
class OperatorParams:
    """Stochastic operator simulator parameters.

    Encodes the three psychophysical regimes used in the controlled
    experiments. Means and standard deviations are in milliseconds and
    follow distributions consistent with the cognitive ergonomics
    literature cited in the paper (Section 6.3).

    Parameters
    ----------
    k, c, sigma : float
        Baseline regime: ``Delta ~ N(k * complexity + c, sigma)``.
    c_fast, eps_fast : float
        Gaming regime: ``Delta ~ N(c_fast, eps_fast)``; complexity-independent.
    c_slow, sigma_high : float
        Fatigue regime: ``Delta ~ N(c_slow, sigma_high)``; high variance.
    p_correct_baseline, p_correct_gaming, p_correct_fatigue : float
        Probability that the operator returns the true label in each
        regime. Setting any to ``-1`` activates adversarial mode (always
        wrong) for stress testing.
    """

    # Baseline
    k: float = 650.0
    c: float = 270.0
    sigma: float = 88.0

    # Gaming
    c_fast: float = 600.0
    eps_fast: float = 25.0

    # Fatigue
    c_slow: float = 925.0
    sigma_high: float = 675.0

    # Correctness
    p_correct_baseline: float = 0.965
    p_correct_gaming: float = 0.22
    p_correct_fatigue: float = 0.45


@dataclass(frozen=True)
class PoDParams:
    """Proof-of-Deliberation verification-layer parameters.

    Three independent checks gate every queried label:

    1. **Authenticity** (``coupling_*``) -- Spearman rho between
       deliberation time and predictive entropy over a sliding window.
    2. **Stability** (``gaming_*``, ``fatigue_*``) -- coefficient-of-variation
       bounds on recent response times across short and long windows.
    3. **Gatekeeping** (``gate_*``) -- per-label admissibility window
       around the difficulty-adjusted expected deliberation time.

    Parameters
    ----------
    coupling_window : int
        Length of the sliding window over which Spearman rho is computed.
    coupling_epsilon : float
        Minimum rho required to pass the authenticity check. Setting to
        a value <= -1 disables the check (used for some real-data configs).
    gate_a, gate_b : float
        Linear coefficients of the expected-deliberation function
        ``mu(C) = gate_a * C + gate_b``.
    gate_lo_frac, gate_hi_frac : float
        Lower/upper relative tolerance of the admission window
        (``mu * (1 - lo_frac)`` to ``mu * (1 + hi_frac)``).
    gate_floor_ms, gate_ceil_ms : float
        Absolute floor/ceiling clamps on the admission window in ms.
    gaming_window, gaming_mu_max_ms, gaming_cv_max : float
        Short-window CV/mean bounds that flag mechanical/robotic clicking.
    fatigue_window, fatigue_cv_min : float
        Long-window CV bound that flags loss of attentional control.
    persist_k : int
        Number of *consecutive* violations of any check required before a
        label is actually rejected; smooths over single-step noise.
    """

    coupling_window: int = 120
    coupling_epsilon: float = 0.15

    gate_a: float = 650.0
    gate_b: float = 245.0
    gate_lo_frac: float = 0.58
    gate_hi_frac: float = 0.86
    gate_floor_ms: float = 120.0
    gate_ceil_ms: float = 3200.0

    gaming_window: int = 80
    gaming_mu_max_ms: float = 310.0
    gaming_cv_max: float = 0.06

    fatigue_window: int = 80
    fatigue_cv_min: float = 0.27

    persist_k: int = 3


@dataclass(frozen=True)
class SynthParams:
    """Synth-Boundary stream-generator parameters.

    Parameters
    ----------
    d : int
        Feature dimensionality.
    lambda_complexity : float
        Decay rate of the intrinsic-complexity function
        ``C* = exp(-lambda * |w . x|)`` (Eq. in Section 6.1).
    rotation_per_step : float
        Default per-step angular rotation of the separating hyperplane
        during the Baseline regime.
    noise_std : float
        Standard deviation of the feature-vector Gaussian generator.
    """

    d: int = 10
    lambda_complexity: float = 3.0
    rotation_per_step: float = 0.004
    noise_std: float = 1.0
