"""
Out-of-fold predictive-entropy estimation for the ATC validation.

Implements the three core probability-quality improvements over the
v1 script (see paper Section 6.5):

1. **Out-of-fold prediction** with stratified K-fold cross-validation
   -- every example is scored by a model that never saw it.
2. **Isotonic calibration** via
   :class:`sklearn.calibration.CalibratedClassifierCV` to widen the
   entropy dynamic range.
3. **Post-hoc temperature scaling** to minimise NLL on a held-out tail.
4. **Multi-seed ensembling** for variance reduction.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import optimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from pod.validation.labels import LABELS


def make_pipeline(seed: int = 42) -> Pipeline:
    """TF-IDF (word 1+2-gram, char 3-4-gram) -> isotonically calibrated LR."""
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
        max_features=8_000,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=3,
        sublinear_tf=True,
        norm="l2",
        max_features=4_000,
    )
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])

    base_clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
        class_weight="balanced",
        multi_class="multinomial",
    )
    calibrated = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    return Pipeline([("features", features), ("clf", calibrated)])


def temperature_scale(
    proba_matrix: np.ndarray, labels: np.ndarray, n_val: int = 200
) -> float:
    """Optimise the post-hoc temperature on the last ``n_val`` rows.

    Returns ``T*`` such that ``proba ** (1/T*)`` (re-normalised) has
    minimum negative log-likelihood on the validation slice. The 1.0
    fallback is returned when the matrix is too short.
    """
    if len(proba_matrix) < n_val + 10:
        return 1.0
    val_p = proba_matrix[-n_val:]
    val_y = labels[-n_val:]

    def nll(log_t: np.ndarray) -> float:
        t = np.exp(log_t[0])
        logp = np.log(np.clip(val_p, 1e-12, 1.0)) / t
        logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))
        return -float(logp[np.arange(len(val_y)), val_y].mean())

    res = optimize.minimize(
        nll, x0=[0.0], method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-4},
    )
    T = float(np.exp(res.x[0]))
    print(f"  Temperature scaling: T* = {T:.3f}")
    return T


def apply_temperature(proba_matrix: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature ``T`` to a probability matrix and re-normalise."""
    if abs(T - 1.0) < 1e-4:
        return proba_matrix
    logp = np.log(np.clip(proba_matrix, 1e-12, 1.0)) / T
    logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))
    return np.exp(logp)


def compute_oof_entropies(
    pairs: list,
    n_splits: int = 5,
    n_seeds: int = 3,
    apply_temp_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Shannon entropies via stratified K-fold ensembling.

    Every example is scored by a fresh pipeline trained on the other
    ``K - 1`` folds; this is repeated for ``n_seeds`` random partitions
    and the per-seed probability matrices are averaged before
    temperature scaling.

    Returns
    -------
    entropies : np.ndarray
        Predictive entropies in nats, one per pair.
    delays : np.ndarray
        Controller response delays in seconds, aligned with
        ``entropies``.
    """
    X = np.array([p["x_text"] for p in pairs])
    y = np.array([p["y"] for p in pairs])
    delays = np.array([p["delay"] for p in pairs])

    n = len(X)
    ens_prob = np.zeros((n, len(LABELS)))

    for seed in range(n_seeds):
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed * 17 + 3
        )
        seed_prob = np.zeros((n, len(LABELS)))

        for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            pipe = make_pipeline(seed=seed * 100 + fold_i)
            pipe.fit(X[train_idx], y[train_idx])
            proba = pipe.predict_proba(X[val_idx])

            clf_labels = pipe.named_steps["clf"].classes_
            full_proba = np.zeros((len(val_idx), len(LABELS)))
            for j, cls_id in enumerate(clf_labels):
                if cls_id < len(LABELS):
                    full_proba[:, cls_id] = proba[:, j]
            row_sums = full_proba.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            full_proba /= row_sums

            seed_prob[val_idx] += full_proba

        rs = seed_prob.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        seed_prob /= rs
        ens_prob += seed_prob

    ens_prob /= n_seeds
    rs = ens_prob.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    ens_prob /= rs

    if apply_temp_scale:
        T = temperature_scale(ens_prob, y)
        ens_prob = apply_temperature(ens_prob, T)

    safe = np.clip(ens_prob, 1e-12, 1.0)
    entropies = -(safe * np.log(safe)).sum(axis=1)

    return entropies, delays
