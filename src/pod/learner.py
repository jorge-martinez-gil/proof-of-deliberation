"""
Base online learner used by every method.

The paper uses an averaged-SGD logistic-regression classifier
(:class:`sklearn.linear_model.SGDClassifier` with ``loss='log_loss'``)
because it (i) supports incremental ``partial_fit`` updates, (ii) is
fast enough for tight closed-loop experiments, and (iii) is the model
under which the bounded-noise stability assumption used in
Proposition 2 of the paper is well established.

This module wraps the classifier together with the standardisation
pipeline so that the experiment runner can stay agnostic to the
learner's mechanics.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import SGDClassifier

from pod.utils import sigmoid, softmax_stable


def make_classifier(
    classes: np.ndarray,
    seed: int,
    *,
    clf_alpha: float = 2e-5,
    clf_eta0: float = 0.01,
    clf_average: bool = True,
    class_weight_dict: Optional[Dict[int, float]] = None,
) -> SGDClassifier:
    """Instantiate the canonical SGD classifier used in the paper.

    The classifier is returned *unfitted*; the caller is responsible for
    calling :meth:`SGDClassifier.partial_fit` on the warm-up batch with
    the explicit ``classes=`` argument.
    """
    kwargs: Dict[str, object] = dict(
        loss="log_loss",
        alpha=float(clf_alpha),
        average=bool(clf_average),
        learning_rate="adaptive",
        eta0=float(clf_eta0),
        random_state=seed,
    )
    if class_weight_dict is not None:
        kwargs["class_weight"] = class_weight_dict
    return SGDClassifier(**kwargs)


def proba_from_decision_function(
    clf: SGDClassifier, X: np.ndarray, n_classes: int
) -> np.ndarray:
    """Probability estimates from a (possibly two-class) SGD classifier.

    ``SGDClassifier(loss='log_loss')`` exposes ``decision_function`` but
    not ``predict_proba`` in older sklearn versions. This helper
    bridges the gap with a numerically stable sigmoid (binary case) or
    softmax (multi-class).
    """
    df = clf.decision_function(X)
    if n_classes <= 2:
        if df.ndim == 2 and df.shape[1] == 1:
            df = df[:, 0]
        p1 = sigmoid(df.reshape(-1, 1))
        p0 = 1.0 - p1
        return np.hstack([p0, p1])
    if df.ndim == 1:
        df = df.reshape(1, -1)
    return softmax_stable(df)
