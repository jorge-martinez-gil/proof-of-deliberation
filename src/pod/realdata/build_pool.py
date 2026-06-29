"""
Freeze a stratified, complexity-annotated task pool for the labeling app.

The labeling app (``PoD_code/labeling/app.html``) reads a JSON pool of
ready-to-render trials. Each trial bundles:

* a small, human-readable feature snapshot,
* the ground-truth label (UP/DOWN),
* the task complexity ``c_star`` estimated as a calibrated logistic-regression
  predictive entropy on the row,
* a complexity stratum (``low`` / ``mid`` / ``high``) used to sample a
  balanced pool across difficulty levels.

The script is deterministic: identical OpenML id, identical seed, and
identical scikit-learn version reproduce the pool bit-for-bit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pod.streams import OpenMLSpec, load_openml_any
from pod.utils import ensure_dir

# The elec2 columns we *display* to participants. Other columns are
# still used to compute c_star but are hidden from the UI to keep the
# stimulus readable.
ELEC2_VISIBLE_COLUMNS: Tuple[str, ...] = (
    "day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer",
)
"""Human-readable elec2 features shown in the labeling app's UI."""

ELEC2_FEATURE_NAMES: Dict[str, str] = {
    "day": "Day of week (1-7)",
    "period": "Half-hour period (1-48)",
    "nswprice": "NSW electricity price (norm.)",
    "nswdemand": "NSW demand (norm.)",
    "vicprice": "Victoria price (norm.)",
    "vicdemand": "Victoria demand (norm.)",
    "transfer": "Inter-state transfer (norm.)",
}
"""Display labels for each elec2 feature."""


def _predictive_entropy_unit(probs: np.ndarray) -> np.ndarray:
    """Row-wise binary entropy normalised to ``[0, 1]``."""
    p = np.clip(probs[:, 1], 1e-12, 1.0 - 1e-12)
    H = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return H / math.log(2.0)


def _stratify_by_quantile(
    c_star: np.ndarray, n_bins: int = 3
) -> Tuple[np.ndarray, List[str]]:
    """Map a continuous complexity vector onto ``n_bins`` quantile strata."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    cuts = np.quantile(c_star, qs)
    bin_idx = np.searchsorted(cuts, c_star, side="right")
    names = ["low", "mid", "high"][:n_bins] if n_bins == 3 else [
        f"q{i+1}" for i in range(n_bins)
    ]
    return bin_idx, names


def _stratified_sample(
    bin_idx: np.ndarray,
    n_per_bin: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample exactly ``n_per_bin`` indices from each bin without replacement."""
    selected: List[int] = []
    for b in sorted(set(bin_idx.tolist())):
        members = np.where(bin_idx == b)[0]
        if len(members) < n_per_bin:
            raise ValueError(
                f"Not enough items in complexity bin {b}: "
                f"need {n_per_bin}, got {len(members)}"
            )
        pick = rng.choice(members, size=n_per_bin, replace=False)
        selected.extend(pick.tolist())
    arr = np.array(selected, dtype=int)
    rng.shuffle(arr)
    return arr


def build_elec2_pool(
    openml_id: int,
    seed: int,
    n_per_bin: int,
    n_bins: int = 3,
    n_fit_rows: int = 5000,
) -> Dict[str, Any]:
    """Build a complexity-stratified pool of elec2 trials.

    Parameters
    ----------
    openml_id : int
        OpenML data id for Electricity. The reference value is 44156.
    seed : int
        Seed for the stratified sampler. The classifier fit is itself
        deterministic.
    n_per_bin : int
        Number of trials drawn from each complexity stratum.
    n_bins : int, default 3
        Number of complexity strata (3 = low / mid / high).
    n_fit_rows : int, default 5000
        Number of rows used to fit the complexity classifier. Held out
        rows form the candidate pool.
    """
    spec = OpenMLSpec("elec2", int(openml_id))
    X, y = load_openml_any(spec, seed=seed)

    if len(X) < n_fit_rows + n_per_bin * n_bins + 100:
        raise ValueError(
            f"elec2 too small to build pool of {n_per_bin*n_bins} items "
            f"after a {n_fit_rows}-row fit set; need at least "
            f"{n_fit_rows + n_per_bin * n_bins + 100} rows"
        )

    X_fit, X_pool = X[:n_fit_rows], X[n_fit_rows:]
    y_fit = y[:n_fit_rows]

    imp = SimpleImputer(strategy="mean").fit(X_fit)
    sc = StandardScaler().fit(imp.transform(X_fit))
    base = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(sc.transform(imp.transform(X_fit)), y_fit)

    proba_pool = clf.predict_proba(sc.transform(imp.transform(X_pool)))
    c_star = _predictive_entropy_unit(proba_pool)
    bin_idx, bin_names = _stratify_by_quantile(c_star, n_bins=n_bins)

    rng = np.random.default_rng(seed)
    picked = _stratified_sample(bin_idx, n_per_bin=n_per_bin, rng=rng)

    # We need the *displayable* feature subset. The OpenML pandas frame
    # was already converted to a NumPy matrix by load_openml_any, so we
    # re-fetch the frame here for the visible columns. We pull from the
    # same OpenML data id, keep the same row ordering, and then index by
    # the permutation seed used in load_openml_any (= 123).
    from sklearn.datasets import fetch_openml

    ds = fetch_openml(data_id=int(openml_id), as_frame=True)
    frame = ds.frame
    perm_seed = 123  # mirrors load_openml_any default
    perm = np.random.default_rng(perm_seed).permutation(len(frame))
    frame = frame.iloc[perm].reset_index(drop=True)

    pool_frame = frame.iloc[n_fit_rows:].reset_index(drop=True)
    visible = [c for c in ELEC2_VISIBLE_COLUMNS if c in pool_frame.columns]
    if not visible:
        raise ValueError(
            "None of the expected elec2 feature columns found; "
            "is the OpenML schema unchanged?"
        )

    items: List[Dict[str, Any]] = []
    for ii, idx in enumerate(picked.tolist()):
        row = pool_frame.iloc[idx]
        features = {col: float(row[col]) for col in visible}
        items.append(
            {
                "item_id": int(idx),
                "trial_order": int(ii),
                "features": features,
                "feature_labels": {c: ELEC2_FEATURE_NAMES.get(c, c) for c in visible},
                "y_true": int(y[n_fit_rows + idx]),
                "c_star": float(c_star[idx]),
                "complexity_bin": bin_names[int(bin_idx[idx])],
            }
        )

    return {
        "schema_version": 1,
        "dataset": "elec2",
        "openml_id": int(openml_id),
        "n_items": len(items),
        "n_per_bin": int(n_per_bin),
        "n_bins": int(n_bins),
        "complexity_bin_names": bin_names,
        "visible_columns": list(visible),
        "feature_labels": {c: ELEC2_FEATURE_NAMES.get(c, c) for c in visible},
        "label_space": {"0": "DOWN", "1": "UP"},
        "items": items,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pod-build-pool",
        description=(
            "Build a complexity-stratified labeling pool for the real-user "
            "data-collection experiment described in the paper's real-data "
            "section."
        ),
    )
    parser.add_argument("--openml-id", type=int, default=44156)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--n-per-bin", type=int, default=40,
                        help="40*3=120 trials = ~12 min session (default).")
    parser.add_argument("--n-bins", type=int, default=3)
    parser.add_argument("--n-fit-rows", type=int, default=5000)
    parser.add_argument(
        "--out", type=str,
        default="PoD_code/labeling/tasks.json",
        help="Output path for the JSON pool (default: PoD_code/labeling/tasks.json).",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    pool = build_elec2_pool(
        openml_id=int(args.openml_id),
        seed=int(args.seed),
        n_per_bin=int(args.n_per_bin),
        n_bins=int(args.n_bins),
        n_fit_rows=int(args.n_fit_rows),
    )
    ensure_dir(os.path.dirname(os.path.abspath(args.out)) or ".")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2)
    print(
        f"[pod-build-pool] Wrote {pool['n_items']} trials "
        f"({args.n_per_bin}/bin x {args.n_bins} bins) to {args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
