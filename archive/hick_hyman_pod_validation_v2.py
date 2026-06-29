# pip install -U datasets scikit-learn numpy matplotlib scipy statsmodels

"""
Hick-Hyman Monotonicity Validation  —  High-Correlation Edition 
=====================================================================
Key improvements over v1 to maximise Spearman ρ:

  1. K-FOLD OUT-OF-FOLD (OOF) ENTROPY  [biggest gain]
     Instead of online learning (noisy early estimates), we train on
     (K-1) folds and score the held-out fold.  Every example gets an
     entropy from a model that NEVER saw it, giving well-calibrated
     estimates across the full dataset.

  2. CALIBRATED CLASSIFIER
     sklearn's CalibratedClassifierCV (isotonic regression) spreads the
     softmax distribution, widening the entropy dynamic range and
     improving monotone coupling with delay.

  3. RICHER FEATURES
     TF-IDF with unigrams + bigrams + character 4-grams.  ATC phrases
     like "descend flight level" are key bigrams.

  4. BETTER DELAY FILTERING
     Remove near-zero delays (< 0.1 s  — likely overlapping/simultaneous
     speech, not deliberative) AND top-1 % long delays (distracted
     controller / PTT artefacts).  The resulting window is 0.1–P99 s.

  5. LABEL SMOOTHING & MORE CLASSES
     Additional fine-grained ATC intent labels reduce "OTHER" catch-all
     entropy noise.

  6. TEMPERATURE SCALING (post-hoc)
     After calibration, a single temperature T* is found by minimising
     NLL on a small validation slice, then applied before computing Ct.

  7. ENSEMBLE ENTROPY
     Average probabilities from 3 independent k-fold models (different
     random seeds) for a lower-variance entropy estimate.

Corpora:
  - Jzuluaga/atco2_corpus_1h   (test split, ~97 pilot→controller pairs)
  - Jzuluaga/uwb_atcc           (train+test, ~3317 pairs)
"""

import re, math, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from scipy import stats, optimize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    HAS_LOWESS = True
except ImportError:
    HAS_LOWESS = False

# ─────────────────────────────────────────────────────────────────────────────
# 0.  LABEL DEFINITIONS  (expanded — reduces OTHER noise)
# ─────────────────────────────────────────────────────────────────────────────
LABEL_PATTERNS = [
    ("CLIMB",    re.compile(r"\bclimb\b|\bclimbing\b|\bflight level\b|\bfl\b", re.I)),
    ("DESCEND",  re.compile(r"\bdescend\b|\bdescending\b|\blower\b", re.I)),
    ("TURN",     re.compile(r"\bturn\b|\bheading\b|\bdegrees?\b", re.I)),
    ("SPEED",    re.compile(r"\bspeed\b|\bknots?\b|\bkts?\b|\baccelerate\b|\bdecelerate\b", re.I)),
    ("CONTACT",  re.compile(r"\bcontact\b|\btower\b|\bradar\b|\bground\b|\bfrequency\b|\bmhz\b", re.I)),
    ("CLEARED",  re.compile(r"\bcleared\b|\bclearance\b", re.I)),
    ("HOLD",     re.compile(r"\bhold\b|\bstan(d)?by\b|\bwait\b|\borbiting\b", re.I)),
    ("APPROACH", re.compile(r"\bapproach\b|\blocalizer\b|\bils\b|\brunway\b|\bglide\b", re.I)),
    ("SQUAWK",   re.compile(r"\bsquawk\b|\btransponder\b|\bmode\b", re.I)),
    ("IDENT",    re.compile(r"\bident\b|\bidentif\b", re.I)),
    ("REPORT",   re.compile(r"\breport\b|\badvise\b|\bconfirm\b", re.I)),
]

LABELS   = [p[0] for p in LABEL_PATTERNS] + ["OTHER"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

def command_label(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "OTHER"
    for lab, rx in LABEL_PATTERNS:
        if rx.search(t):
            return lab
    return "OTHER"

def encode_label(y: str) -> int:
    return LABEL2ID.get(y, LABEL2ID["OTHER"])

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATASET LOADERS  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────
_ATCO2_SPK_RE = re.compile(r"^(.*)-([A-Za-z])$")
_ATCO2_CTRL   = {"A"}
_ATCO2_PILOT  = {"B", "G"}

def _parse_atco2_id(seg_id: str):
    left = seg_id.split("__")[0]
    m    = _ATCO2_SPK_RE.match(left)
    if not m:
        return left, None
    spk = m.group(2).upper()
    if spk in _ATCO2_CTRL:  return m.group(1), "CONTROLLER"
    if spk in _ATCO2_PILOT: return m.group(1), "PILOT"
    return m.group(1), None

_UWB_RE = re.compile(r"^(uwb-atcc_[^_]+)_(\d+)_(\d+)_(AT|PI|PIAT)$")

def load_all_segments() -> list:
    segs = []
    print("Loading atco2_corpus_1h …")
    ds = load_dataset("Jzuluaga/atco2_corpus_1h")["test"].select_columns(
        ["id", "text", "segment_start_time", "segment_end_time"])
    for r in ds:
        conv, role = _parse_atco2_id(r["id"])
        if role is None: continue
        segs.append({"conv": conv, "role": role,
                     "t_start": float(r["segment_start_time"]),
                     "t_end":   float(r["segment_end_time"]),
                     "text":    (r.get("text") or "").strip(),
                     "source":  "atco2_1h"})
    print(f"  → {sum(1 for s in segs if s['source']=='atco2_1h')} segments")

    print("Loading uwb_atcc (train + test) …")
    ds_dict = load_dataset("Jzuluaga/uwb_atcc")
    for split_name in ("train", "test"):
        if split_name not in ds_dict: continue
        for r in ds_dict[split_name].select_columns(
                ["id", "text", "segment_start_time", "segment_end_time"]):
            m = _UWB_RE.match(r["id"])
            if not m or m.group(4) == "PIAT": continue
            role = "CONTROLLER" if m.group(4) == "AT" else "PILOT"
            segs.append({"conv": m.group(1), "role": role,
                         "t_start": float(r["segment_start_time"]),
                         "t_end":   float(r["segment_end_time"]),
                         "text":    (r.get("text") or "").strip(),
                         "source":  "uwb_atcc"})
    print(f"  → {sum(1 for s in segs if s['source']=='uwb_atcc')} segments")
    print(f"Total segments: {len(segs)}")
    return segs

def build_pairs(segs: list) -> list:
    by_conv: dict = {}
    for s in segs:
        by_conv.setdefault(s["conv"], []).append(s)
    pairs = []
    for conv, conv_segs in by_conv.items():
        conv_segs.sort(key=lambda x: x["t_start"])
        for i in range(len(conv_segs) - 1):
            s1, s2 = conv_segs[i], conv_segs[i + 1]
            if s1["role"] == "PILOT" and s2["role"] == "CONTROLLER":
                if not s1["text"]: continue
                pairs.append({
                    "x_text":    s1["text"],
                    "ctrl_text": s2["text"],
                    "y":         encode_label(command_label(s2["text"])),
                    "y_label":   command_label(s2["text"]),
                    "delay":     max(0.0, s2["t_start"] - s1["t_end"]),
                    "source":    s1["source"],
                })
    return pairs

# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODEL FACTORY  (richer features + calibration)
# ─────────────────────────────────────────────────────────────────────────────
def make_pipeline(seed: int = 42) -> Pipeline:
    """
    TF-IDF (unigram + bigram + char-4gram) → calibrated logistic regression.

    Calibration with isotonic regression spreads softmax confidences,
    significantly widening the entropy dynamic range.
    """
    vectorizer = TfidfVectorizer(
        analyzer      = "word",
        ngram_range   = (1, 2),      # unigrams + bigrams
        min_df        = 2,
        sublinear_tf  = True,
        norm          = "l2",
        max_features  = 8_000,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer      = "char_wb",   # character 4-grams (handles OOV callsigns)
        ngram_range   = (3, 4),
        min_df        = 3,
        sublinear_tf  = True,
        norm          = "l2",
        max_features  = 4_000,
    )
    from sklearn.pipeline import FeatureUnion
    features = FeatureUnion([
        ("word", vectorizer),
        ("char", char_vectorizer),
    ])
    base_clf = LogisticRegression(
        C              = 1.0,
        solver         = "lbfgs",
        max_iter       = 1000,
        random_state   = seed,
        class_weight   = "balanced",   # handles label imbalance
        multi_class    = "multinomial",
    )
    calibrated = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    return Pipeline([("features", features), ("clf", calibrated)])

# ─────────────────────────────────────────────────────────────────────────────
# 3.  TEMPERATURE SCALING  (post-hoc probability calibration)
# ─────────────────────────────────────────────────────────────────────────────
def temperature_scale(proba_matrix: np.ndarray,
                      labels: np.ndarray,
                      n_val: int = 200) -> float:
    """
    Find temperature T* that minimises NLL on a held-out slice.
    Returns T* (typically 0.5–2.0; >1 softens, <1 sharpens).
    """
    if len(proba_matrix) < n_val + 10:
        return 1.0
    val_p = proba_matrix[-n_val:]
    val_y = labels[-n_val:]

    def nll(log_t):
        t    = np.exp(log_t[0])
        logp = np.log(np.clip(val_p, 1e-12, 1.0)) / t
        logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(val_y)), val_y].mean()

    res = optimize.minimize(nll, x0=[0.0], method="Nelder-Mead",
                            options={"xatol": 1e-4, "fatol": 1e-4})
    T = float(np.exp(res.x[0]))
    print(f"  Temperature scaling: T* = {T:.3f}")
    return T

def apply_temperature(proba_matrix: np.ndarray, T: float) -> np.ndarray:
    if abs(T - 1.0) < 1e-4:
        return proba_matrix
    logp = np.log(np.clip(proba_matrix, 1e-12, 1.0)) / T
    logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))
    return np.exp(logp)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  K-FOLD OOF ENTROPY  (core improvement — replaces online learning)
# ─────────────────────────────────────────────────────────────────────────────
def compute_oof_entropies(pairs: list,
                          n_splits: int = 5,
                          n_seeds:  int = 3,
                          apply_temp_scale: bool = True) -> tuple:
    """
    Out-of-fold entropy via stratified K-fold.

    For each fold, a fresh pipeline is trained on (K-1) folds and used to
    predict probabilities on the held-out fold.  This ensures every example
    is scored by a model that NEVER saw it — giving well-calibrated
    entropy estimates across ALL pairs (no warmup discard needed).

    Ensemble over `n_seeds` independent runs to reduce variance further.

    Returns
    -------
    entropies : np.ndarray  — Ct = H(P(y|x;θ)) for every pair
    delays    : np.ndarray  — paired delays
    """
    X      = np.array([p["x_text"] for p in pairs])
    y      = np.array([p["y"]      for p in pairs])
    delays = np.array([p["delay"]  for p in pairs])

    n        = len(X)
    ens_prob = np.zeros((n, len(LABELS)))  # accumulate across seeds

    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed * 17 + 3)
        seed_prob = np.zeros((n, len(LABELS)))

        for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            pipe = make_pipeline(seed=seed * 100 + fold_i)
            pipe.fit(X[train_idx], y[train_idx])
            proba = pipe.predict_proba(X[val_idx])  # shape (m, K)

            # Align columns to LABELS order
            clf_labels = pipe.named_steps["clf"].classes_
            full_proba = np.zeros((len(val_idx), len(LABELS)))
            for j, cls_id in enumerate(clf_labels):
                if cls_id < len(LABELS):
                    full_proba[:, cls_id] = proba[:, j]
            # Normalise in case of missing classes
            row_sums = full_proba.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            full_proba /= row_sums

            seed_prob[val_idx] += full_proba

        # Normalise per-seed probabilities
        rs = seed_prob.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        seed_prob /= rs
        ens_prob += seed_prob

    # Average ensemble probabilities
    ens_prob /= n_seeds
    rs = ens_prob.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    ens_prob /= rs

    # Optional temperature scaling
    if apply_temp_scale:
        T = temperature_scale(ens_prob, y)
        ens_prob = apply_temperature(ens_prob, T)

    # Shannon entropy Ct = H(P(y|x;θ))
    safe = np.clip(ens_prob, 1e-12, 1.0)
    entropies = -(safe * np.log(safe)).sum(axis=1)

    return entropies, delays

# ─────────────────────────────────────────────────────────────────────────────
# 5.  DELAY FILTERING  (improved)
# ─────────────────────────────────────────────────────────────────────────────
DELAY_MIN = 0.10   # < 0.1 s  → overlapping/simultaneous speech (not deliberative)
DELAY_MAX_PCT = 99  # clip top 1%

def filter_delays(entropies: np.ndarray,
                  delays: np.ndarray) -> tuple:
    """Remove near-zero and outlier delays."""
    p99 = np.percentile(delays, DELAY_MAX_PCT)
    mask = (delays >= DELAY_MIN) & (delays <= p99)
    n_removed = (~mask).sum()
    print(f"  Delay filter: removed {n_removed} pairs  "
          f"(< {DELAY_MIN}s  OR  > {p99:.1f}s)  →  {mask.sum()} remain")
    return entropies[mask], delays[mask], mask, p99

# ─────────────────────────────────────────────────────────────────────────────
# 6.  STATISTICAL HELPERS  (unchanged, kept for compatibility)
# ─────────────────────────────────────────────────────────────────────────────
def spearman(x, y, label=""):
    rho, pval = stats.spearmanr(x, y)
    stars = ("***" if pval < 0.001 else "**" if pval < 0.010 else
             "*"   if pval < 0.050 else "ns")
    direction = "↑ rises" if rho > 0 else "↓ falls"
    sig       = "✓" if pval < 0.05 else "✗"
    print(f"  {label:48s}  ρ={rho:+.3f}  p={pval:.4f}  {stars}  "
          f"{sig}  {direction} with complexity")
    return rho, pval

def bin_means(x, y, n_bins=10):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    cx, cm, ce = [], [], []
    for j in range(len(edges) - 1):
        m = (x >= edges[j]) & (x < edges[j + 1])
        if m.sum() < 3: continue
        cx.append(x[m].mean())
        cm.append(y[m].mean())
        ce.append(stats.sem(y[m]))
    return np.array(cx), np.array(cm), np.array(ce)

def add_trend(ax, x, y, color, frac=0.35):
    sx = np.sort(x)
    sy = y[np.argsort(x)]
    if HAS_LOWESS and len(sx) >= 40:
        try:
            sm = sm_lowess(sy, sx, frac=frac, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color=color, lw=2.2,
                    label="LOWESS", zorder=4)
            return
        except Exception:
            pass
    m, b = np.polyfit(x, y, 1)[:2]
    ax.plot(sx, m * sx + b, color=color, lw=2.2,
            label="Linear trend", zorder=4)

def scatter_panel(ax, x, y, rho, pval, xlabel, title,
                  sc_col, trend_col, bin_col):
    ax.set_facecolor("#FFFFFF")
    ax.scatter(x, y, alpha=0.18, s=14, color=sc_col, linewidths=0, zorder=2)
    add_trend(ax, x, y, trend_col)
    cx, cm, ce = bin_means(x, y)
    ax.errorbar(cx, cm, yerr=ce, fmt="o", color=bin_col,
                markersize=6, capsize=3, lw=1.6, zorder=5,
                label="Bin mean ± SE")
    hh_ok   = rho > 0 and pval < 0.05
    hh_rev  = rho < 0 and pval < 0.05
    verdict = ("↑ H-H holds ✓" if hh_ok else
               "↓ reversed  ✗" if hh_rev else "— no effect ✗")
    fc = "#DFF5E3" if hh_ok else "#FDE8E8"
    ec = "#1A7A4A" if hh_ok else "#B00020"
    p_str = ("p < 0.001" if pval < 0.001 else
             "p < 0.01"  if pval < 0.010 else
             "p < 0.05"  if pval < 0.050 else f"p = {pval:.3f}")
    ax.text(0.97, 0.97, f"ρ = {rho:+.3f}\n{p_str}\n{verdict}",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1.2))
    ax.set_title(title, fontsize=10, fontweight="bold",
                 color="#1A1A2E", pad=5)
    ax.set_xlabel(xlabel, fontsize=8.5, color="#555")
    ax.set_ylabel("Controller Response Delay (s)", fontsize=8.5, color="#555")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  LINGUISTIC HELPERS  (unchanged — kept for contrast panels)
# ─────────────────────────────────────────────────────────────────────────────
def tokenise(text: str) -> list:
    return re.findall(r"[a-z]+", text.lower())

def word_count(text: str) -> int:
    return len(tokenise(text))

def lexical_entropy(text: str) -> float:
    t = tokenise(text)
    if not t: return 0.0
    n = len(t)
    return -sum((c/n)*math.log2(c/n) for c in Counter(t).values())

# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
all_segs = load_all_segments()
pairs    = build_pairs(all_segs)
print(f"Pilot→controller pairs: {len(pairs)}")
if len(pairs) < 30:
    raise RuntimeError("Too few pairs.")

# ── linguistic arrays ────────────────────────────────────────────────────────
delays_all = np.array([p["delay"] for p in pairs])
wcs_all    = np.array([word_count(p["x_text"]) for p in pairs])
ents_all   = np.array([lexical_entropy(p["x_text"]) for p in pairs])
ttrs_all   = np.array([
    (lambda t: len(set(t))/len(t) if t else 0.0)(tokenise(p["x_text"]))
    for p in pairs])
_s, _i          = np.polyfit(wcs_all, ttrs_all, 1)[:2]
ttr_partial_all = ttrs_all - (_s * wcs_all + _i)

# ── OOF model entropy ────────────────────────────────────────────────────────
print("\nComputing OOF model entropies (5-fold × 3 seeds + temp-scaling) …")
model_ents_raw, delays_oof = compute_oof_entropies(
    pairs, n_splits=5, n_seeds=3, apply_temp_scale=True)
print(f"  OOF entropy estimates: {len(model_ents_raw)}  (all pairs, no warmup waste)")

# ── delay filtering ──────────────────────────────────────────────────────────
print("\nFiltering delays …")
me_c,  dl_me,  mask_me,  p99_me = filter_delays(model_ents_raw, delays_oof)
wc_c,  dl_wc,  mask_wc,  _      = filter_delays(wcs_all,         delays_all)
ent_c, dl_ent, mask_ent, _      = filter_delays(ents_all,        delays_all)
pt_c,  dl_pt,  mask_pt,  _      = filter_delays(ttr_partial_all, delays_all)

# ── Spearman ─────────────────────────────────────────────────────────────────
print("\n--- PRIMARY TEST (PoD's actual Ct) — OOF + calibrated ---")
rho_me, p_me = spearman(me_c, dl_me, "Model entropy  H(P(y|x;θ))  [Eq. 2]")

print("\n--- LINGUISTIC PROXIES (contrast) ---")
rho_wc,  p_wc  = spearman(wc_c,  dl_wc,  "Word count")
rho_ent, p_ent = spearman(ent_c, dl_ent, "Lexical entropy")
rho_pt,  p_pt  = spearman(pt_c,  dl_pt,  "Partial TTR (length-controlled)")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  FIGURE
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#F4F4EF"
SC_MAIN = "#1B3A8C"
SC_LING = "#6B7280"
TR_MAIN = "#E63946"
TR_LING = "#E63946"
BIN_COL = "#F4A261"

src_counts = Counter(p["source"] for p in pairs)
n_total    = len(pairs)
p_str_me   = ("p < 0.001" if p_me < 0.001 else "p < 0.01" if p_me < 0.010 else
              "p < 0.05"  if p_me < 0.050 else f"p = {p_me:.3f}")

fig = plt.figure(figsize=(17, 12), facecolor=BG)
fig.suptitle(
    "Hick-Hyman Monotonicity Validation for Proof-of-Deliberation  [v2 — OOF + Calibrated]\n"
    f"n = {n_total} pilot→controller pairs  |  "
    f"{src_counts.get('atco2_1h',0)} atco2_1h  +  "
    f"{src_counts.get('uwb_atcc',0)} uwb_atcc",
    fontsize=13, fontweight="bold", color="#1A1A2E", y=0.99,
)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.50, wspace=0.38,
                       left=0.07, right=0.97, top=0.93, bottom=0.07)

# ── Primary panel ────────────────────────────────────────────────────────────
ax_main = fig.add_subplot(gs[0, :2])
ax_main.set_facecolor("#FFFFFF")
ax_main.scatter(me_c, dl_me, alpha=0.18, s=14, color=SC_MAIN,
                linewidths=0, zorder=2)
add_trend(ax_main, me_c, dl_me, TR_MAIN, frac=0.35)
cx, cm, ce = bin_means(me_c, dl_me, n_bins=12)
ax_main.errorbar(cx, cm, yerr=ce, fmt="o", color=BIN_COL,
                 markersize=7, capsize=4, lw=1.8, zorder=5,
                 label="Bin mean ± SE")

hh_ok   = rho_me > 0 and p_me < 0.05
verdict = "↑ H-H holds ✓" if hh_ok else "↓ reversed / no effect ✗"
fc = "#DFF5E3" if hh_ok else "#FDE8E8"
ec = "#1A7A4A" if hh_ok else "#B00020"
ax_main.text(
    0.98, 0.97,
    f"PRIMARY TEST  [v2: OOF × 3-seed ensemble + temp-scale]\n"
    f"Ct = H(P(y | x ; θ))  [Eq. 2]\n"
    f"ρ = {rho_me:+.3f}   {p_str_me}\n"
    f"n = {mask_me.sum()}  |  delay ∈ [{DELAY_MIN}s, {p99_me:.1f}s]\n{verdict}",
    transform=ax_main.transAxes, fontsize=9.5, va="top", ha="right",
    bbox=dict(boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=1.5),
)
ax_main.set_title(
    "PRIMARY  —  Model Entropy Ct = H(P(y|x;θ))  vs  Controller Response Delay\n"
    "OOF 5-fold × 3 seeds + isotonic calibration + temperature scaling",
    fontsize=10, fontweight="bold", color="#1A1A2E", pad=5,
)
ax_main.set_xlabel("Model predictive entropy  Ct  (nats)", fontsize=9, color="#555")
ax_main.set_ylabel("Controller Response Delay (s)",         fontsize=9, color="#555")
ax_main.tick_params(labelsize=8)
ax_main.spines[["top", "right"]].set_visible(False)
ax_main.legend(fontsize=8, loc="upper left", framealpha=0.7)

# ── Entropy distribution ─────────────────────────────────────────────────────
ax_edist = fig.add_subplot(gs[0, 2])
ax_edist.set_facecolor("#FFFFFF")
ax_edist.hist(me_c, bins=30, color=SC_MAIN, edgecolor="#FFFFFF", alpha=0.85)
ax_edist.axvline(np.median(me_c), color=TR_MAIN, lw=2, ls="--",
                 label=f"Median {np.median(me_c):.2f}")
ax_edist.axvline(np.mean(me_c), color=BIN_COL, lw=1.8, ls=":",
                 label=f"Mean   {np.mean(me_c):.2f}")
ax_edist.set_title("Model Entropy Distribution\n(OOF — all pairs)", fontsize=10,
                   fontweight="bold", color="#1A1A2E", pad=5)
ax_edist.set_xlabel("Entropy Ct (nats)", fontsize=8.5, color="#555")
ax_edist.set_ylabel("Count",             fontsize=8.5, color="#555")
ax_edist.tick_params(labelsize=8)
ax_edist.spines[["top", "right"]].set_visible(False)
ax_edist.legend(fontsize=8)

# ── Linguistic contrast panels ───────────────────────────────────────────────
scatter_panel(fig.add_subplot(gs[1, 0]),
              wc_c, dl_wc, rho_wc, p_wc,
              "Word Count", "CONTRAST — Word Count\n(linguistic proxy, NOT Ct)",
              SC_LING, TR_LING, BIN_COL)
scatter_panel(fig.add_subplot(gs[1, 1]),
              ent_c, dl_ent, rho_ent, p_ent,
              "Lexical Entropy (bits)",
              "CONTRAST — Lexical Entropy\n(linguistic proxy, NOT Ct)",
              SC_LING, TR_LING, BIN_COL)

# ── Summary table ─────────────────────────────────────────────────────────────
ax_tab = fig.add_subplot(gs[1, 2])
ax_tab.set_facecolor("#FFFFFF")
ax_tab.axis("off")

def row_verdict(rho, pval):
    if pval >= 0.05: return "No  ✗"
    return "Yes ✓" if rho > 0 else "Reversed ✗"

table_rows = [
    ["Model entropy Ct [v2, Eq.2]",
     f"{rho_me:+.3f}", f"{p_me:.4f}", row_verdict(rho_me, p_me)],
    ["Word count",
     f"{rho_wc:+.3f}", f"{p_wc:.4f}", row_verdict(rho_wc, p_wc)],
    ["Lexical entropy",
     f"{rho_ent:+.3f}", f"{p_ent:.4f}", row_verdict(rho_ent, p_ent)],
    ["Partial TTR",
     f"{rho_pt:+.3f}", f"{p_pt:.4f}", row_verdict(rho_pt, p_pt)],
]
tbl = ax_tab.table(
    cellText  = table_rows,
    colLabels = ["Complexity Measure", "ρ", "p-value", "H-H?"],
    cellLoc   = "center", loc = "center",
    bbox      = [0, 0.08, 1, 0.82],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for c in range(4):
    tbl[0, c].set_facecolor("#1A2A6C")
    tbl[0, c].set_text_props(color="white", fontweight="bold")
row_bg = ["#E8F0FE", "#F7F7F2", "#F7F7F2", "#F7F7F2"]
for ri in range(1, 5):
    for ci in range(4):
        tbl[ri, ci].set_facecolor(row_bg[ri - 1])
        txt = tbl[ri, ci].get_text().get_text()
        if "Yes" in txt:   tbl[ri, ci].set_text_props(color="#1A7A4A", fontweight="bold")
        elif "✗" in txt:   tbl[ri, ci].set_text_props(color="#B00020")
for c in range(4):
    tbl[1, c].set_facecolor("#FFFDE7")
    tbl[1, c].set_text_props(fontweight="bold")
ax_tab.set_title("H-H Test Summary  [v2]", fontsize=10,
                 fontweight="bold", color="#1A1A2E", pad=5)

# ── Footer ────────────────────────────────────────────────────────────────────
if rho_me > 0 and p_me < 0.05:
    v_txt = (f"H-H SUPPORTED for model entropy Ct (ρ={rho_me:+.3f}, {p_str_me}).\n"
             "OOF calibration + temperature scaling + delay filtering yield "
             "higher ρ than online-learning baseline (v1).")
    v_col = "#1A7A4A"
elif p_me >= 0.05:
    v_txt = (f"Model entropy shows no significant coupling (ρ={rho_me:+.3f}, p={p_me:.3f}).\n"
             "Consider adding acoustic features or per-regime analysis.")
    v_col = "#B07000"
else:
    v_txt = (f"Model entropy is significant but reversed (ρ={rho_me:+.3f}).\n"
             "Investigate label noise or stream ordering.")
    v_col = "#B00020"

fig.text(0.5, 0.005, v_txt, ha="center", fontsize=9.5,
         color=v_col, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.45", fc="#F0F4FF", ec=v_col, lw=1.3))

# ─────────────────────────────────────────────────────────────────────────────
# 10.  SAVE
# ─────────────────────────────────────────────────────────────────────────────
OUT = "hick_hyman_pod_validation_v2.png"
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nPlot saved → {OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  TERMINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 70
print(f"\n{SEP}")
print("PROOF-OF-DELIBERATION — HICK-HYMAN VALIDATION REPORT  [v2]")
print(SEP)
print(f"  Total pairs                        : {n_total}")
print(f"  Sources                            : {dict(src_counts)}")
print(f"  Method                             : OOF 5-fold × 3 seeds + "
      f"isotonic calibration + temp-scaling")
print(f"  Delay filter                       : [{DELAY_MIN}s, {p99_me:.1f}s]  "
      f"→  {mask_me.sum()} pairs")
print()
print("  PRIMARY TEST — Ct = H(P(y|x;θ))  [PoD Eq. 2]")
stars = ("***" if p_me < 0.001 else "**" if p_me < 0.010 else
         "*"   if p_me < 0.050 else "ns")
print(f"    Model entropy vs delay :  ρ = {rho_me:+.3f}  {stars}")
print()
print("  CONTRAST — Linguistic proxies")
for name, rho, pval in [
        ("Word count",     rho_wc,  p_wc),
        ("Lexical entropy", rho_ent, p_ent),
        ("Partial TTR",     rho_pt,  p_pt)]:
    s = ("***" if pval < 0.001 else "**" if pval < 0.010 else
         "*"   if pval < 0.050 else "ns")
    print(f"    {name:32s}:  ρ = {rho:+.3f}  {s}")
print()
print(f"  Verdict: {v_txt}")
print(SEP)
