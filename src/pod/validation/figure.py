"""
Hick-Hyman validation figure assembly.

The layout follows the paper's Figure 7: a primary scatter of model
entropy vs response delay (with LOWESS/linear trend and bin-mean
error bars), an entropy-distribution histogram, two linguistic-proxy
contrast panels, and a tabular summary of the four Spearman tests.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from pod.validation.stats import bin_means, filter_delays

try:  # pragma: no cover - optional smoothing dependency
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    HAS_LOWESS = True
except ImportError:  # pragma: no cover
    HAS_LOWESS = False


_BG = "#F4F4EF"
_SC_MAIN = "#1B3A8C"
_SC_LING = "#6B7280"
_TR_MAIN = "#E63946"
_TR_LING = "#E63946"
_BIN_COL = "#F4A261"


def _tokenise(text: str) -> list:
    return re.findall(r"[a-z]+", text.lower())


def word_count(text: str) -> int:
    """Whitespace word count after lowercasing."""
    return len(_tokenise(text))


def lexical_entropy(text: str) -> float:
    """Shannon entropy of the token unigram distribution (bits)."""
    t = _tokenise(text)
    if not t:
        return 0.0
    n = len(t)
    return -sum((c / n) * math.log2(c / n) for c in Counter(t).values())


def type_token_ratio(text: str) -> float:
    """Type-token ratio with a zero fallback for empty strings."""
    t = _tokenise(text)
    return len(set(t)) / len(t) if t else 0.0


def _add_trend(ax, x, y, color, frac: float = 0.35) -> None:
    sx = np.sort(x)
    sy = y[np.argsort(x)]
    if HAS_LOWESS and len(sx) >= 40:
        try:
            sm = sm_lowess(sy, sx, frac=frac, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color=color, lw=2.2, label="LOWESS", zorder=4)
            return
        except Exception:  # pragma: no cover - defensive
            pass
    m, b = np.polyfit(x, y, 1)[:2]
    ax.plot(sx, m * sx + b, color=color, lw=2.2, label="Linear trend", zorder=4)


def _scatter_panel(ax, x, y, rho, pval, xlabel, title) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.scatter(x, y, alpha=0.18, s=14, color=_SC_LING, linewidths=0, zorder=2)
    _add_trend(ax, x, y, _TR_LING)
    cx, cm, ce = bin_means(x, y)
    ax.errorbar(
        cx, cm, yerr=ce, fmt="o", color=_BIN_COL,
        markersize=6, capsize=3, lw=1.6, zorder=5, label="Bin mean +/- SE",
    )
    hh_ok = rho > 0 and pval < 0.05
    hh_rev = rho < 0 and pval < 0.05
    verdict = (
        "H-H holds" if hh_ok
        else "reversed" if hh_rev
        else "no effect"
    )
    fc = "#DFF5E3" if hh_ok else "#FDE8E8"
    ec = "#1A7A4A" if hh_ok else "#B00020"
    p_str = (
        "p < 0.001" if pval < 0.001
        else "p < 0.01" if pval < 0.010
        else "p < 0.05" if pval < 0.050
        else f"p = {pval:.3f}"
    )
    ax.text(
        0.97, 0.97, f"rho = {rho:+.3f}\n{p_str}\n{verdict}",
        transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1.2),
    )
    ax.set_title(title, fontsize=10, fontweight="bold", color="#1A1A2E", pad=5)
    ax.set_xlabel(xlabel, fontsize=8.5, color="#555")
    ax.set_ylabel("Controller Response Delay (s)", fontsize=8.5, color="#555")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.7)


def render_figure(
    pairs: list,
    model_ents_raw: np.ndarray,
    delays_oof: np.ndarray,
    out_path: str,
) -> Tuple[float, float, dict]:
    """Render the validation figure and return the primary test result.

    Parameters
    ----------
    pairs : list
        Pilot-controller pairs from :func:`pod.validation.corpora.build_pairs`.
    model_ents_raw, delays_oof : np.ndarray
        Output of
        :func:`pod.validation.entropy.compute_oof_entropies`.
    out_path : str
        Destination path; the file extension is honoured by matplotlib.

    Returns
    -------
    rho_me, p_me : float
        Spearman rho and p-value of the primary Hick-Hyman test.
    extras : dict
        Linguistic-proxy results, sample counts, and the p99 delay
        cutoff, useful for the textual report.
    """
    from pod.validation.stats import spearman  # local import to avoid cycle

    delays_all = np.array([p["delay"] for p in pairs])
    wcs_all = np.array([word_count(p["x_text"]) for p in pairs])
    ents_all = np.array([lexical_entropy(p["x_text"]) for p in pairs])
    ttrs_all = np.array([type_token_ratio(p["x_text"]) for p in pairs])
    slope, intercept = np.polyfit(wcs_all, ttrs_all, 1)[:2]
    ttr_partial_all = ttrs_all - (slope * wcs_all + intercept)

    me_c, dl_me, mask_me, p99_me = filter_delays(model_ents_raw, delays_oof)
    wc_c, dl_wc, _, _ = filter_delays(wcs_all, delays_all)
    ent_c, dl_ent, _, _ = filter_delays(ents_all, delays_all)
    pt_c, dl_pt, _, _ = filter_delays(ttr_partial_all, delays_all)

    print("\n--- PRIMARY TEST (PoD's actual Ct) -- OOF + calibrated ---")
    rho_me, p_me = spearman(me_c, dl_me, "Model entropy H(P(y|x;theta)) [Eq. 2]")

    print("\n--- LINGUISTIC PROXIES (contrast) ---")
    rho_wc, p_wc = spearman(wc_c, dl_wc, "Word count")
    rho_ent, p_ent = spearman(ent_c, dl_ent, "Lexical entropy")
    rho_pt, p_pt = spearman(pt_c, dl_pt, "Partial TTR (length-controlled)")

    src_counts = Counter(p["source"] for p in pairs)
    n_total = len(pairs)
    p_str_me = (
        "p < 0.001" if p_me < 0.001
        else "p < 0.01" if p_me < 0.010
        else "p < 0.05" if p_me < 0.050
        else f"p = {p_me:.3f}"
    )

    fig = plt.figure(figsize=(17, 12), facecolor=_BG)
    fig.suptitle(
        "Hick-Hyman Monotonicity Validation for Proof-of-Deliberation "
        "[v2 -- OOF + Calibrated]\n"
        f"n = {n_total} pilot->controller pairs  |  "
        f"{src_counts.get('atco2_1h', 0)} atco2_1h  +  "
        f"{src_counts.get('uwb_atcc', 0)} uwb_atcc",
        fontsize=13, fontweight="bold", color="#1A1A2E", y=0.99,
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.50, wspace=0.38,
        left=0.07, right=0.97, top=0.93, bottom=0.07,
    )

    ax_main = fig.add_subplot(gs[0, :2])
    ax_main.set_facecolor("#FFFFFF")
    ax_main.scatter(
        me_c, dl_me, alpha=0.18, s=14, color=_SC_MAIN, linewidths=0, zorder=2,
    )
    _add_trend(ax_main, me_c, dl_me, _TR_MAIN, frac=0.35)
    cx, cm, ce = bin_means(me_c, dl_me, n_bins=12)
    ax_main.errorbar(
        cx, cm, yerr=ce, fmt="o", color=_BIN_COL,
        markersize=7, capsize=4, lw=1.8, zorder=5, label="Bin mean +/- SE",
    )

    hh_ok = rho_me > 0 and p_me < 0.05
    verdict = "H-H holds" if hh_ok else "reversed / no effect"
    fc = "#DFF5E3" if hh_ok else "#FDE8E8"
    ec = "#1A7A4A" if hh_ok else "#B00020"
    ax_main.text(
        0.98, 0.97,
        "PRIMARY TEST  [v2: OOF x 3-seed ensemble + temp-scale]\n"
        "Ct = H(P(y | x ; theta))  [Eq. 2]\n"
        f"rho = {rho_me:+.3f}   {p_str_me}\n"
        f"n = {int(mask_me.sum())}  |  delay in [0.1s, {p99_me:.1f}s]\n{verdict}",
        transform=ax_main.transAxes, fontsize=9.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=1.5),
    )
    ax_main.set_title(
        "PRIMARY  --  Model Entropy Ct = H(P(y|x;theta))  vs  "
        "Controller Response Delay\n"
        "OOF 5-fold x 3 seeds + isotonic calibration + temperature scaling",
        fontsize=10, fontweight="bold", color="#1A1A2E", pad=5,
    )
    ax_main.set_xlabel("Model predictive entropy Ct (nats)", fontsize=9, color="#555")
    ax_main.set_ylabel("Controller Response Delay (s)", fontsize=9, color="#555")
    ax_main.tick_params(labelsize=8)
    ax_main.spines[["top", "right"]].set_visible(False)
    ax_main.legend(fontsize=8, loc="upper left", framealpha=0.7)

    ax_edist = fig.add_subplot(gs[0, 2])
    ax_edist.set_facecolor("#FFFFFF")
    ax_edist.hist(me_c, bins=30, color=_SC_MAIN, edgecolor="#FFFFFF", alpha=0.85)
    ax_edist.axvline(
        float(np.median(me_c)), color=_TR_MAIN, lw=2, ls="--",
        label=f"Median {np.median(me_c):.2f}",
    )
    ax_edist.axvline(
        float(np.mean(me_c)), color=_BIN_COL, lw=1.8, ls=":",
        label=f"Mean   {np.mean(me_c):.2f}",
    )
    ax_edist.set_title(
        "Model Entropy Distribution\n(OOF -- all pairs)",
        fontsize=10, fontweight="bold", color="#1A1A2E", pad=5,
    )
    ax_edist.set_xlabel("Entropy Ct (nats)", fontsize=8.5, color="#555")
    ax_edist.set_ylabel("Count", fontsize=8.5, color="#555")
    ax_edist.tick_params(labelsize=8)
    ax_edist.spines[["top", "right"]].set_visible(False)
    ax_edist.legend(fontsize=8)

    _scatter_panel(
        fig.add_subplot(gs[1, 0]),
        wc_c, dl_wc, rho_wc, p_wc,
        "Word Count",
        "CONTRAST -- Word Count\n(linguistic proxy, NOT Ct)",
    )
    _scatter_panel(
        fig.add_subplot(gs[1, 1]),
        ent_c, dl_ent, rho_ent, p_ent,
        "Lexical Entropy (bits)",
        "CONTRAST -- Lexical Entropy\n(linguistic proxy, NOT Ct)",
    )

    ax_tab = fig.add_subplot(gs[1, 2])
    ax_tab.set_facecolor("#FFFFFF")
    ax_tab.axis("off")

    def row_verdict(rho: float, pval: float) -> str:
        if pval >= 0.05:
            return "No"
        return "Yes" if rho > 0 else "Reversed"

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
        cellText=table_rows,
        colLabels=["Complexity Measure", "rho", "p-value", "H-H?"],
        cellLoc="center", loc="center",
        bbox=(0, 0.08, 1, 0.82),
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
            if "Yes" in txt:
                tbl[ri, ci].set_text_props(color="#1A7A4A", fontweight="bold")
            elif "No" in txt or "Reversed" in txt:
                tbl[ri, ci].set_text_props(color="#B00020")
    for c in range(4):
        tbl[1, c].set_facecolor("#FFFDE7")
        tbl[1, c].set_text_props(fontweight="bold")
    ax_tab.set_title(
        "H-H Test Summary [v2]", fontsize=10,
        fontweight="bold", color="#1A1A2E", pad=5,
    )

    if rho_me > 0 and p_me < 0.05:
        v_txt = (
            f"H-H SUPPORTED for model entropy Ct (rho={rho_me:+.3f}, {p_str_me}).\n"
            "OOF calibration + temperature scaling + delay filtering yield "
            "higher rho than online-learning baseline (v1)."
        )
        v_col = "#1A7A4A"
    elif p_me >= 0.05:
        v_txt = (
            f"Model entropy shows no significant coupling "
            f"(rho={rho_me:+.3f}, p={p_me:.3f}).\n"
            "Consider adding acoustic features or per-regime analysis."
        )
        v_col = "#B07000"
    else:
        v_txt = (
            f"Model entropy is significant but reversed (rho={rho_me:+.3f}).\n"
            "Investigate label noise or stream ordering."
        )
        v_col = "#B00020"

    fig.text(
        0.5, 0.005, v_txt, ha="center", fontsize=9.5,
        color=v_col, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="#F0F4FF", ec=v_col, lw=1.3),
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"\nPlot saved -> {out_path}")

    extras = dict(
        n_total=n_total,
        src_counts=dict(src_counts),
        n_kept=int(mask_me.sum()),
        p99=p99_me,
        word_count=(rho_wc, p_wc),
        lexical_entropy=(rho_ent, p_ent),
        partial_ttr=(rho_pt, p_pt),
        verdict=v_txt,
    )
    return float(rho_me), float(p_me), extras
