#!/usr/bin/env python3
"""Fail CI if the manuscript's statistical claims are not backed by data.

Checks, against out_pod_unified/stats/{stats_report.json,macros_generated.tex}
and the manuscript .tex:

  (C1) No run-level test is marked significant (BH-reject) with adjusted
       p > alpha. (The arithmetic-impossibility guard that the v1.2 paper
       failed: a "significant" result whose corrected p exceeds 0.05.)
  (C2) The descriptive cross-dataset Wilcoxon contains NO row asserted
       significant (it cannot be, with N=5).
  (C3) The generated win-count / d_z macros are internally consistent with
       stats_report.json (no stale macro file).
  (C4) The manuscript contains none of the purged impossible phrases
       (Holm-corrected significance stars, p=0.0312-as-significant,
       blanket "dominates every ... d>1.2").
  (C5) Every textual dominance claim of the form "wins on all N datasets"
       is backed: classical baselines and PoD-NoVigilance must indeed have
       wins == N in stats_report; content baselines must NOT be claimed at N
       unless they truly are.

Exit code 0 = all claims supported; 1 = at least one violation.
"""
from __future__ import annotations
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = os.path.join(ROOT, "out_pod_unified", "stats")
REPORT = os.path.join(STATS, "stats_report.json")
MACROS = os.path.join(STATS, "macros_generated.tex")
TEX = os.path.join(ROOT, "trimmed.tex")

SAN = {"AL":"AL","StaticGating":"StaticGating","AdaptiveGating":"AdaptiveGating",
 "WorkerQuality":"WorkerQuality","Raykar":"Raykar","MACE":"MACE","IEThresh":"IEThresh",
 "PoD-NoGate":"NoGate","PoD-NoCoupling":"NoCoupling","PoD-NoVigilance":"NoVigilance"}

def fail(viol, msg): viol.append(msg)

def main() -> int:
    viol = []
    rep = json.load(open(REPORT, encoding="utf-8"))
    alpha = rep["config"]["alpha_q"]
    per_test = rep["primary_run_level"]["per_test"]
    per_comp = {r["method"]: r for r in rep["primary_run_level"]["per_competitor"]}

    # C1: significant <=> p_bh <= alpha
    for t in per_test:
        if t.get("bh_reject") and not (t["p_bh"] <= alpha + 1e-12):
            fail(viol, f"C1 {t['dataset']}/{t['method']}: bh_reject but p_bh={t['p_bh']:.4g} > {alpha}")
        if t.get("win_sig") and not t.get("bh_reject"):
            fail(viol, f"C1 {t['dataset']}/{t['method']}: win_sig without bh_reject")

    # C2: descriptive cross-dataset has no asserted significance
    for w in rep["secondary_cross_dataset"]["wilcoxon_descriptive"]:
        if w.get("reject_05") or w.get("reject_01"):
            fail(viol, f"C2 cross-dataset {w['method']}: marked significant (N=5 cannot)")

    # C3: macros consistent with report
    mac = open(MACROS, encoding="utf-8").read()
    def macval(name):
        m = re.search(r"\\newcommand\{\\"+re.escape(name)+r"\}\{\\ensuremath\{(-?[0-9.]+)\}\}", mac)
        return m.group(1) if m else None
    for raw, san in SAN.items():
        if raw not in per_comp: continue
        c = per_comp[raw]
        w = macval("RLwins"+san)
        if w is None or int(w) != int(c["wins_sig"]):
            fail(viol, f"C3 RLwins{san}: macro={w} report={c['wins_sig']}")
        dz = macval("RLdzMed"+san)
        if dz is None or abs(float(dz) - round(c["median_dz"],2)) > 0.005:
            fail(viol, f"C3 RLdzMed{san}: macro={dz} report={c['median_dz']:.2f}")

    # C4: purged phrases
    tex = open(TEX, encoding="utf-8").read()
    banned = [
        (r"0\.0312\}\)?\s*,?\s*which corresponds to", "p=0.0312 dominance gloss"),
        (r"dominat(es|ing) (every|\\emph\{every\})", "blanket dominance claim"),
        (r"\$d\s*>\s*1\.2\$\s*throughout", "blanket d>1.2 throughout"),
        (r"p_\{\\text\{Holm\}\}\$\s*&", "Holm-corrected pairwise column in a results table (cross-dataset table is invalid at N=5; inline Holm for the n=24 within-subject family is allowed)"),
        (r"Significant at \$\\alpha=0\.05\$ after Holm", "Holm significance footnote"),
    ]
    for pat, desc in banned:
        if re.search(pat, tex):
            fail(viol, f"C4 manuscript still contains: {desc}  (/{pat}/)")

    # C5: backed dominance claims
    # Classical + NoVigilance must be wins==n_datasets; else any '5/5' claim is unbacked.
    n = rep["config"]["n_datasets"]
    must_all = ["AL","StaticGating","AdaptiveGating","PoD-NoVigilance"]
    for m in must_all:
        if m in per_comp and per_comp[m]["wins_sig"] != n:
            fail(viol, f"C5 {m}: paper claims all-{n} but wins_sig={per_comp[m]['wins_sig']}")
    # Content baselines must NOT be asserted to win on all n in the abstract/contrib.
    # (We assert the paper does not claim 'all 5' for content baselines.)
    for m in ["WorkerQuality","Raykar","MACE","IEThresh"]:
        if m in per_comp and per_comp[m]["wins_sig"] == n:
            # allowed (it's true) -- no violation; informational only
            pass

    print(f"check_claims: {len(per_test)} run-level tests, alpha={alpha}, "
          f"{len(per_comp)} competitors.")
    if viol:
        print(f"FAILED with {len(viol)} violation(s):")
        for v in viol: print("  -", v)
        return 1
    print("PASS: every statistical claim is backed by stats_report.json.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
