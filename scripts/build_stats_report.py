#!/usr/bin/env python3
"""Generate the PoD statistical report and LaTeX macros from per-run CSVs.

PRIMARY (confirmatory): per-dataset, run-level paired Wilcoxon of PoD vs each
competitor on the 20 seed-matched runs (identical stream + per-run seed across
methods, guaranteed by run_suite_generic). Family-wise error across the
5 x (k-1) tests is controlled with Benjamini-Hochberg FDR at q=0.05.
Effect size is paired Cohen's d_z = mean(diff)/sd(diff) with a 10k percentile
bootstrap CI.

SECONDARY (descriptive, Demsar): Friedman + Nemenyi CD across the N=5 datasets,
plus the cross-dataset Wilcoxon reported WITHOUT correction stars (its min
two-sided p with N=5 is 0.0625, below which no corrected test can pass 0.05).

Writes into <in>/stats/: run_level_tests.csv, run_level_summary.csv,
stats_report.json, macros_generated.tex (and refreshes the Demsar artifacts).

This mirrors the integrated logic in pod.stats (functions run_level_paired,
benjamini_hochberg, paired_dz_ci, write_stats_macros).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scipy.stats import wilcoxon
from pod.stats import (collect_per_run_scores, ScoreSpec, average_ranks,
                       friedman_test, nemenyi_critical_diff, wilcoxon_holm)

MACRO_NAME = {"AL":"AL","StaticGating":"StaticGating","AdaptiveGating":"AdaptiveGating",
 "WorkerQuality":"WorkerQuality","Raykar":"Raykar","MACE":"MACE","IEThresh":"IEThresh",
 "PoD-NoGate":"NoGate","PoD-NoCoupling":"NoCoupling","PoD-NoVigilance":"NoVigilance"}
METHOD_ORDER = ["AL","StaticGating","AdaptiveGating","WorkerQuality","Raykar","MACE",
 "IEThresh","PoD","PoD-NoGate","PoD-NoCoupling","PoD-NoVigilance"]

def benjamini_hochberg(pvals, q=0.05):
    pvals=np.asarray(pvals,float); n=pvals.size
    reject=np.zeros(n,bool); p_adj=np.ones(n,float)
    if n==0: return reject,p_adj
    order=np.argsort(pvals); crit=0
    for rank,idx in enumerate(order,1):
        if np.isfinite(pvals[idx]) and pvals[idx]<=(rank/n)*q: crit=rank
    if crit>0: reject[order[:crit]]=True
    prev=1.0
    for rank,idx in zip(range(n,0,-1),order[::-1]):
        val=pvals[idx]*n/rank if np.isfinite(pvals[idx]) else float("nan")
        if np.isfinite(val): prev=min(prev,val); p_adj[idx]=prev
        else: p_adj[idx]=float("nan")
    return reject,p_adj

def paired_dz_ci(diffs,n_boot=10000,alpha=0.05,seed=0):
    diffs=np.asarray(diffs,float); diffs=diffs[np.isfinite(diffs)]
    if diffs.size<2: return float("nan"),float("nan"),float("nan")
    sd=float(np.std(diffs,ddof=1)); dz=float(np.mean(diffs)/sd) if sd>0 else 0.0
    rng=np.random.default_rng(seed); n=diffs.size
    idx=rng.integers(0,n,size=(int(n_boot),n)); res=diffs[idx]
    rsd=res.std(axis=1,ddof=1); rm=res.mean(axis=1)
    with np.errstate(divide="ignore",invalid="ignore"):
        bz=np.where(rsd>0,rm/rsd,0.0)
    return dz,float(np.quantile(bz,alpha/2)),float(np.quantile(bz,1-alpha/2))

def run_level_paired(scores,methods,ref="PoD",q=0.05,n_boot=10000,seed=0):
    comp=[m for m in methods if m!=ref]; rows=[]
    for d in sorted(scores["dataset"].unique()):
        piv=scores[scores["dataset"]==d].pivot(index="run",columns="method",values="score")
        if ref not in piv.columns: continue
        r0=piv[ref].to_numpy(float)
        for m in comp:
            if m not in piv.columns: continue
            diffs=r0-piv[m].to_numpy(float)
            if np.allclose(diffs,0.0): W,p=float("nan"),1.0
            else:
                try: rr=wilcoxon(diffs,zero_method="wilcox",correction=False); W,p=float(rr.statistic),float(rr.pvalue)
                except ValueError: W,p=float("nan"),1.0
            dz,lo,hi=paired_dz_ci(diffs,n_boot,q,seed)
            rows.append(dict(dataset=d,reference=ref,method=m,n_runs=int(diffs.size),
                W=W,p=p,mean_diff=float(diffs.mean()),median_diff=float(np.median(diffs)),
                dz=dz,dz_ci_lo=lo,dz_ci_hi=hi))
    df=pd.DataFrame(rows)
    rej,padj=benjamini_hochberg(df["p"].to_numpy(float),q)
    df["p_bh"]=padj; df["bh_reject"]=rej
    df["win_sig"]=df["bh_reject"]&(df["median_diff"]>0)
    df["loss_sig"]=df["bh_reject"]&(df["median_diff"]<0)
    return df

def summarize(rl,methods,ref="PoD"):
    rows=[]
    for m in [x for x in methods if x!=ref]:
        g=rl[rl["method"]==m]
        if g.empty: continue
        rows.append(dict(method=m,n_datasets=int(len(g)),wins_sig=int(g["win_sig"].sum()),
            losses_sig=int(g["loss_sig"].sum()),median_dz=float(g["dz"].median()),
            min_dz=float(g["dz"].min()),max_dz=float(g["dz"].max())))
    return pd.DataFrame(rows)

def fmt_p(p):
    if not np.isfinite(p): return r"\mathrm{n/a}"
    return r"<\!10^{-4}" if p<1e-4 else f"{p:.4g}"

def write_macros(path,rep):
    cfg=rep["config"]; fr=rep["secondary_cross_dataset"]["friedman"]
    cd=rep["secondary_cross_dataset"]["nemenyi_CD"]
    pc={r["method"]:r for r in rep["primary_run_level"]["per_competitor"]}
    L=["%% =====================================================================",
       "%% AUTO-GENERATED by scripts/build_stats_report.py (pod-stats). DO NOT EDIT.",
       "%% Every number below is produced from out_pod_unified/<dataset>/runs/ and",
       "%% is the single source of truth checked by scripts/check_claims.py.",
       "%% Regenerate: python scripts/build_stats_report.py --in out_pod_unified",
       "%% ====================================================================="]
    L+=[f"\\newcommand{{\\NDatasets}}{{\\ensuremath{{{cfg['n_datasets']}}}}}",
        f"\\newcommand{{\\NMethods}}{{\\ensuremath{{{cfg['n_methods']}}}}}",
        f"\\newcommand{{\\NCompetitors}}{{\\ensuremath{{{cfg['n_methods']-1}}}}}",
        f"\\newcommand{{\\FDRtests}}{{\\ensuremath{{{cfg['fdr_n_tests']}}}}}",
        "% --- Cross-dataset omnibus (SECONDARY / descriptive) ---",
        f"\\newcommand{{\\FriedmanChiSq}}{{\\ensuremath{{{fr['chi2']:.2f}}}}}",
        f"\\newcommand{{\\FriedmanChiSqP}}{{\\ensuremath{{{fmt_p(fr['p_chi2'])}}}}}",
        f"\\newcommand{{\\FriedmanF}}{{\\ensuremath{{{fr['F']:.2f}}}}}",
        f"\\newcommand{{\\FriedmanFP}}{{\\ensuremath{{{fmt_p(fr['p_F'])}}}}}",
        f"\\newcommand{{\\NemenyiCD}}{{\\ensuremath{{{cd:.2f}}}}}",
        "% --- Run-level BH-FDR results (PRIMARY / confirmatory) ---"]
    for raw,san in MACRO_NAME.items():
        if raw not in pc: continue
        r=pc[raw]
        L+=[f"\\newcommand{{\\RLwins{san}}}{{\\ensuremath{{{int(r['wins_sig'])}}}}}",
            f"\\newcommand{{\\RLlosses{san}}}{{\\ensuremath{{{int(r['losses_sig'])}}}}}",
            f"\\newcommand{{\\RLdzMed{san}}}{{\\ensuremath{{{r['median_dz']:.2f}}}}}",
            f"\\newcommand{{\\RLdzMin{san}}}{{\\ensuremath{{{r['min_dz']:.2f}}}}}",
            f"\\newcommand{{\\RLdzMax{san}}}{{\\ensuremath{{{r['max_dz']:.2f}}}}}"]
    open(path,"w").write("\n".join(L)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",dest="in_dir",default="out_pod_unified")
    ap.add_argument("--alpha",type=float,default=0.05)
    ap.add_argument("--n-boot",type=int,default=10000)
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args()
    out=os.path.join(a.in_dir,"stats"); os.makedirs(out,exist_ok=True)
    scores=collect_per_run_scores(a.in_dir,ScoreSpec("end",50))
    present=list(scores["method"].unique())
    methods=[m for m in METHOD_ORDER if m in present]+[m for m in present if m not in METHOD_ORDER]
    avg=average_ranks(scores,methods); fried=friedman_test(scores,methods)
    cd=nemenyi_critical_diff(len(methods),int(avg["n_datasets"].iloc[0]),a.alpha)
    wil=wilcoxon_holm(scores,methods,"PoD")
    rl=run_level_paired(scores,methods,"PoD",a.alpha,a.n_boot,a.seed)
    rl.to_csv(os.path.join(out,"run_level_tests.csv"),index=False)
    rls=summarize(rl,methods,"PoD"); rls.to_csv(os.path.join(out,"run_level_summary.csv"),index=False)
    pod_rank=float(avg.loc[avg["method"]=="PoD","avg_rank"].iloc[0])
    gap={r["method"]:{"avg_rank":float(r["avg_rank"]),"gap_to_ref":float(r["avg_rank"])-pod_rank,
         "exceeds_CD":bool(float(r["avg_rank"])-pod_rank>cd)} for _,r in avg.iterrows() if r["method"]!="PoD"}
    rep={"config":{"reference":"PoD","score_mode":"end","end_window":50,"alpha_q":a.alpha,
            "n_boot":a.n_boot,"seed":a.seed,"n_datasets":int(avg["n_datasets"].iloc[0]),
            "n_methods":len(methods),"methods":methods,"fdr_method":"benjamini_hochberg",
            "fdr_n_tests":int(len(rl))},
         "primary_run_level":{"per_test":json.loads(rl.to_json(orient="records")),
            "per_competitor":json.loads(rls.to_json(orient="records"))},
         "secondary_cross_dataset":{"friedman":fried,"nemenyi_CD":cd,
            "average_ranks":json.loads(avg.to_json(orient="records")),"nemenyi_gap":gap,
            "wilcoxon_descriptive":json.loads(wil.to_json(orient="records"))}}
    json.dump(rep,open(os.path.join(out,"stats_report.json"),"w"),indent=2)
    write_macros(os.path.join(out,"macros_generated.tex"),rep)
    print(f"[build_stats] {len(rl)} run-level tests; BH q={a.alpha}; CD={cd:.2f}")
    for _,r in rls.iterrows():
        print(f"  {r['method']:16s} wins {int(r['wins_sig'])}/{int(r['n_datasets'])} "
              f"loss {int(r['losses_sig'])}/{int(r['n_datasets'])} med d_z={r['median_dz']:.2f} "
              f"[{r['min_dz']:.2f},{r['max_dz']:.2f}]")
    print(f"[build_stats] wrote stats_report.json + macros_generated.tex to {out}")

if __name__=="__main__": main()
