"""
Analyze base-learner robustness outputs: per-run final-window F1, run-level
two-sided Wilcoxon signed-rank of PoD vs each competitor, paired Cohen's d_z,
and BH-FDR across the competitor family. Emits a JSON summary and LaTeX macros.

Final-window score per (method, run) = mean F1 over the last K=50 evaluation
points, matching the paper's cross-dataset reduction.
"""
from __future__ import annotations
import glob, json, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon

K = 50
METHODS = ["AL","StaticGating","AdaptiveGating","WorkerQuality","Raykar","MACE",
           "IEThresh","PoD","PoD-NoGate","PoD-NoCoupling","PoD-NoVigilance"]
COMPETITORS = [m for m in METHODS if m != "PoD"]
NAMEPREFIX = {"synth":"Synth-Boundary","gas":"uci224_gas_drift"}

def final_scores(runs_dir, prefix, method):
    fs = sorted(glob.glob(os.path.join(runs_dir, f"{prefix}_{method}_run*.csv")),
                key=lambda p: int(p.split("run")[-1].split(".")[0]))
    return np.array([pd.read_csv(f)["f1"].to_numpy()[-K:].mean() for f in fs])

def dz(diff):
    sd = diff.std(ddof=1)
    return float(diff.mean()/sd) if sd>1e-12 else float("inf") if diff.mean()>0 else 0.0

def bh(pvals, q=0.05):
    p = np.asarray(pvals); n=len(p); order=np.argsort(p); ranks=np.empty(n,int)
    ranks[order]=np.arange(1,n+1); adj=p*n/ranks
    # enforce monotonicity
    adj_sorted=adj[order]; adj_sorted=np.minimum.accumulate(adj_sorted[::-1])[::-1]
    out=np.empty(n); out[order]=np.clip(adj_sorted,0,1); return out

def analyze(out_root, dataset, learner):
    runs_dir=os.path.join(out_root, f"{dataset}_{learner}", "runs")
    prefix=NAMEPREFIX[dataset]
    pod=final_scores(runs_dir,prefix,"PoD")
    res={"dataset":dataset,"learner":learner,"n_runs":int(len(pod)),
         "pod_mean_final_f1":float(pod.mean()),"competitors":{}}
    pvals=[]; comps=[]
    for c in COMPETITORS:
        x=final_scores(runs_dir,prefix,c)
        n=min(len(pod),len(x)); d=pod[:n]-x[:n]
        if np.allclose(d,0): p=1.0
        else:
            try: p=float(wilcoxon(pod[:n],x[:n],zero_method="wilcox",alternative="two-sided").pvalue)
            except Exception: p=1.0
        res["competitors"][c]={"comp_mean_final_f1":float(x[:n].mean()),
            "median_paired_diff":float(np.median(d)),"d_z":dz(d),"p_raw":p}
        pvals.append(p); comps.append(c)
    padj=bh(pvals)
    for c,pa in zip(comps,padj):
        sig = bool(pa<0.05 and res["competitors"][c]["d_z"]>0)
        res["competitors"][c]["p_bh"]=float(pa); res["competitors"][c]["pod_wins_sig"]=sig
    return res

def main():
    out_root="/tmp/rob"; results=[]
    for dataset in ["synth","gas"]:
        for learner in ["sgd","mlp","nb"]:
            d=os.path.join(out_root,f"{dataset}_{learner}","runs")
            if os.path.isdir(d) and glob.glob(os.path.join(d,"*_PoD_run*.csv")):
                results.append(analyze(out_root,dataset,learner))
    os.makedirs(out_root,exist_ok=True)
    json.dump(results, open(os.path.join(out_root,"robustness_summary.json"),"w"), indent=2)
    # console table
    for r in results:
        print(f"\n=== {r['dataset']} / {r['learner']} (n={r['n_runs']}) PoD final-F1={r['pod_mean_final_f1']:.3f} ===")
        for c in COMPETITORS:
            cc=r["competitors"][c]
            print(f"  vs {c:16s} dz={cc['d_z']:+6.2f} pBH={cc['p_bh']:.3g} win={cc['pod_wins_sig']} (PoD {r['pod_mean_final_f1']:.3f} vs {cc['comp_mean_final_f1']:.3f})")
    print("\nwrote /tmp/rob/robustness_summary.json")

if __name__=="__main__":
    main()
