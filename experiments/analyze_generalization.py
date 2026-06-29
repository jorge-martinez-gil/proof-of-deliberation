"""Aggregate ask#3: (A) base-learner robustness across datasets x learners and
(B) second-operator-model robustness. Reuses the analyze_robustness reduction
(final-window K=50 F1, paired Wilcoxon, Cohen's d_z, BH-FDR over the competitor
family). Writes JSON + console tables + a summary figure."""
import glob, os, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

GEN=_os.path.join(_ROOT,"out_generalization")
ROB=_os.path.join(_ROOT,"out_robustness")
PREFIX={"synth":"Synth-Boundary","synth_hd":"Synth-HighDim","gas":"uci224_gas_drift"}
COMP=["AL","StaticGating","AdaptiveGating","WorkerQuality","Raykar","MACE","IEThresh"]
K=50

def dz(d):
    sd=d.std(ddof=1); return float(d.mean()/sd) if sd>1e-12 else (float("inf") if d.mean()>0 else 0.0)
def bh(p):
    p=np.asarray(p); n=len(p); o=np.argsort(p); r=np.empty(n,int); r[o]=np.arange(1,n+1)
    adj=p*n/r; a=adj[o]; a=np.minimum.accumulate(a[::-1])[::-1]; out=np.empty(n); out[o]=np.clip(a,0,1); return out
def finals(runs_dir, pref, m):
    fs=sorted(glob.glob(os.path.join(runs_dir,f"{pref}_{m}_run*.csv")), key=lambda p:int(p.split("run")[-1].split(".")[0]))
    return np.array([pd.read_csv(f)["f1"].to_numpy()[-K:].mean() for f in fs])

def analyze_dir(runs_dir, pref, label):
    pod=finals(runs_dir,pref,"PoD")
    res={"label":label,"n":int(len(pod)),"pod_f1":float(pod.mean()),"comp":{}}
    pv=[]; names=[]
    for c in COMP:
        x=finals(runs_dir,pref,c)
        if len(x)==0: continue
        n=min(len(pod),len(x)); d=pod[:n]-x[:n]
        p=1.0 if np.allclose(d,0) else float(wilcoxon(pod[:n],x[:n],alternative="two-sided").pvalue)
        res["comp"][c]={"f1":float(x[:n].mean()),"dz":dz(d),"p":p,"median_diff":float(np.median(d))}
        pv.append(p); names.append(c)
    for c,pa in zip(names,bh(pv)):
        res["comp"][c]["p_bh"]=float(pa); res["comp"][c]["win"]=bool(pa<0.05 and res["comp"][c]["dz"]>0)
    res["n_sig_wins"]=int(sum(res["comp"][c]["win"] for c in names))
    res["n_comp"]=len(names)
    return res

def main():
    out={"base_learner":{}, "operator_v2":{}}
    # (A) base-learner robustness: synth, synth_hd (new) x {sgd,mlp,nb}; gas from out_robustness
    for ds in ["synth","synth_hd"]:
        for lr in ["sgd","mlp","nb"]:
            d=os.path.join(GEN,f"{ds}_{lr}_v1","runs")
            if glob.glob(os.path.join(d,"*_PoD_run*.csv")):
                out["base_learner"][f"{ds}/{lr}"]=analyze_dir(d,PREFIX[ds],f"{ds}/{lr}")
    for lr in ["sgd","mlp","nb"]:
        d=os.path.join(ROB,f"gas_{lr}","runs")
        if glob.glob(os.path.join(d,"*_PoD_run*.csv")):
            out["base_learner"][f"gas/{lr}"]=analyze_dir(d,PREFIX["gas"],f"gas/{lr}")
    # (B) operator v2 vs v1 (sgd)
    for ds in ["synth","gas"]:
        v2=os.path.join(GEN,f"{ds}_sgd_v2","runs")
        if glob.glob(os.path.join(v2,"*_PoD_run*.csv")):
            out["operator_v2"][f"{ds}/v2"]=analyze_dir(v2,PREFIX[ds],f"{ds}/v2")
    # v1 references: synth from GEN, gas from ROB
    if glob.glob(os.path.join(GEN,"synth_sgd_v1","runs","*_PoD_run*.csv")):
        out["operator_v2"]["synth/v1"]=analyze_dir(os.path.join(GEN,"synth_sgd_v1","runs"),PREFIX["synth"],"synth/v1")
    if glob.glob(os.path.join(ROB,"gas_sgd","runs","*_PoD_run*.csv")):
        out["operator_v2"]["gas/v1"]=analyze_dir(os.path.join(ROB,"gas_sgd","runs"),PREFIX["gas"],"gas/v1")

    json.dump(out, open(os.path.join(GEN,"generalization_summary.json"),"w"), indent=2)

    print("=== (A) BASE-LEARNER ROBUSTNESS: PoD final-F1 and #significant wins vs 7 competitors ===")
    print(f"  {'cell':16s} {'n':>3} {'PoD_F1':>7} {'sigwins':>8}   strongest-competitor")
    for k,r in out["base_learner"].items():
        best=max(r["comp"].items(), key=lambda kv: kv[1]["f1"]) if r["comp"] else (None,None)
        print(f"  {k:16s} {r['n']:>3} {r['pod_f1']:>7.3f} {r['n_sig_wins']:>3}/{r['n_comp']:<3}   "
              f"{best[0]}={best[1]['f1']:.3f} (dz={best[1]['dz']:+.2f},pBH={best[1]['p_bh']:.2g})")
    print("\n=== (B) SECOND OPERATOR MODEL (lognormal v2) vs v1, sgd ===")
    print(f"  {'cell':12s} {'n':>3} {'PoD_F1':>7} {'sigwins':>8}")
    for k,r in out["operator_v2"].items():
        print(f"  {k:12s} {r['n']:>3} {r['pod_f1']:>7.3f} {r['n_sig_wins']:>3}/{r['n_comp']:<3}")

    # figure: grouped PoD vs best-AL bars per cell (A) + v1/v2 PoD-AL (B)
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    cells=list(out["base_learner"]); xs=np.arange(len(cells))
    podf=[out["base_learner"][c]["pod_f1"] for c in cells]
    alf=[out["base_learner"][c]["comp"].get("AL",{}).get("f1",np.nan) for c in cells]
    ax[0].bar(xs-0.2,podf,0.4,label="PoD",color="#1b7837")
    ax[0].bar(xs+0.2,alf,0.4,label="AL",color="#888")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(cells,rotation=45,ha="right",fontsize=8)
    ax[0].set_ylabel("final-window F1"); ax[0].set_title("(A) Base-learner robustness: PoD vs AL",fontsize=10); ax[0].legend(fontsize=8)
    # B
    b_cells=[("synth","synth"),("gas","gas")]; xs2=np.arange(2)
    for i,(lbl,ds) in enumerate(b_cells):
        for j,(ver,col) in enumerate([("v1","#4575b4"),("v2","#d73027")]):
            r=out["operator_v2"].get(f"{ds}/{ver}")
            if r:
                pod=r["pod_f1"]; al=r["comp"].get("AL",{}).get("f1",np.nan)
                ax[1].bar(i-0.2+0.4*j, pod-al, 0.36, color=col, label=(f"{ver}" if i==0 else None))
    ax[1].axhline(0,color="k",lw=1); ax[1].set_xticks(xs2); ax[1].set_xticklabels(["synth","gas"])
    ax[1].set_ylabel("PoD − AL final F1"); ax[1].set_title("(B) PoD advantage under operator v1 vs v2 (lognormal)",fontsize=10); ax[1].legend(fontsize=8,title="operator")
    plt.tight_layout(); fig.savefig(os.path.join(GEN,"fig_generalization.pdf")); fig.savefig(os.path.join(GEN,"fig_generalization.png"),dpi=140)
    print("\nwrote generalization_summary.json + fig_generalization.{pdf,png}")

if __name__=="__main__": main()
