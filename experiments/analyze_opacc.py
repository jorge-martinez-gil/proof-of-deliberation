"""Aggregate the operator-accuracy sweep: paired PoD-AL final-F1 vs corruption
severity (=1-a), per-point Wilcoxon, and the PoD>AL crossover accuracy. Writes
JSON + a 2-panel figure (absolute F1 curves; paired PoD-AL with CI + crossover)."""
import glob, os, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=_os.path.join(_ROOT,"out_opacc")
PREFIX={"synth":"Synth-Boundary","gas":"uci224_gas_drift"}
A_GRID=[0.05,0.15,0.25,0.35,0.45,0.55,0.65]

def load(ds, tag, m):
    fs=sorted(glob.glob(os.path.join(ROOT,ds,f"{PREFIX[ds]}_{tag}_{m}_run*.json")),
              key=lambda p:int(p.split('run')[-1].split('.')[0]))
    return np.array([json.load(open(f))["final_f1"] for f in fs])

def crossover(acc, diff):
    """linear-interpolate accuracy where paired diff crosses 0 (from + to -)."""
    for i in range(len(acc)-1):
        d0,d1=diff[i],diff[i+1]
        if d0>=0 and d1<0:
            a0,a1=acc[i],acc[i+1]
            return float(a0+(a1-a0)*(d0/(d0-d1)))
    return None

def analyze(ds):
    pts=[("a%.2f"%v,v) for v in A_GRID]
    accs=[]; pod_m=[]; al_m=[]; diff_m=[]; diff_ci=[]; pvals=[]
    for tag,a in pts:
        pod=load(ds,tag,"PoD"); al=load(ds,tag,"AL"); n=min(len(pod),len(al))
        d=pod[:n]-al[:n]
        accs.append(a); pod_m.append(float(pod[:n].mean())); al_m.append(float(al[:n].mean()))
        diff_m.append(float(d.mean())); diff_ci.append(float(1.96*d.std(ddof=1)/np.sqrt(n)))
        try: p=float(wilcoxon(pod[:n],al[:n],alternative="two-sided").pvalue) if not np.allclose(d,0) else 1.0
        except Exception: p=1.0
        pvals.append(p)
    # calibrated point
    podc=load(ds,"calib","PoD"); alc=load(ds,"calib","AL"); nc=min(len(podc),len(alc))
    dc=podc[:nc]-alc[:nc]
    calib=dict(p_gam=0.57,p_fat=0.53,acc_eff=0.55,pod=float(podc[:nc].mean()),
               al=float(alc[:nc].mean()),diff=float(dc.mean()),
               diff_ci=float(1.96*dc.std(ddof=1)/np.sqrt(nc)),
               p=float(wilcoxon(podc[:nc],alc[:nc],alternative="two-sided").pvalue) if not np.allclose(dc,0) else 1.0)
    xo=crossover(accs,diff_m)
    return dict(acc=accs,pod=pod_m,al=al_m,diff=diff_m,diff_ci=diff_ci,p=pvals,
                crossover_acc=xo,calib=calib,n=int(n))

def main():
    out={}
    for ds in ["synth","gas"]:
        out[ds]=analyze(ds)
    json.dump(out, open(os.path.join(ROOT,"opacc_summary.json"),"w"), indent=2)
    for ds in ["synth","gas"]:
        r=out[ds]; print(f"\n=== {ds} (n={r['n']}) crossover_acc={r['crossover_acc']} ===")
        for a,pod,al,d,ci,p in zip(r['acc'],r['pod'],r['al'],r['diff'],r['diff_ci'],r['p']):
            star="*" if p<0.05 else " "
            print(f"  a={a:.2f} sev={1-a:.2f}: PoD={pod:.3f} AL={al:.3f}  PoD-AL={d:+.3f}±{ci:.3f} p={p:.1e}{star}")
        c=r['calib']; print(f"  CALIB(0.57/0.53): PoD={c['pod']:.3f} AL={c['al']:.3f} PoD-AL={c['diff']:+.3f}±{c['diff_ci']:.3f} p={c['p']:.1e}")

    fig,ax=plt.subplots(2,2,figsize=(11,7.5))
    for i,ds in enumerate(["synth","gas"]):
        r=out[ds]; sev=[1-a for a in r['acc']]
        a0=ax[i][0]
        a0.plot(r['acc'],r['pod'],"o-",color="#1b7837",label="PoD")
        a0.plot(r['acc'],r['al'],"s--",color="#888",label="AL (unfiltered)")
        a0.scatter([r['calib']['acc_eff']],[r['calib']['pod']],marker="*",s=180,color="#1b7837",zorder=5,edgecolor="k")
        a0.scatter([r['calib']['acc_eff']],[r['calib']['al']],marker="*",s=180,color="#888",zorder=5,edgecolor="k")
        a0.set_xlabel("operator accuracy a (degraded regimes)"); a0.set_ylabel("final-window F1")
        a0.set_title(f"{ds}: PoD vs AL across corruption",fontsize=10); a0.legend(fontsize=8)
        a1=ax[i][1]
        d=np.array(r['diff']); ci=np.array(r['diff_ci'])
        a1.axhline(0,color="k",lw=1)
        a1.plot(r['acc'],d,"o-",color="#d73027",label="PoD − AL (paired)")
        a1.fill_between(r['acc'],d-ci,d+ci,color="#d73027",alpha=0.15)
        a1.scatter([r['calib']['acc_eff']],[r['calib']['diff']],marker="*",s=200,color="#762a83",zorder=5,edgecolor="k",label="calibrated (0.57/0.53)")
        if r['crossover_acc']:
            a1.axvline(r['crossover_acc'],color="#4575b4",ls=":",lw=1.5)
            a1.text(r['crossover_acc'],a1.get_ylim()[1]*0.9,f" crossover a≈{r['crossover_acc']:.2f}",fontsize=8,color="#4575b4")
        a1.set_xlabel("operator accuracy a"); a1.set_ylabel("PoD − AL final F1")
        a1.set_title(f"{ds}: PoD advantage vs corruption",fontsize=10); a1.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(ROOT,"fig_opacc_sweep.pdf")); fig.savefig(os.path.join(ROOT,"fig_opacc_sweep.png"),dpi=140)
    print("\nwrote opacc_summary.json + fig_opacc_sweep.{pdf,png}")

if __name__=="__main__": main()
