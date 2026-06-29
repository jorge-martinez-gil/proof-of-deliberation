"""Aggregate coupling-ablation: degraded-phase final-window F1, paired Wilcoxon +
Cohen's d_z + BH-FDR for PoD vs each method; sweep curve (F1 + leakage vs eps).
Writes JSON + LaTeX macros + a 2-panel figure. Reuses the exact statistical
reductions from experiments/analyze_robustness.py."""
import sys, glob, os, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _os.path.join(_ROOT,"out_coupling")
PREFIX = {"synth": "Synth-Boundary", "gas": "uci224_gas_drift"}
BASELINE = 2000   # degraded phase begins here
K = 50
PANEL = ["AL", "PoD-NoGate", "PoD-NoCoupling", "PoD-NoVigilance"]  # vs PoD
EPS_MAIN = 0.20

def dz(diff):
    sd = diff.std(ddof=1)
    return float(diff.mean()/sd) if sd > 1e-12 else (float("inf") if diff.mean() > 0 else 0.0)

def bh(p):
    p = np.asarray(p); n = len(p); order = np.argsort(p); ranks = np.empty(n, int)
    ranks[order] = np.arange(1, n+1); adj = p*n/ranks
    adj_sorted = adj[order]; adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(adj_sorted, 0, 1); return out

def deg_final(path):
    df = pd.read_csv(path)
    deg = df[df.t >= BASELINE]["f1"].to_numpy()
    return float(np.mean(deg[-K:])) if len(deg) >= 1 else float("nan")

def load_scores(dataset, method):
    d = os.path.join(ROOT, "main", f"eps{EPS_MAIN:.2f}", dataset, "runs")
    fs = sorted(glob.glob(os.path.join(d, f"{PREFIX[dataset]}_{method}_run*.csv")),
                key=lambda p: int(p.split("run")[-1].split(".")[0]))
    return np.array([deg_final(f) for f in fs])

def load_leak(dataset, method):
    d = os.path.join(ROOT, "main", f"eps{EPS_MAIN:.2f}", dataset, "runs")
    fs = sorted(glob.glob(os.path.join(d, f"{PREFIX[dataset]}_{method}_run*_instr.json")),
                key=lambda p: int(p.split("run")[-1].split("_")[0]))
    return np.array([json.load(open(f))["deg_leak"] for f in fs])

def analyze_main(dataset):
    pod = load_scores(dataset, "PoD")
    res = {"dataset": dataset, "n_runs": int(len(pod)),
           "pod_deg_f1": float(pod.mean()), "pod_deg_f1_sd": float(pod.std(ddof=1)),
           "pod_leak": float(load_leak(dataset, "PoD").mean()), "methods": {}}
    pvals, names = [], []
    for m in PANEL:
        x = load_scores(dataset, m); n = min(len(pod), len(x)); d = pod[:n]-x[:n]
        if np.allclose(d, 0): p = 1.0
        else:
            try: p = float(wilcoxon(pod[:n], x[:n], zero_method="wilcox", alternative="two-sided").pvalue)
            except Exception: p = 1.0
        res["methods"][m] = {"deg_f1": float(x[:n].mean()), "deg_f1_sd": float(x[:n].std(ddof=1)),
                             "leak": float(load_leak(dataset, m).mean()),
                             "median_diff": float(np.median(d)), "d_z": dz(d), "p_raw": p}
        pvals.append(p); names.append(m)
    for m, pa in zip(names, bh(pvals)):
        res["methods"][m]["p_bh"] = float(pa)
        res["methods"][m]["pod_wins_sig"] = bool(pa < 0.05 and res["methods"][m]["d_z"] > 0)
    return res

def analyze_sweep(dataset):
    fs = glob.glob(os.path.join(ROOT, "sweep", dataset, f"{PREFIX[dataset]}_PoD_eps*_run*.json"))
    rows = [json.load(open(f)) for f in fs]
    df = pd.DataFrame(rows)
    g = df.groupby("eps").agg(f1_mean=("deg_final_f1","mean"), f1_sd=("deg_final_f1","std"),
                              leak_mean=("deg_leak","mean"), leak_sd=("deg_leak","std"),
                              n=("run","count")).reset_index().sort_values("eps")
    return g

def trajectory(dataset, method):
    d = os.path.join(ROOT, "main", f"eps{EPS_MAIN:.2f}", dataset, "runs")
    fs = sorted(glob.glob(os.path.join(d, f"{PREFIX[dataset]}_{method}_run*.csv")),
                key=lambda p: int(p.split("run")[-1].split(".")[0]))
    mats = [pd.read_csv(f) for f in fs]
    L = min(len(m) for m in mats); t = mats[0]["t"].to_numpy()[:L]
    M = np.vstack([m["f1"].to_numpy()[:L] for m in mats])
    return t, M.mean(0), 1.96*M.std(0, ddof=1)/np.sqrt(len(mats))

def main():
    results = {"epsilon_main": EPS_MAIN, "datasets": {}}
    for ds in ["synth", "gas"]:
        results["datasets"][ds] = {"main": analyze_main(ds),
                                   "sweep": analyze_sweep(ds).to_dict(orient="list")}
    os.makedirs(ROOT, exist_ok=True)
    json.dump(results, open(os.path.join(ROOT, "coupling_summary.json"), "w"), indent=2)

    # console
    for ds in ["synth", "gas"]:
        r = results["datasets"][ds]["main"]
        print(f"\n=== {ds}  PoD degraded-F1={r['pod_deg_f1']:.3f} (leak {r['pod_leak']:.3f}, n={r['n_runs']}) ===")
        for m in PANEL:
            mm = r["methods"][m]
            print(f"  PoD vs {m:16s}: {mm['deg_f1']:.3f} (leak {mm['leak']:.3f})  dz={mm['d_z']:+5.2f} "
                  f"pBH={mm['p_bh']:.2e} sig={mm['pod_wins_sig']}")
        g = results["datasets"][ds]["sweep"]
        print("  sweep eps->F1/leak:", [f"{e:+.2f}:{f:.3f}/{l:.3f}" for e,f,l in
              zip(g["eps"], g["f1_mean"], g["leak_mean"])])

    # figure: 2 rows (synth, gas) x 2 cols (trajectories, sweep)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    colors = {"AL":"#888","PoD":"#1b7837","PoD-NoGate":"#7570b3",
              "PoD-NoCoupling":"#d73027","PoD-NoVigilance":"#fc8d59"}
    for i, ds in enumerate(["synth", "gas"]):
        a = ax[i][0]
        for m in ["AL","PoD","PoD-NoGate","PoD-NoCoupling","PoD-NoVigilance"]:
            t, mu, ci = trajectory(ds, m)
            ls = "-" if m in ("PoD","AL") else "--"
            lw = 2.4 if m=="PoD" else (2.0 if m=="PoD-NoCoupling" else 1.3)
            a.plot(t, mu, ls, color=colors[m], lw=lw, label=m)
            a.fill_between(t, mu-ci, mu+ci, color=colors[m], alpha=0.12)
        a.axvline(BASELINE, color="k", ls=":", lw=1)
        a.text(BASELINE*0.5, a.get_ylim()[0], "baseline", ha="center", va="bottom", fontsize=8, color="gray")
        a.text(BASELINE*1.5, a.get_ylim()[0], "mimicry (decoupled timing)", ha="center", va="bottom", fontsize=8, color="gray")
        a.set_title(f"{ds}: F1 trajectory (ε={EPS_MAIN})", fontsize=10)
        a.set_xlabel("stream step"); a.set_ylabel("holdout F1")
        if i==0: a.legend(fontsize=7, loc="lower left", ncol=2)
        # sweep panel
        b = ax[i][1]
        g = analyze_sweep(ds)
        b.plot(g["eps"], g["f1_mean"], "o-", color="#1b7837", label="PoD degraded F1")
        b.fill_between(g["eps"], g["f1_mean"]-1.96*g["f1_sd"]/np.sqrt(g["n"]),
                       g["f1_mean"]+1.96*g["f1_sd"]/np.sqrt(g["n"]), color="#1b7837", alpha=0.15)
        b.set_ylabel("PoD degraded F1", color="#1b7837"); b.set_xlabel("coupling ε")
        b.axvline(-1.0, color="gray", ls=":", lw=1); b.text(-1.0, b.get_ylim()[1], " ε=-1\n(coupling off)", fontsize=7, va="top", color="gray")
        c = b.twinx()
        c.plot(g["eps"], g["leak_mean"], "s--", color="#d73027", label="bad-label leakage")
        c.set_ylabel("bad-label leakage", color="#d73027")
        b.set_title(f"{ds}: PoD vs coupling ε", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(ROOT, "fig_coupling_ablation.pdf"))
    fig.savefig(os.path.join(ROOT, "fig_coupling_ablation.png"), dpi=140)
    print("\nwrote", os.path.join(ROOT, "coupling_summary.json"), "and fig_coupling_ablation.{pdf,png}")

if __name__ == "__main__":
    main()
