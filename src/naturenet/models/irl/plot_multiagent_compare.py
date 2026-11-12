"""
Side-by-side comparison of multi-agent runs (e.g. coupled HIAVI vs coupled SWIRL, with / without kinematics).

    python -m naturenet.models.irl.plot_multiagent_compare --dirs <out>/multiagent <out>/multiagent_swirl \
        [--labels HIAVI SWIRL] [--K 2 | --K 3 5] --out <out>/ma_compare

Each dir holds ma_summary.json + ma_cv.csv from run_hiavi_multiagent / run_swirl_multiagent. Writes
<out>.png (pooled held-out gain vs time-shift null, per-step gain per held-out animal, coupling alpha,
synchrony index) and <out>.csv (one row per run x variant).
Note: held-out gains are differences of LLs from the same model family, so they are comparable across
families in nats; absolute LLs are not (HIAVI vs SWIRL likelihoods differ, and kinematic runs add the
step/turn factors to both the coupled and independent fits).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_run(d, K):
    with open(os.path.join(d, "ma_summary.json")) as f:
        s = json.load(f)
    cv = pd.read_csv(os.path.join(d, "ma_cv.csv"))
    print(cv["K"])
    return s, cv[cv["K"] == K], s["K"][str(K)]


def table(runs, labels, Ks):
    rows = []
    for lab, K, (s, cv, sk) in zip(labels, Ks, runs):
        for v, e in sk["variants"].items():
            rows.append(dict(run=lab, model=s.get("model", "HIAVI"), kinematics=bool(s.get("kinematics")), K=K,
                             variant=v, pooled_gain_nats=e["pooled_d_total"], gain_per_step=e["pooled_d_per_step"],
                             gain_context_nats=e["pooled_d_conc"], n_context_steps=e["n_conc_steps"],
                             null_mean=e.get("null_pooled_mean"), null_sd=e.get("null_pooled_sd"),
                             p_gain=e.get("p_gain"), alpha_median=e["alpha_median"],
                             alpha_full=e["full_data_alpha"], animals_improved=f"{e['animals_improved']}/{e['n_animals']}",
                             sync_obs=sk["sync"]["observed"], sync_null=sk["sync"]["null_mean"],
                             sync_chance=sk["sync"]["chance"], p_sync=sk["sync"]["p"]))
    return pd.DataFrame(rows)


def plot(runs, labels, Ks, path):
    df = table(runs, labels, Ks)
    variants = list(dict.fromkeys(df["variant"]))
    nr = len(labels)
    w = 0.8 / nr
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    ax = axes[0]
    for j, lab in enumerate(labels):
        d = df[df["run"] == lab].set_index("variant").reindex(variants)
        x = np.arange(len(variants)) + (j - (nr - 1) / 2) * w
        ax.bar(x, d["pooled_gain_nats"], w, color=f"C{j}", label=lab)
        if d["null_mean"].notna().any():
            ax.errorbar(x, d["null_mean"], yerr=d["null_sd"], fmt="_", color="k", capsize=4, ms=14)
        for xi, (g, p) in zip(x, zip(d["pooled_gain_nats"], d["p_gain"])):
            if pd.notna(p):
                ax.text(xi, g, f"p={p:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_ylabel("pooled held-out LL gain (nats)")
    ax.set_title("Coupled - independent (bars) vs null mean +/- sd", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    for j, (lab, (_, cv, _)) in enumerate(zip(labels, runs)):
        o = cv[cv["null_id"] < 0]
        for i, v in enumerate(variants):
            ov = o[o["variant"] == v]
            ax.scatter(np.full(len(ov), i + (j - (nr - 1) / 2) * w), ov["d_per_step"], color=f"C{j}",
                       edgecolor="k", s=36, label=lab if i == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_ylabel("held-out gain per step (nats)")
    ax.set_title("Per held-out animal", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    for j, lab in enumerate(labels):
        d = df[df["run"] == lab].set_index("variant").reindex(variants)
        x = np.arange(len(variants)) + (j - (nr - 1) / 2) * w
        ax.bar(x, d["alpha_median"], w, color=f"C{j}", alpha=0.6, label=f"{lab} CV median")
        ax.scatter(x, d["alpha_full"], color=f"C{j}", edgecolor="k", zorder=3, marker="D", s=30)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_ylim(0, 1)
    ax.set_ylabel("coupling alpha (bar: CV median, diamond: full data)")
    ax.set_title("Coupling strength", fontsize=10)

    ax = axes[3]
    s0 = df.drop_duplicates("run").set_index("run").loc[labels]
    x = np.arange(nr)
    ax.bar(x, s0["sync_obs"], 0.5, color=[f"C{j}" for j in range(nr)])
    ax.scatter(x, s0["sync_null"], color="k", marker="_", s=400, label="time-shift null mean", zorder=3)
    ax.scatter(x, s0["sync_chance"], color="k", marker="x", label="chance", zorder=3)
    for xi, (o, p) in zip(x, zip(s0["sync_obs"], s0["p_sync"])):
        if pd.notna(p):
            ax.text(xi, o, f"p={p:.2f}", ha="center", va="bottom", fontsize=7)
    lo = float(np.nanmin(s0[["sync_obs", "sync_null", "sync_chance"]].to_numpy()))
    ax.set_ylim(max(0, lo - 0.1), 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Synchrony index (independent fits)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    kk = sorted(set(int(k) for k in Ks))
    fig.suptitle("Multi-agent comparison, " + ", ".join(f"{lab}: K={k}" for lab, k in zip(labels, Ks))
                 if len(kk) > 1 else f"Multi-agent comparison, K={kk[0]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--K", type=int, nargs="+", default=[2],
                    help="one K for every dir, or one per dir (e.g. --K 3 5 for HIQL K=3 vs SWIRL K=5); gains are "
                         "coupled - independent differences within a model, so they compare across K in nats")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Ks = a.K * len(a.dirs) if len(a.K) == 1 else a.K
    if len(Ks) != len(a.dirs):
        raise SystemExit(f"--K: give one value or one per dir ({len(a.dirs)})")
    runs = [load_run(d, k) for d, k in zip(a.dirs, Ks)]
    labels = a.labels or [f"{r[0].get('model', 'HIAVI')}{'+kin' if r[0].get('kinematics') else ''}" for r in runs]
    if len(set(labels)) < len(labels):
        labels = [f"{lab} ({os.path.basename(os.path.normpath(d))})" for lab, d in zip(labels, a.dirs)]
    df = plot(runs, labels, Ks, a.out + ".png")
    df.to_csv(a.out + ".csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
