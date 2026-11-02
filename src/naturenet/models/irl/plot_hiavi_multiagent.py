"""
Figures and a results table for run_hiavi_multiagent.py.

    python -m naturenet.models.irl.plot_hiavi_multiagent -y config/swirl/hiavi_multiagent.yaml [--K 2]

Reads <out_dir>/<ma_subdir>/ma_*.{csv,npz,json}; writes into <ma_subdir>/plots/:
  ma_concurrency.png      track timelines coloured by partner availability w; partner distance / time-lag
                          histograms; steps with context per animal
  ma_cv_K<K>.png          held-out gain (coupled - independent) per animal and variant with the time-shift
                          null; pooled gain vs null; coupling alpha vs null
  ma_sync_K<K>.png        synchrony index vs null and chance; calendar-aligned dominant intention per animal
                          (independent and coupled fits)
  ma_results_K<K>.csv     manuscript table: one row per variant (pooled gain, per-step gain, p, alpha, n steps)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

VCOL = {"temporal": "#1f77b4", "proximity": "#d62728"}


def _dates(days):
    return pd.Timestamp("1970-01-01") + pd.to_timedelta(np.asarray(days), unit="D")


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")


def _short(an):
    return str(an).split("_")[-1]


def plot_concurrency(d, conf, path, final=None):
    conc = pd.read_csv(os.path.join(d, "ma_concurrency.csv"))
    pairs = np.load(os.path.join(d, "ma_pairs.npz"))
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
    ax0 = fig.add_subplot(gs[0, :])
    if final is not None:
        an_all = final["animal"]
        orig = final["original"]
        w = final.get("w_temporal")
        for i, an in enumerate(sorted(np.unique(an_all))):
            m = (an_all == an) & orig
            t = final["times"][m].ravel()
            col = w[m].ravel() if w is not None else np.zeros_like(t)
            sc = ax0.scatter(_dates(t), np.full(len(t), i), c=col, cmap="viridis", vmin=0, vmax=1, s=9)
        ax0.set_yticks(range(len(np.unique(an_all))))
        ax0.set_yticklabels([_short(a) for a in sorted(np.unique(an_all))])
        fig.colorbar(sc, ax=ax0, label="partner availability w (temporal)", pad=0.01)
        _date_axis(ax0)
        ax0.set_title("Original-track steps by calendar time")
    ax1 = fig.add_subplot(gs[1, 0])
    v = "temporal" if "temporal_dist_km" in pairs.files else pairs.files[0].rsplit("_dist_km", 1)[0]
    dist = pairs[f"{v}_dist_km"]
    if len(dist):
        ax1.hist(dist, bins=np.linspace(0, max(300, np.percentile(dist, 99)), 40), color="0.5")
        for x in (50, 100):
            ax1.axvline(x, color="k", ls="--", lw=0.8)
    ax1.set_xlabel("partner distance (km)")
    ax1.set_ylabel("step pairs")
    ax1.set_title(f"Partner distance ({v} kernel)")
    ax2 = fig.add_subplot(gs[1, 1])
    dt = pairs[f"{v}_dt_days"]
    if len(dt):
        ax2.hist(dt, bins=30, color="0.5")
    ax2.set_xlabel("|time lag| (days)")
    ax2.set_title("Partner time lag")
    ax3 = fig.add_subplot(gs[1, 2])
    piv = conc.pivot(index="animal", columns="variant", values="steps_with_context")
    tot = conc.groupby("animal")["steps"].first()
    x = np.arange(len(piv))
    ax3.bar(x - 0.3, tot.loc[piv.index], 0.2, color="0.8", label="all steps")
    for j, col in enumerate(piv.columns):
        ax3.bar(x - 0.1 + 0.2 * j, piv[col], 0.2, color=VCOL.get(col, f"C{j}"), label=f"with context ({col})")
    ax3.set_xticks(x)
    ax3.set_xticklabels([_short(a) for a in piv.index])
    ax3.legend(fontsize=7, frameon=False)
    ax3.set_title("Steps with partner context")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cv(cv, summ, K, path):
    cv = cv[cv["K"] == K]
    obs, nul = cv[cv["null_id"] < 0], cv[cv["null_id"] >= 0]
    variants = list(dict.fromkeys(obs["variant"]))
    animals = list(dict.fromkeys(obs["heldout"]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for j, v in enumerate(variants):
        o = obs[obs["variant"] == v].set_index("heldout").loc[animals]
        x = np.arange(len(animals)) + (j - (len(variants) - 1) / 2) * 0.25
        ax.scatter(x, o["d_per_step"], color=VCOL.get(v, f"C{j}"), s=50, zorder=3, label=v, edgecolor="k")
        n = nul[nul["variant"] == v]
        for i, an in enumerate(animals):
            vals = n.loc[n["heldout"] == an, "d_per_step"]
            if len(vals):
                ax.scatter(np.full(len(vals), x[i]) + np.random.default_rng(i).uniform(-0.05, 0.05, len(vals)),
                           vals, color=VCOL.get(v, f"C{j}"), s=8, alpha=0.3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(animals)))
    ax.set_xticklabels([f"{_short(a)}\n(n ctx={int(obs[obs.heldout == a].n_conc_steps.iloc[0])})" for a in animals],
                       fontsize=8)
    ax.set_ylabel("held-out LL gain per step (nats)")
    ax.set_title("Held-out gain by animal (large = observed, small = null)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    sk = summ["K"][str(K)]["variants"]
    for j, v in enumerate(variants):
        n = nul[nul["variant"] == v]
        if len(n):
            pooled = n.groupby("null_id")["d_total"].sum()
            ax.hist(pooled, bins=15, color=VCOL.get(v, f"C{j}"), alpha=0.35, label=f"{v} null")
        ax.axvline(sk[v]["pooled_d_total"], color=VCOL.get(v, f"C{j}"), lw=2,
                   label=f"{v} obs" + (f" (p={sk[v]['p_gain']:.3f})" if "p_gain" in sk[v] else ""))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("pooled held-out LL gain (nats, all held-out steps)")
    ax.set_title("Pooled gain vs time-shift null", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    for j, v in enumerate(variants):
        o = obs[obs["variant"] == v]
        ax.scatter(np.full(len(o), j), o["alpha"], color=VCOL.get(v, f"C{j}"), s=50, edgecolor="k", zorder=3)
        n = nul[nul["variant"] == v]
        if len(n):
            ax.violinplot(n["alpha"], positions=[j + 0.35], widths=0.3, showextrema=False)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_ylabel("coupling alpha (per fold)")
    ax.set_title("Coupling strength: observed (dots) vs null (violin)", fontsize=10)
    fig.suptitle(f"Coupled-intention HIAVI, K={K}, leave-one-animal-out", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sync(sync, summ, final, K, path, coupled="temporal"):
    s = sync[sync["K"] == K]
    sk = summ["K"][str(K)]["sync"]
    fig = plt.figure(figsize=(15, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.6, 1.6])
    ax = fig.add_subplot(gs[0])
    ax.hist(s.loc[s["kind"] == "time_shift", "agreement"], bins=25, color="0.6", label="time-shift null")
    ax.axvline(sk["observed"], color="k", lw=2, label=f"observed (p={sk['p']:.3f})")
    ax.axvline(sk["chance"], color="k", ls=":", label="chance")
    ax.set_xlabel("mean expected intention agreement")
    ax.set_title(f"Synchrony index (n={sk['n_steps']} steps)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    cmap = plt.get_cmap("tab10")
    for p, (gk, ttl) in enumerate([("gamma_indep", "independent HIAVI"), (f"gamma_{coupled}", f"coupled ({coupled})")]):
        ax = fig.add_subplot(gs[p + 1])
        if gk not in final:
            continue
        an_all, orig = final["animal"], final["original"]
        animals = sorted(np.unique(an_all))
        for i, an in enumerate(animals):
            m = (an_all == an) & orig
            t = np.floor(final["times"][m].ravel())
            g = final[gk][m].reshape(-1, K)
            df = pd.DataFrame(g).assign(day=t).groupby("day").mean()
            dom = df.to_numpy().argmax(1)
            conf_ = df.to_numpy().max(1)
            ax.scatter(_dates(df.index.to_numpy()), np.full(len(df), i), c=[cmap(z) for z in dom],
                       s=14, alpha=np.clip(conf_, 0.2, 1))
        ax.set_yticks(range(len(animals)))
        ax.set_yticklabels([_short(a) for a in animals])
        _date_axis(ax)
        ax.set_title(f"Daily dominant intention: {ttl}", fontsize=10)
        for k in range(K):
            ax.scatter([], [], color=cmap(k), label=f"z={k}")
        ax.legend(fontsize=7, frameon=False, ncol=K, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def results_table(summ, K, path):
    rows = []
    sk = summ["K"][str(K)]
    for v, e in sk["variants"].items():
        rows.append(dict(K=K, variant=v, pooled_gain_nats=e["pooled_d_total"], gain_per_step=e["pooled_d_per_step"],
                         gain_context_steps_nats=e["pooled_d_conc"], n_context_steps=e["n_conc_steps"],
                         animals_improved=f"{e['animals_improved']}/{e['n_animals']}",
                         alpha_median_cv=e["alpha_median"], alpha_full_data=e["full_data_alpha"],
                         effective_coupling=e.get("effective_coupling"),
                         p_gain=e.get("p_gain"), p_alpha=e.get("p_alpha"), n_null=e.get("n_null"),
                         sync_obs=sk["sync"]["observed"], sync_null_mean=sk["sync"]["null_mean"],
                         sync_chance=sk["sync"]["chance"], p_sync=sk["sync"]["p"]))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    return df


def run_plots(yml_conf, Ks=None):
    m = yml_conf.get("multiagent") or {}
    d = os.path.join(yml_conf["out_dir"], m.get("ma_subdir", "multiagent"))
    pdir = os.path.join(d, "plots")
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(d, "ma_summary.json")) as f:
        summ = json.load(f)
    cv = pd.read_csv(os.path.join(d, "ma_cv.csv"))
    sync = pd.read_csv(os.path.join(d, "ma_sync.csv"))
    Ks = Ks or [int(k) for k in summ["K"]]
    final0 = None
    for K in Ks:
        fp = os.path.join(d, f"ma_final_K{K}.npz")
        final = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else None
        final0 = final0 or final
        plot_cv(cv, summ, K, os.path.join(pdir, f"ma_cv_K{K}.png"))
        if final is not None:
            plot_sync(sync, summ, final, K, os.path.join(pdir, f"ma_sync_K{K}.png"))
        results_table(summ, K, os.path.join(pdir, f"ma_results_K{K}.csv"))
    plot_concurrency(d, m, os.path.join(pdir, "ma_concurrency.png"), final0)
    print("wrote", pdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--K", type=int, nargs="*", default=None)
    args = ap.parse_args()
    with open(args.yaml) as f:
        run_plots(yaml.safe_load(f), args.K)
