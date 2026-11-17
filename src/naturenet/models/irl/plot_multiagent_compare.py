"""
Side-by-side comparison of multi-agent runs (e.g. coupled HIAVI vs coupled SWIRL, with / without kinematics).

    python -m naturenet.models.irl.plot_multiagent_compare --dirs <out>/multiagent <out>/multiagent_swirl \
        [--labels HIAVI SWIRL] [--K 2 | --K 3 5] --out <out>/ma_compare

Each dir holds ma_summary.json + ma_cv.csv from run_hiavi_multiagent / run_swirl_multiagent. Writes
<out>.png (pooled held-out gain vs time-shift null, per-step gain per held-out animal, coupling alpha,
synchrony index) and <out>.csv (one row per run x variant).
Coupling forms: mixture variants show alpha in the coupling-strength panel; log-linear variants (align / matrix)
show align (beta, or the mean centred diagonal of B; > 0 = follow partners, < 0 = avoid them) with p_align, and
B_fro. Each K row also carries the selection-aware p (p_gain_max: best observed variant vs the null max over
null_variants). With --B, <out>_B.png draws the full-data B of every matrix variant from ma_final_K<K>.npz.
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
    sk = dict(s["K"][str(K)])
    sk["_dir"], sk["_K"] = d, K
    return s, cv[cv["K"] == K], sk


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
                             sync_chance=sk["sync"]["chance"], p_sync=sk["sync"]["p"],
                             coupling=e.get("coupling", "mixture"), align_median=e.get("align_median", np.nan),
                             align_full=e.get("full_data_align", np.nan), p_align=e.get("p_align", np.nan),
                             null_align_mean=e.get("null_align_mean", np.nan),
                             B_fro_median=e.get("B_fro_median", np.nan),
                             p_alpha=e.get("p_alpha", np.nan),
                             p_gain_max=(sk.get("selection") or {}).get("p_gain_max", np.nan),
                             best_variant=(sk.get("selection") or {}).get("best_variant")))
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
        ll = d["coupling"].fillna("mixture") != "mixture"
        med = np.where(ll, d["align_median"], d["alpha_median"])
        full = np.where(ll, d["align_full"], d["alpha_full"])
        ax.bar(x, med, w, color=f"C{j}", alpha=0.6, label=f"{lab} CV median", hatch=None)
        ax.scatter(x, full, color=f"C{j}", edgecolor="k", zorder=3, marker="D", s=30)
        for xi, m_, l_, p in zip(x, med, ll, np.where(ll, d["p_align"], d["p_alpha"])):
            if pd.notna(p) and pd.notna(m_):
                ax.text(xi, m_, f"{'b' if l_ else 'a'} p={p:.2f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=30 if len(variants) > 3 else 0, ha="right" if len(variants) > 3 else "center",
                       fontsize=7 if len(variants) > 3 else None)
    if (df["coupling"].fillna("mixture") == "mixture").all():
        ax.set_ylim(0, 1)
    ax.set_ylabel("alpha (mixture) / align (log-linear)\nbar: CV median, diamond: full data")
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


def plot_B(runs, labels, path):
    """Full-data B (K x K) of every matrix / align variant (rows: destination intention, cols: partner)."""
    items = []
    for lab, (s, cv, sk) in zip(labels, runs):
        f = os.path.join(sk["_dir"], f"ma_final_K{sk['_K']}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        for v, e in sk["variants"].items():
            if e.get("coupling", "mixture") != "mixture" and f"B_{v}" in z:
                items.append((f"{lab} {v}", np.asarray(z[f"B_{v}"], float)))
    if not items:
        print("no log-linear B in the ma_final npz files; skipping", path)
        return
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
    vmax = max(np.abs(B).max() for _, B in items) or 1.0
    for ax, (t, B) in zip(axes[0], items):
        im = ax.imshow(B, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for (i, j), b in np.ndenumerate(B):
            ax.text(j, i, f"{b:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(t, fontsize=8)
        ax.set_xlabel("partner intention")
        ax.set_ylabel("destination intention")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="B (full data)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--K", type=int, nargs="+", default=[2],
                    help="one K for every dir, or one per dir (e.g. --K 3 5 for HIQL K=3 vs SWIRL K=5); gains are "
                         "coupled - independent differences within a model, so they compare across K in nats")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", action="store_true", help="also draw <out>_B.png (full-data B of log-linear variants)")
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
    if a.B:
        plot_B(runs, labels, a.out + "_B.png")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
