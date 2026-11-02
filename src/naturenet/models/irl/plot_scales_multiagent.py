"""
Figures for run_scales_multiagent.

    python -m naturenet.models.irl.plot_scales_multiagent -y config/swirl/scales_multiagent.yaml [--K 2]
           [--coupling temporal] [--out <out_dir>/ma_scales_plots]

1  ma_scales_gain.png      held-out gain per step (coupled - independent) by scale variant and backend, per-animal
                           points, pooled bar; the time-shift null (mean +- sd of the pooled gain / step) if run
2  ma_scales_coupling.png  alpha and beta (fold medians) by variant, with the null replicates
3  ma_scales_hazard.png    durations variants: full-data hazard h(tau) per intention with w (1 - c[z]) = 0 (no partner
                           or partners agree) vs = 1 (partners in another intention)
4  ma_scales_regimes_<backend>_<variant>.png   regime / intention posteriors of the full-data coupled fit vs time
                           for each animal (first R2 variant per backend), with the partner weight w
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from naturenet.models.irl.run_hiavi_multiagent import read_conf  # noqa: E402
from naturenet.models.irl.run_scales_multiagent import scale_variants  # noqa: E402


def _pair_dir(out_dir, backend, name):
    return os.path.join(out_dir, f"multiagent_{backend}_{name}")


def load_cv(out_dir, backends, variants):
    frames = []
    for b in backends:
        for name, step, _ in variants:
            f = os.path.join(_pair_dir(out_dir, b, name), "ma_cv.csv")
            if os.path.exists(f):
                frames.append(pd.read_csv(f).assign(backend=b, scale_variant=name, step=step))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_gain(cv, K, coupling, path):
    d = cv[(cv["K"] == K) & (cv["variant"] == coupling)]
    if d.empty:
        return False
    obs, nul = d[d["null_id"] < 0], d[d["null_id"] >= 0]
    backends = sorted(obs["backend"].unique())
    names = list(dict.fromkeys(obs.sort_values("step")["scale_variant"]))
    fig, axes = plt.subplots(1, len(backends), figsize=(1.1 * len(names) + 3, 4), squeeze=False, sharey=True)
    for ax, b in zip(axes[0], backends):
        for i, n in enumerate(names):
            o = obs[(obs["backend"] == b) & (obs["scale_variant"] == n)]
            if o.empty:
                continue
            pooled = o["d_total"].sum() / max(o["n_test_steps"].sum(), 1)
            ax.bar(i, pooled, color="C%d" % int(o["step"].iloc[0] or 0), alpha=0.6)
            ax.scatter(np.full(len(o), i) + np.linspace(-0.2, 0.2, len(o)), o["d_per_step"], s=14, color="k", zorder=3)
            nu = nul[(nul["backend"] == b) & (nul["scale_variant"] == n)]
            if len(nu):
                g = nu.groupby("null_id").apply(lambda x: x["d_total"].sum() / max(x["n_test_steps"].sum(), 1))
                ax.errorbar(i + 0.3, g.mean(), yerr=g.std(), fmt="s", color="0.4", ms=4, capsize=2)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{b} (K={K}, {coupling})", fontsize=9)
    axes[0, 0].set_ylabel("held-out dLL / step (coupled - indep.)")
    fig.text(0.01, 0.01, "bars: pooled; dots: held-out animals; grey: time-shift null mean +- sd", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_coupling(cv, K, coupling, path):
    d = cv[(cv["K"] == K) & (cv["variant"] == coupling)]
    if d.empty:
        return False
    params = [p for p in ("alpha", "beta") if p in d and d[p].abs().sum() > 0]
    if not params:
        return False
    keys = list(dict.fromkeys(zip(d.sort_values("step")["backend"], d.sort_values("step")["scale_variant"])))
    fig, axes = plt.subplots(1, len(params), figsize=(0.8 * len(keys) + 3, 3.8), squeeze=False)
    for ax, p in zip(axes[0], params):
        for i, (b, n) in enumerate(keys):
            s = d[(d["backend"] == b) & (d["scale_variant"] == n)]
            ob = s[s["null_id"] < 0][p].median()
            nu = s[s["null_id"] >= 0].groupby("null_id")[p].median()
            if len(nu):
                ax.boxplot(nu.to_numpy(), positions=[i], widths=0.5, showfliers=False)
            ax.scatter([i], [ob], color="C3", zorder=3, s=30)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"{b.split('_')[0]}:{n}" for b, n in keys], rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(p, fontsize=10)
    fig.suptitle("fold medians: red = observed calendar, boxes = time-shift null", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_hazard(out_dir, backends, variants, K, coupling, path):
    panels = []
    for b in backends:
        for name, step, vc in variants:
            f = os.path.join(_pair_dir(out_dir, b, name), f"ma_final_K{K}.npz")
            if not vc.get("durations") or not os.path.exists(f):
                continue
            z = np.load(f, allow_pickle=True)
            if f"hazard_{coupling}" in z.files:
                panels.append((b, name, z[f"hazard_{coupling}"], z[f"hazard_partner_{coupling}"]
                               if f"hazard_partner_{coupling}" in z.files else None,
                               float(z[f"beta_{coupling}"]) if f"beta_{coupling}" in z.files else 0.0))
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.2), squeeze=False)
    for ax, (b, name, h, hp, beta) in zip(axes[0], panels):
        tau = np.arange(1, h.shape[-1] + 1)
        for k in range(h.shape[1]):
            ax.plot(tau, h[0, k], color=f"C{k}", label=f"z={k}, w(1-c)=0")
            if hp is not None:
                ax.plot(tau, hp[0, k], color=f"C{k}", ls="--", label=f"z={k}, w(1-c)=1")
        ax.set_title(f"{b.split('_')[0]}:{name}  beta={beta:+.2f}", fontsize=8)
        ax.set_xlabel("dwell tau (steps)")
        ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel("switch hazard (regime 0)")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_regimes(out_dir, backend, name, K, coupling, path, max_animals=4):
    f = os.path.join(_pair_dir(out_dir, backend, name), f"ma_final_K{K}.npz")
    if not os.path.exists(f):
        return False
    z = np.load(f, allow_pickle=True)
    if f"gamma_r_{coupling}" not in z.files:
        return False
    gr, g, w = z[f"gamma_r_{coupling}"], z[f"gamma_{coupling}"], z[f"w_{coupling}"]
    times, animal, orig = z["times"], z["animal"].astype(str), z["original"]
    ans = sorted(np.unique(animal[orig]))[:max_animals]
    fig, axes = plt.subplots(len(ans), 1, figsize=(10, 1.7 * len(ans)), squeeze=False, sharex=True)
    cmap = plt.get_cmap("tab10")
    for ax, an in zip(axes[:, 0], ans):
        for n in np.flatnonzero((animal == an) & orig):              # one segment per sequence (no joins)
            t = times[n]
            ax.fill_between(t, 0, w[n], color="0.88", step="mid", lw=0)
            if gr.shape[-1] > 1:
                ax.plot(t, gr[n, :, -1], color="k", lw=1)
            ax.scatter(t, np.full(len(t), -0.08), c=[cmap(k) for k in g[n].argmax(-1)], s=4, marker="s", lw=0)
        ax.set_ylim(-0.15, 1)
        ax.set_ylabel(an[-10:], fontsize=8)
    handles = [plt.Line2D([], [], color="k", label=f"P(regime {gr.shape[-1] - 1})"),
               plt.Rectangle((0, 0), 1, 1, color="0.88", label="partner weight w")] + \
              [plt.Line2D([], [], color=cmap(k), marker="s", ls="", label=f"MAP z={k}") for k in range(g.shape[-1])]
    axes[0, 0].legend(handles=handles, fontsize=6, ncol=len(handles), loc="upper right")
    axes[-1, 0].set_xlabel("time (days since 1970)")
    fig.suptitle(f"{backend}:{name} ({coupling}, K={K}) full-data coupled fit", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--coupling", default="temporal")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    conf = read_conf(a.yaml)
    m = conf.get("multiagent") or {}
    K = a.K or int((m.get("Ks") or [conf.get("n_hidden", 2)])[0])
    out_dir = conf["out_dir"]
    out = a.out or os.path.join(out_dir, "ma_scales_plots")
    os.makedirs(out, exist_ok=True)
    backends = list(m.get("scales_backends") or ["hiql_scales", "swirl_scales"])
    variants = scale_variants(conf)
    cv = load_cv(out_dir, backends, variants)
    done = []
    if not cv.empty:
        if plot_gain(cv, K, a.coupling, os.path.join(out, "ma_scales_gain.png")):
            done.append("ma_scales_gain.png")
        if plot_coupling(cv, K, a.coupling, os.path.join(out, "ma_scales_coupling.png")):
            done.append("ma_scales_coupling.png")
    if plot_hazard(out_dir, backends, variants, K, a.coupling, os.path.join(out, "ma_scales_hazard.png")):
        done.append("ma_scales_hazard.png")
    for b in backends:
        for name, step, vc in variants:
            if vc.get("hierarchy"):
                p = os.path.join(out, f"ma_scales_regimes_{b}_{name}.png")
                if plot_regimes(out_dir, b, name, K, a.coupling, p):
                    done.append(os.path.basename(p))
                break
    print("wrote", out, done)


if __name__ == "__main__":
    main()
