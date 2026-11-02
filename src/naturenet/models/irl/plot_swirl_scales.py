"""
Plots and summaries for run_swirl_scales_kfold.

    python -m naturenet.models.irl.plot_swirl_scales -y config/swirl/swirl_scales.yaml [--fold 0] [--n_seq 6]

Reads <out_dir>/<run_uid>_scales_cv.csv and the per-variant params / LL files. Writes <out_dir>/<run_uid>_scales_*:
  _ll_vs_order.png     held-out LL per step vs context order N, one line per latent configuration
                       (mean +/- SE over folds); star = effective memory N* (smallest N within 1 SE of the best);
                       dotted lines = the standard SWIRL S-1 / S-2 runs if their LL files exist
  _layer_gain.png      paired held-out gain vs the baseline (N1) per variant: fold dots, mean +/- SE
  _hazard.png          durations variants: hazard h_z(tau) and the implied dwell pmf vs the dwell-time histogram of
                       the decoded test paths (posterior argmax), with the baseline's (geometric) dwell histogram
  _regimes.png         hierarchy variants: P(regime) and decoded intention for the first n_seq test sequences
  _history.png         order >= 2: reward variance share by context depth and mean KL(pi(.|h) || pi(.|x_t)) by depth
  _switch_weights.png  switching-history variants: input-weight magnitude per feature block (and hazard weights)
  _summary.csv         per variant: test / train LL mean and SE, gain vs N1 (mean, SE, folds won), params, time
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from naturenet.models.irl import swirl_scales as sc

from sit_fuse.utils import read_yaml


def _se(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def _files(conf, tag, k):
    out_dir, run_uid, seed, K = conf["out_dir"], conf["run_uid"], conf["seed"], int(conf["n_hidden"])
    uid = f"{run_uid}_SC_{tag}"
    f = lambda s: os.path.join(out_dir, f"{uid}_{K}_{seed}{s}_KF_{k}.npz")  # noqa: E731
    return f("_scales"), f("_naturenet_S1_iter2_LL1"), f("_naturenet_S1_iter2_LL1_test")


def summarize(df):
    base = df[df["variant"] == "N1"].set_index("fold")["LL_test"] if "LL_test" in df else None
    out = []
    for tag, g in df.groupby("variant", sort=False):
        r = dict(variant=tag, order=int(g["order"].iloc[0]), latent=g["latent"].iloc[0], n_folds=len(g),
                 n_params=int(g["n_params"].mean()), fit_sec=float(g["fit_sec"].mean()),
                 LL_train=g["LL_train"].mean(), LL_train_se=_se(g["LL_train"]))
        if base is not None and g["LL_test"].notna().any():
            gain = g.set_index("fold")["LL_test"] - base.reindex(g["fold"]).values
            r.update(LL_test=g["LL_test"].mean(), LL_test_se=_se(g["LL_test"]), gain=float(gain.mean()),
                     gain_se=_se(gain.dropna()), folds_won=int((gain > 0).sum()))
        out.append(r)
    s = pd.DataFrame(out).sort_values(["latent", "order"])
    s["N_star"] = np.nan
    for lat, g in s.groupby("latent"):
        if "LL_test" not in g or g["LL_test"].isna().all():
            continue
        b = g.loc[g["LL_test"].idxmax()]
        thr = b["LL_test"] - (b["LL_test_se"] if np.isfinite(b["LL_test_se"]) else 0)
        s.loc[g.index, "N_star"] = int(g[g["LL_test"] >= thr]["order"].min())
    return s


def plot_ll_vs_order(s, conf, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for lat, g in s.groupby("latent"):
        g = g.sort_values("order")
        ln = ax.errorbar(g["order"], g["LL_test"], yerr=g["LL_test_se"], marker="o", capsize=3, label=lat)
        ns = g["N_star"].iloc[0]
        if np.isfinite(ns):
            row = g[g["order"] == ns].iloc[0]
            ax.plot(ns, row["LL_test"], "*", ms=14, color=ln[0].get_color(), mec="k")
    out_dir, run_uid, seed, K = conf["out_dir"], conf["run_uid"], conf["seed"], int(conf["n_hidden"])
    for name, suf, ls in (("SWIRL S-1", "_naturenet_S1_iter2_LL1_test", ":"), ("SWIRL S-2", "_naturenet_S2_iter2_LL2_test", "-.")):
        v = [float(np.load(p)["LL"]) for k in range(int(conf.get("n_folds", 5)))
             for p in [os.path.join(out_dir, f"{run_uid}_{K}_{seed}{suf}_KF_{k}.npz")] if os.path.exists(p)]
        if v:
            ax.axhline(np.mean(v), color="gray", ls=ls, lw=1, label=f"{name} run ({len(v)} folds)")
    ax.set_xlabel("context order N (action memory N-1 states, reward on N-grams)")
    ax.set_ylabel("held-out LL per step")
    ax.set_xticks(sorted(s["order"].unique()))
    ax.legend(fontsize=7, title="latent layers", title_fontsize=7)
    ax.set_title("Held-out LL vs order (star = N*, within 1 SE of best)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_layer_gain(df, s, path):
    base = df[df["variant"] == "N1"].set_index("fold")["LL_test"]
    tags = [t for t in s.sort_values("gain")["variant"] if t != "N1"]
    if not tags:
        return
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(tags) + 2), 4))
    for i, t in enumerate(tags):
        g = df[df["variant"] == t].set_index("fold")["LL_test"]
        d = (g - base.reindex(g.index)).dropna()
        r = s[s["variant"] == t].iloc[0]
        ax.bar(i, r["gain"], yerr=r["gain_se"], color="#4c72b0", alpha=0.7, capsize=3)
        ax.plot(np.full(len(d), i) + np.linspace(-0.15, 0.15, len(d)), d.values, "k.", ms=5)
        ax.text(i, r["gain"], f" {int(r['folds_won'])}/{len(d)}", fontsize=6,
                ha="center", va="bottom")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("held-out LL per step - N1 (paired by fold)")
    ax.set_title("Gain per decision layer (dots = folds; label = folds won)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _load(conf, tag, k):
    f_par, _, f_llt = _files(conf, tag, k)
    if not (os.path.exists(f_par) and os.path.exists(f_llt)):
        return None
    z = np.load(f_par, allow_pickle=True)
    r = dict(cfg=sc.cfg_from_json(z["sc_cfg"]), params=sc.params_from_npz(z))
    return r, z, np.load(f_llt, allow_pickle=True)


def plot_hazard(conf, tags, k, path):
    tags = [t for t in tags if "_dur-" in t]
    if not tags:
        return
    base = _load(conf, "N1", k)
    fig, axes = plt.subplots(2, len(tags), figsize=(4.2 * len(tags), 6), squeeze=False)
    for j, t in enumerate(tags):
        got = _load(conf, t, k)
        if got is None:
            continue
        r, z, llt = got
        h, pmf = sc.hazard_curves(r, z["phi_mean"] if r["cfg"].durations.hazard_features else None)
        R, K, D = h.shape
        tau = np.arange(1, D + 1)
        dw = sc.dwell_times(llt["jax_path_vmap"])
        dwb = sc.dwell_times(base[2]["jax_path_vmap"]) if base is not None else {}
        for zi in range(K):
            c = f"C{zi}"
            for ri in range(R):
                ls = ["-", "--", "-.", ":"][ri % 4]
                lab = f"z{zi}" + (f" r{ri}" if R > 1 else "")
                axes[0, j].plot(tau, h[ri, zi], color=c, ls=ls, label=lab)
                axes[1, j].plot(tau, pmf[ri, zi], color=c, ls=ls)
            if dw.get(zi):
                v = np.minimum(dw[zi], D)
                axes[1, j].hist(v, bins=np.arange(0.5, D + 1.5), density=True, color=c, alpha=0.25)
            if dwb.get(zi):
                v = np.minimum(dwb[zi], D)
                hb = np.bincount(v, minlength=D + 1)[1:] / len(v)
                axes[1, j].step(tau, hb, where="mid", color=c, ls=":", lw=0.8)
        axes[0, j].set_title(t, fontsize=8)
        axes[0, j].set_ylabel("hazard h_z(tau)")
        axes[1, j].set_ylabel("dwell pmf (line) / test dwell hist (bars)")
        axes[1, j].set_xlabel("steps in intention (bucket D = >= D)")
        axes[0, j].legend(fontsize=7)
    fig.suptitle(f"Intention durations, fold {k} (dotted = N1 decoded dwell, geometric model)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_regimes(conf, tags, k, n_seq, path):
    tags = [t for t in tags if re.search(r"_R\d+", t)]
    if not tags:
        return
    fig, axes = plt.subplots(n_seq, len(tags), figsize=(5 * len(tags), 1.4 * n_seq + 0.8), squeeze=False)
    for j, t in enumerate(tags):
        got = _load(conf, t, k)
        if got is None:
            continue
        _, _, llt = got
        gr, path_z = llt["gamma_r"], llt["jax_path_vmap"]
        for i in range(min(n_seq, gr.shape[0])):
            ax = axes[i, j]
            ax.stackplot(np.arange(gr.shape[1]), gr[i].T, alpha=0.6)
            ax.scatter(np.arange(gr.shape[1]), np.full(gr.shape[1], -0.08), c=path_z[i], cmap="tab10", s=4,
                       marker="s", vmin=0, vmax=9)
            ax.set_ylim(-0.15, 1)
            ax.set_yticks([])
            lab = str(llt["seq_group"][i]) if "seq_group" in llt.files else f"seq {i}"
            ax.set_ylabel(lab, fontsize=6)
        axes[0, j].set_title(f"{t}: P(regime) (areas), decoded z (strip)", fontsize=8)
        axes[-1, j].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(conf, tags, n_folds, path):
    tags = [t for t in tags if not t.startswith("N1")]
    if not tags:
        return
    fig, axes = plt.subplots(2, len(tags), figsize=(3.6 * len(tags), 5.5), squeeze=False)
    for j, t in enumerate(tags):
        sh, kl = [], []
        for k in range(n_folds):
            got = _load(conf, t, k)
            if got is not None:
                sh.append(got[1]["reward_depth_share"])
                kl.append(got[1]["history_kl"])
        if not sh:
            continue
        sh, kl = np.nanmean(sh, 0), np.nanmean(kl, 0)
        K, N = sh.shape
        bottom = np.zeros(K)
        for d in range(N):
            axes[0, j].bar(np.arange(K), sh[:, d], bottom=bottom, label=f"depth {d + 1}")
            bottom += np.nan_to_num(sh[:, d])
        for zi in range(K):
            axes[1, j].plot(np.arange(1, N + 1), kl[zi], "o-", label=f"z{zi}")
        axes[0, j].set_title(t, fontsize=8)
        axes[0, j].set_xlabel("intention")
        axes[0, j].set_ylabel("reward variance share")
        axes[1, j].set_xlabel("context depth")
        axes[1, j].set_ylabel("KL(pi(.|h) || pi(.|x_t))")
        axes[0, j].legend(fontsize=6)
        axes[1, j].legend(fontsize=6)
    fig.suptitle("History in what is done (fold mean)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _block(name):
    m = re.match(r"(lag\d+|decay[\d.]+)_s\d+", name)
    return m.group(1) if m else name


def plot_switch_weights(conf, tags, n_folds, path):
    tags = [t for t in tags if "_hist-" in t]
    if not tags:
        return
    fig, axes = plt.subplots(1, len(tags), figsize=(4.5 * len(tags), 3.8), squeeze=False)
    for j, t in enumerate(tags):
        blocks = {}
        for k in range(n_folds):
            got = _load(conf, t, k)
            if got is None:
                continue
            r, z, _ = got
            names = list(z["phi_names"])
            W = np.abs(np.asarray(r["params"]["W1f"])).sum(1)
            u = np.abs(np.asarray(r["params"].get("hz", {}).get("u", np.zeros((1, 0))))).sum(0)
            for i, nm in enumerate(names):
                b = blocks.setdefault(_block(nm), [0.0, 0.0])
                b[0] += W[i] / n_folds
                if len(u):
                    b[1] += u[i] / n_folds
        keys = list(blocks)
        x = np.arange(len(keys))
        axes[0, j].bar(x - 0.2, [blocks[b][0] for b in keys], 0.4, label="switch MLP |W|")
        if any(blocks[b][1] for b in keys):
            axes[0, j].bar(x + 0.2, [blocks[b][1] for b in keys], 0.4, label="hazard |u|")
        axes[0, j].set_xticks(x)
        axes[0, j].set_xticklabels(keys, rotation=30, ha="right", fontsize=7)
        axes[0, j].set_title(t, fontsize=8)
        axes[0, j].legend(fontsize=7)
    fig.suptitle("History in when intentions switch: weight per feature block (fold mean)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_swirl_scales(conf, fold=0, n_seq=6):
    out = os.path.join(conf["out_dir"], conf["run_uid"] + "_scales")
    df = pd.read_csv(out + "_cv.csv")
    s = summarize(df)
    s.to_csv(out + "_summary.csv", index=False)
    print(s.to_string(index=False))
    tags = list(s["variant"])
    n_folds = int(df["fold"].max()) + 1
    if "LL_test" in df:
        plot_ll_vs_order(s, conf, out + "_ll_vs_order.png")
        plot_layer_gain(df, s, out + "_layer_gain.png")
    plot_hazard(conf, tags, fold, out + "_hazard.png")
    plot_regimes(conf, tags, fold, n_seq, out + "_regimes.png")
    plot_history(conf, tags, n_folds, out + "_history.png")
    plot_switch_weights(conf, tags, n_folds, out + "_switch_weights.png")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_seq", type=int, default=6)
    a = ap.parse_args()
    plot_swirl_scales(read_yaml(a.yaml), a.fold, a.n_seq)
