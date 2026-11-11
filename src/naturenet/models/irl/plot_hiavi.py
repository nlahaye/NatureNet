"""
Plots, CV summary and per-animal time-series export for HIAVI runs (replaces plot_hiql.py, which laid the
states out on an arbitrary sqrt(S) grid and read hard-coded fold_11 / fold_0 / range(5) paths).

HIAVI-specific figures (written to plot_dir, default <out_dir>/plots, tag _hiavi_s1):
- hiavi_cv_ll.png / hiavi_cv_ll_summary.csv: train / test LL per step vs K (mean +- sd over folds and runs)
  from ll_hiavi.csv (or rebuilt from the _hiavi_LL_KF_k files);
- hiavi_latent_dynamics: initial distribution and K x K intention transition matrix, with expected dwell
  times 1 / (1 - Lambda_kk);
- hiavi_reward_sa_k: IAVI reward r_k(s, a) per intention (S x A, shared colour scale);
- hiavi_value: V_k(s) = max_a q_k(s, a) and policy entropy H(pi_k(. | s)) per state;
- hiavi_em: EM log-likelihood history; hiavi_occupancy: fraction of decoded steps and run lengths per
  intention (from the train / test latent files);
- with hiavi_env_map (an (H, W) .npy of abstract env states) and env_lon_bounds / env_lat_bounds: geographic
  maps of the policy-weighted reward E_pi_k[r_k(s, a)] and V_k (IAVI fixes r only up to a per-state shift,
  with mean_a r_k(s, a) = 0, so state maps use E_pi[r]; differences across actions within a state and V are
  the interpretable quantities).
Then plot_swirl_new.run_plots with swirl_model = hiavi: time series per animal (series_hiavi_s1/), tracks
coloured by intention, densities, input-feature plots, reward / env GIFs, state_reward_hiavi_s1.npy. The
exported series feed correlation_analysis_shark.py and exploit_explore_init.py unchanged.

Config: the plot_swirl_new keys (out_dir, run_uid, seed, n_hidden, n_states, df_dir, df_run_uid, actions,
positions, ...) plus hiavi_env_map, hiavi_plot_series (True), hiavi_cv_plot (True), and optionally
min_latent / max_latent / num_runs (for the CV summary when ll_hiavi.csv is missing).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from naturenet.models.irl.plot_swirl_new import map_reward_heat_to_env, run_plots  # noqa: E402
from naturenet.models.irl.swirl_data import as_list  # noqa: E402
from naturenet.models.irl.swirl_track_series import load_latents, swirl_file  # noqa: E402

TAG = "_hiavi_s1"


def load_hiavi(yml_conf, tag=""):
    f = swirl_file(yml_conf, "hiavi", "s1", "reward", tag)
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} not found (run run_hiavi_kfold.py --holdout / run_hiql_hiavi.py first)")
    p = np.load(f, allow_pickle=True)
    return {k: p[k] for k in p.files}


def policy_entropy(pis):
    p = np.clip(pis, 1e-300, 1)
    return -(pis * np.log(p)).sum(-1)


def cv_summary(yml_conf):
    """Rows (K, seed, run_uid, fold, train_ll, test_ll) of the K-fold runs; summary per K."""
    path = os.path.join(yml_conf["out_dir"], "ll_hiavi.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        from naturenet.models.irl.run_hiavi_kfold import summary_from_files
        from naturenet.models.irl.swirl_data import iter_loop_configs
        confs = iter_loop_configs(yml_conf) if "min_latent" in yml_conf else [yml_conf]
        df = pd.concat([summary_from_files(c) for c in confs] or [pd.DataFrame()], ignore_index=True)
    if df.empty:
        return df, df
    df = df[df["fold"].astype(str) != "holdout"]
    summ = df.groupby("K").agg(train_mean=("train_ll", "mean"), train_sd=("train_ll", "std"),
                               test_mean=("test_ll", "mean"), test_sd=("test_ll", "std"),
                               n=("test_ll", "size")).reset_index()
    return df, summ


def plot_cv(summ, out_dir):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for col, lab in (("train", "train"), ("test", "test (held-out animals)")):
        ax.errorbar(summ["K"], summ[col + "_mean"], yerr=summ[col + "_sd"].fillna(0), marker="o", capsize=3,
                    label=lab)
    ax.set_xticks(summ["K"].astype(int).tolist())
    ax.set_xlabel("number of intentions K")
    ax.set_ylabel("log-likelihood per step")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "hiavi_cv_ll.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_latent_dynamics(p, out_dir, tag=TAG):
    pi0, Lam = np.exp(p["logp_init"]), np.exp(p["logp_tr"])
    K = len(pi0)
    dwell = 1.0 / np.maximum(1.0 - np.diag(Lam), 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(5 + 0.6 * K, 3 + 0.3 * K), gridspec_kw={"width_ratios": [1, K]})
    fig.subplots_adjust(wspace=0.35)
    axes[0].imshow(pi0[:, None], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(K):
        axes[0].text(0, i, f"{pi0[i]:.2f}", ha="center", va="center", fontsize=7)
    axes[0].set_xticks([])
    axes[0].set_yticks(range(K))
    axes[0].set_ylabel("intention")
    axes[0].set_title("initial")
    im = axes[1].imshow(Lam, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            axes[1].text(j, i, f"{Lam[i, j]:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if Lam[i, j] > 0.5 else "black")
        axes[1].text(K - 0.4, i, f" {dwell[i]:.3g}" if dwell[i] < 1e4 else " >1e4", ha="left", va="center",
                     fontsize=7)
    axes[1].text(K - 0.4, -0.65, " dwell", ha="left", va="center", fontsize=7)
    axes[1].set_xticks(range(K))
    axes[1].set_yticks(range(K))
    axes[1].set_xlabel("to intention")
    axes[1].set_ylabel("from intention")
    axes[1].set_title("transitions")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.18)
    fig.savefig(os.path.join(out_dir, f"hiavi_latent_dynamics{tag}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_reward_sa(r, out_dir, tag=TAG):
    K, S, A = r.shape
    vmin, vmax = np.nanmin(r), np.nanmax(r)
    ncol = min(K, 4)
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 0.5 + max(2.5, S * 0.06) * nrow), squeeze=False)
    for k in range(nrow * ncol):
        ax = axes[k // ncol, k % ncol]
        if k >= K:
            ax.axis("off")
            continue
        im = ax.imshow(r[k], aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"intention {k}")
        ax.set_xlabel("action")
        ax.set_ylabel("state")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="IAVI reward r(s, a)")
    fig.savefig(os.path.join(out_dir, f"hiavi_reward_sa{tag}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_value_entropy(p, out_dir, tag=TAG):
    V = p["q"].max(-1)
    H = policy_entropy(p["pis"])
    fig, axes = plt.subplots(2, 1, figsize=(max(6, V.shape[1] * 0.15), 2 + 0.8 * V.shape[0]), sharex=True)
    for ax, M, lab, cm in ((axes[0], V, "V(s) = max_a q(s, a)", "viridis"),
                           (axes[1], H, "policy entropy (nats)", "magma")):
        im = ax.imshow(M, aspect="auto", cmap=cm, interpolation="nearest")
        ax.set_ylabel("intention")
        ax.set_yticks(range(M.shape[0]))
        fig.colorbar(im, ax=ax, label=lab)
    axes[1].set_xlabel("environment state")
    fig.savefig(os.path.join(out_dir, f"hiavi_value{tag}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return V, H


def plot_em(p, out_dir, tag=TAG):
    h = np.asarray(p.get("ll_history", []))
    if h.size == 0:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(h, marker=".")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("train log-likelihood per step")
    if "restart_train_ll" in p:
        ax.set_title(f"best of {len(p['restart_train_ll'])} restarts")
    fig.savefig(os.path.join(out_dir, f"hiavi_em{tag}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_lengths(z):
    z = np.asarray(z)
    cut = np.r_[0, np.nonzero(np.diff(z))[0] + 1, len(z)]
    return z[cut[:-1]], np.diff(cut)


def plot_occupancy(yml_conf, K, out_dir, tag=TAG):
    splits = tuple(as_list(yml_conf.get("splits", ["train", "test"])))
    lat = load_latents(yml_conf, "hiavi", "s1", splits)
    occ, runs = {}, {k: [] for k in range(K)}
    for sp in np.unique(lat["split"]):
        # padded steps (segment_policy: variable) are decoded as -1: drop them before counting
        zz = [np.ravel(z).astype(np.int64) for z, s in zip(lat["zs"], lat["split"]) if s == sp]
        zz = [z[z >= 0] for z in zz]
        allz = np.concatenate(zz) if zz else np.zeros(0, dtype=np.int64)
        occ[sp] = np.bincount(allz, minlength=K) / max(len(allz), 1)
        for z in zz:
            if z.size == 0:
                continue
            ks, ls = run_lengths(z)
            for k, l in zip(ks, ls):
                runs[int(k)].append(int(l))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    w = 0.8 / max(len(occ), 1)
    for i, (sp, o) in enumerate(occ.items()):
        axes[0].bar(np.arange(K) + i * w, o, width=w, label=sp)
    axes[0].set_xticks(np.arange(K) + w * (len(occ) - 1) / 2)
    axes[0].set_xticklabels(range(K))
    axes[0].set_xlabel("intention")
    axes[0].set_ylabel("fraction of steps")
    axes[0].legend()
    axes[1].boxplot([runs[k] if runs[k] else [0] for k in range(K)], positions=range(K))
    axes[1].set_xlabel("intention")
    axes[1].set_ylabel("run length (steps)")
    fig.savefig(os.path.join(out_dir, f"hiavi_occupancy{tag}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return occ


def plot_env_value_maps(env_map, maps, labels, out_dir, lon_bounds, lat_bounds, tag=TAG):
    """maps: list of (name, (K, S)); one row per map, one panel per intention, shared scale per row."""
    K = maps[0][1].shape[0]
    extent = [min(lon_bounds), max(lon_bounds), min(lat_bounds), max(lat_bounds)]
    fig, axes = plt.subplots(len(maps), K, figsize=(3.5 * K, 3 * len(maps)), squeeze=False)
    for r_, (name, M) in enumerate(maps):
        vmin, vmax = np.nanmin(M), np.nanmax(M)
        for k in range(K):
            im = axes[r_, k].imshow(map_reward_heat_to_env(M[k], env_map), extent=extent, vmin=vmin, vmax=vmax,
                                    cmap="viridis")
            axes[r_, k].set_title(f"{name}, intention {k}", fontsize=9)
        fig.colorbar(im, ax=axes[r_].tolist(), label=labels[r_])
    fig.savefig(os.path.join(out_dir, f"hiavi_env_maps{tag}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_hiavi_plots(yml_conf):
    conf = dict(yml_conf)
    conf.setdefault("n_hidden", conf.get("n_hidden_init"))
    out_dir = conf.get("plot_dir", os.path.join(conf["out_dir"], "plots"))
    os.makedirs(out_dir, exist_ok=True)
    if conf.get("hiavi_cv_plot", True):
        rows, summ = cv_summary(conf)
        if len(summ):
            summ.to_csv(os.path.join(out_dir, "hiavi_cv_ll_summary.csv"), index=False)
            plot_cv(summ, out_dir)
    p = load_hiavi(conf)
    K = p["r"].shape[0]
    if K != int(conf["n_hidden"]):
        raise ValueError(f"model file has K={K}, config n_hidden={conf['n_hidden']}")
    plot_latent_dynamics(p, out_dir)
    plot_reward_sa(p["r"], out_dir)
    V, H = plot_value_entropy(p, out_dir)
    np.save(os.path.join(out_dir, f"state_value{TAG}.npy"), V)
    plot_em(p, out_dir)
    plot_occupancy(conf, K, out_dir)
    if conf.get("hiavi_env_map"):
        env_map = np.load(conf["hiavi_env_map"])
        lonb = conf.get("env_lon_bounds", conf.get("lon_bounds"))
        latb = conf.get("env_lat_bounds", conf.get("lat_bounds"))
        er = (p["r"] * p["pis"]).sum(-1)
        plot_env_value_maps(env_map, [("E_pi reward", er), ("value", V)], ["E_pi[r(s, a)]", "V(s)"],
                            out_dir, lonb, latb)
    if conf.get("hiavi_plot_series", True):
        run_plots(dict(conf, swirl_model="hiavi", swirl_orders=["s1"]))


if __name__ == '__main__':
    from sit_fuse.utils import read_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    run_hiavi_plots(read_yaml(args.yaml))
