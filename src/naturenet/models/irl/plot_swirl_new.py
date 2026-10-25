"""
Plots and per-animal time-series export for a final SWIRL run (linear or NN, S-1 and/or S-2).

Rewritten; the extraction now lives in swirl_track_series.py. Main fixes vs. the old version:
- latent rows mapped to tracks by the sequence ids saved with each LL file (was a running counter that
  skipped / duplicated rows); tracks loaded with the training loader; no hard-coded uid / paths;
- every exported column computed from its own quantity (the old copy-paste blocks wrote the env series
  into distance_traveled, reward_likelihood, turn_angle, ship_density, bathymetry, distance_to_coast);
- haversine step length / speed from lat / lon and dt; turn angles from forward differences, NaN when
  stationary; equal column lengths; timestamps parsed with pd.to_datetime;
- S-2 linear reward reshaped (K, S, M) with an (S, M) mask from prev_state_map (was (K, S, n_actions) and
  params3 instead of params4); NN S-2 reward table (K, S, S, A);
- path[1:-1] alignment everywhere (latent t <-> fix seq_start + 1 + t), maps masked with < 0 (not <= 0,
  which hid cluster 0), no per-mode min-max scaling (reward_normalize: none | global);
- maps drawn in geographic coordinates on their own extent (env_lat_bounds / env_lon_bounds), so env
  maps and tracks are never indexed with the movement-grid cell indices;
- plot_density_inputs / plot_time_series_input share the generic code (were broken: NameError, overwritten
  dicts, inputs off by one); no prints in loops, no plt.show, GIFs at 120 dpi with PillowWriter.

Config: out_dir, run_uid, seed, n_hidden (or n_hidden_init), n_states, df_dir, df_run_uid, actions,
positions, lon_bounds, lat_bounds, swirl_model (linear | nn), swirl_orders ([s1, s2]), splits
([train, test]), plot_trajs, plot_gifs, plot_inputs, input_keys, static_layers, envs (pickle
{uid: [segment][fix] -> (H, W)} for GIFs), env_lat_bounds / env_lon_bounds, max_gifs (5), plot_dir.
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

from naturenet.models.irl.swirl_data import as_list  # noqa: E402
from naturenet.models.irl.swirl_track_series import (  # noqa: E402
    build_track_series,
    export_series,
    load_latents,
    load_reward_model,
)

DEFAULT_COLUMNS = ["env", "step_km", "speed_kmh", "turn_angle", "reward_likelihood"]


def _colors(K):
    cmap = plt.get_cmap("tab10" if K <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(K)]


def map_reward_heat_to_env(state_reward_k, env_map):
    """Reward of each cell's env state for one mode; invalid / land cells (< 0) -> NaN (cluster 0 kept)."""
    env_map = np.asarray(env_map)
    out = np.full(env_map.shape, np.nan)
    ok = (env_map >= 0) & (env_map < len(state_reward_k))
    out[ok] = np.asarray(state_reward_k)[env_map[ok].astype(int)]
    return out


def plot_reward_heatmaps(state_reward, out_dir, tag=""):
    """(K, S) state reward, one shared colour scale so modes are comparable."""
    fig, ax = plt.subplots(figsize=(max(6, state_reward.shape[1] * 0.15), 1 + 0.4 * state_reward.shape[0]))
    im = ax.imshow(state_reward, aspect="auto", cmap="viridis")
    ax.set_xlabel("environment state")
    ax.set_ylabel("latent mode")
    ax.set_yticks(range(state_reward.shape[0]))
    fig.colorbar(im, ax=ax, label="reward")
    fig.savefig(os.path.join(out_dir, f"reward_heatmap{tag}.png"), bbox_inches="tight")
    plt.close(fig)


def plot_trajs(series, out_dir, K, lon_bounds=None, lat_bounds=None, tag=""):
    cols = _colors(K)
    fig, ax = plt.subplots(figsize=(7, 6))
    for uid, d in series.items():
        ax.cla()
        for seg, ds in d.groupby("segment"):
            ax.plot(ds["longitude"], ds["latitude"], color="0.7", lw=0.5, zorder=1)
        ax.scatter(d["longitude"], d["latitude"], c=[cols[int(z)] for z in d["latent"]], s=6, zorder=2)
        if lon_bounds is not None:
            ax.set_xlim(*sorted(lon_bounds))
        if lat_bounds is not None:
            ax.set_ylim(*sorted(lat_bounds))
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(f"{uid}{tag}")
        for k in range(K):
            ax.scatter([], [], color=cols[k], label=f"mode {k}")
        ax.legend(loc="best", fontsize=7)
        fig.savefig(os.path.join(out_dir, f"traj_{uid}{tag}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_time_series(series, out_dir, columns, K, tag=""):
    """One figure per animal: each column vs Time, coloured by decoded mode (gaps between segments kept)."""
    cols = _colors(K)
    columns = [c for c in columns if any(c in d for d in series.values())]
    if not columns:
        return
    for uid, d in series.items():
        fig, axes = plt.subplots(len(columns), 1, figsize=(10, 1.8 * len(columns)), sharex=True, squeeze=False)
        c = [cols[int(z)] for z in d["latent"]]
        for ax, col in zip(axes[:, 0], columns):
            if col not in d:
                continue
            ax.scatter(d["Time"], d[col], c=c, s=4)
            ax.set_ylabel(col, fontsize=8)
        axes[-1, 0].set_xlabel("hours since first fix")
        axes[0, 0].set_title(f"{uid}{tag}")
        fig.savefig(os.path.join(out_dir, f"time_series_{uid}{tag}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_time_series_input(series, out_dir, input_keys, K, tag=""):
    plot_time_series(series, out_dir, input_keys, K, tag="_inputs" + tag)


def plot_density(series, columns, K, out_dir, tag="", bins=40):
    """Per column: distribution under each decoded mode (all animals pooled, NaN excluded)."""
    import pandas as pd
    allv = pd.concat(series.values(), ignore_index=True)
    cols = _colors(K)
    for col in columns:
        if col not in allv:
            continue
        v = allv[col].to_numpy(dtype=float)
        fin = np.isfinite(v)
        if fin.sum() < 2:
            continue
        edges = np.histogram_bin_edges(v[fin], bins=bins)
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in range(K):
            vk = v[(allv["latent"].to_numpy() == k) & fin]
            if len(vk):
                ax.hist(vk, bins=edges, density=True, histtype="step", color=cols[k], label=f"mode {k} (n={len(vk)})")
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(out_dir, f"density_{col}{tag}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_density_inputs(series, input_keys, K, out_dir, tag=""):
    plot_density(series, input_keys, K, out_dir, tag="_inputs" + tag)


def wrap_to_pi(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def plot_env_maps(env_map, state_reward, uid, out_dir, lon_bounds, lat_bounds, tag=""):
    """Reward heat map of one env map under every mode (shared colour scale)."""
    K = state_reward.shape[0]
    extent = [min(lon_bounds), max(lon_bounds), min(lat_bounds), max(lat_bounds)]
    vmin, vmax = np.nanmin(state_reward), np.nanmax(state_reward)
    fig, axes = plt.subplots(1, K, figsize=(4 * K, 3.5), squeeze=False)
    for k, ax in enumerate(axes[0]):
        im = ax.imshow(map_reward_heat_to_env(state_reward[k], env_map), extent=extent, vmin=vmin, vmax=vmax,
                       cmap="viridis")
        ax.set_title(f"mode {k}")
    fig.colorbar(im, ax=axes[0].tolist(), label="reward")
    fig.savefig(os.path.join(out_dir, f"env_reward_maps_{uid}{tag}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def animate_segment(d_seg, env_maps, state_reward, out_file, lon_bounds, lat_bounds, K, dpi=120, fps=4,
                    kind="reward"):
    """
    GIF over the labelled fixes of one segment: reward heat map (kind="reward") or env cluster map
    (kind="env") at that fix, and the animal position coloured by decoded mode. env_maps[pos] is the map at fix pos, so frame t uses the same fix as the
    latent (no [1:-1] offset). Maps are drawn on their own extent (env bounds).
    """
    cols = _colors(K)
    extent = [min(lon_bounds), max(lon_bounds), min(lat_bounds), max(lat_bounds)]
    if kind == "reward":
        frame = lambda k, m: map_reward_heat_to_env(state_reward[k], m)  # noqa: E731
        vmin, vmax, cmap, label = np.nanmin(state_reward), np.nanmax(state_reward), "viridis", "reward"
    else:
        frame = lambda k, m: np.where(np.asarray(m) < 0, np.nan, np.asarray(m, dtype=float))  # noqa: E731
        vmin, vmax, cmap, label = 0, state_reward.shape[1] - 1, "tab20", "env state"
    fig, ax = plt.subplots(figsize=(6, 5))
    pos0 = int(d_seg["seg_pos"].iloc[0])
    im = ax.imshow(frame(int(d_seg["latent"].iloc[0]), env_maps[pos0]),
                   extent=extent, origin="upper", vmin=vmin, vmax=vmax, cmap=cmap)
    trail, = ax.plot([], [], color="w", lw=0.8)
    (pt,) = ax.plot([], [], "o", ms=7, mec="k")
    fig.colorbar(im, ax=ax, label=label)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(labelsize=8)
    lon, lat = d_seg["longitude"].to_numpy(), d_seg["latitude"].to_numpy()

    def update(i):
        row = d_seg.iloc[i]
        k = int(row["latent"])
        im.set_data(frame(k, env_maps[int(row["seg_pos"])]))
        trail.set_data(lon[:i + 1], lat[:i + 1])
        pt.set_data([lon[i]], [lat[i]])
        pt.set_color(cols[k])
        ax.set_title(f"{row['uid']} seg {row['segment']}  {str(row['timestamp'])[:16]}  mode {k}", fontsize=8)
        return im, trail, pt

    ani = animation.FuncAnimation(fig, update, frames=len(d_seg), blit=False)
    ani.save(out_file, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def run_plots(yml_conf):
    yml_conf = dict(yml_conf)
    yml_conf.setdefault("n_hidden", yml_conf.get("n_hidden_init"))
    model = yml_conf.get("swirl_model", "linear")
    K = int(yml_conf["n_hidden"])
    out_dir = yml_conf.get("plot_dir", os.path.join(yml_conf["out_dir"], "plots"))
    os.makedirs(out_dir, exist_ok=True)
    input_keys = as_list(yml_conf.get("input_keys")) if yml_conf.get("plot_inputs", False) else []
    if not yml_conf.get("plot_inputs", False):
        yml_conf["input_keys"] = []
    layer_cols = list((yml_conf.get("static_layers") or {}).keys())
    envs = None
    if yml_conf.get("plot_gifs", False) and yml_conf.get("envs"):
        with open(yml_conf["envs"], "rb") as f:
            envs = pickle.load(f)

    for order in as_list(yml_conf.get("swirl_orders", ["s1", "s2"])):
        tag = f"_{model}_{order}"
        try:
            rm = load_reward_model(yml_conf, model, order)
            lat = load_latents(yml_conf, model, order, tuple(as_list(yml_conf.get("splits", ["train", "test"]))))
        except FileNotFoundError as e:
            print("Skipping", order, ":", e)
            continue
        series = build_track_series(yml_conf, model, order, rewards=rm, latents=lat)
        export_series(series, os.path.join(out_dir, "series" + tag), prefix="")
        plot_reward_heatmaps(rm.state_reward, out_dir, tag)
        np.save(os.path.join(out_dir, f"state_reward{tag}.npy"), rm.state_reward)  # for exploit_explore_init
        if yml_conf.get("plot_trajs", True):
            plot_trajs(series, out_dir, K, yml_conf.get("lon_bounds"), yml_conf.get("lat_bounds"), tag)
        plot_time_series(series, out_dir, DEFAULT_COLUMNS + layer_cols, K, tag)
        plot_density(series, DEFAULT_COLUMNS + layer_cols, K, out_dir, tag)
        if input_keys:
            plot_time_series_input(series, out_dir, input_keys, K, tag)
            plot_density_inputs(series, input_keys, K, out_dir, tag)
        if envs is not None:
            lonb = yml_conf.get("env_lon_bounds", yml_conf.get("lon_bounds"))
            latb = yml_conf.get("env_lat_bounds", yml_conf.get("lat_bounds"))
            n_gif = 0
            for uid, d in series.items():
                if uid not in envs:
                    continue
                for seg, ds in d.groupby("segment"):
                    if n_gif >= int(yml_conf.get("max_gifs", 5)):
                        break
                    maps = envs[uid][int(seg)]
                    if len(maps) <= int(ds["seg_pos"].max()):
                        print(f"WARNING: {uid} seg {seg}: {len(maps)} env maps for the fixes; GIF skipped")
                        continue
                    if n_gif == 0:
                        plot_env_maps(maps[int(ds["seg_pos"].iloc[0])], rm.state_reward, uid, out_dir, lonb, latb, tag)
                    for kind in ("reward", "env"):
                        animate_segment(ds, maps, rm.state_reward,
                                        os.path.join(out_dir, f"anim_{kind}_{uid}_{seg}{tag}.gif"), lonb, latb, K,
                                        kind=kind)
                    n_gif += 1


if __name__ == '__main__':
    from sit_fuse.utils import read_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    run_plots(read_yaml(args.yaml))
