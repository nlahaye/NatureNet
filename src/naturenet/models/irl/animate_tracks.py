"""
Track animations for representative animals (or groups of animals) from any fitted NatureNet behavior model.

Replaces the GIF code of the old plot_swirl.py (plot_env_heat_map_traj_animations / plot_env_map_traj_animations),
which needed grid-index paths, hard-coded file names and dpi 800. This module reads only exported outputs, so the
same code animates single-animal finals, decision-scale variants and multi-agent fits of any model kind:

    # from src/naturenet:
    python -m naturenet.models.irl.animate_tracks -y config/v2/plots/whales_animations.yaml
    python -m naturenet.models.irl.animate_tracks -y <yaml> --only kin_finals --uids 2015_825 --format mp4
    python -m naturenet.models.irl.animate_tracks -y <yaml> --dry-run        # selection table only, no rendering

Panel sources (kind):
  series       a series_<tag>/ dir of <uid>_time_series_pd.csv (plot_hiavi / plot_swirl_new / plot_swirl_nn
               exports; columns timestamp, latitude, longitude, latent [, p_exploit, latent_class]).
  latents      a glob of LL / latent .npz files with sequence ids (seq_key, seq_seg, seq_start [, seq_offset]),
               e.g. the out-of-fold decision-scale test files <run_uid>_SC_<variant>_<K>_<seed>_..._test_KF_*.npz
               (jax_path_vmap; gamma_r -> regime) or K-fold LL files. Positions come from dfs_pkl.
  multiagent   an ma_final_K<K>.npz of run_hiavi_multiagent / run_swirl_multiagent / run_scales_multiagent
               (times, lat, lon, animal, key, original, valid, gamma_<fit> [, p_exploit_<fit>, kin_labels_<fit>]);
               fit: indep | temporal | proximity | ... (one panel per fit).

Backgrounds (background):
  none     tracks only.
  env      the abstract environment (SIT-FUSE cluster) map of the frame's day, from the merged env-map pickle
           env_maps ({uid: [segment][fix] -> (H, W) int map, -1 invalid}, env_data_format output) indexed by the
           fix dates in dfs_pkl. Cells < 0 are transparent.
  reward   state_reward (K, S) .npy of the panel (state_reward_<tag>.npy) mapped onto that env map for the mode
           decoded at the frame (single-animal jobs; multi-agent jobs fall back to env).
Map extent: env_lon_bounds / env_lat_bounds, or grid.lon_bounds / lat_bounds of env_meta_json; the view is zoomed
to the animated tracks (zoom: track, pad_deg) or shows the whole grid (zoom: env).

Selection (select): every panel of a job shows the same animal(s) over the same time window, chosen on the first
(reference) panel, so models / variants / fits are compared on identical data.
  single jobs  only animals present in every panel are candidates (common_animals: True; partial runs), then
               uids: [...] explicit, else strategy coverage (most decoded modes, then most switches, then most
               fixes; default), longest, or by_mode (one animal per mode, the one with the largest share of that
               mode); n animals; min_fixes. Each animal is cut to the window of max_frames consecutive fixes with
               the most distinct modes and switches (window: best | start | all).
  multi jobs   all animals of the reference panel (or uids), on a calendar clock of frame_days; window_days long,
               placed where the most animals are tracked together (or start / end dates); max_animals keeps the
               most-tracked animals of the window; heads older than max_gap_days (3) are hidden.
Mode numbers are model-specific (mode 2 of HIQL is not mode 2 of SWIRL); labels come from the exploit / explore
classification exported with the fit (latent_class, kin_labels_<fit>) or mode_labels.
Writes <out_dir>/<job>/<job>_<uid or group>.gif|mp4, a still of the last frame (_still.png, for figures) and
selection.csv. format: gif (PillowWriter) | mp4 (ffmpeg; falls back to gif when ffmpeg is missing).

Config keys (top level = defaults for every job; each entry of `animations` overrides them):
  root, out_dir, dfs_pkl (or df_dir + df_run_uid), df_run_uid, env_maps, env_meta_json, env_lon_bounds,
  env_lat_bounds, env_lat_ascending (False), n_clusters (colour range), background, env_alpha (0.55), env_max_days (1), zoom, pad_deg (1.0),
  fps (4), dpi (110), format, figsize, legend_loc, mode_labels ({k: name}), select, mode (single | multi), panels.
  Strings may use {root}. Panel keys: name, kind, path, state_reward, fit, regime (True: draw the regime strip for
  latents files with gamma_r), mode_labels.
"""
import argparse
import glob
import json
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

AUG_RE = re.compile(r"__aug\d+")
COLS = ["uid", "timestamp", "latitude", "longitude", "latent"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def read_conf(path):
    try:
        from sit_fuse.utils import read_yaml
        return read_yaml(path)
    except ImportError:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)


def _fmt(v, root):
    if isinstance(v, str) and root is not None:
        return v.replace("{root}", str(root))
    if isinstance(v, dict):
        return {k: _fmt(x, root) for k, x in v.items()}
    if isinstance(v, list):
        return [_fmt(x, root) for x in v]
    return v


def job_confs(conf):
    """[(job dict)] with top-level defaults merged in and {root} expanded."""
    base = {k: v for k, v in conf.items() if k != "animations"}
    jobs = conf.get("animations") or [dict(name="animation")]
    out = []
    for j in jobs:
        c = dict(base)
        sel = dict(base.get("select") or {})
        sel.update(j.get("select") or {})
        c.update(j)
        c["select"] = sel
        out.append(_fmt(c, c.get("root")))
    return out


def _dfs_path(c):
    if c.get("dfs_pkl"):
        return c["dfs_pkl"]
    if c.get("df_dir") and c.get("df_run_uid"):
        return os.path.join(c["df_dir"], c["df_run_uid"] + "_dfs.pkl")
    return None


_DFS = {}


def load_dfs(c):
    p = _dfs_path(c)
    if p is None:
        raise ValueError("dfs_pkl (or df_dir + df_run_uid) is needed for latents panels and env / reward backgrounds")
    if p not in _DFS:
        _DFS[p] = pd.read_pickle(p)
    return _DFS[p]


def _naive(ts):
    t = pd.to_datetime(pd.Series(ts))
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    return t.to_numpy()


def _short(an, df_run_uid=None):
    s = str(an)
    if df_run_uid and s.startswith(str(df_run_uid) + "_"):
        s = s[len(str(df_run_uid)) + 1:]
    return AUG_RE.sub("", s)


# ---------------------------------------------------------------------------
# Panel sources -> tidy frames (uid, timestamp, latitude, longitude, latent [, p_exploit, latent_class, regime])
# ---------------------------------------------------------------------------

def load_series_dir(path):
    files = sorted(glob.glob(os.path.join(path, "*_time_series_pd.csv")))
    if not files:
        raise FileNotFoundError(f"no *_time_series_pd.csv in {path}")
    keep = COLS + ["segment", "seg_pos", "p_exploit", "latent_class", "speed_kmh"]
    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=lambda c: c in keep, dtype={"uid": str})
        d["uid"] = d["uid"].astype(str)
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["timestamp"] = _naive(d["timestamp"])
    return d


def load_latent_files(pattern, c):
    """Out-of-fold (or any) latent paths with sequence ids, positioned with dfs_pkl."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files match {pattern}")
    dfs, run_uid = load_dfs(c), c.get("df_run_uid")
    rows, seen = [], set()
    for f in files:
        z = np.load(f, allow_pickle=True)
        zs = np.asarray(z["learnt_zs"] if "learnt_zs" in z.files else z["jax_path_vmap"])
        if not {"seq_key", "seq_seg", "seq_start"} <= set(z.files):
            raise KeyError(f"{f}: no seq_key / seq_seg / seq_start (rerun with the current runner)")
        off = int(z["seq_offset"]) if "seq_offset" in z.files else 1
        gr = np.asarray(z["gamma_r"]) if "gamma_r" in z.files else None
        gr = gr if gr is not None and gr.ndim == 3 and gr.shape[-1] > 1 else None
        pe = np.asarray(z["p_exploit"], float) if "p_exploit" in z.files else None
        for i in range(len(zs)):
            key = str(z["seq_key"][i])
            if AUG_RE.search(key):
                continue
            dkey = _short(key, run_uid)
            seg, st = int(z["seq_seg"][i]), int(z["seq_start"][i])
            if dkey not in dfs:
                print(f"WARNING: {f}: no movement DataFrame for {dkey}; skipped")
                continue
            df = dfs[dkey][seg]
            ts = _naive(df["timestamp"])
            for t, k in enumerate(zs[i]):
                pos = st + off + t
                if int(k) < 0 or pos >= len(df) or (dkey, seg, pos) in seen:
                    continue
                seen.add((dkey, seg, pos))
                r = dict(uid=dkey, segment=seg, seg_pos=pos, timestamp=ts[pos],
                         latitude=float(df["latitude"].iloc[pos]), longitude=float(df["longitude"].iloc[pos]),
                         latent=int(k))
                if gr is not None:
                    r["regime"] = int(np.argmax(gr[i, t]))
                if pe is not None and pe.ndim == 2:
                    r["p_exploit"] = float(pe[i, t])
                rows.append(r)
    if not rows:
        raise ValueError(f"{pattern}: no decoded steps")
    return pd.DataFrame(rows)


def load_multiagent(path, fit, c):
    z = np.load(path, allow_pickle=True)
    g = f"gamma_{fit}"
    if g not in z.files:
        fits = sorted(k[6:] for k in z.files if k.startswith("gamma_"))
        raise KeyError(f"{path}: no {g} (fits: {fits})")
    gam, valid = np.asarray(z[g]), np.asarray(z["valid"], bool)
    keep = np.asarray(z["original"], bool) if "original" in z.files else np.ones(len(gam), bool)
    pe = np.asarray(z[f"p_exploit_{fit}"], float) if f"p_exploit_{fit}" in z.files else None
    times = pd.Timestamp("1970-01-01") + pd.to_timedelta(np.asarray(z["times"])[valid & keep[:, None]], unit="D")
    an = np.broadcast_to(np.asarray(z["animal"]).astype(str)[:, None], valid.shape)
    m = valid & keep[:, None]
    d = pd.DataFrame(dict(uid=[_short(a, c.get("df_run_uid")) for a in an[m]], timestamp=times.to_numpy(),
                          latitude=np.asarray(z["lat"])[m], longitude=np.asarray(z["lon"])[m],
                          latent=np.argmax(gam, -1)[m]))
    if pe is not None:
        d["p_exploit"] = pe[m]
    labels = None
    if f"kin_labels_{fit}" in z.files:
        labels = {k: str(v) for k, v in enumerate(np.asarray(z[f"kin_labels_{fit}"]))}
    return d.drop_duplicates(["uid", "timestamp"]), labels


def load_panel(p, c):
    kind = p.get("kind", "series")
    labels = None
    if kind == "series":
        d = load_series_dir(p["path"])
    elif kind == "latents":
        d = load_latent_files(p["path"], c)
    elif kind == "multiagent":
        d, labels = load_multiagent(p["path"], p.get("fit", "indep"), c)
    else:
        raise ValueError(f"panel kind must be series | latents | multiagent, got {kind}")
    d = d.dropna(subset=["latitude", "longitude", "latent"]).sort_values(["uid", "timestamp"]).reset_index(drop=True)
    d["latent"] = d["latent"].astype(int)
    if labels is None and "latent_class" in d:
        lc = d.dropna(subset=["latent_class"]).groupby("latent")["latent_class"].agg(lambda s: s.mode().iloc[0])
        labels = {int(k): str(v) for k, v in lc.items() if str(v)}
    labels = dict(labels or {})
    labels.update({int(k): str(v) for k, v in (c.get("mode_labels") or {}).items()})
    labels.update({int(k): str(v) for k, v in (p.get("mode_labels") or {}).items()})
    sr = np.load(p["state_reward"]) if p.get("state_reward") else None
    K = int(p.get("K", max(int(d["latent"].max()) + 1, sr.shape[0] if sr is not None else 0)))
    return dict(name=p.get("name", kind), data=d, labels=labels, state_reward=sr, K=K,
                regime=bool(p.get("regime", True)) and "regime" in d)


# ---------------------------------------------------------------------------
# Environment background
# ---------------------------------------------------------------------------

def env_bounds(c):
    lon_b, lat_b = c.get("env_lon_bounds"), c.get("env_lat_bounds")
    if c.get("env_meta_json") and (lon_b is None or lat_b is None):
        with open(c["env_meta_json"]) as f:
            g = json.load(f)["grid"]
        lon_b = g["lon_bounds"] if lon_b is None else lon_b
        lat_b = g["lat_bounds"] if lat_b is None else lat_b
    if lon_b is None or lat_b is None:
        lon_b, lat_b = c.get("lon_bounds"), c.get("lat_bounds")
    return lon_b, lat_b


class EnvMaps:
    """Day -> abstract env map, from the merged env-map pickle and the fix dates in dfs_pkl."""

    def __init__(self, c):
        with open(c["env_maps"], "rb") as f:
            envs = pickle.load(f)
        dfs = load_dfs(c)
        self.by_day = {}
        for uid, segs in envs.items():
            if uid not in dfs:
                continue
            for seg, maps in enumerate(segs):
                if seg >= len(dfs[uid]):
                    break
                days = pd.DatetimeIndex(_naive(dfs[uid][seg]["timestamp"])).normalize()
                for pos, m in enumerate(maps):
                    if m is not None and pos < len(days) and days[pos] not in self.by_day:
                        self.by_day[days[pos]] = np.asarray(m)
        if not self.by_day:
            raise ValueError(f"{c['env_maps']}: no env maps matched the dfs_pkl keys")
        self.days = np.array(sorted(self.by_day), dtype="datetime64[ns]")
        self.max_days = float(c.get("env_max_days", 1))
        if c.get("env_lat_ascending", False):
            self.by_day = {k: v[::-1] for k, v in self.by_day.items()}
        self.n_clusters = int(c.get("n_clusters") or max(int(np.max(v)) for v in self.by_day.values()) + 1)

    def at(self, ts):
        t = np.datetime64(pd.Timestamp(ts).normalize(), "ns")
        j = np.searchsorted(self.days, t, side="right") - 1
        if j < 0 or (t - self.days[j]) / np.timedelta64(1, "D") > self.max_days:
            return None
        return self.by_day[pd.Timestamp(self.days[j])]


_ENV = {}


def env_maps(c):
    key = (c["env_maps"], _dfs_path(c), bool(c.get("env_lat_ascending", False)), float(c.get("env_max_days", 1)))
    if key not in _ENV:
        _ENV[key] = EnvMaps(c)
    return _ENV[key]


def categorical_cmap(n, seed=0):
    rng = np.random.default_rng(seed)
    hsv = np.c_[rng.permutation(np.linspace(0, 1, n, endpoint=False)), 0.35 + 0.4 * rng.random(n),
                0.65 + 0.3 * rng.random(n)]
    return ListedColormap(matplotlib.colors.hsv_to_rgb(hsv))


def env_image(m):
    return None if m is None else np.where(np.asarray(m) < 0, np.nan, np.asarray(m, float))


def reward_image(sr_k, m):
    if m is None:
        return None
    m = np.asarray(m)
    out = np.full(m.shape, np.nan)
    ok = (m >= 0) & (m < len(sr_k))
    out[ok] = np.asarray(sr_k)[m[ok].astype(int)]
    return out


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _switches(z):
    z = np.asarray(z)
    return int((z[1:] != z[:-1]).sum()) if len(z) > 1 else 0


def best_window(z, n):
    """Start index of the n-step window with the most distinct modes, then the most switches."""
    if len(z) <= n:
        return 0
    z = np.asarray(z)
    sw = np.r_[0, np.cumsum(z[1:] != z[:-1])]
    best, arg = None, 0
    for s in range(0, len(z) - n + 1, max(1, n // 20)):
        sc = (len(np.unique(z[s:s + n])), sw[s + n - 1] - sw[s])
        if best is None or sc > best:
            best, arg = sc, s
    return arg


def select_single(ref, sel):
    d = ref["data"]
    min_fixes = int(sel.get("min_fixes", 30))
    st = []
    for uid, g in d.groupby("uid"):
        z = g["latent"].to_numpy()
        st.append(dict(uid=uid, n_fixes=len(z), n_modes=len(np.unique(z)), n_switches=_switches(z),
                       days=(g["timestamp"].max() - g["timestamp"].min()) / pd.Timedelta(days=1),
                       **{f"share_{k}": float(np.mean(z == k)) for k in range(ref["K"])}))
    st = pd.DataFrame(st)
    if sel.get("uids"):
        want = [str(u) for u in sel["uids"]]
        miss = sorted(set(want) - set(st["uid"]))
        if miss:
            print("WARNING: not in the reference panel:", miss)
        chosen = st[st["uid"].isin(want)].assign(reason="requested")
    else:
        n = int(sel.get("n", 3))
        pool = st[st["n_fixes"] >= min_fixes]
        pool = pool if len(pool) else st
        strat = sel.get("strategy", "coverage")
        if strat == "longest":
            chosen = pool.sort_values("n_fixes", ascending=False).head(n).assign(reason="longest")
        elif strat == "by_mode":
            picks, used = [], set()
            for k in range(ref["K"]):
                cand = pool[~pool["uid"].isin(used)].sort_values(f"share_{k}", ascending=False)
                if len(cand) and cand[f"share_{k}"].iloc[0] > 0:
                    used.add(cand["uid"].iloc[0])
                    picks.append(cand.head(1).assign(reason=f"mode {k}"))
            chosen = pd.concat(picks) if picks else pool.head(0)
        elif strat == "coverage":
            chosen = pool.sort_values(["n_modes", "n_switches", "n_fixes"], ascending=False).head(n)
            chosen = chosen.assign(reason="coverage")
        else:
            raise ValueError("select.strategy must be coverage | longest | by_mode")
    max_frames, win = int(sel.get("max_frames", 240)), sel.get("window", "best")
    out = []
    for _, r in chosen.iterrows():
        g = d[d["uid"] == r["uid"]].sort_values("timestamp")
        n = len(g) if win == "all" else min(max_frames, len(g))
        s = best_window(g["latent"].to_numpy(), n) if win == "best" else 0
        out.append(dict(r, start=g["timestamp"].iloc[s], end=g["timestamp"].iloc[s + n - 1], frames=n,
                        label=str(r["uid"])))
    return pd.DataFrame(out)


def select_multi(ref, sel):
    d = ref["data"]
    if sel.get("uids"):
        d = d[d["uid"].isin([str(u) for u in sel["uids"]])]
    day = pd.DatetimeIndex(d["timestamp"]).normalize()
    if sel.get("start"):
        start = pd.Timestamp(sel["start"])
        end = pd.Timestamp(sel["end"]) if sel.get("end") else start + pd.Timedelta(days=float(sel.get("window_days", 60)))
    else:
        wd = int(sel.get("window_days", 60))
        cnt = pd.Series(1, index=pd.MultiIndex.from_arrays([day, d["uid"]])).groupby(level=[0, 1]).size()
        per_day = cnt.groupby(level=0).size().asfreq("D", fill_value=0)
        roll = per_day.rolling(wd, min_periods=1).sum()
        end = roll.idxmax() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=wd)
    w = d[(d["timestamp"] >= start) & (d["timestamp"] < end)]
    uids = sorted(w["uid"].unique())
    if sel.get("max_animals") and len(uids) > int(sel["max_animals"]):     # the most-tracked animals in the window
        uids = sorted(w.groupby("uid").size().sort_values(ascending=False).index[:int(sel["max_animals"])])
        w = w[w["uid"].isin(uids)]
    fd = float(sel.get("frame_days", 1.0))
    n = int(np.ceil((end - start) / pd.Timedelta(days=fd)))
    label = sel.get("label") or f"{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}"
    return pd.DataFrame([dict(uid=",".join(uids), n_animals=len(uids), n_fixes=len(w), start=start, end=end,
                              frames=n, label=label, reason="most concurrent animals" if not sel.get("start")
                              else "requested")])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def mode_colors(K):
    cmap = plt.get_cmap("tab10" if K <= 10 else "tab20")
    return np.array([cmap(i % cmap.N) for i in range(max(K, 1))])


def _legend_text(k, labels):
    return f"mode {k}" + (f" ({labels[k]})" if labels.get(k) else "")


def _extent(c, tracks, envm):
    lon_b, lat_b = env_bounds(c)
    env_ext = None if lon_b is None else [min(lon_b), max(lon_b), min(lat_b), max(lat_b)]
    if c.get("zoom", "track") == "env" and env_ext is not None:
        return env_ext, env_ext
    lon = np.concatenate([t["longitude"].to_numpy() for t in tracks])
    lat = np.concatenate([t["latitude"].to_numpy() for t in tracks])
    pad = float(c.get("pad_deg", 1.0))
    v = [np.nanmin(lon) - pad, np.nanmax(lon) + pad, np.nanmin(lat) - pad, np.nanmax(lat) + pad]
    if env_ext is not None:
        v = [max(v[0], env_ext[0]), min(v[1], env_ext[1]), max(v[2], env_ext[2]), min(v[3], env_ext[3])]
    return v, env_ext


def render(c, panels, row, multi, out_file):
    """One animation: one map + one timeline per panel, frames on the reference panel's clock."""
    start, end = pd.Timestamp(row["start"]), pd.Timestamp(row["end"])
    uids = str(row["uid"]).split(",") if multi else [str(row["uid"])]
    sub = []
    for p in panels:
        d = p["data"]
        sub.append(d[d["uid"].isin(uids) & (d["timestamp"] >= start) & (d["timestamp"] <= end)].copy())
    if not len(sub[0]):
        print("WARNING: no fixes for", row["label"])
        return None
    if multi:
        fd = pd.Timedelta(days=float(c["select"].get("frame_days", 1.0)))
        clock = pd.date_range(start + fd, end, freq=fd).to_numpy()
    else:
        clock = sub[0]["timestamp"].to_numpy()
    bg = c.get("background", "none")
    envm = env_maps(c) if bg in ("env", "reward") and c.get("env_maps") else None
    if bg != "none" and envm is None:
        print("WARNING: background", bg, "needs env_maps + dfs_pkl; drawing tracks only")
        bg = "none"
    if multi and bg == "reward":
        bg = "env"
    view, env_ext = _extent(c, [s for s in sub if len(s)], envm)
    if envm is not None and env_ext is None:
        print("WARNING: env background needs env_lon_bounds / env_lat_bounds or env_meta_json; drawing tracks only")
        envm, bg = None, "none"
    nP = len(panels)
    fw, fh = c.get("figsize") or (4.6 * nP + 0.8, 5.6 if not multi else 6.2)
    fig = plt.figure(figsize=(fw, fh))
    gs = fig.add_gridspec(2, nP, height_ratios=[4, 1.3 if not multi else 1.8], hspace=0.32, wspace=0.38)
    arts = []
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]
    vlim = None
    if bg == "reward":
        allr = np.concatenate([p["state_reward"].ravel() for p in panels if p["state_reward"] is not None])
        vlim = np.nanpercentile(allr, [2, 98]) if len(allr) else None
    for j, (p, s) in enumerate(zip(panels, sub)):
        ax = fig.add_subplot(gs[0, j])
        axt = fig.add_subplot(gs[1, j])
        cols = mode_colors(p["K"])
        im = None
        if envm is not None:
            m0 = envm.at(clock[0])
            if bg == "env":
                im = ax.imshow(env_image(m0) if m0 is not None else np.full((2, 2), np.nan), extent=env_ext,
                               origin="upper", cmap=categorical_cmap(envm.n_clusters), vmin=0,
                               vmax=envm.n_clusters - 1, alpha=float(c.get("env_alpha", 0.55)),
                               interpolation="nearest", zorder=0)
            elif p["state_reward"] is not None:
                k0 = int(s["latent"].iloc[0]) if len(s) else 0
                im = ax.imshow(reward_image(p["state_reward"][k0], m0) if m0 is not None else np.full((2, 2), np.nan),
                               extent=env_ext, origin="upper", cmap="viridis",
                               vmin=None if vlim is None else vlim[0], vmax=None if vlim is None else vlim[1],
                               interpolation="nearest", zorder=0)
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="state reward (decoded mode)")
        ax.set_xlim(view[0], view[1])
        ax.set_ylim(view[2], view[3])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("longitude", fontsize=8)
        if j == 0:
            ax.set_ylabel("latitude", fontsize=8)
        tr = []
        for a_i, u in enumerate(uids):
            g = s[s["uid"] == u]
            line, = ax.plot([], [], color="0.25", lw=0.7, alpha=0.8, zorder=2)
            dots = ax.scatter([], [], s=9, zorder=3, linewidths=0)
            head = ax.scatter([], [], s=70, zorder=4, edgecolors="k", linewidths=0.8,
                              marker=marker_cycle[a_i % len(marker_cycle)] if multi else "o")
            tr.append(dict(t=g["timestamp"].to_numpy(), x=g["longitude"].to_numpy(), y=g["latitude"].to_numpy(),
                           z=g["latent"].to_numpy(), line=line, dots=dots, head=head, uid=u))
        handles = [plt.Line2D([], [], marker="o", ls="", color=cols[k], label=_legend_text(k, p["labels"]))
                   for k in range(p["K"])]
        ax.legend(handles=handles, fontsize=6.5, loc=c.get("legend_loc", "lower left"), framealpha=0.85).set_zorder(10)
        # timeline
        if multi:
            for a_i, t in enumerate(tr):
                axt.scatter(t["t"], np.full(len(t["t"]), a_i), c=cols[t["z"]], s=6, marker="s", linewidths=0)
            axt.set_yticks(range(len(tr)))
            axt.set_yticklabels([t["uid"] for t in tr], fontsize=5.5 if len(tr) > 12 else 7)
            axt.set_ylim(-0.7, len(tr) - 0.3)
        else:
            t0 = tr[0]
            axt.scatter(t0["t"], np.zeros(len(t0["t"])), c=cols[t0["z"]], s=14, marker="|", linewidths=3)
            axt.set_yticks([0])
            axt.set_yticklabels(["mode"], fontsize=7)
            nrow = 1
            if p["regime"] and "regime" in s:
                axt.scatter(t0["t"], np.full(len(t0["t"]), 1), c=plt.get_cmap("Greys")(0.35 + 0.5 * s["regime"].to_numpy()),
                            s=14, marker="|", linewidths=3)
                axt.set_yticks([0, 1])
                axt.set_yticklabels(["mode", "regime"], fontsize=7)
                nrow = 2
            axt.set_ylim(-0.6, nrow - 0.4)
            if "p_exploit" in s and np.isfinite(s["p_exploit"]).any():
                ax2 = axt.twinx()
                ax2.plot(t0["t"], s["p_exploit"].to_numpy(), color="k", lw=0.8)
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_ylabel("P(exploit)", fontsize=7)
                ax2.tick_params(labelsize=6)
        axt.set_xlim(start, end)
        axt.tick_params(axis="x", labelsize=6, rotation=25)
        cur = axt.axvline(clock[0], color="k", lw=1.0)
        title = ax.set_title(p["name"], fontsize=9)
        arts.append(dict(ax=ax, im=im, tr=tr, cur=cur, title=title, p=p, cols=cols))
    sup = fig.suptitle("", fontsize=10)
    who = row["label"] if not multi else f"{len(uids)} animals"

    def update(i):
        now = clock[i]
        m = envm.at(now) if envm is not None else None
        out = []
        for a in arts:
            last_k = None
            for t in a["tr"]:
                n = int(np.searchsorted(t["t"], now, side="right"))
                xy = np.c_[t["x"][:n], t["y"][:n]]
                t["line"].set_data(t["x"][:n], t["y"][:n])
                t["dots"].set_offsets(xy if n else np.empty((0, 2)))
                if n:
                    t["dots"].set_facecolors(a["cols"][t["z"][:n]])
                    stale = multi and (now - t["t"][n - 1]) / np.timedelta64(1, "D") > float(
                        c["select"].get("max_gap_days", 3))
                    t["head"].set_offsets(np.empty((0, 2)) if stale else xy[-1:])
                    t["head"].set_facecolors(a["cols"][t["z"][n - 1:n]])
                    last_k = int(t["z"][n - 1])
                else:
                    t["head"].set_offsets(np.empty((0, 2)))
                out += [t["line"], t["dots"], t["head"]]
            if a["im"] is not None and m is not None:
                if bg == "env":
                    a["im"].set_data(env_image(m))
                elif last_k is not None:
                    a["im"].set_data(reward_image(a["p"]["state_reward"][last_k], m))
            if not multi and last_k is not None:
                a["title"].set_text(f"{a['p']['name']}: {_legend_text(last_k, a['p']['labels'])}")
            a["cur"].set_xdata([now, now])
        sup.set_text(f"{who}   {pd.Timestamp(now):%Y-%m-%d %H:%M}")
        return out

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fmt = c.get("format", "gif")
    if fmt == "mp4" and not animation.writers.is_available("ffmpeg"):
        print("WARNING: ffmpeg not available; writing a GIF")
        fmt = "gif"
    out_file = os.path.splitext(out_file)[0] + "." + fmt
    fps, dpi = int(c.get("fps", 4)), int(c.get("dpi", 110))
    writer = animation.PillowWriter(fps=fps) if fmt == "gif" else animation.FFMpegWriter(fps=fps, bitrate=2400)
    ani = animation.FuncAnimation(fig, update, frames=len(clock), blit=False)
    ani.save(out_file, writer=writer, dpi=dpi)
    update(len(clock) - 1)
    fig.savefig(os.path.splitext(out_file)[0] + "_still.png", dpi=max(dpi, 150), bbox_inches="tight")
    plt.close(fig)
    return out_file


def run_job(c, uids=None, dry_run=False):
    name = c.get("name", "animation")
    multi = c.get("mode", "single") == "multi"
    sel = dict(c["select"])
    if uids:
        sel["uids"] = uids
    c["select"] = sel
    panels = []
    for p in c.get("panels") or []:
        try:
            panels.append(load_panel(p, c))
        except (FileNotFoundError, KeyError) as e:
            print(f"[{name}] panel {p.get('name')}: {e}; skipped")
    if not panels:
        print(f"[{name}] no panels available; skipped")
        return []
    ref = panels[0]
    if not multi and len(panels) > 1 and sel.get("common_animals", True):
        # partial runs (e.g. decision-scale folds still running): choose among animals every panel has
        common = set(ref["data"]["uid"])
        for p in panels[1:]:
            common &= set(p["data"]["uid"])
        if common and len(common) < ref["data"]["uid"].nunique():
            print(f"[{name}] {len(common)} animals in every panel (reference has {ref['data']['uid'].nunique()})")
            ref = dict(ref, data=ref["data"][ref["data"]["uid"].isin(common)])
        elif not common:
            print(f"[{name}] WARNING: no animal is in every panel; selecting on the reference panel")
    table = select_multi(ref, sel) if multi else select_single(ref, sel)
    od = os.path.join(c["out_dir"], name)
    os.makedirs(od, exist_ok=True)
    table.assign(job=name, reference=panels[0]["name"]).to_csv(os.path.join(od, "selection.csv"), index=False)
    print(f"[{name}] panels: {[p['name'] for p in panels]}")
    show = table.drop(columns=[x for x in table.columns if x.startswith("share_")])
    if multi:
        show = show.assign(uid=show["uid"].str.slice(0, 60) + "...")
    print(show.to_string(index=False))
    if dry_run:
        return []
    files = []
    for _, r in table.iterrows():
        f = render(c, panels, r, multi, os.path.join(od, f"{name}_{r['label']}"))
        if f:
            print("wrote", f)
            files.append(f)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="job names to run (default: all)")
    ap.add_argument("--uids", nargs="*", default=None, help="animals to animate (overrides select)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--strategy", choices=["coverage", "longest", "by_mode"], default=None)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--format", choices=["gif", "mp4"], default=None)
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--background", choices=["none", "env", "reward"], default=None)
    ap.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    a = ap.parse_args(argv)
    files = []
    for c in job_confs(read_conf(a.yaml)):
        if a.only and c.get("name") not in a.only:
            continue
        for k in ("format", "dpi", "background"):
            if getattr(a, k) is not None:
                c[k] = getattr(a, k)
        for k in ("n", "strategy", "max_frames"):
            if getattr(a, k) is not None:
                c["select"] = dict(c["select"], **{k: getattr(a, k)})
        files += run_job(c, a.uids, a.dry_run)
    return files


if __name__ == "__main__":
    main()
