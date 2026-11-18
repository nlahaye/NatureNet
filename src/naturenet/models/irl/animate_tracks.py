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
  windows      {uid: [start, end]} (or select.start / select.end): explicit calendar windows instead of window.
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
  fps (4), dpi (110), format, figsize, legend_loc, mode_labels ({k: name}), select, mode (single | multi | season), panels.
  Strings may use {root}. Panel keys: name, kind, path, state_reward, fit, regime (True: draw the regime strip for
  latents files with gamma_r), mode_labels, labels_csv, p_exploit.

Figure-oriented options (all optional; defaults reproduce the earlier behaviour):
  jump_filter   {max_speed_kmh: 20, max_step_km: null}: drop plotted fixes whose speed (or step) from the last kept
                fix exceeds the limit (location-error spikes; display only, the fits are unchanged).
  break_gap_h   do not draw the track line across gaps longer than this many hours (e.g. 48).
  colour_by     mode (default; tab10 by mode number) | class (exploit = blues, explore = reds, ambiguous = greys,
                one shade per mode), so the same colour means the same behaviour in every panel and model.
  labels_csv    a per-mode class table (exploit_explore_init output: state + class, or kinematic: state + label,
                or decision-context mode_alignment: mode + cls, rows of labels_model); overrides latent_class.
                Panel key (job level only for single-panel jobs).
  p_exploit     auto (default: hidden when constant, e.g. degenerate labels) | on | off.
  align_modes   True: relabel each panel's modes to the reference panel's by Hungarian matching on shared steps
                (same animals and times; e.g. decision-scale variants whose mode numbers are arbitrary).
  stills        frames to also save as PNG: floats in [0, 1] (fraction of the clock), ints (frame index; -1 last),
                or timestamps ("2023-04-01" -> nearest frame; skipped outside the window). Written as <file>_still_<frame>.png at still_dpi (200);
                still_panels: True also writes one crop per panel (<file>_still_<frame>_p<j>.png).
  animate       False: write only the stills (fast re-render for figures).
  mode_names    panel (or job) {k: name}: a short behaviour name appended to the legend / heatmap row, e.g.
                {3: transit}, so modes of one class stay identifiable under colour_by: class.
  reward_scale  shared (default: one colour range over all panels) | panel (own 2-98 percentile range and
                colour bar per panel; use when panels come from models with different reward scales, e.g. SWIRL
                S-2 vs HIQL) | mode (range of the decoded mode's map, updated each frame: shows spatial contrast
                within a mode even when modes differ mainly by an offset; values are not comparable across frames). reward_label: colour-bar label (panel or job key).
  clip_to_env   True (default): the track view is clipped to the env grid bounds. False: pad beyond the grid so
                animals at the grid edge are not cut off (the background just ends).
Season jobs (mode: season): all fixes of the selected animals (select.uids, default all) are pooled by calendar
  month (season.bins: month | [[12, 1, 2], [3, 4, 5], ...]; season.bin_names), one frame per bin, coloured by mode or
  class over the faint full track set, with mode shares per bin below. Also writes <job>_<panel>_grid.png (one small
  map per bin plus a bin x mode share heatmap) and <job>_<panel>_shares.csv; stills may name bins ("Apr").
  Background: none (pooled days have no single env map). Every panel of a season job is rendered separately.
  season.min_animals (3): bins with fewer animals are hatched in the bar / heatmap and flagged in the csv.
  season.class_rows (True with labels): adds 'all <class>' rows (summed shares per class) to the heatmap and
  share_<class> columns to the csv. Also writes <job>_<panel>_heatmap.png (heatmap only, for composite figures).
  Grid / heatmap labels show n animals and n fixes per bin.
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


def _haversine_km(lat1, lon1, lat2, lon2):
    r = np.radians
    a = np.sin(r(lat2 - lat1) / 2) ** 2 + np.cos(r(lat1)) * np.cos(r(lat2)) * np.sin(r(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def jump_filter(d, jf):
    """Drop fixes whose speed / step from the last kept fix of the same animal exceeds the limits (display only)."""
    if not jf:
        return d
    vmax = jf.get("max_speed_kmh")
    smax = jf.get("max_step_km")
    if vmax is None and smax is None:
        return d
    keep = np.ones(len(d), bool)
    for _, idx in d.groupby("uid").indices.items():
        idx = np.sort(idx)
        lat, lon = d["latitude"].to_numpy()[idx], d["longitude"].to_numpy()[idx]
        t = d["timestamp"].to_numpy()[idx]
        last = 0
        for i in range(1, len(idx)):
            km = _haversine_km(lat[last], lon[last], lat[i], lon[i])
            dt = (t[i] - t[last]) / np.timedelta64(1, "h")
            bad = smax is not None and km > float(smax)
            if vmax is not None and not bad:
                bad = (km / dt > float(vmax)) if dt > 0 else km > 0.5
            if bad:
                keep[idx[i]] = False
            else:
                last = i
    n = int((~keep).sum())
    if n:
        print(f"jump_filter: dropped {n} of {len(d)} plotted fixes ({100.0 * n / len(d):.1f}%)")
    return d[keep].reset_index(drop=True)


def read_labels_csv(path, model=None):
    """{mode: class} from an exploit_explore_init / kinematic-label / decision-context mode_alignment table
    (mode_alignment: rows of `model`, default the first model in the file)."""
    t = pd.read_csv(path)
    kcol = next((k for k in ("state", "mode", "latent") if k in t), None)
    ccol = next((k for k in ("class", "label", "cls", "latent_class") if k in t), None)
    if kcol is None or ccol is None:
        raise KeyError(f"{path}: needs a state / mode column and a class / label / cls column")
    if "model" in t and t["model"].nunique() > 1:
        t = t[t["model"] == (model if model is not None else t["model"].iloc[0])]
    return {int(k): str(v) for k, v in zip(t[kcol], t[ccol]) if pd.notna(v)}


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
    d = jump_filter(d, c.get("jump_filter"))
    if labels is None and "latent_class" in d:
        lc = d.dropna(subset=["latent_class"]).groupby("latent")["latent_class"].agg(lambda s: s.mode().iloc[0])
        labels = {int(k): str(v) for k, v in lc.items() if str(v)}
    labels = dict(labels or {})
    overridden = False
    lcsv = p.get("labels_csv", c.get("labels_csv") if len(c.get("panels") or []) <= 1 else None)
    if lcsv:
        if os.path.exists(lcsv):
            labels.update(read_labels_csv(lcsv, p.get("labels_model")))
            overridden = True
        else:
            print(f"WARNING: labels_csv {lcsv} not found; keeping exported labels")
    labels.update({int(k): str(v) for k, v in (c.get("mode_labels") or {}).items()})
    labels.update({int(k): str(v) for k, v in (p.get("mode_labels") or {}).items()})
    overridden = overridden or bool(p.get("mode_labels") or c.get("mode_labels"))
    pe_mode = str(p.get("p_exploit", c.get("p_exploit", "auto"))).lower()
    if "p_exploit" in d:
        v = pd.to_numeric(d["p_exploit"], errors="coerce").to_numpy(float)
        constant = not np.isfinite(v).any() or np.nanmax(v) - np.nanmin(v) < 1e-6
        if pe_mode in ("off", "false", "0") or (pe_mode == "auto" and (constant or overridden)):
            # P(exploit) is computed from the exported labels; hide it when those are degenerate or replaced
            d = d.drop(columns=["p_exploit"])
    sr = None
    if p.get("state_reward"):
        if os.path.exists(p["state_reward"]):
            sr = np.load(p["state_reward"])
        elif c.get("background", "none") == "reward":
            raise FileNotFoundError(p["state_reward"])
        else:
            print(f"NOTE: {p['state_reward']} not found (only needed for background: reward)")
    K = int(p.get("K", max(int(d["latent"].max()) + 1, sr.shape[0] if sr is not None else 0)))
    names = {int(k): str(v) for k, v in (c.get("mode_names") or {}).items()} if len(c.get("panels") or []) <= 1 else {}
    names.update({int(k): str(v) for k, v in (p.get("mode_names") or {}).items()})
    return dict(name=p.get("name", kind), data=d, labels=labels, state_reward=sr, K=K, names=names,
                reward_label=p.get("reward_label", c.get("reward_label")),
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
    wins = {str(k): v for k, v in (sel.get("windows") or {}).items()}
    for _, r in chosen.iterrows():
        g = d[d["uid"] == r["uid"]].sort_values("timestamp")
        w = wins.get(str(r["uid"]), [sel.get("start"), sel.get("end")])
        if w[0] is not None or w[1] is not None:  # explicit calendar window (e.g. to match a published figure)
            lo = pd.Timestamp(w[0]) if w[0] is not None else g["timestamp"].iloc[0]
            hi = pd.Timestamp(w[1]) if w[1] is not None else g["timestamp"].iloc[-1]
            g = g[(g["timestamp"] >= lo) & (g["timestamp"] <= hi)]
            if not len(g):
                print(f"WARNING: {r['uid']}: no fixes in window {w}")
                continue
            if len(g) > max_frames:
                print(f"NOTE: {r['uid']}: window has {len(g)} fixes (> max_frames {max_frames}); all are animated")
            out.append(dict(r, start=g["timestamp"].iloc[0], end=g["timestamp"].iloc[-1], frames=len(g),
                            label=str(r["uid"])))
            continue
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

CLASS_CMAPS = {"exploit": "Blues", "explore": "Reds", "ambiguous": "Greys"}


def mode_colors(K, labels=None, colour_by="mode"):
    """(K, 4) RGBA per mode: tab10 by mode number, or by class (one shade per mode within its class)."""
    K = max(int(K), 1)
    if colour_by != "class":
        cmap = plt.get_cmap("tab10" if K <= 10 else "tab20")
        return np.array([cmap(i % cmap.N) for i in range(K)])
    labels = labels or {}
    cls = [str(labels.get(k, "ambiguous")).lower() for k in range(K)]
    cls = [x if x in CLASS_CMAPS else "ambiguous" for x in cls]
    out = np.zeros((K, 4))
    for name, cm in CLASS_CMAPS.items():
        ks = [k for k in range(K) if cls[k] == name]
        shades = np.linspace(0.85, 0.5, len(ks)) if len(ks) > 1 else [0.7]
        for k, v in zip(ks, shades):
            out[k] = plt.get_cmap(cm)(v)
    return out


def align_modes(ref, p):
    """Relabel panel p's modes to the reference panel's by maximum co-occurrence on shared (uid, timestamp)."""
    from scipy.optimize import linear_sum_assignment
    a = ref["data"][["uid", "timestamp", "latent"]]
    b = p["data"][["uid", "timestamp", "latent"]]
    m = a.merge(b, on=["uid", "timestamp"], suffixes=("_r", "_p"))
    if not len(m):
        print(f"align_modes: {p['name']} shares no steps with {ref['name']}; not aligned")
        return p
    Kr, Kp = ref["K"], p["K"]
    C = np.zeros((Kp, max(Kr, Kp)))
    np.add.at(C, (m["latent_p"].to_numpy(int), m["latent_r"].to_numpy(int)), 1)
    rows, cols = linear_sum_assignment(-C)
    perm = dict(zip(rows.tolist(), cols.tolist()))
    d = p["data"].copy()
    d["latent"] = d["latent"].map(lambda k: perm.get(int(k), int(k))).astype(int)
    labels = {perm.get(int(k), int(k)): v for k, v in p["labels"].items()}
    names = {perm.get(int(k), int(k)): v for k, v in (p.get("names") or {}).items()}
    agree = float(np.trace(C[rows][:, cols]) / C.sum()) if C.sum() else np.nan
    print(f"align_modes: {p['name']} -> {ref['name']} map {perm} (matched agreement {agree:.2f})")
    K_new = max(Kp, max(perm.values()) + 1)
    sr = p["state_reward"]
    if sr is not None:
        sr_new = np.full((K_new,) + sr.shape[1:], np.nan)
        for k_old, k_new in perm.items():
            if k_old < len(sr):
                sr_new[k_new] = sr[k_old]
        sr = sr_new
    return dict(p, data=d, labels=labels, names=names, state_reward=sr, K=K_new)


def _legend_text(k, labels, names=None):
    bits = [str(x) for x in ((labels or {}).get(k), (names or {}).get(k)) if x]
    return f"mode {k}" + (f" ({', '.join(bits)})" if bits else "")


def _extent(c, tracks, envm):
    lon_b, lat_b = env_bounds(c)
    env_ext = None if lon_b is None else [min(lon_b), max(lon_b), min(lat_b), max(lat_b)]
    if c.get("zoom", "track") == "env" and env_ext is not None:
        return env_ext, env_ext
    lon = np.concatenate([t["longitude"].to_numpy() for t in tracks])
    lat = np.concatenate([t["latitude"].to_numpy() for t in tracks])
    pad = float(c.get("pad_deg", 1.0))
    v = [np.nanmin(lon) - pad, np.nanmax(lon) + pad, np.nanmin(lat) - pad, np.nanmax(lat) + pad]
    if env_ext is not None and c.get("clip_to_env", True):
        v = [max(v[0], env_ext[0]), min(v[1], env_ext[1]), max(v[2], env_ext[2]), min(v[3], env_ext[3])]
    return v, env_ext


def _line_with_breaks(t, x, y, gap_h):
    """Line coordinates with NaN breaks across gaps > gap_h hours; li[i] = line index of fix i."""
    n = len(t)
    if not gap_h or n < 2:
        return x.astype(float), y.astype(float), np.arange(n)
    gap = np.r_[False, np.diff(t) / np.timedelta64(1, "h") > float(gap_h)]
    li = np.arange(n) + np.cumsum(gap)
    xl, yl = np.full(n + int(gap.sum()), np.nan), np.full(n + int(gap.sum()), np.nan)
    xl[li], yl[li] = x, y
    return xl, yl, li


def _still_index(spec, clock):
    """Frame index of a still spec: float fraction, int index (negative from the end) or timestamp string."""
    n = len(clock)
    if isinstance(spec, bool):
        raise ValueError(f"bad still spec {spec}")
    if isinstance(spec, int):
        return int(np.clip(spec if spec >= 0 else n + spec, 0, n - 1))
    if isinstance(spec, float):
        return int(np.clip(round(spec * (n - 1)), 0, n - 1))
    ts = np.datetime64(pd.Timestamp(str(spec)), "ns")
    c64 = clock.astype("datetime64[ns]")
    if ts < c64.min() - np.timedelta64(1, "D") or ts > c64.max() + np.timedelta64(1, "D"):
        return None  # date outside this animation's window (stills lists may mix animals)
    return int(np.clip(np.argmin(np.abs(clock.astype("datetime64[ns]") - ts)), 0, n - 1))


def _save_stills(c, fig, arts, update, clock, out_file):
    specs = c.get("stills") or []
    if not isinstance(specs, (list, tuple)):
        specs = [specs]
    dpi = int(c.get("still_dpi", 200))
    base = os.path.splitext(out_file)[0]
    written = []
    for sp in specs:
        try:
            i = _still_index(sp, clock)
        except (ValueError, TypeError) as e:
            print("WARNING: still", sp, e)
            continue
        if i is None:
            continue
        update(i)
        f = f"{base}_still_{i:04d}.png"
        fig.savefig(f, dpi=dpi, bbox_inches="tight")
        written.append(f)
        if c.get("still_panels"):
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            for j, a in enumerate(arts):
                axs = [a["ax"], a["axt"]] + list(a.get("extra", []))
                bb = matplotlib.transforms.Bbox.union([x.get_tightbbox(r) for x in axs])
                bb = bb.transformed(fig.dpi_scale_trans.inverted()).padded(0.05)
                fig.savefig(f"{base}_still_{i:04d}_p{j}.png", dpi=dpi, bbox_inches=bb)
    if written:
        print("stills:", ", ".join(os.path.basename(w) for w in written))
    return written


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
    rscale = str(c.get("reward_scale", "shared")).lower()
    per_panel = rscale in ("panel", "mode")

    def _mode_lim(sr_k):
        v = np.asarray(sr_k, float)
        v = v[np.isfinite(v)]
        if not len(v):
            return None
        lo, hi = np.nanpercentile(v, [2, 98])
        return (lo, hi) if hi > lo else (lo - 1e-6, lo + 1e-6)
    if bg == "reward" and not per_panel:
        allr = np.concatenate([p["state_reward"].ravel() for p in panels if p["state_reward"] is not None])
        vlim = np.nanpercentile(allr, [2, 98]) if len(allr) else None
    for j, (p, s) in enumerate(zip(panels, sub)):
        ax = fig.add_subplot(gs[0, j])
        axt = fig.add_subplot(gs[1, j])
        cols = mode_colors(p["K"], p["labels"], c.get("colour_by", "mode"))
        im, cb, extra_axes = None, None, []
        if envm is not None:
            m0 = envm.at(clock[0])
            if bg == "env":
                im = ax.imshow(env_image(m0) if m0 is not None else np.full((2, 2), np.nan), extent=env_ext,
                               origin="upper", cmap=categorical_cmap(envm.n_clusters), vmin=0,
                               vmax=envm.n_clusters - 1, alpha=float(c.get("env_alpha", 0.55)),
                               interpolation="nearest", zorder=0)
            elif p["state_reward"] is not None:
                k0 = int(s["latent"].iloc[0]) if len(s) else 0
                if per_panel:
                    v = p["state_reward"][np.isfinite(p["state_reward"])]
                    vlim = np.nanpercentile(v, [2, 98]) if len(v) else None
                    if vlim is not None and vlim[1] <= vlim[0]:
                        vlim = [vlim[0] - 1e-6, vlim[0] + 1e-6]
                    if rscale == "mode":
                        vlim = _mode_lim(p["state_reward"][k0])
                    print(f"  {p['name']}: reward colour range {vlim}" + (" (first mode; re-scaled per mode)"
                                                                          if rscale == "mode" else ""))
                im = ax.imshow(reward_image(p["state_reward"][k0], m0) if m0 is not None else np.full((2, 2), np.nan),
                               extent=env_ext, origin="upper", cmap="viridis",
                               vmin=None if vlim is None else vlim[0], vmax=None if vlim is None else vlim[1],
                               interpolation="nearest", zorder=0)
                cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=p.get("reward_label") or "state reward (decoded mode)")
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
            tt, xx, yy = g["timestamp"].to_numpy(), g["longitude"].to_numpy(), g["latitude"].to_numpy()
            xl, yl, li = _line_with_breaks(tt, xx, yy, c.get("break_gap_h"))
            tr.append(dict(t=tt, x=xx, y=yy, xl=xl, yl=yl, li=li,
                           z=g["latent"].to_numpy(), line=line, dots=dots, head=head, uid=u))
        handles = [plt.Line2D([], [], marker="o", ls="", color=cols[k], label=_legend_text(k, p["labels"], p.get("names")))
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
                extra_axes.append(ax2)
        axt.set_xlim(start, end)
        axt.tick_params(axis="x", labelsize=6, rotation=25)
        cur = axt.axvline(clock[0], color="k", lw=1.0)
        title = ax.set_title(p["name"], fontsize=9)
        arts.append(dict(ax=ax, axt=axt, im=im, tr=tr, cur=cur, title=title, p=p, cols=cols,
                         extra=([cb.ax] if cb is not None else []) + extra_axes))
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
                e = int(t["li"][n - 1]) + 1 if n else 0
                t["line"].set_data(t["xl"][:e], t["yl"][:e])
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
                    if rscale == "mode":
                        lim = _mode_lim(a["p"]["state_reward"][last_k])
                        if lim is not None:
                            a["im"].set_clim(*lim)
            if not multi and last_k is not None:
                a["title"].set_text(f"{a['p']['name']}: {_legend_text(last_k, a['p']['labels'], a['p'].get('names'))}")
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
    if c.get("animate", True):
        writer = animation.PillowWriter(fps=fps) if fmt == "gif" else animation.FFMpegWriter(fps=fps, bitrate=2400)
        ani = animation.FuncAnimation(fig, update, frames=len(clock), blit=False)
        ani.save(out_file, writer=writer, dpi=dpi)
    stills = _save_stills(c, fig, arts, update, clock, out_file)
    update(len(clock) - 1)
    fig.savefig(os.path.splitext(out_file)[0] + "_still.png", dpi=max(dpi, 150), bbox_inches="tight")
    plt.close(fig)
    return out_file if c.get("animate", True) else (stills[0] if stills else os.path.splitext(out_file)[0] + "_still.png")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _season_bins(c):
    sc = c.get("season") or {}
    bins = sc.get("bins", "month")
    if bins == "month":
        return [[m] for m in range(1, 13)], MONTHS
    bins = [[int(m) for m in (b if isinstance(b, (list, tuple)) else [b])] for b in bins]
    names = sc.get("bin_names") or ["-".join(MONTHS[m - 1] for m in b) if len(b) <= 2 else
                                    f"{MONTHS[b[0] - 1]}-{MONTHS[b[-1] - 1]}" for b in bins]
    return bins, names


def render_season(c, p, sel, dry_run=False):
    """Pooled calendar view: one frame per season bin, fixes of that bin coloured by mode / class over the faint
    full track set, with per-bin mode shares below (current bin highlighted). Also a facet grid and a shares csv."""
    name = c.get("name", "season")
    d = p["data"]
    if sel.get("uids"):
        d = d[d["uid"].isin([str(u) for u in sel["uids"]])]
    if not len(d):
        print(f"[{name}] {p['name']}: no fixes")
        return []
    bins, names = _season_bins(c)
    month = d["timestamp"].dt.month.to_numpy()
    b_of = np.full(len(d), -1)
    for i, b in enumerate(bins):
        b_of[np.isin(month, b)] = i
    K = p["K"]
    cols = mode_colors(K, p["labels"], c.get("colour_by", "mode"))
    cnt = np.zeros((len(bins), K))
    np.add.at(cnt, (b_of[b_of >= 0], d["latent"].to_numpy(int)[b_of >= 0]), 1)
    if (c.get("season") or {}).get("drop_empty", True) and (cnt.sum(1) > 0).any():
        keep = np.flatnonzero(cnt.sum(1) > 0)          # e.g. whales: Jul-Dec only
        remap = np.full(len(bins), -1)
        remap[keep] = np.arange(len(keep))
        b_of = np.where(b_of >= 0, remap[np.maximum(b_of, 0)], -1)
        bins, names, cnt = [bins[i] for i in keep], [names[i] for i in keep], cnt[keep]
    share = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
    sc = c.get("season") or {}
    n_an = np.array([d[b_of == i]["uid"].nunique() for i in range(len(bins))])
    n_fx = cnt.sum(1).astype(int)
    low = n_an < int(sc.get("min_animals", 3))
    # class rows: summed shares per class (e.g. all explore = transit + reversal)
    cls_of = {k: str(p["labels"].get(k, "")).lower() for k in range(K)}
    classes = [x for x in ("exploit", "explore", "ambiguous") if x in cls_of.values()]
    use_cls = bool(sc.get("class_rows", True)) and len(classes) > 1
    cshare = np.stack([share[:, [k for k in range(K) if cls_of[k] == x]].sum(1) for x in classes], 1) \
        if use_cls else np.zeros((len(bins), 0))
    od = os.path.join(c["out_dir"], name)
    os.makedirs(od, exist_ok=True)
    tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in p["name"])
    base = os.path.join(od, f"{name}_{tag}")
    tab = pd.DataFrame(share, columns=[f"share_{k}" for k in range(K)]).assign(
        bin=names, n_fixes=n_fx, n_animals=n_an, low_n=low)
    for j, x in enumerate(classes if use_cls else []):
        tab[f"share_{x}"] = cshare[:, j]
    tab[["bin", "n_fixes", "n_animals", "low_n"] + [f"share_{k}" for k in range(K)] +
        [f"share_{x}" for x in (classes if use_cls else [])]].to_csv(base + "_shares.csv", index=False)
    print(f"[{name}] {p['name']}: {d['uid'].nunique()} animals, {len(d)} fixes")
    print(tab[["bin", "n_fixes", "n_animals", "low_n"]].to_string(index=False))
    if dry_run:
        return []
    view, env_ext = _extent(c, [d], None)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cols[k], label=_legend_text(k, p["labels"], p.get("names")))
               for k in range(K)]
    x, y, z = d["longitude"].to_numpy(), d["latitude"].to_numpy(), d["latent"].to_numpy(int)

    def faint(ax):
        for _, g in d.groupby("uid"):
            xl, yl, _ = _line_with_breaks(g["timestamp"].to_numpy(), g["longitude"].to_numpy(),
                                          g["latitude"].to_numpy(), c.get("break_gap_h"))
            ax.plot(xl, yl, color="0.8", lw=0.4, zorder=1)
        ax.set_xlim(view[0], view[1])
        ax.set_ylim(view[2], view[3])
        ax.set_aspect("equal")

    def bars(axb):
        bottom = np.zeros(len(bins))
        for k in range(K):
            axb.bar(range(len(bins)), share[:, k], bottom=bottom, color=cols[k], width=0.85, linewidth=0)
            bottom += share[:, k]
        for i in np.flatnonzero(low):
            axb.bar(i, 1, color="none", edgecolor="0.3", hatch="///", width=0.85, linewidth=0)
        axb.set_xticks(range(len(bins)))
        axb.set_xticklabels([f"{names[i]}\n{n_an[i]}" for i in range(len(bins))], fontsize=6.5)
        axb.set_ylim(0, 1)
        axb.set_ylabel("mode share", fontsize=7)
        axb.tick_params(labelsize=6)

    fig = plt.figure(figsize=c.get("figsize") or (6.2, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.4], hspace=0.25)
    ax, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    faint(ax)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("longitude", fontsize=8)
    ax.set_ylabel("latitude", fontsize=8)
    ax.legend(handles=handles, fontsize=6.5, loc=c.get("legend_loc", "lower left"), framealpha=0.85).set_zorder(10)
    dots = ax.scatter([], [], s=8, zorder=3, linewidths=0)
    bars(axb)
    hl = matplotlib.patches.Rectangle((-0.5, 0), 1.0, 1.0, color="k", alpha=0.12, zorder=0)
    axb.add_patch(hl)
    title = ax.set_title("", fontsize=9)

    def update(i):
        m = b_of == i
        dots.set_offsets(np.c_[x[m], y[m]] if m.any() else np.empty((0, 2)))
        if m.any():
            dots.set_facecolors(cols[z[m]])
        hl.set_x(i - 0.5)
        title.set_text(f"{p['name']}: {names[i]}  ({int(m.sum())} fixes, {d[m]['uid'].nunique()} animals)")
        return [dots, hl]

    fmt = c.get("format", "gif")
    if fmt == "mp4" and not animation.writers.is_available("ffmpeg"):
        fmt = "gif"
    out = []
    if c.get("animate", True):
        f = base + "." + fmt
        fps = int((c.get("season") or {}).get("fps", 1))
        writer = animation.PillowWriter(fps=fps) if fmt == "gif" else animation.FFMpegWriter(fps=fps, bitrate=2400)
        animation.FuncAnimation(fig, update, frames=len(bins), blit=False).save(f, writer=writer,
                                                                               dpi=int(c.get("dpi", 110)))
        out.append(f)
        print("wrote", f)
    clock = np.array([np.datetime64(f"2000-{b[0]:02d}-15") for b in bins])
    arts = [dict(ax=ax, axt=axb, extra=[])]
    for sp in (c.get("stills") or []):
        i = names.index(sp) if isinstance(sp, str) and sp in names else _still_index(sp, clock)
        if i is None:
            continue
        update(i)
        fs = f"{base}_still_{names[i]}.png"
        fig.savefig(fs, dpi=int(c.get("still_dpi", 200)), bbox_inches="tight")
        out.append(fs)
    plt.close(fig)
    # facet grid of every bin + share heatmap
    rows_lab = [_legend_text(k, p["labels"], p.get("names")) for k in range(K)] + [f"all {x}" for x in
                                                                                 (classes if use_cls else [])]
    H = np.concatenate([share, cshare], 1).T                     # (rows, bins)
    min_an = int(sc.get("min_animals", 3))

    def heat(ah, fig):
        im = ah.imshow(H, aspect="auto", cmap="magma_r", vmin=0, vmax=1 if use_cls else max(float(share.max()), 1e-6))
        ah.set_xticks(range(nb))
        ah.set_xticklabels([f"{names[i]}\n{n_an[i]} an. / {n_fx[i]}" for i in range(nb)], fontsize=6)
        ah.set_yticks(range(len(rows_lab)))
        ah.set_yticklabels(rows_lab, fontsize=6.5)
        if use_cls:
            ah.axhline(K - 0.5, color="w", lw=2)
        vm = 1 if use_cls else share.max()
        for i in range(nb):
            for r in range(len(rows_lab)):
                ah.text(i, r, f"{H[r, i]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="w" if H[r, i] > 0.5 * vm else "k")
            if low[i]:
                ah.add_patch(matplotlib.patches.Rectangle((i - 0.5, -0.5), 1, len(rows_lab), fill=False,
                                                          hatch="///", edgecolor="0.5", lw=0))
        if low.any():
            ah.set_xlabel(f"hatched: fewer than {min_an} animals", fontsize=6)
        fig.colorbar(im, ax=ah, fraction=0.03, pad=0.01, label="share of fixes")
        return im

    nb = len(bins)
    ncol = int((c.get("season") or {}).get("grid_cols", 4 if nb > 4 else nb))
    nrow = int(np.ceil(nb / ncol))
    aspect = (view[3] - view[2]) / max(view[1] - view[0], 1e-6)
    fig = plt.figure(figsize=(2.3 * ncol, 2.3 * nrow * min(max(aspect, 0.5), 1.6) + 2.0))
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1] * nrow + [0.9 + 0.15 * cshare.shape[1]], hspace=0.3, wspace=0.08)
    for i in range(nb):
        a = fig.add_subplot(gs[i // ncol, i % ncol])
        faint(a)
        m = b_of == i
        a.scatter(x[m], y[m], c=cols[z[m]], s=3, linewidths=0, zorder=3)
        a.set_title(f"{names[i]} ({n_an[i]} animals, {int(m.sum())} fixes)" + (" *" if low[i] else ""),
                    fontsize=6.5)
        a.set_xticks([])
        a.set_yticks([])
    ah = fig.add_subplot(gs[nrow, :])
    heat(ah, fig)
    unit = "month" if all(len(b) == 1 for b in bins) else "season"
    fig.suptitle(f"{p['name']}: mode by {unit} ({d['uid'].nunique()} animals)", fontsize=10, y=1.0)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=min(K, 5), fontsize=7,
               frameon=False)
    fg = base + "_grid.png"
    fig.savefig(fg, dpi=int(c.get("still_dpi", 200)), bbox_inches="tight")
    plt.close(fig)
    print("wrote", fg)
    fig, ah = plt.subplots(figsize=(max(3.2, 0.75 * nb + 2.4), 0.32 * len(rows_lab) + 1.2))
    heat(ah, fig)
    ah.set_title(f"{p['name']}: mode share by {unit} ({d['uid'].nunique()} animals)", fontsize=8)
    fhm = base + "_heatmap.png"
    fig.savefig(fhm, dpi=int(c.get("still_dpi", 200)), bbox_inches="tight")
    plt.close(fig)
    print("wrote", fhm)
    return out + [fg, fhm]


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
    if c.get("align_modes", c.get("align_to_reference")) and len(panels) > 1:
        panels = [panels[0]] + [align_modes(panels[0], p) for p in panels[1:]]
    if c.get("mode") == "season":
        return [f for p in panels for f in render_season(c, p, sel, dry_run)]
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
    ap.add_argument("--stills-only", "--stills_only", dest="stills_only", action="store_true",
                    help="write only the PNG stills (no gif / mp4)")
    a = ap.parse_args(argv)
    files = []
    for c in job_confs(read_conf(a.yaml)):
        if a.only and c.get("name") not in a.only:
            continue
        for k in ("format", "dpi", "background"):
            if getattr(a, k) is not None:
                c[k] = getattr(a, k)
        if a.stills_only:
            c["animate"] = False
        for k in ("n", "strategy", "max_frames"):
            if getattr(a, k) is not None:
                c["select"] = dict(c["select"], **{k: getattr(a, k)})
        files += run_job(c, a.uids, a.dry_run)
    return files


if __name__ == "__main__":
    main()
