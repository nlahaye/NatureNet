"""
Per-animal time series from SWIRL outputs (used by plot_swirl_new.py, correlation_analysis_shark.py and
exploit_explore_init.py).

Replaces the counter-based matching in the old plot_swirl_new.run_plots / plot_time_series:
- latent rows are mapped to tracks with the sequence ids saved in every LL file (seq_key / seq_seg /
  seq_start; latent step t of a row is segment position seq_start + seq_offset + t, seq_offset 1 (the SWIRL
  alignment) unless the LL file stores another (HIAVI hiavi_align: full -> 0));
- tracks / states come from the same loader as training (swirl_data.load_segments), lat / lon / time from
  the movement DataFrames;
- every column is computed on the full segment and then subset, so all columns have equal length:
  step_km (haversine, into the fix), speed_kmh (/ dt), turn_angle (forward differences, wrapped to
  [-pi, pi], NaN for stationary / end steps), env (abstract state), reward_likelihood (reward of the
  visited state / pair / action under the decoded mode; invalid -> NaN), month sin / cos, static layers
  sampled by lat / lon on their own extent, gathered input features;
- rewards are not min-max scaled per mode (scaling per mode makes modes incomparable); reward_normalize:
  "global" scales all modes together.
"""
import os
import pickle
import re

import numpy as np
import pandas as pd

from naturenet.environment.pipeline_checks import (
    INVALID_STATE,
    check_latent_file,
    check_series_columns,
    check_turn_angles,
)
from naturenet.models.irl.swirl_data import as_list, load_segments

EARTH_RADIUS_KM = 6371.0088

# (model, order) -> (reward / model file suffix, train LL suffix, test LL suffix) for the final (holdout) runs.
SWIRL_FILES = {
    ("linear", "s1"): ("_S1_naturenet_iter2_temp_2", "_S1_naturenet_iter2_LL1", "_S1_naturenet_iter2_test_LL1"),
    ("linear", "s2"): ("_S2_naturenet_iter2", "_S2_naturenet_iter2_LL2", "_S2_naturenet_iter2_test_LL2"),
    ("nn", "s1"): ("_naturenet_MLP_S1_new1_reward_filtered", "_naturenet_MLP_S1_new1_LL", "_naturenet_MLP_S1_new1_LL_test"),
    ("nn", "s2"): ("_naturenet_MLP_S2_new2_reward_filtered", "_naturenet_MLP_S2_new2_LL", "_naturenet_MLP_S2_new2_LL_test"),
    ("hiavi", "s1"): ("_hiavi", "_hiavi_LL", "_hiavi_LL_test"),
}


def _base(yml_conf):
    return os.path.join(yml_conf["out_dir"], f'{yml_conf["run_uid"]}_{yml_conf["n_hidden"]}_{yml_conf["seed"]}')


def swirl_file(yml_conf, model, order, kind, tag=""):
    """kind: 'reward' | 'train' | 'test'. tag='_KF_k' for K-fold outputs."""
    i = {"reward": 0, "train": 1, "test": 2}[kind]
    return _base(yml_conf) + SWIRL_FILES[(model, order)][i] + tag + ".npz"


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

class RewardModel:
    """
    reward(k, s, p, a) for the decoded series, and a (K, S) state reward for maps.

    linear s1: new_reward (K, 1, S)          -> R[k, s]
    linear s2: new_reward (K, 1, S*M), layout x*M + slot, slot = index of p in prev_state_map[s]
               -> (K, S, M) with an (S, M) validity mask (old code reshaped to (K, S, n_actions), which
               only worked while M happened to equal A, and masked whole (K, S) rows)
    nn s1:     reward_filtered (K, S, A)     -> R[k, s, a]
    nn s2:     reward_filtered (K, S, S, A)  -> R[k, s, p, a], valid_pairs (S, S)
    hiavi s1:  r (K, S, A) (IAVI reward)     -> R[k, s, a] (as nn s1); state_reward = E_{a ~ pi_k}[r_k(s, a)]
               (IAVI fixes r only up to a per-state shift and returns mean_a r(s, a) = 0, so the plain action
               mean would be identically 0)
    """

    def __init__(self, model, order, R, n_states, prev_state_map=None, valid_pairs=None, normalize="none",
                 action_weights=None):
        self.model, self.order, self.S = model, order, n_states
        R = np.array(R, dtype=float)  # copy: never modify the caller's array
        if model == "linear":
            R = R[:, 0, :]
            if order == "s2":
                if prev_state_map is None:
                    raise ValueError("linear S-2 reward needs prev_state_map (saved by the updated run_swirl)")
                M = R.shape[1] // n_states
                if M * n_states != R.shape[1]:
                    raise ValueError(f"S-2 reward length {R.shape[1]} is not a multiple of n_states={n_states}")
                R = R.reshape(R.shape[0], n_states, M)
                mask = np.zeros((n_states, M), dtype=bool)
                self.slot = {}
                for s in range(n_states):
                    prev = prev_state_map.get(s, []) if isinstance(prev_state_map, dict) else prev_state_map[s]
                    prev = [int(v) for v in np.asarray(prev).ravel()]
                    mask[s, :len(prev)] = True
                    for m, p in enumerate(prev):
                        self.slot[(s, int(p))] = m
                R = np.where(mask[None], R, np.nan)
                self.state_reward = np.nanmean(R, axis=2)
            else:
                self.state_reward = R
        else:
            if order == "s2":
                if valid_pairs is not None:
                    R = np.where(np.asarray(valid_pairs, dtype=bool)[None, :, :, None], R, np.nan)
                self.state_reward = np.nanmean(R, axis=(2, 3))
            elif action_weights is not None:  # (K, S, A) policy: E_pi[r]
                w = np.asarray(action_weights, dtype=float)
                self.state_reward = (R * w).sum(axis=2) / np.maximum(w.sum(axis=2), 1e-300)
            else:
                self.state_reward = R.mean(axis=2)
        if normalize == "global":
            lo, hi = np.nanmin(R), np.nanmax(R)
            R = (R - lo) / (hi - lo if hi > lo else 1.0)
            self.state_reward = (self.state_reward - lo) / (hi - lo if hi > lo else 1.0)
        self.R = R
        self.K = R.shape[0]

    def lookup(self, k, s, p, a):
        """Vectorised; returns NaN where any index is invalid."""
        k, s, p, a = (np.asarray(v) for v in (k, s, p, a))
        out = np.full(k.shape, np.nan)
        ok = (k >= 0) & (s >= 0) & (s < self.S)
        if self.order == "s2":
            ok &= (p >= 0) & (p < self.S)
        if self.model in ("nn", "hiavi"):
            ok &= (a >= 0) & (a < self.R.shape[-1])
        idx = np.nonzero(ok)[0]
        if self.model == "linear" and self.order == "s1":
            out[idx] = self.R[k[idx], s[idx]]
        elif self.model == "linear":
            for i in idx:
                m = self.slot.get((int(s[i]), int(p[i])))
                if m is not None:
                    out[i] = self.R[k[i], s[i], m]
        elif self.order == "s1":
            out[idx] = self.R[k[idx], s[idx], a[idx]]
        else:
            out[idx] = self.R[k[idx], s[idx], p[idx], a[idx]]
        return out


def load_reward_model(yml_conf, model, order, tag=""):
    f = swirl_file(yml_conf, model, order, "reward", tag)
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} not found (run the final SWIRL run first)")
    p = np.load(f, allow_pickle=True)
    S = int(yml_conf["n_states"])
    norm = yml_conf.get("reward_normalize", "none")
    if model == "linear":
        key = yml_conf.get("reward_key", "new_reward")
        psm = p["prev_state_map"] if "prev_state_map" in p.files else None
        if psm is not None and psm.ndim == 0:  # a pickled dict
            psm = psm.item()
        return RewardModel(model, order, p[key], S, prev_state_map=psm, normalize=norm)
    if model == "hiavi":
        if order != "s1":
            raise ValueError("HIAVI has first-order (s1) rewards only")
        return RewardModel(model, order, p[yml_conf.get("reward_key", "r")], S, normalize=norm, action_weights=p["pis"])
    return RewardModel(model, order, p["reward_filtered"], S,
                       valid_pairs=p["valid_pairs"] if "valid_pairs" in p.files else None, normalize=norm)


# ---------------------------------------------------------------------------
# Latents
# ---------------------------------------------------------------------------

def load_latents(yml_conf, model, order, splits=("train", "test"), tag=""):
    """Concatenated latent paths with their sequence ids; split column says which file each row came from."""
    rows = []
    for split in splits:
        f = swirl_file(yml_conf, model, order, split, tag)
        if not os.path.exists(f):
            print("WARNING: missing", f)
            continue
        z = np.load(f, allow_pickle=True)
        zs = z["learnt_zs"] if "learnt_zs" in z.files else z["jax_path_vmap"]
        meta = {k: (z[k] if k in z.files else None) for k in ("seq_key", "seq_seg", "seq_start")}
        check_latent_file(zs, meta["seq_key"], name=os.path.basename(f))
        off = int(z["seq_offset"]) if "seq_offset" in z.files else 1  # latent t <-> fix seq_start + off + t
        # kinematic runs: per-step P(exploit) and the per-intention exploit / explore labels
        pe = np.asarray(z["p_exploit"], float) if "p_exploit" in z.files else np.full(np.shape(zs), np.nan)
        kl = ",".join(map(str, z["kin_labels"])) if "kin_labels" in z.files else ""
        rows.append(dict(zs=np.asarray(zs), split=np.full(len(zs), split), seq_offset=np.full(len(zs), off),
                         p_exploit=pe, kin_labels=np.full(len(zs), kl, dtype=object), **meta))
    if not rows:
        raise FileNotFoundError("No LL / latent files found for " + _base(yml_conf) + f" ({model}, {order})")
    return {k: np.concatenate([r[k] for r in rows]) for k in ("zs", "split", "seq_key", "seq_seg", "seq_start",
                                                                "seq_offset", "p_exploit", "kin_labels")}


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing (radians, clockwise from north)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.arctan2(y, x)


def naive_utc(ts):
    """Any timestamps (strings, tz-aware / naive Timestamps, datetime64) -> tz-naive UTC datetime64[ns] array.
    (A tz-aware Series .to_numpy() is an object array of Timestamps, which numpy cannot divide by a timedelta64.)"""
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(np.asarray(ts, dtype=object)), utc=True, format="mixed")) \
        .tz_convert(None).to_numpy("datetime64[ns]")


def kinematics(lat, lon, t, min_step_km=1e-3):
    """
    Per fix i: step_km / dt_h / speed_kmh of the step INTO fix i (NaN at i = 0); turn_angle at fix i =
    bearing(i -> i+1) - bearing(i-1 -> i), wrapped to [-pi, pi] (positive = right turn), NaN at the ends
    and when either step is shorter than min_step_km (stationary steps have no heading).
    """
    n = len(lat)
    step = np.full(n, np.nan)
    dt = np.full(n, np.nan)
    turn = np.full(n, np.nan)
    if n >= 2:
        step[1:] = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
        t = t if np.asarray(t).dtype.kind == "M" else naive_utc(t)
        dt[1:] = (t[1:] - t[:-1]) / np.timedelta64(1, "h")
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.nan)
    if n >= 3:
        b = bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])  # b[j]: j -> j+1
        d = b[1:] - b[:-1]  # turn at fix j+1
        d = (d + np.pi) % (2 * np.pi) - np.pi
        moving = step[1:] >= min_step_km  # step into j+1
        ok = moving[:-1] & moving[1:]
        turn[1:-1] = np.where(ok, d, np.nan)
    return step, dt, speed, turn


def season_of(month):
    """Meteorological season (northern hemisphere): 0 DJF, 1 MAM, 2 JJA, 3 SON."""
    return (np.asarray(month) % 12) // 3


# ---------------------------------------------------------------------------
# Static layers (ship density, bathymetry, distance to coast, ...)
# ---------------------------------------------------------------------------

def _load_layer_array(path):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        z = np.load(path)
        return z[z.files[0]]
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return np.asarray(pickle.load(f))
    from osgeo import gdal  # GeoTIFF etc.
    ds = gdal.Open(path)
    return ds.GetRasterBand(1).ReadAsArray()


class Layer:
    """2-D raster with its own geographic extent; nearest-cell sampling, off-extent / nodata -> NaN."""

    def __init__(self, arr, lat_bounds, lon_bounds, lat_ascending=False, nodata=None):
        self.arr = np.asarray(arr, dtype=float)
        if self.arr.ndim != 2:
            raise ValueError(f"layer must be 2-D, got {self.arr.shape}")
        self.lat_min, self.lat_max = sorted(map(float, lat_bounds))
        self.lon_min, self.lon_max = sorted(map(float, lon_bounds))
        self.asc, self.nodata = lat_ascending, nodata

    def sample(self, lat, lon):
        lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        H, W = self.arr.shape
        fr = (lat - self.lat_min) / (self.lat_max - self.lat_min) if self.asc else \
            (self.lat_max - lat) / (self.lat_max - self.lat_min)
        fc = (lon - self.lon_min) / (self.lon_max - self.lon_min)
        ok = (fr >= 0) & (fr <= 1) & (fc >= 0) & (fc <= 1)
        r = np.clip(np.floor(np.nan_to_num(fr * H)).astype(np.int64), 0, H - 1)
        c = np.clip(np.floor(np.nan_to_num(fc * W)).astype(np.int64), 0, W - 1)
        out = np.full(lat.shape, np.nan)
        out[ok] = self.arr[r[ok], c[ok]]
        if self.nodata is not None:
            out[out == self.nodata] = np.nan
        return out


def load_layers(yml_conf):
    """static_layers: {name: {file, lat_bounds, lon_bounds, lat_ascending (False), nodata (None)}}."""
    out = {}
    for name, spec in (yml_conf.get("static_layers") or {}).items():
        out[name] = Layer(_load_layer_array(spec["file"]), spec.get("lat_bounds", yml_conf.get("lat_bounds")),
                          spec.get("lon_bounds", yml_conf.get("lon_bounds")), spec.get("lat_ascending", False),
                          spec.get("nodata"))
    return out


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------

def _df_key(seq_key, df_run_uid):
    k = str(seq_key)
    pre = str(df_run_uid) + "_"
    return k[len(pre):] if k.startswith(pre) else k


def build_track_series(yml_conf, model="linear", order="s1", splits=("train", "test"), tag="",
                       include_aug=False, drop_unlabelled=True, rewards=None, latents=None):
    """
    Returns {animal key: DataFrame}, one row per fix (rows without a decoded latent dropped unless
    drop_unlabelled=False). Columns: uid, segment, seg_pos, split, timestamp, Time (h since the animal's
    first fix), dt_h, month, month_sin, month_cos, season, latitude, longitude, env, latent, step_km
    (alias distance_traveled), speed_kmh, turn_angle, reward_likelihood, static layers, input_keys; kinematic
    runs (LL files with kin_labels / p_exploit) add latent_class (exploit / explore / ambiguous) and p_exploit.
    step_out_km / speed_out_kmh: the move i -> i+1 that the latent at row i explains (speed_kmh is i-1 -> i).
    """
    df_uid = yml_conf["df_run_uid"]
    with open(os.path.join(yml_conf["df_dir"], df_uid + "_dfs.pkl"), "rb") as f:
        movement_dfs = pd.read_pickle(f)
    segs = {(s["key"], s["seg"]): s for s in load_segments(yml_conf)}
    rewards = rewards if rewards is not None else load_reward_model(yml_conf, model, order, tag)
    lat_ = latents if latents is not None else load_latents(yml_conf, model, order, splits, tag)
    layers = load_layers(yml_conf)
    input_keys = as_list(yml_conf.get("input_keys"))
    min_step = float(yml_conf.get("min_step_km", 1e-3))

    # latent per (key, seg, pos)
    lab = {}
    for i in range(len(lat_["zs"])):
        key, seg, st = str(lat_["seq_key"][i]), int(lat_["seq_seg"][i]), int(lat_["seq_start"][i])
        off = int(lat_["seq_offset"][i]) if "seq_offset" in lat_ else 1
        if not include_aug and re.search(r"__aug\d+", key):
            continue
        d = lab.setdefault((key, seg), {})
        pe_i = lat_["p_exploit"][i] if "p_exploit" in lat_ else np.full(len(lat_["zs"][i]), np.nan)
        kl_i = str(lat_["kin_labels"][i]).split(",") if "kin_labels" in lat_ and lat_["kin_labels"][i] else None
        for t, z in enumerate(lat_["zs"][i]):
            if int(z) < 0:                      # padded step (segment_policy: variable): no fix
                continue
            cls_ = kl_i[int(z)] if kl_i is not None and 0 <= int(z) < len(kl_i) else ""
            d[st + off + t] = (int(z), str(lat_["split"][i]), float(pe_i[t]), cls_)

    inputs_cache = {}
    out = {}
    for (key, seg), pos2z in sorted(lab.items()):
        dkey = _df_key(key, df_uid)
        animal = re.sub(r"__aug\d+", "", dkey)
        if dkey not in movement_dfs:
            print("WARNING: no movement DataFrame for", dkey, "- skipped")
            continue
        df = movement_dfs[dkey][seg]
        sg = segs.get((key, seg))
        if sg is None:
            raise KeyError(f"{key} segment {seg} in the LL file but not in the actions / positions files")
        states, actions = sg["states"], sg["actions"]
        if len(df) != len(states):
            raise ValueError(f"{dkey} segment {seg}: {len(df)} fixes vs {len(states)} states "
                             "(preprocessing outputs out of date?)")
        n = len(df)
        lat = df["latitude"].to_numpy(dtype=float)
        lon = df["longitude"].to_numpy(dtype=float)
        ts = naive_utc(df["timestamp"])  # tz-naive UTC datetime64[ns]
        step, dt, speed, turn = kinematics(lat, lon, ts, min_step)

        latent = np.full(n, np.nan)
        split = np.full(n, "", dtype=object)
        p_expl = np.full(n, np.nan)
        lclass = np.full(n, "", dtype=object)
        for pos, (z, sp, pe_, cl_) in pos2z.items():
            if pos < n:
                latent[pos], split[pos], p_expl[pos], lclass[pos] = z, sp, pe_, cl_
        env = np.where(states == INVALID_STATE, np.nan, states.astype(float))
        prev = np.r_[INVALID_STATE, states[:-1]]
        act = np.r_[actions, -1]  # action taken AT fix i (i -> i+1)
        k = np.where(np.isfinite(latent), latent, -1).astype(np.int64)
        rew = rewards.lookup(k, states, prev, act)

        month = pd.DatetimeIndex(ts).month.to_numpy()
        cols = dict(uid=np.full(n, animal), segment=np.full(n, seg), seg_pos=np.arange(n), split=split,
                    timestamp=pd.DatetimeIndex(ts), dt_h=dt, month=month,
                    month_sin=np.sin(2 * np.pi * (month - 1) / 12), month_cos=np.cos(2 * np.pi * (month - 1) / 12),
                    season=season_of(month), latitude=lat, longitude=lon, env=env, latent=latent, step_km=step,
                    distance_traveled=step, speed_kmh=speed, turn_angle=turn, reward_likelihood=rew)
        # the move OUT of fix i (i -> i+1) is the step the latent at row i emits / explains (speed_kmh is INTO i)
        cols["step_out_km"] = np.r_[step[1:], np.nan]
        cols["speed_out_kmh"] = np.r_[speed[1:], np.nan]
        if any(lclass):  # kinematic runs: exploit / explore label of the decoded intention + P(exploit)
            cols["latent_class"] = lclass
            cols["p_exploit"] = p_expl
        for name, layer in layers.items():
            cols[name] = layer.sample(lat, lon)
        if input_keys:
            if dkey not in inputs_cache:
                fin = os.path.join(yml_conf["df_dir"], df_uid + "_" + dkey + "_states_input.pkl")
                with open(fin, "rb") as f:
                    inputs_cache[dkey] = pickle.load(f)
            feats = np.asarray(inputs_cache[dkey][seg], dtype=float)
            if feats.ndim == 2 and feats.shape == (n, 0):
                # no env scene for any fix of this segment (e.g. track dates outside the scene period):
                # gather_input wrote a zero-width array; treat as all-missing
                feats = np.full((n, len(input_keys)), np.nan)
            if feats.ndim != 2 or feats.shape[0] != n or feats.shape[1] < len(input_keys):
                raise ValueError(f"{dkey} segment {seg}: input features {feats.shape} for {n} fixes / "
                                 f"{len(input_keys)} input_keys")
            for j, name in enumerate(input_keys):
                cols[name] = feats[:, j]
        check_series_columns({c: v for c, v in cols.items() if c not in ("uid", "split", "timestamp", "latent_class",
                                                                         "p_exploit")}, name=f"{dkey} segment {seg}")
        check_turn_angles(turn, name=f"{dkey} segment {seg}")
        sdf = pd.DataFrame(cols)
        if drop_unlabelled:
            sdf = sdf[np.isfinite(sdf["latent"])]
        out.setdefault(animal, []).append(sdf)

    series = {}
    for animal, parts in out.items():
        d = pd.concat(parts, ignore_index=True).sort_values(["timestamp", "segment", "seg_pos"]).reset_index(drop=True)
        d.insert(4, "Time", (d["timestamp"] - d["timestamp"].min()) / pd.Timedelta(hours=1))
        if drop_unlabelled and d["latent"].isna().any():
            raise AssertionError(f"{animal}: NaN latents after filtering")
        d["latent"] = d["latent"].astype(int) if drop_unlabelled else d["latent"]
        series[animal] = d
    return series


def export_series(series, out_dir, prefix=""):
    """One <prefix><uid>_time_series_pd.xlsx (and .csv) per animal (same file name as before)."""
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for uid, d in series.items():
        f = os.path.join(out_dir, prefix + uid + "_time_series_pd")
        d.to_csv(f + ".csv", index=False)
        try:
            d.to_excel(f + ".xlsx", header=True, index=False)
        except ImportError:
            print("openpyxl not installed; wrote CSV only")
        files.append(f + ".csv")
    return files
