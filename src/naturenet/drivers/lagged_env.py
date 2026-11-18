"""
Scene-derived covariates at each fix that sample_env_inputs cannot give: lagged / rolling means of daily channels and
per-scene feature thresholds (habitat analyses 1 and 2; any v2 cmems_env scene set, any species).

  lagged   For every fix with date D and every window w = [a, b] (days, a <= b): mean of channel c over the scene
           days D - b .. D - a at the fix's grid cell (same cell lookup as sample_env_inputs). A value is kept when at
           least min_frac of the window days have a usable scene (else NaN). [0, 0] = same day, [0, 7] = 8-day mean
           ending on D, [7, 13] = mean of the week one week before D, ...
           Writes <out_dir>/<df_run_uid>_<key>_lagged_input.pkl (list per segment of float32 (len(segment), W * C),
           aligned row by row with the states_input files) and <out_dir>/<df_run_uid>_lagged_input_channels.json
           (column names "<channel>__<window name>").
  scene    Per scene day with fixes: domain statistics of the selected channels over valid (non-fill) pixels, e.g.
           the 90th percentile of the SST gradient magnitude (front threshold, Belkin & O'Reilly-style per-scene
           decile, as in SIT-FUSE cluster_interpret.front_mask) and the std of the Okubo-Weiss parameter (eddy cores
           W < -0.2 std(W), Isern-Fontanet et al. 2003, as in cluster_interpret.eddy_mask).
           Writes <csv> with columns date, <channel>_p<q>, <channel>_sd, <channel>_n.

Each scene day is read once (selected channels, bounding box of that day's points), as in sample_env_inputs.

YAML (the "lagged" / "scene_stats" blocks of a habitat_tests spec, or standalone):
  sample_inputs_yaml: config/v2/environment/whales_sample_inputs_v2.yaml   # df_dir, df_run_uid, scenes, meta
  lagged:
    channels: [wind_speed, ekman_pumping, VHM0]
    windows: {d0: [0, 0], m8: [0, 7], m30: [0, 29], lag1w: [7, 13], lag2w: [14, 20], lag3w: [21, 27]}
    min_frac: 0.5
    out_dir: <out>/lagged_inputs/          # used fixes;   <out>/lagged_inputs_avl/ for the available points
    workers: 4
    python -m naturenet.drivers.lagged_env -y <spec> --part lagged [--avl]
    python -m naturenet.drivers.lagged_env -y <spec> --part scene
"""
import argparse
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from naturenet.drivers.common import abs_cfg, atomic_csv, read_yaml


# ----------------------------------------------------------------------------- scenes
def scene_setup(sconf):
    """(calendar {day: path}, sorted days, channel list, grid meta, fill value) of a sample_env_inputs config."""
    from naturenet.environment.sample_env_inputs import read_calendar
    with open(sconf["env_meta_json"]) as f:
        meta = json.load(f)
    cal = read_calendar(sconf)
    return cal, sorted(cal), list(meta["channels"]), meta["grid"], float(meta.get("fill_value", -999999.0))


def read_block(path, chan_idx, r0, r1, c0, c1, fill):
    """(C, r1 - r0, c1 - c0) float32 block of a scene zarr (NaN = fill). Replaced in the synthetic tests."""
    import zarr
    arr = zarr.open_array(path, mode="r")
    b = np.asarray(arr.oindex[list(chan_idx), r0:r1, c0:c1], dtype=np.float32)
    b[b == np.float32(fill)] = np.nan
    return b


def fix_cells(sconf, df_dir, keys=None):
    """Per fix of every (selected) key: key, seg, pos, date (day), row, col, ok (inside the env grid)."""
    from naturenet.environment.grid_utils import (grid_from_shape, in_grid_bounds, lat_lon_to_grid_vec,
                                                  _fix_swapped_lat_lon)
    from naturenet.environment.sample_env_inputs import _fix_dates, _load_grid
    with open(sconf["env_meta_json"]) as f:
        g = json.load(f)["grid"]
    H, W = g["shape"]
    uid = sconf["df_run_uid"]
    with open(os.path.join(df_dir, uid + "_dfs.pkl"), "rb") as f:
        dfs = pd.read_pickle(f)
    frames = []
    for key, segs in dfs.items():
        if keys is not None and not keys(key):
            continue
        grid = _load_grid(df_dir, uid, key)
        eg = grid_from_shape(grid, H, W, lon_bounds=g["lon_bounds"], lat_bounds=g["lat_bounds"])
        for si, seg in enumerate(segs):
            n = len(seg)
            if not n:
                continue
            lat, lon = _fix_swapped_lat_lon(grid, seg["latitude"].to_numpy(), seg["longitude"].to_numpy())
            ok = np.isfinite(lat) & np.isfinite(lon)
            ok &= in_grid_bounds(eg, np.where(ok, lat, 0.0), np.where(ok, lon, 0.0))
            rows = np.full(n, -1, np.int64)
            cols = np.full(n, -1, np.int64)
            if ok.any():
                r, c = lat_lon_to_grid_vec(eg, lat[ok], lon[ok])
                rows[ok], cols[ok] = r, c
            frames.append(pd.DataFrame(dict(key=key, seg=si, pos=np.arange(n), date=_fix_dates(seg), row=rows,
                                            col=cols, ok=ok)))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["key", "seg", "pos", "date", "row", "col", "ok"])
    out["ok"] &= pd.notna(out["date"])
    return out, dfs


# ----------------------------------------------------------------------------- lagged / rolling means
def parse_windows(windows):
    out = {}
    for name, w in (windows or {}).items():
        a, b = (int(w), int(w)) if np.isscalar(w) else (int(w[0]), int(w[1]))
        if a < 0 or b < a:
            raise ValueError(f"lag window {name}: need 0 <= a <= b, got {w}")
        out[str(name)] = (a, b)
    if not out:
        raise ValueError("lagged.windows is empty")
    return out


def lagged_means(cells, cal_days, cal_paths, chan_idx, windows, fill, min_frac=0.5, workers=2, reader=None):
    """(N, W, C) window means at the fix cells (NaN where < min_frac of the window days have a value)."""
    reader = reader or read_block
    names = list(windows)
    A = np.array([windows[n][0] for n in names])
    B = np.array([windows[n][1] for n in names])
    N, Wn, C = len(cells), len(names), len(chan_idx)
    S = np.zeros((N, Wn, C))
    K = np.zeros((N, Wn, C))
    ok = cells["ok"].to_numpy(bool)
    D = pd.to_datetime(cells["date"]).to_numpy("datetime64[D]")
    rows, cols = cells["row"].to_numpy(), cells["col"].to_numpy()
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(D[idx], kind="stable")]
    Ds = D[order]
    days = np.asarray(cal_days, dtype="datetime64[D]")

    def work(j):
        d = days[j]
        lo, hi = np.searchsorted(Ds, d + A.min(), "left"), np.searchsorted(Ds, d + B.max(), "right")
        if hi <= lo:
            return j, None, None
        f = order[lo:hi]
        r, c = rows[f], cols[f]
        blk = reader(cal_paths[j], chan_idx, int(r.min()), int(r.max()) + 1, int(c.min()), int(c.max()) + 1, fill)
        return j, f, blk[:, r - r.min(), c - c.min()].T            # (len(f), C)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for j, f, v in ex.map(work, range(len(days))):
            if f is None:
                continue
            lag = (D[f] - days[j]).astype(np.int64)                   # D - d >= 0 inside the window
            fin = np.isfinite(v)
            for w in range(Wn):
                m = (lag >= A[w]) & (lag <= B[w])
                if m.any():
                    S[f[m], w] += np.where(fin[m], v[m], 0.0)
                    K[f[m], w] += fin[m]
    need = np.maximum(1, np.ceil(min_frac * (B - A + 1)))[None, :, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(K >= need, S / np.maximum(K, 1), np.nan)
    return M


def write_lagged(spec, avl=False, reader=None):
    lg = spec["lagged"]
    sconf = read_yaml(abs_cfg(spec["sample_inputs_yaml"]))
    uid = sconf["df_run_uid"]
    if avl:
        df_dir = (spec.get("availability") or {}).get("avl_dir")
        out_dir = lg.get("avl_out_dir") or os.path.join(spec["out_dir"], "lagged_inputs_avl")
    else:
        df_dir = sconf["df_dir"]
        out_dir = lg.get("out_dir") or os.path.join(spec["out_dir"], "lagged_inputs")
    if not df_dir or not os.path.exists(os.path.join(df_dir, uid + "_dfs.pkl")):
        raise FileNotFoundError(f"{df_dir}/{uid}_dfs.pkl (run naturenet.drivers.availability first for --avl)")
    os.makedirs(out_dir, exist_ok=True)
    cal, days, chans, _, fill = scene_setup(sconf)
    want = list(lg["channels"])
    bad = [c for c in want if c not in chans]
    if bad:
        print(f"lagged_env: channels {bad} not in the scene meta - dropped")
        want = [c for c in want if c in chans]
    if not want:
        raise ValueError("lagged_env: none of lagged.channels is in the scenes")
    windows = parse_windows(lg["windows"])
    keys = None if avl else (lambda k: "__aug" not in k)
    cells, dfs = fix_cells(sconf, df_dir, keys)
    M = lagged_means(cells, days, [cal[d] for d in days], [chans.index(c) for c in want], windows, fill,
                     float(lg.get("min_frac", 0.5)), int(lg.get("workers", 2)), reader)
    names = [f"{c}__{w}" for w in windows for c in want]
    flat = M.reshape(len(cells), -1).astype(np.float32)        # (N, W * C), window-major as in names
    cells = cells.reset_index(drop=True)
    for key, g in cells.groupby("key", sort=False):
        segs = [np.full((len(s), len(names)), np.nan, np.float32) for s in dfs[key]]
        for si, gg in g.groupby("seg"):
            segs[int(si)][gg["pos"].to_numpy()] = flat[gg.index.to_numpy()]
        tmp = os.path.join(out_dir, f"{uid}_{key}_lagged_input.pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(segs, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, tmp[:-4])
    with open(os.path.join(out_dir, uid + "_lagged_input_channels.json"), "w") as f:
        json.dump(names, f, indent=1)
    cov = np.isfinite(flat).mean(0)
    print(f"lagged_env: {len(cells)} fixes, {len(names)} columns -> {out_dir}; coverage "
          + ", ".join(f"{n} {c:.2f}" for n, c in zip(names, cov)))
    return out_dir, names


# ----------------------------------------------------------------------------- per-scene feature statistics
def feature_stat_channels(spec):
    """Default scene_stats channels from a habitat_tests features block (front percentile, eddy std)."""
    ft = spec.get("features") or {}
    out = {}
    for f in (ft.get("fronts") or {}).values():
        out.setdefault(f["channel"], dict(percentiles=[]))["percentiles"].append(float(f.get("percentile", 90)))
    if ft.get("eddy"):
        out.setdefault(ft["eddy"]["channel"], dict(percentiles=[]))
    return out


def scene_stats(spec, reader=None):
    """Per scene day that has fixes (used + available): percentiles / std of the feature channels over the domain."""
    st = spec.get("scene_stats") or {}
    sconf = read_yaml(abs_cfg(spec["sample_inputs_yaml"]))
    uid = sconf["df_run_uid"]
    cal, days, chans, grid, fill = scene_setup(sconf)
    reader = reader or read_block
    H, W = grid["shape"]
    req = st.get("channels") or feature_stat_channels(spec)
    q = {c: [float(x) for x in (v.get("percentiles") or [])] for c, v in req.items()}
    use = [c for c in q if c in chans]
    if not use:
        raise ValueError(f"scene_stats: none of {list(q)} in the scene channels")
    dates = set()
    dirs = [sconf["df_dir"]]
    avl = (spec.get("availability") or {}).get("avl_dir")
    if avl and os.path.exists(os.path.join(avl, uid + "_dfs.pkl")):
        dirs.append(avl)
    from naturenet.environment.sample_env_inputs import _fix_dates
    for d in dirs:
        with open(os.path.join(d, uid + "_dfs.pkl"), "rb") as f:
            dfs = pd.read_pickle(f)
        for segs in dfs.values():
            for s in segs:
                if len(s):
                    dates.update(pd.to_datetime(_fix_dates(s)).dropna().normalize())
    sd = {pd.Timestamp(d).normalize(): d for d in days}
    todo = sorted(d for d in sd if d in dates)
    ci = [chans.index(c) for c in use]

    def work(day):
        blk = reader(cal[sd[day]], ci, 0, H, 0, W, fill)
        r = dict(date=day.strftime("%Y-%m-%d"))
        for j, c in enumerate(use):
            v = blk[j][np.isfinite(blk[j])]
            r[f"{c}_n"] = int(v.size)
            r[f"{c}_sd"] = float(np.std(v)) if v.size else np.nan
            for p in q[c]:
                r[f"{c}_p{p:g}"] = float(np.percentile(v, p)) if v.size else np.nan
        return r

    with ThreadPoolExecutor(max_workers=max(1, int(st.get("workers", 2)))) as ex:
        rows = list(ex.map(work, todo))
    out = st.get("csv") or os.path.join(spec["out_dir"], "scene_feature_stats.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    atomic_csv(pd.DataFrame(rows), out)
    print(f"scene_stats: {len(rows)} scene days, channels {use} -> {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True, help="habitat_tests spec (or a file with the same keys)")
    ap.add_argument("--part", choices=("lagged", "scene"), required=True)
    ap.add_argument("--avl", action="store_true", help="lagged: the available points (availability avl_dir)")
    a = ap.parse_args(argv)
    spec = read_yaml(abs_cfg(a.yaml))
    spec = spec.get("habitat_tests", spec)
    if a.part == "lagged":
        write_lagged(spec, a.avl)
    else:
        scene_stats(spec)


if __name__ == "__main__":
    main()
