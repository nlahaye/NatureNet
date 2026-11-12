"""
Raw environmental inputs at each fix, read straight from the cmems_env scene zarrs.

    python -m naturenet.environment.sample_env_inputs -y config/v2/environment/sharks_sample_inputs_v2.yaml

Replaces env_data_format (raw scenes) + movement_preprocess_pipeline gather_input for the v2 scenes: the same
output, <df_dir>/<df_run_uid>_<key>_states_input.pkl = list (per segment) of float32 (len(segment), C), aligned
with the abstract states (len T) and actions (len T-1). NaN = no scene for that day / off the env grid / land or
nodata (scene value == fill_value). Column j = input_channels[j] (= input_keys in the swirl / hiql / iavi
configs). No per-fix scene copies are pickled: each scene day is read once (selected channels, bounding box of
that day's fixes only), so memory stays at one scene per worker plus the (n_fixes, C) outputs.

Config keys
  df_dir, df_run_uid          movement run (reads <df_run_uid>_dfs.pkl and the per-key / shared _grid.pkl)
  scene_index_csv             cmems_env <tag>_scene_index.csv (t, date, path, complete)
  complete_scenes_only        True (default): skip rows with complete == False
  scene_window                optional {start, end} (inclusive dates)
  env_meta_json               cmems_env <tag>_env_meta.json (channel order, grid bounds / shape, fill_value)
  input_channels              list of channel names (null = every channel, in scene order)
  nearest_scene_days          0 (default) = same-day scene only; as env_data_format nearest_scene_days
  uids                        optional subset of track keys (default every key incl. __aug copies)
  workers                     threads for scene reads (default 2)
  channels_out                optional JSON path listing the written column names
                              (default <df_dir>/<df_run_uid>_states_input_channels.json)
"""
import argparse
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml

from naturenet.environment.grid_utils import (grid_from_shape, in_grid_bounds, lat_lon_to_grid_vec,
                                              _fix_swapped_lat_lon)


def _naive_day(x):
    ts = pd.Timestamp(x)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _is_true(v):
    return v is True or str(v).strip().lower() in ("true", "1", "yes")


def read_calendar(conf):
    """{date: path} of the usable scenes."""
    df = pd.read_csv(conf["scene_index_csv"])
    if conf.get("complete_scenes_only", True) and "complete" in df.columns:
        df = df[df["complete"].map(_is_true)]
    w = conf.get("scene_window") or {}
    days = [_naive_day(d) for d in df["date"]]
    cal = {d: p for d, p in zip(days, df["path"])
           if (w.get("start") is None or d >= _naive_day(w["start"]))
           and (w.get("end") is None or d <= _naive_day(w["end"]))}
    if not cal:
        raise ValueError(f"no usable scenes in {conf['scene_index_csv']}")
    return cal


def match_days(dates, sorted_days, nearest_days):
    """Index into sorted_days per fix date (or -1)."""
    sd = np.asarray(sorted_days, dtype="datetime64[ns]")
    d = np.asarray(dates, dtype="datetime64[ns]")
    j = np.searchsorted(sd, d)
    out = np.full(len(d), -1, dtype=np.int64)
    best = np.full(len(d), np.inf)
    for jj in (j - 1, j):
        ok = (jj >= 0) & (jj < len(sd))
        gap = np.full(len(d), np.inf)
        gap[ok] = np.abs((sd[jj[ok]] - d[ok]) / np.timedelta64(1, "D"))
        take = ok & (gap <= nearest_days) & (gap < best)
        out[take], best[take] = jj[take], gap[take]
    return out


def _fix_dates(seg):
    ts = pd.to_datetime(seg["timestamp"], utc=True, format="mixed", errors="coerce")
    return ts.dt.tz_convert(None).dt.normalize().to_numpy()


def _load_grid(df_dir, df_uid, key):
    p = os.path.join(df_dir, df_uid + "_" + key + "_grid.pkl")
    if not os.path.exists(p):
        p = os.path.join(df_dir, df_uid + "_grid.pkl")
    with open(p, "rb") as f:
        return pickle.load(f)


def _read_points(path, chan, rows, cols, fill):
    """(len(rows), C) float32 at (rows, cols) of one scene; reads only the bounding box of the points."""
    import zarr
    arr = zarr.open_array(path, mode="r")
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    ch = slice(None) if chan is None else chan
    block = np.asarray(arr.oindex[ch, r0:r1, c0:c1], dtype=np.float32)
    v = block[:, rows - r0, cols - c0].T.copy()
    v[v == np.float32(fill)] = np.nan
    return v


def sample(conf):
    df_dir, df_uid = conf["df_dir"], conf["df_run_uid"]
    with open(conf["env_meta_json"]) as f:
        meta = json.load(f)
    chans = list(meta["channels"])
    names = list(conf.get("input_channels") or chans)
    bad = [n for n in names if n not in chans]
    if bad:
        raise ValueError(f"input_channels not in {conf['env_meta_json']}: {bad}")
    chan = None if names == chans else [chans.index(n) for n in names]
    g = meta["grid"]
    if not g.get("north_up", True):
        raise ValueError("sample_env_inputs expects north-up scenes")
    H, W = g["shape"]
    fill = float(meta.get("fill_value", -999999.0))
    cal = read_calendar(conf)
    sdays = sorted(cal)
    near = int(conf.get("nearest_scene_days", 0))
    uids = conf.get("uids")
    uids = None if uids is None else {str(u) for u in (uids if isinstance(uids, (list, tuple)) else [uids])}

    with open(os.path.join(df_dir, df_uid + "_dfs.pkl"), "rb") as f:
        movement_dfs = pd.read_pickle(f)
    keys = [k for k in movement_dfs if uids is None or k in uids]

    # Pass 1: (key, seg, fix) -> (day index, row, col)
    jobs = {}           # day index -> list of (key, seg, positions, rows, cols)
    outs = {}
    n_tot = n_hit = 0
    for key in keys:
        grid = _load_grid(df_dir, df_uid, key)
        env_grid = grid_from_shape(grid, H, W, lon_bounds=g["lon_bounds"], lat_bounds=g["lat_bounds"])
        outs[key] = []
        for si, seg in enumerate(movement_dfs[key]):
            n = len(seg)
            outs[key].append(np.full((n, len(names)), np.nan, dtype=np.float32))
            if not n:
                continue
            lat, lon = _fix_swapped_lat_lon(grid, seg["latitude"].to_numpy(), seg["longitude"].to_numpy())
            ok = np.isfinite(lat) & np.isfinite(lon)
            ok &= in_grid_bounds(env_grid, np.where(ok, lat, 0.0), np.where(ok, lon, 0.0))
            di = match_days(_fix_dates(seg), sdays, near)
            ok &= di >= 0
            n_tot += n
            if not ok.any():
                continue
            rows, cols = lat_lon_to_grid_vec(env_grid, lat[ok], lon[ok])
            rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
            pos = np.flatnonzero(ok)
            for d in np.unique(di[ok]):
                m = di[ok] == d
                jobs.setdefault(int(d), []).append((key, si, pos[m], rows[m], cols[m]))
            n_hit += int(ok.sum())
    print(f"{len(keys)} keys, {n_tot} fixes, {n_hit} on a scene day and inside the env grid; "
          f"{len(jobs)}/{len(sdays)} scene days read, {len(names)} channels")

    # Pass 2: one read per scene day
    def work(d):
        items = jobs[d]
        rows = np.concatenate([it[3] for it in items])
        cols = np.concatenate([it[4] for it in items])
        return d, _read_points(cal[sdays[d]], chan, rows, cols, fill)

    with ThreadPoolExecutor(max_workers=int(conf.get("workers", 2))) as ex:
        for d, vals in ex.map(work, sorted(jobs)):
            o = 0
            for key, si, pos, rows, _ in jobs[d]:
                outs[key][si][pos] = vals[o:o + len(pos)]
                o += len(pos)

    for key in keys:
        out = os.path.join(df_dir, df_uid + "_" + str(key) + "_states_input.pkl")
        with open(out, "wb") as f:
            pickle.dump(outs[key], f, protocol=pickle.HIGHEST_PROTOCOL)
    cpath = conf.get("channels_out") or os.path.join(df_dir, df_uid + "_states_input_channels.json")
    with open(cpath, "w") as f:
        json.dump(names, f, indent=1)
    print("wrote", len(keys), "states_input files and", cpath)
    return outs, names


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("-y", "--yaml", required=True)
    a = p.parse_args(argv)
    with open(a.yaml) as f:
        sample(yaml.safe_load(f))


if __name__ == "__main__":
    main()
