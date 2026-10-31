import argparse
import glob
import json
import os
import pickle
import re
import warnings
from datetime import datetime, timedelta
from pprint import pprint

import cv2
import numpy as np
import pandas as pd
import zarr
from osgeo import gdal

from sit_fuse.utils import read_yaml

AUG_RE = re.compile(r"__aug\d+")


# ---------------------------------------------------------------------------
# Scene discovery / loading
# ---------------------------------------------------------------------------

def _scene_file(yml_conf, i):
    """
    Exactly one file for day index i.

    Preferred: ``scene_file_template`` with an ``{i}`` placeholder (exact name).
    Fallback: glob_start + i + glob_end, excluding matches where i is directly preceded or followed by
    another digit (so "t1*" no longer matches t10, t11, ...). Raises unless exactly one file remains.
    """
    tmpl = yml_conf.get("scene_file_template")
    if tmpl:
        fname = tmpl.format(i=i)
        if not os.path.exists(fname):
            raise FileNotFoundError(fname)
        return fname
    gs, ge, si = yml_conf["glob_start"], yml_conf["glob_end"], str(i)
    cands = set(glob.glob(gs + si + ge))
    cands -= set(glob.glob(gs + si + "[0-9]" + ge))
    cands -= set(glob.glob(gs + "[0-9]" + si + ge))
    cands = sorted(cands)
    if len(cands) != 1:
        raise ValueError(f"Day {i}: expected exactly one scene file for {gs + si + ge}, found {cands}")
    return cands[0]


def _load_scene(fname):
    if fname.endswith(".tif") or fname.endswith(".tiff") or ".tif" in os.path.basename(fname):
        ds = gdal.Open(fname)
        return ds.ReadAsArray(), ds.GetGeoTransform()
    if "zarr" in fname:
        return zarr.load(fname), None
    raise ValueError(f"Unsupported scene format: {fname}")


def _scene_times(yml_conf):
    """
    Date of each day index. Uses ``scene_dates_file`` (JSON {"i": "ISO date"}, written by
    preprocess_copernicus.py) when given; otherwise assumes start_time + i days.
    """
    n_days = yml_conf["n_days"]
    dates_file = yml_conf.get("scene_dates_file")
    if dates_file:
        with open(dates_file) as f:
            d = json.load(f)
        missing = [i for i in range(n_days) if str(i) not in d]
        if missing:
            raise ValueError(f"{dates_file} is missing day indices {missing[:10]}")
        return [_naive_day(d[str(i)]) for i in range(n_days)]
    tme = _naive_day(yml_conf["start_time"])
    return [tme + pd.Timedelta(days=i) for i in range(n_days)]


def _naive_day(x):
    """Any date / ISO string (tz-aware or not) -> tz-naive UTC midnight, the key type used for matching.
    (A tz-aware key, e.g. from "2023-11-01T00:00:00+00:00" in scene_dates_file, never equals the tz-naive
    fix dates, so every fix was reported as outside the scene dates.)"""
    ts = pd.Timestamp(x)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def coverage_report(movement_dfs, times, nearest_days=0, strict=True):
    """
    Scene-date coverage per animal (original keys only): fixes, date range, fixes inside the scene
    calendar (+- nearest_days). Raises if no fix of any animal falls inside it (wrong start_time / n_days /
    scene_dates_file), instead of silently writing all-None env maps.
    """
    t0, t1 = min(times), max(times)
    lo, hi = t0 - pd.Timedelta(days=nearest_days), t1 + pd.Timedelta(days=nearest_days)
    tot_in, tot = 0, 0
    print(f"Scene calendar: {t0.date()} .. {t1.date()} ({len(times)} days, nearest_scene_days={nearest_days})")
    for uid in sorted(movement_dfs):
        if AUG_RE.search(uid) or not len(movement_dfs[uid]):
            continue
        d = pd.concat([_fix_dates(s) for s in movement_dfs[uid]])
        n_in = int(((d >= lo) & (d <= hi)).sum())
        tot_in, tot = tot_in + n_in, tot + len(d)
        flag = "" if n_in else "   <- no overlap"
        print(f"  {uid}: {len(d)} fixes, {d.min().date()} .. {d.max().date()}, {n_in} in calendar{flag}")
    print(f"  total: {tot_in}/{tot} fixes inside the scene calendar")
    if strict and tot and tot_in == 0:
        raise ValueError("No fixes fall inside the scene calendar. Check start_time / n_days / scene_dates_file "
                         "against the track dates above.")
    return tot_in, tot


def _extent(gt, shape_hw):
    h, w = shape_hw
    return np.array([gt[0], gt[3], gt[0] + gt[1] * w, gt[3] + gt[5] * h])


def _check_geotransform(mask_gt, mask_shape, scene_gt, scene_shape, tol_deg=1e-3):
    if mask_gt is None or scene_gt is None:
        return
    em, es = _extent(mask_gt, mask_shape), _extent(scene_gt, scene_shape)
    if not np.allclose(em, es, atol=tol_deg):
        warnings.warn(f"Water mask extent {em.tolist()} differs from scene extent {es.tolist()}; "
                      "resizing by shape will misalign land/water.")
    if np.sign(mask_gt[5]) != np.sign(scene_gt[5]):
        warnings.warn("Water mask and scene have opposite latitude orientation (north-up vs south-up).")


def _land_fill(yml_conf):
    """
    Value written over land. -1 (default) for cluster-id scenes (ids are >= 0, -1 = invalid).
    Use land_fill: nan for raw / standardised feature scenes, where -1 is a valid value.
    """
    v = yml_conf.get("land_fill", -1)
    return np.nan if (v is None or (isinstance(v, str) and v.lower() == "nan")) else float(v)


def _load_scenes(yml_conf, apply_mask=True):
    fill = _land_fill(yml_conf)
    mask_ds = gdal.Open(yml_conf["mask_path"])
    initial_water_mask = mask_ds.ReadAsArray()
    land_value = yml_conf["land_value"]
    resized_water_mask = None
    land = None

    scenes = []
    for i in range(yml_conf["n_days"]):
        fname = _scene_file(yml_conf, i)
        dat, gt = _load_scene(fname)
        if apply_mask:
            hw = dat.shape[-2:]
            if resized_water_mask is None:
                print("SHAPING WATER MASK", dat.shape)
                _check_geotransform(mask_ds.GetGeoTransform(), initial_water_mask.shape, gt, hw)
                # Nearest neighbour keeps the mask categorical (cubic created non-integer coast values).
                resized_water_mask = cv2.resize(initial_water_mask.astype(np.float32), (hw[1], hw[0]),
                                                interpolation=cv2.INTER_NEAREST)
                land = np.rint(resized_water_mask) == land_value
            if np.isnan(fill) and not np.issubdtype(dat.dtype, np.floating):
                dat = dat.astype(np.float32)
            elif not np.issubdtype(dat.dtype, np.floating) and not np.issubdtype(dat.dtype, np.signedinteger):
                dat = dat.astype(np.int32)
            if dat.ndim > 2:
                dat[:, land] = fill
            else:
                dat[land] = fill
        print("SCENE", i, "LOADED", fname)
        scenes.append(dat)
    return scenes


# ---------------------------------------------------------------------------
# Track -> scene matching
# ---------------------------------------------------------------------------

def _fix_dates(movement_df):
    ts = pd.to_datetime(movement_df["timestamp"], utc=True, format="mixed", errors="coerce")
    return ts.dt.tz_convert(None).dt.normalize()


def _segments_match(segs_a, segs_b):
    if len(segs_a) != len(segs_b):
        return False
    for a, b in zip(segs_a, segs_b):
        if len(a) != len(b):
            return False
        if not np.array_equal(pd.to_datetime(a["timestamp"], utc=True, format="mixed").to_numpy(),
                              pd.to_datetime(b["timestamp"], utc=True, format="mixed").to_numpy()):
            return False
    return True


def _match_track(segments, scenes, date_to_idx, nearest_days, sorted_dates):
    """Per segment, list (len = n fixes) of the scene for that fix's date, or None if out of range."""
    out, n_missing = [], 0
    for seg in segments:
        dates = _fix_dates(seg)
        maps = []
        for dt in dates:
            idx = date_to_idx.get(dt)
            if idx is None and nearest_days > 0 and len(sorted_dates):
                j = np.searchsorted(sorted_dates, dt)
                best = None
                for jj in (j - 1, j):
                    if 0 <= jj < len(sorted_dates):
                        gap = abs((sorted_dates[jj] - dt).days)
                        if gap <= nearest_days and (best is None or gap < best[0]):
                            best = (gap, date_to_idx[sorted_dates[jj]])
                idx = None if best is None else best[1]
            if idx is None:
                n_missing += 1
                maps.append(None)
            else:
                maps.append(scenes[idx])
        out.append(maps)
    return out, n_missing


def _pkl_path(out_dir, output_uid, uid):
    return os.path.join(out_dir, output_uid + uid + ".pkl")


def gen_scene_list(yml_conf):
    df_uid = yml_conf["df_uid"]
    df_dir = yml_conf["df_dir"]
    out_dir = yml_conf["out_dir"]
    output_uid = yml_conf["output_uid"]  # "final_env_maps_"
    nearest_days = int(yml_conf.get("nearest_scene_days", 0))

    times = _scene_times(yml_conf)
    with open(os.path.join(df_dir, df_uid + "_dfs.pkl"), "rb") as f:
        coverage_report(pd.read_pickle(f), times, nearest_days, strict=yml_conf.get("strict_scene_coverage", True))
    scenes = _load_scenes(yml_conf)
    date_to_idx = {t: i for i, t in enumerate(times)}
    sorted_dates = sorted(date_to_idx)

    with open(os.path.join(df_dir, df_uid + "_dfs.pkl"), "rb") as f:
        movement_dfs = pd.read_pickle(f)

    for uid in movement_dfs:
        pkl_file = _pkl_path(out_dir, output_uid, uid)
        if AUG_RE.search(uid):
            orig = AUG_RE.sub("", uid)
            # Augmented copies reuse the original's scenes only if they segment identically
            # (same timestamps per segment). Otherwise they are processed on their own.
            if orig in movement_dfs and _segments_match(movement_dfs[uid], movement_dfs[orig]):
                if os.path.lexists(pkl_file):
                    os.remove(pkl_file)
                os.symlink(_pkl_path(out_dir, output_uid, orig), pkl_file)
                continue
            print(f"{uid}: segmentation differs from {orig}; processing separately")

        print("Matching movement tracks to scenes for track", uid)
        scenes_per_uid, n_missing = _match_track(movement_dfs[uid], scenes, date_to_idx, nearest_days, sorted_dates)
        n_fix = sum(len(s) for s in movement_dfs[uid])
        if n_missing:
            print(f"WARNING: {uid}: {n_missing}/{n_fix} fixes fall outside the scene dates; env map set to None")

        if os.path.islink(pkl_file):
            os.remove(pkl_file)
        with open(pkl_file, "wb") as f:
            pickle.dump({uid: scenes_per_uid}, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Cluster relabelling
# ---------------------------------------------------------------------------

def _quantise(arr, scale):
    arr = np.asarray(arr, dtype=np.float64)
    q = np.full(arr.shape, -1, dtype=np.int64)
    valid = np.isfinite(arr) & (arr > 0.0)
    q[valid] = np.rint(arr[valid] * scale).astype(np.int64)
    q[valid & (q <= 0)] = -1
    return q


def build_class_mapper(scenes, scale=1000):
    """Valid (> 0) quantised cluster ids -> 0..K-1. Invalid (<= 0, land, nodata) -> -1."""
    lst = np.unique(np.concatenate([np.unique(_quantise(s, scale)) for s in scenes]))
    keys = lst[lst > 0]
    return keys


def remap_clusters(arr, keys, scale=1000):
    """Vectorised remap (no chained in-place relabelling). Returns int16 with -1 for invalid."""
    q = _quantise(arr, scale)
    out = np.full(q.shape, -1, dtype=np.int16)
    valid = q > 0
    idx = np.searchsorted(keys, q[valid])
    idx_c = np.clip(idx, 0, len(keys) - 1)
    unknown = keys[idx_c] != q[valid]
    if np.any(unknown):
        raise ValueError(f"{int(unknown.sum())} cluster values are not in the mapper, e.g. {q[valid][unknown][:5]}")
    out[valid] = idx_c.astype(np.int16)
    return out


def compress_clusters(yml_conf):
    df_uid = yml_conf["df_uid"]
    df_dir = yml_conf["df_dir"]
    out_dir = yml_conf["out_dir"]
    output_uid = yml_conf["output_uid"]
    scale = yml_conf.get("cluster_scale", 1000)

    scenes = _load_scenes(yml_conf, apply_mask=False)
    keys = build_class_mapper(scenes, scale)
    del scenes
    mapper = {float(k): i for i, k in enumerate(keys)}
    mapper[-1.0] = -1
    print("MAPPER: K =", len(keys), "valid clusters (set n_clusters >= K; -1 = invalid)")
    pprint(mapper)
    with open(os.path.join(out_dir, output_uid + "_class_mapper.pkl"), "wb") as f:
        pickle.dump(mapper, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(df_dir, df_uid + "_dfs.pkl"), "rb") as f:
        movement_dfs = pd.read_pickle(f)

    for key in movement_dfs.keys():
        pkl_file = _pkl_path(out_dir, output_uid, key)
        if os.path.islink(pkl_file):  # aliased augmented copy; original is processed
            continue
        with open(pkl_file, "rb") as f:
            abstract_grid = pickle.load(f)

        cache = {}  # scenes are shared across fixes: remap each unique array once and keep sharing
        new_segs = []
        for seg in abstract_grid[key]:
            new_maps = []
            for m in seg:
                if m is None:
                    new_maps.append(None)
                    continue
                if id(m) not in cache:
                    cache[id(m)] = (m, remap_clusters(m, keys, scale))
                new_maps.append(cache[id(m)][1])
            new_segs.append(new_maps)
        abstract_grid[key] = new_segs

        with open(pkl_file, "wb") as f:
            pickle.dump(abstract_grid, f, protocol=pickle.HIGHEST_PROTOCOL)


def merge_compressed_maps(yml_conf):
    out_dir = yml_conf["out_dir"]
    output_uid = yml_conf["output_uid"]
    df_uid = yml_conf["df_uid"]
    out_fle = os.path.join(out_dir, output_uid + df_uid + ".pkl")

    with open(os.path.join(yml_conf["df_dir"], df_uid + "_dfs.pkl"), "rb") as f:
        movement_dfs = pd.read_pickle(f)

    final_dict, aliases, missing = {}, [], []
    # Iterate the movement keys explicitly (the old glob also picked up the class mapper and old merges).
    for key in movement_dfs.keys():
        pkl_file = _pkl_path(out_dir, output_uid, key)
        if not os.path.exists(pkl_file):
            missing.append(key)
            continue
        if os.path.islink(pkl_file):
            aliases.append(key)
            continue
        with open(pkl_file, "rb") as f:
            final_dict[key] = pickle.load(f)[key]

    for key in aliases:
        orig = AUG_RE.sub("", key)
        if orig in final_dict:
            final_dict[key] = final_dict[orig]  # same object; pickled once
        else:
            missing.append(key)
    if missing:
        print(f"WARNING: no env maps for {len(missing)} keys: {missing[:10]}")

    with open(out_fle, "wb") as f:
        pickle.dump(final_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    yml_conf = read_yaml(args.yaml)

    gen_scene_list(yml_conf)

    if yml_conf["sit_fuse_output"]:
        compress_clusters(yml_conf)

    merge_compressed_maps(yml_conf)
