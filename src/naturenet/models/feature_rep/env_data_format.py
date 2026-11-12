"""
Match movement fixes to daily environment scenes (SIT-FUSE cluster maps or raw scenes) and write
<out_dir>/<output_uid><key>.pkl per track plus the merged <out_dir><output_uid><df_uid>.pkl.

    python -m naturenet.models.feature_rep.env_data_format -y config/v2/environment/sharks_env_format_clusters_v2.yaml

Scene calendar (no hardcoded dates):
  scene_index_csv: cmems_env <tag>_scene_index.csv (t, date, path, complete); gaps allowed, incomplete rows
                   skipped unless complete_scenes_only: False. Preferred for the v2 scenes.
  or scene_dates_file (JSON {"i": date}) + n_days, or start_time + n_days (legacy, contiguous days).
  scene_window: {start, end} optional inclusive restriction.
Scene files: scene_file_template with {i} = t (wildcards allowed, one match per t) or glob_start / glob_end.
  allow_missing_scenes: True drops calendar days without a file (e.g. no SIT-FUSE output) instead of failing.
Only the scenes some fix uses are loaded. Other keys: df_uid, df_dir, out_dir, output_uid, nearest_scene_days,
strict_scene_coverage, mask_path / land_value / land_fill, scene_fill_value (scene nodata -> land_fill),
input_channels + env_meta_json (raw-scene channel subset), store_paths (write file names, not arrays),
sit_fuse_output (relabel cluster ids and write <output_uid>_class_mapper.pkl), cluster_scale.

Cluster ids: id = rint(raster value * cluster_scale) (SIT-FUSE rasters store id / 1000). zero_label_valid: True keeps
id 0 (a real cluster; SIT-FUSE no data is negative); the legacy default False treats it as invalid like before.
Environment levels (sit_fuse_output: True; SIT-FUSE sit_fuse.postprocessing.cluster_hierarchy):
  cluster_level: fine      heir ids as before (default)
                 parent    first-stage cluster = fine // num_child (label_input: fine), or the no_heir rasters as
                           they are (label_input: parent, scene_file_template -> <scene>.no_heir.clust.data_*)
                 coarse    parent -> agglomerated cluster from hierarchy_json cut level_k
  num_child (5 = SIT-FUSE cluster.heir.num_classes), label_input (fine | parent), hierarchy_json, level_k,
  unmapped_parent (error | invalid). The class mapper keys are the level ids; <output_uid>_level.json records
  the level settings. Use one out_dir / output_uid per level.
"""
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

def _scene_file(yml_conf, i, required=True):
    """
    Exactly one file for day index i (None if missing and not required).

    Preferred: ``scene_file_template`` with an ``{i}`` placeholder. Shell wildcards in the formatted name are
    expanded (e.g. SIT-FUSE output names ".../cop_env_sharks_t{i}.zarr.clust.data_*clusters.no_geo.tif", where *
    is the per-run cluster count); more than one match raises.
    Fallback: glob_start + i + glob_end, excluding matches where i is directly preceded or followed by
    another digit (so "t1*" no longer matches t10, t11, ...).
    """
    tmpl = yml_conf.get("scene_file_template")
    if tmpl:
        fname = tmpl.format(i=i)
        if glob.has_magic(fname):
            hits = sorted(glob.glob(fname))
            if len(hits) > 1:
                raise ValueError(f"Day {i}: {fname} matches {len(hits)} files: {hits[:5]}")
            fname = hits[0] if hits else fname
        if not os.path.exists(fname):
            if required:
                raise FileNotFoundError(fname)
            return None
        return fname
    gs, ge, si = yml_conf["glob_start"], yml_conf["glob_end"], str(i)
    cands = set(glob.glob(gs + si + ge))
    cands -= set(glob.glob(gs + si + "[0-9]" + ge))
    cands -= set(glob.glob(gs + "[0-9]" + si + ge))
    cands = sorted(cands)
    if not cands and not required:
        return None
    if len(cands) != 1:
        raise ValueError(f"Day {i}: expected exactly one scene file for {gs + si + ge}, found {cands}")
    return cands[0]


def _channel_index(yml_conf):
    """Channel positions for input_channels (names, resolved with env_meta_json["channels"]), or None = all."""
    names = yml_conf.get("input_channels")
    if not names:
        return None
    meta = yml_conf.get("env_meta_json")
    if not meta:
        raise ValueError("input_channels needs env_meta_json (channel order of the scenes)")
    with open(meta) as f:
        chans = list(json.load(f)["channels"])
    bad = [n for n in names if n not in chans]
    if bad:
        raise ValueError(f"input_channels not in {meta}: {bad}")
    return [chans.index(n) for n in names]


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


def _is_true(v):
    return v is True or (isinstance(v, (int, np.integer, np.bool_)) and bool(v)) or \
        str(v).strip().lower() in ("true", "1", "yes")


def scene_calendar(yml_conf):
    """
    [(t, date)] of the scenes to use, t = the scene file index ({i} in scene_file_template).

    scene_index_csv (written by naturenet.environment.cmems_env: columns t, date, path, complete) gives the
    scene days directly, with gaps allowed; rows with complete == False are skipped unless
    complete_scenes_only: False. Otherwise scene_dates_file / start_time + n_days as before.
    Optional scene_window: {start: <date>, end: <date>} (inclusive) restricts either calendar.
    """
    csv = yml_conf.get("scene_index_csv")
    if csv:
        df = pd.read_csv(csv)
        if yml_conf.get("complete_scenes_only", True) and "complete" in df.columns:
            df = df[df["complete"].map(_is_true)]
        cal = [(int(t), _naive_day(d)) for t, d in zip(df["t"], df["date"])]
    else:
        cal = list(enumerate(_scene_times(yml_conf)))
    w = yml_conf.get("scene_window") or {}
    if w.get("start") is not None:
        cal = [c for c in cal if c[1] >= _naive_day(w["start"])]
    if w.get("end") is not None:
        cal = [c for c in cal if c[1] <= _naive_day(w["end"])]
    if not cal:
        raise ValueError("Empty scene calendar (check scene_index_csv / start_time / n_days / scene_window)")
    days = [d for _, d in cal]
    if len(set(days)) != len(days):
        raise ValueError("Scene calendar has more than one scene for the same day")
    return cal


def scene_files(yml_conf, cal=None):
    """{t: file} for the calendar. Missing files raise unless allow_missing_scenes: True (then they are
    dropped with a warning, e.g. scenes SIT-FUSE did not run inference on)."""
    cal = scene_calendar(yml_conf) if cal is None else cal
    allow = yml_conf.get("allow_missing_scenes", False)
    files, missing = {}, []
    for t, _ in cal:
        f = _scene_file(yml_conf, t, required=not allow)
        if f is None:
            missing.append(t)
        else:
            files[t] = f
    if missing:
        print(f"WARNING: {len(missing)}/{len(cal)} scene files missing (allow_missing_scenes), e.g. t={missing[:10]}")
    if not files:
        raise ValueError("No scene files found")
    return files


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


def _load_scenes(yml_conf, files, apply_mask=True):
    """{t: scene} for the given {t: file}, loaded one at a time (only the scenes the tracks use).
    store_paths: True keeps the file names instead of the arrays (no loading, no mask)."""
    if yml_conf.get("store_paths", False):
        return dict(files)
    fill = _land_fill(yml_conf)
    chan = _channel_index(yml_conf)
    land_value = yml_conf.get("land_value", 0)
    mask_ds, initial_water_mask = None, None
    if apply_mask and yml_conf.get("mask_path"):
        mask_ds = gdal.Open(yml_conf["mask_path"])
        initial_water_mask = mask_ds.ReadAsArray()
    land = None

    scenes = {}
    for t in sorted(files):
        fname = files[t]
        dat, gt = _load_scene(fname)
        if chan is not None:
            if dat.ndim != 3:
                raise ValueError(f"input_channels given but {fname} has shape {dat.shape}")
            dat = dat[chan]
        if mask_ds is not None:
            hw = dat.shape[-2:]
            if land is None:
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
        nodata = yml_conf.get("scene_fill_value")   # e.g. -999999 in cmems_env scenes -> land_fill
        if nodata is not None and np.issubdtype(dat.dtype, np.floating):
            dat[dat == float(nodata)] = fill
        print("SCENE", t, "LOADED", fname)
        scenes[t] = dat
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


def _match_track(segments, date_to_idx, nearest_days, sorted_dates):
    """Per segment, list (len = n fixes) of the scene index t for that fix's date, or None if out of range."""
    out, n_missing = [], 0
    for seg in segments:
        dates = _fix_dates(seg)
        idxs = []
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
            idxs.append(idx)
        out.append(idxs)
    return out, n_missing


def _pkl_path(out_dir, output_uid, uid):
    return os.path.join(out_dir, output_uid + uid + ".pkl")


def gen_scene_list(yml_conf):
    df_uid = yml_conf["df_uid"]
    df_dir = yml_conf["df_dir"]
    out_dir = yml_conf["out_dir"]
    output_uid = yml_conf["output_uid"]  # "final_env_maps_"
    if yml_conf.get("store_paths", False) and yml_conf.get("sit_fuse_output", False):
        raise ValueError("store_paths: True writes file names; cluster compression needs sit_fuse_output: False")
    nearest_days = int(yml_conf.get("nearest_scene_days", 0))

    files = scene_files(yml_conf)
    cal = [(t, d) for t, d in scene_calendar(yml_conf) if t in files]
    with open(os.path.join(df_dir, df_uid + "_dfs.pkl"), "rb") as f:
        movement_dfs = pd.read_pickle(f)
    coverage_report(movement_dfs, [d for _, d in cal], nearest_days,
                    strict=yml_conf.get("strict_scene_coverage", True))
    date_to_idx = {d: t for t, d in cal}
    sorted_dates = sorted(date_to_idx)

    # Match first, then load only the scenes some fix uses.
    matches = {uid: _match_track(movement_dfs[uid], date_to_idx, nearest_days, sorted_dates)
               for uid in movement_dfs}
    needed = {t for m, _ in matches.values() for seg in m for t in seg if t is not None}
    print(f"{len(needed)}/{len(cal)} scenes used by the tracks")
    scenes = _load_scenes(yml_conf, {t: files[t] for t in needed})

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
        idxs, n_missing = matches[uid]
        scenes_per_uid = [[None if t is None else scenes[t] for t in seg] for seg in idxs]
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

def _quantise(arr, scale, zero_valid=False):
    """Raster values -> int64 ids, -1 = invalid (NaN, negative, and 0 unless zero_valid)."""
    arr = np.asarray(arr, dtype=np.float64)
    q = np.full(arr.shape, -1, dtype=np.int64)
    valid = np.isfinite(arr) & ((arr >= 0.0) if zero_valid else (arr > 0.0))
    q[valid] = np.rint(arr[valid] * scale).astype(np.int64)
    q[valid & ((q < 0) if zero_valid else (q <= 0))] = -1
    return q


LEVELS = ("fine", "parent", "coarse")


def level_spec(yml_conf):
    """Level settings from the config (None = legacy fine ids). Loads the coarse map from hierarchy_json."""
    lv = yml_conf.get("cluster_level", "fine") or "fine"
    if lv not in LEVELS:
        raise ValueError(f"cluster_level must be one of {LEVELS}, got {lv}")
    spec = dict(level=lv, num_child=int(yml_conf.get("num_child", 5)), label_input=yml_conf.get("label_input", "fine"),
                scale=yml_conf.get("cluster_scale", 1000), zero_valid=bool(yml_conf.get("zero_label_valid", False)),
                unmapped=yml_conf.get("unmapped_parent", "error"), level_k=None, hierarchy_json=None, keys=None)
    if spec["label_input"] not in ("fine", "parent"):
        raise ValueError("label_input must be fine | parent")
    if lv == "fine" and spec["label_input"] != "fine":
        raise ValueError("cluster_level fine needs fine (heir) rasters (label_input: fine)")
    if lv == "coarse":
        spec["hierarchy_json"], spec["level_k"] = yml_conf["hierarchy_json"], int(yml_conf["level_k"])
        with open(spec["hierarchy_json"]) as f:
            h = json.load(f)
        cut = h["cuts"].get(str(spec["level_k"]))
        if cut is None:
            raise ValueError(f"{spec['hierarchy_json']}: no cut K = {spec['level_k']} (have {sorted(map(int, h['cuts']))})")
        for k in ("num_child", "label_scale"):
            if k in h and float(h[k]) != float(spec["num_child" if k == "num_child" else "scale"]):
                raise ValueError(f"hierarchy {k} = {h[k]} differs from this config")
        m = {int(p): int(c) for p, c in cut.items()}
        spec["keys"] = np.array(sorted(m), dtype=np.int64)
        spec["vals"] = np.array([m[k] for k in spec["keys"]], dtype=np.int64)
    return spec


def level_ids(arr, spec):
    """Raster -> int64 ids at the configured level, -1 = invalid."""
    q = _quantise(arr, spec["scale"], spec["zero_valid"])
    if spec["level"] == "fine":
        return q
    par = q if spec["label_input"] == "parent" else np.where(q >= 0, q // spec["num_child"], -1)
    if spec["level"] == "parent":
        return par
    out = np.full(par.shape, -1, dtype=np.int64)
    v = par >= 0
    keys = spec["keys"]
    j = np.clip(np.searchsorted(keys, par[v]), 0, len(keys) - 1)
    hit = keys[j] == par[v]
    if not hit.all() and spec["unmapped"] == "error":
        bad = np.unique(par[v][~hit])
        raise ValueError(f"{len(bad)} parent clusters not in {spec['hierarchy_json']}, e.g. {bad[:10].tolist()} "
                         "(rebuild the hierarchy over every scene, or unmapped_parent: invalid)")
    res = np.full(j.shape, -1, dtype=np.int64)
    res[hit] = spec["vals"][j[hit]]
    out[v] = res
    return out


def _spec(scale_or_spec):
    if isinstance(scale_or_spec, dict):
        return scale_or_spec
    return dict(level="fine", scale=scale_or_spec, zero_valid=False)


def build_class_mapper(scenes, scale=1000):
    """Valid quantised (level) ids -> 0..K-1. Invalid (land, nodata, unmapped) -> -1.
    scale: cluster_scale (legacy fine ids, id 0 invalid) or a level_spec dict."""
    spec = _spec(scale)
    lst = np.unique(np.concatenate([np.unique(level_ids(s, spec)) for s in scenes]))
    keys = lst[lst >= 0]
    return keys


def remap_clusters(arr, keys, scale=1000):
    """Vectorised remap (no chained in-place relabelling). Returns int16 with -1 for invalid."""
    q = level_ids(arr, _spec(scale))
    out = np.full(q.shape, -1, dtype=np.int16)
    valid = q >= 0
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
    scale = level_spec(yml_conf)

    # Mapper over every scene of the calendar (not only the tracked days), one scene in memory at a time.
    keys = np.zeros(0, dtype=np.int64)
    for t, fname in sorted(scene_files(yml_conf).items()):
        keys = np.union1d(keys, build_class_mapper([_load_scene(fname)[0]], scale))
    if len(keys) > np.iinfo(np.int16).max:
        raise ValueError(f"{len(keys)} clusters do not fit the int16 state maps")
    mapper = {float(k): i for i, k in enumerate(keys)}
    mapper[-1.0] = -1
    print(f"MAPPER ({scale['level']}): K =", len(keys), "valid clusters (set n_clusters >= K; -1 = invalid)")
    pprint(mapper)
    with open(os.path.join(out_dir, output_uid + "_class_mapper.pkl"), "wb") as f:
        pickle.dump(mapper, f, protocol=pickle.HIGHEST_PROTOCOL)
    info = {k: scale[k] for k in ("level", "num_child", "label_input", "scale", "zero_valid", "level_k",
                                  "hierarchy_json", "unmapped")}
    info.update(n_clusters=int(len(keys)), level_ids=[int(k) for k in keys])
    with open(os.path.join(out_dir, output_uid + "_level.json"), "w") as f:
        json.dump(info, f, indent=1)

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


def run(yml_conf):
    """Full step: match tracks to scenes, (sit_fuse_output) relabel cluster ids 0..K-1, merge per-key pickles."""
    if (yml_conf.get("cluster_level") or "fine") != "fine" and not yml_conf.get("sit_fuse_output", False):
        raise ValueError("cluster_level parent / coarse needs sit_fuse_output: True")
    level_spec(yml_conf)             # fail early on a bad level / hierarchy before matching tracks
    gen_scene_list(yml_conf)
    if yml_conf.get("sit_fuse_output", False) and not yml_conf.get("store_paths", False):
        compress_clusters(yml_conf)
    merge_compressed_maps(yml_conf)


def main(argv=None, defaults=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True, help="env_data_format YAML")
    args = parser.parse_args(argv)
    conf = dict(defaults or {})
    conf.update(read_yaml(args.yaml))
    run(conf)


if __name__ == "__main__":
    main()
