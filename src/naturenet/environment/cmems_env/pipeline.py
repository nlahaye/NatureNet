"""Copernicus Marine -> SIT-FUSE environment scenes.

    python -m naturenet.environment.cmems_env.pipeline all      --config cmems_env_sharks.yaml
    python -m naturenet.environment.cmems_env.pipeline discover --config ...   # catalogue CSV
    python -m naturenet.environment.cmems_env.pipeline resolve  --config ...   # channel list only
    python -m naturenet.environment.cmems_env.pipeline static   --config ...   # mask/bathy/coast/ships
    python -m naturenet.environment.cmems_env.pipeline pull     --config ...   # resumable cache
    python -m naturenet.environment.cmems_env.pipeline assemble --config ...   # fill/features/scenes

Stages
  static   water mask (relaxed coast), signed distance to coast, bathymetry (+slope,
           rugosity, isobath distance), ship density -> {work}/static.zarr
  pull     lazy ARCO reads of every source for the needed days, daily means, regrid to
           the target grid in one matmul per block, bit-rounded zstd cache per source
           ({work}/cache/{source}.zarr, (T, V, H, W)); resumable via a `done` flag array
  assemble short-gap temporal fill -> rolling/anomaly features -> push-pull spatial fill
           -> derived features -> static layers -> channel order -> water mask (fill
           value) -> one (C, H, W) float32 zarr per day, plus metadata, scene index,
           QA, per-channel stats and a SIT-FUSE data block.
"""
import argparse
import fnmatch
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

from . import store
from .features import compute_features, default_name, required_inputs
from .fill import missing_fraction, pushpull_fill, temporal_fill
from .grid import Grid, Regridder
from .sources import Source, discover, open_lazy
from .static import build_bathymetry, build_ship_density, build_water_mask

log = logging.getLogger("cmems_env")
ROLL_OPS = ("rolling_mean", "anomaly")


# ------------------------------------------------------------------ config / days
def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("work_dir", os.path.join(cfg["out_dir"], "work"))
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["work_dir"], "cache"), exist_ok=True)
    return cfg


def make_grid(cfg):
    b = cfg["grid"]["bbox"]
    return Grid(b[0], b[1], b[2], b[3], float(cfg["grid"]["res"]))


def track_days(cfg):
    """Unique UTC days with fixes inside the bbox, from the track CSVs."""
    b = cfg["grid"]["bbox"]
    days = []
    for tr in cfg["time"].get("tracks", []):
        cols = [tr.get("time_col", "timestamp"), tr.get("lon_col", "longitude"), tr.get("lat_col", "latitude")]
        df = pd.read_csv(tr["path"], usecols=cols)
        t = pd.to_datetime(df[cols[0]], utc=True, errors="coerce")
        ok = t.notna() & df[cols[1]].between(b[0], b[1]) & df[cols[2]].between(b[2], b[3])
        days.append(pd.DatetimeIndex(t[ok].dt.tz_localize(None).dt.floor("D").unique()))
    if not days:
        return pd.DatetimeIndex([])
    return days[0].append(days[1:]).unique().sort_values()


def plan_days(cfg):
    """(write_days, pull_days, halo dict). Halos: temporal-fill gap and rolling look-back."""
    tc = cfg["time"]
    pad = int(tc.get("pad_days", 1))
    tdays = track_days(cfg) if tc.get("tracks") else pd.DatetimeIndex([])
    start = pd.Timestamp(tc["start"]) if tc.get("start", "auto") != "auto" else tdays.min() - pd.Timedelta(days=pad)
    end = pd.Timestamp(tc["end"]) if tc.get("end", "auto") != "auto" else tdays.max() + pd.Timedelta(days=pad)
    if tc.get("write_days", "track") == "track" and len(tdays):
        w = set()
        for d in tdays:
            w.update(pd.date_range(d - pd.Timedelta(days=pad), d + pd.Timedelta(days=pad)))
        write = pd.DatetimeIndex(sorted(w))
        write = write[(write >= start) & (write <= end)]
    else:
        write = pd.date_range(start, end, freq="D")
    gap = int(cfg.get("fill", {}).get("max_temporal_gap", 3))
    roll = max([int(s.get("days", 1)) - 1 for s in cfg.get("features", []) if s["op"] in ROLL_OPS] + [0])
    halo = dict(gap=gap, roll=roll)
    return write, expand_days(write, gap + roll, gap), halo, start


def expand_days(write, back, fwd):
    p = set()
    for d in write:
        p.update(pd.date_range(d - pd.Timedelta(days=back), d + pd.Timedelta(days=fwd)))
    return pd.DatetimeIndex(sorted(p))


def source_roll(cfg, names):
    """Look-back needed by one source: only sources feeding rolling features pull the long halo."""
    return max([int(s.get("days", 1)) - 1 for s in cfg.get("features", [])
                if s["op"] in ROLL_OPS and s["var"] in names] + [0])


def _runs(idx_days, max_len):
    """Split a sorted DatetimeIndex into contiguous runs of at most max_len days -> (i0, i1)."""
    if len(idx_days) == 0:
        return []
    br = np.where(np.diff(idx_days.values).astype("timedelta64[D]").astype(int) != 1)[0] + 1
    out = []
    for a, b in zip(np.r_[0, br], np.r_[br, len(idx_days)]):
        for s in range(a, b, max_len):
            out.append((s, min(s + max_len, b)))
    return out


def make_sources(cfg, opener=open_lazy):
    margin = float(cfg.get("pull", {}).get("margin_deg", 0.5))
    return [Source(s, cfg["grid"]["bbox"], margin, opener=opener) for s in cfg["sources"] if s.get("enabled", True)]


# ------------------------------------------------------------------ resolve
def cmd_resolve(cfg, opener=open_lazy):
    srcs = make_sources(cfg, opener)
    res = {}
    for s in srcs:
        names = s.resolve()
        res[s.name] = dict(datasets=s.datasets, channels=names,
                           coverage={k: [str(v[0].date()), str(v[1].date())] for k, v in s.coverage.items()})
        log.info("[%s] %d channels: %s", s.name, len(names), ", ".join(names))
    with open(os.path.join(cfg["work_dir"], "resolved.json"), "w") as f:
        json.dump(res, f, indent=1)
    return srcs, res


# ------------------------------------------------------------------ static
def cmd_static(cfg, opener=open_lazy):
    grid = make_grid(cfg)
    layers = {}
    mcfg = cfg["mask"]
    m = build_water_mask(mcfg, grid)
    mask = m["mask"]
    layers["dist_coast_km"] = m["dist_coast_km"]
    layers["water_frac"] = m["water_frac"]
    bc = cfg.get("bathymetry", {})
    if bc.get("enabled", True):
        cm_depth = None
        if not bc.get("path"):
            dsid = bc.get("cmems_static", "cmems_mod_glo_phy_my_0.083deg_static")
            ds = opener(dsid, cfg["grid"]["bbox"], 0.5, variables=["deptho"])
            lon = ds["longitude"].values
            lat = ds["latitude"].values
            cm_depth = Regridder(lon, lat, grid, min_support=0.01)(np.asarray(ds["deptho"].values, np.float32))
        layers.update(build_bathymetry(bc, grid, cm_depth))
    if cfg.get("ship_density"):
        layers.update(build_ship_density(cfg["ship_density"], grid))
    keep = cfg.get("static_layers")  # optional explicit subset
    if keep:
        layers = {k: v for k, v in layers.items() if k in keep}
    names = list(layers)
    arr = np.stack([layers[k] for k in names]).astype(np.float32)
    p = os.path.join(cfg["work_dir"], "static.zarr")
    store.save_scene(p, arr, attrs=dict(channels=names))
    store.save_scene(os.path.join(cfg["work_dir"], "mask.zarr"), mask[None].astype(np.float32))
    if cfg.get("output", {}).get("write_mask_tif", True):
        _write_tif(os.path.join(cfg["out_dir"], f"{cfg['tag']}_water_mask.tif"), mask.astype(np.uint8), grid)
    log.info("static: %s; water cells %d / %d", names, int(mask.sum()), mask.size)
    return names, arr, mask


def _write_tif(path, a, grid):
    try:
        import rasterio
        from rasterio.transform import Affine
        with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                           dtype=a.dtype, crs="EPSG:4326", transform=Affine(*grid.transform()),
                           compress="deflate") as dst:
            dst.write(a, 1)
    except Exception as e:
        log.warning("mask GeoTIFF not written: %s", e)


def load_static(cfg):
    p = os.path.join(cfg["work_dir"], "static.zarr")
    a = store.open_array(p)
    mask = store.open_array(os.path.join(cfg["work_dir"], "mask.zarr"))[0] > 0.5
    return list(a.attrs["channels"]), np.asarray(a[...]), mask


# ------------------------------------------------------------------ pull
def cmd_pull(cfg, opener=open_lazy, srcs=None):
    import dask
    grid = make_grid(cfg)
    pc = cfg.get("pull", {})
    write, _, halo, _ = plan_days(cfg)
    if srcs is None:
        srcs, _ = cmd_resolve(cfg, opener)
    keepbits = pc.get("keepbits", 12)
    tasks, caches, regr = [], {}, {}
    for s in srcs:
        V = len(s.channels)
        pull = expand_days(write, halo["gap"] + source_roll(cfg, [c[2] for c in s.channels]), halo["gap"])
        cp = os.path.join(cfg["work_dir"], "cache", f"{s.name}.zarr")
        dp = os.path.join(cfg["work_dir"], "cache", f"{s.name}.done.zarr")
        attrs = dict(channels=[c[2] for c in s.channels], days=[d.strftime("%Y-%m-%d") for d in pull])
        reuse = False
        if os.path.exists(cp) and os.path.exists(dp):
            old = store.open_array(cp, mode="r+")
            reuse = list(old.attrs.get("channels", [])) == attrs["channels"] and list(old.attrs.get("days", [])) == attrs["days"]
        if reuse:
            arr, done = old, store.open_array(dp, mode="r+")
        else:
            arr = store.create_array(cp, (len(pull), V) + grid.shape, (1, V) + grid.shape, attrs=attrs,
                                     clevel=pc.get("clevel", 5))
            done = store.create_array(dp, (len(pull),), (1,), dtype="u1", fill_value=0)
        caches[s.name] = (arr, done, pull)
        dflags = np.asarray(done[...])
        pick = np.array(s.dataset_for_days(pull), dtype=object)
        for i0, i1 in _runs(pull, s.block_days):
            j = i0
            while j < i1:  # split further where the chosen dataset changes
                k = j
                while k < i1 and pick[k] == pick[j]:
                    k += 1
                if not dflags[j:k].all():
                    tasks.append((s, pick[j], j, k))
                j = k
        log.info("[%s] %d channels, %d pull days", s.name, V, len(pull))
    log.info("pull: %d blocks to fetch over %d sources", len(tasks), len(srcs))

    def run(task):
        s, dsid, i0, i1 = task
        arr, done, pull = caches[s.name]
        if dsid is None:  # no dataset covers these days -> stays NaN (flagged at assembly)
            done[i0:i1] = 2
            return s.name, i0, i1, "uncovered"
        data, lon, lat = s.read_days(dsid, pull[i0:i1])
        key = (s.name, dsid)
        if key not in regr:
            regr[key] = Regridder(lon, lat, grid, min_support=float(pc.get("min_support", 0.25)))
        out = regr[key](data)
        if s.cfg.get("lossless", False) is False:
            out = store.bitround(out, s.cfg.get("keepbits", keepbits))
        arr[i0:i1] = out
        done[i0:i1] = 1
        return s.name, i0, i1, dsid

    with dask.config.set(scheduler="threads", num_workers=int(pc.get("dask_threads", 4))):
        nw = int(pc.get("workers", 4))
        if nw <= 1:
            for t in tasks:
                log.info("pulled %s", run(t))
        else:
            with ThreadPoolExecutor(nw) as ex:
                futs = [ex.submit(run, t) for t in tasks]
                for f in as_completed(futs):
                    try:
                        log.info("pulled %s", f.result())
                    except Exception as e:  # keep going; rerun resumes failed blocks
                        log.error("block failed: %s", e)
    return caches


# ------------------------------------------------------------------ assemble
def channel_order(cfg, raw, feats, static):
    allc = raw + feats + static
    ch = cfg.get("channels", {}) or {}
    if ch.get("order"):
        names = []
        for pat in ch["order"]:
            names += [c for c in allc if fnmatch.fnmatch(c, pat) and c not in names]
    else:
        names = list(allc)
    drop = ch.get("drop", ["water_frac"])
    return [c for c in names if not any(fnmatch.fnmatch(c, p) for p in drop)]


def cmd_assemble(cfg):
    grid = make_grid(cfg)
    fc = cfg.get("fill", {})
    oc = cfg.get("output", {})
    fillv = float(oc.get("fill_value", -999999.0))
    write, _, halo, t0 = plan_days(cfg)
    snames, sarr, mask = load_static(cfg)
    caches = {}
    for s in cfg["sources"]:
        if not s.get("enabled", True):
            continue
        a = store.open_array(os.path.join(cfg["work_dir"], "cache", f"{s['name']}.zarr"))
        caches[s["name"]] = (a, list(a.attrs["channels"]), pd.DatetimeIndex(a.attrs["days"]))
    raw = [c for _, (_, ch, _) in caches.items() for c in ch]
    specs = cfg.get("features", [])
    avail = set(raw)
    feat_names, live_specs = [], []
    for sp in specs:
        if all(k in avail for k in required_inputs(sp)):
            n = default_name(sp)
            feat_names.append(n)
            live_specs.append(sp)
            avail.add(n)
        else:
            log.warning("feature %s skipped (inputs missing)", sp)
    order = channel_order(cfg, raw, feat_names, snames)
    roll_specs = [s for s in live_specs if s["op"] in ROLL_OPS]
    other_specs = [s for s in live_specs if s["op"] not in ROLL_OPS]
    needed_raw = set(order) | {k for s in live_specs for k in required_inputs(s)}
    prefix = oc.get("prefix", f"cop_env_{cfg['tag']}")
    sdir = oc.get("scene_dir", cfg["out_dir"])
    os.makedirs(sdir, exist_ok=True)
    gap, roll = halo["gap"], halo["roll"]
    blk = int(oc.get("assemble_block_days", 8))
    stats = {c: np.zeros(5) for c in order}  # n, sum, sumsq, min, max
    for c in order:
        stats[c][3], stats[c][4] = np.inf, -np.inf
    index_rows, qa_rows = [], []
    wmask = mask
    writer = ThreadPoolExecutor(int(oc.get("write_threads", 2)))
    pending = []
    # split write days into blocks spanning <= blk calendar days
    blocks, cur = [], []
    for d in write:
        if cur and (d - cur[0]).days >= blk:
            blocks.append(cur)
            cur = []
        cur.append(d)
    if cur:
        blocks.append(cur)
    for bdays in blocks:
        bdays = pd.DatetimeIndex(bdays)
        dense = pd.date_range(bdays[0] - pd.Timedelta(days=gap), bdays[-1] + pd.Timedelta(days=gap))
        wpos = dense.get_indexer(bdays)
        fields, incomplete = {}, np.zeros(len(bdays), bool)
        missing_ch = [[] for _ in bdays]
        for sname, (arr, chans, cache_days) in caches.items():
            sel = [k for k, c in enumerate(chans) if c in needed_raw]
            if not sel:
                continue
            x = _read_dense(arr, cache_days, dense, sel)
            for j, k in enumerate(sel):
                c = chans[k]
                pre = missing_fraction(x[wpos, j], wmask)
                for i, d in enumerate(bdays):
                    qa_rows.append((d.strftime("%Y-%m-%d"), c, float(pre[i])))
                y = temporal_fill(x[:, j], gap)
                for sp in roll_specs:
                    if sp["var"] == c:
                        ext = pd.date_range(dense[0] - pd.Timedelta(days=int(sp["days"]) - 1), dense[-1])
                        xe = _read_dense(arr, cache_days, ext, [k])[:, 0]
                        xe = temporal_fill(xe, gap)
                        r = _rolling_mean(xe, int(sp["days"]))[-len(dense):]
                        v = r if sp["op"] == "rolling_mean" else y - r
                        fields[default_name(sp)] = _fill_write(v[wpos], fc)
                yw = y[wpos]
                miss = missing_fraction(yw, wmask)
                for i in np.where(miss >= float(fc.get("max_missing_frac", 0.999)))[0]:
                    incomplete[i] = True
                    missing_ch[i].append(c)
                fields[c] = _fill_write(yw, fc)
            del x
        compute_features(fields, other_specs, grid)
        for k, n in enumerate(snames):
            fields[n] = np.broadcast_to(sarr[k], (len(bdays),) + grid.shape)
        for i, d in enumerate(bdays):
            t = int((d - t0).days)
            path = os.path.join(sdir, f"{prefix}_t{t}.zarr")
            ok = not incomplete[i]
            index_rows.append(dict(t=t, date=d.strftime("%Y-%m-%d"), path=path, complete=ok,
                                   missing_channels=";".join(missing_ch[i])))
            if not ok and not fc.get("allow_incomplete", False):
                continue
            scene = np.stack([fields[c][i] for c in order]).astype(np.float32)
            scene = np.where(np.isfinite(scene), scene, fillv)
            scene[:, ~wmask] = fillv
            for c_i, c in enumerate(order):
                v = scene[c_i][wmask]
                v = v[v != fillv].astype(np.float64)
                if v.size:
                    st = stats[c]
                    st[0] += v.size
                    st[1] += v.sum()
                    st[2] += (v * v).sum()
                    st[3] = min(st[3], v.min())
                    st[4] = max(st[4], v.max())
            pending.append(writer.submit(store.save_scene, path, scene,
                                         dict(channels=order, date=d.strftime("%Y-%m-%d"), t=t, fill_value=fillv),
                                         oc.get("clevel", 5)))
        pending = [p for p in pending if not p.done()]
        log.info("assembled %s .. %s", bdays[0].date(), bdays[-1].date())
    for p in pending:
        p.result()
    writer.shutdown()
    return _write_outputs(cfg, grid, order, raw, feat_names, snames, index_rows, qa_rows, stats, t0, fillv)


def _fill_write(y, fc):
    return pushpull_fill(y, smooth_iters=int(fc.get("smooth_iters", 4)))


def _rolling_mean(x, days):
    from .features import _rolling
    return _rolling(x, days)


def _read_dense(arr, cache_days, dense, sel):
    """(len(dense), len(sel), H, W) float32 from the cache; days not cached -> NaN."""
    pos = cache_days.get_indexer(dense)
    out = np.full((len(dense), len(sel)) + tuple(arr.shape[2:]), np.nan, np.float32)
    have = np.where(pos >= 0)[0]
    if len(have):
        # contiguous cache runs -> one slice read each
        for a, b in _runs(pd.DatetimeIndex(dense[have]), 10 ** 6):
            i0, i1 = pos[have[a]], pos[have[b - 1]] + 1
            blk = np.asarray(arr[i0:i1])
            out[have[a]:have[b - 1] + 1] = blk[:, sel]
    return out


def _write_outputs(cfg, grid, order, raw, feats, static, index_rows, qa_rows, stats, t0, fillv):
    tag, od = cfg["tag"], cfg["out_dir"]
    idx = pd.DataFrame(index_rows)
    idx.to_csv(os.path.join(od, f"{tag}_scene_index.csv"), index=False)
    pd.DataFrame(qa_rows, columns=["date", "channel", "missing_frac_prefill"]).to_csv(
        os.path.join(od, f"{tag}_qa_missing.csv"), index=False)
    srows = []
    for c in order:
        n, s, ss, mn, mx = stats[c]
        mu = s / n if n else np.nan
        sd = np.sqrt(max(ss / n - mu * mu, 0)) if n else np.nan
        srows.append(dict(channel=c, n=int(n), mean=mu, std=sd, min=mn if n else np.nan, max=mx if n else np.nan))
    pd.DataFrame(srows).to_csv(os.path.join(od, f"{tag}_channel_stats.csv"), index=False)
    resolved = {}
    rp = os.path.join(cfg["work_dir"], "resolved.json")
    if os.path.exists(rp):
        with open(rp) as f:
            resolved = json.load(f)
    meta = dict(tag=tag, channels=order, n_channels=len(order), raw_channels=raw, derived_channels=feats,
                static_channels=static, grid=grid.to_dict(), fill_value=fillv, time_start=str(t0.date()),
                t_definition="days since time_start", sources=resolved,
                features=cfg.get("features", []), mask=cfg.get("mask", {}))
    try:
        import copernicusmarine
        meta["copernicusmarine_version"] = copernicusmarine.__version__
    except Exception:
        pass
    with open(os.path.join(od, f"{tag}_env_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    ok = idx[idx["complete"]] if len(idx) else idx
    every = int(cfg.get("output", {}).get("train_every_days", 7))
    files = list(ok["path"]) if len(ok) else []
    tr = [p for p, t in zip(files, ok["t"] if len(ok) else []) if t % every == 0]
    block = dict(data=dict(number_channels=len(order), fill_value=fillv,
                           valid_min=float(cfg.get("output", {}).get("valid_min", -8000.0)),
                           valid_max=1.0e8, reader_type="zarr_to_numpy", chan_dim=0,
                           files_train=tr, files_test=files))
    with open(os.path.join(od, f"{tag}_sitfuse_data_block.yaml"), "w") as f:
        f.write(f"# Paste into the SIT-FUSE model config data: block. {len(order)} channels; order in {tag}_env_meta.json\n")
        yaml.safe_dump(block, f, sort_keys=False)
    log.info("wrote %d scenes (%d incomplete) with %d channels", int(idx["complete"].sum()) if len(idx) else 0,
             int((~idx["complete"]).sum()) if len(idx) else 0, len(order))
    return meta


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["discover", "resolve", "static", "pull", "assemble", "all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--log", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    if a.workers:
        cfg.setdefault("pull", {})["workers"] = a.workers
    if a.stage == "discover":
        w, p, _, _ = plan_days(cfg)
        out = os.path.join(cfg["out_dir"], f"{cfg['tag']}_cmems_catalogue.csv")
        df = discover(cfg["grid"]["bbox"], p.min(), p.max(), out_csv=out,
                      include_monthly=cfg.get("discover", {}).get("include_monthly", True))
        log.info("%d dataset/variable rows -> %s", len(df), out)
    elif a.stage == "resolve":
        cmd_resolve(cfg)
    elif a.stage == "static":
        cmd_static(cfg)
    elif a.stage == "pull":
        cmd_pull(cfg)
    elif a.stage == "assemble":
        cmd_assemble(cfg)
    else:
        cmd_static(cfg)
        cmd_pull(cfg)
        cmd_assemble(cfg)


if __name__ == "__main__":
    main()
