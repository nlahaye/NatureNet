"""
MBON / NOAA Seascapes (Kavanaugh et al. 2016) on the cmems_env scene grid, one class raster per scene.

    python -m naturenet.environment.seascapes download -y config/v2/environment/sharks_seascapes_v2.yaml
    python -m naturenet.environment.seascapes regrid   -y config/v2/environment/sharks_seascapes_v2.yaml
    python -m naturenet.environment.seascapes report   -y config/v2/environment/sharks_seascapes_v2.yaml
    python -m naturenet.environment.seascapes all      -y ...

download  ERDDAP griddap subsets (CLASS, P) of the scene bbox + margin, one netCDF per calendar year, over the scene
          index dates (+ lookback for the composite that covers the first day). Resumable: complete files are kept.
regrid    Each composite once onto the scene grid (env_meta_json grid: bbox, res, north-up, pixel centres):
            mode     majority class of the source cells whose centres fall in the target cell (ties -> lower id);
                     invalid if the classified share of the source cells < min_support
            nearest  source cell nearest to the target centre
          min_prob masks source cells with class probability P < min_prob first. coast_fill_cells: N > 0 gives
          unclassified cells the class of the nearest classified cell within N cells (coastal gaps of the 5 km
          product; land is masked later by env_data_format). Each complete scene (t, date)
          gets the composite that covers its date (time_stamp: start = composite start dates, the 8-day product's
          convention; centre = nearest stamp within period/2). One int16 GeoTIFF per scene, out_template with {tag}
          and {t}; class 1..33, -1 = no class (land, coast gaps, ice-free fill, or no composite).
          Writes <out_dir>/seascape_classes.csv (id, name) and <tag>_seascape_scene_map.csv.
report    <tag>_seascape_report.csv: class share over water pixels (mask_path; every pixel without a mask) and at the
          track fixes on scene days inside the grid (cmems_env config time.tracks), including the share without a
          class (coastal gaps of the 5 km product, land, missing composites).

Use with env_data_format cluster_level: categorical (cluster_scale 1, zero_label_valid False) as an environment
level (naturenet.environment.make_level_configs), and with sit_fuse.xai.cluster_interpret `reference` for the
cluster / Seascape agreement tables.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import yaml

from naturenet.environment.cmems_env.grid import Grid
from naturenet.environment.proj_env import ensure_proj_data

CLASS_NAMES = [
    "NORTH_ATLANTIC_SPRING_ACC_TRANSITION", "SUBPOLAR_TRANSITION", "TROPICAL_SUBTROPICAL_TRANSITION",
    "WESTERN_WARM_POOL_SUBTROPICAL", "SUBTROPICAL_GYRE_TRANSITION", "ACC_NUTRIENT_STRESS", "TEMPERATE_TRANSITION",
    "INDOPACIFIC_SUBTROPICAL_GYRE", "EQUATORIAL_TRANSITION", "HIGHLY_OLIGOTROPHIC_SUBTROPICAL_GYRE",
    "TROPICAL_SUBTROPICAL_UPWELLING", "SUBPOLAR", "SUBTROPICAL_GYRE_MESOSCALE_INFLUENCED", "TEMPERATE_BLOOMS_UPWELLING",
    "TROPICAL_SEAS", "MEDITTERANEAN_RED_SEA", "SUBTROPICAL_TRANSITION_LOW_NUTRIENT_STRESS", "MEDITERRANEAN_RED_SEA",
    "ARTIC_SUBPOLAR_SHELVES", "SUBTROPICAL_FRESH_INFLUENCED_COASTAL", "WARM_BLOOMS_HIGH_NUTS", "ARCTIC_LATE_SUMMER",
    "FRESHWATER_INFLUENCED_POLAR_SUBPOLAR_SHELVES", "ANTARCTIC_SHELVES", "ICE_PACK_LARGE_POLYNAS",
    "ANTARCTIC_ICE_EDGE", "HYPERSALINE_EUTROPHIC_PERSIAN_GULF_RED_SEA", "ARCTIC_ICE_EDGE", "ANTARCTIC",
    "ICE_EDGE_BLOOM", "1_30percent_ICE_PRESENT", "30_80percent_MARGINAL_ICE", "PACK_ICE"]   # ERDDAP CLASS flag_meanings
N_CLASS = len(CLASS_NAMES)          # ids 1..33, 0 = _FillValue
NODATA = -1

DEFAULTS = dict(erddap="https://cwcgom.aoml.noaa.gov/erddap", dataset_id="noaa_aoml_seascapes_8day",
                variables=["CLASS", "P"], period_days=8, time_stamp="start", margin_deg=0.1, lookback_days=40,
                regrid="mode", min_support=0.5, min_prob=None, coast_fill_cells=0, complete_scenes_only=True,
                out_template="{out_dir}/seascape_{tag}_t{t}.tif", timeout_s=600, retries=4, overwrite=False)


# ---------------------------------------------------------------------------
# Config, grid, calendar
# ---------------------------------------------------------------------------

def load_conf(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    conf = dict(DEFAULTS)
    conf.update(c)
    for k in ("raw_dir", "out_dir"):
        os.makedirs(conf[k], exist_ok=True)
    return conf


def scene_grid(conf):
    with open(conf["env_meta_json"]) as f:
        g = json.load(f)["grid"]
    grid = Grid(g["lon_bounds"][0], g["lon_bounds"][1], g["lat_bounds"][0], g["lat_bounds"][1], float(g["res_deg"]))
    if list(grid.shape) != list(g["shape"]):
        raise ValueError(f"grid shape {grid.shape} differs from the meta JSON {g['shape']}")
    return grid


def scene_dates(conf):
    """[(t, date)] of the scenes (complete ones unless complete_scenes_only: False)."""
    idx = pd.read_csv(conf["scene_index_csv"])
    if conf.get("complete_scenes_only", True) and "complete" in idx:
        idx = idx[idx["complete"].astype(str).str.lower().isin(["true", "1"])]
    return [(int(t), pd.Timestamp(d).normalize()) for t, d in zip(idx["t"], idx["date"])]


def out_path(conf, t):
    return conf["out_template"].format(out_dir=conf["out_dir"].rstrip("/"), tag=conf["tag"], t=t)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def griddap_url(conf, t0, t1, bbox):
    from urllib.parse import quote
    lo0, lo1, la0, la1 = bbox
    sel = f"[({t0}):1:({t1})][({la0:.4f}):1:({la1:.4f})][({lo0:.4f}):1:({lo1:.4f})]"
    q = ",".join(v + sel for v in conf["variables"])
    return f"{conf['erddap'].rstrip('/')}/griddap/{conf['dataset_id']}.nc?" + quote(q, safe=",:()")


def _fetch(url, path, timeout, retries):
    """Stream url to path (via .part); retries with backoff. Returns the path."""
    import requests
    err = None
    for k in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:            # ERDDAP: "Your query produced no matching results."
                    msg = r.text[:2000].lower()
                    if "no matching" in msg or "no data" in msg:
                        return None
                r.raise_for_status()
                with open(path + ".part", "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(path + ".part", path)
            return path
        except Exception as e:          # noqa: BLE001 - network errors of any kind are retried
            err = e
            print(f"  attempt {k + 1}/{retries} failed: {e}")
            time.sleep(min(60, 5 * 2 ** k))
    raise RuntimeError(f"download failed after {retries} attempts: {url}") from err


def _ok_nc(path):
    try:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            return "CLASS" in ds and ds.sizes.get("time", 0) > 0
    except Exception:                   # noqa: BLE001
        return False


def time_axis(conf):
    """(first, last) composite time of the dataset from ERDDAP, or None if the query fails."""
    import requests
    url = f"{conf['erddap'].rstrip('/')}/griddap/{conf['dataset_id']}.csv0?time%5B0:1:last%5D"
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        t = pd.to_datetime(r.text.split(), utc=True).tz_convert(None)
        return t.min(), t.max()
    except Exception as e:              # noqa: BLE001
        print(f"WARNING: time axis query failed ({e}); requesting the full scene range")
        return None


def download(conf, fetch=None, axis="query"):
    """axis: "query" (ERDDAP time axis, clips the requests), None (no clipping) or (first, last)."""
    if axis == "query":
        axis = time_axis(conf) if fetch is None else None
    fetch = fetch or _fetch
    grid = scene_grid(conf)
    m = float(conf["margin_deg"])
    bbox = (grid.lon_min - m, grid.lon_max + m, grid.lat_min - m, grid.lat_max + m)
    dates = [d for _, d in scene_dates(conf)]
    start = min(dates) - pd.Timedelta(days=int(conf["lookback_days"]))
    end = max(dates) + pd.Timedelta(days=1)
    if axis is not None:
        if end < axis[0] or start > axis[1] + pd.Timedelta(days=int(conf["period_days"])):
            raise ValueError(f"scene dates {start.date()}..{end.date()} outside {conf['dataset_id']} "
                             f"({axis[0].date()}..{axis[1].date()})")
        start, end = max(start, axis[0]), min(end, axis[1])
    files = []
    for y in range(start.year, end.year + 1):
        a, b = max(start, pd.Timestamp(f"{y}-01-01")), min(end, pd.Timestamp(f"{y}-12-31T23:59:59"))
        path = os.path.join(conf["raw_dir"], f"{conf['dataset_id']}_{conf['tag']}_{y}.nc")
        if os.path.exists(path) and _ok_nc(path) and not conf.get("overwrite"):
            print("have", path)
            files.append(path)
            continue
        url = griddap_url(conf, a.strftime("%Y-%m-%dT00:00:00Z"), b.strftime("%Y-%m-%dT23:59:59Z"), bbox)
        print("GET", url)
        got = fetch(url, path, int(conf["timeout_s"]), int(conf["retries"]))
        if got is None:
            print(f"  no composites in {y}")
            continue
        if not _ok_nc(path):
            raise RuntimeError(f"{path} is not a readable Seascapes netCDF (check the ERDDAP response)")
        files.append(path)
    return files


# ---------------------------------------------------------------------------
# Regrid
# ---------------------------------------------------------------------------

def _raw_files(conf):
    return sorted(glob.glob(os.path.join(conf["raw_dir"], f"{conf['dataset_id']}_{conf['tag']}_*.nc")))


def open_composites(conf):
    """(times DatetimeIndex, lon (nx,), lat (ny,), CLASS loader(i), P loader(i) or None) over the raw files."""
    import xarray as xr
    ds = xr.open_mfdataset(_raw_files(conf), combine="by_coords") if len(_raw_files(conf)) > 1 else \
        xr.open_dataset(_raw_files(conf)[0])
    ds = ds.sortby("time")
    _, keep = np.unique(ds["time"].values, return_index=True)
    ds = ds.isel(time=np.sort(keep))
    times = pd.DatetimeIndex(ds["time"].values).tz_localize(None)
    lon = ((ds["longitude"].values + 180.0) % 360.0) - 180.0
    lat = ds["latitude"].values
    cls = lambda i: np.nan_to_num(np.asarray(ds["CLASS"].isel(time=i).values, np.float64), nan=0).astype(np.int16)
    prob = (lambda i: np.asarray(ds["P"].isel(time=i).values, np.float64)) if "P" in ds else None
    return ds, times, lon, lat, cls, prob


def composite_for(times, date, mode="start", period_days=8):
    """Index of the composite covering `date`, or -1."""
    d = pd.Timestamp(date).normalize() + pd.Timedelta(hours=12)
    if mode == "start":
        i = int(np.searchsorted(times.values, d.to_datetime64(), side="right")) - 1   # last stamp <= date
        if i < 0:
            return -1
        end = times[i] + pd.Timedelta(days=period_days)
        if i + 1 < len(times):
            end = min(end, times[i + 1])
        return i if d < end else -1
    if mode == "centre":
        i = int(np.argmin(np.abs((times - d).values)))
        return i if abs(times[i] - d) <= pd.Timedelta(days=period_days / 2.0) else -1
    raise ValueError("time_stamp must be start | centre")


class Regridder:
    """Categorical source grid (lon, lat 1-D) -> scene grid, by majority (mode) or nearest neighbour."""

    def __init__(self, lon, lat, grid, method="mode", min_support=0.5):
        self.grid, self.method, self.min_support = grid, method, float(min_support)
        if method == "mode":
            ix = np.floor((lon - grid.lon_min) / grid.res).astype(np.int64)
            iy = np.floor((grid.lat_max - lat) / grid.res).astype(np.int64)
            IY, IX = np.meshgrid(iy, ix, indexing="ij")
            ok = (IX >= 0) & (IX < grid.nx) & (IY >= 0) & (IY < grid.ny)
            self.ok = ok
            self.cell = (IY * grid.nx + IX)[ok]
            self.n_src = np.bincount(self.cell, minlength=grid.ny * grid.nx)
        elif method == "nearest":
            self.jy = np.abs(lat[None, :] - grid.lat[:, None]).argmin(1)
            self.jx = np.abs(lon[None, :] - grid.lon[:, None]).argmin(1)
            self.dy = np.abs(lat[self.jy] - grid.lat).max() if len(lat) else 0
        else:
            raise ValueError("regrid must be mode | nearest")

    def __call__(self, cls):
        g = self.grid
        cls = np.where((cls >= 1) & (cls <= N_CLASS), cls, 0).astype(np.int64)
        if self.method == "nearest":
            out = cls[np.ix_(self.jy, self.jx)].astype(np.int16)
            out[out == 0] = NODATA
            return out
        c = cls[self.ok]
        cnt = np.bincount(self.cell * (N_CLASS + 1) + c, minlength=g.ny * g.nx * (N_CLASS + 1))
        cnt = cnt.reshape(g.ny * g.nx, N_CLASS + 1)
        cnt[:, 0] = 0
        n_valid = cnt.sum(1)
        best = cnt.argmax(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            support = np.where(self.n_src > 0, n_valid / np.maximum(self.n_src, 1), 0.0)
        out = np.where((n_valid > 0) & (support >= self.min_support), best, NODATA).astype(np.int16)
        return out.reshape(g.shape)


def coast_fill(arr, n_cells):
    """Unclassified cells within n_cells of a classified cell take the nearest class."""
    if n_cells <= 0:
        return arr
    from scipy import ndimage
    bad = arr < 1
    if not bad.any() or bad.all():
        return arr
    dist, (iy, ix) = ndimage.distance_transform_edt(bad, return_indices=True)
    fill = bad & (dist <= n_cells)
    out = arr.copy()
    out[fill] = arr[iy[fill], ix[fill]]
    return out


def write_tif(path, arr, grid):
    ensure_proj_data(verbose=True)     # a PROJ_DATA without proj.db breaks EPSG lookups
    import rasterio
    from rasterio.transform import Affine
    prof = dict(driver="GTiff", height=grid.ny, width=grid.nx, count=1, dtype="int16", nodata=NODATA,
                crs="EPSG:4326", transform=Affine(*grid.transform()), compress="deflate")
    with rasterio.open(path + ".part.tif", "w", **prof) as dst:
        dst.write(arr.astype(np.int16), 1)
    os.replace(path + ".part.tif", path)


def regrid(conf):
    grid = scene_grid(conf)
    ds, times, lon, lat, cls, prob = open_composites(conf)
    rg = Regridder(lon, lat, grid, conf["regrid"], conf["min_support"])
    cache, rows = {}, []
    mp = conf.get("min_prob")
    for t, d in scene_dates(conf):
        i = composite_for(times, d, conf["time_stamp"], int(conf["period_days"]))
        path = out_path(conf, t)
        rows.append(dict(t=t, date=d.strftime("%Y-%m-%d"), composite=str(times[i].date()) if i >= 0 else "",
                         path=path))
        if os.path.exists(path) and not conf.get("overwrite"):
            continue
        if i not in cache:
            if i < 0:
                cache = {i: np.full(grid.shape, NODATA, np.int16)}
            else:
                c = cls(i)
                if mp is not None and prob is not None:
                    c = np.where(np.nan_to_num(prob(i), nan=0.0) >= float(mp), c, 0)
                cache = {i: coast_fill(rg(c), int(conf.get("coast_fill_cells") or 0))}   # one composite kept
        write_tif(path, cache[i], grid)
    ds.close()
    pd.DataFrame(rows).to_csv(os.path.join(conf["out_dir"], f"{conf['tag']}_seascape_scene_map.csv"), index=False)
    pd.DataFrame(dict(id=np.arange(1, N_CLASS + 1), name=CLASS_NAMES)).to_csv(
        os.path.join(conf["out_dir"], "seascape_classes.csv"), index=False)
    n_none = sum(r["composite"] == "" for r in rows)
    print(f"{len(rows)} scenes -> {conf['out_dir']} ({n_none} without a covering composite -> all {NODATA})")
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _read_tif(path):
    ensure_proj_data()
    import rasterio
    with rasterio.open(path) as ds:
        return ds.read(1)


def _tracks(conf):
    """Concatenated fixes (time, lon, lat) from the cmems_env config `tracks` (or conf['tracks'])."""
    tr = conf.get("tracks")
    if tr is None and conf.get("cmems_config"):
        with open(conf["cmems_config"]) as f:
            tr = (yaml.safe_load(f).get("time") or {}).get("tracks")
    out = []
    for e in tr or []:
        if not os.path.exists(e["path"]):
            print("WARNING: track file missing:", e["path"])
            continue
        df = pd.read_csv(e["path"], usecols=[e["time_col"], e["lon_col"], e["lat_col"]])
        out.append(pd.DataFrame(dict(
            time=pd.to_datetime(df[e["time_col"]], utc=True, format="mixed", errors="coerce").dt.tz_convert(None),
            lon=df[e["lon_col"]].astype(float), lat=df[e["lat_col"]].astype(float))))
    return pd.concat(out, ignore_index=True).dropna() if out else None


def report(conf):
    grid = scene_grid(conf)
    sm = pd.read_csv(os.path.join(conf["out_dir"], f"{conf['tag']}_seascape_scene_map.csv"))
    water = None
    if conf.get("mask_path") and os.path.exists(conf["mask_path"]):
        water = _read_tif(conf["mask_path"]) != int(conf.get("land_value", 0))
        if water.shape != grid.shape:
            raise ValueError(f"mask {water.shape} differs from the scene grid {grid.shape}")
    px = np.zeros(N_CLASS + 1, np.int64)
    fx = np.zeros(N_CLASS + 1, np.int64)
    tracks = _tracks(conf)
    by_day = {}
    if tracks is not None:
        tracks["day"] = tracks["time"].dt.normalize().dt.strftime("%Y-%m-%d")
        by_day = {d: g for d, g in tracks.groupby("day")}
    for _, r in sm.iterrows():
        a = _read_tif(r["path"]).astype(np.int64)
        v = a[water] if water is not None else a.reshape(-1)
        px += np.bincount(np.where(v > 0, v, 0), minlength=N_CLASS + 1)
        g = by_day.get(r["date"])
        if g is not None:
            iy = np.floor((grid.lat_max - g["lat"].to_numpy()) / grid.res).astype(int)
            ix = np.floor((g["lon"].to_numpy() - grid.lon_min) / grid.res).astype(int)
            ok = (iy >= 0) & (iy < grid.ny) & (ix >= 0) & (ix < grid.nx)
            vals = a[iy[ok], ix[ok]]                       # fixes outside the grid are not counted
            fx += np.bincount(np.where(vals > 0, vals, 0), minlength=N_CLASS + 1)
    rows = [dict(id=0, name="NO_CLASS", water_px=int(px[0]), water_share=px[0] / max(px.sum(), 1),
                 fixes=int(fx[0]), fix_share=fx[0] / max(fx.sum(), 1))]
    rows += [dict(id=k, name=CLASS_NAMES[k - 1], water_px=int(px[k]), water_share=px[k] / max(px.sum(), 1),
                  fixes=int(fx[k]), fix_share=fx[k] / max(fx.sum(), 1)) for k in range(1, N_CLASS + 1)
             if px[k] or fx[k]]
    df = pd.DataFrame(rows)
    path = os.path.join(conf["out_dir"], f"{conf['tag']}_seascape_report.csv")
    df.to_csv(path, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    n_cls = int(((df.id > 0) & (df.fixes > 0)).sum()) if fx.sum() else int((df.id > 0).sum())
    print(f"wrote {path}: {n_cls} classes along the tracks; no class at {rows[0]['fix_share']:.1%} of fixes and "
          f"{rows[0]['water_share']:.1%} of water pixels")
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["download", "regrid", "report", "all"])
    p.add_argument("-y", "--yaml", required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    conf = load_conf(a.yaml)
    if a.overwrite:
        conf["overwrite"] = True
    if a.stage in ("download", "all"):
        download(conf)
    if a.stage in ("regrid", "all"):
        regrid(conf)
    if a.stage in ("report", "all"):
        report(conf)


if __name__ == "__main__":
    main()
