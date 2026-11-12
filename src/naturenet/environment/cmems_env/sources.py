"""Copernicus Marine sources: discovery, channel resolution, and daily block reads.

Data access uses the lazy ARCO interface (`copernicusmarine.open_dataset`), subset to
the bounding box + margin server-side, so only needed chunks are transferred and no
intermediate NetCDF files are written.
"""
import fnmatch
import logging
import time

import numpy as np
import pandas as pd

log = logging.getLogger("cmems_env")


_HTTP = None


def configure_http(pc=None):
    """Size the S3 connection pool of the Copernicus Marine toolbox to the pull concurrency.

    copernicusmarine (2.x) opens one boto3 client per dataset with botocore's default pool of 10
    connections, while one pull block issues up to workers x dask_threads x zarr_concurrency chunk
    requests through that client. Extra connections are then closed after each request ("Connection
    pool is full, discarding connection: s3.waw3-1.cloudferro.com") and every later request pays a new
    TLS handshake. Must run before the first dataset read (clients are created lazily, on the first get).

    pull.http_pool         pool size per client (default: workers * dask_threads * zarr_concurrency,
                           at least 10, at most 128)
    pull.zarr_concurrency  concurrent chunk requests per zarr read (zarr >= 3 'async.concurrency';
                           default 4, zarr's own default is 10)
    pull.read_timeout      seconds without data before an S3 read fails (default 60, botocore's default)
    pull.connect_timeout   seconds to open an S3 connection (default 20; botocore's default is 60)
                           copernicusmarine retries each request up to 10 times (adaptive mode), so a dead
                           socket can take about 10 x (read_timeout + backoff) to surface; pull.block_timeout
                           bounds a whole block on top of this.
    Returns the pool size used.
    """
    pc = pc or {}
    zc = int(pc.get("zarr_concurrency", 4))
    try:
        import zarr
        if not zarr.__version__.startswith("2"):
            zarr.config.set({"async.concurrency": zc})
    except ImportError:
        pass
    n = pc.get("http_pool")
    if n is None:
        n = int(pc.get("workers", 4)) * int(pc.get("dask_threads", 4)) * zc
        n = min(max(n, 10), 128)
    n = int(n)
    try:
        import botocore.config
        # copernicusmarine builds Config(signature_version=..., retries=...) without max_pool_connections,
        # so the class default applies to every client it creates from now on.
        botocore.config.Config.OPTION_DEFAULTS["max_pool_connections"] = n
        # Same mechanism for the socket timeouts (copernicusmarine does not pass them either).
        rt, ct = float(pc.get("read_timeout", 60)), float(pc.get("connect_timeout", 20))
        botocore.config.Config.OPTION_DEFAULTS["read_timeout"] = rt
        botocore.config.Config.OPTION_DEFAULTS["connect_timeout"] = ct
    except ImportError:
        rt = ct = None
    global _HTTP
    if _HTTP != (n, zc, rt, ct):
        log.info("S3 connection pool %d per dataset, zarr concurrency %d, read/connect timeout %s/%s s",
                 n, zc, rt, ct)
        _HTTP = (n, zc, rt, ct)
    return n

LON_NAMES = ("longitude", "lon", "x")
LAT_NAMES = ("latitude", "lat", "y")
DEPTH_NAMES = ("depth", "elevation", "deptht")
DEFAULT_EXCLUDE = ["err_*", "*_err", "*_error", "*_bias", "*_sdd", "*_dv", "number_of_*",
                   "flag_*", "*_flag", "tpa_correction", "mask", "*_uncertainty", "sea_ice_fraction"]


def _coord(ds, names):
    for n in names:
        if n in ds.coords or n in ds.dims:
            return n
    return None


def open_lazy(dataset_id, bbox, margin, variables=None, depth_range=None, retries=3, **kw):
    import copernicusmarine as cm
    args = dict(dataset_id=dataset_id,
                minimum_longitude=bbox[0] - margin, maximum_longitude=bbox[1] + margin,
                minimum_latitude=bbox[2] - margin, maximum_latitude=bbox[3] + margin,
                coordinates_selection_method="outside")
    if variables:
        args["variables"] = list(variables)
    if depth_range is not None:
        args["minimum_depth"], args["maximum_depth"] = depth_range
    args.update(kw)
    for k in range(retries):
        try:
            return cm.open_dataset(**args)
        except Exception as e:  # network / auth / transient
            if k == retries - 1:
                raise
            log.warning("open_dataset(%s) failed (%s); retry %d", dataset_id, e, k + 1)
            time.sleep(5 * (k + 1))


class Source:
    """One logical source = ordered dataset candidates sharing variable names.

    For every day the first candidate whose time axis covers it is used (e.g. MY
    reanalysis, then the interim/NRT continuation).
    """

    def __init__(self, cfg, bbox, margin, opener=open_lazy):
        self.cfg = cfg
        self.name = cfg["name"]
        self.datasets = cfg["datasets"] if isinstance(cfg["datasets"], list) else [cfg["datasets"]]
        self.bbox, self.margin, self.opener = bbox, float(cfg.get("margin_deg", margin)), opener
        self.temporal = cfg.get("temporal", "daily")  # daily | prior (monthly, ffill)
        self.circular = set(cfg.get("circular", []))
        self.transforms = cfg.get("transforms", {})  # var: {scale, offset}
        self.rename = cfg.get("rename", {})
        self.block_days = int(cfg.get("block_days", 16))
        self._ds = {}
        self.channels = None  # list of (var, depth_value_or_None, out_name)
        self.coverage = {}

    # ---------------------------------------------------------------- resolve
    def _depth_range(self):
        dmax = 0.0
        for v in (self.cfg.get("depths") or {}).values():
            dmax = max(dmax, max(v))
        return (0.0, dmax + 1.0) if self.cfg.get("depths") else None

    def dataset(self, dsid):
        if dsid not in self._ds:
            want = self.cfg.get("variables", "all")
            vars_ = None if want == "all" else list(want)
            self._ds[dsid] = self.opener(dsid, self.bbox, self.margin, variables=vars_,
                                         depth_range=self._depth_range())
        return self._ds[dsid]

    def resolve(self):
        """Open candidates lazily (metadata only), list channels, record coverage."""
        ds0 = None
        for dsid in self.datasets:
            try:
                ds = self.dataset(dsid)
            except Exception as e:
                log.warning("[%s] cannot open %s: %s", self.name, dsid, e)
                continue
            t = pd.DatetimeIndex(ds["time"].values).floor("D")
            self.coverage[dsid] = (t.min(), t.max())
            ds0 = ds if ds0 is None else ds0
        if ds0 is None:
            raise RuntimeError(f"[{self.name}] no dataset could be opened: {self.datasets}")
        lon, lat = _coord(ds0, LON_NAMES), _coord(ds0, LAT_NAMES)
        dep = _coord(ds0, DEPTH_NAMES)
        excl = DEFAULT_EXCLUDE + list(self.cfg.get("exclude", []))
        want = self.cfg.get("variables", "all")
        names = list(ds0.data_vars) if want == "all" else list(want)
        depths = self.cfg.get("depths") or {}
        chans = []
        for v in names:
            if v not in ds0.data_vars or any(fnmatch.fnmatch(v, p) for p in excl):
                continue
            dims = ds0[v].dims
            if "time" not in dims or lon not in dims or lat not in dims:
                continue
            base = self.rename.get(v, v)
            comps = [("sin", np.sin), ("cos", np.cos)] if v in self.circular else [(None, None)]
            if dep is not None and dep in dims:
                levels = np.asarray(ds0[dep].values, float)
                req = depths.get(v, depths.get("default", [0]))
                for d in req:
                    lev = float(levels[np.argmin(np.abs(levels - d))])
                    for c, _ in comps:
                        chans.append((v, lev, f"{base}_{int(round(d))}m" + (f"_{c}" if c else ""), c))
            else:
                for c, _ in comps:
                    chans.append((v, None, base + (f"_{c}" if c else ""), c))
        self.channels = chans
        self.lon_name, self.lat_name, self.depth_name = lon, lat, dep
        return [c[2] for c in chans]

    def dataset_for_days(self, days):
        """Map each day to the first candidate dataset covering it (or None)."""
        out = []
        for d in days:
            pick = None
            for dsid in self.datasets:
                cov = self.coverage.get(dsid)
                if cov is not None and cov[0] <= d <= cov[1] + pd.Timedelta(days=31 if self.temporal == "prior" else 0):
                    pick = dsid
                    break
            out.append(pick)
        return out

    # ---------------------------------------------------------------- read
    def read_days(self, dsid, days):
        """Daily fields for `days` (sorted DatetimeIndex) from one dataset.

        Returns (data (T, V, Hs, Ws) float32, lon, lat). Sub-daily data is averaged to
        daily means (direction variables as sin/cos first); monthly data is forward
        filled from the most recent time step.
        """
        ds = self.dataset(dsid)
        lon = np.asarray(ds[self.lon_name].values, float)
        lat = np.asarray(ds[self.lat_name].values, float)
        t_all = pd.DatetimeIndex(ds["time"].values)
        if self.temporal == "prior":
            idx = np.searchsorted(t_all.values, (days + pd.Timedelta(hours=23, minutes=59)).values, side="right") - 1
            idx = np.clip(idx, 0, len(t_all) - 1)
            uniq, inv = np.unique(idx, return_inverse=True)
            sub = ds.isel(time=uniq)
            pos = inv
        else:
            sub = ds.sel(time=slice(days[0], days[-1] + pd.Timedelta(hours=23, minutes=59, seconds=59)))
            tday = pd.DatetimeIndex(sub["time"].values).floor("D")
            pos = None
        T, V = len(days), len(self.channels)
        out = np.full((T, V, len(lat), len(lon)), np.nan, np.float32)
        groups = {}
        for k, (v, lev, _, comp) in enumerate(self.channels):
            groups.setdefault((v, lev), []).append((k, comp))
        for (v, lev), items in groups.items():
            da = sub[v]
            if lev is not None:
                da = da.sel({self.depth_name: lev}, method="nearest")
            x = self._load(da)
            # order dims to (time, lat, lon)
            x = np.moveaxis(x, [da.dims.index("time"), da.dims.index(self.lat_name), da.dims.index(self.lon_name)], [0, 1, 2])
            tr = self.transforms.get(v, {})
            if tr:
                x = x * float(tr.get("scale", 1.0)) + float(tr.get("offset", 0.0))
            for k, comp in items:
                y = x
                if comp is not None:
                    r = np.deg2rad(x)
                    y = np.sin(r) if comp == "sin" else np.cos(r)
                if pos is not None:
                    out[:, k] = y[pos]
                else:
                    out[:, k] = _daily_mean(y, tday, days)
        return out, lon, lat

    def _load(self, da, retries=3):
        for k in range(retries):
            try:
                return np.asarray(da.values, np.float32)
            except Exception as e:
                if k == retries - 1:
                    raise
                log.warning("[%s] read failed (%s); retry %d", self.name, e, k + 1)
                time.sleep(5 * (k + 1))


def _daily_mean(x, tday, days):
    """NaN-aware mean of (Tsub, H, W) grouped by calendar day onto `days`."""
    out = np.full((len(days),) + x.shape[1:], np.nan, np.float32)
    if len(tday) == 0:
        return out
    codes = days.get_indexer(tday)
    for i in np.unique(codes[codes >= 0]):
        sel = x[codes == i]
        if sel.shape[0] == 1:
            out[i] = sel[0]
        else:
            with np.errstate(invalid="ignore"):
                n = np.isfinite(sel).sum(0)
                s = np.nansum(sel, 0)
                out[i] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


# -------------------------------------------------------------------- discover
def discover(bbox, t0, t1, max_step_days=1.0, include_monthly=False, out_csv=None, contains=None):
    """List every CMEMS dataset/variable overlapping bbox and [t0, t1] at <= daily step.

    Writes a catalogue CSV (product, dataset, variable, units, time range, step, depth
    flag, resolution) to choose sources from. Requires network; ~1-2 min.
    """
    import copernicusmarine as cm
    cat = cm.describe(contains=contains or [], disable_progress_bar=True)
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    rows = []
    for p in cat.products:
        for d in p.datasets:
            for ver in d.versions:
                for part in ver.parts:
                    for svc in part.services:
                        if svc.service_short_name not in ("timeseries", "geoseries", "arco-time-series", "arco-geo-series", "static-arco"):
                            continue
                        for var in svc.variables:
                            co = {c.coordinate_id: c for c in var.coordinates}
                            tc = co.get("time")
                            xs, ys = co.get("longitude") or co.get("x"), co.get("latitude") or co.get("y")
                            if xs is None or ys is None:
                                continue
                            try:
                                if xs.maximum_value < bbox[0] or xs.minimum_value > bbox[1] or ys.maximum_value < bbox[2] or ys.minimum_value > bbox[3]:
                                    continue
                            except TypeError:
                                pass
                            row = dict(product=p.product_id, dataset=d.dataset_id, version=ver.label,
                                       variable=var.short_name, units=getattr(var, "units", ""),
                                       has_depth="depth" in co, res_deg=getattr(xs, "step", None))
                            if tc is not None and tc.minimum_value is not None:
                                ts = pd.to_datetime(tc.minimum_value, unit="ms")
                                te = pd.to_datetime(tc.maximum_value, unit="ms")
                                step = (tc.step or 0) / 86400000.0
                                if te < t0 or ts > t1:
                                    continue
                                if step and step > max_step_days and not (include_monthly and step <= 31):
                                    continue
                                row.update(time_start=ts, time_end=te, step_days=step)
                            rows.append(row)
                        break
    df = pd.DataFrame(rows).drop_duplicates(["dataset", "variable"])
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df
