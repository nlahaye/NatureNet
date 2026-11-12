"""Target grid, separable NaN-aware regridding, and streaming raster aggregation."""
from dataclasses import dataclass

import numpy as np

R_EARTH_KM = 6371.0088
KM_PER_DEG = np.pi * R_EARTH_KM / 180.0
OMEGA = 7.2921159e-5


@dataclass
class Grid:
    """Regular lon/lat grid, pixel-centre registered, north-up (row 0 = northernmost)."""
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    res: float

    def __post_init__(self):
        self.nx = int(round((self.lon_max - self.lon_min) / self.res))
        self.ny = int(round((self.lat_max - self.lat_min) / self.res))
        self.lon = self.lon_min + (np.arange(self.nx) + 0.5) * self.res
        self.lat = self.lat_max - (np.arange(self.ny) + 0.5) * self.res

    @property
    def shape(self):
        return (self.ny, self.nx)

    def supersample(self, k):
        return Grid(self.lon_min, self.lon_max, self.lat_min, self.lat_max, self.res / k)

    def expand(self, n_cells):
        d = n_cells * self.res
        return Grid(self.lon_min - d, self.lon_max + d, max(self.lat_min - d, -90.0),
                    min(self.lat_max + d, 90.0), self.res)

    def dy_km(self):
        return self.res * KM_PER_DEG

    def dx_km(self):
        """Per-row east-west cell size (ny,)."""
        return self.res * KM_PER_DEG * np.cos(np.deg2rad(self.lat))

    def cell_area_km2(self):
        return (self.dx_km() * self.dy_km())[:, None] * np.ones((1, self.nx))

    def coriolis(self):
        return 2 * OMEGA * np.sin(np.deg2rad(self.lat))

    def transform(self):
        """GDAL/affine geotransform tuple (a, b, c, d, e, f) for rasterio.Affine."""
        return (self.res, 0.0, self.lon_min, 0.0, -self.res, self.lat_max)

    def to_dict(self):
        return dict(lon_bounds=[self.lon_min, self.lon_max], lat_bounds=[self.lat_min, self.lat_max],
                    res_deg=self.res, shape=[self.ny, self.nx], north_up=True,
                    registration="pixel_centre")


def _edges(c):
    c = np.asarray(c, float)
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def axis_weights(src, tgt, tgt_res, method="auto"):
    """(n_tgt, n_src) weights for one axis. `src` must be ascending; `tgt` any order.

    area   : 1-D overlap length (conservative box average) - used when the source is finer.
    linear : 2-point linear interpolation - used when the source is coarser or similar.
    """
    src = np.asarray(src, float)
    tgt = np.asarray(tgt, float)
    src_res = float(np.median(np.diff(src))) if len(src) > 1 else tgt_res
    if method == "auto":
        method = "area" if src_res < 0.75 * tgt_res else "linear"
    W = np.zeros((len(tgt), len(src)), np.float32)
    if method == "area":
        se = _edges(src)
        t0 = (tgt - tgt_res / 2)[:, None]
        t1 = (tgt + tgt_res / 2)[:, None]
        W[:] = np.clip(np.minimum(t1, se[None, 1:]) - np.maximum(t0, se[None, :-1]), 0, None)
    else:
        pos = np.interp(tgt, src, np.arange(len(src)), left=np.nan, right=np.nan)
        ok = np.isfinite(pos)
        i0 = np.clip(np.floor(pos[ok]).astype(int), 0, max(len(src) - 2, 0))
        w1 = pos[ok] - i0
        rows = np.where(ok)[0]
        if len(src) == 1:
            W[rows, 0] = 1.0
        else:
            W[rows, i0] = 1.0 - w1
            W[rows, i0 + 1] += w1
    return W


class Regridder:
    """Separable NaN-aware regridding: out = (Wy @ (v*m) @ Wx) / (Wy @ m @ Wx).

    Works on any (..., H_src, W_src) stack in one BLAS call. Cells with less than
    `min_support` of their full weight covered by valid source data become NaN (the
    later fill step handles them), so land values never leak into the ocean.
    """

    def __init__(self, src_lon, src_lat, grid, method="auto", min_support=0.25):
        lon = ((np.asarray(src_lon, float) + 180.0) % 360.0) - 180.0
        lat = np.asarray(src_lat, float)
        self.ix = np.argsort(lon, kind="stable")
        self.iy = np.argsort(lat, kind="stable")
        self.Wy = axis_weights(lat[self.iy], grid.lat, grid.res, method)
        self.Wx = np.ascontiguousarray(axis_weights(lon[self.ix], grid.lon, grid.res, method).T)
        self.full = np.outer(self.Wy.sum(1), self.Wx.sum(0)).astype(np.float32)
        self.min_support = min_support
        self.sorted_y = np.all(self.iy == np.arange(len(self.iy)))
        self.sorted_x = np.all(self.ix == np.arange(len(self.ix)))

    def __call__(self, a):
        a = np.asarray(a, np.float32)
        if not self.sorted_y:
            a = np.take(a, self.iy, axis=-2)
        if not self.sorted_x:
            a = np.take(a, self.ix, axis=-1)
        valid = np.isfinite(a)
        v = np.where(valid, a, 0.0).astype(np.float32)
        num = self.Wy @ v @ self.Wx
        den = self.Wy @ valid.astype(np.float32) @ self.Wx
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        bad = (den < self.min_support * self.full) | (self.full <= 0)
        out[np.broadcast_to(bad, out.shape)] = np.nan
        return out.astype(np.float32)


def aggregate_raster(path, grid, band=1, func=None, stats=("mean",), strip_rows=32, offraster=np.nan):
    """Stream a (usually much finer) raster onto `grid` with exact per-cell statistics.

    Reads strips of target rows so memory stays bounded even for 10 GB global rasters
    (GEBCO, World Bank ship density, water masks). `func` maps the raw band to the
    quantity to aggregate (e.g. `lambda x: x == 1` for a water mask). Supported stats:
    mean, sum, std, count. Coarser rasters are bilinearly interpolated instead.
    Non-EPSG:4326 rasters are read through a WarpedVRT.
    """
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds, Window

    out = {s: np.full(grid.shape, np.nan, np.float32) for s in stats}
    with rasterio.open(path) as src0:
        epsg = src0.crs.to_epsg() if src0.crs is not None else 4326
        src = src0 if epsg == 4326 else WarpedVRT(src0, crs="EPSG:4326")
        try:
            nodata = src.nodata
            rx, ry = abs(src.transform.a), abs(src.transform.e)
            if rx > 0.75 * grid.res:
                return _interp_coarse_raster(src, grid, band, func, nodata, stats, out)
            sums = np.zeros(grid.shape, np.float64)
            sq = np.zeros(grid.shape, np.float64) if "std" in stats else None
            cnt = np.zeros(grid.shape, np.float64)
            for r0 in range(0, grid.ny, strip_rows):
                r1 = min(r0 + strip_rows, grid.ny)
                top = grid.lat_max - r0 * grid.res
                bot = grid.lat_max - r1 * grid.res
                win = from_bounds(grid.lon_min, bot, grid.lon_max, top, src.transform)
                win = win.round_offsets(op="floor").round_lengths(op="ceil")
                full = Window(0, 0, src.width, src.height)
                try:
                    win = win.intersection(full)
                except Exception:
                    continue
                if win.width <= 0 or win.height <= 0:
                    continue
                x = src.read(band, window=win).astype(np.float64)
                valid = np.ones(x.shape, bool)
                if nodata is not None:
                    valid &= x != nodata
                valid &= np.isfinite(x)
                if func is not None:
                    x = np.asarray(func(x), np.float64)
                    valid &= np.isfinite(x)
                wt = src.window_transform(win)
                cols = wt.c + (np.arange(x.shape[1]) + 0.5) * wt.a
                rows = wt.f + (np.arange(x.shape[0]) + 0.5) * wt.e
                ti = np.floor((grid.lat_max - rows) / grid.res).astype(int)
                tj = np.floor((cols - grid.lon_min) / grid.res).astype(int)
                okr = (ti >= r0) & (ti < r1)
                okc = (tj >= 0) & (tj < grid.nx)
                x = x[okr][:, okc]
                valid = valid[okr][:, okc]
                ti, tj = ti[okr], tj[okc]
                flat = (ti[:, None] * grid.nx + tj[None, :])[valid]
                xv = x[valid]
                n = grid.ny * grid.nx
                sums += np.bincount(flat, weights=xv, minlength=n).reshape(grid.shape)
                cnt += np.bincount(flat, minlength=n).reshape(grid.shape)
                if sq is not None:
                    sq += np.bincount(flat, weights=xv * xv, minlength=n).reshape(grid.shape)
        finally:
            if src is not src0:
                src.close()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / cnt
        if "mean" in stats:
            out["mean"] = np.where(cnt > 0, mean, offraster).astype(np.float32)
        if "sum" in stats:
            out["sum"] = np.where(cnt > 0, sums, offraster).astype(np.float32)
        if "count" in stats:
            out["count"] = cnt.astype(np.float32)
        if "std" in stats:
            var = np.maximum(sq / cnt - mean * mean, 0.0)
            out["std"] = np.where(cnt > 1, np.sqrt(var), np.nan).astype(np.float32)
    return out


def _interp_coarse_raster(src, grid, band, func, nodata, stats, out):
    from rasterio.windows import from_bounds
    m = 2 * abs(src.transform.a)
    win = from_bounds(grid.lon_min - m, grid.lat_min - m, grid.lon_max + m, grid.lat_max + m, src.transform)
    win = win.round_offsets(op="floor").round_lengths(op="ceil")
    x = src.read(band, window=win, boundless=True, fill_value=nodata if nodata is not None else np.nan)
    x = x.astype(np.float32)
    if nodata is not None:
        x[x == nodata] = np.nan
    if func is not None:
        x = np.asarray(func(x), np.float32)
    wt = src.window_transform(win)
    lon = wt.c + (np.arange(x.shape[1]) + 0.5) * wt.a
    lat = wt.f + (np.arange(x.shape[0]) + 0.5) * wt.e
    v = Regridder(lon, lat, grid, method="linear")(x)
    for s in stats:
        out[s] = v if s in ("mean", "sum") else np.full(grid.shape, np.nan, np.float32)
    return out


def block_reduce_mean(a, k):
    """Mean over k x k blocks of a 2-D array whose shape is divisible by k."""
    h, w = a.shape
    return a.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def lonlat_to_xyz(lon, lat):
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def nearest_distance_km(src_lon, src_lat, q_lon, q_lat):
    """Great-circle distance from each query point to the nearest source point (KD-tree)."""
    from scipy.spatial import cKDTree
    if len(src_lon) == 0:
        return np.full(np.shape(q_lon), np.nan, np.float32)
    tree = cKDTree(lonlat_to_xyz(np.ravel(src_lon), np.ravel(src_lat)))
    d, _ = tree.query(lonlat_to_xyz(np.ravel(q_lon), np.ravel(q_lat)), k=1, workers=-1)
    km = 2 * R_EARTH_KM * np.arcsin(np.clip(d / 2, 0, 1))
    return km.reshape(np.shape(q_lon)).astype(np.float32)
