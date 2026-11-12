"""Static layers: relaxed water mask, distance to coast, bathymetry, ship density."""
import numpy as np
from scipy import ndimage

from .grid import Grid, aggregate_raster, block_reduce_mean, nearest_distance_km


def _ocean_connected(water, covered=None):
    """Keep water connected to the open boundary (same rule as environment/build_ocean_mask.py).

    The open boundary is the array edge plus, when `covered` is given, any cell next to a
    cell the mask raster does not cover. Without the second part, a mask raster clipped to
    the study area (with land assumed off-raster) has an all-land margin, so no water
    touches the edge and the whole domain is removed.
    """
    lab, _ = ndimage.label(water, structure=np.ones((3, 3), np.uint8))
    border = np.zeros(water.shape, bool)
    border[0], border[-1], border[:, 0], border[:, -1] = True, True, True, True
    if covered is not None and not covered.all():
        border |= ndimage.binary_dilation(~covered, structure=np.ones((3, 3), bool))
    edge = np.unique(lab[border])
    edge = edge[edge > 0]
    return np.isin(lab, edge)


def _seed_connected(water, grid, seeds, path=""):
    """Keep water connected to known open-ocean points (lon, lat) instead of the edge.

    Use this when the domain edge or a corner is land, or the raster is clipped to the
    study area. A seed that falls on land raises an error, which also catches an inverted mask.
    """
    lab, _ = ndimage.label(water, structure=np.ones((3, 3), np.uint8))
    keep = []
    for lon, lat in seeds:
        i = int(np.floor((grid.lat_max - lat) / grid.res))
        j = int(np.floor((lon - grid.lon_min) / grid.res))
        if not (0 <= i < grid.ny and 0 <= j < grid.nx):
            raise ValueError("ocean seed (%g, %g) is outside the grid" % (lon, lat))
        if lab[i, j] == 0:
            raise ValueError("ocean seed (%g, %g) is not water in %s: check the seed, "
                             "mask.water_value, or whether the mask is inverted" % (lon, lat, path))
        keep.append(lab[i, j])
    return np.isin(lab, keep)


def _disk(ry, rx):
    ry, rx = max(int(np.ceil(ry)), 0), max(int(np.ceil(rx)), 0)
    y, x = np.ogrid[-ry:ry + 1, -rx:rx + 1]
    return ((y / max(ry, 1e-9)) ** 2 + (x / max(rx, 1e-9)) ** 2) <= 1.0 if (ry or rx) else np.ones((1, 1), bool)


def build_water_mask(cfg, grid):
    """Water mask on the target grid with a relaxed land boundary.

    Uses the same raster and `water_value` convention as the land-jitter step
    (environment/jitter_bad_points.py WaterMask). Steps:
      1. stream the mask onto a k-times finer grid (with a margin, for distance to coast);
      2. optionally keep only ocean-connected water (drops lakes): water connected to
         `ocean_seeds` ([[lon, lat], ...] known open-ocean points) if given, otherwise
         water connected to the domain edge or to the raster's own boundary;
      3. a target cell is water if its water fraction > `min_water_frac` (default 0: any water);
      4. relax the coastline by dilating water by `coast_buffer_km`, so coastal fixes and
         partially-wet cells keep valid (interpolated) values instead of fill.
    Returns dict(mask, water_frac, dist_coast_km).
    """
    k = int(cfg.get("supersample", 4))
    margin = int(cfg.get("margin_cells", 24))
    wv = cfg.get("water_value", 1)
    off = 1.0 if cfg.get("offraster_is_water", False) else 0.0
    big = grid.expand(margin)
    fine = big.supersample(k)
    frac_f = aggregate_raster(cfg["path"], fine, func=lambda x: (x == wv).astype(np.float64),
                              stats=("mean",), offraster=np.nan)["mean"]
    covered = np.isfinite(frac_f)
    if not covered.any():
        raise ValueError("water mask %s does not overlap the grid bbox %s (check its extent/CRS)"
                         % (cfg["path"], (grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max)))
    water_raw = np.nan_to_num(frac_f, nan=off) >= 0.5
    water_f = water_raw
    seeds = cfg.get("ocean_seeds")  # optional [[lon, lat], ...] known open-ocean points
    if seeds:
        water_f = _seed_connected(water_raw, fine, seeds, cfg["path"])
    elif cfg.get("ocean_connected", True):
        water_f = _ocean_connected(water_raw, covered)
    # inner region (target grid) on the fine grid
    i0, j0 = margin * k, margin * k
    inner = water_f[i0:i0 + grid.ny * k, j0:j0 + grid.nx * k]
    frac = block_reduce_mean(inner.astype(np.float32), k)
    water = frac > float(cfg.get("min_water_frac", 0.0))
    if not water.any():
        try:
            import rasterio
            with rasterio.open(cfg["path"]) as src:
                rinfo = "raster nodata=%r bounds=%s crs=%s" % (src.nodata, tuple(src.bounds), src.crs)
        except Exception as e:  # pragma: no cover
            rinfo = "raster info unavailable (%s)" % e
        raw = block_reduce_mean(water_raw[i0:i0 + grid.ny * k, j0:j0 + grid.nx * k].astype(np.float32), k)
        raise ValueError(
            "water mask is empty on the grid: %d cells are water before the ocean-connected step "
            "(%d after), raster covers %.1f%% of the grid. Check mask.water_value (now %r), "
            "mask.offraster_is_water, or set mask.ocean_connected: false. %s"
            % (int((raw > 0).sum()), int((frac > 0).sum()),
               100.0 * float(block_reduce_mean(covered[i0:i0 + grid.ny * k, j0:j0 + grid.nx * k]
                                               .astype(np.float32), k).mean()), wv, rinfo))
    buf_km = float(cfg.get("coast_buffer_km", 0.0))
    if buf_km > 0:
        ry = buf_km / grid.dy_km()
        rx = buf_km / float(np.mean(grid.dx_km()))
        water = ndimage.binary_dilation(water, structure=_disk(ry, rx))
    # distance to coast: nearest land-cell-adjacent-to-water on the fine grid
    land = ~water_f
    coast = land & ndimage.binary_dilation(water_f, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    cy, cx = np.nonzero(coast)
    qlon, qlat = np.meshgrid(grid.lon, grid.lat)
    dist = nearest_distance_km(fine.lon[cx], fine.lat[cy], qlon, qlat)
    if np.all(np.isnan(dist)):
        dist = np.full(grid.shape, float(cfg.get("no_coast_distance_km", 1000.0)), np.float32)
    dist = np.where(frac > 0, dist, -dist).astype(np.float32)  # negative over full-land cells
    return dict(mask=water, water_frac=frac.astype(np.float32), dist_coast_km=dist)


def slope_deg(depth_m, grid):
    gy, gx = np.gradient(depth_m.astype(np.float64))
    dzdy = gy / (grid.dy_km() * 1000.0)
    dzdx = gx / (grid.dx_km()[:, None] * 1000.0)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)


def isobath_distance_km(depth_m, grid, isobath):
    """Signed distance to an isobath (positive on the deeper side), e.g. 200 m shelf break."""
    deep = depth_m >= isobath
    edge = deep & ndimage.binary_dilation(~deep & np.isfinite(depth_m))
    cy, cx = np.nonzero(edge)
    qlon, qlat = np.meshgrid(grid.lon, grid.lat)
    d = nearest_distance_km(grid.lon[cx], grid.lat[cy], qlon, qlat)
    return np.where(deep, d, -d).astype(np.float32)


def build_bathymetry(cfg, grid, cmems_static=None):
    """Depth (m, positive down) and derived terrain layers.

    Preferred: GEBCO (15 arc-sec; elevation negative below sea level) via `path`.
    Fallback: CMEMS model bathymetry (`deptho`) passed in as `cmems_static`.
    """
    out = {}
    if cfg.get("path"):
        a = aggregate_raster(cfg["path"], grid, band=int(cfg.get("band", 1)),
                             stats=("mean", "std"))
        sign = -1.0 if cfg.get("elevation", True) else 1.0
        depth = sign * a["mean"]
        out["rugosity_m"] = a["std"]
    elif cmems_static is not None:
        depth = cmems_static
    else:
        raise ValueError("bathymetry: set `path` (GEBCO) or enable the CMEMS static fallback")
    depth = np.where(np.isfinite(depth), depth, np.nan).astype(np.float32)
    wet = np.clip(np.nan_to_num(depth, nan=0.0), 0, None)
    out["depth_m"] = wet
    out["log10_depth"] = np.log10(wet + 1.0).astype(np.float32)
    out["slope_deg"] = slope_deg(wet, grid)
    for iso in cfg.get("isobaths", [200]):
        out[f"dist_{int(iso)}m_isobath_km"] = isobath_distance_km(wet, grid, float(iso))
    if "rugosity_m" in out:
        out["rugosity_m"] = np.nan_to_num(out["rugosity_m"], nan=0.0).astype(np.float32)
    return out


def build_ship_density(layers, grid):
    """Ship-density rasters (World Bank/IMF AIS: counts of hourly AIS positions, 2015-2021).

    Raster values are summed per target cell and divided by cell area (km^2). Per layer:
      scale      multiply raster values first (default 1). For a raster min-max normalised
                 from counts with min 0, set this to the original max count to recover counts.
      transform  log1p (default; right for counts), log10 (log10(density + eps); use for a
                 normalised raster when the original max is unknown, as it only shifts by a
                 constant under rescaling), or none.
      eps        log10 offset (default 1e-12; keep it below the smallest non-zero density).
    """
    out = {}
    area = grid.cell_area_km2()
    for lay in layers:
        scale = float(lay.get("scale", 1.0))
        s = aggregate_raster(lay["path"], grid, band=int(lay.get("band", 1)),
                             func=lambda x: np.clip(x, 0, None) * scale, stats=("sum",), offraster=0.0)["sum"]
        dens = np.nan_to_num(s, nan=0.0) / area
        tr = str(lay.get("transform", "log1p")).lower()
        if tr == "log1p":
            v = np.log1p(dens)
        elif tr == "log10":
            v = np.log10(dens + float(lay.get("eps", 1e-12)))
        elif tr == "none":
            v = dens
        else:
            raise ValueError("ship_density transform must be log1p, log10 or none, got %r" % tr)
        out[lay.get("name", "ship_density")] = v.astype(np.float32)
    return out
