import os
import argparse

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

from sit_fuse.utils import read_yaml
from naturenet.environment.jitter_from_argos import (
    apply_gps_ellipse,
    meters_to_degree_offsets,
    sample_argos_ellipse_offsets,
)


class WaterMask:
    """
    Vectorised water lookup on a raster mask.

    - Reprojects lon/lat (EPSG:4326) into the raster CRS if needed.
    - Points outside the raster bounds or on nodata are NOT water by default
      (``offraster_is_water``), instead of silently reading the fill value.
    """

    def __init__(self, src, water_value, offraster_is_water=False):
        self.src = src
        self.water_value = water_value
        self.offraster_is_water = offraster_is_water
        self.crs_is_4326 = src.crs is None or src.crs.to_epsg() == 4326
        if src.crs is None:
            print("WARNING: water mask has no CRS; assuming EPSG:4326.")
        self.band = src.read(1)
        self.nodata = src.nodata

    def __call__(self, lon, lat):
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        out = np.full(lon.shape, self.offraster_is_water, dtype=bool)
        finite = np.isfinite(lon) & np.isfinite(lat)
        if not finite.any():
            return out
        x, y = lon[finite], lat[finite]
        if not self.crs_is_4326:
            x, y = warp_transform("EPSG:4326", self.src.crs, x, y)
            x, y = np.asarray(x), np.asarray(y)
        rows, cols = rasterio.transform.rowcol(self.src.transform, x, y)
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < self.band.shape[0]) & (cols >= 0) & (cols < self.band.shape[1])
        res = np.full(x.shape, self.offraster_is_water, dtype=bool)
        vals = self.band[rows[inside], cols[inside]]
        ok = vals == self.water_value
        if self.nodata is not None:
            nod = (vals == self.nodata) | (np.isnan(vals) if np.issubdtype(vals.dtype, np.floating) else False)
            ok = np.where(nod, self.offraster_is_water, ok)
        res[inside] = ok
        out[finite] = res
        return out


def _original_position(group_df, lon_col, lat_col):
    """
    Recover the un-jittered fix for each row, so land points are re-drawn from the
    ORIGINAL Argos ellipse (one draw), not original + eps1 + eps2 (double variance).
    """
    lon = pd.to_numeric(group_df[lon_col], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(group_df[lat_col], errors="coerce").to_numpy(dtype=float)
    o_lon_col, o_lat_col = f"orig_{lon_col}", f"orig_{lat_col}"
    if o_lon_col in group_df.columns and o_lat_col in group_df.columns:
        o_lon = pd.to_numeric(group_df[o_lon_col], errors="coerce").to_numpy(dtype=float)
        o_lat = pd.to_numeric(group_df[o_lat_col], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(o_lon), o_lon, lon), np.where(np.isfinite(o_lat), o_lat, lat)
    if "argos_dx_m" in group_df.columns and "argos_dy_m" in group_df.columns:
        dx = pd.to_numeric(group_df["argos_dx_m"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        dy = pd.to_numeric(group_df["argos_dy_m"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return meters_to_degree_offsets(lon, lat, -dx, -dy)
    return lon, lat


def jitter_path_off_land(
    path_df,
    mask_path,
    group_cols=["uid"],
    lon_col="longitude",
    lat_col="latitude",
    semi_major_col="semi_major",
    semi_minor_col="semi_minor",
    orientation_col="err_orient",
    id_col="uid",
    water_value=0,
    max_tries=100,
    time_col="timestamp",
    lc_col="lc",
    offraster_is_water=False,
    seed=123,
):
    """
    Re-draw fixes that fall on land from their Argos error ellipse until they fall on water.
    If no water draw is found after ``max_tries``, fall back to the previous fix
    (or keep the original for the first fix).

    Adds column ``land_jitter_status``:
        0 = already water, 1 = re-drawn onto water, 2 = fell back to previous fix,
        3 = first fix kept on land.

    Note: status 2 creates zero-length "stay" steps; filter or down-weight them if
    they bias stay counts.
    """
    out = path_df.copy()
    out = apply_gps_ellipse(out, semi_major_col, semi_minor_col, orientation_col, lc_col=lc_col)
    rng = np.random.default_rng(seed)

    grouped = [("all_rows", out)] if not group_cols else list(out.groupby(group_cols, dropna=False, sort=False))

    out_list = []
    n_fixed, n_fallback = 0, 0
    with rasterio.open(mask_path) as src:
        is_water = WaterMask(src, water_value, offraster_is_water=offraster_is_water)

        for group_key, group_df in grouped:
            # Sort on parsed time, not the raw string (e.g. M/D/YYYY sorts wrongly as text).
            t_parsed = pd.to_datetime(group_df[time_col], utc=True, errors="coerce", format="mixed")
            group_df = group_df.assign(_t=t_parsed).sort_values("_t", kind="stable").drop(columns="_t")
            group_df = group_df.copy()

            lon = pd.to_numeric(group_df[lon_col], errors="coerce").to_numpy(dtype=float)
            lat = pd.to_numeric(group_df[lat_col], errors="coerce").to_numpy(dtype=float)
            status = np.zeros(len(group_df), dtype=int)

            land = ~is_water(lon, lat)
            if land.any():
                o_lon, o_lat = _original_position(group_df, lon_col, lat_col)
                smaj = pd.to_numeric(group_df[semi_major_col], errors="coerce").fillna(0.0).to_numpy()
                smin = pd.to_numeric(group_df[semi_minor_col], errors="coerce").fillna(0.0).to_numpy()
                ori = pd.to_numeric(group_df[orientation_col], errors="coerce").fillna(0.0).to_numpy()

                for i in np.where(land)[0]:
                    # Draw all candidates at once and keep the first that lands on water.
                    offs = sample_argos_ellipse_offsets(
                        np.repeat(smaj[i], max_tries), np.repeat(smin[i], max_tries),
                        np.repeat(ori[i], max_tries), rng=rng)
                    c_lon, c_lat = meters_to_degree_offsets(
                        np.repeat(o_lon[i], max_tries), np.repeat(o_lat[i], max_tries),
                        offs[:, 0], offs[:, 1])
                    wet = np.where(is_water(c_lon, c_lat))[0]
                    if len(wet) > 0:
                        j = wet[0]
                        lon[i], lat[i] = c_lon[j], c_lat[j]
                        status[i] = 1
                        n_fixed += 1
                    elif i > 0:
                        lon[i], lat[i] = lon[i - 1], lat[i - 1]
                        status[i] = 2
                        n_fallback += 1
                    else:
                        status[i] = 3

            group_df[lon_col] = lon
            group_df[lat_col] = lat
            group_df["land_jitter_status"] = status
            out_list.append(group_df)

    print(f"Land re-jitter: {n_fixed} fixes moved to water, {n_fallback} fell back to previous fix.")
    out_final = pd.concat(out_list, ignore_index=True)
    return out_final


def process_csv_with_land_mask(yml_conf):

    movement_csvs = yml_conf["movement_csvs"]

    for i in range(len(movement_csvs)):

        # 1. Read the input CSV
        df = pd.read_csv(movement_csvs[i])

        # 2. Apply the jittering function to adjust points off land
        df_updated = jitter_path_off_land(
            path_df=df,
            mask_path=yml_conf["water_mask"],
            group_cols=["uid"],
            lon_col="longitude",
            lat_col="latitude",
            semi_major_col="semi_major",
            semi_minor_col="semi_minor",
            orientation_col="err_orient",
            id_col="uid",
            water_value=yml_conf["water_value"],  # 1
            max_tries=yml_conf.get("max_tries", 100),
            offraster_is_water=yml_conf.get("offraster_is_water", False),
        )

        # 3. Write the updated DataFrame next to the input CSV
        df_updated.to_csv(os.path.splitext(movement_csvs[i])[0] + ".land_jittered.csv", index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    # Translate config to dictionary
    yml_conf = read_yaml(args.yaml)
    process_csv_with_land_mask(yml_conf)
