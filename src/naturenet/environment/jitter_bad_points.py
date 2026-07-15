
import pandas as pd
import rasterio
import os
import argparse

from osgeo import osr, gdal

import numpy as np
 
from sit_fuse.utils import read_yaml 
from naturenet.environment.jitter_from_argos import jitter_argos_positions

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

):
    """
    Jitter points until they fall on water.
    If a point still falls on land after max_tries, reset it to the previous point.

    Parameters
    ----------
    path_df : pd.DataFrame
        DataFrame containing longitude/latitude columns.
    mask_path : str
        Path to the raster land/water mask.
    jitter_point_fn : callable
        Function like jitter_point_fn(lon, lat, **kwargs) -> (new_lon, new_lat)
    lon_col, lat_col : str
        Coordinate column names.
    water_value : int or float
        Raster value representing water.
    max_tries : int
        Maximum jitter attempts per point.
    jitter_kwargs : dict
        Extra keyword arguments passed to jitter_point_fn.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with adjusted coordinates.
    """
    out = path_df.copy()

    rng = np.random.default_rng(123)

    grouped = [("all_rows", out)] if not group_cols else list(out.groupby(group_cols, dropna=False, sort=False))


    out_list = []
    with rasterio.open(mask_path) as src:

        def is_water(lon, lat):
            val = next(src.sample([(lon, lat)]))[0]
            return val == water_value

 
        for group_key, group_df in grouped:
                group_df = group_df.sort_values(by=["timestamp"], ascending=[True]) 
 
                print(group_key)
                for idx in group_df.index:
                    lon = group_df.at[idx, lon_col]
                    lat = group_df.at[idx, lat_col]

                    original_lon = lon
                    original_lat = lat

                    tries = 0
                    out_row = None
                    if not is_water(lon, lat):
                        while not is_water(lon, lat) and tries < max_tries:
   
                            
                            out_row = jitter_argos_positions(
                                 group_df.loc[idx].to_frame().transpose(),
                                 lon_col=lon_col,
                                 lat_col=lat_col,
                                 semi_major_col=semi_major_col,
                                 semi_minor_col=semi_minor_col,
                                 orientation_col=orientation_col,
                                 rng=rng,
                                 output_offset_cols=True,
                              )
    
                            lon = float(out_row[lon_col].iloc[0])
                            lat = float(out_row[lat_col].iloc[0])
    
                            tries += 1
                        print("TRIES", group_key, tries, idx, is_water(lon, lat))
                        if out_row is not None and is_water(lon, lat):
                            group_df.at[idx, lon_col] = float(out_row[lon_col].iloc[0])
                            group_df.at[idx, lat_col] = float(out_row[lat_col].iloc[0])
                        elif out_row is not None:
                            # Fallback: use the previous point if available
                            if idx != group_df.index[0]:
                                prev_idx = group_df.index[group_df.index.get_loc(idx) - 1]
                                group_df.at[idx, lon_col] = group_df.at[prev_idx, lon_col]
                                group_df.at[idx, lat_col] = group_df.at[prev_idx, lat_col]
                            else:
                                # For the first point, keep the original if no previous point exists
                                group_df.at[idx, lon_col] = original_lon
                                group_df.at[idx, lat_col] = original_lat

                out_list.append(group_df)
 
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
            water_value=yml_conf["water_value"], #1
            max_tries=100,
        )

        # 3. Write the updated DataFrame back to the same CSV (overwrite)
        df_updated.to_csv(os.path.splitext(movement_csvs[i])[0] + ".land_jittered.csv", index=False)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)
    process_csv_with_land_mask(yml_conf)

