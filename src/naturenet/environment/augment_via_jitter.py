import os
import pandas as pd
import numpy as np
import argparse

from sit_fuse.utils import read_yaml

from naturenet.environment.grid_utils import combine_errs_and_paths, streamline_columns
from naturenet.environment.jitter_from_argos import augment_argos_tracks
 

def augment_via_jitter(yml_conf):

    n_augmented_per_group = yml_conf["n_augmented_per_group"]

    for i in range(len(yml_conf["movement_csvs"])):

        if "err_csvs" in yml_conf:

            print("Streamlining and merging error and movement CSVs")

            init_err = pd.read_csv(yml_conf["err_csvs"][i])
            strm_err = streamline_columns(init_err)
            strm_err.to_csv(os.path.splitext(yml_conf["err_csvs"][i])[0] + ".streamlined.csv", index=False)

            init_mvmt = pd.read_csv(yml_conf["movement_csvs"][i])
            strm_mvmt = streamline_columns(init_mvmt) 
            strm_mvmt.to_csv(os.path.splitext(yml_conf["movement_csvs"][i])[0] + ".streamlined.csv", index=False)

            print("Saving merged CSV", os.path.splitext(yml_conf["movement_csvs"][i])[0] + ".Argos_err.csv")
            merged_df = combine_errs_and_paths(strm_mvmt, strm_err)
            merged_df.to_csv(os.path.splitext(yml_conf["movement_csvs"][i])[0] + ".Argos_err.csv", index=False)
        else:
            print("Loading pre-merged", yml_conf["movement_csvs"][i])
            merged_df = pd.read_csv(yml_conf["movement_csvs"][i])
            if yml_conf["streamline"]:
                merged_df = streamline_columns(merged_df)
            merged_df.to_csv(os.path.splitext(yml_conf["movement_csvs"][i])[0] + ".streamlined.csv", index=False)

 
        print("Creating", n_augmented_per_group, "Jittered path augmentations")
        augmented = augment_argos_tracks(
                merged_df,
                group_cols=["uid"],
                n_augmented_per_group=n_augmented_per_group,
                lon_col="longitude",
                lat_col="latitude",
                semi_major_col="semi_major",
                semi_minor_col="semi_minor",
                orientation_col="err_orient",
                id_col="uid",
                random_seed=123,
        )
        augmented.to_csv(os.path.splitext(yml_conf["movement_csvs"][i])[0] + ".augmented.csv", index=False)   
        print("Saving", os.path.splitext(yml_conf["movement_csvs"][i])[0] + ".augmented.csv")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    
    augment_via_jitter(yml_conf)


