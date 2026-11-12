"""
SIT-FUSE cluster maps along the shark tracks.

Superseded by naturenet.models.feature_rep.env_data_format; kept as a config-driven wrapper so old commands
still work. Nothing is hardcoded any more (the old version fixed start 2023-11-01, 398 days, COP_SHARK_DEMO and Hammerhead_Out_v1 paths).

    python -m naturenet.models.misc.sit_fuse_data_format_sharks -y config/v2/environment/sharks_env_format_clusters_v2.yaml

The YAML takes the env_data_format keys (scene_index_csv or start_time + n_days, scene_file_template, df_uid,
df_dir, out_dir, output_uid, ...). Defaults applied here: {'sit_fuse_output': True, 'output_uid': 'final_env_maps_', 'land_fill': -1}.
Output: <out_dir>/<output_uid><key>.pkl ({key: [segment -> per-fix cluster map (int16, -1 invalid) or None]}) and the merged
<out_dir><output_uid><df_uid>.pkl. Fixes are matched by date (the old loop only advanced the date when a scene
file existed, so days after a missing scene were shifted).
"""
from naturenet.models.feature_rep.env_data_format import main

DEFAULTS = {'sit_fuse_output': True, 'output_uid': 'final_env_maps_', 'land_fill': -1}

if __name__ == "__main__":
    main(defaults=DEFAULTS)
