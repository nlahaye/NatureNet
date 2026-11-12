"""
Scene file names along the shark tracks.

Superseded by naturenet.models.feature_rep.env_data_format; kept as a config-driven wrapper so old commands
still work. Nothing is hardcoded any more (the old version fixed start 2023-11-01, 398 days and the v1 paths).

    python -m naturenet.models.misc.gather_shark_env_input_fnames -y config/v2/environment/sharks_env_fnames_v2.yaml

The YAML takes the env_data_format keys (scene_index_csv or start_time + n_days, scene_file_template, df_uid,
df_dir, out_dir, output_uid, ...). Defaults applied here: {'store_paths': True, 'sit_fuse_output': False, 'output_uid': 'env_maps_fnames_shark'}.
Output: <out_dir>/<output_uid><key>.pkl ({key: [segment -> per-fix scene file name or None]}) and the merged
<out_dir><output_uid><df_uid>.pkl. Fixes are matched by date (the old loop only advanced the date when a scene
file existed, so days after a missing scene were shifted).
"""
from naturenet.models.feature_rep.env_data_format import main

DEFAULTS = {'store_paths': True, 'sit_fuse_output': False, 'output_uid': 'env_maps_fnames_shark'}

if __name__ == "__main__":
    main(defaults=DEFAULTS)
