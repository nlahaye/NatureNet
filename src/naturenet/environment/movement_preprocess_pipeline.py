
import os
import argparse
import pickle
import sparse
import pandas as pd
from sit_fuse.utils import read_yaml
import numpy as np

from naturenet.environment.grid_utils import Grid, compute_transition_probabilities, split_movement_streams,\
 compute_transition_probs_abstracted_env, lat_lon_to_grid, split_pre_split_streams, streamline_columns, gather_input_features,\
 counts_to_trans_prob, grid_from_shape, N_ACTIONS
from naturenet.environment.pipeline_checks import INVALID_STATE, check_input_features, check_segment_alignment, check_trans_prob


def _bounds_from_conf(yml_conf):
    if "lon_bounds" in yml_conf and "lat_bounds" in yml_conf:
        return (tuple(yml_conf["lon_bounds"]), tuple(yml_conf["lat_bounds"]))
    return None


def _merge_counts(dst, src):
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v
    return dst


def _segment_kwargs(yml_conf, default_max_len):
    return dict(
        max_gap_days=yml_conf.get("max_gap_days", 2.0),
        max_len=yml_conf.get("max_segment_len", default_max_len),
        min_len=yml_conf.get("min_segment_len", 2),
        overlap=yml_conf.get("segment_overlap", True),
        bounds=_bounds_from_conf(yml_conf),
    )


def run_movement_track_preprocess(yml_conf):


    df_uid = yml_conf["df_run_uid"]
    df_dir = yml_conf["df_dir"]

    out_dir = yml_conf["out_dir"]
    skip_uids = yml_conf.get("skip_uids", []) or []

    df_fname = os.path.join(df_dir, df_uid + '_dfs.pkl')

    if "pre_split" in yml_conf and yml_conf["pre_split"]:
        ids = []
        dfs_by_id = []

        for i in range(len(yml_conf["pre_split_movement_csvs"])):
            dfs_by_id.append(streamline_columns(pd.read_csv(yml_conf["pre_split_movement_csvs"][i])))
            ids.append((os.path.splitext(os.path.basename(yml_conf["pre_split_movement_csvs"][i]))[0]).replace(" ", "_"))

        movement_dfs = split_pre_split_streams(dfs_by_id, ids, yml_conf["out_dir"], yml_conf["run_uid"],
                                               **_segment_kwargs(yml_conf, 29))

    elif os.path.exists(df_fname):

        movement_dfs = None
        with open(df_fname, "rb") as f:
            movement_dfs = pd.read_pickle(f)
    else:
        movement_df = pd.read_csv(yml_conf["movement_csv"])
        movement_dfs = split_movement_streams(movement_df, yml_conf["out_dir"], yml_conf["run_uid"],
                                              **_segment_kwargs(yml_conf, 49))

    # One grid shared by every key. Without lon/lat_bounds, use the extent of ALL keys
    # (per-key grids gave inconsistent flat indices in the pooled transition model).
    bounds = _bounds_from_conf(yml_conf)
    if bounds is not None:
        (min_lon, max_lon), (min_lat, max_lat) = bounds
    else:
        all_df = pd.concat([seg for segs in movement_dfs.values() for seg in segs])
        min_lon, max_lon = all_df["longitude"].min(), all_df["longitude"].max()
        min_lat, max_lat = all_df["latitude"].min(), all_df["latitude"].max()
        print("WARNING: lon_bounds/lat_bounds not set; using the extent of all tracks")
    grid = Grid(min_lon, min_lat, max_lon, max_lat, yml_conf["grid_res_lon"], yml_conf["grid_res_lat"])
    n_states = grid.lon_tiles * grid.lat_tiles

    points = {}
    total_count = {}
    for key in movement_dfs.keys():
        key = str(key)
        if key in skip_uids:
            continue

        run_uid = yml_conf["run_uid"] + "_" + key
        tp_fname = os.path.join(yml_conf["out_dir"], run_uid + "_trans_prob.npz")
        key_counts_fname = os.path.join(yml_conf["out_dir"], run_uid + "_key_counts.pkl")
        print("HERE", tp_fname, os.path.exists(tp_fname))
        if os.path.exists(tp_fname) and os.path.exists(key_counts_fname):
            # Resume: still pool this key's counts so the global model is complete.
            with open(key_counts_fname, "rb") as f:
                kc = pickle.load(f)
            _merge_counts(points, kc["points"])
            _merge_counts(total_count, kc["total_count"])
            continue

        movement_sub_df = movement_dfs[key]

        single_unit_coords_full = []
        actions_full = []
        key_points, key_total = {}, {}

        for i in range(len(movement_sub_df)):
            key_total, key_points, _, actions, single_unit_coords, _ = compute_transition_probabilities(
                movement_sub_df[i], grid, total_count=key_total, points=key_points)
            check_segment_alignment(single_unit_coords, actions, name=f"{key} segment {i}")

            single_unit_coords_full.append(single_unit_coords)
            actions_full.append(actions)

        _merge_counts(points, key_points)
        _merge_counts(total_count, key_total)

        # Per-key model is built from THIS key's transitions only (it used to be cumulative).
        trans_prob = counts_to_trans_prob(key_points, key_total, n_states, N_ACTIONS)
        sparse.save_npz(os.path.join(yml_conf["out_dir"], run_uid + "_trans_prob"), trans_prob)

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_grid.pkl"), "wb") as f:
             pickle.dump(grid, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_total_count.pkl"), "wb") as f:
             pickle.dump(key_total, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_points.pkl"), "wb") as f:
             pickle.dump(key_points, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_actions.pkl"), "wb") as f:
             pickle.dump(actions_full, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(yml_conf["out_dir"], run_uid + "_single_unit_coords.pkl"), "wb") as f:
             pickle.dump(single_unit_coords_full, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Written last: its presence marks the key as complete for resume.
        with open(key_counts_fname, "wb") as f:
             pickle.dump({"points": key_points, "total_count": key_total}, f, protocol=pickle.HIGHEST_PROTOCOL)

    trans_prob = counts_to_trans_prob(points, total_count, n_states, N_ACTIONS)
    sparse.save_npz(os.path.join(yml_conf["out_dir"], yml_conf["run_uid"] +  "_trans_prob"), trans_prob)

    with open(os.path.join(yml_conf["out_dir"], yml_conf["run_uid"] + "_grid.pkl"), "wb") as f:
        pickle.dump(grid, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(yml_conf["out_dir"],  yml_conf["run_uid"] +  "_total_count.pkl"), "wb") as f:
        pickle.dump(total_count, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(yml_conf["out_dir"],  yml_conf["run_uid"] +  "_points.pkl"), "wb") as f:
        pickle.dump(points, f, protocol=pickle.HIGHEST_PROTOCOL)


def gen_grid_point_paths(yml_conf):

    movement_dfs = None
    grid = None

    df_uid = yml_conf["df_run_uid"]
    df_dir = yml_conf["df_dir"]

    with open(os.path.join(df_dir, df_uid + '_dfs.pkl'), "rb") as f:
        movement_dfs = pickle.load(f)

    for key in movement_dfs.keys():
        grid = None

        paths = []
        df_uid = yml_conf["df_run_uid"] + "_" + key
        with open(os.path.join(df_dir, df_uid + "_grid.pkl"), "rb") as f:
            grid = pickle.load(f)

        movement_sub_df = movement_dfs[key]
        for i in range(len(movement_sub_df)):
            path = []
            for index, row in movement_sub_df[i].iterrows():
                y,x = lat_lon_to_grid(grid, row["latitude"], row["longitude"])
                path.append({"y":y,"x":x})
            paths.append(path)

        with open(os.path.join(df_dir, yml_conf["df_run_uid"] + "_" + str(key) + "_grid_paths.pkl"), "wb") as f:
            pickle.dump(paths, f, protocol=pickle.HIGHEST_PROTOCOL)


def _env_grid_for(abstract_maps, grid, yml_conf):
    """
    Grid used to index the env maps. If the env map matches the movement grid, use it.
    Otherwise derive a grid with the same extent (or env_lon_bounds / env_lat_bounds)
    and the env map's shape, and say so. Assumes north-up maps; set env_lat_ascending: True
    for south-to-north rasters (they are flipped on load).
    """
    shp = None
    for m in abstract_maps:
        if m is not None:
            shp = np.shape(m)[-2:]
            break
    if shp is None or tuple(shp) == (grid.lat_tiles, grid.lon_tiles):
        return None
    lon_b = yml_conf.get("env_lon_bounds")
    lat_b = yml_conf.get("env_lat_bounds")
    env_grid = grid_from_shape(grid, shp[0], shp[1], lon_bounds=lon_b, lat_bounds=lat_b)
    print(f"Env map shape {tuple(shp)} != movement grid {(grid.lat_tiles, grid.lon_tiles)}; indexing env with "
          f"res {env_grid.grid_res_lon_deg:.5f} x {env_grid.grid_res_lat_deg:.5f} deg over the "
          f"{'configured env' if lon_b else 'movement-grid'} extent")
    exp_res = yml_conf.get("grid_res_lon")
    if exp_res is not None and not np.isclose(env_grid.grid_res_lon_deg, exp_res, rtol=0.02):
        print(f"WARNING: derived env lon resolution {env_grid.grid_res_lon_deg:.5f} != grid_res_lon {exp_res}. "
              f"Check that the env raster covers lon_bounds/lat_bounds.")
    return env_grid


def reprocess_env_stats_abstract(yml_conf):
    """
    Abstract-state trajectories + transition model.

    Output per key: list (per segment) of state arrays of length T (INCLUDING the initial
    fix), aligned with <df_run_uid>_<key>_actions.pkl (length T-1).
    Invalid fixes (no env map, land, nodata) are INVALID_STATE (-1) unless
    invalid_state: "absorbing", in which case they map to an extra state n_clusters.
    """

    movement_dfs = None
    grid = None
    abstract_grid = None

    df_uid = yml_conf["df_run_uid"]
    df_dir = yml_conf["df_dir"]

    run_uid = yml_conf["run_uid"]
    out_dir = yml_conf["out_dir"]

    with open(os.path.join(df_dir, df_uid + '_dfs.pkl'), "rb") as f:
        movement_dfs = pd.read_pickle(f)

    with open(os.path.join(out_dir, "final_env_maps_" + run_uid + ".pkl"), "rb") as f: #TODO - _simple* vs not simple
        abstract_grid = pickle.load(f)

    n_clusters = yml_conf["n_clusters"]
    invalid_policy = yml_conf.get("invalid_state", "drop")
    n_states = n_clusters + 1 if invalid_policy == "absorbing" else n_clusters
    flip_lat = yml_conf.get("env_lat_ascending", False)

    total_count = {}
    points = {}
    n_missing_keys = 0
    for key in movement_dfs.keys():
        if key not in abstract_grid:
            print(f"WARNING: no env maps for {key}; skipping (re-run env_data_format if this is an augmented key)")
            n_missing_keys += 1
            continue
        grid = None
        actions = None

        df_uid = yml_conf["df_run_uid"] + "_" + key
        grid_fname = os.path.join(df_dir, df_uid + "_grid.pkl")
        if not os.path.exists(grid_fname):
            grid_fname = os.path.join(df_dir, yml_conf["df_run_uid"] + "_grid.pkl")
        with open(grid_fname, "rb") as f:
            grid = pickle.load(f)

        with open(os.path.join(df_dir, df_uid + "_actions.pkl"), "rb") as f:
            actions = pickle.load(f)

        abstract_grid_sub = abstract_grid[key]
        movement_sub_df = movement_dfs[key]
        if len(abstract_grid_sub) != len(movement_sub_df):
            raise ValueError(f"{key}: {len(abstract_grid_sub)} env segments vs {len(movement_sub_df)} movement segments")

        states_full = []
        n_invalid = 0
        for i in range(len(movement_sub_df)):
            maps = list(abstract_grid_sub[i])
            if len(maps) != len(movement_sub_df[i]):
                raise ValueError(f"{key} segment {i}: {len(maps)} env maps for {len(movement_sub_df[i])} fixes")
            if flip_lat:
                maps = [None if m is None else np.flipud(np.asarray(m)) for m in maps]
            env_grid = _env_grid_for(maps, grid, yml_conf)

            total_count, points, _, states = compute_transition_probs_abstracted_env(
                movement_sub_df[i], maps, actions[i], grid, n_clusters, points=points, total_count=total_count,
                env_grid=env_grid)
            if invalid_policy == "absorbing":
                states = np.where(states == INVALID_STATE, n_clusters, states)
            n_invalid += int((states == INVALID_STATE).sum())
            check_segment_alignment(states, actions[i], name=f"{key} segment {i}")
            states_full.append(states)

        if n_invalid:
            print(f"{key}: {n_invalid} fixes have no valid env state (marked {INVALID_STATE})")

        with open(os.path.join(df_dir, yml_conf["df_run_uid"] + "_" + str(key) + "_states_abstract_env.pkl"), "wb") as f:
            pickle.dump(states_full, f, protocol=pickle.HIGHEST_PROTOCOL)

    if invalid_policy == "absorbing":
        # Re-count with the absorbing state included.
        from naturenet.environment.grid_utils import accumulate_counts
        total_count, points = {}, {}
        for key in movement_dfs.keys():
            fn = os.path.join(df_dir, yml_conf["df_run_uid"] + "_" + str(key) + "_states_abstract_env.pkl")
            if key not in abstract_grid or not os.path.exists(fn):
                continue
            with open(fn, "rb") as f:
                st = pickle.load(f)
            with open(os.path.join(df_dir, yml_conf["df_run_uid"] + "_" + key + "_actions.pkl"), "rb") as f:
                ac = pickle.load(f)
            for s, a in zip(st, ac):
                points, total_count = accumulate_counts(s, a, counts=points, sa_counts=total_count)

    if n_missing_keys:
        print(f"WARNING: {n_missing_keys} keys had no env maps")

    trans_prob = counts_to_trans_prob(points, total_count, n_states, N_ACTIONS,
                                      fill_unobserved=yml_conf.get("fill_unobserved", "self_loop"))
    check_trans_prob(trans_prob, n_actions=N_ACTIONS, name="abstract-env trans_prob")
    sparse.save_npz(os.path.join(df_dir, yml_conf["df_run_uid"] + "_trans_prob_abstract_env"), trans_prob)

    with open(os.path.join(df_dir, yml_conf["df_run_uid"] + "_total_count_abstract_env.pkl"), "wb") as f:
         pickle.dump(total_count, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(df_dir, yml_conf["df_run_uid"] + "_points_abstract_env.pkl"), "wb") as f:
         pickle.dump(points, f, protocol=pickle.HIGHEST_PROTOCOL)


def _feature_pkl(feature_dir, feature_uid, key):
    """env_data_format writes output_uid + uid (no separator); older runs used feature_uid + '_' + uid."""
    cands = [os.path.join(feature_dir, feature_uid + key + ".pkl")]
    if not feature_uid.endswith("_"):
        cands.append(os.path.join(feature_dir, feature_uid + "_" + key + ".pkl"))
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"No feature file for {key}; tried {cands}")


def generate_input_feature_states(yml_conf):
    """
    Raw input features at each fix (gather_input: True).

    Output per key: <df_run_uid>_<key>_states_input.pkl, a list (per segment) of float32 arrays
    (len(segment), C), aligned with the states (len T) and actions (len T-1). NaN = no scene / off-grid /
    land (when env_data_format was run with land_fill: nan). Config: uids (optional; default all keys),
    feature_dir, feature_uid, env_lat_ascending, env_lon_bounds / env_lat_bounds (as the abstract step).
    """
    df_uid = yml_conf["df_run_uid"]
    df_dir = yml_conf["df_dir"]
    uids = yml_conf.get("uids")
    uids = None if uids is None else set(str(u) for u in (uids if isinstance(uids, (list, tuple)) else [uids]))
    feature_dir = yml_conf["feature_dir"]
    feature_uid = yml_conf["feature_uid"]
    flip_lat = yml_conf.get("env_lat_ascending", False)

    with open(os.path.join(df_dir, df_uid + '_dfs.pkl'), "rb") as f:
        movement_dfs = pd.read_pickle(f)

    for key in movement_dfs.keys():
        if uids is not None and key not in uids:
            continue
        fname = _feature_pkl(feature_dir, feature_uid, key)
        with open(fname, "rb") as f:
            feature_grid = pickle.load(f)
        if key not in feature_grid:
            raise KeyError(f"{fname} has no entry for {key} (keys: {list(feature_grid)[:5]})")

        grid_fname = os.path.join(df_dir, df_uid + "_" + key + "_grid.pkl")
        if not os.path.exists(grid_fname):
            grid_fname = os.path.join(df_dir, df_uid + "_grid.pkl")
        with open(grid_fname, "rb") as f:
            grid = pickle.load(f)

        feature_grid_sub = feature_grid[key]
        movement_sub_df = movement_dfs[key]
        if len(feature_grid_sub) != len(movement_sub_df):
            raise ValueError(f"{key}: {len(feature_grid_sub)} feature segments vs {len(movement_sub_df)} movement segments")

        states_full = []
        for i in range(len(movement_sub_df)):
            maps = list(feature_grid_sub[i])
            if len(maps) != len(movement_sub_df[i]):
                raise ValueError(f"{key} segment {i}: {len(maps)} feature maps for {len(movement_sub_df[i])} fixes")
            env_grid = _env_grid_for(maps, grid, yml_conf)
            states = gather_input_features(movement_sub_df[i], maps, grid, env_grid=env_grid, flip_lat=flip_lat)
            check_input_features(states, len(movement_sub_df[i]), name=f"{key} segment {i}")
            states_full.append(states)

        out = os.path.join(df_dir, df_uid + "_" + str(key) + "_states_input.pkl")
        print("SAVING", out)
        with open(out, "wb") as f:
            pickle.dump(states_full, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary
    yml_conf = read_yaml(args.yaml)


    if yml_conf["init_preprocess"]:
        run_movement_track_preprocess(yml_conf)
    elif "gather_input" in yml_conf and yml_conf["gather_input"]:
        generate_input_feature_states(yml_conf)
    else:
        reprocess_env_stats_abstract(yml_conf)
