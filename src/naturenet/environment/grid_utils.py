
import os
import pickle
import math
import re
from collections import defaultdict

import sparse
import numpy as np
import pandas as pd

from naturenet.environment.pipeline_checks import (
    INVALID_STATE,
    check_env_grid_alignment,
    check_segment_alignment,
    check_trans_prob,
)

N_ACTIONS = 9

# Action codes: 0 = stay, then the 8 neighbours clockwise starting at NW.
#   1 NW   2 N   3 NE
#   8 W    0 .   4 E
#   7 SW   6 S   5 SE
# Keys are (sign(d_row), sign(d_col)); rows grow southward, cols grow eastward.
ACTION_FROM_STEP = {
    (0, 0): 0,
    (-1, -1): 1, (-1, 0): 2, (-1, 1): 3,
    (0, 1): 4,
    (1, 1): 5, (1, 0): 6, (1, -1): 7,
    (0, -1): 8,
}
_ACTION_LUT = np.zeros((3, 3), dtype=np.int64)
for (_dr, _dc), _a in ACTION_FROM_STEP.items():
    _ACTION_LUT[_dr + 1, _dc + 1] = _a


class Grid(object):
    def __init__(self, min_lon, min_lat, max_lon, max_lat, grid_res_lon_deg, grid_res_lat_deg):
        self.min_lon = min_lon
        self.min_lat = min_lat

        self.max_lon = max_lon
        self.max_lat = max_lat

        self.grid_res_lon_deg = grid_res_lon_deg
        self.grid_res_lat_deg = grid_res_lat_deg

        self.lon_size = (max_lon - min_lon) / self.grid_res_lon_deg
        self.lat_size = (max_lat - min_lat) / self.grid_res_lat_deg

        self.lon_tiles = math.ceil(self.lon_size)
        self.lat_tiles = math.ceil(self.lat_size)

        print(self.lon_tiles, self.lat_tiles)


def grid_from_shape(ref_grid, n_rows, n_cols, lon_bounds=None, lat_bounds=None):
    """
    Grid with the same (or given) extent as ``ref_grid`` but a resolution implied
    by an (n_rows, n_cols) raster. Used when the env map is coarser than the
    movement grid. Assumes the raster is north-up (row 0 = max_lat).
    """
    min_lon, max_lon = lon_bounds if lon_bounds is not None else (ref_grid.min_lon, ref_grid.max_lon)
    min_lat, max_lat = lat_bounds if lat_bounds is not None else (ref_grid.min_lat, ref_grid.max_lat)
    g = Grid(min_lon, min_lat, max_lon, max_lat,
             (max_lon - min_lon) / n_cols, (max_lat - min_lat) / n_rows)
    # Guard against ceil() round-off adding a phantom row/col.
    g.lon_tiles, g.lat_tiles = n_cols, n_rows
    return g


def grid_to_lat_lon(grid, row: int, col: int) -> tuple[float, float]:
    """Return the center of a cell in a north-up, top-left-origin grid."""
    if not (0 <= col < grid.lon_tiles and 0 <= row < grid.lat_tiles):
        raise IndexError(
            f"Cell ({col}, {row}) is outside grid dimensions "
            f"({grid.lon_tiles}, {grid.lat_tiles})"
        )

    lon = grid.min_lon + (col + 0.5) * grid.grid_res_lon_deg
    lat = grid.max_lat - (row + 0.5) * grid.grid_res_lat_deg
    return lat, lon


def in_grid_bounds(grid, lat, lon):
    """Vectorised: True where lon/lat lies inside the closed grid extent."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    return (lon >= grid.min_lon) & (lon <= grid.max_lon) & (lat >= grid.min_lat) & (lat <= grid.max_lat)


def lat_lon_to_grid_vec(grid, lat, lon, eps=1e-9):
    """
    Vectorised lat/lon -> (row, col). Points exactly on the east (max_lon) or
    south (min_lat) edge go into the last col/row instead of raising.
    Points outside the extent raise ValueError (filter with ``in_grid_bounds`` first).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    col = np.floor((lon - grid.min_lon) / grid.grid_res_lon_deg + eps).astype(np.int64)
    row = np.floor((grid.max_lat - lat) / grid.grid_res_lat_deg + eps).astype(np.int64)
    # Closed outer edges
    col = np.where((col == grid.lon_tiles) & np.isclose(lon, grid.max_lon), grid.lon_tiles - 1, col)
    row = np.where((row == grid.lat_tiles) & np.isclose(lat, grid.min_lat), grid.lat_tiles - 1, row)
    bad = (col < 0) | (col >= grid.lon_tiles) | (row < 0) | (row >= grid.lat_tiles) | ~np.isfinite(lat) | ~np.isfinite(lon)
    if np.any(bad):
        i = int(np.argmax(bad))
        raise ValueError(
            f"Coordinate ({np.ravel(lon)[i]}, {np.ravel(lat)[i]}) lies outside the grid extent: "
            f"[{grid.min_lon}, {grid.min_lon + grid.lon_tiles * grid.grid_res_lon_deg}] x "
            f"[{grid.max_lat - grid.lat_tiles * grid.grid_res_lat_deg}, {grid.max_lat}]"
        )
    return row, col


def lat_lon_to_grid(grid, lat: float, lon: float) -> tuple[int, int]:
    """Return the zero-based (row, col) of the cell containing lon/lat.

    Assumes:
      - min_lon is the west outer grid edge
      - max_lat is the north outer grid edge
      - col increases west -> east
      - row increases north -> south
      - both resolutions are positive degree magnitudes
    Fixes exactly on max_lon / min_lat are put in the last col / row.
    """
    row, col = lat_lon_to_grid_vec(grid, [lat], [lon])
    return int(row[0]), int(col[0])


def grid_to_ind(grid, row, col):
    """Convert zero-based (row, col) to a C-order flat index (scalars or arrays)."""

    width = grid.lon_tiles
    return row * width + col


def ind_to_grid(index: int, width: int) -> tuple[int, int]:
    """Convert a C-order flat index to zero-based (row, col)."""
    row, col = divmod(index, width)
    return row, col


def distance_grid(grid, row, col, deg_grid=None):
    """
    Euclidean (degree) distance from cell (row, col) to every cell. Vectorised and
    only computed on request; it was previously built per fix in pure Python and
    never saved.
    """
    lat, lon = grid_to_lat_lon(grid, row, col)
    if deg_grid is None:
        lats = grid.max_lat - (np.arange(grid.lat_tiles) + 0.5) * grid.grid_res_lat_deg
        lons = grid.min_lon + (np.arange(grid.lon_tiles) + 0.5) * grid.grid_res_lon_deg
        return np.sqrt((lats[:, None] - lat) ** 2 + (lons[None, :] - lon) ** 2)
    return np.sqrt((deg_grid[..., 0] - lat) ** 2 + (deg_grid[..., 1] - lon) ** 2)


def _fix_swapped_lat_lon(grid, lat, lon):
    """
    Swap lat/lon for rows where 'latitude' is a valid longitude but not a valid
    latitude. Only meaningful when the grid bounds are fixed (lon_bounds/lat_bounds),
    because data-derived bounds already include the swapped points.
    """
    lat = np.asarray(lat, dtype=float).copy()
    lon = np.asarray(lon, dtype=float).copy()
    swap = (lat >= grid.min_lon) & (lat <= grid.max_lon) & ((lat > grid.max_lat) | (lat < grid.min_lat))
    if np.any(swap):
        print(f"WARNING: swapping lat/lon for {int(swap.sum())} fixes")
        lat[swap], lon[swap] = lon[swap], lat[swap].copy()
    return lat, lon


def df_to_cells(movement_df, grid):
    lat, lon = _fix_swapped_lat_lon(grid, movement_df["latitude"].to_numpy(), movement_df["longitude"].to_numpy())
    return lat_lon_to_grid_vec(grid, lat, lon)


def compute_actions(rows, cols):
    """Action a[k] for the move cell[k] -> cell[k+1] (length len(rows) - 1)."""
    dr = np.sign(np.diff(np.asarray(rows))).astype(np.int64)
    dc = np.sign(np.diff(np.asarray(cols))).astype(np.int64)
    return _ACTION_LUT[dr + 1, dc + 1]


def compute_action(prev_state, current_state):
    """Scalar version of compute_actions, for (row, col) tuples."""
    dr = int(np.sign(current_state[0] - prev_state[0]))
    dc = int(np.sign(current_state[1] - prev_state[1]))
    return ACTION_FROM_STEP[(dr, dc)]


def accumulate_counts(states, actions, counts=None, sa_counts=None, invalid=INVALID_STATE):
    """
    Add transitions states[k] --actions[k]--> states[k+1] to the count dicts.
    ``counts[(s, a, s')]`` and ``sa_counts[(s, a)]``. Transitions touching an
    invalid state are skipped. No extra count is added for the final state.
    """
    counts = defaultdict(int) if counts is None else counts
    sa_counts = defaultdict(int) if sa_counts is None else sa_counts
    states = np.asarray(states)
    actions = np.asarray(actions)
    check_segment_alignment(states, actions)
    for s, a, s2 in zip(states[:-1], actions, states[1:]):
        s, a, s2 = int(s), int(a), int(s2)
        if s == invalid or s2 == invalid:
            continue
        counts[(s, a, s2)] = counts.get((s, a, s2), 0) + 1
        sa_counts[(s, a)] = sa_counts.get((s, a), 0) + 1
    return counts, sa_counts


def counts_to_trans_prob(counts, sa_counts, n_states, n_actions=N_ACTIONS, fill_unobserved=None,
                         fill_states=None, check=True):
    """
    T[s, a, s'] = count(s, a, s') / count(s, a)   (rows sum to 1 for observed (s, a)).

    fill_unobserved:
        None        - leave unobserved (s, a) rows at zero (acts as an implicit absorbing state in VI)
        "self_loop" - T[s, a, s] = 1 for unobserved (s, a), for s in ``fill_states``
                      (default: every state if n_states * n_actions <= 1e6, else states seen in counts)
    """
    coords = [[], [], []]
    data = []
    for (s, a, s2), c in counts.items():
        coords[0].append(s)
        coords[1].append(a)
        coords[2].append(s2)
        data.append(c / sa_counts[(s, a)])

    if fill_unobserved == "self_loop":
        if fill_states is None:
            if n_states * n_actions <= 1_000_000:
                fill_states = range(n_states)
            else:
                fill_states = sorted({k[0] for k in sa_counts} | {k[2] for k in counts})
        for s in fill_states:
            for a in range(n_actions):
                if (s, a) not in sa_counts:
                    coords[0].append(s)
                    coords[1].append(a)
                    coords[2].append(s)
                    data.append(1.0)
    elif fill_unobserved is not None:
        raise ValueError(f"Unknown fill_unobserved: {fill_unobserved}")

    trans_prob = sparse.COO(np.array(coords, dtype=np.int64).reshape(3, -1), np.array(data, dtype=float),
                            shape=(n_states, n_actions, n_states))
    if check:
        check_trans_prob(trans_prob, n_actions=n_actions)
    return trans_prob


def streamline_columns(movement_df, rename_dict = {"lon":"longitude", "lat":"latitude", "TimeValue":"timestamp", "date":"timestamp", "datetime":"timestamp",\
    "animal_id":"uid", "id":"uid", "smaj":"semi_major", "smin":"semi_minor", "eor":"err_orient", "coords_x":"longitude", "coords_y":"latitude"}):

    rnm_dict = {}
    for key in rename_dict.keys():
        if key in movement_df:
            rnm_dict[key] = rename_dict[key]

    movement_df.rename(columns=rnm_dict, inplace=True)

    movement_df["uid"]  = movement_df["uid"].astype(str).str.strip()
    movement_df["timestamp"]  = pd.to_datetime(movement_df["timestamp"],  utc=True, errors="coerce", format="mixed")
    # Sort AFTER parsing, so non-ISO strings sort chronologically.
    movement_df = movement_df.sort_values(['uid', 'timestamp'], ascending=[True, True], kind="stable")
    print(movement_df.head())

    return movement_df

def combine_errs_and_paths(movement_df, err_df):


    # Inner join: keep only rows where both id AND date match in df1 and df2
    merged = pd.merge(
        movement_df,
        err_df,
        on=["uid", "timestamp", "longitude", "latitude"],   # two-key match
        how="inner",
        #suffixes=("_df1", "_df2")
    )

    #merged["longitude"] = merged["longitude_x"]
    #merged["latitude"] = merged["latitude_x"]
    return merged


def gather_input_features(movement_df, feature_grid, grid, env_grid=None, flip_lat=False):
    """
    Feature vector at each fix: float32 array (len(movement_df), C), one row per fix INCLUDING the
    initial fix (same length as the abstract-state and pixel paths). Rows are NaN where the fix has no
    feature map or lies outside the grid.

    feature_grid[t] is the (C, H, W) or (H, W) feature scene for fix t (or None). The scenes are on the
    env grid, so pass ``env_grid`` (from movement_preprocess_pipeline._env_grid_for) when it differs from
    the movement grid; ``flip_lat`` flips south-up rasters. Values are gathered first and then cast, so
    the full scene is never converted (the old float16 copy per fix overflowed above 65504 and quantised
    SST to 0.25 K).
    """
    g = env_grid if env_grid is not None else grid
    lat, lon = _fix_swapped_lat_lon(grid, movement_df["latitude"].to_numpy(), movement_df["longitude"].to_numpy())
    n = len(movement_df)
    ok = in_grid_bounds(g, lat, lon) & np.isfinite(lat) & np.isfinite(lon)
    rows = np.zeros(n, dtype=np.int64)
    cols = np.zeros(n, dtype=np.int64)
    if ok.any():
        rows[ok], cols[ok] = lat_lon_to_grid_vec(g, lat[ok], lon[ok])

    n_feat = None
    out = [None] * n
    for t in range(n):
        fg = feature_grid[t]
        if fg is None:
            continue
        fg = np.asarray(fg)
        if fg.ndim == 2:
            fg = fg[None]
        if fg.shape[-2:] != (g.lat_tiles, g.lon_tiles):
            raise ValueError(f"feature scene t={t} has shape {fg.shape[-2:]}, grid is {(g.lat_tiles, g.lon_tiles)}")
        if n_feat is None:
            n_feat = fg.shape[0]
        elif fg.shape[0] != n_feat:
            raise ValueError(f"feature scene t={t} has {fg.shape[0]} channels, expected {n_feat}")
        if not ok[t]:
            continue
        r = g.lat_tiles - 1 - rows[t] if flip_lat else rows[t]
        out[t] = fg[:, r, cols[t]].astype(np.float32)
    if n_feat is None:
        return np.full((n, 0), np.nan, dtype=np.float32)
    feats = np.full((n, n_feat), np.nan, dtype=np.float32)
    for t, v in enumerate(out):
        if v is not None:
            feats[t] = v
    if (~ok).any():
        print(f"WARNING: {int((~ok).sum())}/{n} fixes outside the feature grid; set to NaN")
    return feats


def _lookup_abstract_states(movement_df, abstract_grid, grid, env_grid=None):
    """
    Abstract (cluster) state at every fix. Invalid / missing -> INVALID_STATE.
    ``abstract_grid[t]`` is the 2D cluster map for fix t, or None.
    """
    lat, lon = _fix_swapped_lat_lon(grid, movement_df["latitude"].to_numpy(), movement_df["longitude"].to_numpy())
    states = np.full(len(movement_df), INVALID_STATE, dtype=np.int64)
    for t in range(len(movement_df)):
        amap = abstract_grid[t]
        if amap is None:
            continue
        amap = np.asarray(amap)
        if amap.ndim != 2:
            raise ValueError(f"abstract map for fix {t} must be 2D (H, W), got {amap.shape}")
        g = env_grid if env_grid is not None else grid
        check_env_grid_alignment(amap, g, name=f"abstract map t={t}")
        r, c = lat_lon_to_grid_vec(g, [lat[t]], [lon[t]])
        v = amap[r[0], c[0]]
        if np.isfinite(v) and v >= 0:
            states[t] = int(v)
    return states


def compute_transition_probs_abstracted_env(movement_df, abstract_grid, actions, grid, n_clusters, total_count=None,
                                            points=None, env_grid=None, fill_unobserved="self_loop"):
    """
    Abstract-env transitions.

    Returns
    -------
    total_count : dict (s, a) -> count     (pooled across calls)
    points      : dict (s, a, s') -> count (pooled across calls)
    trans_prob  : sparse.COO (n_clusters, 9, n_clusters), rows normalised by count(s, a)
    states      : np.ndarray, length len(movement_df), INCLUDING the initial state.
                  actions[k] is the move states[k] -> states[k+1]. INVALID_STATE (-1)
                  marks fixes with no env map / land / nodata.
    """
    if len(actions) != len(movement_df) - 1:
        raise ValueError(f"Expected {len(movement_df) - 1} actions for {len(movement_df)} fixes, got {len(actions)}")

    states = _lookup_abstract_states(movement_df, abstract_grid, grid, env_grid=env_grid)
    if states.max(initial=-1) >= n_clusters:
        raise ValueError(f"State id {states.max()} >= n_clusters ({n_clusters}). Increase n_clusters.")

    points, total_count = accumulate_counts(states, actions, counts=points, sa_counts=total_count)
    trans_prob = counts_to_trans_prob(points, total_count, n_clusters, N_ACTIONS, fill_unobserved=fill_unobserved)
    return total_count, points, trans_prob, states


#Not requiring adjacency for now, as animals can cross larger distances in some cases
#   will grid based on stats on movement distances, etc.

def compute_transition_probabilities(movement_df, grid, total_count=None, points=None, return_distance=False,
                                     build_trans_prob=False):
    """
    Pixel-grid transitions for one segment.

    Returns (total_count, points, trans_prob, actions, single_ind_positions, distance)
      - points / total_count: pooled count dicts keyed (s, a, s') / (s, a) on flat indices
      - actions: length T-1, actions[k] is the move fix k -> fix k+1
      - single_ind_positions: length T flat indices (includes the initial fix)
      - trans_prob: only built if ``build_trans_prob`` (normally built once at the end
        from the pooled counts; rebuilding the COO for every segment is O(N^2))
      - distance: [] unless ``return_distance`` (vectorised, memory heavy)
    """
    rows, cols = df_to_cells(movement_df, grid)
    actions = compute_actions(rows, cols)
    single_ind_positions = grid_to_ind(grid, rows, cols).astype(np.int64)

    points, total_count = accumulate_counts(single_ind_positions, actions, counts=points, sa_counts=total_count)

    trans_prob = None
    if build_trans_prob:
        trans_prob = counts_to_trans_prob(points, total_count, grid.lon_tiles * grid.lat_tiles, N_ACTIONS)

    distance = []
    if return_distance:
        distance = [distance_grid(grid, r, c) for r, c in zip(rows, cols)]

    return total_count, points, trans_prob, actions.tolist(), single_ind_positions.tolist(), distance


#For now, split - later, can investigate use of loss functions that account for missing data:
#https://arxiv.org/pdf/1911.06930

def _segment_track(track_df, time_col="timestamp", max_gap_days=2.0, max_len=49, min_len=2, overlap=True):
    """
    Split one animal's time-sorted track into segments:
      - break where the gap between fixes is > ``max_gap_days`` (strict, not rounded)
      - cap segment length at ``max_len`` fixes; with ``overlap`` the next chunk starts
        at the last fix of the previous one, so no transition is lost at the cap
      - keep segments with >= ``min_len`` fixes; the final fix of the track is kept
    """
    t = pd.to_datetime(track_df[time_col], utc=True, errors="coerce", format="mixed")
    keep = t.notna().to_numpy()
    track_df = track_df.loc[keep]
    t = t[keep]
    order = np.argsort(t.to_numpy(), kind="stable")
    track_df = track_df.iloc[order]
    t = t.iloc[order]
    if len(track_df) == 0:
        return []
    gaps = t.diff().dt.total_seconds().to_numpy() / 86400.0
    breaks = np.where(gaps > max_gap_days)[0]  # index i: gap between i-1 and i
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(track_df)]])

    segs = []
    step = max_len - 1 if overlap else max_len
    for s, e in zip(starts, ends):
        i = s
        while i < e:
            j = min(i + max_len, e)
            if j - i >= min_len:
                segs.append(track_df.iloc[i:j])
            if j >= e:
                break
            i = i + step
    return segs


def _filter_to_bounds(movement_df, bounds):
    if bounds is None:
        return movement_df
    (min_lon, max_lon), (min_lat, max_lat) = bounds
    lat = pd.to_numeric(movement_df["latitude"], errors="coerce")
    lon = pd.to_numeric(movement_df["longitude"], errors="coerce")
    inside = (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)
    n_out = int((~inside).sum())
    if n_out:
        print(f"Dropping {n_out} fixes outside lon {min_lon, max_lon} / lat {min_lat, max_lat} "
              f"before segmenting (gaps they leave are handled by the time-gap split)")
    return movement_df.loc[inside]


def split_pre_split_streams(dfs_by_id, ids, out_dir, run_uid, max_gap_days=2.0, max_len=29, min_len=2,
                            overlap=True, bounds=None):

    final_dfs = {}
    for i in range(len(dfs_by_id)):
        df = _filter_to_bounds(dfs_by_id[i], bounds)
        final_dfs[ids[i]] = _segment_track(df, max_gap_days=max_gap_days, max_len=max_len,
                                           min_len=min_len, overlap=overlap)
        print(ids[i], len(final_dfs[ids[i]]), "segments")

    with open(os.path.join(out_dir, run_uid + '_dfs.pkl'), 'wb') as f:
         pickle.dump(final_dfs, f, protocol=pickle.HIGHEST_PROTOCOL)

    return final_dfs


def split_movement_streams(movement_df, out_dir, run_uid, max_gap_days=2.0, max_len=49, min_len=2,
                           overlap=True, bounds=None):
    """
    Split every animal's track into segments (see ``_segment_track``).
    Keys are str(uid), so augmented ids ("123__aug1") and numeric ids are handled the same way.
    """
     
    print("SPLITTING Movement streams")
    movement_df = movement_df.copy()
    movement_df["uid"] = movement_df["uid"].astype(str).str.strip()
    movement_df = _filter_to_bounds(movement_df, bounds)

    final_dfs = {}
    n_fix_in, n_fix_kept = 0, 0
    for uid, track in movement_df.groupby("uid", sort=True):
        segs = _segment_track(track, max_gap_days=max_gap_days, max_len=max_len, min_len=min_len, overlap=overlap)
        final_dfs[uid] = segs
        n_fix_in += len(track)
        n_fix_kept += len(pd.concat(segs).index.unique()) if segs else 0
    print(f"Segmented {len(final_dfs)} tracks; {n_fix_kept}/{n_fix_in} fixes are in a segment")

    print("SAVING", os.path.join(out_dir, run_uid + '_dfs.pkl'))
    with open(os.path.join(out_dir, run_uid + '_dfs.pkl'), 'wb') as f:
        pickle.dump(final_dfs, f, protocol=pickle.HIGHEST_PROTOCOL)

    return final_dfs


def base_uid(uid):
    """Animal id without the augmentation suffix: '123__aug4' -> '123'."""
    return re.sub(r"__aug\d+$", "", str(uid))
