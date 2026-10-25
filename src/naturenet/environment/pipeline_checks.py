"""
Cheap consistency checks run between pipeline steps.

Each check raises ``PipelineCheckError`` (an AssertionError subclass) with a
message that says which step produced the bad output. They are plain functions
so they can also be called from notebooks or tests.
"""
import numpy as np

INVALID_STATE = -1


class PipelineCheckError(AssertionError):
    pass


def _fail(msg):
    raise PipelineCheckError(msg)


def _dense(trans_prob):
    return trans_prob.todense() if hasattr(trans_prob, "todense") else np.asarray(trans_prob)


def check_trans_prob(trans_prob, n_actions=None, atol=1e-6, require_full=False, name="trans_prob"):
    """
    T[s, a, :] must sum to 1 for every observed (s, a) (and for every (s, a) if
    ``require_full``). Works for dense arrays and sparse.COO.
    """
    if trans_prob.ndim != 3 or trans_prob.shape[0] != trans_prob.shape[2]:
        _fail(f"{name}: expected shape (S, A, S), got {trans_prob.shape}")
    if n_actions is not None and trans_prob.shape[1] != n_actions:
        _fail(f"{name}: expected {n_actions} actions, got {trans_prob.shape[1]}")
    if hasattr(trans_prob, "coords"):  # sparse.COO: sum over nnz only (memory ~ nnz, not S*A)
        data = np.asarray(trans_prob.data)
        if np.any(data < -atol) or not np.all(np.isfinite(data)):
            _fail(f"{name}: negative or non-finite probabilities")
        S, A, _ = trans_prob.shape
        keys = trans_prob.coords[0].astype(np.int64) * A + trans_prob.coords[1]
        uniq, inv = np.unique(keys, return_inverse=True)
        sums = np.bincount(inv, weights=data)
        bad = ~np.isclose(sums, 1.0, atol=atol)
        if np.any(bad):
            bk = uniq[bad][:5]
            _fail(f"{name}: T[s,a,:] must sum to 1 for observed (s,a); first bad (s,a): "
                  f"{[(int(k // A), int(k % A)) for k in bk]}, sums: {sums[bad][:5].tolist()}")
        if require_full and len(uniq) < S * A:
            _fail(f"{name}: {S * A - len(uniq)} (s,a) rows are all zero; fill unobserved rows before SWIRL")
        return True
    else:
        T = np.asarray(trans_prob)
        if np.any(T < -atol) or not np.all(np.isfinite(T)):
            _fail(f"{name}: negative or non-finite probabilities")
        rs = T.sum(-1)
    observed = rs > 0
    if not np.allclose(rs[observed], 1.0, atol=atol):
        bad = np.argwhere(observed & ~np.isclose(rs, 1.0, atol=atol))[:5]
        _fail(f"{name}: T[s,a,:] must sum to 1 for observed (s,a); first bad (s,a): {bad.tolist()}, "
              f"sums: {[float(rs[tuple(b)]) for b in bad]}")
    if require_full and not observed.all():
        n_missing = int((~observed).sum())
        _fail(f"{name}: {n_missing} (s,a) rows are all zero; fill unobserved rows before SWIRL")
    return True


def check_segment_alignment(states, actions, name="segment"):
    """One trajectory: len(states) == len(actions) + 1, actions[k] : states[k] -> states[k+1]."""
    if len(states) != len(actions) + 1:
        _fail(f"{name}: expected len(states) == len(actions) + 1, got {len(states)} states and "
              f"{len(actions)} actions. Re-run the preprocessing step that wrote these files "
              f"(old outputs dropped the initial state).")
    return True


def check_swirl_inputs(states, actions, n_states, n_actions=9, name="SWIRL input"):
    """Batched (N, T) states and (N, T-1) actions before one-hot encoding."""
    states = np.asarray(states)
    actions = np.asarray(actions)
    if states.ndim != 2 or actions.ndim != 2:
        _fail(f"{name}: expected 2D arrays, got states {states.shape}, actions {actions.shape}")
    if states.shape[1] != actions.shape[1] + 1:
        _fail(f"{name}: states.shape[1] must equal actions.shape[1] + 1, got {states.shape}, {actions.shape}")
    if states.min() < 0 or states.max() >= n_states:
        _fail(f"{name}: states must be in [0, {n_states}), got [{states.min()}, {states.max()}]")
    if actions.min() < 0 or actions.max() >= n_actions:
        _fail(f"{name}: actions must be in [0, {n_actions}), got [{actions.min()}, {actions.max()}]")
    return True


def check_env_grid_alignment(env_map, grid, name="env map"):
    """Env map (H, W) must match the grid it will be indexed with."""
    shp = np.shape(env_map)[-2:]
    if tuple(shp) != (grid.lat_tiles, grid.lon_tiles):
        _fail(f"{name}: shape {tuple(shp)} != grid (lat_tiles, lon_tiles) = {(grid.lat_tiles, grid.lon_tiles)}")
    return True


def check_folds(groups, train_idx, test_idx, name="fold"):
    """No animal (base uid) may appear in both train and test."""
    groups = np.asarray(groups)
    overlap = set(groups[train_idx]).intersection(groups[test_idx])
    if overlap:
        _fail(f"{name}: animals in both train and test: {sorted(overlap)[:10]}")
    if len(test_idx) == 0 or len(train_idx) == 0:
        _fail(f"{name}: empty train or test split")
    return True


def check_s2_prev_slots(n_prev_slots, n_states, n_actions, max_dense_entries):
    entries = (n_states * n_prev_slots) ** 2 * n_actions
    if entries > max_dense_entries:
        _fail(f"S-2 augmented transition tensor would have {entries:.3g} entries "
              f"({n_states} states x {n_prev_slots} predecessor slots). Raise s2_max_dense_entries, "
              f"coarsen the state space, or set run_s2: False.")
    return True


# ---------------------------------------------------------------------------
# Viz / analysis / NN-SWIRL stage checks
# ---------------------------------------------------------------------------

def check_input_features(feats, n_fixes, name="input features"):
    """gather_input -> plotting: float32 (n_fixes, C), no +-inf (NaN = missing is allowed)."""
    feats = np.asarray(feats)
    if feats.ndim != 2 or feats.shape[0] != n_fixes:
        _fail(f"{name}: shape {feats.shape}, expected ({n_fixes}, C)")
    if feats.dtype != np.float32:
        _fail(f"{name}: dtype {feats.dtype}, expected float32")
    if np.isinf(feats).any():
        _fail(f"{name}: contains inf (float16 overflow or bad nodata)")
    return True


def check_grid_indices(rows, cols, shape, name="grid indices"):
    """All sampled (row, col) are inside an (H, W) map (negative indices would wrap around)."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    H, W = shape[-2:]
    if rows.size and (rows.min() < 0 or cols.min() < 0 or rows.max() >= H or cols.max() >= W):
        _fail(f"{name}: indices outside map {H}x{W}: rows [{rows.min()}, {rows.max()}], "
              f"cols [{cols.min()}, {cols.max()}]")
    return True


def check_latent_file(zs, seq_key, path_lens=None, name="LL file"):
    """SWIRL outputs -> plotting: one sequence id per latent row; latent length = len(states) - 2."""
    zs = np.asarray(zs)
    if seq_key is None:
        _fail(f"{name}: no seq_key array; re-run SWIRL with the current code (LL files now store sequence ids)")
    if len(seq_key) != zs.shape[0]:
        _fail(f"{name}: {len(seq_key)} sequence ids for {zs.shape[0]} latent paths")
    if path_lens is not None:
        bad = [i for i, L in enumerate(path_lens) if L - 2 != zs.shape[1]]
        if bad:
            _fail(f"{name}: latent length {zs.shape[1]} != len(path) - 2 for rows {bad[:5]}")
    return True


def check_series_columns(columns, env_col="env", name="time series"):
    """Before export: equal column lengths, and at least one column differs from env (copy-paste guard)."""
    lens = {k: len(v) for k, v in columns.items()}
    if len(set(lens.values())) > 1:
        _fail(f"{name}: unequal column lengths {lens}")
    if env_col in columns:
        env = np.asarray(columns[env_col], dtype=float)
        others = [k for k in columns if k not in (env_col, "latent", "Time", "month", "uid", "segment")]
        same = [k for k in others if np.array_equal(np.asarray(columns[k], dtype=float), env, equal_nan=True)]
        if others and len(same) == len(others) and len(env) > 1:
            _fail(f"{name}: every exported column equals {env_col}: {same}")
    return True


def check_turn_angles(theta, name="turn angle"):
    theta = np.asarray(theta, dtype=float)
    fin = theta[np.isfinite(theta)]
    if fin.size and (fin.min() < -np.pi - 1e-9 or fin.max() > np.pi + 1e-9):
        _fail(f"{name}: values outside [-pi, pi]: [{fin.min()}, {fin.max()}]")
    return True


def check_single_series(df, id_cols=("uid", "segment"), name="series"):
    """Before correlation: a series must come from one animal / segment."""
    for c in id_cols:
        if c in df and df[c].nunique(dropna=False) > 1:
            _fail(f"{name}: spans {df[c].nunique()} values of '{c}'; compute statistics per series and pool")
    return True


def check_policy(logpi, atol=1e-6, name="policy"):
    """NN SWIRL: log-policy finite, rows of exp(logpi) sum to 1 over actions (last axis)."""
    logpi = np.asarray(logpi, dtype=float)
    if not np.isfinite(logpi).all():
        _fail(f"{name}: non-finite log-policy entries")
    err = np.abs(np.exp(logpi).sum(-1) - 1).max()
    if err > atol:
        _fail(f"{name}: rows sum to 1 +- {err:.2e}")
    return True


def check_finite(x, name="value"):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        _fail(f"{name}: non-finite values")
    return True
