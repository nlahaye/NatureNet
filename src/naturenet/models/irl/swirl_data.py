"""
Shared data loading, alignment, grouped K-fold splits and per-fold transition tensors for the
ARHMM / SWIRL scripts (run_arhmm*.py, run_swirl*.py, get_stats*.py).

Input format (written by movement_preprocess_pipeline.reprocess_env_stats_abstract):
    positions file : list of int arrays, one per segment, length T_i, INCLUDING the initial state
                     (-1 = no valid env state)
    actions file   : list of int arrays, one per segment, length T_i - 1;
                     actions[k] is the move states[k] -> states[k+1]
"""

import copy
import glob
import json
import os
import pickle

import numpy as np

from naturenet.environment.grid_utils import N_ACTIONS, accumulate_counts, base_uid, counts_to_trans_prob
from naturenet.environment.pipeline_checks import (
    INVALID_STATE,
    PipelineCheckError,
    check_folds,
    check_swirl_inputs,
    check_trans_prob,
)


def as_list(x):
    """YAML may give a single path string or a list of paths."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001 - legacy np.save'd object arrays
        return list(np.load(path, allow_pickle=True))


def group_from_filename(path, suffix="_actions.pkl"):
    """'<df_run_uid>_<key>_actions.pkl' -> base animal id (augmentation suffix removed)."""
    name = os.path.basename(path)
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return base_uid(name)


def _key_from_filename(path, suffix="_actions.pkl"):
    """'<df_run_uid>_<key>_actions.pkl' -> '<df_run_uid>_<key>' (augmentation suffix kept)."""
    name = os.path.basename(path)
    return name[: -len(suffix)] if name.endswith(suffix) else os.path.splitext(name)[0]


def as_rs(rs):
    """ARHMM MLP params (W1, b1, W2, b2) as a tuple of float arrays (np.savez(dtype=object) round-trips)."""
    return tuple(np.asarray(r, dtype=np.float64) for r in rs)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _valid_runs(states, actions):
    """Split one segment at invalid states into maximal valid runs (states, actions, start)."""
    states = np.asarray(states, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    valid = states != INVALID_STATE
    runs, start = [], None
    for k in range(len(states) + 1):
        if k < len(states) and valid[k]:
            if start is None:
                start = k
        elif start is not None:
            runs.append((states[start:k], actions[start:k - 1], start))
            start = None
    return runs


def _chunk(states, actions, L):
    """Windows of L states (L-1 actions, offset); consecutive windows share their boundary state."""
    out = []
    step = L - 1
    for s0 in range(0, len(states) - L + 1, step):
        out.append((states[s0:s0 + L], actions[s0:s0 + L - 1], s0))
    return out


def _split_variable(states, actions, L):
    """
    One valid run -> pieces of at most L states (L None: no cap). A longer run is cut into the fewest
    near-equal pieces that fit; consecutive pieces share their boundary state (as in _chunk).
    """
    n = len(states)
    if L is None or n <= L:
        return [(states, actions, 0)]
    p = int(np.ceil((n - 1) / (L - 1)))
    cuts = np.round(np.linspace(0, n - 1, p + 1)).astype(int)
    return [(states[a:b + 1], actions[a:b], int(a)) for a, b in zip(cuts[:-1], cuts[1:])]


PAD_ACTION = -1   # aligned action index of a padded step (segment_policy: variable)


def is_variable(data):
    return bool(data.get("variable", False))


def seq_lengths(data):
    """(N,) real states per sequence (= seq_len for the fixed-length policies)."""
    if data.get("lengths") is not None:
        return np.asarray(data["lengths"], dtype=np.int64)
    st = np.asarray(data["states"])
    return np.full(st.shape[0], st.shape[1], dtype=np.int64)


def step_mask(data, offset=1, index=None):
    """
    (N, T) bool, True on real (not padded) model steps. offset = states dropped before step 0
    (1 for the SWIRL alignment x = states[:, 1:-1], 0 for HIAVI "full"); T = L - 1 - offset.
    All True for the fixed-length policies.
    """
    lens = seq_lengths(data)
    if index is not None:
        lens = lens[np.asarray(index)]
    T = np.asarray(data["states"]).shape[1] - 1 - int(offset)
    return np.arange(T)[None, :] < (lens - 1 - int(offset))[:, None]


def _pad_rows(states, actions):
    """Pad at the END to the longest row: states forward-filled with the last real state, actions 0."""
    lens = np.array([len(s) for s in states], dtype=np.int64)
    L = int(lens.max())
    S = np.empty((len(states), L), dtype=np.int64)
    A = np.zeros((len(states), L - 1), dtype=np.int64)
    for i, (s, a) in enumerate(zip(states, actions)):
        S[i, :len(s)] = s
        S[i, len(s):] = s[-1]
        A[i, :len(a)] = a
    return S, A, lens


def input_files(yml_conf):
    """
    (actions files, positions files). actions may be a list or a glob ("<dir>/hammerhead_v1_*_actions.pkl");
    positions may be a list or "auto" (each <...>_actions.pkl paired with <...>_states_abstract_env.pkl).
    """
    acts = yml_conf["actions"]
    if isinstance(acts, str) and any(ch in acts for ch in "*?["):
        acts = sorted(glob.glob(acts))
        if not acts:
            raise FileNotFoundError(f"no files match {yml_conf['actions']}")
    act_files = as_list(acts)
    pos = yml_conf.get("positions", "auto")
    if isinstance(pos, str) and pos == "auto":
        bad = [f for f in act_files if not f.endswith("_actions.pkl")]
        if bad:
            raise ValueError(f"positions: auto needs <...>_actions.pkl files, got {bad[:3]}")
        pos_files = [f[:-len("_actions.pkl")] + "_states_abstract_env.pkl" for f in act_files]
    else:
        pos_files = as_list(pos)
    return act_files, pos_files


def load_segments(yml_conf):
    """All segments as a list of dicts {group, states, actions} (no length filtering)."""
    act_files, pos_files = input_files(yml_conf)
    grp_override = as_list(yml_conf.get("groups"))
    if len(act_files) != len(pos_files):
        raise ValueError(f"{len(act_files)} actions files vs {len(pos_files)} positions files")
    if grp_override and len(grp_override) != len(act_files):
        raise ValueError("'groups' must have one entry per actions file")

    segs = []
    for i, (af, pf) in enumerate(zip(act_files, pos_files)):
        acts = _load_pickle(af)
        poss = _load_pickle(pf)
        if len(acts) != len(poss):
            raise ValueError(f"{af}: {len(acts)} action segments vs {len(poss)} position segments in {pf}")
        grp = str(grp_override[i]) if grp_override else group_from_filename(af)
        key = _key_from_filename(af)
        for j, (a, s) in enumerate(zip(acts, poss)):
            a = np.asarray(a).astype(np.int64).ravel()
            s = np.asarray(s).astype(np.int64).ravel()
            if len(s) == len(a):
                raise PipelineCheckError(
                    f"{pf} segment {j}: len(states) == len(actions) == {len(s)}. This is the legacy format "
                    f"(initial state dropped). Re-run reprocess_env_stats_abstract with the updated code.")
            if len(s) != len(a) + 1:
                raise PipelineCheckError(f"{pf} segment {j}: {len(s)} states vs {len(a)} actions")
            segs.append({"group": grp, "key": key, "seg": j, "states": s, "actions": a})
    if not segs:
        raise ValueError("No segments loaded; check the 'actions' / 'positions' paths")
    return segs


def drop_report(segs, segment_policy="exact", seq_len=None):
    """Per animal: segments, states, fraction invalid (-1), segments that are all -1 / partly -1 / clean."""
    per = {}
    for sg in segs:
        s = np.asarray(sg["states"])
        d = per.setdefault(sg["group"], dict(segs=0, states=0, invalid=0, all_bad=0, some_bad=0, clean=0, lens=[]))
        nb = int((s == INVALID_STATE).sum())
        d["segs"] += 1
        d["states"] += len(s)
        d["invalid"] += nb
        d["all_bad" if nb == len(s) else "some_bad" if nb else "clean"] += 1
        d["lens"].append(len(s))
    print(f"Invalid-state report (policy={segment_policy}, seq_len={seq_len}; -1 = no env scene for the fix date, "
          f"land / nodata, or outside the env grid):")
    for g in sorted(per):
        d = per[g]
        print(f"  {g}: {d['segs']} segs, {d['invalid']}/{d['states']} states invalid "
              f"({100.0 * d['invalid'] / max(d['states'], 1):.1f}%); all-invalid {d['all_bad']}, "
              f"partly {d['some_bad']}, clean {d['clean']}; segment length {min(d['lens'])}..{max(d['lens'])}")
    if segment_policy == "variable":
        print("  variable: every valid run kept as one sequence (runs longer than seq_len split into near-equal "
              "pieces); rows padded at the end and masked.")
    if segment_policy == "exact" and seq_len is None:
        print("  NOTE: exact + seq_len None keeps only clean segments of the MAXIMUM length; "
              "use segment_policy: chunk with seq_len to keep shorter / partly valid segments.")
    return per


def build_dataset(segs, seq_len=None, segment_policy="exact", min_len=3, verbose=True, return_meta=False,
                  min_seq_len=None):
    """
    (N, L) states / (N, L-1) actions for SWIRL.

    segment_policy:
        "exact"    - keep valid segments with exactly L states (old behaviour; L defaults to the max length)
        "chunk"    - cut every valid run into L-state windows (keeps far more data from short/long tracks)
        "variable" - every valid run with >= min_seq_len states is one sequence; runs longer than seq_len
                     (a MAXIMUM here; None = no cap) are split into near-equal pieces sharing their boundary
                     state. Rows are padded at the END to the longest one (states forward-filled, actions 0)
                     and the real lengths are returned in meta[i]["length"] (see step_mask).
    Segments containing invalid (-1) states are split ("chunk", "variable") or dropped ("exact").
    return_meta=True also returns one dict per sequence: {key, seg, start, length} = file key (incl. __aug),
    segment index in that file, offset (in states) of the sequence inside the segment and its real length.
    For "variable" the returned L is the padded width (max length).
    """
    if segment_policy not in ("exact", "chunk", "variable"):
        raise ValueError(f"Unknown segment_policy {segment_policy}")
    variable = segment_policy == "variable"
    min_seq_len = int(min_len if min_seq_len is None else min_seq_len)
    if min_seq_len < min_len:
        raise ValueError(f"min_seq_len={min_seq_len} < {min_len}; SWIRL needs at least 3 states per sequence")
    runs = []
    for sg in segs:
        m = {"key": sg.get("key", sg["group"]), "seg": int(sg.get("seg", -1))}
        if segment_policy == "exact":
            if np.all(sg["states"] != INVALID_STATE):
                runs.append((sg["group"], sg["states"], sg["actions"], dict(m, start=0)))
        else:
            for s, a, st in _valid_runs(sg["states"], sg["actions"]):
                runs.append((sg["group"], s, a, dict(m, start=int(st))))
    if variable:
        runs = [r for r in runs if len(r[1]) >= min_seq_len]
    if verbose or not runs:
        drop_report(segs, segment_policy, seq_len)
    if not runs:
        raise ValueError("No valid (fully observed) segments: every segment contains invalid (-1) states. "
                         "See the report above (all -1 = no env scene for those dates / outside the env grid)."
                         + (f" (variable: no valid run has >= min_seq_len={min_seq_len} states)" if variable else ""))
    if seq_len is None and not variable:
        seq_len = max(len(r[1]) for r in runs)
    if seq_len is not None and seq_len < min_len:
        raise ValueError(f"seq_len={seq_len} < {min_len}; SWIRL needs at least 3 states per sequence")
    if variable and seq_len is not None and seq_len < min_seq_len:
        raise ValueError(f"seq_len={seq_len} < min_seq_len={min_seq_len}")

    states, actions, groups, meta = [], [], [], []
    for g, s, a, m in runs:
        if segment_policy == "exact":
            pieces = [(s, a, 0)] if len(s) == seq_len else []
        elif segment_policy == "chunk":
            pieces = _chunk(s, a, seq_len)
        else:
            pieces = _split_variable(s, a, seq_len)
        for ps, pa, off in pieces:
            assert len(ps) == len(pa) + 1
            states.append(ps)
            actions.append(pa)
            groups.append(g)
            meta.append(dict(m, start=m["start"] + int(off), length=int(len(ps))))
    if not states:
        longest = max(len(r[1]) for r in runs)
        raise ValueError(f"No sequences of length {seq_len} with segment_policy={segment_policy}: the longest valid "
                         f"run has {longest} states. Movement segments are capped at max_segment_len fixes in "
                         "movement preprocessing (default 49); lower seq_len, use segment_policy: variable, or "
                         "re-segment with a larger max_segment_len (null = no cap).")

    if variable:
        states, actions, lens = _pad_rows(states, actions)
        assert lens.min() >= min_seq_len and (seq_len is None or lens.max() <= seq_len)
    else:
        states = np.stack(states)
        actions = np.stack(actions)
        lens = np.full(len(states), states.shape[1], dtype=np.int64)
    groups = np.asarray(groups)
    L = int(states.shape[1])

    if verbose:
        tot = {}
        for sg in segs:
            tot[sg["group"]] = tot.get(sg["group"], 0) + len(sg["actions"])
        kept = {}
        for g, n in zip(groups, lens):
            kept[g] = kept.get(g, 0) + int(n) - 1
        if variable:
            q = np.percentile(lens, [0, 25, 50, 75, 100]).astype(int)
            print(f"SWIRL dataset: {len(states)} sequences, policy=variable (seq_len max={seq_len}, "
                  f"min_seq_len={min_seq_len}); states per sequence min/q25/median/q75/max = {'/'.join(map(str, q))}; "
                  f"padded width {L}, padding {100.0 * (1 - lens.sum() / lens.size / L):.1f}% of the batch")
        else:
            print(f"SWIRL dataset: {len(states)} sequences of {seq_len} states, policy={segment_policy}")
        for g in sorted(tot):
            print(f"  {g}: kept {kept.get(g, 0)}/{tot[g]} transitions ({100.0 * kept.get(g, 0) / max(tot[g], 1):.1f}%)")
        missing = sorted(set(tot) - set(kept))
        if missing:
            print(f"  WARNING: {len(missing)} animals contribute no sequences: {missing[:10]}")
    if return_meta:
        return states, actions, groups, L, meta
    return states, actions, groups, L


def align_for_swirl(states, actions, lengths=None):
    """
    Emission p(a_t | x_t) uses the state where the action was TAKEN:
        x      = states[:, 1:-1]   (x_t,   t = 1..L-2)
        x_prev = states[:, :-2]    (x_t-1, for S-2)
        a      = actions[:, 1:]    (a_t : x_t -> x_t+1)
    (The old code paired a_t with the state reached by a_t.)
    lengths (N,) real states per row (segment_policy: variable): padded steps get a = PAD_ACTION (-1) and
    (x, x_prev) = the row's first real step, so every state / pair lookup stays a valid, observed index;
    the models ignore those steps through the mask (a < 0).
    """
    states = np.asarray(states)
    actions = np.asarray(actions)
    if states.shape[1] != actions.shape[1] + 1:
        raise PipelineCheckError(f"states {states.shape} / actions {actions.shape} are not aligned")
    x, x_prev, a = states[:, 1:-1], states[:, :-2], actions[:, 1:]
    assert x.shape == x_prev.shape == a.shape
    if lengths is not None:
        lengths = np.asarray(lengths)
        if lengths.min() < 3 or lengths.max() > states.shape[1]:
            raise PipelineCheckError(f"sequence lengths must be in [3, {states.shape[1]}]")
        pad = np.arange(x.shape[1])[None, :] >= (lengths - 2)[:, None]
        if pad.any():
            x, x_prev, a = x.copy(), x_prev.copy(), a.copy()
            x[pad] = np.broadcast_to(x[:, :1], x.shape)[pad]
            x_prev[pad] = np.broadcast_to(x_prev[:, :1], x.shape)[pad]
            a[pad] = PAD_ACTION
    return x, x_prev, a


def align_data(data):
    """align_for_swirl on a load_swirl_dataset dict (padding-aware)."""
    return align_for_swirl(data["states"], data["actions"], data.get("lengths") if is_variable(data) else None)


def load_swirl_dataset(yml_conf, verbose=True, allow_variable=False):
    """
    Load + filter + check once (hoisted out of the fold loop).
    segment_policy: variable needs allow_variable=True (the caller must handle padded rows; see step_mask).
    """
    policy = yml_conf.get("segment_policy", "exact")
    if policy == "variable" and not allow_variable:
        raise ValueError("segment_policy: variable (padded, masked sequences) is supported by run_swirl_kfold / "
                         "get_stats / get_stats_2 / run_arhmm_kfold, run_hiavi_kfold / get_stats_hiavi, "
                         "run_swirl_scales_kfold, run_hiavi_multiagent / run_swirl_multiagent / "
                         "run_scales_multiagent and check_inputs. This script needs fixed-length sequences: "
                         "use segment_policy: chunk (or exact).")
    segs = load_segments(yml_conf)
    states, actions, groups, L, meta = build_dataset(segs, yml_conf.get("seq_len"), policy, verbose=verbose,
                                                     return_meta=True, min_seq_len=yml_conf.get("min_seq_len"))
    n_states = yml_conf.get("n_states")
    if n_states is not None:
        check_swirl_inputs(states, actions, n_states, N_ACTIONS)
    lengths = np.array([m["length"] for m in meta], dtype=np.int64)
    return {"segments": segs, "states": states, "actions": actions, "groups": groups, "seq_len": L, "meta": meta,
            "lengths": lengths, "variable": policy == "variable"}


def seq_meta_arrays(data, index=None):
    """
    Per-sequence ids to store next to every LL / latent-path file, so plots can map row i of
    jax_path_vmap back to (file key, segment, offset) instead of guessing from file order.
    Latent step t of row i is the state at segment position seq_start + 1 + t (x = states[:, 1:-1]).
    seq_length = real states in row i: only steps t < seq_length - 2 (SWIRL) are real; later entries of a
    padded (segment_policy: variable) row are padding (latent path -1).
    """
    meta = data.get("meta")
    if meta is None:
        return {}
    idx = np.arange(len(meta)) if index is None else np.asarray(index)
    return {"seq_key": np.array([meta[i]["key"] for i in idx]),
            "seq_group": np.asarray(data["groups"])[idx],
            "seq_seg": np.array([meta[i]["seg"] for i in idx], dtype=np.int64),
            "seq_start": np.array([meta[i]["start"] for i in idx], dtype=np.int64),
            "seq_length": np.array([meta[i].get("length", len(data["states"][i])) for i in idx], dtype=np.int64)}


# ---------------------------------------------------------------------------
# Grouped K-fold
# ---------------------------------------------------------------------------

def make_group_folds(groups, n_folds=5, seed=0):
    """Assign whole animals to folds (shuffled with ``seed``, greedy size balancing)."""
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    if len(uniq) < n_folds:
        raise ValueError(f"{len(uniq)} animals < n_folds={n_folds}; lower n_folds")
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]  # largest first, ties in shuffled order
    fold_groups = [[] for _ in range(n_folds)]
    fold_size = np.zeros(n_folds, dtype=np.int64)
    for i in order:
        k = int(np.argmin(fold_size))
        fold_groups[k].append(str(uniq[i]))
        fold_size[k] += counts[i]
    return fold_groups


def fold_indices(groups, fold_groups):
    groups = np.asarray(groups)
    out = []
    for k, fg in enumerate(fold_groups):
        test = np.where(np.isin(groups, fg))[0]
        train = np.where(~np.isin(groups, fg))[0]
        check_folds(groups, train, test, name=f"fold {k}")
        out.append((train, test))
    return out


def get_folds(yml_conf, groups):
    """
    Grouped folds shared by the ARHMM and SWIRL runs (and by every K / run in get_stats*):
    saved to <folds_dir or out_dir>/folds_<n>_seed<fold_seed>.json and reused if the animals match.
    """
    n_folds = int(yml_conf.get("n_folds", 5))
    fold_seed = int(yml_conf.get("fold_seed", 0))
    fdir = yml_conf.get("folds_dir", yml_conf["out_dir"])
    os.makedirs(fdir, exist_ok=True)
    path = os.path.join(fdir, f"folds_{n_folds}_seed{fold_seed}.json")
    uniq = sorted(set(map(str, groups)))
    fold_groups = None
    if os.path.exists(path):
        with open(path) as f:
            fold_groups = json.load(f)
        if sorted(g for fg in fold_groups for g in fg) != uniq:
            print(f"WARNING: {path} was built for a different set of animals; rebuilding")
            fold_groups = None
    if fold_groups is None:
        fold_groups = make_group_folds(groups, n_folds, fold_seed)
        with open(path, "w") as f:
            json.dump(fold_groups, f, indent=1)
    return fold_indices(groups, fold_groups)


def subsample_index(index, groups, lengths, max_n, seed=0, min_per_group=1):
    """
    Representative subset of ``index`` (sequence rows) with at most ``max_n`` rows, for faster K-fold runs:
      - every animal keeps a share proportional to its sequence count (largest remainder, at least
        ``min_per_group`` while max_n allows), so the per-animal mix of the fold is preserved;
      - within an animal, rows are taken at evenly spaced quantiles of sequence length (ties broken with
        ``seed``), so the length distribution (variable segments) and the time coverage are preserved.
    Deterministic for a given (index, seed). max_n None / <= 0 / >= len(index) -> index unchanged.
    """
    index = np.asarray(index)
    if max_n is None or int(max_n) <= 0 or len(index) <= int(max_n):
        return index
    max_n = int(max_n)
    g = np.asarray(groups)[index]
    L = np.asarray(lengths)[index]
    uniq, counts = np.unique(g, return_counts=True)
    floor_min = min_per_group if max_n >= min_per_group * len(uniq) else 0
    share = counts / counts.sum() * max_n
    alloc = np.minimum(counts, np.maximum(np.floor(share).astype(np.int64), floor_min))
    rem = max_n - alloc.sum()
    order = np.argsort(-(share - np.floor(share)), kind="stable")
    i = 0
    while rem > 0 and i < 10 * len(order):   # hand out the remainder (largest fractional share first)
        j = order[i % len(order)]
        if alloc[j] < counts[j]:
            alloc[j] += 1
            rem -= 1
        i += 1
    while alloc.sum() > max_n:               # floor_min pushed us over: trim the largest allocations
        alloc[np.argmax(alloc)] -= 1
    rng = np.random.RandomState(seed)
    keep = []
    for gi, n_keep in zip(uniq, alloc):
        if n_keep <= 0:
            continue
        rows = np.where(g == gi)[0]
        rows = rows[np.lexsort((rng.permutation(len(rows)), L[rows]))]       # by length, random ties
        pick = np.unique(np.round(np.linspace(0, len(rows) - 1, int(n_keep))).astype(np.int64))
        keep.extend(rows[pick].tolist())
    out = np.sort(index[np.asarray(keep, dtype=np.int64)])
    assert len(out) <= max_n and set(np.asarray(groups)[out]) <= set(uniq)
    return out


def subsample_fold(yml_conf, data, train, test, kf_ind=0):
    """
    Apply yml max_train_seqs / max_test_seqs (null = all) to one fold. Seeded by fold_seed + fold index only,
    so the same rows are used for every K and run in get_stats* (LLs stay comparable across K). Animals stay
    disjoint between train and test (only rows are dropped). T (per_fold_trans_prob) still uses all TRAIN
    animals' segments.
    """
    mtr, mte = yml_conf.get("max_train_seqs"), yml_conf.get("max_test_seqs")
    if not mtr and not mte:
        return train, test
    seed = int(yml_conf.get("fold_seed", 0)) + 1000 * (int(kf_ind) + 1)
    lengths, groups = seq_lengths(data), np.asarray(data["groups"])
    tr = subsample_index(train, groups, lengths, mtr, seed)
    te = subsample_index(test, groups, lengths, mte, seed + 1)
    if len(tr) < len(train) or len(te) < len(test):
        print(f"fold {kf_ind}: subsampled train {len(train)} -> {len(tr)}, test {len(test)} -> {len(te)} "
              f"(per-animal and length stratified)")
    check_folds(groups, tr, te, name=f"fold {kf_ind} (subsampled)")
    return tr, te


def holdout_split(yml_conf, groups):
    """Final-run train/test split by animal. holdout_fraction 0 -> train on everything (test empty)."""
    groups = np.asarray(groups)
    hold = as_list(yml_conf.get("holdout_groups"))
    if not hold:
        frac = float(yml_conf.get("holdout_fraction", 0.2))
        uniq = np.unique(groups)
        n_hold = int(round(frac * len(uniq)))
        if frac > 0 and n_hold == 0:
            n_hold = 1
        rng = np.random.RandomState(int(yml_conf.get("fold_seed", 0)))
        hold = list(rng.permutation(uniq)[:n_hold])
    test = np.where(np.isin(groups, hold))[0]
    train = np.where(~np.isin(groups, hold))[0]
    if len(test):
        check_folds(groups, train, test, name="holdout")
    return train, test, [str(h) for h in hold]


# ---------------------------------------------------------------------------
# Per-fold transitions / rewards
# ---------------------------------------------------------------------------

def trans_prob_from_segments(segs, n_states, n_actions=N_ACTIONS, keep_groups=None, fill_unobserved="self_loop"):
    """
    Empirical T from the given segments (all lengths, invalid transitions skipped). Pass the TRAIN
    animals as ``keep_groups`` so the test animals' transitions do not leak into T.
    """
    counts, sa_counts = {}, {}
    keep = None if keep_groups is None else set(map(str, keep_groups))
    for sg in segs:
        if keep is not None and sg["group"] not in keep:
            continue
        counts, sa_counts = accumulate_counts(sg["states"], sg["actions"], counts=counts, sa_counts=sa_counts)
    T = counts_to_trans_prob(counts, sa_counts, n_states, n_actions, fill_unobserved=fill_unobserved,
                             fill_states=range(n_states) if fill_unobserved else None)
    check_trans_prob(T, n_actions=n_actions, require_full=fill_unobserved is not None, name="per-fold trans_prob")
    return T


def init_rewards(n_hidden, n_states, n_actions, seed):
    """R_start (K, S, A) and the state-only R_start2 (K, S). First two modes start at 0, the rest random."""
    rng = np.random.RandomState(seed)
    R = [np.zeros((n_states, n_actions)) for _ in range(min(n_hidden, 2))]
    for _ in range(2, n_hidden):
        R.append(np.tile(rng.rand(n_states)[:, None], (1, n_actions)))
    R = np.asarray(R)
    return R, R.mean(axis=-1)


# ---------------------------------------------------------------------------
# get_stats loops
# ---------------------------------------------------------------------------

def iter_loop_configs(yml_conf, n_hidden=None, run=None):
    """
    One config per (K, run): K in [min_latent, max_latent] (inclusive), seed = base seed + run.
    ``n_hidden`` / ``run`` restrict the loop (e.g. one job-array task each).
    """
    base_uid_ = yml_conf["run_uid"]
    base_seed = int(yml_conf["seed"])
    for K in range(int(yml_conf["min_latent"]), int(yml_conf["max_latent"]) + 1):
        if n_hidden is not None and K != n_hidden:
            continue
        for i in range(int(yml_conf["num_runs"])):
            if run is not None and i != run:
                continue
            conf = copy.deepcopy(yml_conf)
            conf["n_hidden"] = K
            conf["n_hidden_init"] = K
            conf["seed"] = base_seed + i
            conf["run_uid"] = base_uid_ + "_loop_" + str(i) + "_latent_" + str(K) + "_states"
            yield conf
