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


def load_segments(yml_conf):
    """All segments as a list of dicts {group, states, actions} (no length filtering)."""
    act_files = as_list(yml_conf["actions"])
    pos_files = as_list(yml_conf["positions"])
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
    if segment_policy == "exact" and seq_len is None:
        print("  NOTE: exact + seq_len None keeps only clean segments of the MAXIMUM length; "
              "use segment_policy: chunk with seq_len to keep shorter / partly valid segments.")
    return per


def build_dataset(segs, seq_len=None, segment_policy="exact", min_len=3, verbose=True, return_meta=False):
    """
    Fixed-length (N, L) states / (N, L-1) actions for SWIRL.

    segment_policy:
        "exact" - keep valid segments with exactly L states (old behaviour; L defaults to the max length)
        "chunk" - cut every valid run into L-state windows (keeps far more data from short/long tracks)
    Segments containing invalid (-1) states are split ("chunk") or dropped ("exact").
    return_meta=True also returns one dict per sequence: {key, seg, start} = file key (incl. __aug),
    segment index in that file and offset (in states) of the sequence inside the segment.
    """
    runs = []
    for sg in segs:
        m = {"key": sg.get("key", sg["group"]), "seg": int(sg.get("seg", -1))}
        if segment_policy == "exact":
            if np.all(sg["states"] != INVALID_STATE):
                runs.append((sg["group"], sg["states"], sg["actions"], dict(m, start=0)))
        else:
            for s, a, st in _valid_runs(sg["states"], sg["actions"]):
                runs.append((sg["group"], s, a, dict(m, start=int(st))))
    if verbose or not runs:
        drop_report(segs, segment_policy, seq_len)
    if not runs:
        raise ValueError("No valid (fully observed) segments: every segment contains invalid (-1) states. "
                         "See the report above (all -1 = no env scene for those dates / outside the env grid).")
    if seq_len is None:
        seq_len = max(len(r[1]) for r in runs)
    if seq_len < min_len:
        raise ValueError(f"seq_len={seq_len} < {min_len}; SWIRL needs at least 3 states per sequence")

    states, actions, groups, meta = [], [], [], []
    for g, s, a, m in runs:
        if segment_policy == "exact":
            pieces = [(s, a, 0)] if len(s) == seq_len else []
        elif segment_policy == "chunk":
            pieces = _chunk(s, a, seq_len)
        else:
            raise ValueError(f"Unknown segment_policy {segment_policy}")
        for ps, pa, off in pieces:
            states.append(ps)
            actions.append(pa)
            groups.append(g)
            meta.append(dict(m, start=m["start"] + int(off)))
    if not states:
        raise ValueError(f"No sequences of length {seq_len} with segment_policy={segment_policy}")

    states = np.stack(states)
    actions = np.stack(actions)
    groups = np.asarray(groups)

    if verbose:
        tot = {}
        for sg in segs:
            tot[sg["group"]] = tot.get(sg["group"], 0) + len(sg["actions"])
        kept = {}
        for g in groups:
            kept[g] = kept.get(g, 0) + seq_len - 1
        print(f"SWIRL dataset: {len(states)} sequences of {seq_len} states, policy={segment_policy}")
        for g in sorted(tot):
            print(f"  {g}: kept {kept.get(g, 0)}/{tot[g]} transitions ({100.0 * kept.get(g, 0) / max(tot[g], 1):.1f}%)")
        missing = sorted(set(tot) - set(kept))
        if missing:
            print(f"  WARNING: {len(missing)} animals contribute no sequences: {missing[:10]}")
    if return_meta:
        return states, actions, groups, seq_len, meta
    return states, actions, groups, seq_len


def align_for_swirl(states, actions):
    """
    Emission p(a_t | x_t) uses the state where the action was TAKEN:
        x      = states[:, 1:-1]   (x_t,   t = 1..L-2)
        x_prev = states[:, :-2]    (x_t-1, for S-2)
        a      = actions[:, 1:]    (a_t : x_t -> x_t+1)
    (The old code paired a_t with the state reached by a_t.)
    """
    states = np.asarray(states)
    actions = np.asarray(actions)
    if states.shape[1] != actions.shape[1] + 1:
        raise PipelineCheckError(f"states {states.shape} / actions {actions.shape} are not aligned")
    x, x_prev, a = states[:, 1:-1], states[:, :-2], actions[:, 1:]
    assert x.shape == x_prev.shape == a.shape
    return x, x_prev, a


def load_swirl_dataset(yml_conf, verbose=True):
    """Load + filter + check once (hoisted out of the fold loop)."""
    segs = load_segments(yml_conf)
    states, actions, groups, L, meta = build_dataset(segs, yml_conf.get("seq_len"),
                                                     yml_conf.get("segment_policy", "exact"), verbose=verbose,
                                                     return_meta=True)
    n_states = yml_conf.get("n_states")
    if n_states is not None:
        check_swirl_inputs(states, actions, n_states, N_ACTIONS)
    return {"segments": segs, "states": states, "actions": actions, "groups": groups, "seq_len": L, "meta": meta}


def seq_meta_arrays(data, index=None):
    """
    Per-sequence ids to store next to every LL / latent-path file, so plots can map row i of
    jax_path_vmap back to (file key, segment, offset) instead of guessing from file order.
    Latent step t of row i is the state at segment position seq_start + 1 + t (x = states[:, 1:-1]).
    """
    meta = data.get("meta")
    if meta is None:
        return {}
    idx = np.arange(len(meta)) if index is None else np.asarray(index)
    return {"seq_key": np.array([meta[i]["key"] for i in idx]),
            "seq_group": np.asarray(data["groups"])[idx],
            "seq_seg": np.array([meta[i]["seg"] for i in idx], dtype=np.int64),
            "seq_start": np.array([meta[i]["start"] for i in idx], dtype=np.int64)}


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
