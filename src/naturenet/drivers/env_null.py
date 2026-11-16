"""
Test 1 - environment ablation nulls as state "levels".

The movement (actions, kinematics, sequences, folds) is unchanged; only the environment label of every fix is
replaced, so the paired test-LL difference (real level - null level) measures how much the environment explains
beyond the movement itself. Label permutations are NOT a null (every model is invariant to relabelling the
states); these modes break the label <-> location link instead:

  shift     circular shift of each animal's valid labels along its track (offset U[n/4, 3n/4]); keeps the
            animal's own label composition and run structure, breaks where / when. __aug copies share the offset.
  shuffle   within-animal time permutation of the valid labels (keeps composition, breaks runs)
  marginal  iid draws from the pooled occupancy of all animals (keeps only the population frequencies)
  single    one environment state (every valid label -> 0): the "no environment" model, K_env = 1

Invalid (-1) and absorbing (K = land / no data) positions are never touched, so segment splits and the absorbing
state stay where the real environment put them.

Outputs in dst_dir (a level dir such as <levels_dir>/null_shift/):
  <df_run_uid>_<key>_states_abstract_env.pkl   (same format as movement_preprocess_pipeline)
  <df_run_uid>_trans_prob_abstract_env.npz     (pooled T, for holdout fits; K-fold runs build T per fold)
  env_clusters/<output_uid>_class_mapper.pkl   (copy of the source mapper; {0: 0} for single) so sync_v2_configs
                                               sets n_states as for any other level
  null_meta.json                               (mode, seed, source, per-animal offsets)
Movement files are symlinked from source_dir first (link_movement_run, abstract_env products excluded).

    python -m naturenet.drivers.env_null -y config/v2/levels/whales/null_shift/null.yaml
"""
import argparse
import glob
import hashlib
import os
import pickle
import shutil

import numpy as np

from naturenet.drivers.common import atomic_json, base_id, read_yaml

MODES = ("shift", "shuffle", "marginal", "single")
SUFFIX = "_states_abstract_env.pkl"


def _rng(uid, seed):
    h = int(hashlib.sha1(f"{base_id(uid)}|{int(seed)}".encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h)


def n_valid(mapper_path):
    with open(mapper_path, "rb") as f:
        m = pickle.load(f)
    return len({int(v) for v in m.values() if int(v) >= 0})


def null_segments(segs, K, mode, rng, pooled_p=None):
    """segs: list of int arrays (one animal). Returns (new segs, info). Only labels 0..K-1 are changed."""
    segs = [np.asarray(s).astype(np.int64).copy() for s in segs]
    lens = [len(s) for s in segs]
    flat = np.concatenate(segs) if segs else np.zeros(0, np.int64)
    ok = (flat >= 0) & (flat < K)
    vals = flat[ok]
    n = len(vals)
    info = {"n_valid": int(n)}
    if n:
        if mode == "shift":
            off = int(rng.integers(max(1, n // 4), max(2, (3 * n) // 4 + 1))) if n > 1 else 0
            vals = np.roll(vals, off)
            info["offset"] = off
        elif mode == "shuffle":
            vals = vals[rng.permutation(n)]
        elif mode == "marginal":
            vals = rng.choice(K, size=n, p=pooled_p)
        elif mode == "single":
            vals = np.zeros(n, np.int64)
        else:
            raise ValueError(f"mode must be one of {MODES}")
    out = flat.copy()
    out[ok] = vals
    if mode == "single":
        out[flat == K] = 1                     # absorbing K -> 1 (K_env = 1, n_states = 2)
    res, i = [], 0
    for L, s in zip(lens, segs):
        res.append(out[i:i + L].astype(s.dtype if s.dtype.kind == "i" else np.int64))
        i += L
    return res, info


def _link(conf):
    from naturenet.environment.link_movement_run import link_run
    link_run(dict(src_dir=conf["source_dir"], dst_dir=conf["dst_dir"], df_run_uid=conf["df_run_uid"],
                  exclude=["*_states_abstract_env.pkl", "*_trans_prob_abstract_env.npz",
                           "*_total_count_abstract_env.pkl", "*_points_abstract_env.pkl"],
                  copy_dirs=[], dry_run=False))


def run(conf):
    mode = conf.get("mode", "shift")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    src, dst, uid = conf["source_dir"], conf["dst_dir"], conf["df_run_uid"]
    seed = int(conf.get("seed", 0))
    os.makedirs(dst, exist_ok=True)
    if conf.get("link", True):
        _link(conf)
    mapper = conf.get("source_mapper") or os.path.join(src, "env_clusters",
                                                       conf.get("output_uid", "final_env_maps_") + "_class_mapper.pkl")
    K = n_valid(mapper)
    files = sorted(glob.glob(os.path.join(src, f"{uid}_*{SUFFIX}")))
    files = [f for f in files if not os.path.basename(f).startswith(f"{uid}_trans_prob")]
    if not files:
        raise FileNotFoundError(f"no {uid}_*{SUFFIX} in {src}")
    data = {os.path.basename(f)[len(uid) + 1:-len(SUFFIX)]: list(pickle.load(open(f, "rb"))) for f in files}
    pooled = np.zeros(K)
    for segs in data.values():
        for s in segs:
            s = np.asarray(s)
            pooled += np.bincount(s[(s >= 0) & (s < K)].astype(int), minlength=K)
    p = pooled / max(pooled.sum(), 1)
    meta = dict(mode=mode, seed=seed, source_dir=src, source_mapper=mapper, K_source=K, animals={})
    acts_all, states_all = [], []
    for key, segs in data.items():
        rng = _rng(key, seed) if mode == "shift" else np.random.default_rng(
            int(hashlib.sha1(f"{key}|{seed}".encode()).hexdigest()[:12], 16))
        new, info = null_segments(segs, K, mode, rng, p)
        meta["animals"][key] = info
        out = os.path.join(dst, f"{uid}_{key}{SUFFIX}")
        if os.path.islink(out):
            os.remove(out)
        with open(out + ".tmp", "wb") as f:
            pickle.dump(new, f)
        os.replace(out + ".tmp", out)
        af = os.path.join(src, f"{uid}_{key}_actions.pkl")
        if os.path.exists(af):
            acts_all.append(list(pickle.load(open(af, "rb"))))
            states_all.append(new)
    Kn = 1 if mode == "single" else K
    md = os.path.join(dst, "env_clusters")
    os.makedirs(md, exist_ok=True)
    mout = os.path.join(md, conf.get("output_uid", "final_env_maps_") + "_class_mapper.pkl")
    if mode == "single":
        with open(mout, "wb") as f:
            pickle.dump({0: 0}, f)
    else:
        shutil.copyfile(mapper, mout)
    if acts_all and conf.get("write_trans_prob", True):
        _write_T(dst, uid, states_all, acts_all, Kn + 1)
    meta["K_null"] = Kn
    atomic_json(meta, os.path.join(dst, "null_meta.json"))
    print(f"env_null {mode}: {len(data)} animals, K {K} -> {Kn}; wrote {dst}")
    return meta


def _write_T(dst, uid, states_all, acts_all, n_states):
    import sparse
    from naturenet.environment.grid_utils import accumulate_counts, counts_to_trans_prob
    counts, sa = {}, {}
    for segs, acts in zip(states_all, acts_all):
        for s, a in zip(segs, acts):
            if len(s) == len(a) + 1:
                counts, sa = accumulate_counts(s, a, counts=counts, sa_counts=sa)
    if not counts:
        return
    T = counts_to_trans_prob(counts, sa, n_states, fill_unobserved="self_loop", fill_states=range(n_states))
    out = os.path.join(dst, f"{uid}_trans_prob_abstract_env.npz")
    if os.path.islink(out):
        os.remove(out)
    sparse.save_npz(out, T if isinstance(T, sparse.SparseArray) else sparse.COO.from_numpy(np.asarray(T)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    a = ap.parse_args(argv)
    run(read_yaml(a.yaml))


if __name__ == "__main__":
    main()
