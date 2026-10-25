
#Code adapted from https://github.com/BRAINML-GT/SWIRL
import argparse
import os

import numpy as np
import sparse

import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial

from naturenet.models.irl.run_arhmm_kfold import run_arhmm
from naturenet.models.irl.swirl_training import *  # noqa: F401,F403
from naturenet.models.irl.swirl_training_top_level import *  # noqa: F401,F403
from naturenet.models.irl.swirl_utils import *  # noqa: F401,F403
from naturenet.models.irl.swirl_data import (
    align_for_swirl,
    as_list,
    seq_meta_arrays,
    as_rs,
    get_folds,
    init_rewards,
    load_swirl_dataset,
    trans_prob_from_segments,
)
from naturenet.environment.pipeline_checks import check_swirl_inputs, check_trans_prob

from sit_fuse.utils import read_yaml

jax.config.update("jax_enable_x64", True)
_platform = os.environ.get("NATURENET_JAX_PLATFORM", "gpu")
if _platform:
    try:
        jax.config.update("jax_platform_name", _platform)
    except Exception as e:  # noqa: BLE001
        print("Could not select JAX platform", _platform, ":", e)


#K = number of hidden modes, D_obs = 1, C = number of (abstract) states


# ---------------------------------------------------------------------------
# Shared preparation (done ONCE, outside the fold loop)
# ---------------------------------------------------------------------------

def prepare_data(yml_conf, data=None):
    data = data if data is not None else load_swirl_dataset(yml_conf)
    if "xohs" in data:  # already prepared
        return data
    n_states = int(yml_conf["n_states"])
    n_actions = 9
    x, x_prev, a = align_for_swirl(data["states"], data["actions"])
    check_swirl_inputs(data["states"], data["actions"], n_states, n_actions)

    one_hotx = vmap(partial(one_hotx_partial, n_states=n_states))
    one_hota = vmap(partial(one_hota_partial, n_actions=n_actions))
    data.update(x=x, x_prev=x_prev, a=a, n_states=n_states, n_actions=n_actions,
                xohs=one_hotx(jnp.asarray(x)), aohs=one_hota(jnp.asarray(a)))
    return data


def fold_trans_prob(yml_conf, data, train_index):
    """Per-fold T from the TRAIN animals only (default), or the global T from yml 'trans_probs'."""
    n_states, n_actions = data["n_states"], data["n_actions"]
    key = ("T", _idx_key(train_index), yml_conf.get("per_fold_trans_prob", True))
    cache = data.setdefault("_cache", {})  # reused across (K, run) in get_stats loops
    if key in cache:
        return cache[key]
    if yml_conf.get("per_fold_trans_prob", True) and train_index is not None:
        train_groups = set(data["groups"][train_index])
        T = trans_prob_from_segments(data["segments"], n_states, n_actions, keep_groups=train_groups,
                                     fill_unobserved=yml_conf.get("fill_unobserved", "self_loop"))
    else:
        T = sparse.load_npz(yml_conf["trans_probs"])
    if T.shape[0] != n_states:
        raise ValueError(f"trans_prob has {T.shape[0]} states but n_states = {n_states}")
    check_trans_prob(T, n_actions=n_actions, require_full=yml_conf.get("require_full_trans_prob", True))
    cache[key] = T
    return T


def _idx_key(idx):
    return None if idx is None else hash(np.asarray(idx, dtype=np.int64).tobytes())


def s2_inputs(yml_conf, data, T):
    """Prev-slot table (from T support + all observed pairs), S-2 tensor and one-hots over (x, slot)."""
    n_states, n_actions = data["n_states"], data["n_actions"]
    cache = data.setdefault("_cache", {})
    key = ("S2", id(T))
    if key in cache:
        return cache[key]
    prev_state_map = compute_prev_state_map(T, n_states, n_actions, observed_pairs=[(data["x"], data["x_prev"])])
    M = n_prev_slots(prev_state_map)
    new_T, _ = construct_new_trans_probs_limited(
        T, prev_state_map, None, n_states, n_actions, n_slots=M,
        max_dense_entries=float(yml_conf.get("s2_max_dense_entries", DEFAULT_S2_MAX_DENSE_ENTRIES)))
    check_trans_prob(new_T, n_actions=n_actions, name="S-2 trans_prob")
    xs_prev = preprocess_xs_prev_np(data["x"], data["x_prev"], prev_state_map, n_actions, n_states)
    one_hotx2 = vmap(partial(one_hotx2_partial, n_states=n_states, n_actions=M))
    xohs2 = one_hotx2(jnp.asarray(data["x"]), jnp.asarray(xs_prev))
    out = (jnp.asarray(new_T), xohs2, M)
    cache[("S2psm", id(T))] = prev_state_map  # small; kept for plots (reward2 layout = x * M + slot)
    if yml_conf.get("cache_fold_inputs", True):  # set False if 5 folds of S-2 tensors do not fit in memory
        cache[key] = out
    return out


def load_arhmm(out_dir, run_uid, n_hidden, seed, kf_ind=None):
    suffix = '_seed_arhmm_s.npz' if kf_ind is None else '_seed_arhmm_s_KF_' + str(kf_ind) + '.npz'
    p = np.load(os.path.join(out_dir, run_uid + "_" + str(n_hidden) + "_hidden_" + str(seed) + suffix), allow_pickle=True)
    Rs = as_rs((p['W1_start'], p['b1_start'], p['W2_start'], p['b2_start']))
    return jnp.asarray(p['logpi0_start']), jnp.asarray(p['log_Ps_start']), Rs


def _save(path, **kw):
    if "new_Rs" in kw:
        kw["new_Rs"] = np.array(kw["new_Rs"], dtype=object)
    jnp.savez(path, **kw)


def _temps(yml_conf, key, K, default):
    t = yml_conf.get(key)
    t = default if t is None else list(as_list(t))
    if len(t) != K:
        raise ValueError(f"{key} has {len(t)} entries, expected n_hidden = {K}")
    return jnp.array(t, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Fit stages (shared with run_swirl.py)
# ---------------------------------------------------------------------------

def fit_swirl_init(yml_conf, data, T, train_index, arhmm, tag):
    """S-1 with all temps = 1 (default). Saves *_init0{tag}.npz and *_init1{tag}.npz."""
    out_dir, run_uid, seed = yml_conf["out_dir"], yml_conf["run_uid"], yml_conf["seed"]
    K = int(yml_conf["n_hidden_init"])
    S, A = data["n_states"], data["n_actions"]
    tol = yml_conf.get("em_tol")
    temps = _temps(yml_conf, "temps_init", K, [1.0] * K)
    _, R_start2 = init_rewards(K, S, A, seed)
    logpi0, log_Ps, Rs = arhmm
    xo, ao = data["xohs"][train_index], data["aohs"][train_index]
    Td = jnp.asarray(T.todense() if hasattr(T, "todense") else T)

    print("Training initial model step 1")
    new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_temp(
        logpi0, log_Ps, Rs, jnp.array(R_start2)[:, None], temps, Td, xo, ao,
        int(yml_conf.get("epochs_init", 50)), init=False, trans=False, tol=tol)
    _save(os.path.join(out_dir, run_uid + "_swirl_run_" + str(K) + '_' + str(seed) + "_naturenet_init0" + tag + ".npz"),
          new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=new_Rs, new_reward=new_reward, LL_list=LL_list, temps=temps)

    print("Training initial model step 2")
    new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_temp(
        jnp.array(new_logpi0), jnp.array(new_log_Ps), new_Rs, jnp.array(new_reward), temps, Td, xo, ao,
        int(yml_conf.get("epochs_init_2", 30)), tol=tol)
    # Raw reward is saved (and used downstream); the (r-min)^4 version is for plots only.
    _save(os.path.join(out_dir, run_uid + "_swirl_run_" + str(K) + '_' + str(seed) + "_naturenet_init1" + tag + ".npz"),
          new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=new_Rs, new_reward=new_reward,
          new_reward_normalized=normalize_reward(np.asarray(new_reward)), LL_list=LL_list, temps=temps)


def fit_swirl_final(yml_conf, data, T, train_index, test_index, arhmm, tag, name_fmt="kfold"):
    """
    S-2 (optional) and S-1 with temps_final, then train / test LLs with the RAW rewards.
    name_fmt "kfold" keeps the run_swirl_kfold file names, "final" keeps the run_swirl ones.
    """
    out_dir, run_uid, seed = yml_conf["out_dir"], yml_conf["run_uid"], yml_conf["seed"]
    K, K_init = int(yml_conf["n_hidden"]), int(yml_conf["n_hidden_init"])
    if K != K_init:
        raise ValueError(f"n_hidden ({K}) must equal n_hidden_init ({K_init}): the ARHMM / init params are for "
                         f"{K_init} modes")
    S, A = data["n_states"], data["n_actions"]
    tol = yml_conf.get("em_tol")
    epochs, epochs_2 = int(yml_conf.get("epochs", 50)), int(yml_conf.get("epochs_2", 30))
    temps = _temps(yml_conf, "temps_final", K, [0.01] + [1.0] * (K - 1))
    _, R_start2 = init_rewards(K, S, A, seed)
    R_start2 = jnp.array(R_start2)[:, None]  # (K, 1, S)
    if yml_conf.get("use_init_reward", False):
        p = jnp.load(os.path.join(out_dir, run_uid + "_swirl_run_" + str(K_init) + '_' + str(seed) +
                                  "_naturenet_init1" + tag + ".npz"), allow_pickle=True)
        R_start2 = jnp.asarray(p["new_reward"])
        print("Warm-starting final S-1 / S-2 rewards from init1")
    logpi0, log_Ps, Rs = arhmm
    Td = jnp.asarray(T.todense() if hasattr(T, "todense") else T)
    has_test = test_index is not None and len(test_index) > 0
    xo, ao = data["xohs"], data["aohs"]

    if name_fmt == "kfold":
        n = lambda s: os.path.join(out_dir, run_uid + "_" + str(K) + '_' + str(seed) + s + tag + ".npz")  # noqa: E731
        f_s2i, f_s2, f_s1i, f_s1 = "_naturenet_iter2_S2_init", "_naturenet_iter2_S2", "_naturenet_iter2_S1_temp_init", "_naturenet_iter2_S1_temp_2"
        f_ll1, f_ll1t, f_ll2, f_ll2t = "_naturenet_S1_iter2_LL1", "_naturenet_S1_iter2_LL1_test", "_naturenet_S2_iter2_LL2", "_naturenet_S2_iter2_LL2_test"
    else:
        n = lambda s: os.path.join(out_dir, run_uid + "_" + str(K) + '_' + str(seed) + s + ".npz")  # noqa: E731
        f_s2i, f_s2, f_s1i, f_s1 = "_S2_naturenet_iter2_init", "_S2_naturenet_iter2", "_S1_naturenet_iter2_temp_init", "_S1_naturenet_iter2_temp_2"
        f_ll1, f_ll1t, f_ll2, f_ll2t = "_S1_naturenet_iter2_LL1", "_S1_naturenet_iter2_test_LL1", "_S2_naturenet_iter2_LL2", "_S2_naturenet_iter2_test_LL2"

    if yml_conf.get("run_s2", True):
        new_T, xohs2, M = s2_inputs(yml_conf, data, T)
        psm = data["_cache"][("S2psm", id(T))]
        psm_arr = np.empty(S, dtype=object)  # psm is a dict {state: [predecessors in slot order]}
        for s_ in range(S):
            psm_arr[s_] = np.asarray(psm.get(s_, []), dtype=np.int64)
        s2_layout = dict(n_slots=M, prev_state_map=psm_arr)
        R2 = jnp.repeat(R_start2[:, :, :, None], M, axis=3).reshape(K, 1, S * M)  # (K, 1, S*M)
        x2 = xohs2[train_index]
        print("Training second model step 1 (S-2, M =", M, "predecessor slots)")
        r = em_train2_naturenet(logpi0, log_Ps, Rs, R2, new_T, xo[train_index], x2, ao[train_index], epochs, tol=tol)
        _save(n(f_s2i), new_logpi0=r[0], new_log_Ps=r[1], new_Rs=r[2], new_reward=r[3], LL_list=r[4], temps=temps)
        print("Training second model step 2")
        r = em_train2_temp(jnp.array(r[0]), jnp.array(r[1]), r[2], jnp.array(r[3]), temps, new_T, xo[train_index], x2,
                           ao[train_index], epochs_2, tol=tol)
        _save(n(f_s2), new_logpi0=r[0], new_log_Ps=r[1], new_Rs=r[2], new_reward=r[3],
              new_reward_normalized=normalize_reward(np.asarray(r[3])), LL_list=r[4], temps=temps, **s2_layout)

        print("Final computations", "S2")
        LL, path = learnt_LL2(r[0], r[1], r[2], jnp.asarray(r[3]), temps, S, A, new_T, xo[train_index], x2, ao[train_index])
        print("S2 train LL", LL)
        np.savez(n(f_ll2), LL=LL, jax_path_vmap=np.asarray(path), **seq_meta_arrays(data, train_index))
        if has_test:
            LL, path = learnt_LL2(r[0], r[1], r[2], jnp.asarray(r[3]), temps, S, A, new_T, xo[test_index],
                                  xohs2[test_index], ao[test_index])
            print("S2 test LL", LL)
            np.savez(n(f_ll2t), LL=LL, jax_path_vmap=np.asarray(path), **seq_meta_arrays(data, test_index))

    print("Training third model step 1 (S-1)")
    r = em_train_naturenet(logpi0, log_Ps, Rs, R_start2, Td, xo[train_index], ao[train_index], epochs, tol=tol)
    _save(n(f_s1i), new_logpi0=r[0], new_log_Ps=r[1], new_Rs=r[2], new_reward=r[3], LL_list=r[4], temps=temps)
    print("Training third model step 2")
    r = em_train_temp(jnp.array(r[0]), jnp.array(r[1]), r[2], jnp.array(r[3]), temps, Td, xo[train_index],
                      ao[train_index], epochs_2, tol=tol)
    _save(n(f_s1), new_logpi0=r[0], new_log_Ps=r[1], new_Rs=r[2], new_reward=r[3],
          new_reward_normalized=normalize_reward(np.asarray(r[3])), LL_list=r[4], temps=temps)

    # LLs use the RAW learned reward (previously normalize_reward() was applied first, so the reported
    # LLs were for a different reward than the one that was fit).
    print("Final computations", "S1")
    LL, path = learnt_LL1(r[0], r[1], r[2], jnp.asarray(r[3]), temps, S, A, Td, xo[train_index], ao[train_index])
    print("S1 train LL", LL)
    np.savez(n(f_ll1), LL=LL, jax_path_vmap=np.asarray(path), **seq_meta_arrays(data, train_index))
    if has_test:
        LL, path = learnt_LL1(r[0], r[1], r[2], jnp.asarray(r[3]), temps, S, A, Td, xo[test_index], ao[test_index])
        print("S1 test LL", LL)
        np.savez(n(f_ll1t), LL=LL, jax_path_vmap=np.asarray(path), **seq_meta_arrays(data, test_index))


# ---------------------------------------------------------------------------
# K-fold drivers
# ---------------------------------------------------------------------------

def _fold_iter(yml_conf, data, folds_to_run):
    folds = get_folds(yml_conf, data["groups"])
    sel = folds_to_run if folds_to_run is not None else yml_conf.get("folds_to_run")
    sel = None if sel is None else set(int(k) for k in as_list(sel))
    for kf_ind, (tr, te) in enumerate(folds):
        if sel is None or kf_ind in sel:
            yield kf_ind, tr, te


def run_swirl_init(yml_conf, data=None, folds_to_run=None):
    data = prepare_data(yml_conf, data)
    out_dir, run_uid, seed, K = yml_conf["out_dir"], yml_conf["run_uid"], yml_conf["seed"], yml_conf["n_hidden_init"]
    for kf_ind, tr, te in _fold_iter(yml_conf, data, folds_to_run):
        tag = "_KF_" + str(kf_ind)
        out = os.path.join(out_dir, run_uid + "_swirl_run_" + str(K) + '_' + str(seed) + "_naturenet_init1" + tag + ".npz")
        if os.path.exists(out) and not yml_conf.get("overwrite", False):
            print("init fold", kf_ind, "exists; skipping")
            continue
        print("SWIRL init fold", kf_ind, ":", len(tr), "train /", len(te), "test sequences")
        T = fold_trans_prob(yml_conf, data, tr)
        fit_swirl_init(yml_conf, data, T, tr, load_arhmm(out_dir, run_uid, K, seed, kf_ind), tag)


def run_swirl_final(yml_conf, data=None, folds_to_run=None):
    data = prepare_data(yml_conf, data)
    out_dir, run_uid, seed, K = yml_conf["out_dir"], yml_conf["run_uid"], yml_conf["seed"], yml_conf["n_hidden"]
    for kf_ind, tr, te in _fold_iter(yml_conf, data, folds_to_run):
        tag = "_KF_" + str(kf_ind)
        out = os.path.join(out_dir, run_uid + "_" + str(K) + '_' + str(seed) + "_naturenet_S1_iter2_LL1_test" + tag + ".npz")
        if os.path.exists(out) and not yml_conf.get("overwrite", False):
            print("final fold", kf_ind, "exists; skipping")
            continue
        print("SWIRL final fold", kf_ind, ":", len(tr), "train /", len(te), "test sequences")
        T = fold_trans_prob(yml_conf, data, tr)
        fit_swirl_final(yml_conf, data, T, tr, te, load_arhmm(out_dir, run_uid, yml_conf["n_hidden_init"], seed, kf_ind), tag)


def run_all(yml_conf, folds_to_run=None):
    """ARHMM + SWIRL init + final with the data loaded once."""
    data = None
    if yml_conf.get("run_arhmm", False) or yml_conf.get("run_swirl_init", False) or yml_conf.get("run_swirl_final", False):
        data = load_swirl_dataset(yml_conf)
    if yml_conf.get("run_arhmm", False):
        run_arhmm(yml_conf, data=data, folds_to_run=folds_to_run)
    if yml_conf.get("run_swirl_init", False) or yml_conf.get("run_swirl_final", False):
        data = prepare_data(yml_conf, data)
    if yml_conf.get("run_swirl_init", False):
        run_swirl_init(yml_conf, data=data, folds_to_run=folds_to_run)
    if yml_conf.get("run_swirl_final", False):
        run_swirl_final(yml_conf, data=data, folds_to_run=folds_to_run)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    parser.add_argument("--fold", type=int, nargs="*", default=None, help="Only run these fold indices (job arrays).")
    args = parser.parse_args()

    yml_conf = read_yaml(args.yaml)
    run_all(yml_conf, folds_to_run=args.fold)
