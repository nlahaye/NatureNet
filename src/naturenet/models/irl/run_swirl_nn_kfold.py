
#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
NN-SWIRL K-fold (S-1 here; S-2 in run_swirl_nn_2_kfold.py; final holdout run in run_swirl_nn.py).

Fixes vs. the original:
- data via swirl_data (aligned x / x_prev / a; the old positions[:, 1:] / actions[:, 1:] no longer match
  the preprocessing output), animal-grouped folds shared with the ARHMM, per-fold T from train animals;
- per-(run, fold) PRNG keys (was PRNGKey(0) everywhere, so all runs were identical apart from the ARHMM),
  independent keys for the two networks;
- output names include run_uid (the model files were overwritten by every run) and have no stray space;
- "train" LL is computed on the train split (the old _LL file used all data, test fold included); one LL
  per file (the old train_LL was the same number computed twice);
- the saved S-2 reward is the full pair table (K, S, S, A) (was the S-1 expanded evaluation);
- no dense S^2 x A x S^2 tensor (it was built even for S-1), no bfloat16, finished folds skipped.

Config keys (defaults): nn_hidden_size 32, nn_learning_rate 5e-3, nn_reward_bound None,
nn_epochs_init / nn_epochs (S-1 20 / 20, S-2 10 / 10), nn_emit_iters 800, nn_emit_tol None,
nn_s1_warmup_iters (S-2 warm start on the S-1 objective, 0), discount 0.95, vi_iters 100, em_tol None.
"""
import argparse
import os

import numpy as np
import sparse

import jax
import jax.numpy as jnp
from flax import serialization

from naturenet.models.irl.run_arhmm_kfold_nn import arhmm_nn_fname, run_arhmm
from naturenet.models.irl.swirl_data import (
    align_for_swirl,
    as_list,
    get_folds,
    load_swirl_dataset,
    seq_meta_arrays,
    trans_prob_from_segments,
)
from naturenet.models.irl.swirl_nn_model import create_train_state
from naturenet.models.irl.swirl_training_top_level_nn import em_train_nn, learnt_LL_nn, reward_tables
from naturenet.environment.pipeline_checks import check_folds, check_swirl_inputs, check_trans_prob

from sit_fuse.utils import read_yaml

jax.config.update("jax_enable_x64", True)
_platform = os.environ.get("NATURENET_JAX_PLATFORM", "gpu")
if _platform:
    try:
        jax.config.update("jax_platform_name", _platform)
    except Exception as e:  # noqa: BLE001
        print("Could not select JAX platform", _platform, ":", e)

N_ACTIONS = 9
_DEFAULT_EPOCHS = {"s1": (20, 20), "s2": (10, 10)}


# ---------------------------------------------------------------------------
# Data / settings (prepared once, reused across folds and (K, run))
# ---------------------------------------------------------------------------

def prepare_nn_data(yml_conf, data=None):
    data = data if data is not None else load_swirl_dataset(yml_conf)
    if "nn_xoh" in data:
        return data
    S = int(yml_conf["n_states"])
    check_swirl_inputs(data["states"], data["actions"], S, N_ACTIONS)
    x, x_prev, a = align_for_swirl(data["states"], data["actions"])
    data.update(n_states=S, n_actions=N_ACTIONS, x=x, x_prev=x_prev, a=a,
                nn_xoh=jax.nn.one_hot(jnp.asarray(x), S, dtype=jnp.float64)[:, :, None, :],
                nn_xe={"s1": jnp.asarray(x, dtype=jnp.int32),
                       "s2": jnp.asarray(x * S + x_prev, dtype=jnp.int32)})  # pair index s * S + p
    return data


def nn_settings(yml_conf, mode):
    e0, e1 = _DEFAULT_EPOCHS[mode]
    rb = yml_conf.get("nn_reward_bound")
    return dict(hidden_size=int(yml_conf.get("nn_hidden_size", 32)),
                learning_rate=float(yml_conf.get("nn_learning_rate", 5e-3)),
                reward_bound=None if rb is None else float(rb),
                epochs_init=int(yml_conf.get("nn_epochs_init" + ("_s2" if mode == "s2" else ""), e0)),
                epochs=int(yml_conf.get("nn_epochs" + ("_s2" if mode == "s2" else ""), e1)),
                em=dict(emit_iters=int(yml_conf.get("nn_emit_iters", 800)),
                        emit_tol=yml_conf.get("nn_emit_tol"),
                        s1_warmup_iters=int(yml_conf.get("nn_s1_warmup_iters", 0)) if mode == "s2" else 0,
                        discount=float(yml_conf.get("discount", 0.95)),
                        vi_iters=int(yml_conf.get("vi_iters", 100)),
                        tol=yml_conf.get("em_tol")))


def fold_trans_prob_nn(yml_conf, data, train_index):
    """Dense (S, A, S) T from the train animals (cached per split; reused by S-1 and S-2 and every (K, run))."""
    S, A = data["n_states"], data["n_actions"]
    cache = data.setdefault("_cache", {})
    per_fold = yml_conf.get("per_fold_trans_prob", True) and train_index is not None
    key = ("T_nn", None if not per_fold else hash(np.asarray(train_index, dtype=np.int64).tobytes()))
    if key not in cache:
        if per_fold:
            T = trans_prob_from_segments(data["segments"], S, A, keep_groups=set(data["groups"][train_index]),
                                         fill_unobserved=yml_conf.get("fill_unobserved", "self_loop"))
        else:
            T = sparse.load_npz(yml_conf["trans_probs"])
        if T.shape[0] != S:
            raise ValueError(f"trans_prob has {T.shape[0]} states but n_states = {S}")
        check_trans_prob(T, n_actions=A, require_full=yml_conf.get("require_full_trans_prob", True))
        cache[key] = jnp.asarray(T.todense() if hasattr(T, "todense") else T, dtype=jnp.float64)
    return cache[key]


def load_arhmm_nn(out_dir, run_uid, n_hidden, seed, kf_ind=None):
    f = arhmm_nn_fname(out_dir, run_uid, n_hidden, seed, kf_ind)
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} not found: run get_stats_nn.py / run_arhmm_kfold_nn.py "
                                f"({'--holdout' if kf_ind is None else 'K-fold'}) first")
    p = np.load(f, allow_pickle=True)
    return (jnp.asarray(p["logpi0_start"], dtype=jnp.float64), jnp.asarray(p["log_Ps_start"], dtype=jnp.float64),
            jnp.asarray(np.asarray(p["Rs_start"], dtype=np.float64)))


def _names(yml_conf, mode, tag):
    out_dir, run_uid = yml_conf["out_dir"], yml_conf["run_uid"]
    K, seed = yml_conf["n_hidden"], yml_conf["seed"]
    base = os.path.join(out_dir, run_uid + "_" + str(K) + "_" + str(seed))
    lvl = {"s1": ("S1", "net1", "new1"), "s2": ("S2", "net2", "new2")}[mode]
    return {"model": base + "_MLP_" + lvl[0] + "_" + lvl[1] + tag + ".npz",
            "reward": base + "_naturenet_MLP_" + lvl[0] + "_" + lvl[2] + "_reward_filtered" + tag + ".npz",
            "ll": base + "_naturenet_MLP_" + lvl[0] + "_" + lvl[2] + "_LL" + tag + ".npz",
            "ll_test": base + "_naturenet_MLP_" + lvl[0] + "_" + lvl[2] + "_LL_test" + tag + ".npz"}


def _rng(seed, kf_ind):
    key = jax.random.PRNGKey(int(seed))
    return jax.random.split(jax.random.fold_in(key, 0 if kf_ind is None else int(kf_ind) + 1), 2)


# ---------------------------------------------------------------------------
# One fit (shared by K-fold and final runs)
# ---------------------------------------------------------------------------

def check_arhmm_nn(arhmm, K, S):
    """Stage check: ARHMM init shapes / finiteness (Rs is (S, 1, K): dot(one_hot(x)[:, 0, :], Rs[:, 0, :]))."""
    logpi0, log_Ps, Rs = arhmm
    for name, v, shp in (("logpi0", logpi0, (K,)), ("log_Ps", log_Ps, (K, K)), ("Rs", Rs, (S, 1, K))):
        if tuple(v.shape) != shp:
            raise ValueError(f"ARHMM init {name} has shape {tuple(v.shape)}, expected {shp}")
        if not bool(jnp.all(jnp.isfinite(v))):
            raise ValueError(f"ARHMM init {name} has non-finite values")


def fit_swirl_nn(yml_conf, data, train_index, test_index, arhmm, mode, tag, kf_ind=None):
    S, A = data["n_states"], data["n_actions"]
    K = int(yml_conf["n_hidden"])
    if K != int(yml_conf["n_hidden_init"]):
        raise ValueError("n_hidden must equal n_hidden_init (the ARHMM init is for n_hidden_init modes)")
    check_arhmm_nn(arhmm, K, S)
    st = nn_settings(yml_conf, mode)
    names = _names(yml_conf, mode, tag)
    T = fold_trans_prob_nn(yml_conf, data, train_index)
    xoh, xe, a = data["nn_xoh"], data["nn_xe"][mode], jnp.asarray(data["a"], dtype=jnp.int32)
    tr = np.asarray(train_index)

    _, k_net = _rng(yml_conf["seed"], kf_ind)
    R_state = create_train_state(k_net, st["learning_rate"], S, K, st["hidden_size"], A, st["reward_bound"])

    logpi0, log_Ps, Rs = arhmm
    print(f"NN-SWIRL {mode} stage 1 (emission only):", len(tr), "train sequences")
    logpi0, log_Ps, Rs, R_state, LL1 = em_train_nn(logpi0, log_Ps, Rs, R_state, T, xoh[tr], xe[tr], a[tr], mode,
                                                   st["epochs_init"], init=False, trans=False, **st["em"])
    print(f"NN-SWIRL {mode} stage 2 (all parameters)")
    em2 = dict(st["em"], s1_warmup_iters=0)
    logpi0, log_Ps, Rs, R_state, LL2 = em_train_nn(logpi0, log_Ps, Rs, R_state, T, xoh[tr], xe[tr], a[tr], mode,
                                                   st["epochs"], **em2)

    np.savez(names["model"], new_logpi0=np.asarray(logpi0), new_log_Ps=np.asarray(log_Ps), new_Rs=np.asarray(Rs),
             new_R_state=np.array([serialization.to_state_dict(jax.device_get(R_state.params))], dtype=object),
             LL_list=np.concatenate([LL1, LL2]), LL_list_stage1=LL1, mode=mode, hidden_size=st["hidden_size"],
             reward_bound=np.nan if st["reward_bound"] is None else st["reward_bound"])
    reward = reward_tables(R_state, S, mode)
    extra = {}
    if mode == "s2":  # which (s, p) pairs can occur: p -> s possible under T
        extra["valid_pairs"] = (np.asarray(T).sum(axis=1) > 0).T  # [s, p]
        extra["reward_expand"] = reward_tables(R_state, S, "s1")
    np.savez(names["reward"], reward_filtered=reward, reward_layout="k,s,a" if mode == "s1" else "k,s,p,a", **extra)

    LL, paths = learnt_LL_nn(logpi0, log_Ps, Rs, R_state, T, xoh[tr], xe[tr], a[tr], mode,
                             st["em"]["discount"], st["em"]["vi_iters"])
    print(mode, "train LL", LL)
    np.savez(names["ll"], LL=LL, learnt_zs=paths, **seq_meta_arrays(data, tr))
    if test_index is not None and len(test_index):
        te = np.asarray(test_index)
        LL, paths = learnt_LL_nn(logpi0, log_Ps, Rs, R_state, T, xoh[te], xe[te], a[te], mode,
                                 st["em"]["discount"], st["em"]["vi_iters"])
        print(mode, "test LL", LL)
        np.savez(names["ll_test"], LL=LL, learnt_zs=paths, **seq_meta_arrays(data, te))
    return names


def run_swirl_nn_kfold(yml_conf, mode="s1", data=None, folds_to_run=None):
    data = prepare_nn_data(yml_conf, data)
    sel = folds_to_run if folds_to_run is not None else yml_conf.get("folds_to_run")
    sel = None if sel is None else set(int(k) for k in as_list(sel))
    for kf_ind, (tr, te) in enumerate(get_folds(yml_conf, data["groups"])):
        if sel is not None and kf_ind not in sel:
            continue
        check_folds(data["groups"], tr, te, name=f"fold {kf_ind}")
        tag = "_KF_" + str(kf_ind)
        if os.path.exists(_names(yml_conf, mode, tag)["ll_test"]) and not yml_conf.get("overwrite", False):
            print(mode, "fold", kf_ind, "exists; skipping")
            continue
        arhmm = load_arhmm_nn(yml_conf["out_dir"], yml_conf["run_uid"], yml_conf["n_hidden_init"], yml_conf["seed"],
                              kf_ind)
        fit_swirl_nn(yml_conf, data, tr, te, arhmm, mode, tag, kf_ind)


def run_swirl_nn(yml_conf, data=None, folds_to_run=None):
    """S-1 NN K-fold (name kept for get_stats_3.py)."""
    run_swirl_nn_kfold(yml_conf, "s1", data, folds_to_run)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    parser.add_argument("--fold", type=int, nargs="*", default=None)
    args = parser.parse_args()

    yml_conf = read_yaml(args.yaml)
    data = load_swirl_dataset(yml_conf)
    if yml_conf.get("run_arhmm", False):
        run_arhmm(yml_conf, data=data, folds_to_run=args.fold)
    if yml_conf.get("run_swirl_nn", True):
        run_swirl_nn(yml_conf, data=data, folds_to_run=args.fold)
