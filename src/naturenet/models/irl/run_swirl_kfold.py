
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
    align_data,
    align_for_swirl,
    step_mask,
    as_list,
    seq_meta_arrays,
    as_rs,
    get_folds,
    subsample_fold,
    init_rewards,
    load_swirl_dataset,
    trans_prob_from_segments,
)
from naturenet.environment.pipeline_checks import check_swirl_inputs, check_trans_prob
from naturenet.models.irl.kinematics import (
    KinematicEmissions,
    cached_kinematics,
    comparable_ll,
    kin_conf,
    labels_from_conf,
    p_exploit,
    pooled,
)
from naturenet.models.irl.swirl_ext import em_train_ext, params_to_save, swirl_eval

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
    """
    Aligned x / x_prev / a and one-hots. segment_policy: variable: padded steps get all-zero one-hots
    (emission LL 0, excluded from every M-step) and identity transitions (swirl_training.pad_eye), so the
    LLs, posteriors and Viterbi paths on the real steps equal those of the unpadded sequences.
    """
    data = data if data is not None else load_swirl_dataset(yml_conf, allow_variable=True)
    if "xohs" in data:  # already prepared
        return data
    n_states = int(yml_conf["n_states"])
    n_actions = 9
    x, x_prev, a = align_data(data)
    check_swirl_inputs(data["states"], data["actions"], n_states, n_actions)
    m = step_mask(data, offset=1)
    assert m.shape == x.shape and np.all((a >= 0) == m), "padding mask / PAD_ACTION mismatch"
    assert m[:, 0].all(), "every sequence needs at least one real SWIRL step (>= 3 states)"
    mj = jnp.asarray(m, dtype=jnp.float64)[:, :, None, None]
    one_hotx = vmap(partial(one_hotx_partial, n_states=n_states))
    one_hota = vmap(partial(one_hota_partial, n_actions=n_actions))
    data.update(x=x, x_prev=x_prev, a=a, n_states=n_states, n_actions=n_actions, step_mask=m,
                xohs=one_hotx(jnp.asarray(x)) * mj, aohs=one_hota(jnp.asarray(np.maximum(a, 0))) * mj)
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


def s2_structured_trans(T, prev_state_map, n_states, M):
    """S2Trans(T dense (S, A, S), G (S, S)) with G[x, x'] = x' * M + slot(x', x) (0 where x -> x' is impossible)."""
    Td = np.asarray(T.todense() if hasattr(T, "todense") else T, dtype=np.float64)
    lut = prev_slot_lookup(prev_state_map)
    G = np.zeros((n_states, n_states), dtype=np.int32)
    for (xn, xp), j in lut.items():           # x' = xn reached from x = xp
        G[xp, xn] = xn * M + j
    reach = Td.sum(1) > 0                      # (x, x') with T[x, a, x'] > 0 for some a: x must be a slot of x'
    bad = [(int(x), int(y)) for x, y in zip(*np.where(reach)) if (int(y), int(x)) not in lut]
    if bad:
        raise ValueError(f"prev_state_map is missing predecessors present in T, e.g. {bad[:5]}; rebuild it from T")
    return S2Trans(Td, G)


def s2_inputs(yml_conf, data, T):
    """Prev-slot table (from T support + all observed pairs), S-2 tensor and one-hots over (x, slot)."""
    n_states, n_actions = data["n_states"], data["n_actions"]
    cache = data.setdefault("_cache", {})
    key = ("S2", id(T))
    if key in cache:
        return cache[key]
    prev_state_map = compute_prev_state_map(T, n_states, n_actions, observed_pairs=[(data["x"], data["x_prev"])])
    M = n_prev_slots(prev_state_map)
    if yml_conf.get("s2_structured_vi", True):
        # Exact structured form (swirl_training.S2Trans): value iteration on T (S, A, S) plus a gather, instead of
        # the dense (S*M, A, S*M) new_T. M^2 less VI compute and no dense S-2 tensor in host or device memory.
        new_T = s2_structured_trans(T, prev_state_map, n_states, M)
        size = new_T.T.nbytes + new_T.G.nbytes
    else:
        new_T, _ = construct_new_trans_probs_limited(
            T, prev_state_map, None, n_states, n_actions, n_slots=M,
            max_dense_entries=float(yml_conf.get("s2_max_dense_entries", DEFAULT_S2_MAX_DENSE_ENTRIES)))
        check_trans_prob(new_T, n_actions=n_actions, name="S-2 trans_prob")
        new_T = np.asarray(new_T)
        size = new_T.nbytes
    xs_prev = preprocess_xs_prev_np(data["x"], data["x_prev"], prev_state_map, n_actions, n_states)
    print(f"S-2: M = {M} predecessor slots, {n_states * M} augmented states, "
          f"{'structured' if is_s2(new_T) else 'dense'} new_T {size / 1e9:.3f} GB")
    if yml_conf.get("s2_index_emissions", True):
        # (N, T, 1) int32 index x * M + slot (-1 on padded steps) instead of the dense (N, T, 1, S*M) float64
        # one-hot: same LLs (swirl_training.comp_ll_jax) with 2*S*M times less memory.
        x2 = np.asarray(data["x"], dtype=np.int64) * M + np.asarray(xs_prev, dtype=np.int64)
        if "step_mask" in data:
            x2 = np.where(np.asarray(data["step_mask"], dtype=bool), x2, -1)
        if x2.max() >= np.iinfo(np.int32).max:
            raise ValueError("S-2 index exceeds int32")
        xohs2 = x2.astype(np.int32)[:, :, None]
    else:  # previous dense one-hots (needs N * T * S * M * 8 bytes)
        one_hotx2 = vmap(partial(one_hotx2_partial, n_states=n_states, n_actions=M))
        xohs2 = one_hotx2(jnp.asarray(data["x"]), jnp.asarray(xs_prev))
        if "step_mask" in data:   # padded steps: zero one-hots (segment_policy: variable)
            xohs2 = xohs2 * jnp.asarray(data["step_mask"], dtype=xohs2.dtype)[:, :, None, None]
    # new_T stays in HOST memory; each fit moves it to the device once and frees it afterwards (previously
    # every fold's dense new_T stayed on the GPU through the cache).
    out = (new_T, xohs2, M)
    cache[("S2psm", id(T))] = prev_state_map  # small; kept for plots (reward2 layout = x * M + slot)
    if yml_conf.get("cache_fold_inputs", True):  # host memory only; set False if 5 folds of new_T do not fit
        cache[key] = out
    return out


def load_arhmm(out_dir, run_uid, n_hidden, seed, kf_ind=None):
    suffix = '_seed_arhmm_s.npz' if kf_ind is None else '_seed_arhmm_s_KF_' + str(kf_ind) + '.npz'
    p = np.load(os.path.join(out_dir, run_uid + "_" + str(n_hidden) + "_hidden_" + str(seed) + suffix), allow_pickle=True)
    Rs = as_rs((p['W1_start'], p['b1_start'], p['W2_start'], p['b2_start']))
    out = ArhmmInit((jnp.asarray(p['logpi0_start']), jnp.asarray(p['log_Ps_start']), Rs))
    if KinematicEmissions.has_params(p):
        out.kin = KinematicEmissions.from_params(p)   # kinematic ARHMM (run_arhmm_kfold with a kinematics block)
    return out


class ArhmmInit(tuple):
    """(logpi0, log_Ps, Rs) that unpacks as before, plus .kin (KinematicEmissions or None)."""
    kin = None


def _save(path, **kw):
    if "new_Rs" in kw:
        kw["new_Rs"] = np.array(kw["new_Rs"], dtype=object)
    jnp.savez(path, **kw)


def _temps(yml_conf, key, K, default):
    t = yml_conf.get(key)
    t = default if t is None else list(as_list(t))
    if len(t) == 1 and K > 1:       # a single value (e.g. temps_final: 1.0) applies to every intention
        t = t * K
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


def _final_names(yml_conf, tag, name_fmt):
    out_dir, run_uid, seed, K = yml_conf["out_dir"], yml_conf["run_uid"], yml_conf["seed"], int(yml_conf["n_hidden"])
    if name_fmt == "kfold":
        n = lambda s: os.path.join(out_dir, run_uid + "_" + str(K) + '_' + str(seed) + s + tag + ".npz")  # noqa: E731
        return n, ("_naturenet_iter2_S2_init", "_naturenet_iter2_S2", "_naturenet_iter2_S1_temp_init", "_naturenet_iter2_S1_temp_2"), \
            ("_naturenet_S1_iter2_LL1", "_naturenet_S1_iter2_LL1_test", "_naturenet_S2_iter2_LL2", "_naturenet_S2_iter2_LL2_test")
    n = lambda s: os.path.join(out_dir, run_uid + "_" + str(K) + '_' + str(seed) + s + ".npz")  # noqa: E731
    return n, ("_S2_naturenet_iter2_init", "_S2_naturenet_iter2", "_S1_naturenet_iter2_temp_init", "_S1_naturenet_iter2_temp_2"), \
        ("_S1_naturenet_iter2_LL1", "_S1_naturenet_iter2_test_LL1", "_S2_naturenet_iter2_LL2", "_S2_naturenet_iter2_test_LL2")


def swirl_kinematics(yml_conf, data, kcfg=None):
    """KinFeatures aligned with SWIRL's x = states[:, 1:-1] (step t = fix start + 1 + t -> next fix)."""
    kcfg = kcfg or kin_conf(yml_conf)
    return None if kcfg is None else cached_kinematics(yml_conf, data, kcfg, offset=1,
                                                       shape=tuple(np.asarray(data["xohs"]).shape[:2]))


def _init_kin_model(yml_conf, kcfg, K, kin_tr, arhmm_kin, seed):
    if arhmm_kin is not None:
        print("Kinematic emissions warm-started from the kinematic ARHMM")
        return arhmm_kin
    return KinematicEmissions.from_conf(K, kcfg).init_quantiles(kin_tr, rng=np.random.default_rng(seed))


def _kin_ll_file(path, ev_joint, ev_action, kin_null, kin, labels, data, idx, extra=None):
    cmp_, joint, null = comparable_ll(ev_joint["LL_total"], kin_null, kin, ev_joint["n_steps"])
    pe, px = p_exploit(ev_joint["gamma"], labels)
    np.savez(path, LL=cmp_, LL_joint=joint, LL_kin_null=null, LL_action=ev_action["LL"],
             jax_path_vmap=ev_joint["path"], gamma=ev_joint["gamma"], p_exploit=pe, p_explore=px,
             kin_labels=np.asarray(labels["label"]), kin_step=kin.step, kin_turn=kin.turn,
             **(extra or {}), **seq_meta_arrays(data, idx))
    return cmp_, joint, null


def _fit_swirl_final_kin(yml_conf, data, T, train_index, test_index, arhmm, tag, name_fmt, kcfg, R_start2, temps,
                         tol, epochs, epochs_2):
    """
    fit_swirl_final with per-intention kinematic emissions (swirl_ext.em_train_ext), same stages / file names.
    LL files: LL = LL_joint - LL_kin_null (comparable with action-only runs; kin null = one-intention kinematic
    fit on the train animals), LL_joint, LL_kin_null, LL_action, gamma, p_exploit / p_explore, kin_labels.
    """
    n, (f_s2i, f_s2, f_s1i, f_s1), (f_ll1, f_ll1t, f_ll2, f_ll2t) = _final_names(yml_conf, tag, name_fmt)
    K, seed = int(yml_conf["n_hidden"]), int(yml_conf["seed"])
    S = data["n_states"]
    logpi0, log_Ps, Rs = arhmm[:3]
    arhmm_kin = getattr(arhmm, "kin", None)
    if arhmm_kin is not None and arhmm_kin.K != K:
        raise ValueError(f"ARHMM kinematics have {arhmm_kin.K} intentions, n_hidden = {K}")
    Td = jnp.asarray(T.todense() if hasattr(T, "todense") else T)
    xo, ao = data["xohs"], data["aohs"]
    tr = np.asarray(train_index)
    te = None if test_index is None or len(test_index) == 0 else np.asarray(test_index)
    kin = swirl_kinematics(yml_conf, data, kcfg)
    k_tr = kin[tr]
    print("Kinematics coverage (train):", k_tr.coverage())
    kin_null = pooled(k_tr, kcfg)
    km0 = _init_kin_model(yml_conf, kcfg, K, k_tr, arhmm_kin, seed)

    def _stages(xe_all, Tm, R0, f_init, f_fin, f_ll, f_llt, layout=None):
        xe_tr = None if xe_all is None else xe_all[tr]
        r = em_train_ext(logpi0, log_Ps, Rs, R0, Tm, xo[tr], ao[tr], xohs_emit=xe_tr, kin=k_tr, kin_model=km0,
                         iters=epochs, tol=tol)
        _save(f_init, **params_to_save(r), temps=temps)
        r = em_train_ext(jnp.array(r["logpi0"]), jnp.array(r["log_Ps"]), r["Rs"], jnp.array(r["rewards"]), Tm,
                         xo[tr], ao[tr], xohs_emit=xe_tr, temps=temps, kin=k_tr, kin_model=r["kin_model"],
                         iters=epochs_2, tol=tol)
        labels = labels_from_conf(r["kin_model"], kcfg)
        _save(f_fin, **params_to_save(r), new_reward_normalized=normalize_reward(np.asarray(r["rewards"])),
              kin_labels=np.asarray(labels["label"]), kin_step_feature=kcfg["step_feature"], **(layout or {}))
        labels.to_csv(f_fin.replace(".npz", "_kinematics.csv"), index=False)
        print(labels[["state", "label", "step_median", "persistence", "p_stationary"]].to_string(index=False))
        for idx, path, name in ((tr, f_ll, "train"), (te, f_llt, "test")):
            if idx is None:
                continue
            xe = None if xe_all is None else xe_all[idx]
            evj = swirl_eval(r, Tm, xo[idx], ao[idx], xohs_emit=xe, kin=kin[idx])
            eva = swirl_eval(r, Tm, xo[idx], ao[idx], xohs_emit=xe, kin=None, use_kin=False, paths=False)
            c, j, nl = _kin_ll_file(path, evj, eva, kin_null, kin[idx], labels, data, idx)
            print(f"{name} LL (comparable) {c:.4f}  joint {j:.4f}  kinematic null {nl:.4f}  action-only {eva['LL']:.4f}")
        return r

    if yml_conf.get("run_s2", True):
        new_T, xohs2, M = s2_inputs(yml_conf, data, T)
        psm = data["_cache"][("S2psm", id(T))]
        psm_arr = np.empty(S, dtype=object)
        for s_ in range(S):
            psm_arr[s_] = np.asarray(psm.get(s_, []), dtype=np.int64)
        R2 = jnp.repeat(R_start2[:, :, :, None], M, axis=3).reshape(K, 1, S * M)
        print("Training S-2 with kinematic emissions (M =", M, "predecessor slots)")
        _stages(xohs2, new_T, R2, n(f_s2i), n(f_s2), n(f_ll2), n(f_ll2t),
                layout=dict(n_slots=M, prev_state_map=psm_arr))
    print("Training S-1 with kinematic emissions")
    _stages(None, Td, R_start2, n(f_s1i), n(f_s1), n(f_ll1), n(f_ll1t))


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

    n, (f_s2i, f_s2, f_s1i, f_s1), (f_ll1, f_ll1t, f_ll2, f_ll2t) = _final_names(yml_conf, tag, name_fmt)
    kcfg = kin_conf(yml_conf)
    if kcfg is not None:
        return _fit_swirl_final_kin(yml_conf, data, T, train_index, test_index, arhmm, tag, name_fmt, kcfg,
                                    R_start2, temps, tol, epochs, epochs_2)

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
            tr, te = subsample_fold(yml_conf, data, tr, te, kf_ind)
            yield kf_ind, tr, te


def run_swirl_init(yml_conf, data=None, folds_to_run=None):
    print("M-step options:", set_mstep_options(yml_conf))
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
    print("M-step options:", set_mstep_options(yml_conf))
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
        data = load_swirl_dataset(yml_conf, allow_variable=True)
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
