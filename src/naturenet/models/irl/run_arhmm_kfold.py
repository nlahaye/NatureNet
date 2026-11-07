
#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
ARHMM initialisation for SWIRL (ssm fork). With a `kinematics` YAML block the ARHMM is the kinematic one
(ssm dynamics "arcategorical_kinematic": data rows [state, step feature, turn]), so the SWIRL intentions
start from modes that already separate slow / tortuous from fast / directed movement; the kinematic
parameters are saved in the ARHMM file (kin_*) and picked up by run_swirl_kfold.load_arhmm. The LL files
then store train_ll / test_ll = (joint - one-intention kinematic null) per step, comparable with the
action-only runs (train_ll_joint / test_ll_joint are saved as well).

The raw ARHMM LL scores the next STATE (n_states-way) and is not on the same event space as SWIRL / HIAVI. With
arhmm_action_ll: True (default) the LL file also stores train_ll_act / test_ll_act: the ARHMM re-scored as an
action model on the SWIRL steps (arhmm_action_ll.py), plotted as "arhmm_act" by plot_ll_comparison.py. The AR
tables are saved to *_arhmm_s_logits_KF_<k>.npz. Existing folds without the action LLs are re-scored from the
logits file or, for older runs, re-fit with the same seed (the existing init file used by SWIRL is kept).

arhmm_pseudo_count (default 1e-5, the previous fixed value) is the Dirichlet pseudo-count of the AR tables;
arhmm_pseudo_support: "T" applies it only to moves reachable under the fold's T (others get arhmm_pseudo_floor).
Use a separate run_uid / out_dir per value (finished folds are skipped).
"""

import numpy as np
import numpy.random as npr
import os

import argparse

from ssm.swirl import ARHMMs
from ssm.observations import ARCategoricalObservations

from sit_fuse.utils import read_yaml

from naturenet.models.irl.swirl_data import align_data, as_list, get_folds, load_swirl_dataset, step_mask
from naturenet.models.irl.kinematics import KinFeatures, KinematicEmissions, cached_kinematics, kin_conf, pooled
from naturenet.models.irl.arhmm_action_ll import arhmm_action_lls, arhmm_ar_logits


def arhmm_split_ll(arhmm_s, list_x):
    """Per-step LL of one split: ARHMMs.log_likelihood (summed over sequences) / total steps (SWIRL's LL / (N*T))."""
    if list_x is None or len(list_x) == 0:
        return np.nan
    n_steps = int(sum(len(x) for x in list_x))
    return float(arhmm_s.log_likelihood(list_x)) / n_steps


def _as_list_x(positions):
    return [np.expand_dims(np.asarray(p).astype(np.int32), axis=1) for p in positions]


def kin_datas(positions, kin):
    """ARHMM data rows [state, step, turn] (float) for the kinematic ARHMM; row t = move x_t -> x_{t+1}."""
    return [np.c_[np.asarray(p, dtype=float), kin.step[i, :len(p)], kin.turn[i, :len(p)]] for i, p in enumerate(positions)]


def fit_arhmm_params(positions, n_hidden_states, n_states, emission_dim=1, latent_state_dim=1, epochs=100,
                     kin=None, kin_cfg=None, transitions="mlprecurrent", inputs=None, transition_kwargs=None):
    """
    Fit an ARHMM on (N, T) state sequences; kin (KinFeatures aligned with positions) + kin_cfg -> kinematic
    ARHMM. Returns (arhmm, list_x, EM LL trace).
    """
    M = 0 if inputs is None else int(np.asarray(inputs[0]).shape[1])
    if kin is not None:
        # bin order drawn from the global numpy seed (run_arhmm seeds it per fold)
        km = KinematicEmissions.from_conf(n_hidden_states, kin_cfg).init_quantiles(
            kin, rng=np.random.default_rng(npr.randint(2 ** 31 - 1)))
        list_x = kin_datas(positions, kin)
        arhmm_s = ARHMMs(emission_dim, n_hidden_states, latent_state_dim, n_states, M=M, transitions=transitions,
                         transition_kwargs=transition_kwargs, dynamics="arcategorical_kinematic",
                         dynamics_kwargs=dict(kin_model=km), single_subspace=True)
    else:
        list_x = [np.expand_dims(np.array(p), axis=1) for p in positions]
        arhmm_s = ARHMMs(emission_dim, n_hidden_states, latent_state_dim, n_states, M=M, transitions=transitions,
                         transition_kwargs=transition_kwargs, dynamics="arcategorical", single_subspace=True)
    lls = arhmm_s.initialize(list_x, inputs=inputs, num_init_iters=epochs)
    return arhmm_s, list_x, lls


def arhmm_start_params(arhmm_s):
    """(logpi0, log_Ps, (W1, b1, W2, b2), kin_model or None) for SWIRL."""
    t = arhmm_s.transitions
    return (np.asarray(arhmm_s.init_state_distn.log_pi0), np.asarray(t.log_Ps),
            (np.asarray(t.W1), np.asarray(t.b1), np.asarray(t.W2), np.asarray(t.b2)),
            getattr(arhmm_s.dynamics, "kin_model", None))


def _kin_null_split(kin_null, kin, lengths=None):
    """Pooled kinematic LL on the ARHMM moves (steps 0..n_i-2 of every sequence; n_i = real steps, default T)."""
    st, tu = kin.step.copy(), kin.turn.copy()
    n = np.full(st.shape[0], st.shape[1]) if lengths is None else np.asarray(lengths)
    drop = np.arange(st.shape[1])[None, :] >= (n - 1)[:, None]
    st[drop] = np.nan
    tu[drop] = np.nan
    return kin_null.total_loglik(KinFeatures(st, tu)) * kin_null.weight


def run_arhmm_internal(seed, emission_dim, n_hidden_states, latent_state_dim, n_states, positions, out_dir, run_uid, kf_ind,
                       test_positions=None, kin_train=None, kin_test=None, kin_cfg=None, epochs=100, write_init=True):
    """Fit one fold. write_init False keeps an existing init file (re-fit only to add LLs). Returns the ARHMM."""

    os.makedirs(out_dir, exist_ok=True)
    arhmm_s, list_x, lls_arhmm = fit_arhmm_params(positions, n_hidden_states, n_states, emission_dim,
                                                  latent_state_dim, epochs, kin=kin_train, kin_cfg=kin_cfg)
    init_start = arhmm_s.init_state_distn.initial_state_distn
    logpi0_start = arhmm_s.init_state_distn.log_pi0
    log_Ps_start = arhmm_s.transitions.log_Ps
    Rs_start = arhmm_s.transitions.W1, arhmm_s.transitions.b1, arhmm_s.transitions.W2, arhmm_s.transitions.b2

    fname = run_uid + "_" + str(n_hidden_states) + "_hidden_" + str(seed) + '_seed_arhmm_s_KF_' + str(kf_ind) + '.npz'
    fname = os.path.join(out_dir, fname) 

    fname_ll = run_uid + "_" + str(n_hidden_states) + "_hidden_" + str(seed) + '_seed_arhmm_s_LLs_KF_' + str(kf_ind) + '.npz'
    fname_ll = os.path.join(out_dir, fname_ll)

    km = getattr(arhmm_s.dynamics, "kin_model", None)
    np.savez(arhmm_logits_fname(out_dir, run_uid, n_hidden_states, seed, kf_ind), ar_logits=arhmm_ar_logits(arhmm_s),
             pseudo_count=ARCategoricalObservations.pseudo_count,
             pseudo_support=ARCategoricalObservations.pseudo_support is not None)
    if write_init or not os.path.exists(fname):
        np.savez(fname, init_start=init_start, logpi0_start=logpi0_start, log_Ps_start=log_Ps_start, W1_start=Rs_start[0],
                 b1_start=Rs_start[1], W2_start=Rs_start[2], b2_start=Rs_start[3], **({} if km is None else km.params()))
    # train_ll / test_ll: per-step LLs of the fitted ARHMM (used by plot_ll_comparison.py); log_likelihood is the EM trace.
    if km is None:
        test_x = _as_list_x(test_positions) if test_positions is not None else None
    else:
        test_x = kin_datas(test_positions, kin_test) if test_positions is not None else None
    train_ll, test_ll = arhmm_split_ll(arhmm_s, list_x), arhmm_split_ll(arhmm_s, test_x)
    extra = {}
    if km is not None:
        # comparable LL: joint minus the one-intention kinematic null fitted on the train animals
        kin_null = pooled(kin_train, kin_cfg)
        n_tr = sum(len(x) for x in list_x)
        extra = dict(train_ll_joint=train_ll, test_ll_joint=test_ll)
        train_ll = train_ll - _kin_null_split(kin_null, kin_train, [len(x) for x in list_x]) / n_tr
        if test_x:
            test_ll = test_ll - _kin_null_split(kin_null, kin_test, [len(x) for x in test_x]) / \
                sum(len(x) for x in test_x)
    print("ARHMM train LL", train_ll, "test LL", test_ll)
    np.savez(fname_ll, log_likelihood=lls_arhmm, train_ll=train_ll, test_ll=test_ll,
             n_train_steps=int(sum(len(x) for x in list_x)),
             n_test_steps=int(sum(len(x) for x in test_x)) if test_x else 0, **extra)
    return arhmm_s


def arhmm_fname(out_dir, run_uid, n_hidden, seed, kf_ind):
    return os.path.join(out_dir, run_uid + "_" + str(n_hidden) + "_hidden_" + str(seed) + '_seed_arhmm_s_KF_' + str(kf_ind) + '.npz')


def arhmm_ll_fname(out_dir, run_uid, n_hidden, seed, kf_ind):
    return os.path.join(out_dir, run_uid + "_" + str(n_hidden) + "_hidden_" + str(seed) + '_seed_arhmm_s_LLs_KF_' + str(kf_ind) + '.npz')


def arhmm_logits_fname(out_dir, run_uid, n_hidden, seed, kf_ind):
    return os.path.join(out_dir, run_uid + "_" + str(n_hidden) + "_hidden_" + str(seed) + '_seed_arhmm_s_logits_KF_' + str(kf_ind) + '.npz')


def _has_split_lls(path, key="test_ll"):
    if not os.path.exists(path):
        return False
    with np.load(path, allow_pickle=True) as f:
        return key in f.files


def set_pseudo_count(yml_conf, T=None):
    """ARCategoricalObservations pseudo-count / support from the YAML (support "T" needs the fold's T)."""
    ARCategoricalObservations.pseudo_count = float(yml_conf.get("arhmm_pseudo_count", 1e-5))
    ARCategoricalObservations.pseudo_floor = float(yml_conf.get("arhmm_pseudo_floor", 1e-5))
    sup = yml_conf.get("arhmm_pseudo_support")
    if sup in (None, False, "none", "all"):
        ARCategoricalObservations.pseudo_support = None
    elif sup == "T":
        Td = np.asarray(T.todense() if hasattr(T, "todense") else T)
        ARCategoricalObservations.pseudo_support = Td.sum(axis=1) > 0           # (S, S): x -> x' reachable
    else:
        raise ValueError(f"arhmm_pseudo_support must be null or 'T', got {sup!r}")


def add_action_lls(yml_conf, data, train_index, test_index, kf_ind, arhmm_s=None):
    """Write train_ll_act / test_ll_act into the fold's LL file (ARHMM re-scored as an action model)."""
    from naturenet.models.irl.run_swirl_kfold import fold_trans_prob, load_arhmm, prepare_data  # (import cycle)
    out_dir, run_uid, seed, K = yml_conf["out_dir"], yml_conf["run_uid"], yml_conf["seed"], yml_conf["n_hidden_init"]
    data = prepare_data(yml_conf, data)
    T = fold_trans_prob(yml_conf, data, train_index)
    if arhmm_s is not None:   # parameters of this fit (consistent with its logits)
        t = arhmm_s.transitions
        params = (np.asarray(arhmm_s.init_state_distn.log_pi0), np.asarray(t.log_Ps),
                  (np.asarray(t.W1), np.asarray(t.b1), np.asarray(t.W2), np.asarray(t.b2)))
        old = load_arhmm(out_dir, run_uid, K, seed, kf_ind)
        d = float(np.max(np.abs(np.asarray(old[1]) - params[1])))
        if d > 1e-6:
            print(f"  note: re-fit ARHMM log_Ps differ from the existing init file by {d:.2e} (init file kept)")
    else:
        params = load_arhmm(out_dir, run_uid, K, seed, kf_ind)
    with np.load(arhmm_logits_fname(out_dir, run_uid, K, seed, kf_ind)) as f:
        logits = f["ar_logits"]
    tr, te, _, _ = arhmm_action_lls(params, logits, T, data, train_index, test_index)
    path = arhmm_ll_fname(out_dir, run_uid, K, seed, kf_ind)
    with np.load(path, allow_pickle=True) as f:
        out = {k: f[k] for k in f.files}
    out.update(train_ll_act=tr, test_ll_act=te)
    np.savez(path, **out)
    print(f"ARHMM fold {kf_ind} action LL: train {tr:.4f} test {te:.4f} (state LL: train {float(out['train_ll']):.4f} "
          f"test {float(out['test_ll']):.4f})")


def run_arhmm(yml_conf, data=None, folds_to_run=None):
    """
    ARHMM init per grouped fold, trained on the SAME aligned state sequences (x = states[:, 1:-1])
    and the SAME animal-grouped folds that run_swirl_kfold uses.
    """
    seed = yml_conf["seed"]
    n_hidden = yml_conf["n_hidden_init"]
    emission_dim = yml_conf["emission_dim"]
    latent_state_dim = yml_conf["latent_state_dim"]
    n_states = yml_conf["n_states"]
    out_dir = yml_conf["out_dir"]
    run_uid = yml_conf["run_uid"]
    overwrite = yml_conf.get("overwrite", False)
    folds_to_run = folds_to_run if folds_to_run is not None else yml_conf.get("folds_to_run")

    data = data if data is not None else load_swirl_dataset(yml_conf, allow_variable=True)
    xs, _, _ = align_data(data)
    folds = get_folds(yml_conf, data["groups"])
    if data.get("variable", False):   # padded rows -> per-sequence arrays of the real steps (ssm takes lists)
        n_real = step_mask(data, offset=1).sum(1)
        xs_rows = np.empty(len(xs), dtype=object)
        for i in range(len(xs)):
            xs_rows[i] = xs[i, :n_real[i]]
    else:
        xs_rows = xs
    kcfg = kin_conf(yml_conf)
    kin = None if kcfg is None else cached_kinematics(yml_conf, data, kcfg, offset=1, shape=np.asarray(xs).shape)

    for kf_ind, (train_index, test_index) in enumerate(folds):
        if folds_to_run is not None and kf_ind not in as_list(folds_to_run):
            continue
        f_init = arhmm_fname(out_dir, run_uid, n_hidden, seed, kf_ind)
        f_ll = arhmm_ll_fname(out_dir, run_uid, n_hidden, seed, kf_ind)
        want_act = bool(yml_conf.get("arhmm_action_ll", True))
        has_ll = os.path.exists(f_init) and _has_split_lls(f_ll)
        has_act = has_ll and _has_split_lls(f_ll, "test_ll_act")
        has_te = len(test_index) > 0
        if not overwrite and has_ll and (has_act or not want_act):
            print("ARHMM fold", kf_ind, "exists; skipping")
            continue
        if not overwrite and has_ll and os.path.exists(arhmm_logits_fname(out_dir, run_uid, n_hidden, seed, kf_ind)):
            print("ARHMM fold", kf_ind, "exists; adding action LLs from the saved AR tables")
            add_action_lls(yml_conf, data, train_index, test_index if has_te else None, kf_ind)
            continue
        # Re-fit: new folds, folds whose LL file predates the train / test LLs, or (older runs) no saved AR tables.
        print("ARHMM fold", kf_ind, ":", len(train_index), "train /", len(test_index), "test sequences")
        if yml_conf.get("arhmm_pseudo_support") == "T":
            from naturenet.models.irl.run_swirl_kfold import fold_trans_prob, prepare_data  # (import cycle)
            set_pseudo_count(yml_conf, fold_trans_prob(yml_conf, prepare_data(yml_conf, data), train_index))
        else:
            set_pseudo_count(yml_conf)
        npr.seed(seed + kf_ind)
        arhmm_s = run_arhmm_internal(seed, emission_dim, n_hidden, latent_state_dim, n_states, xs_rows[train_index],
                                     out_dir, run_uid, kf_ind, test_positions=xs_rows[test_index] if has_te else None,
                                     kin_train=None if kin is None else kin[train_index],
                                     kin_test=None if kin is None or not has_te else kin[test_index], kin_cfg=kcfg,
                                     epochs=int(yml_conf.get("arhmm_epochs", 100)),
                                     write_init=overwrite or not os.path.exists(f_init))
        if want_act:
            add_action_lls(yml_conf, data, train_index, test_index if has_te else None, kf_ind, arhmm_s=arhmm_s)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    parser.add_argument("--fold", type=int, nargs="*", default=None, help="Only run these fold indices.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_arhmm(yml_conf, folds_to_run=args.fold)
