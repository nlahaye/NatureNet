
#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
ARHMM initialisation for NN-SWIRL ("recurrent" transitions -> Rs_start).

Same fixes as run_arhmm_kfold.py: aligned states (x = states[:, 1:-1]), animal-grouped folds shared
with every other K-fold script, seeded per fold, finished folds skipped. ``run_arhmm_holdout`` writes the
init for the final NN run (run_swirl_nn.py) from the TRAIN animals of the holdout split only.
"""

import numpy as np
import numpy.random as npr
import os

import argparse

from sit_fuse.utils import read_yaml

from naturenet.models.irl.swirl_data import align_for_swirl, as_list, get_folds, holdout_split, load_swirl_dataset


def arhmm_nn_fname(out_dir, run_uid, n_hidden, seed, kf_ind=None):
    suffix = '_seed_arhmm_nn_s.npz' if kf_ind is None else '_seed_arhmm_nn_s_KF_' + str(kf_ind) + '.npz'
    return os.path.join(out_dir, run_uid + "_" + str(n_hidden) + "_hidden_" + str(seed) + suffix)


def _has_split_lls(path):
    if not os.path.exists(path):
        return False
    with np.load(path, allow_pickle=True) as f:
        return "test_ll" in f.files


def run_arhmm_internal(seed, emission_dim, n_hidden_states, latent_state_dim, n_states, positions, out_dir, run_uid,
                       kf_ind=None, num_init_iters=100, test_positions=None):
    from ssm.swirl import ARHMMs

    arhmm_s = ARHMMs(emission_dim, n_hidden_states, latent_state_dim, n_states,
                     transitions="recurrent", dynamics="arcategorical", single_subspace=True)
    list_x = [np.expand_dims(np.asarray(p).astype(np.int32), axis=1) for p in positions]
    os.makedirs(out_dir, exist_ok=True)
    lls_arhmm = arhmm_s.initialize(list_x, num_init_iters=num_init_iters)

    fname = arhmm_nn_fname(out_dir, run_uid, n_hidden_states, seed, kf_ind)
    np.savez(fname, init_start=arhmm_s.init_state_distn.initial_state_distn,
             logpi0_start=arhmm_s.init_state_distn.log_pi0, log_Ps_start=arhmm_s.transitions.log_Ps,
             Rs_start=arhmm_s.transitions.Rs)
    # train_ll / test_ll: per-step LLs (ARHMMs.log_likelihood / total steps) for plot_ll_comparison.py.
    test_x = [np.expand_dims(np.asarray(p).astype(np.int32), axis=1) for p in test_positions] \
        if test_positions is not None and len(test_positions) else None
    n_tr = int(sum(len(x) for x in list_x))
    n_te = int(sum(len(x) for x in test_x)) if test_x else 0
    train_ll = float(arhmm_s.log_likelihood(list_x)) / n_tr
    test_ll = float(arhmm_s.log_likelihood(test_x)) / n_te if test_x else np.nan
    print("ARHMM-nn train LL", train_ll, "test LL", test_ll)
    np.savez(fname.replace("_seed_arhmm_nn_s", "_seed_arhmm_nn_s_LLs"), log_likelihood=lls_arhmm,
             train_ll=train_ll, test_ll=test_ll, n_train_steps=n_tr, n_test_steps=n_te)


def _conf(yml_conf):
    return (yml_conf["seed"], yml_conf["n_hidden_init"], yml_conf["emission_dim"], yml_conf["latent_state_dim"],
            yml_conf["n_states"], yml_conf["out_dir"], yml_conf["run_uid"])


def run_arhmm(yml_conf, data=None, folds_to_run=None):
    """Per-fold ARHMM on the train animals of each shared grouped fold."""
    seed, K, D, L, S, out_dir, run_uid = _conf(yml_conf)
    folds_to_run = folds_to_run if folds_to_run is not None else yml_conf.get("folds_to_run")
    data = data if data is not None else load_swirl_dataset(yml_conf)
    xs, _, _ = align_for_swirl(data["states"], data["actions"])
    for kf_ind, (train_index, test_index) in enumerate(get_folds(yml_conf, data["groups"])):
        if folds_to_run is not None and kf_ind not in as_list(folds_to_run):
            continue
        f_model = arhmm_nn_fname(out_dir, run_uid, K, seed, kf_ind)
        if not yml_conf.get("overwrite", False) and os.path.exists(f_model) and \
                _has_split_lls(f_model.replace("_seed_arhmm_nn_s", "_seed_arhmm_nn_s_LLs")):
            print("ARHMM-nn fold", kf_ind, "exists; skipping")
            continue
        print("ARHMM-nn fold", kf_ind, ":", len(train_index), "train /", len(test_index), "test sequences")
        npr.seed(seed + kf_ind)
        run_arhmm_internal(seed, D, K, L, S, xs[train_index], out_dir, run_uid, kf_ind,
                           int(yml_conf.get("arhmm_init_iters", 100)),
                           test_positions=xs[test_index] if len(test_index) else None)


def run_arhmm_holdout(yml_conf, data=None):
    """ARHMM init for the final NN run (train animals of the holdout split)."""
    seed, K, D, L, S, out_dir, run_uid = _conf(yml_conf)
    data = data if data is not None else load_swirl_dataset(yml_conf)
    xs, _, _ = align_for_swirl(data["states"], data["actions"])
    train_index, test_index, _ = holdout_split(yml_conf, data["groups"])
    npr.seed(seed)
    run_arhmm_internal(seed, D, K, L, S, xs[train_index], out_dir, run_uid, None,
                       int(yml_conf.get("arhmm_init_iters_final", 500)),
                       test_positions=xs[test_index] if test_index is not None and len(test_index) else None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    parser.add_argument("--fold", type=int, nargs="*", default=None, help="Only run these fold indices.")
    parser.add_argument("--holdout", action="store_true", help="Init for the final run instead of K-fold.")
    args = parser.parse_args()

    yml_conf = read_yaml(args.yaml)
    if args.holdout:
        run_arhmm_holdout(yml_conf)
    else:
        run_arhmm(yml_conf, folds_to_run=args.fold)
