
#Code adapted from https://github.com/BRAINML-GT/SWIRL

import numpy as np
import numpy.random as npr
import os

import argparse

from ssm.swirl import ARHMMs 

from sit_fuse.utils import read_yaml

from naturenet.models.irl.swirl_data import align_for_swirl, as_list, get_folds, load_swirl_dataset


def run_arhmm_internal(seed, emission_dim, n_hidden_states, latent_state_dim, n_states, positions, out_dir, run_uid, kf_ind):

    arhmm_s = ARHMMs( emission_dim, n_hidden_states, latent_state_dim, n_states,\
        transitions="mlprecurrent", dynamics="arcategorical",\
        single_subspace=True)

    #positions = np.array(positions)
    #positions = np.array(positions[:-1])

    list_x = []
    for path in range(len(positions)):
        p = positions[path]
        list_x.append(np.expand_dims(np.array(p), axis=1))

        #for p2 in range(p.shape[0]):
        #    print(p[p2])
        #    list_x.append(p[p2])


    #list_x = [row for row in positions[:, :, np.newaxis].astype(int)]
    epochs = 100
    os.makedirs(out_dir, exist_ok=True)
    lls_arhmm = arhmm_s.initialize(list_x, num_init_iters=epochs)
    init_start = arhmm_s.init_state_distn.initial_state_distn
    logpi0_start = arhmm_s.init_state_distn.log_pi0
    log_Ps_start = arhmm_s.transitions.log_Ps
    Rs_start = arhmm_s.transitions.W1, arhmm_s.transitions.b1, arhmm_s.transitions.W2, arhmm_s.transitions.b2

    fname = run_uid + "_" + str(n_hidden_states) + "_hidden_" + str(seed) + '_seed_arhmm_s_KF_' + str(kf_ind) + '.npz'
    fname = os.path.join(out_dir, fname) 

    fname_ll = run_uid + "_" + str(n_hidden_states) + "_hidden_" + str(seed) + '_seed_arhmm_s_LLs_KF_' + str(kf_ind) + '.npz'
    fname_ll = os.path.join(out_dir, fname_ll)

    np.savez(fname, init_start=init_start, logpi0_start=logpi0_start, log_Ps_start=log_Ps_start, W1_start=Rs_start[0], b1_start=Rs_start[1], W2_start=Rs_start[2], b2_start=Rs_start[3]) 
    np.savez(fname_ll, log_likelihood=lls_arhmm)


def arhmm_fname(out_dir, run_uid, n_hidden, seed, kf_ind):
    return os.path.join(out_dir, run_uid + "_" + str(n_hidden) + "_hidden_" + str(seed) + '_seed_arhmm_s_KF_' + str(kf_ind) + '.npz')


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

    data = data if data is not None else load_swirl_dataset(yml_conf)
    xs, _, _ = align_for_swirl(data["states"], data["actions"])
    folds = get_folds(yml_conf, data["groups"])

    for kf_ind, (train_index, test_index) in enumerate(folds):
        if folds_to_run is not None and kf_ind not in as_list(folds_to_run):
            continue
        if not overwrite and os.path.exists(arhmm_fname(out_dir, run_uid, n_hidden, seed, kf_ind)):
            print("ARHMM fold", kf_ind, "exists; skipping")
            continue
        print("ARHMM fold", kf_ind, ":", len(train_index), "train /", len(test_index), "test sequences")
        npr.seed(seed + kf_ind)
        run_arhmm_internal(seed, emission_dim, n_hidden, latent_state_dim, n_states, xs[train_index], out_dir, run_uid, kf_ind)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    parser.add_argument("--fold", type=int, nargs="*", default=None, help="Only run these fold indices.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_arhmm(yml_conf, folds_to_run=args.fold)
