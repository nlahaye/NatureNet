
#Code adapted from https://github.com/BRAINML-GT/SWIRL

import numpy as np
import numpy.random as npr
import os

import argparse

from ssm.swirl import ARHMMs 

from sit_fuse.utils import read_yaml

from naturenet.models.irl.swirl_data import align_for_swirl, holdout_split, load_swirl_dataset


def run_arhmm_internal(seed, emission_dim, n_hidden_states, latent_state_dim, n_states, positions, out_dir, run_uid):

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
    os.makedirs(out_dir, exist_ok=True)
    lls_arhmm = arhmm_s.initialize(list_x, num_init_iters=500)
    init_start = arhmm_s.init_state_distn.initial_state_distn
    logpi0_start = arhmm_s.init_state_distn.log_pi0
    log_Ps_start = arhmm_s.transitions.log_Ps
    Rs_start = arhmm_s.transitions.W1, arhmm_s.transitions.b1, arhmm_s.transitions.W2, arhmm_s.transitions.b2

    fname = run_uid + "_" + str(n_hidden_states) + "_hidden_" + str(seed) + '_seed_arhmm_s.npz'
    fname = os.path.join(out_dir, fname) 

    np.savez(fname, init_start=init_start, logpi0_start=logpi0_start, log_Ps_start=log_Ps_start, W1_start=Rs_start[0], b1_start=Rs_start[1], W2_start=Rs_start[2], b2_start=Rs_start[3])


def run_arhmm(yml_conf, data=None):
    """ARHMM init for the final run, on the aligned states of the TRAIN animals of the holdout split."""
    seed = yml_conf["seed"]
    n_hidden = yml_conf["n_hidden_init"]
    emission_dim = yml_conf["emission_dim"]
    latent_state_dim = yml_conf["latent_state_dim"]
    n_states = yml_conf["n_states"]
    out_dir = yml_conf["out_dir"]
    run_uid = yml_conf["run_uid"]

    data = data if data is not None else load_swirl_dataset(yml_conf)
    xs, _, _ = align_for_swirl(data["states"], data["actions"])
    train_index, _, _ = holdout_split(yml_conf, data["groups"])

    npr.seed(seed)
    run_arhmm_internal(seed, emission_dim, n_hidden, latent_state_dim, n_states, xs[train_index], out_dir, run_uid)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_arhmm(yml_conf)
