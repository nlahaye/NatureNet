
#Code adapted from https://github.com/BRAINML-GT/SWIRL

import numpy as np
import numpy.random as npr
import os

import argparse

from ssm.swirl import ARHMMs 

from sit_fuse.utils import read_yaml

from sklearn.model_selection import KFold


def run_arhmm_internal(seed, emission_dim, n_hidden_states, latent_state_dim, n_states, positions, out_dir, run_uid, kf_ind):

    #print(seed, emission_dim, n_hidden_states, latent_state_dim, n_states, len(positions), "HERE TEST", positions)
    arhmm_s = ARHMMs( emission_dim, n_hidden_states, latent_state_dim, n_states,\
        transitions="recurrent", dynamics="arcategorical",\
        single_subspace=True) #mlprecurrent

    #positions = np.array(positions)
    #positions = np.array(positions[:-1])
    
    list_x = []
    for path in range(len(positions)):
        #print(path, len(positions[path]))
        p = positions[path]
        list_x.append(np.expand_dims(np.array(p).astype(np.int32), axis=1))
        #for p2 in range(p.shape[0]):
        #    print(p[p2])
        #    list_x.append(p[p2])
            

    #list_x = [row for row in positions[:, :, np.newaxis].astype(int)]
    #print(list_x)
    #print(len(list_x))
    epochs = 100
    lls_arhmm = arhmm_s.initialize(list_x, num_init_iters=epochs)
    init_start = arhmm_s.init_state_distn.initial_state_distn
    logpi0_start = arhmm_s.init_state_distn.log_pi0
    log_Ps_start = arhmm_s.transitions.log_Ps
    Rs_start = arhmm_s.transitions.Rs
    #Rs_start = arhmm_s.transitions.W1, arhmm_s.transitions.b1, arhmm_s.transitions.W2, arhmm_s.transitions.b2

    fname = run_uid + "_" + str(n_hidden_states) + "_hidden_" + str(seed) + '_seed_arhmm_s_KF_' + str(kf_ind) + '.npz'
    fname = os.path.join(out_dir, fname) 

    fname_ll = run_uid + "_" + str(n_hidden_states) + "_hidden_" + str(seed) + '_seed_arhmm_s_LLs_KF_' + str(kf_ind) + '.npz'
    fname_ll = os.path.join(out_dir, fname_ll)
 
    np.savez(fname, init_start=init_start, logpi0_start=logpi0_start, log_Ps_start=log_Ps_start, Rs_start=Rs_start) #W1_start=Rs_start[0], b1_start=Rs_start[1], W2_start=Rs_start[2], b2_start=Rs_start[3])
    np.savez(fname_ll, log_likelihood=lls_arhmm)

def run_arhmm(yml_conf):
 
    seed = yml_conf["seed"]
    n_hidden = yml_conf["n_hidden_init"]
    emission_dim = yml_conf["emission_dim"]
    latent_state_dim = yml_conf["latent_state_dim"]
    n_states = yml_conf["n_states"]
    out_dir = yml_conf["out_dir"]
    run_uid = yml_conf["run_uid"]
    actions_fpath = yml_conf["actions"]
    positions_fpath = yml_conf["positions"]

    actions = []
    positions = []
    max_len = -1
    for i in range(len(actions_fpath)):
        actions_tmp = np.load(actions_fpath[i], allow_pickle=True)
        positions_tmp = np.load(positions_fpath[i], allow_pickle=True)

        for j in range(len(positions_tmp)):
            max_len = max(max_len, len(positions_tmp[j]))

        actions.extend(actions_tmp)
        positions.extend(positions_tmp)

    #for now, at least, all path lengths have to be the same for SWIRL :(
    positions_final = []
    actions_final = []
    for j in range(len(positions)):
        if len(positions[j]) == max_len:
            print(len(positions[j]), len(actions[j]), max_len)
            positions_final.append(positions[j])
            actions_final.append(actions[j])


    kf = KFold(n_splits=5)
    kf_ind = 0
    for train_index, test_index in kf.split(actions_final):

        print(train_index, len(actions))
        actions = np.array(actions_final)[train_index]
        positions = np.array(positions_final)[train_index]

        run_arhmm_internal(seed, emission_dim, n_hidden, latent_state_dim, n_states, positions, out_dir, run_uid, kf_ind)


        kf_ind = kf_ind + 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_arhmm(yml_conf)

