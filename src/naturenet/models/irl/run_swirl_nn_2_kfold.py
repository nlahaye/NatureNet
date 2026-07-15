
#Code adapted from https://github.com/BRAINML-GT/SWIRL
from naturenet.models.irl.run_arhmm import run_arhmm
from naturenet.models.irl.swirl_training import *
from naturenet.models.irl.swirl_training_top_level_nn import *
from naturenet.models.irl.swirl_utils import *
from naturenet.models.irl.swirl_nn_model import *

from sklearn.model_selection import KFold

import numpy as np
import numpy.random as npr
from scipy.special import logsumexp
import os
import pickle
import sparse

import sparse
import scipy

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit

#from jax.lib import xla_bridge

import jax.extend
 
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

from sit_fuse.utils import read_yaml

import argparse

 
#K = 2 number of hidden states
#D_obs = 1 number of observed dimensions
#D = number of latent dimensions
#C = 25 number of states

def get_reward_m(trans_probs, R_params, apply_fn):
    n_states, n_actions, _ = trans_probs.shape
    reshape_func = lambda x: (jnp.tile(jnp.expand_dims(x, axis=-1), (1,) * (x.ndim) + (n_states,)) / n_states).reshape(*x.shape[:-1], x.shape[-1] * x.shape[-1])
    def get_reward_single(curr_s):
        one_hot_input = jax.nn.one_hot(curr_s, n_states)
        # Combine one-hot encodings
        return apply_fn({'params': R_params}, reshape_func(one_hot_input))
        
    reward_net = vmap(get_reward_single)(jnp.arange(n_states)).transpose(1, 0, 2)
    return reward_net


def run_swirl_nn(yml_conf):

    seed = yml_conf["seed"]

    n_hidden = yml_conf["n_hidden"]
    n_hidden_init = yml_conf["n_hidden_init"]
    emission_dim = yml_conf["emission_dim"]

    trans_prob_fpath = yml_conf["trans_probs"]
    actions_fpath = yml_conf["actions"]
    positions_fpath = yml_conf["positions"]
    out_dir = yml_conf["out_dir"]
 
    run_uid = yml_conf["run_uid"]
    arhmm_s_fname = os.path.join(out_dir, run_uid + "_" + str(n_hidden_init) + "_hidden_" + str(seed) + '_seed_arhmm_s.npz')
    #time_interval_params = run_uid + "_time_interval_run_" + str(n_hidden_init) + '_' + str(seed) + "_naturenet_init1.npz"

    trans_probs = sparse.load_npz(trans_prob_fpath)

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
            positions_final.append(positions[j])
            actions_final.append(actions[j])
    actions = actions_final
    positions = positions_final
    positions = np.array(positions)
    actions  = np.array(actions)
 

    n_states, n_actions, _ = trans_probs.shape


    arhmm_s_params = np.load(arhmm_s_fname, allow_pickle=True)
    logpi0_start = arhmm_s_params['logpi0_start']
    log_Ps_start = arhmm_s_params['log_Ps_start']
    Rs_start = arhmm_s_params["Rs_start"] #arhmm_s_params['W1_start'], arhmm_s_params['b1_start'], arhmm_s_params['W2_start'], arhmm_s_params['b2_start']
  
    rng = jax.random.PRNGKey(0)
    hidden_size = 32
    learning_rate = 5e-3

    # Initialize the model and training state
    R_state = create_train_state(rng, learning_rate, n_states, n_hidden, hidden_size, n_actions)
    R_state2 = create_train_state(rng, learning_rate, n_states, n_hidden, hidden_size, n_actions) 
 
    trans_probs = trans_probs.todense()
 
    new_trans_probs = np.zeros((n_states * n_states, n_actions, n_states * n_states))
    for s_prev in range(n_states):
        for s in range(n_states):
            for a in range(n_actions):
                for s_prime in range(n_states):
                    if trans_probs[s, a, s_prime] > 0:
                        new_trans_probs[s * n_states + s_prev, a, s_prime * n_states + s] = trans_probs[s, a, s_prime]
  
    print("Finalizing preprocessing") 
    all_xohs = vmap(one_hotx_partial_nn)(positions[:, 1:])
    all_xohs2, all_xs2 = vmap(one_hotx2_partial_nn)(positions[:, 1:], jnp.roll(positions[:, 1:], 1))
    all_aohs = vmap(one_hota_partial_nn)(actions[:, 1:])

    trans_probs = trans_probs.astype(jnp.bfloat16)
    new_trans_probs = new_trans_probs.astype(jnp.bfloat16)
    all_xohs2 = all_xohs2.astype(jnp.bfloat16)
    all_xohs = all_xohs.astype(jnp.bfloat16)
    all_aohs = all_aohs.astype(jnp.bfloat16)
    logpi0_start = jnp.array(logpi0_start, dtype = jnp.bfloat16)
    Rs_start = jnp.array(Rs_start, dtype = jnp.bfloat16)


    kf = KFold(n_splits=5)
    kf_ind = 0
    for train_index, test_index in kf.split(all_aohs):


        train_aohs, test_aohs = all_aohs[train_index], all_aohs[test_index]
        train_xohs, test_xohs = all_xohs[train_index], all_xohs[test_index]
        train_xohs2, test_xohs2 = all_xohs2[train_index], all_xohs2[test_index]

 
        ## S-2
        print(logpi0_start.shape, log_Ps_start.shape, len(Rs_start))
        new_logpi02, new_log_Ps2, new_Rs2, new_R_state2, LL_list2 = em_train_jaxopt_netadam2(train_xohs2, train_xohs, train_aohs, logpi0_start, log_Ps_start, Rs_start, R_state, trans_probs,\
            new_trans_probs, 10, init=False, trans=False)

        new_logpi02 = jnp.array(new_logpi02, dtype=jnp.bfloat16)
        new_log_Ps2 = jnp.array(new_log_Ps2, dtype=jnp.bfloat16)
        new_Rs2 = jnp.array(new_Rs2, dtype=jnp.bfloat16)
        #new_R_state2 = new_R_state2.astype(jnp.bfloat16)
    

        ## jnp.savez(folder + '/DAar1_rand/' + str(K) + '_' + str(seed) + '_NM_DAsa_net2_Ronly.npz', new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=new_Rs, new_R_state=new_R_state.params, LL_list=LL_list)
        new_logpi02, new_log_Ps2, new_Rs2, new_R_state2, LL_list2 = em_train_jaxopt_netadam2(train_xohs2, train_xohs, train_aohs, new_logpi02, new_log_Ps2, new_Rs2, new_R_state2, trans_probs,\
            new_trans_probs, 10)
        jnp.savez(os.path.join(out_dir, str(n_hidden) + '_' + str(seed) +' _MLP_S2_net2_KF_' + str(kf_ind) + '.npz'), new_logpi0=new_logpi02, new_log_Ps=new_log_Ps2, new_Rs=new_Rs2, new_R_state=new_R_state2.params, LL_list=LL_list2)


        reward_m2 = get_reward_m(trans_probs, new_R_state2.params, R_state.apply_fn)
        reward_m2_filtered = np.copy(reward_m2).reshape((n_hidden, n_states, n_actions))
        fname = os.path.join(out_dir, run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_MLP_S2_new2_reward_filtered_KF_" + str(kf_ind) + '.npz')
        jnp.savez(fname, reward_filtered=reward_m2_filtered)

        LL, train_LL, learnt_zs = learnt_LL22(new_logpi02, new_log_Ps2, new_Rs2, new_R_state2.params, R_state.apply_fn, trans_probs, train_aohs, train_xohs, train_xohs2)
        fname = os.path.join(out_dir, run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_MLP_S2_new2_LL_KF_" + str(kf_ind) + '.npz')
        print(LL, train_LL)
        jnp.savez(fname, LL=LL, train_LL=train_LL, learnt_zs=learnt_zs) #test_LL

        LL, train_LL, learnt_zs = learnt_LL22(new_logpi02, new_log_Ps2, new_Rs2, new_R_state2.params, R_state.apply_fn, trans_probs, test_aohs, test_xohs, test_xohs2)
        fname = os.path.join(out_dir, run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_MLP_S2_new2_LL_test_KF_" + str(kf_ind) + ".npz")
        print(LL, train_LL)
        jnp.savez(fname, LL=LL, train_LL=train_LL, learnt_zs=learnt_zs) #test_LL

        kf_ind = kf_ind + 1

        logpi0_start = jnp.array(logpi0_start, dtype=jnp.bfloat16)
        log_Ps_start = jnp.array(log_Ps_start, dtype=jnp.bfloat16)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)
 
    if yml_conf["run_arhmm"]:
        run_arhmm(yml_conf)    
 
    if yml_conf["run_swirl_nn"]:
        run_swirl_nn(yml_conf)

  
