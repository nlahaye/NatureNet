
#Code adapted from https://github.com/BRAINML-GT/SWIRL
from naturenet.models.swirl.run_arhmm import run_arhmm
from naturenet.models.swirl.swirl_training import *
from naturenet.models.swirl.swirl_training_top_level import *
from naturenet.models.swirl.swirl_utils import *

import numpy as np
import numpy.random as npr
from scipy.special import logsumexp
import os
import pickle
import sparse

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
from jax.lib import xla_bridge
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from sit_fuse.utils import read_yaml

import argparse

 
#K = 2 number of hidden states
#D_obs = 1 number of observed dimensions
#D = number of latent dimensions
#C = 25 number of states

def run_swirl_init(yml_conf):

    seed = yml_conf["seed"]
    
    n_hidden_init = yml_conf["n_hidden_init"]

    trans_prob_fpath = yml_conf["trans_probs"]
    actions_fpath = yml_conf["actions"]
    positions_fpath = yml_conf["positions"]
    out_dir = yml_conf["out_dir"]
    run_uid = yml_conf["run_uid"]

    trans_prob = sparse.load_npz(trans_prob_fpath)
    actions = np.load(actions_fpath, allow_pickle=True)
    positions = np.load(positions_fpath, allow_pickle=True)

    for i in range(len(positions)):
        print(len(positions[i]))

    positions = np.array(positions[:-1])
    actions  = np.array(actions[:-1])

    n_states, n_actions, _ = trans_prob.shape

    #TODO train test split

    # Compute the prev_state_map based on the transitions
 
    #next_state_map_fname = os.path.join(out_dir, run_uid + "_next_state_map_init.pkl")
    prev_state_map_fname = os.path.join(out_dir, run_uid + "_prev_state_map_init.pkl")

 
    #if os.path.exists(next_state_map_fname):
    #    next_state_map = np.load(next_state_map_fname, allow_pickle=True)
    #else: 
    #    print("Computing next state map")
    #    next_state_map = compute_next_state_map(trans_prob, n_states, n_actions)
    #    with open(next_state_map_fname, "wb") as f:
    #         pickle.dump(next_state_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.exists(prev_state_map_fname):
        prev_state_map = np.load(prev_state_map_fname, allow_pickle=True)
    else:
        print("Computing prev state map")
        prev_state_map = compute_prev_state_map(trans_prob, n_states, n_actions)

        with open(prev_state_map_fname, "wb") as f:
             pickle.dump(prev_state_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    r1 = np.zeros((n_states, n_actions))
    r2 = np.zeros((n_states, n_actions))

    np.random.seed(seed)
    if n_hidden_init == 3:
        r3 = npr.rand(n_states)[:, None]
        r3 = jnp.tile(r3, (1, n_actions))
        R_start = np.array([r1, r2, r3])
        R_start2 = R_start.mean(axis=-1)
    elif n_hidden_init == 2:
        r3 = npr.rand(n_states)[:, None]
        r3 = jnp.tile(r3, (1, n_actions))
        R_start = np.array([r1, r3])
        R_start2 = R_start.mean(axis=-1)

  
    arhmm_s_fname = os.path.join(out_dir, run_uid + "_" + str(n_hidden_init) + "_hidden_" + str(seed) + '_seed_arhmm_s.npz')
    arhmm_s_params = np.load(arhmm_s_fname, allow_pickle=True)
    logpi0_start = arhmm_s_params['logpi0_start']
    log_Ps_start = arhmm_s_params['log_Ps_start']
    Rs_start = arhmm_s_params['W1_start'], arhmm_s_params['b1_start'], arhmm_s_params['W2_start'], arhmm_s_params['b2_start']

    print("Preprocessing variables") 
    all_xs_prev = preprocess_xs_prev_np(positions[:, 1:], positions[:, :-1], prev_state_map, n_actions, n_states)
    all_xohs = vmap(one_hotx_partial)(positions[:, 1:], n_states)
    all_xohs_prev = vmap(one_hotx_partial)(positions[:, :-1], n_states)
    all_xohs2 = vmap(one_hotx2_partial)(positions[:, 1:], all_xs_prev, n_states)
    all_aohs = vmap(one_hota_partial)(actions[:, 1:], n_actions)
 
    print(xla_bridge.get_backend().platform)
    temps = jnp.array([1] + [1] * (n_hidden - 1))

    # S-1
    print("Training initial model step 1")
    new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_temp(jnp.array(logpi0_start), jnp.array(log_Ps_start), Rs_start, jnp.array(R_start2)[:, None], temps, 50, init=False, trans=False)
    fname = run_uid + "_time_interval_run_" + str(n_hidden_init) + '_' + str(seed) + "_naturenet_init0.npz"
    fname = os.path.join(out_dir, fname)
    jnp.savez(fname, new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=np.array(new_Rs, dtype=object), new_reward=new_reward, LL_list=LL_list, temps=temps) 

    print("Training initial model step 2")
    new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_temp(jnp.array(new_logpi0), jnp.array(new_log_Ps), new_Rs, jnp.array(new_reward), temps, 30)
    new_reward = normalize_reward(new_reward)
    new_reward = new_reward[[1, 0, 2], ...] #TODO - viz this
    fname = run_uid + "_time_interval_run_" + str(n_hidden_init) + '_' + str(seed) + "_naturenet_init1.npz"
    fname = os.path.join(out_dir, fname)
    jnp.savez(fname, new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=np.array(new_Rs, dtype=object), new_reward=new_reward, LL_list=LL_list, temps=temps)

    #TODO connect observed environmental conditions to emissions / emissions dimensionality


def run_swirl_final(yml_conf):

    seed = yml_conf["seed"]

    n_hidden = yml_conf["n_hidden"]
    emission_dim = yml_conf["emission_dim"]

    trans_prob_fpath = yml_conf["trans_probs"]
    actions_fpath = yml_conf["actions"]
    positions_fpath = yml_conf["positions"]
    out_dir = yml_conf["out_dir"]
 
    run_uid = yml_conf["run_uid"]
    arhmm_s_fname = os.path.join(out_dir, run_uid + "_" + str(n_hidden_init) + "_hidden_" + str(seed) + '_seed_arhmm_s.npz')
    time_interval_params = run_uid + "_time_interval_run_" + str(n_hidden) + '_' + str(seed) + "_naturenet_init1.npz"

    trans_prob = sparse.load_npz(trans_prob_fpath)
    actions = np.load(actions_fpath, allow_pickle=True)
    positions = np.load(positions_fpath, allow_pickle=True)

    positions = np.array(positions[:-1])
    actions  = np.array(actions[:-1])

    n_states, n_actions, _ = trans_prob.shape

    time_interval_learnt_params = jnp.load(time_interval_params, allow_pickle=True)
    init_reward = time_interval_learnt_params['new_reward']

    arhmm_s_params = np.load(arhmm_s_fname, allow_pickle=True)
    logpi0_start = arhmm_s_params['arr_0']
    log_Ps_start = arhmm_s_params['arr_1']
    Rs_start = arhmm_s_params['arr_2'], arhmm_s_params['arr_3'], arhmm_s_params['arr_4'], arhmm_s_params['arr_5']
 
    # Compute the prev_state_map based on the transitions

    #next_state_map_fname = os.path.join(out_dir, run_uid + "_next_state_map.pkl")
    prev_state_map_fname = os.path.join(out_dir, run_uid + "_prev_state_map.pkl")


    #if os.path.exists(next_state_map_fname):
    #    next_state_map = np.load(next_state_map_fname, allow_pickle=True)
    #else: 
    #    print("Computing next state map")
    #    next_state_map = compute_next_state_map(trans_prob, n_states, n_actions)
    #    with open(next_state_map_fname, "wb") as f:
    #         pickle.dump(next_state_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.exists(prev_state_map_fname):
        prev_state_map = np.load(prev_state_map_fname, allow_pickle=True)
    else:
        print("Computing prev state map")
        prev_state_map = compute_prev_state_map(trans_prob, n_states, n_actions)

        with open(prev_state_map_fname, "wb") as f:
             pickle.dump(prev_state_map, f, protocol=pickle.HIGHEST_PROTOCOL)

 
    r1 = init_reward[0].T
    r1 = normalize(r1)
    r1 = np.tile(r1, (1, n_actions))
    r2 = init_reward[1].T
    r2 = normalize(r2)
    r2 = np.tile(r2, (1, n_actions))
    np.random.seed(seed)
    #TODO - generalize this functionality for n_hidden = some K
    if n_hidden == 3:
        r3 = npr.rand(n_states)[:, None]
        r3 = jnp.tile(r3, (1, n_actions))
        R_start = np.array([r1, r2, r3])
        R_start2 = R_start.mean(axis=-1)
    elif n_hidden == 2:
        R_start = np.array([r1, r2])
        R_start2 = R_start.mean(axis=-1)
    elif n_hidden == 4:
        r3 = npr.rand(n_states)[:, None]
        r3 = jnp.tile(r3, (1, n_actions))
        r4 = npr.rand(n_states)[:, None]
        r4 = jnp.tile(r4, (1, n_actions))
        R_start = np.array([r1, r2, r3, r4])
        R_start2 = R_start.mean(axis=-1)
    elif n_hidden == 5:
        r3 = npr.rand(n_states)[:, None]
        r3 = jnp.tile(r3, (1, n_actions))
        r4 = npr.rand(n_states)[:, None]
        r4 = jnp.tile(r4, (1, n_actions))
        r5 = npr.rand(n_states)[:, None]
        r5 = jnp.tile(r5, (1, n_actions))
        R_start = np.array([r1, r2, r3, r4, r5])
        R_start2 = R_start.mean(axis=-1)

   
    print("Finalizing preprocessing") 
    all_xs_prev = preprocess_xs_prev_np(positions[:, 1:], positions[:, :-1], prev_state_map, n_actions, n_states)
    all_xohs = vmap(one_hotx_partial)(positions[:, 1:], n_states)
    all_xohs_prev = vmap(one_hotx_partial)(positions[:, :-1], n_states)
    all_xohs2 = vmap(one_hotx2_partial)(positions[:, 1:], all_xs_prev, n_states)
    all_aohs = vmap(one_hota_partial)(actions[:, 1:], n_actions)

    print(xla_bridge.get_backend().platform)
    temps = jnp.array([0.01] + [1] * (n_hidden - 1))

    # S-2
    print("Training second model step 1")
    new_logpi02, new_log_Ps2, new_Rs2, new_reward2, LL_list2 = em_train2_naturenet(jnp.array(logpi0_start), jnp.array(log_Ps_start), Rs_start, jnp.array(R_start)[:, None].reshape(n_hidden, emission_dim, n_states*n_actions), 50)
    fname = run_uid + "_" + str(n_hidden_init) + '_' + str(seed) + "_naturenet_iter2_init.npz" 
    fname = os.path.join(out_dir, fname) 
    jnp.savez(fname, new_logpi0=new_logpi02, new_log_Ps=new_log_Ps2, new_Rs=np.array(new_Rs2, dtype=object), new_reward=new_reward2, LL_list=LL_list2, temps=temps)

    print("Training second model step 2")
    new_logpi02, new_log_Ps2, new_Rs2, new_reward2, LL_list2 = em_train2_temp(jnp.array(new_logpi02), jnp.array(new_log_Ps2), new_Rs2, jnp.array(new_reward2), temps, 30)
    fname = run_uid + "_" + str(n_hidden_init) + '_' + str(seed) + "_naturenet_iter2.npz"
    fname = os.path.join(out_dir, fname)
    jnp.savez(fname, new_logpi0=new_logpi02, new_log_Ps=new_log_Ps2, new_Rs=np.array(new_Rs2, dtype=object), new_reward=new_reward2, LL_list=LL_list2, temps=temps)
 
    # # S-1
    print("Training third model step 1")
    new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_naturenet(jnp.array(logpi0_start), jnp.array(log_Ps_start), Rs_start, jnp.array(R_start2)[:, None], 50)
    fname = run_uid + "_" + str(n_hidden_init) + '_' + str(seed) + "_naturenet_iter2_temp_init.npz"
    fname = os.path.join(out_dir, fname)
    jnp.savez(fname, new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=np.array(new_Rs, dtype=object), new_reward=new_reward, LL_list=LL_list, temps=temps)

    print("Training third model step 2")
    new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_temp(jnp.array(new_logpi0), jnp.array(new_log_Ps), new_Rs, jnp.array(new_reward), temps, 30)
    fname = run_uid + "_" + str(n_hidden_init) + '_' + str(seed) + "_naturenet_iter2_temp_2.npz"
    fname = os.path.join(out_dir, fname)
    jnp.savez(fname, new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=np.array(new_Rs, dtype=object), new_reward=new_reward, LL_list=LL_list, temps=temps)
 
    # Load S-2 params
    print("Load params and set reward values")
    fname = run_uid + "_" + str(n_hidden_init) + '_' + str(seed) + "_naturenet_iter2.npz"
    fname = os.path.join(out_dir, fname)
    params2 = jnp.load(fname, allow_pickle=True)
    new_logpi02, new_log_Ps2, new_Rs2, new_reward2, LL_list2 = params2['new_logpi0'], params2['new_log_Ps'], params2['new_Rs'], params2['new_reward'], params2['LL_list']

  
    #TODO - change to all, train, test
    print("Final computations")
    LL, train_LL, test_LL, jax_path_vmap = learnt_LL2(new_logpi02, new_log_Ps2, new_Rs2, new_reward2, temps, n_states, n_actions, trans_prob, all_xohs, all_aohs, all_xohs, all_aohs, all_aohs, all_xohs)
    print(LL, train_LL, test_LL)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)
 
    if yml_conf["run_arhmm"]:
        run_arhmm(yml_conf)    

    if yml_conf["run_swirl_init"]:
        run_swirl_init(yml_conf)
 
    if yml_conf["run_swirl_final"]:
        run_swirl_final(yml_conf)



