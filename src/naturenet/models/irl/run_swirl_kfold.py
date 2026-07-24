
#Code adapted from https://github.com/BRAINML-GT/SWIRL
from naturenet.models.irl.run_arhmm_kfold import run_arhmm
from naturenet.models.irl.swirl_training import *
from naturenet.models.irl.swirl_training_top_level import *
from naturenet.models.irl.swirl_utils import *

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

def run_swirl_init(yml_conf):

    seed = yml_conf["seed"]
    
    n_hidden_init = yml_conf["n_hidden_init"]

    trans_prob_fpath = yml_conf["trans_probs"]
    actions_fpath = yml_conf["actions"]
    positions_fpath = yml_conf["positions"]
    out_dir = yml_conf["out_dir"]
    run_uid = yml_conf["run_uid"]

    trans_prob = sparse.load_npz(trans_prob_fpath)
 
    trans_prob = trans_prob.todense()

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

    print(len(positions), len(actions))

    #actions = np.load(actions_fpath, allow_pickle=True)
    #positions = np.load(positions_fpath, allow_pickle=True)

    positions = np.array(positions) #[:-1])
    actions  = np.array(actions) #[:-1])

    while len(positions[-1]) < len(positions[0]):
        positions = positions[:-1]
        actions  = actions[:-1]


    global n_states
    global n_actions
    n_states, n_actions, _ = trans_prob.shape

    #TODO train test split

    # Compute the prev_state_map based on the transitions
 
    next_state_map_fname = os.path.join(out_dir, run_uid + "_next_state_map_init.pkl")
    prev_state_map_fname = os.path.join(out_dir, run_uid + "_prev_state_map_init.pkl")

 
    if os.path.exists(next_state_map_fname):
        next_state_map = np.load(next_state_map_fname, allow_pickle=True)
    else: 
        print("Computing next state map")
        next_state_map = compute_next_state_map(trans_prob, n_states, n_actions)
        with open(next_state_map_fname, "wb") as f:
             pickle.dump(next_state_map, f, protocol=pickle.HIGHEST_PROTOCOL)

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
    if n_hidden_init == 1:
        R_start = [r1]
    elif n_hidden_init >= 2:
        R_start = [r1, r2]
        if n_hidden_init > 2:
            for i in range(2, n_hidden_init):
                r = npr.rand(n_states)[:, None]
                r = jnp.tile(r, (1, n_actions))
                R_start.append(r)

    R_start = np.array(R_start)
    R_start2 = R_start.mean(axis=-1)
  
    """
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
    elif n_hidden_init == 4:
        r3 = npr.rand(n_states)[:, None]
        r3 = jnp.tile(r3, (1, n_actions))
        r4 = npr.rand(n_states)[:, None]
        r4 = jnp.tile(r4, (1, n_actions))
        R_start = np.array([r1, r2, r3, r4])
        R_start2 = R_start.mean(axis=-1)
    elif n_hidden_init == 5:
        r3 = npr.rand(n_states)[:, None]
        r3 = jnp.tile(r3, (1, n_actions))
        r4 = npr.rand(n_states)[:, None]
        r4 = jnp.tile(r4, (1, n_actions))
        r5 = npr.rand(n_states)[:, None]
        r5 = jnp.tile(r5, (1, n_actions))
        R_start = np.array([r1, r2, r3, r4, r5])
        R_start2 = R_start.mean(axis=-1)
    """  

    print("Preprocessing variables")


    one_hotx = vmap(partial(one_hotx_partial, n_states=n_states))
    one_hotx2 = vmap(partial(one_hotx2_partial, n_states=n_states, n_actions=n_actions))
    one_hota = vmap(partial(one_hota_partial, n_actions=n_actions))
 

    print(positions.shape, positions.shape, n_actions, n_states)
    all_xs_prev = preprocess_xs_prev_np(positions[:, 1:], positions[:, :-1], prev_state_map, n_actions, n_states)
    all_xohs = vmap(one_hotx)(positions[:, 1:])
    all_xohs_prev = vmap(one_hotx)(positions[:, :-1])
    all_xohs2 = vmap(one_hotx2)(positions[:, 1:], all_xs_prev)
    all_aohs = vmap(one_hota)(actions[:, 1:])
 
    temps = jnp.array([1] + [1] * (n_hidden_init- 1))

    print(all_xs_prev.shape, all_xohs.shape, all_xohs_prev.shape, all_xohs2.shape, all_aohs.shape, "HERE TEST")

    # S-1
    print("Training initial model step 1")
    epochs = 50 #50
    epochs_2 = 30 #100


    kf = KFold(n_splits=5)
    kf_ind = 0
    for train_index, test_index in kf.split(all_aohs):
        #if kf_ind < 3:
        #    kf_ind = kf_ind + 1
        #    continue

        arhmm_s_fname = os.path.join(out_dir, run_uid + "_" + str(n_hidden_init) + "_hidden_" + str(seed) + '_seed_arhmm_s_KF_' + str(kf_ind) + '.npz') #_KF_' + str(kf_ind) + '.npz')
        arhmm_s_params = np.load(arhmm_s_fname, allow_pickle=True)
        logpi0_start = arhmm_s_params['logpi0_start']
        log_Ps_start = arhmm_s_params['log_Ps_start']

        Rs_start = arhmm_s_params['W1_start'], arhmm_s_params['b1_start'], arhmm_s_params['W2_start'], arhmm_s_params['b2_start']



        train_aohs, test_aohs = all_aohs[train_index], all_aohs[test_index]
        train_xohs, test_xohs = all_xohs[train_index], all_xohs[test_index]
        train_xohs2, test_xohs2 = all_xohs2[train_index], all_xohs2[test_index]

        print(train_aohs.shape, test_aohs.shape, train_aohs.shape)
 
        print(train_xohs.shape, prev_state_map.keys(), n_actions, n_states, len(Rs_start), len(R_start2))



        new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_temp(jnp.array(logpi0_start), jnp.array(log_Ps_start), Rs_start, jnp.array(R_start2)[:, None], temps, trans_prob, train_xohs, train_aohs, epochs, init=False, trans=False)


        fname = run_uid + "_swirl_run_" + str(n_hidden_init) + '_' + str(seed) + "_naturenet_init0_KF_" + str(kf_ind) + ".npz"
        fname = os.path.join(out_dir, fname)
        jnp.savez(fname, new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=np.array(new_Rs, dtype=object), new_reward=new_reward, LL_list=LL_list, temps=temps) 

        print("Training initial model step 2")
        new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_temp(jnp.array(new_logpi0), jnp.array(new_log_Ps), new_Rs, jnp.array(new_reward), temps, trans_prob, train_xohs, train_aohs, epochs_2)

        new_reward = normalize_reward(new_reward)
        print(new_reward.shape, "NEW_REWARD")
        #if new_reward.shape[0] >=  3:
        #    new_reward = new_reward[[1, 0, 2], ...] #TODO - viz this
        #elif new_reward.shape[0] == 2:
        #    new_reward = new_reward[[1, 0], ...]
        #else:
        #    new_reward = new_reward[[0], ...]
        
        fname = run_uid + "_swirl_run_" + str(n_hidden_init) + '_' + str(seed) + "_naturenet_init1_KF_" + str(kf_ind) + ".npz"
        fname = os.path.join(out_dir, fname)
        jnp.savez(fname, new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=np.array(new_Rs, dtype=object), new_reward=new_reward, LL_list=LL_list, temps=temps)

        #TODO connect observed environmental conditions to emissions / emissions dimensionality
        kf_ind = kf_ind + 1

def construct_new_trans_probs_limited(trans_probs, prev_state_map, next_state_map, n_state, n_action):
    new_trans_probs = np.zeros((n_state * n_action, n_action, n_state * n_action))
    invalid_indices = np.ones((n_state, n_action), dtype=bool)
    for x in range(n_state):
        for prev_x_i in np.arange(n_action):
            if prev_x_i < len(prev_state_map[x]):
                invalid_indices[x, prev_x_i] = False
            new_state = x * n_action + prev_x_i
            for a in range(n_action):
                next_x = next_state_map[x, a]
                if next_x < 0:
                    continue
                new_next_state = next_x * n_action + prev_state_map[next_x].index(x)
                new_trans_probs[new_state, a, new_next_state] = trans_probs[x, a, next_x]
    return new_trans_probs, invalid_indices

def run_swirl_final(yml_conf):

    seed = yml_conf["seed"]

    n_hidden = yml_conf["n_hidden"]
    n_hidden_init = yml_conf["n_hidden_init"]
    emission_dim = yml_conf["emission_dim"]

    trans_prob_fpath = yml_conf["trans_probs"]
    actions_fpath = yml_conf["actions"]
    positions_fpath = yml_conf["positions"]
    out_dir = yml_conf["out_dir"]
 
    run_uid = yml_conf["run_uid"]

    trans_prob = sparse.load_npz(trans_prob_fpath)

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
 

    n_states, n_actions, _ = trans_prob.shape

    kf = KFold(n_splits=5)
    kf_ind = 0

    one_hotx = vmap(partial(one_hotx_partial, n_states=n_states))
    one_hotx2 = vmap(partial(one_hotx2_partial, n_states=n_states, n_actions=n_actions))
    one_hota = vmap(partial(one_hota_partial, n_actions=n_actions))

 
    all_aohs = vmap(one_hota)(actions[:, 1:])

    trans_prob = trans_prob.todense()
    for train_index, test_index in kf.split(all_aohs):
        #if kf_ind < 3:
        #    kf_ind = kf_ind + 1
        #    continue


        arhmm_s_fname = os.path.join(out_dir, run_uid + "_" + str(n_hidden_init) + "_hidden_" + str(seed) + '_seed_arhmm_s_KF_' + str(kf_ind) + '.npz')
        swirl_params = run_uid + "_swirl_run_" + str(n_hidden_init) + '_' + str(seed) + "_naturenet_init1_KF_" + str(kf_ind) + ".npz"


        swirl_learnt_params = jnp.load(os.path.join(out_dir,swirl_params), allow_pickle=True)
        init_reward = swirl_learnt_params['new_reward']

        arhmm_s_params = np.load(arhmm_s_fname, allow_pickle=True)
        logpi0_start = arhmm_s_params['logpi0_start']
        log_Ps_start = arhmm_s_params['log_Ps_start']
        Rs_start = arhmm_s_params['W1_start'], arhmm_s_params['b1_start'], arhmm_s_params['W2_start'], arhmm_s_params['b2_start']
  
        # Compute the prev_state_map based on the transitions

        next_state_map_fname = os.path.join(out_dir, run_uid + "_next_state_map.pkl")
        prev_state_map_fname = os.path.join(out_dir, run_uid + "_prev_state_map.pkl")


        if os.path.exists(next_state_map_fname):
            next_state_map = np.load(next_state_map_fname, allow_pickle=True)
        else: 
            print("Computing next state map")
            next_state_map = compute_next_state_map(trans_prob, n_states, n_actions)
            with open(next_state_map_fname, "wb") as f:
                 pickle.dump(next_state_map, f, protocol=pickle.HIGHEST_PROTOCOL)

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
        if n_hidden_init == 1:
            R_start = [r1]
        elif n_hidden_init >= 2:
            R_start = [r1, r2]
            if n_hidden_init > 2:
                for i in range(2, n_hidden_init):
                    r = npr.rand(n_states)[:, None]
                    r = jnp.tile(r, (1, n_actions))
                    R_start.append(r)

        R_start = np.array(R_start)
        R_start2 = R_start.mean(axis=-1)
 
 
        #r1 = init_reward[0].T
        #r1 = normalize(r1)
        #r1 = np.tile(r1, (1, n_actions))
        #r2 = init_reward[1].T
        #r2 = normalize(r2)
        #r2 = np.tile(r2, (1, n_actions))
        #np.random.seed(seed)

        #np.random.seed(seed)
        #R_start = [r1, r2]
        #for i in range(2, n_hidden_init):
        #   r = npr.rand(n_states)[:, None]
        #    r = jnp.tile(r, (1, n_actions))
        #    R_start.append(r)

        #R_start = np.array(R_start)
        #R_start2 = R_start.mean(axis=-1)

    
        print("Finalizing preprocessing") 
        all_xs_prev = preprocess_xs_prev_np(positions[:, 1:], positions[:, :-1], prev_state_map, n_actions, n_states)
        all_xohs = vmap(one_hotx)(positions[:, 1:])
        all_xohs_prev = vmap(one_hotx)(positions[:, :-1])
        all_xohs2 = vmap(one_hotx2)(positions[:, 1:], all_xs_prev)

        train_aohs, test_aohs = all_aohs[train_index], all_aohs[test_index]
        train_xohs, test_xohs = all_xohs[train_index], all_xohs[test_index]
        train_xohs2, test_xohs2 = all_xohs2[train_index], all_xohs2[test_index]
        train_xohs_prev, test_xohs_prev = all_xohs_prev[train_index], all_xohs_prev[test_index]
        train_xs_prev, test_xs_prev = all_xs_prev[train_index], all_xs_prev[test_index]

        temps = jnp.array([0.01] + [1] * (n_hidden - 1))

        new_trans_prob, _ = construct_new_trans_probs_limited(trans_prob, prev_state_map, next_state_map, n_states, n_actions)
        #new_trans_prob = new_trans_prob.todense()


        # S-2
        epochs = 50 #50
        epochs_2 = 30
        print("Training second model step 1")
        new_logpi02, new_log_Ps2, new_Rs2, new_reward2, LL_list2 = em_train2_naturenet(jnp.array(logpi0_start), jnp.array(log_Ps_start), Rs_start, jnp.array(R_start)[:, None].reshape(n_hidden, emission_dim, n_states*n_actions),\
             new_trans_prob, train_xohs, train_xohs2, train_aohs, epochs)
        fname = run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_iter2_S2_init_KF_" + str(kf_ind) + ".npz" 
        fname = os.path.join(out_dir, fname) 
        jnp.savez(fname, new_logpi0=new_logpi02, new_log_Ps=new_log_Ps2, new_Rs=np.array(new_Rs2, dtype=object), new_reward=new_reward2, LL_list=LL_list2, temps=temps)

        print("Training second model step 2")
        new_logpi02, new_log_Ps2, new_Rs2, new_reward2, LL_list2 = em_train2_temp(jnp.array(new_logpi02), jnp.array(new_log_Ps2), new_Rs2, jnp.array(new_reward2), temps, new_trans_prob, train_xohs, train_xohs2, train_aohs, epochs_2)
        fname = run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_iter2_S2_KF_" + str(kf_ind) + ".npz"
        fname = os.path.join(out_dir, fname)
        jnp.savez(fname, new_logpi0=new_logpi02, new_log_Ps=new_log_Ps2, new_Rs=np.array(new_Rs2, dtype=object), new_reward=new_reward2, LL_list=LL_list2, temps=temps)
 
        # # S-1
        print("Training third model step 1")
        new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_naturenet(jnp.array(logpi0_start), jnp.array(log_Ps_start), Rs_start, jnp.array(R_start2)[:, None],\
             trans_prob, train_xohs, train_aohs, epochs)
        fname = run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_iter2_S1_temp_init_KF_" + str(kf_ind) + ".npz"
        fname = os.path.join(out_dir, fname)
        jnp.savez(fname, new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=np.array(new_Rs, dtype=object), new_reward=new_reward, LL_list=LL_list, temps=temps)

        print("Training third model step 2")
        new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_temp(jnp.array(new_logpi0), jnp.array(new_log_Ps), new_Rs, jnp.array(new_reward), temps, trans_prob, train_xohs, train_aohs, epochs_2)
        fname = run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_iter2_S1_temp_2_KF_" + str(kf_ind) + ".npz"
        fname = os.path.join(out_dir, fname)
        jnp.savez(fname, new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=np.array(new_Rs, dtype=object), new_reward=new_reward, LL_list=LL_list, temps=temps)


        new_reward = normalize_reward(new_reward)
        print(new_reward.shape, "NEW_REWARD")
        #if new_reward.shape[0] >=  3:
        #    new_reward = new_reward[[1, 0, 2], ...] #TODO - viz this
        #elif new_reward.shape[0] == 2:
        #    new_reward = new_reward[[1, 0], ...]
        #else:
        #    new_reward = new_reward[[0], ...]


        print("Final computations", "S1")
        LL, jax_path_vmap = learnt_LL1(new_logpi0, new_log_Ps, new_Rs, new_reward, temps, n_states, n_actions, trans_prob, train_xohs, train_aohs)
        print(LL)
        fname = os.path.join(out_dir, run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_S1_iter2_LL1_KF_" + str(kf_ind) + ".npz") 
        jnp.savez(fname, LL=LL, jax_path_vmap=jax_path_vmap)

        LL, jax_path_vmap = learnt_LL1(new_logpi0, new_log_Ps, new_Rs, new_reward, temps, n_states, n_actions, trans_prob, test_xohs, test_aohs)
        print(LL)
        fname = os.path.join(out_dir, run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_S1_iter2_LL1_test_KF_" + str(kf_ind) + ".npz")
        jnp.savez(fname, LL=LL, jax_path_vmap=jax_path_vmap)


  
        # Load S-2 params
        print("Load params and set reward values")
        fname = run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_iter2_S2_KF_" + str(kf_ind) + ".npz"
        fname = os.path.join(out_dir, fname)
        params2 = jnp.load(fname, allow_pickle=True)
        new_logpi02, new_log_Ps2, new_Rs2, new_reward2, LL_list2 = params2['new_logpi0'], params2['new_log_Ps'], params2['new_Rs'], params2['new_reward'], params2['LL_list']

        new_reward2 = normalize_reward(new_reward2)
        print(new_reward2.shape, "NEW_REWARD")
        #if new_reward2.shape[0] >=  3:
        #    new_reward2 = new_reward2[[1, 0, 2], ...] #TODO - viz this
        #elif new_reward2.shape[0] == 2:
        #    new_reward2 = new_reward2[[1, 0], ...]
        #else:
        #    new_reward2 = new_reward2[[0], ...]



        #TODO - change to all, train, test
        print("Final computations", "S2")
        LL, jax_path_vmap = learnt_LL2(new_logpi02, new_log_Ps2, new_Rs2, new_reward2, temps, n_states, n_actions, new_trans_prob, train_xohs, train_xohs2, train_aohs)
        print(LL)
        fname = os.path.join(out_dir, run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_S2_iter2_LL2_KF_" + str(kf_ind) + ".npz")
        jnp.savez(fname, LL=LL, jax_path_vmap=jax_path_vmap)

        LL, jax_path_vmap = learnt_LL2(new_logpi02, new_log_Ps2, new_Rs2, new_reward2, temps, n_states, n_actions, new_trans_prob, test_xohs, test_xohs2, test_aohs)
        print(LL)
        fname = os.path.join(out_dir, run_uid + "_" + str(n_hidden) + '_' + str(seed) + "_naturenet_S2_iter2_LL2_test_KF_" + str(kf_ind) + ".npz")
        jnp.savez(fname, LL=LL, jax_path_vmap=jax_path_vmap)

        kf_ind = kf_ind + 1


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



