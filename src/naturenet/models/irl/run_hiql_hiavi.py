
from hiql.algorithms import HIAVI

import os
import json

import pandas as pd
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

def gen_traj(actions, positions):

    traj_total = []
    for j in range(len(actions)):
        traj = []
        for i in range(len(actions[j])-1):
            print(i, len(positions[j]), len(actions[j]))
            traj.append([positions[j][i], actions[j][i], positions[j][i+1]])
        traj_total.append(traj)
    traj_total = np.array(traj_total)
    return traj_total

def run_hiavi(yml_conf):

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
    for i in range(len(actions_fpath)):
        actions_tmp = np.load(actions_fpath[i], allow_pickle=True)
        positions_tmp = np.load(positions_fpath[i], allow_pickle=True)
        
        actions.extend(actions_tmp)
        positions.extend(positions_tmp)

    print(len(positions), len(actions))

 
    #positions = np.array(positions[:-1])
    #actions  = np.array(actions[:-1])


    global n_states
    global n_actions
    n_states, n_actions, _ = trans_prob.shape

    trajs = gen_traj(actions, positions)
    trans_prob = np.moveaxis(trans_prob, 1,2)
    gamma = 0.9

    np.random.seed(42)
    output_df = pd.DataFrame(columns=['num_trajs', 'fold', 'train_ll', 'test_ll'])

    kf = KFold(n_splits=len(trajs), shuffle=True, random_state=10015)
    for num_trajs in np.arange(0,5):
        for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs)):
            train_trajs = [trajs[train_idx] for train_idx in train_idxes]
            test_trajs = [trajs[test_idx] for test_idx in test_idxes]

            best_test_ll = -np.infty
            best_ll = None
            for repeats in range(10): #num_repeats):
                model = HIAVI(num_latents=5, num_states=n_states, num_actions=n_actions,
                                train_trajs=train_trajs, test_trajs=test_trajs, P=trans_prob, discount=gamma)
                ll, logp_init, logp_tr, agents = model.fit()
                if ll['test'] > best_test_ll:
                    best_test_ll = ll['test']
                    best_ll = ll
                    if num_trajs == 4: 
                        param_dir = os.path.join(out_dir, f'hiavi/fold_{kf_idx}')
                        if not os.path.exists(param_dir):
                            os.makedirs(param_dir)
                        np.save(os.path.join(param_dir, 'logp_init.npy'), logp_init)
                        np.save(os.path.join(param_dir, 'logp_tr.npy'), logp_tr)
                        for agent_idx, agent in enumerate(agents):
                            np.save(os.path.join(param_dir, f'r_{agent_idx}.npy'), agent.r)
                            np.save(os.path.join(param_dir, f'q_{agent_idx}.npy'), agent.q)
            output_df.loc[len(output_df)] = [num_trajs, kf_idx, best_ll['train'], best_ll['test']]
            output_df.to_csv(os.path.join(out_dir, 'll_hiavi.csv'), index=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)
 
    run_hiavi(yml_conf)



