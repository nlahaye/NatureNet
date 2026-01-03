import os

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set(style='ticks', font_scale=1.5)

mpl.use("Agg")
#mpl.rcParams['pdf.fonttype'] = 42
#mpl.rcParams['ps.fonttype'] = 42
#mpl.rcParams["text.usetex"] = True
#mpl.rcParams["mathtext.fontset"] = 'cm'
#mpl.rcParams['font.family'] = ['sans-serif']

from hiql.algorithms import value_iteration, policy_eval

import sparse

from sit_fuse.utils import read_yaml

import argparse

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

def run_plots(yml_conf):

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

    fig_dir = os.path.join(out_dir, "figures")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6, 4.5))
    cmap = 'viridis'

    for p in range(len(positions)):
        print(p, "INITIAL_STATE", positions[p][0])
        print(p, "FINAL_STATE", positions[p][-1])
        print(p, "HIST", np.histogram(positions[p], bins=n_states))

    for i in range(n_states):
        cur_state = i
        r_state_goal = np.zeros(n_states)
        r_state_goal[cur_state] = 1
        v_state_goal_gt = value_iteration(reward=r_state_goal, P = trans_prob, num_actions=n_actions,\
            num_states=n_states, discount=gamma)
        new_arr_size = math.ceil(math.sqrt(v_state_goal_gt.shape[0]))
        new_arr = np.zeros((new_arr_size**2))
        new_arr[:v_state_goal_gt.shape[0]] = v_state_goal_gt
        v_state_goal_gt = new_arr

        print(v_state_goal_gt.shape) 
        im = plt.imshow(v_state_goal_gt.reshape(new_arr_size, new_arr_size), cmap=cmap)
        plt.colorbar(im, location='bottom', pad=0.05, ticks=[np.min(v_state_goal_gt), np.max(v_state_goal_gt)],
                 format='${x:.1f}$')
        plt.tight_layout(h_pad=0.1)
        plt.savefig(os.path.join(fig_dir, "vs_" + str(i) + "_v_state_goal.png"), bbox_inches='tight')
        plt.clf()

        # iavi
        print(out_dir, os.path.join(out_dir, "iavi", "fold_11", "q.npy"))
        pi = np.load(os.path.join(out_dir, "iavi", "fold_11", "q.npy"))
        pi = np.exp(pi) / np.sum(np.exp(pi), axis=-1, keepdims=True)
        v_state_goal = policy_eval(pi, r_state_goal, trans_prob, n_states, gamma)

        new_arr_n = np.zeros((new_arr_size**2))
        new_arr_n[:v_state_goal.shape[0]] = v_state_goal
        v_state_goal = new_arr_n

        im = plt.imshow(v_state_goal.reshape(new_arr_size, new_arr_size), cmap=cmap)
        plt.colorbar(im, location='bottom', pad=0.05, ticks=[np.min(v_state_goal), np.max(v_state_goal)],
                     format='${x:.1f}$')
        plt.tight_layout(h_pad=0.1)
        plt.savefig(os.path.join(fig_dir, "vs_" + str(i) + "_iavi.png"), bbox_inches='tight')
        plt.clf()

        # hiavi
        pis = []
        for l_idx in range(5):
            pi = np.load(os.path.join(out_dir, "hiavi", "fold_0", f'q_{l_idx}.npy'))
            pi = np.exp(pi) / np.sum(np.exp(pi), axis=-1, keepdims=True)
            pis.append(pi)
            v_state_goal = policy_eval(pis[l_idx], r_state_goal, trans_prob, n_states, gamma)

            new_arr_n_2 = np.zeros((new_arr_size**2))
            new_arr_n_2[:v_state_goal.shape[0]] = v_state_goal
            v_state_goal = new_arr_n_2

 
            im = plt.imshow(v_state_goal.reshape(new_arr_size, new_arr_size), cmap=cmap)
            plt.colorbar(im, location='bottom', pad=0.05, ticks=[np.min(v_state_goal), np.max(v_state_goal)],
                     format='${x:.1f}$')
            plt.tight_layout(h_pad=0.1)
            plt.savefig(os.path.join(fig_dir, "vs_" + str(i) + "_hiavi_latent" + str(l_idx) + ".png"), bbox_inches='tight')
            plt.clf() 
    
            for r_idx in range(axs.shape[0]):
                #for c_idx in range(axs.shape[1]):
                axs[r_idx].set_xticks([])
                axs[r_idx].set_yticks([])
 
        #fig.add_artist(plt.Line2D((.36, .36), (0.05, 0.99), color="k", linestyle='dashed', linewidth=2))
 
        #fig.text(0.082, 0.95, 'ground truth', fontsize=20)
        #fig.text(0.61, 0.95, 'HIAVI', fontsize=20)
        #axs[0].set_ylabel("`goal'", fontsize=20)
        #axs[0].set_ylabel("`abandon'", fontsize=20)
        #axs[1].set_title('(1 intention)', fontsize=15)
        #axs[2].set_title('(2 intentions)', fontsize=15)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_plots(yml_conf)

