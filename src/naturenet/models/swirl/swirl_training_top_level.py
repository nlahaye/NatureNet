
#Code adapted from https://github.com/BRAINML-GT/SWIRL

import numpy as np
import numpy.random as npr
from scipy.special import logsumexp

import sparse

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
 
from naturenet.models.swirl.swirl_training import *
from naturenet.models.swirl.swirl_utils import *

jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_platform_name", "gpu")

def em_train_temp(logpi0, log_Ps, Rs, rewards, temps, trans_probs, train_xohs, train_aohs, iter=100, init=True, trans=True, emit=True):
    LL_list = []
    for i in range(iter):
        print(i)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jax_e_step_batch_temp(pi0, log_Ps, Rs, rewards, trans_probs, temps, train_xohs, train_aohs)
        print(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0
        print(new_logpi0)

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_scipy2(log_Ps, Rs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs))
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_rewards = emit_m_step_jax_scipy_temp_reg(rewards, trans_probs, temps,(all_gamma_jax, all_xi_jax), jnp.array(train_xohs), jnp.array(train_aohs))
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list

def em_train2_temp(logpi0, log_Ps, Rs, rewards, temps, new_trans_probs, iter=100, init=True, trans=True, emit=True):
    LL_list = []
    for i in range(iter):
        print(i)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jax_e_step_batch2_temp(pi0, log_Ps, Rs, rewards, new_trans_probs, temps, train_xohs, train_xohs2, train_aohs)
        print(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0
        print(new_logpi0)

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_scipy2(log_Ps, Rs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs))
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_rewards = emit_m_step_jax_scipy2_temp(rewards, new_trans_probs, temps, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs2), jnp.array(train_aohs))
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list


def em_train_naturenet(logpi0, log_Ps, Rs, rewards, iter=100, init=True, trans=True, emit=True):
    LL_list = []
    for i in range(iter):
        print(i)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jax_e_step_batch_labyrinth(pi0, log_Ps, Rs, rewards, trans_probs, train_xohs, train_aohs)
        print(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0
        print(new_logpi0)

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_scipy2(log_Ps, Rs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs))
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_rewards = emit_m_step_jax_scipy2_labyrinth(rewards, trans_probs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs), jnp.array(train_aohs))
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list

def em_train2_naturenet(logpi0, log_Ps, Rs, rewards, iter=100, init=True, trans=True, emit=True):
    LL_list = []
    for i in range(iter):
        print(i)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jax_e_step_batch2_labyrinth(pi0, log_Ps, Rs, rewards, new_trans_probs, train_xohs, train_xohs2, train_aohs)
        print(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0
        print(new_logpi0)

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_scipy2(log_Ps, Rs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs))
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_rewards = emit_m_step_jax_scipy2_labyrinth(rewards, new_trans_probs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs2), jnp.array(train_aohs))
           
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list



def comp_LLloss(pi0, trans_Ps, lls):
    alphas_list = vmap(partial(forward, jnp.array(pi0)))(trans_Ps, lls)
    return jnp.sum(jax_logsumexp(alphas_list[:, -1], axis=-1))

def learnt_LL1(logpi0, log_Ps, Rs, rewards, temps, n_states, n_actions, trans_probs, all_xohs, all_aohs, train_xohs, train_aohs, test_aohs, test_xohs):
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
    pi, _, _ = vmap(partial(vi_temp, trans_probs))(rewards_sa, temps)
    logemit = jnp.log(pi)
    new_lls_jax_vmap = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs), jnp.array(all_aohs))
    new_trans_Ps_vmap = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(all_xohs))
    new_lls_jax_vmap_train = vmap(partial(comp_ll_jax, logemit))(jnp.array(train_xohs), jnp.array(train_aohs))
    new_trans_Ps_vmap_train = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(train_xohs))
    new_lls_jax_vmap_test = vmap(partial(comp_ll_jax, logemit))(jnp.array(test_xohs), jnp.array(test_aohs))
    new_trans_Ps_vmap_test = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(test_xohs))
    jax_path_vmap = vmap(partial(_viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap), jnp.array(new_lls_jax_vmap))
    return comp_LLloss(pi0, new_trans_Ps_vmap, new_lls_jax_vmap) / (all_xohs.shape[0]*all_xohs.shape[1]), \
        comp_LLloss(pi0, new_trans_Ps_vmap_train, new_lls_jax_vmap_train) / (train_xohs.shape[0]*train_xohs.shape[1]),\
        comp_LLloss(pi0, new_trans_Ps_vmap_test, new_lls_jax_vmap_test) / (test_xohs.shape[0]*test_xohs.shape[1]), jax_path_vmap

def learnt_LL2(logpi0, log_Ps, Rs, rewards, temps, n_states, n_actions, trans_probs, all_xohs, all_aohs, train_xohs, train_aohs, test_aohs, test_xohs):
    n_states, n_actions, _ = new_trans_probs.shape
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
    pi, _, _ = vmap(partial(vi_temp, new_trans_probs))(rewards_sa, temps)
    logemit = jnp.log(pi)
    new_lls_jax_vmap = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs2), jnp.array(all_aohs))
    new_trans_Ps_vmap = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(all_xohs))
    new_lls_jax_vmap_train = vmap(partial(comp_ll_jax, logemit))(jnp.array(train_xohs2), jnp.array(train_aohs))
    new_trans_Ps_vmap_train = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(train_xohs))
    new_lls_jax_vmap_test = vmap(partial(comp_ll_jax, logemit))(jnp.array(test_xohs2), jnp.array(test_aohs))
    new_trans_Ps_vmap_test = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(test_xohs))
    jax_path_vmap = vmap(partial(_viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap), jnp.array(new_lls_jax_vmap))
    return comp_LLloss(pi0, new_trans_Ps_vmap, new_lls_jax_vmap) / (all_xohs.shape[0]*all_xohs.shape[1]), comp_LLloss(pi0, new_trans_Ps_vmap_train, new_lls_jax_vmap_train) / (train_xohs.shape[0]*train_xohs.shape[1]), comp_LLloss(pi0, new_trans_Ps_vmap_test, new_lls_jax_vmap_test) / (test_xohs.shape[0]*test_xohs.shape[1]), jax_path_vmap


