
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
 
from naturenet.models.irl.swirl_training_nn import *
from naturenet.models.irl.swirl_training_nn import _viterbi_JAX
from naturenet.models.irl.swirl_utils import *

jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_platform_name", "gpu")

from tqdm import tqdm

def em_train_jaxopt_netadam2(train_xohs2, train_xohs, train_aohs, logpi0, log_Ps, Rs, R_state, trans_probs, new_trans_probs, iter=100, init=True, trans=True, emit=True):
    LL_list = []


    for i in tqdm(range(iter)):
        #print("Epoch", i)
        #print(train_xohs2.shape, train_xohs.shape, train_aohs.shape, logpi0.shape, log_Ps.shape, Rs.shape, new_trans_probs.shape)

        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax = None
        all_xi_jax = None
        all_jax_alphas = None
        for j in range(0, train_xohs.shape[0], 1): 
            agj, axj, aja = jaxnet_e_step_batch2(pi0, log_Ps, Rs, R_state, new_trans_probs, train_xohs[j:j+1], train_xohs2[j:j+1], train_aohs[j:j+1])
            #print(all_gamma_jax.shape, all_xi_jax.shape, all_jax_alphas.shape)
            if all_gamma_jax is None:
                all_gamma_jax = agj
                all_xi_jax = axj
                all_jax_alphas = aja
                #print(all_gamma_jax, agj, all_xi_jax, axj, all_jax_alphas, aja)
            else:
                all_gamma_jax = jnp.concatenate((all_gamma_jax, agj), axis=0)
                all_xi_jax = jnp.concatenate((all_xi_jax, axj), axis=0)
                all_jax_alphas = jnp.concatenate((all_jax_alphas, aja), axis=0)
            #print(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0
        #print(new_logpi0)

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_jaxopt(log_Ps, Rs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs))
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        #print(all_gamma_jax)
        if emit == True:
            new_R_state = emit_m_step_jaxnet_optax2_expand(R_state, jnp.array(trans_probs), all_gamma_jax, jnp.array(train_xohs), jnp.array(train_aohs), num_iters=800)
            new_R_state = emit_m_step_jaxnet_optax2(new_R_state, jnp.array(new_trans_probs), all_gamma_jax, jnp.array(train_xohs2), jnp.array(train_aohs), num_iters=200)
        else:
            new_R_state = R_state
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        logpi0, log_Ps, Rs, R_state = new_logpi0, new_log_Ps, new_Rs, new_R_state
    return logpi0, log_Ps, Rs, R_state, LL_list

def em_train_jaxopt_netadam(train_xohs, train_aohs, logpi0, log_Ps, Rs, R_state, trans_probs, iter=100, init=True, trans=True, emit=True):
    LL_list = []
    for i in tqdm(range(iter)):
        #print(i)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jaxnet_e_step_batch(pi0, log_Ps, Rs, R_state, trans_probs, train_xohs, train_aohs)
        #print(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0
        #print(new_logpi0)

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_jaxopt(log_Ps, Rs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs))
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_R_state = emit_m_step_jaxnet_optax2_expand(R_state, jnp.array(trans_probs), all_gamma_jax, jnp.array(train_xohs), jnp.array(train_aohs), num_iters=800)
        else:
            new_R_state = R_state
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        logpi0, log_Ps, Rs, R_state = new_logpi0, new_log_Ps, new_Rs, new_R_state
    return logpi0, log_Ps, Rs, R_state, LL_list


def comp_LLloss(pi0, trans_Ps, lls):
    alphas_list = vmap(partial(forward, jnp.array(pi0)))(trans_Ps, lls)
    return jnp.sum(jax_logsumexp(alphas_list[:, -1], axis=-1))

def learnt_LL21(logpi0, log_Ps, Rs, params, apply_fn, trans_probs, all_aohs, all_xohs):
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    pi, _, _ = vinet_expand(trans_probs, params, apply_fn)
    logemit = jnp.log(pi)
    new_lls_jax_vmap = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs), jnp.array(all_aohs))
    new_trans_Ps_vmap = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(jnp.array(all_xohs))
    new_lls_jax_vmap_test = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs), jnp.array(all_aohs))
    new_trans_Ps_vmap_test = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(jnp.array(all_xohs))
    jax_path_vmap = vmap(partial(_viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap), jnp.array(new_lls_jax_vmap))
    return comp_LLloss(pi0, new_trans_Ps_vmap, new_lls_jax_vmap) / (all_xohs.shape[0]*all_xohs.shape[1]), comp_LLloss(pi0, new_trans_Ps_vmap_test, new_lls_jax_vmap_test) / (all_xohs.shape[0]*all_xohs.shape[1]), jax_path_vmap


def learnt_LL22(logpi0, log_Ps, Rs, params, apply_fn, trans_probs, all_aohs, all_xohs, all_xohs2):
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    pi, _, _ = vinet_expand(trans_probs, params, apply_fn)
    logemit = jnp.log(pi)
    new_lls_jax_vmap = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs2), jnp.array(all_aohs))
    new_trans_Ps_vmap = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(jnp.array(all_xohs))
    new_lls_jax_vmap_test = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs2), jnp.array(all_aohs))
    new_trans_Ps_vmap_test = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(jnp.array(all_xohs))
    jax_path_vmap = vmap(partial(_viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap), jnp.array(new_lls_jax_vmap))
    return comp_LLloss(pi0, new_trans_Ps_vmap, new_lls_jax_vmap) / (all_xohs.shape[0]*all_xohs.shape[1]), comp_LLloss(pi0, new_trans_Ps_vmap_test, new_lls_jax_vmap_test) / (all_xohs.shape[0]*all_xohs.shape[1]), jax_path_vmap





