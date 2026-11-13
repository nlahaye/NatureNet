
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
 
from naturenet.models.irl.swirl_training import *
from naturenet.models.irl.swirl_utils import *

from naturenet.models.irl.jax_precision import configure as _configure_precision
_configure_precision()   # float64 unless NATURENET_JAX_X64=0 (jax_precision.py)
#jax.config.update("jax_platform_name", "gpu")


def _safe_log(p):
    return jnp.log(jnp.clip(p, 1e-12))


def _dense(tp):
    return tp.todense() if hasattr(tp, "todense") else tp


def _to_device(loc):
    """Convert T and the one-hot inputs to device arrays ONCE (not on every E/M step)."""
    tp = loc.get("trans_probs", loc.get("new_trans_probs"))
    xohs2 = loc.get("train_xohs2")
    return (device_tp(tp), jnp.asarray(loc["train_xohs"]),
            None if xohs2 is None else jnp.asarray(xohs2), jnp.asarray(loc["train_aohs"]))


def _converged(LL_list, tol):
    """Relative LL change below tol (None disables early stopping, matching the old behaviour)."""
    if tol is None or len(LL_list) < 2:
        return False
    prev, cur = float(LL_list[-2]), float(LL_list[-1])
    return abs(cur - prev) <= tol * max(1.0, abs(prev))


def em_train_temp(logpi0, log_Ps, Rs, rewards, temps, trans_probs, train_xohs, train_aohs, iter=100, init=True, trans=True, emit=True, tol=None, verbose=False):
    LL_list = []
    _args = _to_device(locals())
    trans_probs_j, xohs_j, xohs2_j, aohs_j = _args
    for i in range(iter):
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jax_e_step_batch_temp(pi0, log_Ps, Rs, rewards, trans_probs_j, temps, xohs_j, aohs_j)
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        if verbose:
            print("EM iter", i, "LL", float(LL_list[-1]))
        if _converged(LL_list, tol):
            break

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_scipy2(log_Ps, Rs, (all_gamma_jax, all_xi_jax), xohs_j)
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_rewards = emit_m_step_jax_scipy_temp_reg(rewards, trans_probs_j, temps, (all_gamma_jax, all_xi_jax), xohs_j, aohs_j)
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list

def em_train2_temp(logpi0, log_Ps, Rs, rewards, temps, new_trans_probs, train_xohs, train_xohs2, train_aohs, iter=100, init=True, trans=True, emit=True, tol=None, verbose=False):
    LL_list = []
    _args = _to_device(locals())
    trans_probs_j, xohs_j, xohs2_j, aohs_j = _args
    for i in range(iter):
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jax_e_step_batch2_temp(pi0, log_Ps, Rs, rewards, trans_probs_j, temps, xohs_j, xohs2_j, aohs_j)
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        if verbose:
            print("EM iter", i, "LL", float(LL_list[-1]))
        if _converged(LL_list, tol):
            break

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_scipy2(log_Ps, Rs, (all_gamma_jax, all_xi_jax), xohs_j)
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_rewards = emit_m_step_jax_scipy2_temp(rewards, trans_probs_j, temps, (all_gamma_jax, all_xi_jax), xohs2_j, aohs_j)
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list


def em_train_naturenet(logpi0, log_Ps, Rs, rewards, trans_probs, train_xohs, train_aohs, iter=100, init=True, trans=True, emit=True, tol=None, verbose=False):
    LL_list = []
    _args = _to_device(locals())
    trans_probs_j, xohs_j, xohs2_j, aohs_j = _args
    for i in range(iter):
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jax_e_step_batch_naturenet(pi0, log_Ps, Rs, rewards, trans_probs_j, xohs_j, aohs_j)
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        if verbose:
            print("EM iter", i, "LL", float(LL_list[-1]))
        if _converged(LL_list, tol):
            break

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_scipy2(log_Ps, Rs, (all_gamma_jax, all_xi_jax), xohs_j)
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_rewards = emit_m_step_jax_scipy2_naturenet(rewards, trans_probs_j, (all_gamma_jax, all_xi_jax), xohs_j, aohs_j)
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list

def em_train2_naturenet(logpi0, log_Ps, Rs, rewards, new_trans_probs, train_xohs, train_xohs2, train_aohs, iter=100, init=True, trans=True, emit=True, tol=None, verbose=False):
    LL_list = []
    _args = _to_device(locals())
    trans_probs_j, xohs_j, xohs2_j, aohs_j = _args
    for i in range(iter):
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jax_e_step_batch2_naturenet(pi0, log_Ps, Rs, rewards, trans_probs_j, xohs_j, xohs2_j, aohs_j)
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        if verbose:
            print("EM iter", i, "LL", float(LL_list[-1]))
        if _converged(LL_list, tol):
            break

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_scipy2(log_Ps, Rs, (all_gamma_jax, all_xi_jax), xohs_j)
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_rewards = emit_m_step_jax_scipy2_naturenet(rewards, trans_probs_j, (all_gamma_jax, all_xi_jax), xohs2_j, aohs_j)
           
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list



def comp_LLloss(pi0, trans_Ps, lls):
    alphas_list = vmap(partial(forward, jnp.array(pi0)))(trans_Ps, lls)
    return jnp.sum(jax_logsumexp(alphas_list[:, -1], axis=-1))

def _masked_ll_path(ll_total, path, all_aohs):
    """LL per real step and the Viterbi path with -1 on padded steps (segment_policy: variable)."""
    m = oh_mask(jnp.asarray(all_aohs))
    return ll_total / jnp.maximum(m.sum(), 1), jnp.where(m, path, -1)


def learnt_LL1(logpi0, log_Ps, Rs, rewards, temps, n_states, n_actions, trans_probs, all_xohs, all_aohs):
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
    trans_probs = jnp.asarray(_dense(trans_probs))
    pi, _, _ = vmap(partial(vi_temp, trans_probs))(rewards_sa, temps)
    logemit = _safe_log(pi)
    new_lls_jax_vmap = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs), jnp.array(all_aohs))
    new_trans_Ps_vmap = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(all_xohs))
    new_trans_Ps_vmap = vmap(pad_eye)(new_trans_Ps_vmap, jnp.asarray(all_aohs))
    jax_path_vmap = vmap(partial(viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap), jnp.array(new_lls_jax_vmap))
    return _masked_ll_path(comp_LLloss(pi0, new_trans_Ps_vmap, new_lls_jax_vmap), jax_path_vmap, all_aohs)

def learnt_LL2(logpi0, log_Ps, Rs, rewards, temps, n_states, n_actions, new_trans_probs, all_xohs, all_xohs2, all_aohs):
    n_states, n_actions = int(rewards.shape[-1]), tp_n_actions(new_trans_probs)   # S*M augmented states
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
    new_trans_probs = device_tp(new_trans_probs)
    pi, _, _ = vmap(partial(vi_temp, new_trans_probs))(rewards_sa, temps)
    logemit = _safe_log(pi)
    new_lls_jax_vmap = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs2), jnp.array(all_aohs))
    new_trans_Ps_vmap = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(all_xohs))
    new_trans_Ps_vmap = vmap(pad_eye)(new_trans_Ps_vmap, jnp.asarray(all_aohs))
    jax_path_vmap = vmap(partial(viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap), jnp.array(new_lls_jax_vmap))
    return _masked_ll_path(comp_LLloss(pi0, new_trans_Ps_vmap, new_lls_jax_vmap), jax_path_vmap, all_aohs)


