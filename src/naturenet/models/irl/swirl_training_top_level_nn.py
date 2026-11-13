#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
NN-SWIRL EM drivers.

em_train_nn       - S-1 ("s1") or S-2 ("s2") EM with integer inputs, the policy computed once per
                    EM iteration and a single vmapped E-step (the old S-2 loop re-solved the S^2 MDP
                    once per sequence).
learnt_LL_nn      - normalised log-likelihood + Viterbi paths for one model and one data split.

S-2 emission M-step: the old code ran 800 Adam steps on the S-1 (expanded-input) objective and then 200
on the S-2 objective in EVERY EM iteration, pulling the S-2 net back to the S-1 solution each time. Now
the S-1 objective is only used as a warm start before the first iteration (s1_warmup_iters, default 0 =
off) and every EM iteration optimises the S-2 objective.

The old entry points (em_train_jaxopt_netadam, em_train_jaxopt_netadam2, learnt_LL21, learnt_LL22) are
kept as wrappers that take the old one-hot arrays.
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jax_logsumexp

from naturenet.models.irl.swirl_training_nn import (  # noqa: F401  (re-exported for star importers)
    DEFAULT_DISCOUNT,
    DEFAULT_VI_ITERS,
    _idx,
    _viterbi_JAX,
    comp_ll_jax,
    comp_log_transP,
    comp_transP,
    e_step_all,
    emit_m_step,
    expected_states,
    forward,
    backward,
    loglik_and_paths,
    pi0_m_step,
    policy_logits_state,
    reward_tables,
    soft_vi,
    soft_vi_pairs,
    trans_m_step,
    vinet,
    vinet_expand,
)

from naturenet.models.irl.jax_precision import configure as _configure_precision
_configure_precision()   # float64 unless NATURENET_JAX_X64=0 (jax_precision.py)

from tqdm import tqdm


from naturenet.environment.pipeline_checks import check_policy


def _f64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def em_train_nn(logpi0, log_Ps, Rs, R_state, T, xoh, xe, a, mode="s1", iters=20, init=True, trans=True,
                emit=True, emit_iters=800, emit_tol=None, s1_warmup_iters=0, discount=DEFAULT_DISCOUNT,
                vi_iters=DEFAULT_VI_ITERS, tol=None, trans_iters=1000, verbose=True):
    """
    T (S, A, S) dense first-order transitions; xoh (N, T, 1, S) one-hot states; xe (N, T) emission index
    (x for "s1", x * S + x_prev for "s2"); a (N, T) actions.
    tol: stop when the relative change of the total LL between EM iterations is < tol (None = fixed iters).
    Returns logpi0, log_Ps, Rs, R_state, LL_list (LL_list[i] is the LL before the i-th update).
    """
    T, xoh = _f64(T), _f64(xoh)
    xe, a = jnp.asarray(xe, dtype=jnp.int32), jnp.asarray(a, dtype=jnp.int32)
    logpi0, log_Ps, Rs = _f64(logpi0), _f64(log_Ps), _f64(Rs)
    x_s1 = jnp.argmax(xoh[..., 0, :], axis=-1).astype(jnp.int32)
    LL_list = []
    it = tqdm(range(iters)) if verbose else range(iters)
    for i in it:
        logpi = policy_logits_state(R_state, T, discount, mode, vi_iters)
        if i == 0:
            check_policy(logpi, name=f"NN-SWIRL {mode} policy")
        gamma, xi, lls = e_step_all(logpi0, log_Ps, Rs, logpi, xoh, xe, a)
        LL = float(jnp.sum(lls))
        if not np.isfinite(LL):
            raise FloatingPointError(f"EM iteration {i}: log-likelihood is {LL}")
        LL_list.append(LL)
        if tol is not None and i > 0 and abs(LL_list[-1] - LL_list[-2]) <= tol * abs(LL_list[-2]):
            if verbose:
                print(f"EM converged after {i} iterations (rel. change < {tol})")
            break
        new_logpi0 = pi0_m_step(gamma) if init else logpi0
        if trans:
            new_log_Ps, new_Rs = trans_m_step(log_Ps, Rs, gamma, xi, xoh, num_iters=trans_iters)
        else:
            new_log_Ps, new_Rs = log_Ps, Rs
        if emit:
            if mode == "s2" and i == 0 and s1_warmup_iters:
                R_state, _ = emit_m_step(R_state, T, gamma, x_s1, a, "s1", s1_warmup_iters, discount, vi_iters,
                                         emit_tol)
            R_state, _ = emit_m_step(R_state, T, gamma, xe, a, mode, emit_iters, discount, vi_iters, emit_tol)
        logpi0, log_Ps, Rs = new_logpi0, new_log_Ps, new_Rs
    return logpi0, log_Ps, Rs, R_state, np.asarray(LL_list)


def learnt_LL_nn(logpi0, log_Ps, Rs, R_state, T, xoh, xe, a, mode="s1", discount=DEFAULT_DISCOUNT,
                 vi_iters=DEFAULT_VI_ITERS):
    """Total LL / (N * T) and Viterbi paths (N, T) for one split."""
    T, xoh = _f64(T), _f64(xoh)
    logpi = policy_logits_state(R_state, T, discount, mode, vi_iters)
    lls, paths = loglik_and_paths(_f64(logpi0), _f64(log_Ps), _f64(Rs), logpi, xoh,
                                  jnp.asarray(xe, dtype=jnp.int32), jnp.asarray(a, dtype=jnp.int32))
    LL = float(jnp.sum(lls)) / (xoh.shape[0] * xoh.shape[1])
    if not np.isfinite(LL):
        raise FloatingPointError(f"learnt LL is {LL}")
    return LL, np.asarray(paths)


def comp_LLloss(pi0, trans_Ps, lls):
    from jax import vmap
    from functools import partial
    alphas_list = vmap(partial(forward, jnp.array(pi0)))(trans_Ps, lls)
    return jnp.sum(jax_logsumexp(alphas_list[:, -1], axis=-1))


# ---------------------------------------------------------------------------
# Old entry points (one-hot inputs)
# ---------------------------------------------------------------------------

def _pair_index(xohs2):
    return _idx(xohs2)


def em_train_jaxopt_netadam(train_xohs, train_aohs, logpi0, log_Ps, Rs, R_state, trans_probs, iter=100,
                            init=True, trans=True, emit=True, **kw):
    return em_train_nn(logpi0, log_Ps, Rs, R_state, trans_probs, train_xohs, _idx(train_xohs), _idx(train_aohs),
                       "s1", iter, init, trans, emit, **kw)


def em_train_jaxopt_netadam2(train_xohs2, train_xohs, train_aohs, logpi0, log_Ps, Rs, R_state, trans_probs,
                             new_trans_probs=None, iter=100, init=True, trans=True, emit=True, **kw):
    """new_trans_probs (dense S^2 tensor) is no longer needed and is ignored."""
    return em_train_nn(logpi0, log_Ps, Rs, R_state, trans_probs, train_xohs, _pair_index(train_xohs2),
                       _idx(train_aohs), "s2", iter, init, trans, emit, **kw)



def learnt_LL21(logpi0, log_Ps, Rs, params, apply_fn, trans_probs, all_aohs, all_xohs):
    """Old API. Returns (LL, LL, paths) - the old 'train_LL' was the same number computed twice."""
    from flax.training import train_state
    import optax
    st = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=optax.identity())
    LL, paths = learnt_LL_nn(logpi0, log_Ps, Rs, st, trans_probs, all_xohs, _idx(all_xohs), _idx(all_aohs), "s1")
    return LL, LL, paths


def learnt_LL22(logpi0, log_Ps, Rs, params, apply_fn, trans_probs, all_aohs, all_xohs, all_xohs2):
    """Old API, now scored with the S-2 policy (was the S-1 policy on pair one-hots -> shape error)."""
    from flax.training import train_state
    import optax
    st = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=optax.identity())
    LL, paths = learnt_LL_nn(logpi0, log_Ps, Rs, st, trans_probs, all_xohs, _pair_index(all_xohs2),
                             _idx(all_aohs), "s2")
    return LL, LL, paths
