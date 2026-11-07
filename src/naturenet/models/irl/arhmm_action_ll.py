"""
ARHMM predictions re-scored as ACTIONS, on the same steps and per-step scale as SWIRL / HIAVI.

The ARHMM (ssm "arcategorical") scores the next STATE, p_z(x' | x), over n_states cells. SWIRL and HIAVI score the
ACTION, p_z(a | x), over 9 moves, so the raw ARHMM LL is on a different event space (and is inflated when animals
repeat cell-to-cell moves). This module turns the fitted ARHMM into an action model:

    p_z(a | x) ∝ sum_x' p_z(x' | x) q(a | x, x'),   q(a | x, x') = T[x, a, x'] / sum_a' T[x, a', x']

i.e. each predicted next cell is credited to the actions that reach it under the fold's T (train animals only,
the same T SWIRL uses). Mass on cells that T cannot reach from x is dropped and the rest renormalised (uniform if
nothing is reachable). The latent dynamics are the ARHMM's own (logpi0, log_Ps, recurrent MLP W1/b1/W2/b2: the
parameters SWIRL starts from), run through the SAME forward / Viterbi / padding code as learnt_LL1, so
  train_ll_act / test_ll_act  are directly comparable with the SWIRL S-1 / S-2 and HIAVI LLs.
Kinematic ARHMMs are scored on the action part only (their kinematic emissions have no SWIRL counterpart).

Called from run_arhmm_kfold.run_arhmm (yml arhmm_action_ll: True, the default) and plotted by
plot_ll_comparison.py as method "arhmm_act".
"""
from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp as jax_logsumexp

from naturenet.models.irl.swirl_training import comp_ll_jax, comp_transP, pad_eye, viterbi_JAX
from naturenet.models.irl.swirl_training_top_level import _masked_ll_path, comp_LLloss


def arhmm_ar_logits(arhmm_s):
    """(K, S, S) log p_z(x' | x) of a fitted ssm ARHMM (arcategorical or arcategorical_kinematic)."""
    lg = np.asarray(arhmm_s.dynamics.logits, dtype=np.float64)        # (K, D=1, C, C)
    lg = lg[:, 0] if lg.ndim == 4 else lg
    return lg - np.logaddexp.reduce(lg, axis=-1, keepdims=True)


def action_logemit(ar_logits, T, floor=1e-12):
    """(K, S, A) log p_z(a | x) implied by the ARHMM next-state tables and the transition tensor T (S, A, S)."""
    T = np.asarray(T.todense() if hasattr(T, "todense") else T, dtype=np.float64)
    P = np.exp(np.asarray(ar_logits, dtype=np.float64))                 # (K, S, S)
    den = T.sum(axis=1, keepdims=True)                                  # (S, 1, S): sum over actions
    q = np.divide(T, den, out=np.zeros_like(T), where=den > 0)          # q(a | x, x')
    pa = np.einsum("kxy,xay->kxa", P, q)                                # (K, S, A)
    tot = pa.sum(-1, keepdims=True)
    A = T.shape[1]
    pa = np.where(tot > 0, pa / np.where(tot > 0, tot, 1.0), 1.0 / A)
    lp = np.log(pa + floor)
    lp = lp - np.logaddexp.reduce(lp, axis=-1, keepdims=True)
    assert np.all(np.isfinite(lp)) and lp.shape == (P.shape[0], T.shape[0], A)
    return lp


def ll_from_logemit(logpi0, log_Ps, Rs, logemit, xohs, aohs):
    """Per-step LL and Viterbi path for fixed log-emissions (K, S, A): learnt_LL1 without the value iteration."""
    logpi0 = jnp.asarray(logpi0)
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    logemit = jnp.asarray(logemit)
    xohs, aohs = jnp.asarray(xohs), jnp.asarray(aohs)
    lls = vmap(partial(comp_ll_jax, logemit))(xohs, aohs)
    Ps = vmap(partial(comp_transP, jnp.asarray(log_Ps), tuple(jnp.asarray(r) for r in Rs)))(xohs)
    Ps = vmap(pad_eye)(Ps, aohs)
    path = vmap(partial(viterbi_JAX, pi0))(Ps, lls)
    return _masked_ll_path(comp_LLloss(pi0, Ps, lls), path, aohs)


def arhmm_action_lls(arhmm_params, ar_logits, T, data, train_index, test_index=None):
    """(train_ll_act, test_ll_act, train_path, test_path) for one fold; data = run_swirl_kfold.prepare_data(...)."""
    logpi0, log_Ps, Rs = arhmm_params
    le = action_logemit(ar_logits, T)
    xo, ao = data["xohs"], data["aohs"]
    tr_ll, tr_path = ll_from_logemit(logpi0, log_Ps, Rs, le, xo[train_index], ao[train_index])
    te_ll, te_path = (np.nan, None)
    if test_index is not None and len(test_index) > 0:
        te_ll, te_path = ll_from_logemit(logpi0, log_Ps, Rs, le, xo[test_index], ao[test_index])
    return float(tr_ll), float(te_ll), tr_path, te_path
