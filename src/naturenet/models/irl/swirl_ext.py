"""
SWIRL EM with two optional extensions, on the same JAX pieces as swirl_training / swirl_training_top_level:

1. Per-intention kinematic emissions (kinematics.KinematicEmissions): the step log-likelihood becomes
       log pi_k(a_t | x_t) + log f_k(step_t) + log vM_k(turn_t)
   The kinematic term is a fixed (N, T, K) array inside each E-step (recomputed every iteration) and its
   parameters get a closed-form M-step from the posteriors; the reward M-step is unchanged (it sees the
   kinematics only through the posteriors).

2. Partner-coupled (multi-agent) intention transitions, as in hiavi_multiagent:
       P_t(j -> k) = (1 - alpha w_t) P^SWIRL_t(j -> k | x_t) + alpha w_t c_t[k]
   P^SWIRL_t is SWIRL's recurrent transition (log_Ps + MLP(x_t)); (w, c) is the partner context
   (hiavi_multiagent.PartnerContext.context). log_Ps, the MLP and alpha = sigmoid(alpha_raw) are fitted
   jointly by L-BFGS on the expected complete-data log-likelihood of the mixture.
   Log-linear forms (coupling: align | matrix, see hiavi_multiagent): partners shift the switch logits,
       P_t(j -> k) = softmax_k( log P^SWIRL_t(j -> k | x_t) + w_t sum_m B[k, m] c_t[m] )
   with B = beta I (align) or a free K x K matrix (matrix), fitted with the transition parameters under the ridge
   coupling_l2 / 2 * ||B||^2. B = 0 is exactly the independent SWIRL.

kin=None and ctx=None (or alpha = 0) give exactly the original em_train_naturenet / em_train_temp /
em_train2_* updates (the same E-step algebra and the same trans / emit M-step functions).

    r = em_train_ext(logpi0, log_Ps, Rs, rewards, T, xohs, aohs, xohs_emit=None, temps=None,
                     kin=None, kin_model=None, ctx=None, alpha=0.0, fit_alpha=False, iters=50)
    ev = swirl_eval(r, T, xohs, aohs, ...)   # LL, step LLs (N, T), posteriors (N, T, K), Viterbi paths
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.scipy.special import logsumexp as jax_logsumexp

from naturenet.models.irl.swirl_training import (
    MSTEP,
    device_tp,
    _emit_m_step,
    _log_policy,
    _minimize,
    _unflatten_trans,
    backward,
    comp_ll_jax,
    comp_log_transP,
    comp_transP,
    expected_states,
    forward,
    n_real,
    oh_mask,
    pad_eye,
    pi0_m_step,
    trans_m_step_jax_scipy2,
    viterbi_JAX,
)
from naturenet.models.irl.swirl_training_top_level import _converged

ALPHA_MAX = 1.0 - 1e-6


def _dense(tp):
    return tp.todense() if hasattr(tp, "todense") else tp


def _mix(P, w, c, alpha):
    """(T-1, K, K) SWIRL transitions mixed with the partner context of the destination step."""
    aw = (alpha * w[1:])[:, None, None]
    return (1 - aw) * P + aw * c[1:, None, :]


COUPLING_FORMS = ("mixture", "align", "matrix")


def theta_B(form, theta, K):
    """coupling parameter vector -> B (K, K) (log-linear forms)."""
    return theta[0] * jnp.eye(K) if form == "align" else theta.reshape(K, K)


def _couple(P, w, c, theta, form):
    """(T-1, K, K) SWIRL transitions coupled to the partner context (theta = [alpha] | [beta] | B.ravel())."""
    if form == "mixture":
        return _mix(P, w, c, theta[0])
    K = P.shape[-1]
    shift = w[1:, None] * (c[1:] @ theta_B(form, theta, K).T)                 # (T-1, K)
    lg = jnp.log(jnp.clip(P, 1e-300)) + shift[:, None, :]
    return jnp.exp(lg - jax_logsumexp(lg, axis=-1, keepdims=True))


def coupling_theta(form, K, alpha=0.0, B=None):
    """initial parameter vector of a coupling form."""
    if form not in COUPLING_FORMS:
        raise ValueError(f"coupling must be one of {COUPLING_FORMS}, got {form!r}")
    if form == "mixture":
        return jnp.array([float(alpha)])
    B = np.zeros((K, K)) if B is None else np.asarray(B, float)
    return jnp.array([float(np.trace(B)) / K]) if form == "align" else jnp.asarray(B.ravel())


def r_theta(r, K):
    """(form, theta) of a fitted result dict (old results: mixture with r['alpha'])."""
    form = r.get("coupling", "mixture") or "mixture"
    return form, coupling_theta(form, K, r.get("alpha", 0.0), r.get("B"))


def _one_seq(pi0, log_Ps, Rs, logemit, theta, form, xoh, xoh_e, aoh, kll, w, c):
    P = pad_eye(_couple(comp_transP(log_Ps, Rs, xoh), w, c, theta, form), aoh)
    ll = comp_ll_jax(logemit, xoh_e, aoh) + jnp.where(oh_mask(aoh)[:, None], kll, 0.0)
    al = forward(pi0, P, ll)
    be = backward(P, ll)
    g, xi = expected_states(al, be, P, ll)
    return g, xi, al


@partial(jax.jit, static_argnames=("use_temp", "form"))
def e_step_ext(pi0, log_Ps, Rs, rewards, trans_probs, temps, xohs, xohs_e, aohs, kll, w, c, theta, use_temp,
               form="mixture"):
    """theta: coupling parameters (mixture: [alpha]; align: [beta]; matrix: B.ravel())."""
    theta = jnp.atleast_1d(theta)
    logemit = _log_policy(rewards, trans_probs, temps, use_temp=use_temp)
    return vmap(partial(_one_seq, pi0, log_Ps, Rs, logemit, theta, form))(xohs, xohs_e, aohs, kll, w, c)


def _viterbi_one(pi0, log_Ps, Rs, logemit, theta, form, xoh, xoh_e, aoh, kll, w, c):
    P = pad_eye(_couple(comp_transP(log_Ps, Rs, xoh), w, c, theta, form), aoh)
    return viterbi_JAX(pi0, P, comp_ll_jax(logemit, xoh_e, aoh) + jnp.where(oh_mask(aoh)[:, None], kll, 0.0))


@partial(jax.jit, static_argnames=("use_temp", "form"))
def viterbi_ext(pi0, log_Ps, Rs, rewards, trans_probs, temps, xohs, xohs_e, aohs, kll, w, c, theta, use_temp,
                form="mixture"):
    theta = jnp.atleast_1d(theta)
    logemit = _log_policy(rewards, trans_probs, temps, use_temp=use_temp)
    return vmap(partial(_viterbi_one, pi0, log_Ps, Rs, logemit, theta, form))(xohs, xohs_e, aohs, kll, w, c)


# ---------------------------------------------------------------------------
# Coupled transition M-step
# ---------------------------------------------------------------------------

def _ctrans_loss(flat, xis, xohs, w, c, alpha_fixed, n_total, shapes, fit_alpha, form="mixture", n_theta=1,
                 l2=1.0):
    log_Ps, Rs = _unflatten_trans(flat[:-n_theta], shapes)
    raw = flat[-n_theta:]
    P0 = jnp.exp(vmap(lambda x: comp_log_transP(log_Ps, Rs, x))(xohs))       # (N, T-1, K, K)
    pen = 0.0
    if form == "mixture":
        alpha = ALPHA_MAX * jax.nn.sigmoid(raw[0]) if fit_alpha else alpha_fixed
        aw = alpha * w[:, 1:, None, None]
        P = (1 - aw) * P0 + aw * c[:, 1:, None, :]
    else:
        P = vmap(lambda p, ww, cc: _couple(p, ww, cc, raw, form))(P0, w, c)
        pen = 0.5 * l2 * jnp.sum(raw ** 2)
    m = oh_mask(xohs)[:, 1:, None, None]                                       # real destination steps
    return (-jnp.sum(jnp.where(m, xis * jnp.log(jnp.clip(P, 1e-300)), 0.0)) + pen) / n_total


_ctrans_vg = jax.jit(jax.value_and_grad(_ctrans_loss), static_argnames=("shapes", "fit_alpha", "form", "n_theta"))


def _logit(p):
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return np.log(p / (1 - p))


def coupled_trans_m_step(log_Ps, Rs, alpha, xis, xohs, w, c, fit_alpha=True, num_iters=1000, form="mixture",
                         theta=None, l2=1.0):
    """
    Transition M-step with partner coupling. mixture: returns (log_Ps, Rs, alpha). Log-linear forms (align /
    matrix): theta = current parameters (coupling_theta), returns (log_Ps, Rs, theta) with B column-centred.
    """
    W1, b1, W2, b2 = Rs
    parts = [jnp.asarray(p) for p in (log_Ps, W1, b1, W2, b2)]
    shapes = tuple(tuple(int(d) for d in p.shape) for p in parts)
    if form == "mixture":
        th0 = jnp.array([_logit(alpha / ALPHA_MAX)])
    else:
        th0 = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float64))
    flat0 = jnp.concatenate([p.ravel() for p in parts] + [th0])
    n_total = n_real(xohs)
    res = _minimize(_ctrans_vg, flat0, "L-BFGS-B", num_iters, xis=jnp.asarray(xis), xohs=xohs, w=w, c=c,
                    alpha_fixed=jnp.asarray(float(alpha)), n_total=n_total, shapes=shapes, fit_alpha=bool(fit_alpha),
                    form=form, n_theta=int(th0.size), l2=float(l2))
    nt = int(th0.size)
    lp, R = _unflatten_trans(jnp.asarray(res.x[:-nt]), shapes)
    if form == "mixture":
        a = float(ALPHA_MAX / (1 + np.exp(-res.x[-1]))) if fit_alpha else float(alpha)
        return lp, R, a
    th = np.asarray(res.x[-nt:], float)
    if form == "matrix":
        K = int(round(np.sqrt(nt)))
        Bm = th.reshape(K, K)
        th = (Bm - Bm.mean(axis=0, keepdims=True)).ravel()                   # same P, column-centred
    return lp, R, jnp.asarray(th)


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------

def _inputs(trans_probs, xohs, aohs, xohs_emit, temps, K, kin, kin_model, ctx):
    tp = device_tp(trans_probs)
    xo, ao = jnp.asarray(xohs), jnp.asarray(aohs)
    xe = xo if xohs_emit is None else jnp.asarray(xohs_emit)
    N, T = xo.shape[0], xo.shape[1]
    use_temp = temps is not None
    tmp = jnp.asarray(temps, dtype=jnp.float64) if use_temp else jnp.ones(K)
    if ctx is None:
        w, c = jnp.zeros((N, T)), jnp.zeros((N, T, K))
    else:
        w, c = jnp.asarray(ctx[0], dtype=jnp.float64), jnp.asarray(ctx[1], dtype=jnp.float64)
        if w.shape != (N, T) or c.shape != (N, T, K):
            raise ValueError(f"context w {w.shape} / c {c.shape} do not match (N, T, K) = {(N, T, K)}")
    if kin is not None and kin.shape != (N, T):
        raise ValueError(f"kinematics {kin.shape} do not match (N, T) = {(N, T)}")
    return tp, xo, ao, xe, tmp, use_temp, w, c


def _kll(kin, kin_model, N, T, K):
    if kin is None or kin_model is None:
        return jnp.zeros((N, T, K))
    return jnp.asarray(kin_model.loglik(kin))


def em_train_ext(logpi0, log_Ps, Rs, rewards, trans_probs, xohs, aohs, xohs_emit=None, temps=None,
                 kin=None, kin_model=None, ctx=None, alpha=0.0, fit_alpha=False, iters=100, init=True,
                 trans=True, emit=True, kin_mstep=True, tol=None, verbose=False, num_iters=None,
                 coupling="mixture", B=None, coupling_l2=1.0, fit_coupling=None):
    """
    One SWIRL EM run (S-1: xohs_emit=None, trans_probs (S, A, S); S-2: xohs_emit = S-2 one-hots, trans_probs
    the S-2 tensor). temps=None -> naturenet (no temperature) policy, else the temperature policy.
    coupling: mixture (alpha / fit_alpha, as before) | align | matrix (B start, fitted when ctx is given unless
    fit_coupling=False; ridge coupling_l2).
    Returns dict(logpi0, log_Ps, Rs, rewards, alpha, coupling, B, kin_model, LL_list).
    """
    rewards = jnp.asarray(rewards)
    K = int(rewards.shape[0])
    kin_model = None if kin_model is None else kin_model.copy()
    tp, xo, ao, xe, tmp, use_temp, w, c = _inputs(trans_probs, xohs, aohs, xohs_emit, temps, K, kin, kin_model, ctx)
    N, T = xo.shape[0], xo.shape[1]
    theta = coupling_theta(coupling, K, alpha, B)
    if coupling == "mixture":
        coupled = ctx is not None and (fit_alpha or alpha > 0)
    else:
        fit_coupling = True if fit_coupling is None else bool(fit_coupling)
        coupled = ctx is not None and fit_coupling
    LL_list = []
    for i in range(int(iters)):
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        g, xi, al = e_step_ext(pi0, log_Ps, Rs, rewards, tp, tmp, xo, xe, ao, _kll(kin, kin_model, N, T, K),
                               w, c, jnp.array([float(alpha)]) if coupling == "mixture" else theta, use_temp,
                               form=coupling)
        g = g * oh_mask(ao)[..., None]                    # padded steps carry no posterior mass
        LL_list.append(float(jnp.sum(jax_logsumexp(al[:, -1], axis=-1))))
        if not np.isfinite(LL_list[-1]):
            raise FloatingPointError(f"SWIRL EM LL is {LL_list[-1]} at iteration {i}")
        if verbose:
            print("EM iter", i, "LL", LL_list[-1], "alpha", round(float(alpha), 4))
        if _converged(LL_list, tol):
            break
        new_logpi0 = pi0_m_step(g) if init else logpi0
        if trans:
            if coupled and coupling == "mixture":
                new_log_Ps, new_Rs, alpha = coupled_trans_m_step(
                    log_Ps, Rs, alpha, xi, xo, w, c, fit_alpha,
                    MSTEP["trans_iters"] if num_iters is None else num_iters)
            elif coupled:
                new_log_Ps, new_Rs, theta = coupled_trans_m_step(
                    log_Ps, Rs, 0.0, xi, xo, w, c, False,
                    MSTEP["trans_iters"] if num_iters is None else num_iters, form=coupling, theta=theta,
                    l2=coupling_l2)
            else:
                new_log_Ps, new_Rs = trans_m_step_jax_scipy2(log_Ps, Rs, (g, xi), xo, num_iters=num_iters)
        else:
            new_log_Ps, new_Rs = log_Ps, Rs
        new_rewards = _emit_m_step(rewards, tp, tmp if use_temp else None, (g, xi), xe, ao, num_iters, use_temp) \
            if emit else rewards
        if kin_model is not None and kin is not None and kin_mstep:
            kin_model.m_step(np.asarray(g), kin)
        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, jnp.asarray(new_rewards)
    Bout = np.zeros((K, K)) if coupling == "mixture" else np.asarray(theta_B(coupling, jnp.asarray(theta), K))
    return dict(logpi0=logpi0, log_Ps=log_Ps, Rs=Rs, rewards=rewards,
                alpha=float(alpha) if coupling == "mixture" else 0.0, coupling=coupling, B=Bout,
                kin_model=kin_model, temps=None if temps is None else np.asarray(temps), LL_list=np.asarray(LL_list))


def swirl_eval(r, trans_probs, xohs, aohs, xohs_emit=None, kin=None, ctx=None, use_kin=True, paths=True):
    """
    Forward pass of a fitted model r (em_train_ext output) on (xohs, aohs): total LL, LL per step, one-step
    predictive LLs (N, T), posteriors (N, T, K), Viterbi paths (N, T). use_kin=False scores actions only.
    """
    rewards = jnp.asarray(r["rewards"])
    K = int(rewards.shape[0])
    temps = r.get("temps")
    km = r.get("kin_model") if use_kin else None
    tp, xo, ao, xe, tmp, use_temp, w, c = _inputs(trans_probs, xohs, aohs, xohs_emit, temps, K, kin, km, ctx)
    N, T = xo.shape[0], xo.shape[1]
    form, theta = r_theta(r, K) if ctx is not None else ("mixture", jnp.array([0.0]))
    pi0 = jnp.exp(r["logpi0"] - jax_logsumexp(r["logpi0"]))
    kll = _kll(kin, km, N, T, K)
    g, _, al = e_step_ext(pi0, r["log_Ps"], r["Rs"], rewards, tp, tmp, xo, xe, ao, kll, w, c, theta, use_temp,
                          form=form)
    cum = np.asarray(jax_logsumexp(al, axis=-1))                 # (N, T) log p(a_{0:t})
    step = np.diff(cum, axis=1, prepend=0.0)                     # 0 on padded steps
    m = np.asarray(oh_mask(ao))                                  # (N, T) real steps (all True without padding)
    n = int(m.sum())
    out = dict(LL_total=float(cum[:, -1].sum()), LL=float(cum[:, -1].sum() / max(n, 1)),
               step_ll=np.where(m, step, 0.0), gamma=np.asarray(g) * m[..., None], n_steps=n, mask=m)
    if paths:
        out["path"] = np.where(m, np.asarray(viterbi_ext(pi0, r["log_Ps"], r["Rs"], rewards, tp, tmp, xo, xe, ao,
                                                         kll, w, c, theta, use_temp, form=form)), -1)
    return out


def obj_array(items):
    out = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        out[i] = np.asarray(x)
    return out


def params_to_save(r, prefix="new_"):
    """npz-ready dict in the run_swirl_kfold naming (new_logpi0, new_log_Ps, new_Rs, new_reward, ...)."""
    out = {prefix + "logpi0": np.asarray(r["logpi0"]), prefix + "log_Ps": np.asarray(r["log_Ps"]),
           prefix + "Rs": obj_array(r["Rs"]),
           prefix + "reward": np.asarray(r["rewards"]), "LL_list": np.asarray(r["LL_list"]),
           "alpha": float(r.get("alpha", 0.0))}
    if r.get("coupling", "mixture") != "mixture":
        out.update(coupling=r["coupling"], B=np.asarray(r["B"]))
    if r.get("temps") is not None:
        out["temps"] = np.asarray(r["temps"])
    if r.get("kin_model") is not None:
        out.update(r["kin_model"].params())
    return out
