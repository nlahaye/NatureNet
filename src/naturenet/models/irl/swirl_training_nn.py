#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
NN-SWIRL core: soft value iteration, E-step and M-steps.

Changes vs. the original (see NATURENET_VIZ_NN_CHANGES.md):
- Stable soft policy: log pi = Q - logsumexp(Q) (was exp(Q)/sum(exp(Q)) -> inf/inf = NaN for large Q,
  and log(pi) -> -inf on underflow).
- S-2 value iteration is factored. The S-2 tensor is built from first-order dynamics, so the next pair
  after (s, p) under a is (s', s) with prob T[s, a, s']. Memory stays in R2(s, p, a), V2(s, p) and
  pi(a | s, p); only the expected future value depends on (s, a) alone:
      Q2[s, p, a] = R2[s, p, a] + discount * W[s, a],   W = einsum('sat,ts->sa', T, V2)
  O(S^2 A) per iteration instead of O(S^4 A), and no dense S^2 x A x S^2 tensor. A second-order
  T2[s, p, a, s'] (4-D) is also accepted (W then depends on p; O(S^3 A)).
- Emission log-likelihoods are integer gathers logpi[k, x_t, a_t] instead of one-hot einsums.
- E-step: policy computed once per EM iteration, all sequences in one vmapped, jitted call.
- M-steps are jitted once at module level (the old inner @jax.jit step was re-traced every EM
  iteration with the expectations baked in as constants); optional early stopping.
- No bfloat16 anywhere (float64 with jax_enable_x64).
The original function names are kept as thin wrappers at the bottom of the file.
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, vmap
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp

from naturenet.models.irl.swirl_nn_model import reward_table_expand, reward_table_pairs

jax.config.update("jax_enable_x64", True)

DEFAULT_DISCOUNT = 0.95
DEFAULT_VI_ITERS = 100


# ---------------------------------------------------------------------------
# Soft value iteration
# ---------------------------------------------------------------------------

def soft_vi(T, R, discount=DEFAULT_DISCOUNT, n_iter=DEFAULT_VI_ITERS):
    """T (S, A, S'), R (S, A) -> log pi (S, A), V (S)."""
    R = R.astype(jnp.result_type(R, T))
    def backup(v):
        return R + discount * jnp.einsum("sat,t->sa", T, v)

    v, _ = lax.scan(lambda v, _: (jax_logsumexp(backup(v), axis=1), None),
                    jnp.zeros(R.shape[0], R.dtype), None, length=n_iter)
    Q = backup(v)
    return Q - jax_logsumexp(Q, axis=1, keepdims=True), v


def soft_vi_pairs(T, R2, discount=DEFAULT_DISCOUNT, n_iter=DEFAULT_VI_ITERS):
    """
    S-2 soft VI over (s, p) pairs. R2 (S, S, A) indexed [s, p, a]; V2 (S, S) indexed [s, p].
    T (S, A, S') first-order (factored backup) or (S, S, A, S') second-order [s, p, a, s'].
    Returns log pi2 (S, S, A), V2 (S, S).
    """
    second_order = T.ndim == 4
    R2 = R2.astype(jnp.result_type(R2, T))

    def backup(V):
        if second_order:
            return R2 + discount * jnp.einsum("spat,ts->spa", T, V)
        return R2 + discount * jnp.einsum("sat,ts->sa", T, V)[:, None, :]

    S = R2.shape[0]
    V, _ = lax.scan(lambda V, _: (jax_logsumexp(backup(V), axis=2), None),
                    jnp.zeros((S, S), R2.dtype), None, length=n_iter)
    Q = backup(V)
    return Q - jax_logsumexp(Q, axis=2, keepdims=True), V


def policy_logits(params, apply_fn, T, mode="s1", discount=DEFAULT_DISCOUNT, n_iter=DEFAULT_VI_ITERS):
    """
    log pi for every hidden mode:
        "s1": (K, S, A) from the expanded-input reward;
        "s2": (K, S*S, A), pair index s * S + p (same as one_hot_jax2_nn).
    """
    S = T.shape[0]
    if mode == "s1":
        R = reward_table_expand(params, apply_fn, S)
        return vmap(lambda r: soft_vi(T, r, discount, n_iter)[0])(R)
    if mode == "s2":
        R2 = reward_table_pairs(params, apply_fn, S)
        lp = vmap(lambda r: soft_vi_pairs(T, r, discount, n_iter)[0])(R2)
        return lp.reshape(lp.shape[0], S * S, lp.shape[-1])
    raise ValueError(f"mode must be 's1' or 's2', got {mode}")


@partial(jax.jit, static_argnames=("mode", "n_iter"))
def policy_logits_state(R_state, T, discount, mode="s1", n_iter=DEFAULT_VI_ITERS):
    return policy_logits(R_state.params, R_state.apply_fn, T, mode, discount, n_iter)


def reward_tables(R_state, n_states, mode="s1"):
    """Learned reward: "s1" (K, S, A); "s2" (K, S, S, A) indexed [k, s, p, a]."""
    if mode == "s1":
        return np.asarray(reward_table_expand(R_state.params, R_state.apply_fn, n_states))
    return np.asarray(reward_table_pairs(R_state.params, R_state.apply_fn, n_states))


# ---------------------------------------------------------------------------
# HMM pieces (unchanged maths)
# ---------------------------------------------------------------------------

def comp_ll_jax(logits, one_hot_x, one_hot_a):
    logits = logits - jax_logsumexp(logits, axis=-1, keepdims=True)
    transition_probs = jnp.einsum('...i,...ij->...j', one_hot_x, logits)
    lls = jnp.sum(one_hot_a * transition_probs, axis=-1)
    return lls


def emission_ll(logpi, xe, a):
    """logpi (K, X, A), xe / a (T,) int -> (T, K). Same as comp_ll_jax on one-hots, without them."""
    return logpi[:, xe, a].T


def comp_log_transP(log_Ps, Rs, one_hot_x):
    T = one_hot_x.shape[0]
    log_Ps = jnp.tile(log_Ps[None, :, :], (T-1, 1, 1))
    log_Ps = log_Ps + jnp.dot(one_hot_x[:-1, 0, :], Rs[:, 0, :])[:, None, :]
    return log_Ps - jax_logsumexp(log_Ps, axis=2, keepdims=True)


def comp_transP(log_Ps, Rs, one_hot_x):
    return jnp.exp(comp_log_transP(log_Ps, Rs, one_hot_x))


def comp_log_transP_i(log_Ps, one_hot_x):
    T = one_hot_x.shape[0]
    log_Ps = jnp.tile(log_Ps[None, :, :], (T-1, 1, 1))
    return log_Ps - jax_logsumexp(log_Ps, axis=2, keepdims=True)


def comp_transP_i(log_Ps, one_hot_x):
    return jnp.exp(comp_log_transP_i(log_Ps, one_hot_x))


def _viterbi_JAX(pi0, Ps, ll):
    """Viterbi in JAX (Ps stationary (1, K, K) or time-varying (T-1, K, K))."""
    T, K = ll.shape
    hetero = (Ps.shape[0] == T-1)

    def score_fn(carry, inputs):
        score_next_t = carry
        t = inputs
        vals = jnp.log(Ps[t * hetero]) + score_next_t + ll[t+1]

        def comp_arg_score_over_K(vals):
            return jnp.argmax(vals), jnp.max(vals)

        arg_next_t, score_t = vmap(comp_arg_score_over_K)(vals)
        return score_t[None, :], (score_t, arg_next_t)

    _, (scores, args) = lax.scan(score_fn, jnp.zeros((1, K)), jnp.arange(T-2, -1, -1))
    z0 = jnp.array([(scores[-1] + jnp.log(pi0) + ll[0]).argmax()])

    def scan_z(carry, inputs):
        z_t = inputs[carry.astype(int)]
        return z_t, z_t

    _, z = lax.scan(scan_z, z0, args[::-1])
    return jnp.concatenate([z0, z.squeeze(-1)]).astype(int)


def forward(pi0, Ps, log_likes):
    alpha0 = jnp.log(pi0) + log_likes[0]

    def scan_body(alpha_prev, inputs):
        Ps_t, log_like_t = inputs
        m = jnp.max(alpha_prev)
        alpha_t = jnp.log(jnp.dot(jnp.exp(alpha_prev - m), Ps_t)) + m + log_like_t
        return alpha_t, alpha_t

    _, alphas = lax.scan(scan_body, alpha0, (Ps, log_likes[1:]))
    return jnp.concatenate([alpha0[None, :], alphas])


def backward(Ps, log_likes):
    K = log_likes.shape[1]
    betaT = jnp.zeros((K))

    def scan_body(beta_next, inputs):
        Ps_t, log_like_next = inputs
        tmp = log_like_next + beta_next
        m = jnp.max(tmp)
        beta_t = jnp.log(jnp.dot(Ps_t, jnp.exp(tmp - m))) + m
        return beta_t, beta_t

    _, betas = lax.scan(scan_body, betaT, (Ps[::-1], log_likes[1:][::-1]))
    return jnp.concatenate([betas[::-1], betaT[None, :]])


def expected_states(alphas, betas, Ps, ll):
    expected_states = alphas + betas
    expected_states -= jax_logsumexp(expected_states, axis=1, keepdims=True)
    expected_states = jnp.exp(expected_states)

    log_Ps = jnp.log(Ps)
    expected_joints = alphas[:-1, :, None] + betas[1:, None, :] + ll[1:, None, :] + log_Ps
    expected_joints -= expected_joints.max((1, 2))[:, None, None]
    expected_joints = jnp.exp(expected_joints)
    expected_joints /= expected_joints.sum((1, 2))[:, None, None]
    return expected_states, expected_joints


# ---------------------------------------------------------------------------
# E-step / likelihood / Viterbi for all sequences at once
# ---------------------------------------------------------------------------

@jax.jit
def e_step_all(logpi0, log_Ps, Rs, logpi, xoh, xe, a):
    """
    xoh (N, T, 1, S) one-hot states (input-driven transitions), xe (N, T) emission index
    (state for S-1, pair s*S+p for S-2), a (N, T) actions. Returns gamma (N,T,K), xi (N,T-1,K,K),
    per-sequence log-likelihood (N,).
    """
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))

    def one(xoh_i, xe_i, a_i):
        Ps = comp_transP(log_Ps, Rs, xoh_i)
        ll = emission_ll(logpi, xe_i, a_i)
        alpha = forward(pi0, Ps, ll)
        beta = backward(Ps, ll)
        g, xi = expected_states(alpha, beta, Ps, ll)
        return g, xi, jax_logsumexp(alpha[-1])

    return vmap(one)(xoh, xe, a)


@jax.jit
def loglik_and_paths(logpi0, log_Ps, Rs, logpi, xoh, xe, a):
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))

    def one(xoh_i, xe_i, a_i):
        Ps = comp_transP(log_Ps, Rs, xoh_i)
        ll = emission_ll(logpi, xe_i, a_i)
        return jax_logsumexp(forward(pi0, Ps, ll)[-1]), _viterbi_JAX(pi0, Ps, ll)

    return vmap(one)(xoh, xe, a)


# ---------------------------------------------------------------------------
# M-steps (jitted once at module level)
# ---------------------------------------------------------------------------

def pi0_m_step(gammas):
    pi0 = jnp.sum(gammas[:, 0], axis=0) + 1e-8
    return jnp.log(pi0 / pi0.sum())


def pi0_m_step_numpy(expectations):
    pi0 = sum([Ez[0] for Ez, _ in expectations]) + 1e-8
    return np.log(pi0 / pi0.sum())


@partial(jax.jit, static_argnames=("mode", "n_steps", "n_iter"))
def _emit_steps(R_state, T, gamma, xe, a, discount, mode, n_steps, n_iter):
    ntot = gamma.shape[0] * gamma.shape[1]
    g = jnp.moveaxis(gamma, -1, 0)  # (K, N, T)

    def loss_fn(params):
        logpi = policy_logits(params, R_state.apply_fn, T, mode, discount, n_iter)
        return -jnp.sum(g * logpi[:, xe, a]) / ntot

    def body(st, _):
        loss, grads = jax.value_and_grad(loss_fn)(st.params)
        return st.apply_gradients(grads=grads), loss

    return lax.scan(body, R_state, None, length=n_steps)


def emit_m_step(R_state, T, gamma, xe, a, mode="s1", num_iters=800, discount=DEFAULT_DISCOUNT,
                vi_iters=DEFAULT_VI_ITERS, tol=None, chunk=50):
    """
    Adam on the expected emission log-likelihood. Runs in jitted chunks of ``chunk`` steps; with
    ``tol`` it stops once the mean loss of a chunk improves by less than tol (relative).
    """
    T = jnp.asarray(T)
    done, prev, losses = 0, None, None
    while done < num_iters:
        n = int(min(chunk, num_iters - done))
        R_state, losses = _emit_steps(R_state, T, gamma, xe, a, discount, mode, n, vi_iters)
        done += n
        cur = float(jnp.mean(losses))
        if not np.isfinite(cur):
            raise FloatingPointError(f"emission M-step loss is {cur} after {done} steps")
        if tol is not None and prev is not None and abs(prev - cur) <= tol * max(abs(prev), 1e-12):
            break
        prev = cur
    return R_state, (float(losses[-1]) if losses is not None else np.nan)


def _trans_objective(params, gamma, xi, xoh):
    log_Ps, Rs = params

    def one(xoh_i, xi_i):
        return jnp.sum(xi_i * comp_log_transP(log_Ps, Rs, xoh_i))

    return -jnp.sum(vmap(one)(xoh, xi)) / (xoh.shape[0] * xoh.shape[1])


_LBFGS = {}


def trans_m_step(log_Ps, Rs, gamma, xi, xoh, num_iters=1000, tol=1e-4):
    """Input-driven transition M-step (L-BFGS); one solver instance per (maxiter, tol) so it compiles once."""
    from jaxopt import LBFGS
    key = (int(num_iters), float(tol))
    if key not in _LBFGS:
        _LBFGS[key] = LBFGS(fun=_trans_objective, maxiter=key[0], tol=key[1], stop_if_linesearch_fails=True)
    res = _LBFGS[key].run((jnp.asarray(log_Ps), jnp.asarray(Rs)), gamma=gamma, xi=xi, xoh=xoh)
    return res.params


# ---------------------------------------------------------------------------
# Backwards-compatible wrappers (old one-hot API)
# ---------------------------------------------------------------------------

def _idx(oh):
    """(…, 1, D) or (…, D) one-hot -> (…) int index."""
    oh = jnp.asarray(oh)
    if oh.ndim >= 2 and oh.shape[-2] == 1:
        oh = oh[..., 0, :]
    return jnp.argmax(oh, axis=-1)


@partial(jax.jit, static_argnames=['threshold'])
def jax_soft_find_policy(transition_probabilities, reward, discount, threshold=100):
    """Old API: (V, Q, pi) with the stable policy."""
    _, v = soft_vi(transition_probabilities, reward, discount, threshold)
    Q = reward + discount * jnp.einsum("sat,t->sa", transition_probabilities, v)
    return jax_logsumexp(Q, axis=1), Q, jax.nn.softmax(Q, axis=1)


def vinet(trans_probs, R_params, apply_fn, discount=DEFAULT_DISCOUNT):
    """
    Old API: S-2 policy over the S x S pair space, returned as (pi, None, None).
    trans_probs may be the first-order (S, A, S) tensor (factored; recommended) or the legacy dense
    (S^2, A, S^2) pair tensor (exact old behaviour, O(S^4 A)).
    """
    T = jnp.asarray(trans_probs, dtype=jnp.float64)
    n_in = R_params["dense1"]["kernel"].shape[0]
    if T.shape[0] == n_in:  # legacy dense pair tensor
        R = apply_fn({'params': R_params}, jnp.eye(n_in)).transpose(1, 0, 2)
        logpi = vmap(lambda r: soft_vi(T, r, discount)[0])(R)
    else:
        logpi = policy_logits(R_params, apply_fn, T, "s2", discount)
    return jnp.exp(logpi), None, None


def vinet_expand(trans_probs, R_params, apply_fn, discount=DEFAULT_DISCOUNT):
    logpi = policy_logits(R_params, apply_fn, jnp.asarray(trans_probs, dtype=jnp.float64), "s1", discount)
    return jnp.exp(logpi), None, None


def trans_m_step_jax_jaxopt(log_Ps, Rs, expectations, one_hot_xs, num_iters=1000, **kwargs):
    return trans_m_step(log_Ps, Rs, expectations[0], expectations[1], jnp.asarray(one_hot_xs), num_iters)
