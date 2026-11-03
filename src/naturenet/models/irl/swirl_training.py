#Code adapted from https://github.com/BRAINML-GT/SWIRL

import numpy as np
from scipy.optimize import minimize

#JAX
import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax import grad as jax_grad  # noqa: F401 (kept for star-importers)
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import optax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")


@partial(jax.jit, static_argnames=['threshold'])
def jax_soft_find_policy(transition_probabilities, reward, discount,
                threshold=100):
    n_states, n_actions = reward.shape
    v = jnp.ones(n_states)

    Q = jnp.ones((n_states, n_actions))


    def scan_iter(carry, inputs):

        v, Q = carry

        new_Q = vmap(vmap(lambda r_sa, tp_sa: r_sa + discount * jnp.dot(tp_sa, v)))(reward, transition_probabilities)
        new_v = jax.scipy.special.logsumexp(new_Q, axis=1)

        return (new_v, new_Q), 0

    (v, Q), _ = lax.scan(scan_iter, (v, Q), jnp.arange(threshold))

    Q -= Q.max(axis=1).reshape((n_states, 1))

    policy = jnp.exp(Q)/jnp.exp(Q).sum(axis=1, keepdims=True)
    
    return v, Q, policy

def vi(trans_prob, reward, discount=0.95):
    VG, QG, piG = jax_soft_find_policy(jnp.array(trans_prob), reward, discount=discount)
    return piG, QG, VG

@partial(jax.jit, static_argnames=['threshold'])
def jax_soft_find_policy_temp(transition_probabilities, reward, discount, temp, threshold=100):
    n_states, n_actions = reward.shape
    v = jnp.ones(n_states)
    Q = jnp.ones((n_states, n_actions))

    def scan_iter(carry, inputs):
        v, Q = carry
        new_Q = vmap(vmap(lambda r_sa, tp_sa: r_sa + discount * jnp.dot(tp_sa, v)))(
            reward, transition_probabilities)
        new_v = temp * jax.scipy.special.logsumexp(new_Q / temp, axis=1)
        return (new_v, new_Q), 0

    (v, Q), _ = lax.scan(scan_iter, (v, Q), jnp.arange(threshold))
    policy = jnp.exp((Q - v[:, None]) / temp)
    policy += 1e-8
    policy /= jnp.sum(policy, axis=1, keepdims=True)
    return v, Q, policy

def vi_temp(trans_prob, reward, temp, discount=0.95):
    VG, QG, piG = jax_soft_find_policy_temp(
        jnp.array(trans_prob), reward, discount=discount, temp=temp)
    return piG, QG, VG

def comp_ll_jax(logits, one_hot_x, one_hot_a):
    logits = logits - jax_logsumexp(logits, axis=-1, keepdims=True)
    transition_probs = jnp.einsum('...i,...ij->...j', one_hot_x, logits)
    lls = jnp.sum(one_hot_a * transition_probs, axis=-1)
    return lls

def sigmoid_jax(x):
    return 1 / (1 + jnp.exp(-x))

def comp_log_transP(log_Ps, Wbs, one_hot_x):
    T = one_hot_x.shape[0]
    W1, b1, W2, b2 = Wbs
    log_Ps = jnp.tile(log_Ps[None, :, :], (T-1, 1, 1))

    Z1 = jnp.dot(one_hot_x[:-1, 0, :], W1) + b1
    A1 = sigmoid_jax(Z1)
    Z2 = jnp.dot(A1, W2) + b2
    log_Ps = log_Ps + Z2[:, None, :]
    return log_Ps - jax_logsumexp(log_Ps, axis=2, keepdims=True)
    
def comp_transP(log_Ps, Wbs, one_hot_x):
    return jnp.exp(comp_log_transP(log_Ps, Wbs, one_hot_x))

def oh_mask(oh):
    """(..., T) True on real steps: padded steps (segment_policy: variable) have all-zero one-hots."""
    return oh.reshape(oh.shape[:-2] + (-1,)).sum(-1) > 0 if oh.ndim >= 3 else oh.sum(-1) > 0


def pad_eye(Ps, aoh):
    """
    (T-1, K, K) transitions with the identity into padded steps (aoh row all zero). With the zero emission
    LL of a padded step this leaves the forward LL, the real-step posteriors AND the Viterbi path exact (the
    padded tail continues every state with probability 1). No-op without padding.
    """
    m = oh_mask(aoh)[1:]
    return jnp.where(m[:, None, None], Ps, jnp.eye(Ps.shape[-1], dtype=Ps.dtype))


def viterbi_JAX(pi0, Ps, ll):
    """
    Implements the Viterbi algorithm in JAX.
    """
    T, K = ll.shape

    # Check if the transition matrices are stationary or time-varying (hetero)
    hetero = (Ps.shape[0] == T-1)

    def score_fn(carry, inputs):
        score_next_t = carry
        t = inputs

        vals = jnp.log(Ps[t * hetero]) + score_next_t + ll[t+1]
        
        def comp_arg_score_over_K(vals):
            return jnp.argmax(vals), jnp.max(vals)
        
        arg_next_t, score_t = vmap(comp_arg_score_over_K)(vals)

        return score_t[None, :], (score_t, arg_next_t)
    
    _, (scores, args) = lax.scan(score_fn, jnp.zeros((1, K)), jnp.arange(T-2,-1,-1))

    z0 = jnp.array([(scores[-1] + jnp.log(pi0) + ll[0]).argmax()])
    
    def scan_z(carry, inputs):
        z_prev_t = carry
        arg_next_t = inputs

        z_t = arg_next_t[z_prev_t.astype(int)]

        return z_t, z_t

    _, z = lax.scan(scan_z, z0, args[::-1])

    return jnp.concatenate([z0, z.squeeze()]).astype(int)

def forward(pi0, Ps, log_likes):
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    alpha0 = jnp.log(pi0) + log_likes[0]
    def scan_body(carry, inputs):
        alpha_prev = carry
        Ps_t, log_like_t = inputs
        m = jnp.max(alpha_prev)
        alpha_t = jnp.log(jnp.dot(jnp.exp(alpha_prev - m), Ps_t)) + m + log_like_t
        return alpha_t, alpha_t
    _, alphas = lax.scan(scan_body, alpha0, (Ps, log_likes[1:]))
    return jnp.concatenate([alpha0[None, :], alphas])

def backward(Ps, log_likes):
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    betaT = jnp.zeros((K))

    def scan_body(carry, inputs):
        beta_next = carry
        Ps_t, log_like_next = inputs
        tmp = log_like_next + beta_next
        m = jnp.max(tmp)
        beta_t = jnp.log(jnp.dot(Ps_t, jnp.exp(tmp - m))) + m

        return beta_t, beta_t
        
    _, betas = lax.scan(scan_body, betaT, (Ps[::-1], log_likes[1:][::-1]))

    return jnp.concatenate([betas[::-1], betaT[None, :]])

def expected_states(alphas, betas, Ps, ll):
    T, K = ll.shape

    expected_states = alphas + betas
    expected_states -= jax_logsumexp(expected_states, axis=1, keepdims=True)
    expected_states = jnp.exp(expected_states)

    log_Ps = jnp.log(Ps)

    expected_joints = alphas[:-1,:,None] + betas[1:,None,:] + ll[1:,None,:] + log_Ps
    expected_joints -= expected_joints.max((1,2))[:,None, None]
    expected_joints = jnp.exp(expected_joints)
    expected_joints /= expected_joints.sum((1,2))[:,None,None]

    return expected_states, expected_joints

def jax_e_step_logpi2(pi0, log_Ps, Rs, logemit, trans_probs, xoh, xoh2, aoh):
    Ps_jax = pad_eye(comp_transP(jnp.array(log_Ps), Rs, jnp.array(xoh)), aoh)
    log_likes_jax = comp_ll_jax(jnp.array(logemit), jnp.array(xoh2), jnp.array(aoh))
    alpha_jax = forward(pi0, Ps_jax, log_likes_jax)
    beta_jax = backward(Ps_jax, log_likes_jax)
    gamma_jax, xi_jax = expected_states(alpha_jax, beta_jax, Ps_jax, log_likes_jax)
    return gamma_jax, xi_jax, alpha_jax

def jax_e_step_logpi(pi0, log_Ps, Rs, logemit, trans_probs, xoh, aoh):
    Ps_jax = pad_eye(comp_transP(jnp.array(log_Ps), Rs, jnp.array(xoh)), aoh)
    log_likes_jax = comp_ll_jax(jnp.array(logemit), jnp.array(xoh), jnp.array(aoh))
    alpha_jax = forward(pi0, Ps_jax, log_likes_jax)
    beta_jax = backward(Ps_jax, log_likes_jax)
    gamma_jax, xi_jax = expected_states(alpha_jax, beta_jax, Ps_jax, log_likes_jax)
    return gamma_jax, xi_jax, alpha_jax

LOG_EPS = 1e-12


def _safe_log(p):
    return jnp.log(jnp.clip(p, LOG_EPS))


def _rewards_sa(rewards, n_states, n_actions):
    # (K, 1, S) state-only reward broadcast over actions -> (K, S, A)
    return jnp.expand_dims(rewards[:, 0, :], axis=2) * jnp.ones((1, n_states, n_actions))


def _log_policy(rewards, trans_probs, temps=None, use_temp=False):
    """Log policy (K, S, A). Always log(pi): the *_naturenet variants previously used pi itself as logits."""
    n_states, n_actions = trans_probs.shape[0], trans_probs.shape[1]
    rewards_sa = _rewards_sa(rewards, n_states, n_actions)
    if use_temp:
        pi = vmap(lambda r, t: jax_soft_find_policy_temp(trans_probs, r, 0.95, t)[2])(rewards_sa, temps)
    else:
        pi = vmap(lambda r: jax_soft_find_policy(trans_probs, r, 0.95)[2])(rewards_sa)
    return _safe_log(pi)


# ---------------------------------------------------------------------------
# E-step (jitted: compiled once per input shape and reused every EM iteration)
# ---------------------------------------------------------------------------

@jax.jit
def jax_e_step_batch2_naturenet(pi0, log_Ps, Rs, rewards, trans_probs, xoh_list, xoh_list2, aoh_list):
    logemit = _log_policy(rewards, trans_probs)
    return vmap(partial(jax_e_step_logpi2, pi0, log_Ps, Rs, logemit, trans_probs))(xoh_list, xoh_list2, aoh_list)


@jax.jit
def jax_e_step_batch_naturenet(pi0, log_Ps, Rs, rewards, trans_probs, xoh_list, aoh_list):
    logemit = _log_policy(rewards, trans_probs)
    return vmap(partial(jax_e_step_logpi, pi0, log_Ps, Rs, logemit, trans_probs))(xoh_list, aoh_list)


@jax.jit
def jax_e_step_batch2_temp(pi0, log_Ps, Rs, rewards, trans_probs, temps, xoh_list, xoh_list2, aoh_list):
    logemit = _log_policy(rewards, trans_probs, temps, use_temp=True)
    return vmap(partial(jax_e_step_logpi2, pi0, log_Ps, Rs, logemit, trans_probs))(xoh_list, xoh_list2, aoh_list)


@jax.jit
def jax_e_step_batch_temp(pi0, log_Ps, Rs, rewards, trans_probs, temps, xoh_list, aoh_list):
    logemit = _log_policy(rewards, trans_probs, temps, use_temp=True)
    return vmap(partial(jax_e_step_logpi, pi0, log_Ps, Rs, logemit, trans_probs))(xoh_list, aoh_list)


# ---------------------------------------------------------------------------
# M-steps. The objectives are module-level and jitted (value_and_grad), so they are traced once per
# shape instead of being re-built and run op-by-op on every EM iteration; scipy gets f and grad from
# one call (jac=True) instead of two separate passes.
# ---------------------------------------------------------------------------

def _emit_loss(flat, trans_probs, temps, gammas, xohs, aohs, n_total, shape, use_temp):
    rewards = flat.reshape(shape)
    logemit = _log_policy(rewards, trans_probs, temps, use_temp=use_temp)
    lls = vmap(partial(comp_ll_jax, logemit))(xohs, aohs)  # (N, T, K)
    return -jnp.sum(gammas * lls) / n_total


_emit_vg = jax.jit(jax.value_and_grad(_emit_loss), static_argnames=("shape", "use_temp"))


def _unflatten_trans(flat, shapes):
    out, i = [], 0
    for shp in shapes:
        n = int(np.prod(shp))
        out.append(flat[i:i + n].reshape(shp))
        i += n
    return out[0], tuple(out[1:])


def _trans_loss(flat, xis, xohs, n_total, shapes):
    log_Ps, Rs = _unflatten_trans(flat, shapes)
    log_trans = vmap(lambda x: comp_log_transP(log_Ps, Rs, x))(xohs)  # (N, T-1, K, K)
    m = oh_mask(xohs)[:, 1:, None, None]                              # transitions into real steps only
    return -jnp.sum(jnp.where(m, xis * log_trans, 0.0)) / n_total


def n_real(one_hot_xs):
    """Real (unpadded) steps in a (N, T, ...) one-hot batch (= N * T without padding)."""
    return float(np.asarray(oh_mask(jnp.asarray(one_hot_xs))).sum())


_trans_vg = jax.jit(jax.value_and_grad(_trans_loss), static_argnames=("shapes",))


def _minimize(vg, x0, method, num_iters, **kw):
    def f(x):
        v, g = vg(jnp.asarray(x), **kw)
        return float(v), np.asarray(g, dtype=np.float64)

    return minimize(f, np.asarray(x0, dtype=np.float64), jac=True, method=method,
                    options=dict(maxiter=num_iters), tol=1e-4)


def trans_m_step_jax_scipy2(log_Ps, Rs, expectations, one_hot_xs, num_iters=1000, **kwargs):
    one_hot_xs = jnp.asarray(one_hot_xs)
    W1, b1, W2, b2 = Rs
    parts = [jnp.asarray(p) for p in (log_Ps, W1, b1, W2, b2)]
    shapes = tuple(tuple(int(d) for d in p.shape) for p in parts)
    flat0 = jnp.concatenate([p.ravel() for p in parts])
    n_total = n_real(one_hot_xs)
    result = _minimize(_trans_vg, flat0, "L-BFGS-B", num_iters, xis=jnp.asarray(expectations[1]),
                       xohs=one_hot_xs, n_total=n_total, shapes=shapes)
    log_Ps_opt, Rs_opt = _unflatten_trans(result.x, shapes)
    return log_Ps_opt, Rs_opt


def _emit_m_step(rewards, trans_probs, temps, expectations, one_hot_xs, one_hot_acs, num_iters, use_temp):
    rewards = jnp.asarray(rewards)
    one_hot_xs = jnp.asarray(one_hot_xs)
    shape = tuple(int(d) for d in rewards.shape)
    n_total = n_real(one_hot_xs)
    result = _minimize(_emit_vg, rewards.ravel(), "BFGS", num_iters, trans_probs=jnp.asarray(trans_probs),
                       temps=None if temps is None else jnp.asarray(temps), gammas=jnp.asarray(expectations[0]),
                       xohs=one_hot_xs, aohs=jnp.asarray(one_hot_acs), n_total=n_total, shape=shape,
                       use_temp=use_temp)
    return result.x.reshape(shape)


def emit_m_step_jax_scipy2(rewards, trans_probs, expectations, one_hot_xs, one_hot_acs, num_iters=1000, **kwargs):
    return _emit_m_step(rewards, trans_probs, None, expectations, one_hot_xs, one_hot_acs, num_iters, False)


def emit_m_step_jax_scipy2_naturenet(rewards, trans_probs, expectations, one_hot_xs, one_hot_acs, num_iters=1000, **kwargs):
    # Same objective as emit_m_step_jax_scipy2 now that log(pi) is used (was: logemit = pi).
    return _emit_m_step(rewards, trans_probs, None, expectations, one_hot_xs, one_hot_acs, num_iters, False)


def emit_m_step_jax_scipy2_temp(rewards, trans_probs, temps, expectations, one_hot_xs, one_hot_acs, num_iters=1000, **kwargs):
    return _emit_m_step(rewards, trans_probs, temps, expectations, one_hot_xs, one_hot_acs, num_iters, True)


def emit_m_step_jax_scipy_temp_reg(rewards, trans_probs, temps, expectations, one_hot_xs, one_hot_acs, num_iters=1000, **kwargs):
    # Regularisation terms were commented out in the original; identical to the temp M-step.
    return _emit_m_step(rewards, trans_probs, temps, expectations, one_hot_xs, one_hot_acs, num_iters, True)


def pi0_m_step(gammas):
    pi0 = sum(gammas[:, 0]) + 1e-8
    return jnp.log(pi0 / pi0.sum())
