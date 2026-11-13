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
from typing import NamedTuple

from naturenet.models.irl.jax_precision import configure as _configure_precision
_configure_precision()   # float64 unless NATURENET_JAX_X64=0 (jax_precision.py)
# jax.config.update("jax_platform_name", "cpu")


class S2Trans(NamedTuple):
    """
    Structured S-2 transitions. The S-2 tensor over augmented states (x, j), j = slot of the previous state, is

        new_T[(x, j), a, (x', slot(x', x))] = T[x, a, x']        (the same for every j)

    so it never needs to be built: E[v(next) | (x, j), a] = sum_x' T[x, a, x'] v[G[x, x']] with
    G[x, x'] = x' * M + slot(x', x). Value iteration then costs S*A*S per sweep instead of (S*M)^2*A (M^2 less
    compute) and the device holds T (S, A, S) instead of the dense (S*M, A, S*M) tensor. Exact: same policy as
    the dense new_T (checked in test_suff_stats.py).
    """
    T: object   # (S, A, S) base transitions
    G: object   # (S, S) int: augmented index of x' reached from x (0 where T[x, :, x'] == 0)


def is_s2(tp):
    return isinstance(tp, S2Trans)


def device_tp(tp):
    """Transition input on the device: dense (S, A, S') array, or S2Trans with device fields."""
    if is_s2(tp):
        return S2Trans(jnp.asarray(tp.T, dtype=jnp.float64), jnp.asarray(tp.G, dtype=jnp.int32))
    return jnp.asarray(tp.todense() if hasattr(tp, "todense") else tp)


def tp_n_actions(tp):
    return int((tp.T if is_s2(tp) else tp).shape[1])


def tp_n_states(tp):
    """Number of (augmented) states the policy is defined on (S*M for S2Trans needs M: use the reward shape)."""
    if is_s2(tp):
        raise ValueError("S2Trans: take the augmented state count from the reward shape")
    return int(tp.shape[0])


def expected_next_v(tp, v, n_states):
    """(n_states, A) expected next-state value: dense sum_y T[s, a, y] v[y], or the structured S-2 form."""
    if is_s2(tp):
        S = tp.T.shape[0]
        ev = jnp.einsum("xay,xy->xa", tp.T, v[tp.G])           # (S, A)
        return jnp.repeat(ev, n_states // S, axis=0)             # row x*M + j = row x
    return jnp.einsum("say,y->sa", tp, v)


@partial(jax.jit, static_argnames=['threshold'])
def jax_soft_find_policy(transition_probabilities, reward, discount,
                threshold=100):
    n_states, n_actions = reward.shape
    v = jnp.ones(n_states)

    Q = jnp.ones((n_states, n_actions))


    def scan_iter(carry, inputs):

        v, Q = carry

        new_Q = reward + discount * expected_next_v(transition_probabilities, v, n_states)
        new_v = jax.scipy.special.logsumexp(new_Q, axis=1)

        return (new_v, new_Q), 0

    (v, Q), _ = lax.scan(scan_iter, (v, Q), jnp.arange(threshold))

    Q -= Q.max(axis=1).reshape((n_states, 1))

    policy = jnp.exp(Q)/jnp.exp(Q).sum(axis=1, keepdims=True)
    
    return v, Q, policy

def vi(trans_prob, reward, discount=0.95):
    VG, QG, piG = jax_soft_find_policy(device_tp(trans_prob), reward, discount=discount)
    return piG, QG, VG

@partial(jax.jit, static_argnames=['threshold'])
def jax_soft_find_policy_temp(transition_probabilities, reward, discount, temp, threshold=100):
    n_states, n_actions = reward.shape
    v = jnp.ones(n_states)
    Q = jnp.ones((n_states, n_actions))

    def scan_iter(carry, inputs):
        v, Q = carry
        new_Q = reward + discount * expected_next_v(transition_probabilities, v, n_states)
        new_v = temp * jax.scipy.special.logsumexp(new_Q / temp, axis=1)
        return (new_v, new_Q), 0

    (v, Q), _ = lax.scan(scan_iter, (v, Q), jnp.arange(threshold))
    policy = jnp.exp((Q - v[:, None]) / temp)
    policy += 1e-8
    policy /= jnp.sum(policy, axis=1, keepdims=True)
    return v, Q, policy

def vi_temp(trans_prob, reward, temp, discount=0.95):
    VG, QG, piG = jax_soft_find_policy_temp(
        device_tp(trans_prob), reward, discount=discount, temp=temp)
    return piG, QG, VG

def comp_ll_jax(logits, one_hot_x, one_hot_a):
    """
    (T, K) emission LLs. one_hot_x is either the (T, 1, S) one-hot states or an integer (T, 1) / (T,) array of
    state indices (S-2: x * M + slot, -1 on padded steps). The index form gives the same LLs without the dense
    (N, T, S*M) one-hot tensor, which is what runs out of GPU memory for S-2 with many sequences / slots.
    Padded steps have an all-zero one_hot_a row, so their LL is 0 in both forms.
    """
    logits = logits - jax_logsumexp(logits, axis=-1, keepdims=True)
    if jnp.issubdtype(one_hot_x.dtype, jnp.integer):
        idx = jnp.clip(one_hot_x.reshape(one_hot_x.shape[0]), 0, logits.shape[-2] - 1)
        transition_probs = jnp.moveaxis(jnp.take(logits, idx, axis=-2), -2, 0)  # (T, K, A)
    else:
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
    """
    (..., T) True on real steps: padded steps (segment_policy: variable) have all-zero one-hots. Integer
    (..., T, 1) index arrays (S-2 index emissions, see comp_ll_jax) mark padded steps with -1.
    """
    if jnp.issubdtype(oh.dtype, jnp.integer):
        return oh[..., 0] >= 0
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
    n_states, n_actions = int(rewards.shape[-1]), tp_n_actions(trans_probs)
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

# Runtime options (set once per run from the YAML via set_mstep_options; the defaults are the original ones).
#   emit_iters / trans_iters : max BFGS / L-BFGS-B iterations per M-step (generalised EM: each M-step is warm-started
#                              from the previous one, so 50-200 is usually enough; 1000 = original).
#   emit_method              : "auto" (default: BFGS up to emit_bfgs_max_params rewards, i.e. S-1 as before, else
#                              L-BFGS-B), "BFGS" (original everywhere) or "L-BFGS-B". scipy BFGS keeps a dense n x n
#                              inverse Hessian: for S-2 (n = K*S*M, e.g. 3375) its update alone is ~1 s / iteration.
#   suff_stats               : True = exact sufficient-statistic M-steps (see below), False = original per-step losses.
#   emit_lbfgs_ftol          : L-BFGS-B relative f tolerance for the emission M-step (scipy's tol=1e-4 maps to
#                              ftol=1e-4, which stops far too early on the reward objective).
MSTEP = dict(emit_iters=1000, trans_iters=1000, emit_method="auto", emit_bfgs_max_params=1000,
             emit_lbfgs_ftol=1e-10, trans_method="L-BFGS-B", tol=1e-4, suff_stats=True)


def set_mstep_options(yml_conf=None, **kw):
    """Update MSTEP from yml keys (mstep_emit_iters, mstep_trans_iters, mstep_emit_method, mstep_tol,
    mstep_suff_stats) and/or keyword overrides. Returns the active options."""
    if yml_conf is not None:
        for k in ("emit_iters", "trans_iters", "emit_method", "emit_bfgs_max_params", "emit_lbfgs_ftol",
                  "trans_method", "tol", "suff_stats"):
            v = yml_conf.get("mstep_" + k)
            if v is not None:
                MSTEP[k] = v
    MSTEP.update({k: v for k, v in kw.items() if v is not None})
    MSTEP["emit_iters"], MSTEP["trans_iters"] = int(MSTEP["emit_iters"]), int(MSTEP["trans_iters"])
    MSTEP["emit_bfgs_max_params"] = int(MSTEP["emit_bfgs_max_params"])
    MSTEP["tol"], MSTEP["suff_stats"] = float(MSTEP["tol"]), bool(MSTEP["suff_stats"])
    return dict(MSTEP)


# ---------------------------------------------------------------------------
# Sufficient statistics. Both M-step objectives depend on the data only through counts:
#   emission : -sum_{n,t,k} gamma[n,t,k] log pi_k(a_nt | x_nt)            = -sum_{k,s,a} C[k,s,a] log pi_k(a|s)
#   transition: -sum_{n,t,j,k} xi[n,t,j,k] log P(z'=k | z=j, x_nt)        = -sum_{s,j,k} X[s,j,k] log P_s(j,k)
# with C / X accumulated ONCE per E-step. The objective (and its gradient) is then independent of N and T, so
# each BFGS iteration costs one value iteration instead of one value iteration plus a pass over all N*T steps.
# Same optimum and same values as the per-step losses (to float rounding; checked in test_suff_stats.py).
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_rows",))
def emit_counts(gammas, xohs, aohs, n_rows):
    """(K, n_rows, A) expected (state, action) counts per mode. xohs: (N, T, 1, n_rows) one-hots or integer
    (N, T, 1) indices (-1 = pad); aohs (N, T, 1, A) one-hots (all zero on pads, so pads contribute nothing)."""
    N, T, K = gammas.shape
    a = aohs.reshape(N, T, -1)
    if jnp.issubdtype(xohs.dtype, jnp.integer):
        idx = xohs.reshape(N * T)
        w = (gammas[..., :, None] * a[..., None, :]).reshape(N * T, K, -1)          # (NT, K, A)
        w = jnp.where((idx >= 0)[:, None, None], w, 0.0)
        C = jnp.zeros((n_rows, K, a.shape[-1]), w.dtype).at[jnp.clip(idx, 0, n_rows - 1)].add(w)
        return jnp.moveaxis(C, 1, 0)
    return jnp.einsum("ntk,nts,nta->ksa", gammas, xohs.reshape(N, T, -1), a)


@jax.jit
def trans_counts(xis, xohs):
    """(S, K, K) expected mode transitions out of each state (transitions into padded steps excluded)."""
    m = oh_mask(xohs)[:, 1:]
    return jnp.einsum("nts,ntjk->sjk", xohs[:, :-1, 0, :], jnp.where(m[:, :, None, None], xis, 0.0))


def _state_log_transP(log_Ps, Wbs):
    """(S, K, K) log transition matrix for every state (= comp_log_transP on the one-hot of each state)."""
    W1, b1, W2, b2 = Wbs
    Z2 = jnp.dot(sigmoid_jax(W1 + b1), W2) + b2                     # (S, K): one-hot rows pick W1 rows
    lp = log_Ps[None, :, :] + Z2[:, None, :]
    return lp - jax_logsumexp(lp, axis=2, keepdims=True)


def _emit_loss_ss(flat, trans_probs, temps, counts, n_total, shape, use_temp):
    logemit = _log_policy(flat.reshape(shape), trans_probs, temps, use_temp=use_temp)
    logemit = logemit - jax_logsumexp(logemit, axis=-1, keepdims=True)    # as in comp_ll_jax
    return -jnp.sum(counts * logemit) / n_total


_emit_vg_ss = jax.jit(jax.value_and_grad(_emit_loss_ss), static_argnames=("shape", "use_temp"))


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


def _trans_loss_ss(flat, counts, n_total, shapes):
    log_Ps, Rs = _unflatten_trans(flat, shapes)
    return -jnp.sum(counts * _state_log_transP(log_Ps, Rs)) / n_total


def n_real(one_hot_xs):
    """Real (unpadded) steps in a (N, T, ...) one-hot batch (= N * T without padding)."""
    return float(np.asarray(oh_mask(jnp.asarray(one_hot_xs))).sum())


_trans_vg = jax.jit(jax.value_and_grad(_trans_loss), static_argnames=("shapes",))
_trans_vg_ss = jax.jit(jax.value_and_grad(_trans_loss_ss), static_argnames=("shapes",))


def _minimize(vg, x0, method, num_iters, _options=None, **kw):
    def f(x):
        v, g = vg(jnp.asarray(x), **kw)
        return float(v), np.asarray(g, dtype=np.float64)

    if _options:   # explicit tolerances (no scipy 'tol' override)
        return minimize(f, np.asarray(x0, dtype=np.float64), jac=True, method=method,
                        options=dict(maxiter=num_iters, **_options))
    return minimize(f, np.asarray(x0, dtype=np.float64), jac=True, method=method,
                    options=dict(maxiter=num_iters), tol=MSTEP["tol"])


def trans_m_step_jax_scipy2(log_Ps, Rs, expectations, one_hot_xs, num_iters=None, **kwargs):
    num_iters = MSTEP["trans_iters"] if num_iters is None else num_iters
    one_hot_xs = jnp.asarray(one_hot_xs)
    W1, b1, W2, b2 = Rs
    parts = [jnp.asarray(p) for p in (log_Ps, W1, b1, W2, b2)]
    shapes = tuple(tuple(int(d) for d in p.shape) for p in parts)
    flat0 = jnp.concatenate([p.ravel() for p in parts])
    n_total = n_real(one_hot_xs)
    xis = jnp.asarray(expectations[1])
    if MSTEP["suff_stats"]:
        result = _minimize(_trans_vg_ss, flat0, MSTEP["trans_method"], num_iters,
                           counts=trans_counts(xis, one_hot_xs), n_total=n_total, shapes=shapes)
    else:
        result = _minimize(_trans_vg, flat0, MSTEP["trans_method"], num_iters, xis=xis,
                           xohs=one_hot_xs, n_total=n_total, shapes=shapes)
    log_Ps_opt, Rs_opt = _unflatten_trans(result.x, shapes)
    return log_Ps_opt, Rs_opt


def _emit_m_step(rewards, trans_probs, temps, expectations, one_hot_xs, one_hot_acs, num_iters, use_temp):
    num_iters = MSTEP["emit_iters"] if num_iters is None else num_iters
    rewards = jnp.asarray(rewards)
    one_hot_xs = jnp.asarray(one_hot_xs)
    shape = tuple(int(d) for d in rewards.shape)
    n_total = n_real(one_hot_xs)
    tp = device_tp(trans_probs)
    tm = None if temps is None else jnp.asarray(temps)
    gam = jnp.asarray(expectations[0])
    method = MSTEP["emit_method"]
    if method == "auto":
        method = "BFGS" if rewards.size <= MSTEP["emit_bfgs_max_params"] else "L-BFGS-B"
    opts = dict(ftol=float(MSTEP["emit_lbfgs_ftol"]), gtol=MSTEP["tol"]) if method == "L-BFGS-B" else None
    if MSTEP["suff_stats"]:
        counts = emit_counts(gam, one_hot_xs, jnp.asarray(one_hot_acs), n_rows=int(shape[-1]))
        result = _minimize(_emit_vg_ss, rewards.ravel(), method, num_iters, _options=opts, trans_probs=tp,
                           temps=tm, counts=counts, n_total=n_total, shape=shape, use_temp=use_temp)
    else:
        result = _minimize(_emit_vg, rewards.ravel(), method, num_iters, _options=opts, trans_probs=tp,
                           temps=tm, gammas=gam, xohs=one_hot_xs, aohs=jnp.asarray(one_hot_acs),
                           n_total=n_total, shape=shape, use_temp=use_temp)
    return result.x.reshape(shape)


def emit_m_step_jax_scipy2(rewards, trans_probs, expectations, one_hot_xs, one_hot_acs, num_iters=None, **kwargs):
    return _emit_m_step(rewards, trans_probs, None, expectations, one_hot_xs, one_hot_acs, num_iters, False)


def emit_m_step_jax_scipy2_naturenet(rewards, trans_probs, expectations, one_hot_xs, one_hot_acs, num_iters=None, **kwargs):
    # Same objective as emit_m_step_jax_scipy2 now that log(pi) is used (was: logemit = pi).
    return _emit_m_step(rewards, trans_probs, None, expectations, one_hot_xs, one_hot_acs, num_iters, False)


def emit_m_step_jax_scipy2_temp(rewards, trans_probs, temps, expectations, one_hot_xs, one_hot_acs, num_iters=None, **kwargs):
    return _emit_m_step(rewards, trans_probs, temps, expectations, one_hot_xs, one_hot_acs, num_iters, True)


def emit_m_step_jax_scipy_temp_reg(rewards, trans_probs, temps, expectations, one_hot_xs, one_hot_acs, num_iters=None, **kwargs):
    # Regularisation terms were commented out in the original; identical to the temp M-step.
    return _emit_m_step(rewards, trans_probs, temps, expectations, one_hot_xs, one_hot_acs, num_iters, True)


def pi0_m_step(gammas):
    pi0 = sum(gammas[:, 0]) + 1e-8
    return jnp.log(pi0 / pi0.sum())
