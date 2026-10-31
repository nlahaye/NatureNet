"""
Hierarchical inverse action-value iteration (HIAVI; Zhu et al. 2024, "Multi-intention inverse Q-learning
for interpretable behavior representation", https://github.com/nrgrp/hiql), re-implemented for NatureNet.

Model: a K-state Markov chain over intentions z_t (pi0, Lambda) and, under intention k, actions drawn from a
Boltzmann policy pi_k(a | s) whose reward r_k(s, a) is recovered with IAVI on the (fixed) environment
transition model T(s' | s, a).

Differences from hiql.algorithms.HIAVI (same model and fixed points):
- any number of latents (hiql's fit() hard-codes 2 for pi0 / Lambda, so num_latents != 2 fails to broadcast);
- the reported log-likelihood is the forward (marginal) log p(a_{1:T} | s_{1:T}) per step. hiql reports
  mean_t logsumexp(log gamma_t + log pi(a_t|s_t)), which uses the posterior of the step being scored (the
  action itself), so it is not a likelihood and overstates held-out fit;
- IAVI recovers the demonstrated policy exactly (softmax(q_k) = normalised expert policy + eps, the
  defining property of IAVI), so it does not change the EM fixed point. The EM emission M-step is the
  closed form (posterior-weighted (s, a) counts) and IAVI runs once per latent after EM, giving the same
  pi_k, q_k, r_k as running it in every iteration. IAVI itself is vectorised over states (closed form for
  the lstsq step: r = eta - mean_a eta);
- correct backward pass: hiql's _get_mc_probs does `logp_beta_next += lpo` on the array it has just
  appended to the beta list, so every stored beta_t (t >= 1) has the emission at t added again; gamma_t and
  xi_t then double-count the emission at t (verified numerically against hiql.algorithms);
- deterministic soft M-step by default (mstep="soft"); mstep="sample" keeps hiql's stochastic
  responsibilities (Bernoulli(gamma) per step). EM stops on a log-likelihood tolerance, unchanged MAP
  paths, or max_iters (hiql's while-True loop can cycle forever with sampling);
- seeded numpy Generator, vectorised forward-backward / Viterbi over sequences of equal length (N, T).
"""
import numpy as np
from scipy.special import logsumexp

EPS = 1e-6  # hiql's smoothing for unvisited (s, a)


# ---------------------------------------------------------------------------
# IAVI (vectorised)
# ---------------------------------------------------------------------------

def iavi(T, expert_pi, discount, tol=1e-3, max_iters=10000, eps=EPS, q0=None):
    """
    Inverse action-value iteration (Kalweit et al. 2020) for a Boltzmann expert.

    T: (S, A, S') transition probabilities; expert_pi: (S, A) rows sum to 1.
    Returns r (S, A), q (S, A). Same fixed point as hiql.algorithms.IAVI.train (which sweeps states
    sequentially and solves a singular A x A least-squares problem per state; its min-norm solution is
    r = eta - mean_a(eta)).
    """
    S, A, _ = T.shape
    logpi = np.log(expert_pi + eps)
    q = np.zeros((S, A)) if q0 is None else np.array(q0, dtype=float)
    r = np.zeros((S, A))
    for _ in range(int(max_iters)):
        cont = discount * (T @ q.max(axis=1))  # (S, A): gamma * sum_s' T(s'|s,a) max_a' q(s', a')
        eta = logpi - cont
        r_new = eta - eta.mean(axis=1, keepdims=True)
        q = r_new + cont
        delta = np.max(np.abs(r_new - r))
        r = r_new
        if delta < tol:
            break
    return r, q


def softmax_policy(q):
    z = q - q.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# HMM pieces (N sequences of length T, K latents)
# ---------------------------------------------------------------------------

def emission_loglik(logpis, s, a):
    """logpis (K, S, A); s, a (N, T) int -> (N, T, K)."""
    return np.moveaxis(logpis[:, s, a], 0, -1)


def forward_backward(logp_init, logp_tr, lls):
    """
    lls (N, T, K). Returns log gamma (N, T, K), expected transition counts (K, K) summed over n and t,
    and the marginal log-likelihood per sequence (N,).
    """
    N, T, K = lls.shape
    la = np.empty((N, T, K))
    la[:, 0] = logp_init + lls[:, 0]
    for t in range(1, T):
        la[:, t] = logsumexp(la[:, t - 1][:, :, None] + logp_tr[None], axis=1) + lls[:, t]
    lb = np.zeros((N, T, K))
    for t in range(T - 2, -1, -1):
        lb[:, t] = logsumexp(logp_tr[None] + (lls[:, t + 1] + lb[:, t + 1])[:, None, :], axis=2)
    ll = logsumexp(la[:, -1], axis=1)
    lg = la + lb - ll[:, None, None]
    if T > 1:
        lx = la[:, :-1, :, None] + logp_tr[None, None] + (lls[:, 1:] + lb[:, 1:])[:, :, None, :] \
            - ll[:, None, None, None]
        xi = np.exp(logsumexp(lx.reshape(-1, K, K), axis=0))
    else:
        xi = np.zeros((K, K))
    return lg, xi, ll


def viterbi(logp_init, logp_tr, lls):
    N, T, K = lls.shape
    d = logp_init + lls[:, 0]
    bp = np.zeros((N, T, K), dtype=np.int64)
    for t in range(1, T):
        sc = d[:, :, None] + logp_tr[None]
        bp[:, t] = sc.argmax(axis=1)
        d = sc.max(axis=1) + lls[:, t]
    z = np.zeros((N, T), dtype=np.int64)
    z[:, -1] = d.argmax(axis=1)
    for t in range(T - 1, 0, -1):
        z[:, t - 1] = bp[np.arange(N), t, z[:, t]]
    return z


# ---------------------------------------------------------------------------
# HIAVI
# ---------------------------------------------------------------------------

class HIAVI:
    """
    Fit on (N, T) states s and actions a (a[n, t] taken in s[n, t]); T_env (S, A, S').

    hm = HIAVI(K, T_env, discount=0.9, seed=0); hm.fit(s, a)
    hm.loglik(s, a) -> mean log-likelihood per step; hm.decode(s, a) -> (N, T) MAP intentions.
    Attributes after fit: logp_init (K,), logp_tr (K, K), pis (K, S, A), r (K, S, A), q (K, S, A),
    ll_history.
    """

    def __init__(self, n_latents, T_env, discount=0.9, seed=0, mstep="soft", pseudocount=0.0, eps=EPS,
                 max_iters=200, tol=1e-6, sticky=0.95, iavi_tol=1e-3, iavi_max_iters=10000, verbose=False):
        self.K = int(n_latents)
        self.T_env = np.asarray(T_env, dtype=float)
        self.S, self.A, _ = self.T_env.shape
        self.discount = float(discount)
        self.rng = np.random.default_rng(seed)
        if mstep not in ("soft", "sample"):
            raise ValueError("mstep must be 'soft' or 'sample'")
        self.mstep, self.pseudocount, self.eps = mstep, float(pseudocount), float(eps)
        self.max_iters, self.tol, self.sticky = int(max_iters), float(tol), float(sticky)
        self.iavi_tol, self.iavi_max_iters, self.verbose = iavi_tol, iavi_max_iters, verbose

    # -- initialisation (as hiql: random policies, sticky Lambda, random pi0) --
    def _init(self):
        K, S, A = self.K, self.S, self.A
        p_init = np.abs(self.rng.standard_normal(K))
        p_init /= p_init.sum()
        p_tr = self.sticky * np.eye(K) + np.abs(self.rng.normal(0, 0.05, (K, K)))
        p_tr /= p_tr.sum(axis=1, keepdims=True)
        pis = np.abs(self.rng.standard_normal((K, S, A)))
        pis /= pis.sum(axis=-1, keepdims=True)
        return np.log(p_init), np.log(p_tr), np.log(pis)

    def _policies_from_counts(self, w, s, a):
        """w (N, T, K) responsibilities -> (K, S, A) normalised counts (+ pseudocount, eps rows as hiql)."""
        K, S, A = self.K, self.S, self.A
        cnt = np.zeros((K, S * A))
        idx = (s * A + a).ravel()
        for k in range(K):
            cnt[k] = np.bincount(idx, weights=w[..., k].ravel(), minlength=S * A)
        cnt = cnt.reshape(K, S, A) + self.pseudocount
        empty = cnt.sum(axis=-1) == 0
        cnt[empty] = self.eps
        pis = cnt / cnt.sum(axis=-1, keepdims=True)
        return pis

    def _logpis(self, pis):
        # softmax(q_IAVI) == (pi + eps) / sum(pi + eps): the policy IAVI reproduces
        p = pis + self.eps
        return np.log(p / p.sum(axis=-1, keepdims=True))

    def fit(self, s, a):
        s, a = np.asarray(s, dtype=np.int64), np.asarray(a, dtype=np.int64)
        if s.shape != a.shape or s.ndim != 2:
            raise ValueError(f"s and a must both be (N, T); got {s.shape}, {a.shape}")
        if s.min() < 0 or s.max() >= self.S or a.min() < 0 or a.max() >= self.A:
            raise ValueError("state / action index out of range")
        logp_init, logp_tr, logpis = self._init()
        prev_z, prev_ll, hist = None, -np.inf, []
        for it in range(self.max_iters):
            lg, xi, ll = forward_backward(logp_init, logp_tr, emission_loglik(logpis, s, a))
            tot = float(ll.sum() / s.size)
            if not np.isfinite(tot):
                raise FloatingPointError(f"HIAVI log-likelihood is {tot} at iteration {it}")
            hist.append(tot)
            z = lg.argmax(-1)
            if it > 0 and (abs(tot - prev_ll) < self.tol or (self.mstep == "sample" and np.array_equal(z, prev_z))):
                break
            prev_z, prev_ll = z, tot
            g = np.exp(lg)
            w = (self.rng.uniform(size=g.shape) <= g).astype(float) if self.mstep == "sample" else g
            logpis = self._logpis(self._policies_from_counts(w, s, a))
            logp_init = logsumexp(lg[:, 0], axis=0) - np.log(s.shape[0])
            logp_tr = np.log(np.maximum(xi, 1e-300))
            logp_tr -= logsumexp(logp_tr, axis=1, keepdims=True)
            if self.verbose:
                print(f"HIAVI iter {it}: LL/step {tot:.5f}")
        self.logp_init, self.logp_tr, self.ll_history = logp_init, logp_tr, np.asarray(hist)
        # final E-step responsibilities -> expert policies -> IAVI once per latent
        lg, _, _ = forward_backward(logp_init, logp_tr, emission_loglik(logpis, s, a))
        expert = self._policies_from_counts(np.exp(lg), s, a)
        self.r = np.zeros((self.K, self.S, self.A))
        self.q = np.zeros((self.K, self.S, self.A))
        for k in range(self.K):
            self.r[k], self.q[k] = iavi(self.T_env, expert[k], self.discount, self.iavi_tol, self.iavi_max_iters,
                                        self.eps)
        self.pis = softmax_policy(self.q)
        return self

    # -- evaluation with the IAVI policies --
    def _lls(self, s, a):
        return emission_loglik(np.log(self.pis), np.asarray(s, dtype=np.int64), np.asarray(a, dtype=np.int64))

    def loglik(self, s, a):
        """Mean forward log-likelihood per step (comparable to SWIRL's LL / (N * T))."""
        _, _, ll = forward_backward(self.logp_init, self.logp_tr, self._lls(s, a))
        return float(ll.sum() / np.asarray(s).size)

    def posteriors(self, s, a):
        lg, _, _ = forward_backward(self.logp_init, self.logp_tr, self._lls(s, a))
        return np.exp(lg)

    def decode(self, s, a):
        return viterbi(self.logp_init, self.logp_tr, self._lls(s, a))

    def params(self):
        return dict(logp_init=self.logp_init, logp_tr=self.logp_tr, pis=self.pis, r=self.r, q=self.q,
                    ll_history=self.ll_history)
