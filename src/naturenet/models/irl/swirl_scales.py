"""
SWIRL with decision-making at several time scales. Four independent toggles, all off by default (all off and
order = 1 is exactly SWIRL S-1, em_train_ext):

1. Order-N policy (action level).  pi_z(a | h_t), h_t = the longest suffix of (x_{t-N+1}, ..., x_t) that is in a
   context tree built from the data (suffix- and prefix-closed set of observed k-grams, k <= N, optional
   min_count pruning). Soft value iteration runs on the tree: taking a in context h and landing in s' moves to
   the longest in-tree suffix of (h, s') of length <= N, so the policy is the soft-optimal policy of the
   history-augmented MDP (S-2 is the order-2 case with every T-support pair in the tree).
   Memory: with state-only rewards, Q(h, a) = R(h) + g sum_s' T(s'|x_t, a) V(next(h, s')) and next() keeps
   only h[-(N-1):], so the ACTION CHOICE depends on the last N-1 states (x_{t-N+2..t}) and the reward on N-grams.
   Order 2 (= S-2) changes the reward to a transition reward R(x_{t-1}, x_t) but pi(a | x_{t-1}, x_t) =
   pi(a | x_t); order 3 is the first order whose action choice depends on the previous state.
       reward_param "additive": R_z(h) = sum_{k <= |h|} theta_z[h[-k:]]  (state reward + history increments,
                                 ridge penalty reward_l2 * (k - 1) * theta^2 / n_steps on depth k >= 2, i.e. a
                                 Gaussian prior in per-step units; increments start at 0, so
                                 the order-N model starts at the order-1 policy)
       reward_param "free":     R_z(h) = theta_z[h]  (S-2 layout; for equivalence tests)
2. Switching history.  The intention-transition MLP sees [onehot(x_{t-1}), phi_{t-1}], phi = features of the
   state history ("lags": one-hot x_{t-1..t-L}; "decay": exponentially weighted state occupancy;
   "summary": stay fraction, unique states, revisit flag over a window). New input weights start at 0.
3. Intention durations (explicit-duration / semi-Markov, in steps).  Latent (z, tau), tau = steps in z
   (bucket D absorbs >= D). Leave z with hazard h_{r,z}(tau [, phi]) ("logistic": logit h = a_{r,z} +
   b_z log tau [+ u_z . phi]; "nonparametric": a_{r,z,tau} with a random-walk penalty); on leaving,
   z' ~ Q_t(. | z) with the diagonal masked. b = 0 is the geometric (SWIRL) dwell time. With durations on,
   WHEN to switch lives in the hazard, so the history features enter the hazard (hazard_features default) and
   the regimes get their own hazard intercepts; with K = 2, Q is fixed (the other intention).
4. Hierarchical intentions.  Slow regime r with sticky P_R(r' | r); the intention transitions are
   regime-specific (log_Ps[r], and the hazard intercepts with durations on); the policies (rewards) and the
   kinematics are shared across regimes.

Latent chain (r, z, tau), step t-1 -> t:
    r_t ~ P_R(. | r_{t-1})
    stay   (prob 1 - h(r_{t-1}, z, tau, t-1)):   z_t = z_{t-1}, tau_t = min(tau + 1, D)
    switch (prob h(r_{t-1}, z, tau, t-1)):       z_t ~ Q_t(. | z_{t-1}, r_t, x_{t-1}, phi_{t-1}),  tau_t = 1
Durations off: D = 1, h = 1 and Q keeps its diagonal, i.e. the SWIRL recurrent transition.

Emissions: log pi_z(a_t | h_t) (+ per-intention kinematics, kinematics.KinematicEmissions). Multi-agent
partner coupling (swirl_ext ctx) is NOT combined with these toggles.

    tree = ContextTree.build(states, S, T, order=2)
    cfg = ScalesConfig.from_dict(dict(order=2, durations=dict(enabled=True, max_duration=20)))
    p = init_params(cfg, tree, (logpi0, log_Ps, Rs), R_start2, n_feat)
    r = em_train_scales(p, cfg, tree, data_tr, temps=None, iters=50)
    ev = scales_eval(r, cfg, tree, data_te)
"""
from dataclasses import dataclass, field, asdict
from functools import partial
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, vmap
from jax.nn import log_sigmoid
from jax.scipy.special import logsumexp as lse

from naturenet.models.irl.swirl_training import LOG_EPS, _minimize, sigmoid_jax
from naturenet.models.irl.swirl_training_top_level import _converged

jax.config.update("jax_enable_x64", True)

NEG = -1e30          # masked logits (finite, so 0 * NEG = 0 in the expected log-likelihoods)
DISCOUNT = 0.95
VI_ITERS = 100


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HistoryConfig:
    kind: tuple = ()              # subset of ("lags", "decay", "summary"); () = off
    lags: int = 2
    decay_steps: tuple = (3.0, 10.0)
    window: int = 10

    @property
    def enabled(self):
        return len(self.kind) > 0


@dataclass
class DurationConfig:
    enabled: bool = False
    hazard: str = "logistic"      # logistic | nonparametric
    max_duration: int = 20
    hazard_features: object = None  # hazard also sees the switching-history features (None: = history on)
    smooth: float = 1.0           # random-walk penalty (nonparametric)
    l2: float = 1.0               # ridge on b and u (per-step units, like reward_l2)


@dataclass
class HierarchyConfig:
    enabled: bool = False
    n_regimes: int = 2
    sticky: float = 0.95          # initial P_R diagonal
    prior: float = 0.0            # Dirichlet pseudo-count added to the P_R diagonal in the M-step
    init_noise: float = 0.5       # sd of the per-regime log_Ps perturbation (symmetry breaking)


@dataclass
class ScalesConfig:
    order: int = 1
    reward_param: str = "additive"   # additive | free
    reward_l2: float = 1.0           # prior precision / 2 per depth level (per-step units)
    context_min_count: int = 1
    context_source: str = "train"    # train | all
    context_t_support: bool = False  # add every T-support pair (S-2 layout) at depth 2
    reward_optimizer: str = "auto"   # auto | BFGS | L-BFGS-B
    # scipy tol for the reward M-step. The loss is per-step, so gradients on rarely visited deep contexts are
    # O(count / n_total) and the SWIRL default (1e-4) stops BFGS before they move. None -> 1e-4 at order 1
    # (bitwise SWIRL match), 1e-8 at order > 1.
    reward_tol: float = None
    history: HistoryConfig = field(default_factory=HistoryConfig)
    durations: DurationConfig = field(default_factory=DurationConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    seed: int = 0

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        h = d.pop("switch_history", d.pop("history", None))
        du = d.pop("durations", None)
        hi = d.pop("hierarchy", None)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.history = _history_conf(h)
        cfg.durations = _sub_conf(DurationConfig, du)
        cfg.hierarchy = _sub_conf(HierarchyConfig, hi)
        if cfg.durations.hazard_features is None:
            cfg.durations.hazard_features = cfg.history.enabled
        cfg.durations.hazard_features = bool(cfg.durations.hazard_features)
        cfg.order = int(cfg.order)
        if cfg.order < 1:
            raise ValueError("order must be >= 1")
        if cfg.reward_param not in ("additive", "free"):
            raise ValueError(f"reward_param {cfg.reward_param!r}")
        if cfg.durations.hazard not in ("logistic", "nonparametric"):
            raise ValueError(f"durations.hazard {cfg.durations.hazard!r}")
        return cfg

    @property
    def R(self):
        return int(self.hierarchy.n_regimes) if self.hierarchy.enabled else 1

    @property
    def D(self):
        return int(self.durations.max_duration) if self.durations.enabled else 1

    def to_json(self):
        return json.dumps(asdict(self), default=list)

    def tag(self):
        parts = [f"N{self.order}"]
        if self.reward_param != "additive":
            parts.append(self.reward_param)
        if self.history.enabled:
            parts.append("hist-" + "+".join(self.history.kind))
        if self.durations.enabled:
            parts.append(f"dur-{self.durations.hazard[:4]}{self.D}")
        if self.hierarchy.enabled:
            parts.append(f"R{self.R}")
        return "_".join(parts)


def _sub_conf(cls, d):
    if d is None or d is False:
        return cls()
    if d is True:
        return cls(enabled=True)
    d = dict(d)
    d.setdefault("enabled", True)
    return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _history_conf(h):
    if h in (None, False, "off", "none", ""):
        return HistoryConfig()
    if isinstance(h, str):
        return HistoryConfig(kind=tuple(h.split("+")))
    if isinstance(h, (list, tuple)):
        return HistoryConfig(kind=tuple(h))
    h = dict(h)
    if not h.pop("enabled", True):
        return HistoryConfig()
    kind = h.pop("kind", ("lags",))
    kind = tuple(kind.split("+")) if isinstance(kind, str) else tuple(kind)
    bad = set(kind) - {"lags", "decay", "summary"}
    if bad:
        raise ValueError(f"unknown switch_history kinds {bad}")
    if "decay_steps" in h:
        h["decay_steps"] = tuple(float(v) for v in np.atleast_1d(h["decay_steps"]))
    return HistoryConfig(kind=kind, **{k: v for k, v in h.items() if k in HistoryConfig.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Order-N context tree
# ---------------------------------------------------------------------------

def _support(T):
    """Per (s, a): next states and probs, padded to B = max branching. Returns nxt (S, A, B), prob (S, A, B)."""
    if hasattr(T, "coords"):
        c, v = np.asarray(T.coords), np.asarray(T.data)
        S, A = T.shape[0], T.shape[1]
    else:
        T = np.asarray(T)
        S, A = T.shape[0], T.shape[1]
        c = np.stack(np.nonzero(T))
        v = T[tuple(c)]
    lists = [[[] for _ in range(A)] for _ in range(S)]
    for s, a, s2, p in zip(c[0], c[1], c[2], v):
        if p > 0:
            lists[int(s)][int(a)].append((int(s2), float(p)))
    B = max(1, max(len(l) for row in lists for l in row))
    nxt = np.zeros((S, A, B), dtype=np.int64)
    prob = np.zeros((S, A, B))
    for s in range(S):
        for a in range(A):
            for b, (s2, p) in enumerate(lists[s][a]):
                nxt[s, a, b], prob[s, a, b] = s2, p
    return nxt, prob


class ContextTree:
    """
    Nodes: tuples of states (oldest ... newest), depth 1..order. Node ids 0..S-1 are the single states (so
    order = 1 is the standard state space); deeper nodes follow, sorted by depth.
    """

    def __init__(self, nodes, S, order, T):
        self.S, self.order = int(S), int(order)
        self.nodes = list(nodes)
        self.index = {c: i for i, c in enumerate(self.nodes)}
        self.H = len(self.nodes)
        self.depth = np.array([len(c) for c in self.nodes], dtype=np.int64)
        self.last = np.array([c[-1] for c in self.nodes], dtype=np.int64)
        snxt, sprob = _support(T)
        self.A, self.B = snxt.shape[1], snxt.shape[2]
        self.prob = sprob[self.last]                                     # (H, A, B)
        self.nxt = np.zeros((self.H, self.A, self.B), dtype=np.int64)    # (H, A, B) node ids
        for h, c in enumerate(self.nodes):
            keep = c[-(self.order - 1):] if self.order > 1 else ()
            for a in range(self.A):
                for b in range(self.B):
                    if sprob[c[-1], a, b] > 0:
                        self.nxt[h, a, b] = self.lookup(keep + (int(snxt[c[-1], a, b]),))
        anc = np.full((self.H, self.order), self.H, dtype=np.int64)     # suffix chain, pad = H (zero slot)
        for h, c in enumerate(self.nodes):
            for k in range(1, len(c) + 1):
                anc[h, k - 1] = self.index[c[-k:]]
        self.anc = anc

    @classmethod
    def build(cls, state_rows, S, T, order=1, min_count=1, extra_pairs=None):
        """state_rows: (N, L) int states (the full rows; x = rows[:, 1:-1]). extra_pairs: (prev, x) to add."""
        order = int(order)
        nodes = [(s,) for s in range(int(S))]
        if order > 1:
            rows = np.asarray(state_rows)
            counts = {}
            L = rows.shape[1]
            for k in range(2, order + 1):
                if L < k:
                    break
                win = np.lib.stride_tricks.sliding_window_view(rows, k, axis=1).reshape(-1, k)
                u, n = np.unique(win, axis=0, return_counts=True)
                for w, c in zip(map(tuple, u.tolist()), n):
                    # factor-closed: both (k-1)-factors must already be nodes (always true for min_count = 1)
                    if c >= min_count and (k == 2 or (w[:-1] in counts and w[1:] in counts)):
                        counts[w] = int(c)
            deep = set(counts)
            if extra_pairs is not None:
                deep |= {(int(p), int(x)) for p, x in extra_pairs}
            nodes += sorted(deep, key=lambda c: (len(c), c))
        return cls(nodes, S, order, T)

    def lookup(self, seq):
        for k in range(min(len(seq), self.order), 0, -1):
            i = self.index.get(tuple(seq[-k:]))
            if i is not None:
                return i
        raise KeyError(seq)

    def contexts(self, state_rows):
        """(N, T) node ids for x_t = rows[:, t + 1], from the history rows[:, max(0, t+2-order): t+2]."""
        rows = np.asarray(state_rows)
        N, L = rows.shape
        out = np.empty((N, L - 2), dtype=np.int64)
        if self.order == 1:
            return rows[:, 1:-1].astype(np.int64)
        for n in range(N):
            r = tuple(int(v) for v in rows[n])
            for t in range(L - 2):
                out[n, t] = self.lookup(r[max(0, t + 2 - self.order): t + 2])
        return out

    def depth_weights(self):
        """Ridge weight per node: 0 at depth 1, (k - 1) at depth k."""
        return (self.depth - 1).astype(float)

    def arrays(self):
        return dict(nxt=jnp.asarray(self.nxt), prob=jnp.asarray(self.prob), anc=jnp.asarray(self.anc))

    def to_save(self, prefix="ctx_"):
        L = self.order
        mat = np.full((self.H, L), -1, dtype=np.int64)
        for h, c in enumerate(self.nodes):
            mat[h, L - len(c):] = c
        return {prefix + "nodes": mat, prefix + "order": L, prefix + "S": self.S}

    @staticmethod
    def nodes_from_save(p, prefix="ctx_"):
        mat = np.asarray(p[prefix + "nodes"])
        return [tuple(int(v) for v in row if v >= 0) for row in mat]


# ---------------------------------------------------------------------------
# Switching-history features
# ---------------------------------------------------------------------------

def history_features(state_rows, S, hcfg):
    """(N, T, F) features of the history through x_t = rows[:, t+1] (the transition into t+1 uses row t)."""
    rows = np.asarray(state_rows, dtype=np.int64)
    N, L = rows.shape
    T = L - 2
    blocks, names = [], []
    if not hcfg.enabled:
        return np.zeros((N, T, 0)), names
    eye = np.eye(S)
    pos = np.arange(1, L - 1)                                         # row position of x_t
    if "lags" in hcfg.kind:
        for l in range(1, int(hcfg.lags) + 1):
            p = pos - l
            f = np.where((p >= 0)[None, :, None], eye[rows[:, np.maximum(p, 0)]], 0.0)
            blocks.append(f)
            names += [f"lag{l}_s{s}" for s in range(S)]
    if "decay" in hcfg.kind:
        oh = eye[rows]                                                # (N, L, S)
        for tau in hcfg.decay_steps:
            rho = 1.0 / max(float(tau), 1.0)
            o = np.zeros((N, S))
            occ = np.zeros((N, L, S))
            for p in range(L):
                o = (1 - rho) * o + rho * oh[:, p]
                occ[:, p] = o
            blocks.append(occ[:, 1:-1])
            names += [f"decay{tau:g}_s{s}" for s in range(S)]
    if "summary" in hcfg.kind:
        W = int(hcfg.window)
        f = np.zeros((N, T, 3))
        for t, p in enumerate(pos):
            lo = max(0, p - W + 1)
            win = rows[:, lo:p + 1]
            if win.shape[1] > 1:
                f[:, t, 0] = (win[:, 1:] == win[:, :-1]).mean(1)
            srt = np.sort(win, axis=1)
            f[:, t, 1] = (1 + (srt[:, 1:] != srt[:, :-1]).sum(1)) / W
            if p - 2 >= lo:
                f[:, t, 2] = (rows[:, lo:p - 1] == rows[:, p][:, None]).any(1)
        blocks.append(f)
        names += ["stay_frac", "n_unique", "revisit"]
    return np.concatenate(blocks, axis=-1), names


# ---------------------------------------------------------------------------
# Rewards and policy on the tree
# ---------------------------------------------------------------------------

def node_rewards(theta, anc, reward_param):
    """theta (K, H) -> R (K, H)."""
    if reward_param == "free":
        return theta
    th = jnp.concatenate([theta, jnp.zeros((theta.shape[0], 1))], axis=1)
    return th[:, anc].sum(-1)


def _vi(R_h, nxt, prob, temp, use_temp):
    """Soft VI on the tree (same init / iterations / formulas as jax_soft_find_policy[_temp]). -> pi (H, A)."""
    H, A, _ = nxt.shape
    r_sa = R_h[:, None] * jnp.ones((1, A))

    def it(carry, _):
        v, _q = carry
        q = r_sa + DISCOUNT * jnp.sum(prob * v[nxt], axis=-1)
        if use_temp:
            v = temp * lse(q / temp, axis=1)
        else:
            v = lse(q, axis=1)
        return (v, q), None

    (v, q), _ = lax.scan(it, (jnp.ones(H), jnp.ones((H, A))), None, length=VI_ITERS)
    if use_temp:
        pi = jnp.exp((q - v[:, None]) / temp) + 1e-8
    else:
        q = q - q.max(axis=1, keepdims=True)
        pi = jnp.exp(q)
    return pi / pi.sum(axis=1, keepdims=True)


@partial(jax.jit, static_argnames=("reward_param", "use_temp"))
def log_policy(theta, nxt, prob, anc, temps, reward_param, use_temp):
    """(K, H, A) log pi."""
    R = node_rewards(theta, anc, reward_param)
    pi = vmap(lambda r, t: _vi(r, nxt, prob, t, use_temp))(R, temps)
    return jnp.log(jnp.clip(pi, LOG_EPS))


def emission_ll(logpi, ctx, a):
    """(N, T, K) from logpi (K, H, A), ctx (N, T), a (N, T)."""
    return jnp.moveaxis(logpi[:, ctx, a], 0, -1)


# ---------------------------------------------------------------------------
# Transitions and hazards
# ---------------------------------------------------------------------------

def log_Q(log_Ps, Rs, W1f, x, phi, mask_diag):
    """(N, T-1, R, K, K) log Q_t(z' | z, r): log_Ps (R, K, K) + MLP([onehot(x_{t-1}), phi_{t-1}])."""
    W1, b1, W2, b2 = Rs
    Z1 = W1[x[:, :-1]] + b1
    if W1f.shape[0] > 0:
        Z1 = Z1 + phi[:, :-1] @ W1f
    Z2 = sigmoid_jax(Z1) @ W2 + b2                                    # (N, T-1, K)
    logits = log_Ps[None, None] + Z2[:, :, None, None, :]
    if mask_diag:
        logits = logits + NEG * jnp.eye(log_Ps.shape[-1])
    return logits - lse(logits, axis=-1, keepdims=True)


def hazard_logits(hz, D, phi, kind, use_feat):
    """Hazard logits: (R, K, D), or (N, T-1, R, K, D) with features."""
    tau = jnp.log(jnp.arange(1, D + 1, dtype=jnp.float64))
    base = hz["a"][..., None] + hz["b"][None, :, None] * tau if kind == "logistic" else hz["a"]
    if not use_feat:
        return base
    return base[None, None] + jnp.einsum("ntf,kf->ntk", phi[:, :-1], hz["u"])[:, :, None, :, None]


def log_hazards(cfg, hz, phi):
    """(log h, log (1 - h)) with shape (R, K, D) or (N, T-1, R, K, D)."""
    if not cfg.durations.enabled:
        return None, None
    lg = hazard_logits(hz, cfg.D, phi, cfg.durations.hazard, cfg.durations.hazard_features)
    return log_sigmoid(lg), log_sigmoid(-lg)


# ---------------------------------------------------------------------------
# Forward / backward on (r, z, tau)
# ---------------------------------------------------------------------------

def _shift(B):
    """Bn[..., i] = B[..., min(i + 1, D - 1)]."""
    return jnp.concatenate([B[..., 1:], B[..., -1:]], axis=-1)


def _seq_fb(logpi0, logPR, ll, logQ, lh, l1h, want_kd, ht):
    """
    One sequence. logpi0 (R, K), logPR (R, R), ll (T, K), logQ (T-1, R, K, K), lh / l1h (R, K, D) or
    (T-1, R, K, D) (hazard of the PREVIOUS regime). Returns logZ, cum (T,), g_z (T, K), g_r (T, R), g0 (R, K),
    EQ (T-1, R, K, K), S0 / S1 (stay / switch mass, (T-1, R, K, D) if ht else (R, K, D)), ER (R, R),
    g_kd ((T, K, D) or None).
    """
    R, K = logpi0.shape
    D = lh.shape[-1]
    la0 = jnp.full((R, K, D), -jnp.inf).at[:, :, 0].set(logpi0 + ll[0][None])

    def fwd(la, inp):
        lq, l_t, h, h1 = inp
        u = la + h1
        st = jnp.concatenate([jnp.full((R, K, 1), -jnp.inf), u[..., :-1]], axis=-1)
        st = st.at[..., -1].set(jnp.logaddexp(st[..., -1], u[..., -1])) if D > 1 else u
        stay = lse(logPR[:, :, None, None] + st[:, None], axis=0)
        v = lse(la + h, axis=-1)                                      # (R, K)
        w = lse(logPR[:, :, None] + v[:, None, :], axis=0)            # (R', K)
        sw = lse(w[:, :, None] + lq, axis=1)                          # (R', K')
        new = stay.at[..., 0].set(jnp.logaddexp(stay[..., 0], sw))
        new = new + l_t[None, :, None]
        return new, new

    T = ll.shape[0]
    hs = (lh, l1h) if ht else (jnp.broadcast_to(lh, (T - 1,) + lh.shape), jnp.broadcast_to(l1h, (T - 1,) + l1h.shape))
    _, las = lax.scan(fwd, la0, (logQ, ll[1:], hs[0], hs[1]))
    la = jnp.concatenate([la0[None], las], axis=0)                    # (T, R, K, D)
    logZ = lse(la[-1])
    cum = lse(la.reshape(T, -1), axis=-1)

    def bwd(lb, inp):
        la_p, lq, l_t, h, h1 = inp
        Bt = l_t[None, :, None] + lb                                  # (R', K, D)
        Bn = _shift(Bt)
        ms = lse(logPR[:, :, None, None] + Bn[None], axis=1)          # (R, K, D)
        stay = h1 + ms
        G = lse(lq + Bt[:, None, :, 0], axis=-1)                      # (R', K)
        swr = lse(logPR[:, :, None] + G[None], axis=1)                # (R, K)
        lb_p = jnp.logaddexp(stay, h + swr[:, :, None])
        # pairwise statistics for step t-1 -> t
        S0 = jnp.exp(la_p + h1 + ms - logZ)                           # (R, K, D)
        F = la_p + h
        S1 = jnp.exp(F + swr[:, :, None] - logZ)                      # (R, K, D)
        Fz = lse(F, axis=-1)                                          # (R, K)
        Wr = lse(Fz[:, None, :] + logPR[:, :, None], axis=0)          # (R', K)
        EQ = jnp.exp(Wr[:, :, None] + lq + Bt[:, None, :, 0] - logZ)  # (R', K, K')
        ERs = lse((la_p + h1)[:, None] + Bn[None], axis=(2, 3))       # (R, R')
        ERw = lse(Fz[:, None, :] + G[None], axis=-1)                  # (R, R')
        ER = jnp.exp(logPR + jnp.logaddexp(ERs, ERw) - logZ)
        g = la_p + lb_p - logZ
        return lb_p, (g, EQ, S0, S1, ER)

    lbT = jnp.zeros((R, K, D))
    _, (gs, EQ, S0, S1, ER) = lax.scan(bwd, lbT, (la[:-1], logQ, ll[1:], hs[0], hs[1]), reverse=True)
    g = jnp.exp(jnp.concatenate([gs, (la[-1] - logZ)[None]], axis=0))  # (T, R, K, D)
    out = dict(logZ=logZ, cum=cum, g_z=g.sum((1, 3)), g_r=g.sum((2, 3)), g0=g[0, :, :, 0], EQ=EQ,
               S0=S0 if ht else S0.sum(0), S1=S1 if ht else S1.sum(0), ER=ER.sum(0))
    out["g_kd"] = g.sum(1) if want_kd else None
    return out


@partial(jax.jit, static_argnames=("want_kd", "ht"))
def _fb_batch(logpi0, logPR, ll, logQ, lh, l1h, want_kd, ht):
    hax = 0 if ht else None
    f = vmap(lambda l, q, h, h1: _seq_fb(logpi0, logPR, l, q, h, h1, want_kd, ht), in_axes=(0, 0, hax, hax))
    return f(ll, logQ, lh, l1h)


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

def init_params(cfg, tree, base, R_start, n_feat, seed=None):
    """
    base = (logpi0 (K,), log_Ps (K, K), Rs = (W1, b1, W2, b2)) from the ARHMM (or sticky_init); R_start
    (K, S) or (K, 1, S) state rewards. With every toggle off and order 1 the model equals this SWIRL init.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    logpi0, log_Ps, Rs = (np.asarray(base[0], float), np.asarray(base[1], float),
                          tuple(np.asarray(w, float) for w in base[2]))
    K, S = log_Ps.shape[0], tree.S
    R1 = np.asarray(R_start, float).reshape(K, -1)
    if R1.shape[1] != S:
        raise ValueError(f"R_start has {R1.shape[1]} states, the tree {S}")
    theta = np.zeros((K, tree.H))
    theta[:, :S] = R1
    if cfg.reward_param == "free":
        theta = R1[:, tree.last]
    R, D = cfg.R, cfg.D
    lP = np.repeat(log_Ps[None], R, axis=0)
    for r in range(1, R):
        lP[r] += rng.normal(scale=cfg.hierarchy.init_noise, size=(K, K))
    lp0 = logpi0 - np.logaddexp.reduce(logpi0)
    p = dict(theta=theta, logpi0=np.repeat(lp0[None], R, 0) - np.log(R), log_Ps=lP, Rs=Rs,
             W1f=np.zeros((n_feat, Rs[0].shape[1])))
    if R > 1:
        s = cfg.hierarchy.sticky
        P = np.full((R, R), (1 - s) / (R - 1))
        np.fill_diagonal(P, s)
        p["logPR"] = np.log(P)
    else:
        p["logPR"] = np.zeros((1, 1))
    if cfg.durations.enabled:
        if K < 2:
            raise ValueError("durations need n_hidden >= 2 (the switch target excludes the current intention)")
        P = np.exp(log_Ps - np.logaddexp.reduce(log_Ps, axis=1, keepdims=True))
        hz0 = np.clip(1 - np.diag(P), 0.01, 0.99)
        a = np.repeat((np.log(hz0) - np.log1p(-hz0))[None], R, 0)    # (R, K)
        for r in range(1, R):
            a[r] += rng.normal(scale=cfg.hierarchy.init_noise, size=K)
        nf = n_feat if cfg.durations.hazard_features else 0
        if cfg.durations.hazard == "logistic":
            p["hz"] = dict(a=a, b=np.zeros(K), u=np.zeros((K, nf)))
        else:
            p["hz"] = dict(a=np.repeat(a[..., None], D, -1), u=np.zeros((K, nf)))
    return p


def _jnp_tree(p):
    return jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=jnp.float64), p)


def n_params(p, cfg, tree):
    n = p["theta"].size + p["logpi0"].size - 1 + p["log_Ps"].size + sum(np.size(w) for w in p["Rs"]) + \
        np.size(p["W1f"])
    if cfg.R > 1:
        n += cfg.R * (cfg.R - 1)
    if "hz" in p:
        n += sum(np.size(v) for v in p["hz"].values())
    return int(n)


# ---------------------------------------------------------------------------
# Data bundle
# ---------------------------------------------------------------------------

def scales_data(state_rows, a, tree, hcfg, S, kin=None, idx=None):
    """dict(x, ctx, a, phi, kin) for the rows idx (x = rows[:, 1:-1], a aligned with x)."""
    rows = np.asarray(state_rows)
    a = np.asarray(a)
    if idx is not None:
        rows, a = rows[idx], a[idx]
        kin = None if kin is None else kin[idx]
    if a.shape != (rows.shape[0], rows.shape[1] - 2):
        raise ValueError(f"actions {a.shape} do not match x = rows[:, 1:-1] {rows[:, 1:-1].shape}")
    phi, names = history_features(rows, S, hcfg)
    ctx = tree.contexts(rows)
    assert ctx.shape == a.shape and phi.shape[:2] == a.shape
    return dict(x=jnp.asarray(rows[:, 1:-1].astype(np.int64)), ctx=jnp.asarray(ctx), a=jnp.asarray(a, dtype=jnp.int64),
                phi=jnp.asarray(phi), phi_names=names, kin=kin, rows=rows)


# ---------------------------------------------------------------------------
# E-step
# ---------------------------------------------------------------------------

def _temps_arr(temps, K):
    return (jnp.ones(K), False) if temps is None else (jnp.asarray(temps, dtype=jnp.float64), True)


def e_step(p, cfg, tree_arr, d, temps, kin_model=None, want_kd=False, use_kin=True):
    K = p["theta"].shape[0]
    tmp, use_temp = _temps_arr(temps, K)
    logpi = log_policy(p["theta"], tree_arr["nxt"], tree_arr["prob"], tree_arr["anc"], tmp, cfg.reward_param,
                       use_temp)
    ll = emission_ll(logpi, d["ctx"], d["a"])
    if use_kin and kin_model is not None and d["kin"] is not None:
        ll = ll + jnp.asarray(kin_model.loglik(d["kin"]))
    lq = log_Q(p["log_Ps"], p["Rs"], p["W1f"], d["x"], d["phi"], cfg.durations.enabled)
    if cfg.durations.enabled:
        lh, l1h = log_hazards(cfg, p["hz"], d["phi"])
    else:
        R = p["logPR"].shape[0]
        lh, l1h = jnp.zeros((R, K, 1)), jnp.full((R, K, 1), -jnp.inf)
    ht = lh.ndim == 5
    lpi0 = p["logpi0"] - lse(p["logpi0"])
    lPR = p["logPR"] - lse(p["logPR"], axis=1, keepdims=True)
    return _fb_batch(lpi0, lPR, ll, lq, lh, l1h, want_kd, ht)


# ---------------------------------------------------------------------------
# M-steps
# ---------------------------------------------------------------------------

def _flat(parts):
    return jnp.concatenate([jnp.ravel(v) for v in parts])


def _unflat(flat, shapes):
    out, i = [], 0
    for s in shapes:
        n = int(np.prod(s))
        out.append(flat[i:i + n].reshape(s))
        i += n
    return out


def _q_loss(flat, shapes, EQ, x, phi, n_total, mask):
    lP, W1, b1, W2, b2, W1f = _unflat(flat, shapes)
    lq = log_Q(lP, (W1, b1, W2, b2), W1f, x, phi, mask)
    return -jnp.sum(EQ * lq) / n_total


_q_vg = jax.jit(jax.value_and_grad(_q_loss), static_argnames=("shapes", "mask"))


def q_m_step(p, cfg, st, d, n_total, num_iters=1000):
    parts = [p["log_Ps"], *p["Rs"], p["W1f"]]
    shapes = tuple(tuple(int(s) for s in np.shape(v)) for v in parts)
    res = _minimize(_q_vg, _flat([jnp.asarray(v) for v in parts]), "L-BFGS-B", num_iters, shapes=shapes,
                    EQ=st["EQ"], x=d["x"], phi=d["phi"], n_total=n_total, mask=cfg.durations.enabled)
    lP, W1, b1, W2, b2, W1f = [np.asarray(v) for v in _unflat(jnp.asarray(res.x), shapes)]
    return lP, (W1, b1, W2, b2), W1f


def _hz_loss(flat, shapes, keys, S0, S1, phi, n_total, D, kind, use_feat, smooth, l2):
    hz = dict(zip(keys, _unflat(flat, shapes)))
    lg = hazard_logits(hz, D, phi, kind, use_feat)
    nll = -jnp.sum(S1 * log_sigmoid(lg) + S0 * log_sigmoid(-lg)) / n_total
    pen = l2 * jnp.sum(hz["u"] ** 2)
    if kind == "logistic":
        pen = pen + l2 * jnp.sum(hz["b"] ** 2)
    else:
        pen = pen + smooth * jnp.sum(jnp.diff(hz["a"], axis=-1) ** 2)
    return nll + pen / n_total


_hz_vg = jax.jit(jax.value_and_grad(_hz_loss), static_argnames=("shapes", "keys", "D", "kind", "use_feat"))


def hazard_m_step(p, cfg, st, d, n_total, num_iters=1000):
    keys = tuple(sorted(p["hz"]))
    parts = [jnp.asarray(p["hz"][k]) for k in keys]
    shapes = tuple(tuple(int(s) for s in v.shape) for v in parts)
    dc = cfg.durations
    res = _minimize(_hz_vg, _flat(parts), "L-BFGS-B", num_iters, shapes=shapes, keys=keys, S0=st["S0"],
                    S1=st["S1"], phi=d["phi"], n_total=n_total, D=cfg.D, kind=dc.hazard,
                    use_feat=dc.hazard_features, smooth=float(dc.smooth), l2=float(dc.l2))
    return {k: np.asarray(v) for k, v in zip(keys, _unflat(jnp.asarray(res.x), shapes))}


def _r_loss(flat, shape, nxt, prob, anc, temps, g, ctx, a, n_total, reward_param, use_temp, pen_w, l2):
    theta = flat.reshape(shape)
    logpi = log_policy(theta, nxt, prob, anc, temps, reward_param, use_temp)
    ll = emission_ll(logpi, ctx, a)
    return (-jnp.sum(g * ll) + l2 * jnp.sum(pen_w[None] * theta ** 2)) / n_total


_r_vg = jax.jit(jax.value_and_grad(_r_loss), static_argnames=("shape", "reward_param", "use_temp"))


def _minimize_tol(vg, x0, method, num_iters, tol, **kw):
    """_minimize with an explicit scipy tol (see ScalesConfig.reward_tol)."""
    from scipy.optimize import minimize

    def f(x):
        v, g = vg(jnp.asarray(x), **kw)
        return float(v), np.asarray(g, dtype=np.float64)

    return minimize(f, np.asarray(x0, dtype=np.float64), jac=True, method=method,
                    options=dict(maxiter=num_iters), tol=float(tol))


def reward_m_step(p, cfg, tree_arr, pen_w, st, d, temps, n_total, num_iters=1000):
    K = p["theta"].shape[0]
    tmp, use_temp = _temps_arr(temps, K)
    theta = jnp.asarray(p["theta"])
    method = cfg.reward_optimizer
    if method == "auto":
        method = "BFGS" if theta.size <= 3000 else "L-BFGS-B"
    l2 = float(cfg.reward_l2) if cfg.reward_param == "additive" else 0.0
    tol = cfg.reward_tol if cfg.reward_tol is not None else (1e-4 if cfg.order == 1 else 1e-8)
    res = _minimize_tol(_r_vg, theta.ravel(), method, num_iters, tol, shape=tuple(theta.shape), nxt=tree_arr["nxt"],
                    prob=tree_arr["prob"], anc=tree_arr["anc"], temps=tmp, g=st["g_z"], ctx=d["ctx"], a=d["a"],
                    n_total=n_total, reward_param=cfg.reward_param, use_temp=use_temp, pen_w=jnp.asarray(pen_w),
                    l2=l2)
    return np.asarray(res.x).reshape(theta.shape)


def regime_m_step(cfg, st):
    ER = np.asarray(st["ER"]).sum(0) + 1e-8
    ER = ER + cfg.hierarchy.prior * np.eye(ER.shape[0])
    return np.log(ER / ER.sum(1, keepdims=True))


def pi0_m_step(st):
    g0 = np.asarray(st["g0"]).sum(0) + 1e-8
    return np.log(g0 / g0.sum())


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------

def em_train_scales(p, cfg, tree, d, temps=None, kin_model=None, iters=50, tol=None, init=True, trans=True,
                    emit=True, kin_mstep=True, num_iters=1000, verbose=False):
    """
    EM on the training bundle d (scales_data). Same order as em_train_ext: E-step (LL recorded), convergence
    check, then pi0 / regimes / Q / hazard / reward / kinematics M-steps.
    Returns dict(params, kin_model, temps, LL_list, cfg).
    """
    p = _jnp_tree({k: v for k, v in p.items()})
    tree_arr = tree.arrays()
    pen_w = tree.depth_weights()
    kin_model = None if kin_model is None else kin_model.copy()
    N, T = d["a"].shape
    n_total = float(N * T)
    LL_list = []
    for i in range(int(iters)):
        st = e_step(p, cfg, tree_arr, d, temps, kin_model)
        LL_list.append(float(jnp.sum(st["logZ"])))
        if not np.isfinite(LL_list[-1]):
            raise FloatingPointError(f"scales EM LL is {LL_list[-1]} at iteration {i}")
        if verbose:
            print("EM iter", i, "LL", LL_list[-1])
        if _converged(LL_list, tol):
            break
        new = dict(p)
        if init:
            new["logpi0"] = jnp.asarray(pi0_m_step(st))
        if cfg.R > 1 and trans:
            new["logPR"] = jnp.asarray(regime_m_step(cfg, st))
        if trans:
            lP, Rs, W1f = q_m_step(p, cfg, st, d, n_total, num_iters)
            new["log_Ps"], new["Rs"], new["W1f"] = jnp.asarray(lP), tuple(jnp.asarray(w) for w in Rs), jnp.asarray(W1f)
            if cfg.durations.enabled:
                new["hz"] = _jnp_tree(hazard_m_step(p, cfg, st, d, n_total, num_iters))
        if emit:
            new["theta"] = jnp.asarray(reward_m_step(p, cfg, tree_arr, pen_w, st, d, temps, n_total, num_iters))
        if kin_model is not None and d["kin"] is not None and kin_mstep:
            kin_model.m_step(np.asarray(st["g_z"]), d["kin"])
        p = new
    return dict(params=jax.tree_util.tree_map(np.asarray, p), kin_model=kin_model,
                temps=None if temps is None else np.asarray(temps), LL_list=np.asarray(LL_list), cfg=cfg)


def scales_eval(r, tree, d, use_kin=True, want_kd=False):
    """LL_total, LL per step, step_ll (N, T), gamma (N, T, K), gamma_r (N, T, R), path (posterior argmax z)."""
    cfg = r["cfg"]
    p = _jnp_tree(r["params"])
    st = e_step(p, cfg, tree.arrays(), d, r.get("temps"), r.get("kin_model"), want_kd=want_kd, use_kin=use_kin)
    cum = np.asarray(st["cum"])
    N, T = cum.shape
    g = np.asarray(st["g_z"])
    out = dict(LL_total=float(cum[:, -1].sum()), LL=float(cum[:, -1].sum() / (N * T)),
               step_ll=np.diff(cum, axis=1, prepend=0.0), gamma=g, gamma_r=np.asarray(st["g_r"]),
               path=g.argmax(-1), n_steps=int(N * T))
    if want_kd:
        out["gamma_zt"] = np.asarray(st["g_kd"])                      # (N, T, K, D)
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def policy(r, tree):
    cfg, p = r["cfg"], r["params"]
    K = p["theta"].shape[0]
    tmp, use_temp = _temps_arr(r.get("temps"), K)
    ta = tree.arrays()
    return np.exp(np.asarray(log_policy(jnp.asarray(p["theta"]), ta["nxt"], ta["prob"], ta["anc"], tmp,
                                        cfg.reward_param, use_temp)))


def history_kl(r, tree, ctx_counts=None):
    """(K, order) mean KL(pi(.|h) || pi(.|last state of h)) by context depth (weighted by ctx_counts)."""
    pi = policy(r, tree)
    lp = np.log(np.clip(pi, LOG_EPS, None))
    kl = (pi * (lp - lp[:, tree.last])).sum(-1)                        # (K, H)
    w = np.ones(tree.H) if ctx_counts is None else np.asarray(ctx_counts, float)
    out = np.full((pi.shape[0], tree.order), np.nan)
    for k in range(1, tree.order + 1):
        m = (tree.depth == k) & (w > 0)
        if m.any():
            out[:, k - 1] = (kl[:, m] * w[m]).sum(1) / w[m].sum()
    return out


def reward_depth_share(r, tree, ctx_counts=None):
    """(K, order) share of the (count-weighted) reward variance carried by the depth-k increments (additive)."""
    th = np.asarray(r["params"]["theta"])
    w = np.ones(tree.H) if ctx_counts is None else np.asarray(ctx_counts, float)
    K = th.shape[0]
    out = np.zeros((K, tree.order))
    used = w > 0
    if r["cfg"].reward_param != "additive":
        return out * np.nan
    for k in range(1, tree.order + 1):
        a = tree.anc[used, k - 1]
        val = np.concatenate([th, np.zeros((K, 1))], 1)[:, a]
        mu = (val * w[used]).sum(1, keepdims=True) / w[used].sum()
        out[:, k - 1] = ((val - mu) ** 2 * w[used]).sum(1) / w[used].sum()
    s = out.sum(1, keepdims=True)
    return out / np.where(s > 0, s, 1)


def hazard_curves(r, phi_mean=None):
    """(R, K, D) hazard h_{r,z}(tau) (features at phi_mean, default 0) and the implied dwell pmf (R, K, D)."""
    cfg = r["cfg"]
    if not cfg.durations.enabled:
        return None, None
    hz = _jnp_tree(r["params"]["hz"])
    lg = np.asarray(hazard_logits(hz, cfg.D, None, cfg.durations.hazard, False))
    if cfg.durations.hazard_features and phi_mean is not None:
        lg = lg + (np.asarray(hz["u"]) @ np.asarray(phi_mean))[None, :, None]
    h = 1 / (1 + np.exp(-lg))
    surv = np.cumprod(np.concatenate([np.ones(h.shape[:-1] + (1,)), 1 - h[..., :-1]], -1), -1)
    return h, surv * h


def dwell_times(path):
    """Completed run lengths per intention (runs touching either end of a sequence are dropped)."""
    out = {}
    for row in np.asarray(path):
        ch = np.flatnonzero(np.diff(row)) + 1
        b = np.concatenate([[0], ch, [len(row)]])
        for i in range(1, len(b) - 2):
            out.setdefault(int(row[b[i]]), []).append(int(b[i + 1] - b[i]))
    return out


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def params_to_save(r, tree, prefix="sc_"):
    p = r["params"]
    out = {prefix + "theta": p["theta"], prefix + "logpi0": p["logpi0"], prefix + "logPR": p["logPR"],
           prefix + "log_Ps": p["log_Ps"], prefix + "W1f": p["W1f"], prefix + "cfg": r["cfg"].to_json(),
           "LL_list": np.asarray(r["LL_list"])}
    for i, w in enumerate(p["Rs"]):
        out[prefix + f"Rs{i}"] = np.asarray(w)
    for k, v in p.get("hz", {}).items():
        out[prefix + "hz_" + k] = np.asarray(v)
    if r.get("temps") is not None:
        out["temps"] = np.asarray(r["temps"])
    if r.get("kin_model") is not None:
        out.update(r["kin_model"].params())
    out.update(tree.to_save())
    # reward on the depth-1 nodes in the SWIRL (K, 1, S) layout, for the existing plots
    R = np.asarray(node_rewards(jnp.asarray(p["theta"]), jnp.asarray(tree.anc), r["cfg"].reward_param))
    out["new_reward"] = R[:, None, :tree.S]
    return out


def cfg_from_json(s):
    c = json.loads(str(s))
    cfg = ScalesConfig.from_dict({k: v for k, v in c.items() if k not in ("history", "durations", "hierarchy")})
    cfg.history = HistoryConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in c["history"].items()})
    cfg.durations = DurationConfig(**c["durations"])
    cfg.hierarchy = HierarchyConfig(**c["hierarchy"])
    return cfg


def params_from_npz(z, prefix="sc_"):
    p = dict(theta=z[prefix + "theta"], logpi0=z[prefix + "logpi0"], logPR=z[prefix + "logPR"],
             log_Ps=z[prefix + "log_Ps"], W1f=z[prefix + "W1f"],
             Rs=tuple(z[prefix + f"Rs{i}"] for i in range(4)))
    hz = {k[len(prefix) + 3:]: z[k] for k in z.files if k.startswith(prefix + "hz_")}
    if hz:
        p["hz"] = hz
    return p


def load_scales(path, T, prefix="sc_"):
    """-> (r, tree) from a params_to_save npz (T: the fold transition tensor used for the fit)."""
    from naturenet.models.irl.kinematics import KinematicEmissions

    z = np.load(path, allow_pickle=True)
    cfg = cfg_from_json(z[prefix + "cfg"])
    tree = ContextTree(ContextTree.nodes_from_save(z), int(z["ctx_S"]), int(z["ctx_order"]), T)
    p = params_from_npz(z, prefix)
    km = KinematicEmissions.from_params(z) if KinematicEmissions.has_params(z) else None
    return dict(params=p, cfg=cfg, temps=z["temps"] if "temps" in z.files else None, kin_model=km,
                LL_list=z["LL_list"]), tree


def warm_start(p_new, tree_new, r_old, tree_old):
    """Copy a fitted model into a new init: theta by context (shared nodes), and every compatible block."""
    po = r_old["params"]
    out = dict(p_new)
    th = np.array(p_new["theta"], float)
    if r_old["cfg"].reward_param == "additive":
        for h, c in enumerate(tree_old.nodes):
            j = tree_new.index.get(c)
            if j is not None:
                th[:, j] = po["theta"][:, h]
    out["theta"] = th
    for k in ("logpi0", "log_Ps", "logPR", "W1f"):
        if np.shape(po[k]) == np.shape(p_new[k]):
            out[k] = np.asarray(po[k])
    if all(np.shape(a) == np.shape(b) for a, b in zip(po["Rs"], p_new["Rs"])):
        out["Rs"] = tuple(np.asarray(w) for w in po["Rs"])
    if "hz" in p_new and "hz" in po and all(np.shape(po["hz"].get(k)) == np.shape(v) for k, v in p_new["hz"].items()):
        out["hz"] = {k: np.asarray(v) for k, v in po["hz"].items()}
    return out
