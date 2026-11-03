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

Emissions: log pi_z(a_t | h_t) (+ per-intention kinematics, kinematics.KinematicEmissions).
    emission "swirl" (default): pi_z = soft-optimal policy of the reward theta on the context tree (above).
    emission "hiql": HIQL / HIAVI policies, pi_z(a | h) = posterior-weighted (h, a) counts (closed-form M-step,
        hiavi_model.HIAVI at order 1 with every toggle off); deeper contexts back off to their parent context,
        pi(.|h) = (n(h, .) + kappa pi(.|parent(h))) / (n(h) + kappa), kappa = hiql.backoff. The policy is tabular
        in the full context, so at order N the ACTION depends on all N states (order 2 already has one step of
        action memory, unlike the state-reward SWIRL). Rewards / Q are recovered after fitting by IAVI on the
        context MDP (iavi_tree), as HIAVI does once per intention.
    transition_input "state" (SWIRL MLP on onehot(x_{t-1}) [+ phi]) | "none" (HIAVI Lambda; with switching
        history the features enter linearly, logits += phi_{t-1} @ W1f). Default: "state" for swirl, "none" for
        hiql. With no inputs and no coupling the Q M-step is closed form (normalised expected counts).

5. Multi-agent coupling (mean-field, as hiavi_multiagent / swirl_ext). Partner context (w_t, c_t) of the
   destination step (PartnerContext.context of the other animals' posteriors), d["w"] (N, T), d["c"] (N, T, K):
   destination   Q~_t(k | j) = (1 - alpha w_t) Q_t(k | j) + alpha w_t c~_t[k]   (c~ = c; with durations c~ drops
                 the current intention and renormalises, so with K = 2 and durations on alpha has no effect
                 and is fixed at 0)
   timing        logit h_{r,z}(tau, t) += beta * w_t * (1 - c_t[z])   (durations on; beta > 0: a bout ends
                 sooner when the partners are in another intention, beta < 0: partners hold the bout)
   alpha = ALPHA_MAX sigmoid(cpl_alpha) is fitted jointly with Q (L-BFGS), beta jointly with the hazard. alpha =
   beta = 0, or w = 0 (no partner tracked), is exactly the uncoupled model. Regimes stay per animal (each sequence
   has its own regime chain; alpha / beta are shared across regimes).

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
ALPHA_MAX = 1.0 - 1e-6   # as swirl_ext
TINY = 1e-300
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
class CouplingConfig:
    enabled: bool = False
    alpha0: float = 0.1           # initial destination coupling (ignored, and fixed at 0, with durations and K = 2)
    fit_alpha: bool = True
    hazard: object = None         # partner term in the hazard (None: = durations on)
    beta0: float = 0.0
    fit_beta: bool = True
    # scipy tol of the coupled Q / hazard M-steps. The losses are per step and the alpha / beta gradients are
    # O(steps with partners x switch rate), so swirl_ext's 1e-4 barely moves them (with durations not at all).
    # Default 1e-8; 1e-4 reproduces swirl_ext.em_train_ext exactly (test E11).
    tol: float = 1e-8


@dataclass
class HiqlConfig:
    pseudocount: float = 0.0      # as hiavi_pseudocount (depth-1 counts)
    eps: float = 1e-6             # hiql smoothing of unvisited (s, a)
    backoff: float = 5.0          # Dirichlet strength towards the parent-context policy (order > 1; inf = order 1)
    discount: float = 0.9         # IAVI (post-fit rewards only)
    iavi_tol: float = 1e-8
    iavi_max_iters: int = 10000


@dataclass
class ScalesConfig:
    order: int = 1
    emission: str = "swirl"          # swirl | hiql
    transition_input: str = None     # state | none (None: state for swirl, none for hiql)
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
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    hiql: HiqlConfig = field(default_factory=HiqlConfig)
    seed: int = 0

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        h = d.pop("switch_history", d.pop("history", None))
        du = d.pop("durations", None)
        hi = d.pop("hierarchy", None)
        cp = d.pop("coupling", None)
        hq = d.pop("hiql", None)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.history = _history_conf(h)
        cfg.durations = _sub_conf(DurationConfig, du)
        cfg.hierarchy = _sub_conf(HierarchyConfig, hi)
        cfg.coupling = _sub_conf(CouplingConfig, cp)
        cfg.hiql = HiqlConfig(**{k: v for k, v in dict(hq or {}).items() if k in HiqlConfig.__dataclass_fields__})
        cfg.hiql.backoff = float(cfg.hiql.backoff)
        if cfg.emission not in ("swirl", "hiql"):
            raise ValueError(f"emission must be swirl | hiql, got {cfg.emission!r}")
        if cfg.transition_input is None:
            cfg.transition_input = "state" if cfg.emission == "swirl" else "none"
        if cfg.transition_input not in ("state", "none"):
            raise ValueError(f"transition_input must be state | none, got {cfg.transition_input!r}")
        if cfg.coupling.hazard is None:
            cfg.coupling.hazard = cfg.durations.enabled
        cfg.coupling.hazard = bool(cfg.coupling.hazard) and cfg.durations.enabled
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

    @property
    def use_state(self):
        return self.transition_input == "state"

    def to_json(self):
        return json.dumps(asdict(self), default=list)

    def with_coupling(self, enabled=True, **kw):
        """Copy with coupling switched on / off (kw: CouplingConfig fields)."""
        c = cfg_from_json(self.to_json())
        c.coupling.enabled = bool(enabled)
        for k, v in kw.items():
            if k not in CouplingConfig.__dataclass_fields__:
                raise ValueError(f"unknown coupling option {k!r}")
            setattr(c.coupling, k, v)
        if c.coupling.hazard is None:
            c.coupling.hazard = c.durations.enabled
        c.coupling.hazard = bool(c.coupling.hazard) and c.durations.enabled
        return c

    def tag(self):
        parts = [f"N{self.order}"]
        if self.emission != "swirl":
            parts.insert(0, self.emission.upper())
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
    def build(cls, state_rows, S, T, order=1, min_count=1, extra_pairs=None, lengths=None):
        """
        state_rows: (N, L) int states (the full rows; x = rows[:, 1:-1]). extra_pairs: (prev, x) to add.
        lengths (N,): real states per row (segment_policy: variable); windows reaching into padding are skipped.
        """
        order = int(order)
        nodes = [(s,) for s in range(int(S))]
        if order > 1:
            rows = np.asarray(state_rows)
            counts = {}
            L = rows.shape[1]
            for k in range(2, order + 1):
                if L < k:
                    break
                win = np.lib.stride_tricks.sliding_window_view(rows, k, axis=1)       # (N, L-k+1, k)
                if lengths is not None:
                    ok = np.arange(L - k + 1)[None, :] <= (np.asarray(lengths)[:, None] - k)
                    win = win[ok]
                win = win.reshape(-1, k)
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
    """(N, T, K) from logpi (K, H, A), ctx (N, T), a (N, T); 0 on padded steps (a = -1, segment_policy: variable)."""
    a = jnp.asarray(a)
    ll = jnp.moveaxis(logpi[:, ctx, jnp.maximum(a, 0)], 0, -1)
    return jnp.where((a >= 0)[..., None], ll, 0.0)


# ---------------------------------------------------------------------------
# Transitions and hazards
# ---------------------------------------------------------------------------

def log_Q(log_Ps, Rs, W1f, x, phi, mask_diag, use_state=True):
    """
    (N, T-1, R, K, K) log Q_t(z' | z, r): log_Ps (R, K, K) + MLP([onehot(x_{t-1}), phi_{t-1}]) (use_state), or
    log_Ps + phi_{t-1} @ W1f (W1f (F, K); HIAVI Lambda when F = 0).
    """
    N, T1 = x.shape[0], x.shape[1] - 1
    if use_state:
        W1, b1, W2, b2 = Rs
        Z1 = W1[x[:, :-1]] + b1
        if W1f.shape[0] > 0:
            Z1 = Z1 + phi[:, :-1] @ W1f
        Z2 = sigmoid_jax(Z1) @ W2 + b2                                # (N, T-1, K)
    elif W1f.shape[0] > 0:
        Z2 = phi[:, :-1] @ W1f
    else:
        Z2 = jnp.zeros((N, T1, log_Ps.shape[-1]))
    logits = log_Ps[None, None] + Z2[:, :, None, None, :]
    if mask_diag:
        logits = logits + NEG * jnp.eye(log_Ps.shape[-1])
    return logits - lse(logits, axis=-1, keepdims=True)


def hazard_logits(hz, D, phi, kind, use_feat, cpl=None):
    """
    Hazard logits: (R, K, D), or (N, T-1, R, K, D) with features / partner coupling. cpl (N, T-1, K) =
    w_t (1 - c_t[z]) adds beta * cpl (hz["beta"]).
    """
    tau = jnp.log(jnp.arange(1, D + 1, dtype=jnp.float64))
    base = hz["a"][..., None] + hz["b"][None, :, None] * tau if kind == "logistic" else hz["a"]
    if use_feat:
        base = base[None, None] + jnp.einsum("ntf,kf->ntk", phi[:, :-1], hz["u"])[:, :, None, :, None]
    if cpl is not None and "beta" in hz:
        term = jnp.reshape(hz["beta"], ()) * cpl[:, :, None, :, None]
        base = (base if base.ndim == 5 else base[None, None]) + term
    return base


def log_hazards(cfg, hz, phi, cpl=None):
    """(log h, log (1 - h)) with shape (R, K, D) or (N, T-1, R, K, D)."""
    if not cfg.durations.enabled:
        return None, None
    lg = hazard_logits(hz, cfg.D, phi, cfg.durations.hazard, cfg.durations.hazard_features, cpl)
    return log_sigmoid(lg), log_sigmoid(-lg)


def coupled_logQ(lq, alpha, w, c, mask_diag):
    """
    Partner-mixed transitions: lq (N, T-1, R, K, K) normalised, w (N, T), c (N, T, K) (destination step t).
    (1 - alpha w_t) Q + alpha w_t c~_t; with mask_diag (durations) c~ drops the current intention.
    """
    K = lq.shape[-1]
    aw = (alpha * w[:, 1:])[:, :, None, None, None]
    ct = c[:, 1:, None, None, :]
    if mask_diag:
        off = 1.0 - jnp.eye(K)
        ct = ct * off
        tot = ct.sum(-1, keepdims=True)
        ct = jnp.where(tot > 1e-12, ct / jnp.maximum(tot, 1e-12), off / max(K - 1, 1))
    P = (1 - aw) * jnp.exp(lq) + aw * ct
    out = jnp.log(jnp.clip(P, TINY))
    if mask_diag:
        out = jnp.where(jnp.eye(K, dtype=bool), NEG, out)
    return out


def alpha_active(cfg, K):
    """Destination coupling is identifiable (not with durations and K = 2, where Q is fixed)."""
    return cfg.coupling.enabled and not (cfg.durations.enabled and K <= 2)


def coupling_alpha(p, cfg):
    K = np.shape(p["log_Ps"])[-1]
    if not alpha_active(cfg, K) or "cpl_alpha" not in p:
        return 0.0
    return float(ALPHA_MAX / (1 + np.exp(-float(np.asarray(p["cpl_alpha"])))))


def coupling_beta(p, cfg):
    if not (cfg.coupling.enabled and cfg.coupling.hazard) or "beta" not in p.get("hz", {}):
        return 0.0
    return float(np.asarray(p["hz"]["beta"]).reshape(()))


def _ctx(d, K):
    """(w, c) from the data bundle (zeros = no partner, i.e. the uncoupled model)."""
    N, T = d["a"].shape
    w = d.get("w")
    if w is None:
        return jnp.zeros((N, T)), jnp.full((N, T, K), 1.0 / K)
    return jnp.asarray(w, dtype=jnp.float64), jnp.asarray(d["c"], dtype=jnp.float64)


def _cpl_h(cfg, d, K):
    if not (cfg.coupling.enabled and cfg.coupling.hazard):
        return None
    w, c = _ctx(d, K)
    return w[:, 1:, None] * (1.0 - c[:, 1:])                          # (N, T-1, K)


def with_context(d, w, c):
    """Data bundle with the partner context (w (N, T), c (N, T, K))."""
    out = dict(d)
    out["w"], out["c"] = jnp.asarray(w, dtype=jnp.float64), jnp.asarray(c, dtype=jnp.float64)
    return out


def trans_logQ(p, cfg, d):
    """Intention transitions of the bundle d, partner-mixed when coupling is on."""
    lq = log_Q(p["log_Ps"], p["Rs"], p["W1f"], d["x"], d["phi"], cfg.durations.enabled, cfg.use_state)
    K = lq.shape[-1]
    if alpha_active(cfg, K) and "cpl_alpha" in p:
        w, c = _ctx(d, K)
        alpha = ALPHA_MAX * jax.nn.sigmoid(jnp.asarray(p["cpl_alpha"]))
        lq = coupled_logQ(lq, alpha, w, c, cfg.durations.enabled)
    return lq


# ---------------------------------------------------------------------------
# Forward / backward on (r, z, tau)
# ---------------------------------------------------------------------------

def _shift(B):
    """Bn[..., i] = B[..., min(i + 1, D - 1)]."""
    return jnp.concatenate([B[..., 1:], B[..., -1:]], axis=-1)


def _seq_fb(logpi0, logPR, ll, logQ, lh, l1h, want_kd, ht, m=None):
    """
    m (T,) bool: real steps (padding at the end, segment_policy: variable). A padded step carries alpha / beta
    unchanged (identity transition), so logZ and the real-step posteriors are exact; its pairwise statistics
    and posteriors are zeroed, and cum stays flat (step LL 0).
    One sequence. logpi0 (R, K), logPR (R, R), ll (T, K), logQ (T-1, R, K, K), lh / l1h (R, K, D) or
    (T-1, R, K, D) (hazard of the PREVIOUS regime). Returns logZ, cum (T,), g_z (T, K), g_r (T, R), g0 (R, K),
    EQ (T-1, R, K, K), S0 / S1 (stay / switch mass, (T-1, R, K, D) if ht else (R, K, D)), ER (R, R),
    g_kd ((T, K, D) or None).
    """
    R, K = logpi0.shape
    D = lh.shape[-1]
    la0 = jnp.full((R, K, D), -jnp.inf).at[:, :, 0].set(logpi0 + ll[0][None])

    def fwd(la, inp):
        lq, l_t, h, h1, m_t = inp
        u = la + h1
        st = jnp.concatenate([jnp.full((R, K, 1), -jnp.inf), u[..., :-1]], axis=-1)
        st = st.at[..., -1].set(jnp.logaddexp(st[..., -1], u[..., -1])) if D > 1 else u
        stay = lse(logPR[:, :, None, None] + st[:, None], axis=0)
        v = lse(la + h, axis=-1)                                      # (R, K)
        w = lse(logPR[:, :, None] + v[:, None, :], axis=0)            # (R', K)
        sw = lse(w[:, :, None] + lq, axis=1)                          # (R', K')
        new = stay.at[..., 0].set(jnp.logaddexp(stay[..., 0], sw))
        new = new + l_t[None, :, None]
        new = jnp.where(m_t, new, la)
        return new, new

    T = ll.shape[0]
    m = jnp.ones(T, dtype=bool) if m is None else jnp.asarray(m, dtype=bool)
    hs = (lh, l1h) if ht else (jnp.broadcast_to(lh, (T - 1,) + lh.shape), jnp.broadcast_to(l1h, (T - 1,) + l1h.shape))
    _, las = lax.scan(fwd, la0, (logQ, ll[1:], hs[0], hs[1], m[1:]))
    la = jnp.concatenate([la0[None], las], axis=0)                    # (T, R, K, D)
    logZ = lse(la[-1])
    cum = lse(la.reshape(T, -1), axis=-1)

    def bwd(lb, inp):
        la_p, lq, l_t, h, h1, m_t = inp
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
        lb_p = jnp.where(m_t, lb_p, lb)                               # padded step t: identity
        g = la_p + lb_p - logZ
        return lb_p, (g, m_t * EQ, m_t * S0, m_t * S1, m_t * ER)

    lbT = jnp.zeros((R, K, D))
    _, (gs, EQ, S0, S1, ER) = lax.scan(bwd, lbT, (la[:-1], logQ, ll[1:], hs[0], hs[1], m[1:]), reverse=True)
    g = jnp.exp(jnp.concatenate([gs, (la[-1] - logZ)[None]], axis=0))  # (T, R, K, D)
    g = g * m[:, None, None, None]
    out = dict(logZ=logZ, cum=cum, g_z=g.sum((1, 3)), g_r=g.sum((2, 3)), g0=g[0, :, :, 0], EQ=EQ,
               S0=S0 if ht else S0.sum(0), S1=S1 if ht else S1.sum(0), ER=ER.sum(0))
    out["g_kd"] = g.sum(1) if want_kd else None
    return out


@partial(jax.jit, static_argnames=("want_kd", "ht"))
def _fb_batch(logpi0, logPR, ll, logQ, lh, l1h, want_kd, ht, m=None):
    hax = 0 if ht else None
    if m is None:
        m = jnp.ones(ll.shape[:2], dtype=bool)
    f = vmap(lambda l, q, h, h1, mm: _seq_fb(logpi0, logPR, l, q, h, h1, want_kd, ht, mm),
             in_axes=(0, 0, hax, hax, 0))
    return f(ll, logQ, lh, l1h, m)


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

def init_params(cfg, tree, base, R_start, n_feat, seed=None):
    """
    base = (logpi0 (K,), log_Ps (K, K), Rs = (W1, b1, W2, b2)) from the ARHMM (or sticky_init); R_start
    (K, S) or (K, 1, S) state rewards. With every toggle off and order 1 the model equals this SWIRL init.
    HIQL (cfg.emission = "hiql"): base = (logpi0, log_Ps = HIAVI Lambda, Rs or None); R_start = (K, S, A) log
    policies (HIAVI), copied to every context by its last state.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    logpi0, log_Ps = np.asarray(base[0], float), np.asarray(base[1], float)
    K, S = log_Ps.shape[0], tree.S
    if base[2] is None:
        Rs = (np.zeros((S, 1)), np.zeros(1), np.zeros((1, K)), np.zeros(K))
    else:
        Rs = tuple(np.asarray(w, float) for w in base[2])
    hidden = Rs[0].shape[1] if cfg.use_state else K
    if cfg.emission == "hiql":
        lp = np.asarray(R_start, float)
        if lp.shape != (K, S, tree.A):
            raise ValueError(f"HIQL R_start must be (K, S, A) = {(K, S, tree.A)} log policies, got {lp.shape}")
        lp = lp - np.logaddexp.reduce(lp, axis=-1, keepdims=True)
        theta = None
    else:
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
    p = dict(logpi0=np.repeat(lp0[None], R, 0) - np.log(R), log_Ps=lP, Rs=Rs, W1f=np.zeros((n_feat, hidden)))
    if theta is None:
        p["logpi"] = lp[:, tree.last]
    else:
        p["theta"] = theta
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
    cc = cfg.coupling
    if cc.enabled:
        p["cpl_alpha"] = np.asarray(_logit(cc.alpha0 / ALPHA_MAX))
        if cc.hazard:
            if not cfg.durations.enabled:
                raise ValueError("coupling.hazard needs durations.enabled")
            p["hz"]["beta"] = np.asarray(float(cc.beta0))
    return p


def _jnp_tree(p):
    return jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=jnp.float64), p)


def n_params(p, cfg, tree):
    K = int(np.shape(p["log_Ps"])[-1])
    emit = K * tree.H * (tree.A - 1) if cfg.emission == "hiql" else np.size(p["theta"])
    n = emit + np.size(p["logpi0"]) - 1 + np.size(p["log_Ps"]) + np.size(p["W1f"])
    if cfg.use_state:
        n += sum(np.size(w) for w in p["Rs"])
    if cfg.R > 1:
        n += cfg.R * (cfg.R - 1)
    if "hz" in p:
        n += sum(np.size(v) for k, v in p["hz"].items() if k != "beta")
    if alpha_active(cfg, K) and cfg.coupling.fit_alpha:
        n += 1
    if cfg.coupling.enabled and cfg.coupling.hazard and cfg.coupling.fit_beta:
        n += 1
    return int(n)


# ---------------------------------------------------------------------------
# Data bundle
# ---------------------------------------------------------------------------

def data_mask(d):
    """(N, T) bool real steps of a bundle (a = -1 on padded steps of segment_policy: variable)."""
    return jnp.asarray(d["a"]) >= 0


def n_real_steps(d):
    return float(np.asarray(d["a"] >= 0).sum())


def scales_data(state_rows, a, tree, hcfg, S, kin=None, idx=None):
    """
    dict(x, ctx, a, phi, kin) for the rows idx (x = rows[:, 1:-1], a aligned with x). Padded steps
    (segment_policy: variable) are marked by a = -1; their x / ctx / phi are ignored (masked) downstream.
    """
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


def model_log_policy(p, cfg, tree_arr, temps):
    """(K, H, A) log pi for either emission."""
    if cfg.emission == "hiql":
        return jnp.asarray(p["logpi"])
    K = p["theta"].shape[0]
    tmp, use_temp = _temps_arr(temps, K)
    return log_policy(p["theta"], tree_arr["nxt"], tree_arr["prob"], tree_arr["anc"], tmp, cfg.reward_param,
                      use_temp)


def e_step(p, cfg, tree_arr, d, temps, kin_model=None, want_kd=False, use_kin=True):
    K = int(np.shape(p["log_Ps"])[-1])
    logpi = model_log_policy(p, cfg, tree_arr, temps)
    ll = emission_ll(logpi, d["ctx"], d["a"])
    if use_kin and kin_model is not None and d["kin"] is not None:
        ll = ll + jnp.where(data_mask(d)[..., None], jnp.asarray(kin_model.loglik(d["kin"])), 0.0)
    lq = trans_logQ(p, cfg, d)
    if cfg.durations.enabled:
        lh, l1h = log_hazards(cfg, p["hz"], d["phi"], _cpl_h(cfg, d, K))
    else:
        R = p["logPR"].shape[0]
        lh, l1h = jnp.zeros((R, K, 1)), jnp.full((R, K, 1), -jnp.inf)
    ht = lh.ndim == 5
    lpi0 = p["logpi0"] - lse(p["logpi0"])
    lPR = p["logPR"] - lse(p["logPR"], axis=1, keepdims=True)
    return _fb_batch(lpi0, lPR, ll, lq, lh, l1h, want_kd, ht, data_mask(d))


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


def _q_loss(flat, shapes, EQ, x, phi, n_total, mask, use_state, w, c, coupled, alpha_fixed, fit_alpha):
    lP, W1, b1, W2, b2, W1f = _unflat(flat[:-1] if coupled and fit_alpha else flat, shapes)
    lq = log_Q(lP, (W1, b1, W2, b2), W1f, x, phi, mask, use_state)
    if coupled:
        alpha = ALPHA_MAX * jax.nn.sigmoid(flat[-1]) if fit_alpha else alpha_fixed
        lq = coupled_logQ(lq, alpha, w, c, mask)
    return -jnp.sum(EQ * lq) / n_total


_q_vg = jax.jit(jax.value_and_grad(_q_loss), static_argnames=("shapes", "mask", "use_state", "coupled", "fit_alpha"))


def _logit(q):
    q = float(np.clip(q, 1e-4, 1 - 1e-4))
    return float(np.log(q / (1 - q)))


def q_closed_form(cfg, st):
    """Normalised expected transition counts (no inputs, no coupling): HIAVI's Lambda M-step per regime."""
    EQ = np.asarray(st["EQ"]).sum((0, 1))                             # (R, K, K)
    lP = np.log(np.maximum(EQ, TINY))
    if cfg.durations.enabled:
        K = EQ.shape[-1]
        lP = np.where(np.eye(K, dtype=bool)[None], NEG, lP)
    lP = lP - np.logaddexp.reduce(lP, axis=-1, keepdims=True)
    if cfg.durations.enabled:
        lP = np.where(np.eye(lP.shape[-1], dtype=bool)[None], 0.0, lP)
    return lP


def coupling_tol(cfg):
    return float(cfg.coupling.tol)


def q_m_step(p, cfg, st, d, n_total, num_iters=1000):
    """-> log_Ps, Rs, W1f, cpl_alpha (raw logit or None)."""
    K = int(np.shape(p["log_Ps"])[-1])
    coupled = alpha_active(cfg, K) and "cpl_alpha" in p
    fit_alpha = coupled and bool(cfg.coupling.fit_alpha)
    if not cfg.use_state and np.size(p["W1f"]) == 0 and not coupled:
        return q_closed_form(cfg, st), p["Rs"], np.asarray(p["W1f"]), p.get("cpl_alpha")
    parts = [p["log_Ps"], *p["Rs"], p["W1f"]]
    shapes = tuple(tuple(int(s) for s in np.shape(v)) for v in parts)
    flat0 = _flat([jnp.asarray(v) for v in parts])
    if fit_alpha:
        flat0 = jnp.concatenate([flat0, jnp.reshape(jnp.asarray(p["cpl_alpha"], dtype=jnp.float64), (1,))])
    w, c = _ctx(d, K)
    a_fix = ALPHA_MAX * jax.nn.sigmoid(jnp.asarray(p["cpl_alpha"])) if coupled else jnp.asarray(0.0)
    res = _minimize_tol(_q_vg, flat0, "L-BFGS-B", num_iters, coupling_tol(cfg) if coupled else 1e-4,
                    shapes=shapes, EQ=st["EQ"], x=d["x"], phi=d["phi"],
                    n_total=n_total, mask=cfg.durations.enabled, use_state=cfg.use_state, w=w, c=c,
                    coupled=coupled, alpha_fixed=a_fix, fit_alpha=fit_alpha)
    x = jnp.asarray(res.x)
    lP, W1, b1, W2, b2, W1f = [np.asarray(v) for v in _unflat(x[:-1] if fit_alpha else x, shapes)]
    alpha_raw = np.asarray(x[-1]) if fit_alpha else p.get("cpl_alpha")
    return lP, (W1, b1, W2, b2), W1f, alpha_raw


def _hz_loss(flat, shapes, keys, S0, S1, phi, n_total, D, kind, use_feat, smooth, l2, cpl, fixed):
    hz = dict(zip(keys, _unflat(flat, shapes)))
    hz.update(fixed)
    lg = hazard_logits(hz, D, phi, kind, use_feat, cpl)
    nll = -jnp.sum(S1 * log_sigmoid(lg) + S0 * log_sigmoid(-lg)) / n_total
    pen = l2 * jnp.sum(hz["u"] ** 2)
    if kind == "logistic":
        pen = pen + l2 * jnp.sum(hz["b"] ** 2)
    else:
        pen = pen + smooth * jnp.sum(jnp.diff(hz["a"], axis=-1) ** 2)
    return nll + pen / n_total


_hz_vg = jax.jit(jax.value_and_grad(_hz_loss), static_argnames=("shapes", "keys", "D", "kind", "use_feat"))


def hazard_m_step(p, cfg, st, d, n_total, num_iters=1000):
    K = int(np.shape(p["log_Ps"])[-1])
    fixed = {}
    if "beta" in p["hz"] and not (cfg.coupling.enabled and cfg.coupling.hazard and cfg.coupling.fit_beta):
        fixed["beta"] = jnp.asarray(p["hz"]["beta"])
    keys = tuple(sorted(k for k in p["hz"] if k not in fixed))
    parts = [jnp.asarray(p["hz"][k]) for k in keys]
    shapes = tuple(tuple(int(s) for s in v.shape) for v in parts)
    dc = cfg.durations
    cpl = _cpl_h(cfg, d, K)
    tol = coupling_tol(cfg) if cpl is not None else 1e-4
    res = _minimize_tol(_hz_vg, _flat(parts), "L-BFGS-B", num_iters, tol, shapes=shapes, keys=keys, S0=st["S0"],
                    S1=st["S1"], phi=d["phi"], n_total=n_total, D=cfg.D, kind=dc.hazard,
                    use_feat=dc.hazard_features, smooth=float(dc.smooth), l2=float(dc.l2),
                    cpl=cpl, fixed=fixed)
    out = {k: np.asarray(v) for k, v in zip(keys, _unflat(jnp.asarray(res.x), shapes))}
    out.update({k: np.asarray(v) for k, v in fixed.items()})
    return out


# ---------------------------------------------------------------------------
# HIQL emission (closed form) and IAVI on the context MDP
# ---------------------------------------------------------------------------

def hiql_counts(g, ctx, a, H, A):
    """(K, H, A) posterior-weighted (context, action) counts."""
    g = np.asarray(g)
    idx = (np.asarray(ctx) * A + np.maximum(np.asarray(a), 0)).ravel()   # padded steps have g = 0
    K = g.shape[-1]
    return np.stack([np.bincount(idx, weights=g[..., k].ravel(), minlength=H * A) for k in range(K)]).reshape(K, H, A)


def hiql_policies(cnt, tree, hc):
    """
    (K, H, A) policies from own-context counts cnt (each step counts at its data context only).
    pooled_s: counts of every context ending in s (+ pseudocount, empty rows eps) = HIAVI's order-1 M-step.
    Order 1: pi = pooled. Order >= 2, back-off kappa = hc.backoff:
        depth 1:  (n(s) + kappa pooled_s) / (n(s) + kappa)
        depth k:  (n(h) + kappa pi(parent)) / (n(h) + kappa), parent = h without its oldest state
    kappa = inf -> every context uses pooled_s (== order 1); kappa = 0 -> per-context ML (exact M-step; empty
    contexts back off to the parent).
    """
    K, H, A = cnt.shape
    S = tree.S
    pooled = np.zeros((K, S, A))
    for s_ in range(S):
        pooled[:, s_] = cnt[:, tree.last == s_].sum(1)
    pooled = pooled + hc.pseudocount
    empty = pooled.sum(-1) == 0
    pooled[empty] = hc.eps
    pooled = pooled / pooled.sum(-1, keepdims=True)
    pis = np.zeros((K, H, A))
    if tree.order == 1:
        pis[:, :S] = pooled
        return pis
    kap = float(hc.backoff)

    def shrink(n, target):
        if np.isinf(kap):
            return target
        num = n + kap * target
        den = num.sum(-1, keepdims=True)
        return np.where(den > 0, num / np.maximum(den, TINY), target)

    pis[:, :S] = shrink(cnt[:, :S], pooled)
    for k in range(2, tree.order + 1):
        m = np.flatnonzero(tree.depth == k)
        if len(m):
            par = np.array([tree.index[tree.nodes[h][1:]] for h in m])
            pis[:, m] = shrink(cnt[:, m], pis[:, par])
    return pis


def hiql_logpi(pis, eps):
    """HIAVI._logpis: softmax(q_IAVI) == (pi + eps) / sum(pi + eps)."""
    p = pis + eps
    return np.log(p / p.sum(-1, keepdims=True))


def hiql_m_step(cfg, tree, st, d):
    H, A = tree.H, tree.A
    cnt = hiql_counts(st["g_z"], d["ctx"], d["a"], H, A)
    return hiql_logpi(hiql_policies(cnt, tree, cfg.hiql), cfg.hiql.eps)


def iavi_tree(tree, expert_pi, discount, tol=1e-8, max_iters=10000, eps=1e-6):
    """
    IAVI (hiavi_model.iavi) on the context MDP: taking a in context h lands in next(h, s') with T(s' | x_t, a).
    expert_pi (H, A). Returns r (H, A), q (H, A); order 1 equals hiavi_model.iavi on T.
    """
    logpi = np.log(expert_pi + eps)
    H, A = logpi.shape
    q = np.zeros((H, A))
    r = np.zeros((H, A))
    for _ in range(int(max_iters)):
        cont = discount * (tree.prob * q.max(axis=1)[tree.nxt]).sum(-1)
        eta = logpi - cont
        r_new = eta - eta.mean(axis=1, keepdims=True)
        q = r_new + cont
        delta = np.max(np.abs(r_new - r))
        r = r_new
        if delta < tol:
            break
    return r, q


def hiql_finalize(r, tree, d):
    """IAVI once per intention from the final responsibilities (as HIAVI.fit): adds r['iavi'] = (r, q) (K, H, A)."""
    cfg = r["cfg"]
    st = e_step(_jnp_tree(r["params"]), cfg, tree.arrays(), d, None, r.get("kin_model"))
    cnt = hiql_counts(st["g_z"], d["ctx"], d["a"], tree.H, tree.A)
    expert = hiql_policies(cnt, tree, cfg.hiql)
    out = [iavi_tree(tree, expert[k], cfg.hiql.discount, cfg.hiql.iavi_tol, cfg.hiql.iavi_max_iters, cfg.hiql.eps)
           for k in range(expert.shape[0])]
    r = dict(r)
    r["iavi"] = dict(r=np.stack([o[0] for o in out]), q=np.stack([o[1] for o in out]))
    return r


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
    n_total = n_real_steps(d)
    if cfg.emission == "hiql" and temps is not None:
        raise ValueError("temps apply to the SWIRL emission only")
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
            lP, Rs, W1f, a_raw = q_m_step(p, cfg, st, d, n_total, num_iters)
            new["log_Ps"], new["Rs"], new["W1f"] = jnp.asarray(lP), tuple(jnp.asarray(w) for w in Rs), jnp.asarray(W1f)
            if a_raw is not None:
                new["cpl_alpha"] = jnp.asarray(a_raw)
            if cfg.durations.enabled:
                new["hz"] = _jnp_tree(hazard_m_step(p, cfg, st, d, n_total, num_iters))
        if emit:
            if cfg.emission == "hiql":
                new["logpi"] = jnp.asarray(hiql_m_step(cfg, tree, st, d))
            else:
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
    m = np.asarray(data_mask(d))
    n = int(m.sum())
    g = np.asarray(st["g_z"])
    out = dict(LL_total=float(cum[:, -1].sum()), LL=float(cum[:, -1].sum() / n),
               step_ll=np.where(m, np.diff(cum, axis=1, prepend=0.0), 0.0), gamma=g, gamma_r=np.asarray(st["g_r"]),
               path=np.where(m, g.argmax(-1), -1), n_steps=n, mask=m)
    if want_kd:
        out["gamma_zt"] = np.asarray(st["g_kd"])                      # (N, T, K, D)
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def policy(r, tree):
    cfg, p = r["cfg"], r["params"]
    if cfg.emission == "hiql":
        return np.exp(np.asarray(p["logpi"]))
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
    if r["cfg"].emission == "hiql" or r["cfg"].reward_param != "additive":
        K = int(np.shape(r["params"]["log_Ps"])[-1])
        return np.full((K, tree.order), np.nan)
    th = np.asarray(r["params"]["theta"])
    w = np.ones(tree.H) if ctx_counts is None else np.asarray(ctx_counts, float)
    K = th.shape[0]
    out = np.zeros((K, tree.order))
    used = w > 0
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
        row = row[row >= 0]                                           # drop padded steps (path -1)
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
    out = {prefix + "logpi0": p["logpi0"], prefix + "logPR": p["logPR"],
           prefix + "log_Ps": p["log_Ps"], prefix + "W1f": p["W1f"], prefix + "cfg": r["cfg"].to_json(),
           "LL_list": np.asarray(r["LL_list"])}
    for k in ("theta", "logpi", "cpl_alpha"):
        if k in p:
            out[prefix + k] = np.asarray(p[k])
    if "iavi" in r:
        out[prefix + "iavi_r"], out[prefix + "iavi_q"] = r["iavi"]["r"], r["iavi"]["q"]
    for i, w in enumerate(p["Rs"]):
        out[prefix + f"Rs{i}"] = np.asarray(w)
    for k, v in p.get("hz", {}).items():
        out[prefix + "hz_" + k] = np.asarray(v)
    if r.get("temps") is not None:
        out["temps"] = np.asarray(r["temps"])
    if r.get("kin_model") is not None:
        out.update(r["kin_model"].params())
    out.update(tree.to_save())
    if "theta" in p:
        # reward on the depth-1 nodes in the SWIRL (K, 1, S) layout, for the existing plots
        R = np.asarray(node_rewards(jnp.asarray(p["theta"]), jnp.asarray(tree.anc), r["cfg"].reward_param))
        out["new_reward"] = R[:, None, :tree.S]
    return out


def cfg_from_json(s):
    c = json.loads(str(s))
    sub = ("history", "durations", "hierarchy", "coupling", "hiql")
    cfg = ScalesConfig.from_dict({k: v for k, v in c.items() if k not in sub})
    cfg.history = HistoryConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in c["history"].items()})
    cfg.durations = DurationConfig(**c["durations"])
    cfg.hierarchy = HierarchyConfig(**c["hierarchy"])
    if "coupling" in c:
        cfg.coupling = CouplingConfig(**c["coupling"])
    if "hiql" in c:
        cfg.hiql = HiqlConfig(**c["hiql"])
    return cfg


def params_from_npz(z, prefix="sc_"):
    p = dict(logpi0=z[prefix + "logpi0"], logPR=z[prefix + "logPR"],
             log_Ps=z[prefix + "log_Ps"], W1f=z[prefix + "W1f"],
             Rs=tuple(z[prefix + f"Rs{i}"] for i in range(4)))
    for k in ("theta", "logpi", "cpl_alpha"):
        if prefix + k in z.files:
            p[k] = z[prefix + k]
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
    if "theta" in p_new and "theta" in po:
        th = np.array(p_new["theta"], float)
        if r_old["cfg"].reward_param == "additive":
            for h, c in enumerate(tree_old.nodes):
                j = tree_new.index.get(c)
                if j is not None:
                    th[:, j] = po["theta"][:, h]
        out["theta"] = th
    if "logpi" in p_new and "logpi" in po:
        lp = np.array(p_new["logpi"], float)
        old = np.asarray(po["logpi"])
        lp[:] = old[:, [tree_old.index[(int(s),)] for s in tree_new.last]]   # back off to the last state
        for h, c in enumerate(tree_old.nodes):
            j = tree_new.index.get(c)
            if j is not None:
                lp[:, j] = old[:, h]
        out["logpi"] = lp
    for k in ("logpi0", "log_Ps", "logPR", "W1f", "cpl_alpha"):
        if k in p_new and k in po and np.shape(po[k]) == np.shape(p_new[k]):
            out[k] = np.asarray(po[k])
    if all(np.shape(a) == np.shape(b) for a, b in zip(po["Rs"], p_new["Rs"])):
        out["Rs"] = tuple(np.asarray(w) for w in po["Rs"])
    if "hz" in p_new and "hz" in po:
        hz = dict(p_new["hz"])
        for k, v in p_new["hz"].items():
            if k in po["hz"] and np.shape(po["hz"][k]) == np.shape(v):
                hz[k] = np.asarray(po["hz"][k])
        out["hz"] = hz
    return out
