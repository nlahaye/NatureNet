"""
Equivalence / reduction tests for swirl_scales (CPU, ~1-2 min):

    PYTHONPATH=src JAX_PLATFORMS=cpu NATURENET_JAX_PLATFORM=cpu python -m naturenet.models.irl.test_swirl_scales
    (or pytest src/naturenet/models/irl/test_swirl_scales.py)

E1  all toggles off, order 1 == swirl_ext.em_train_ext (EM trace, rewards, posteriors, step LLs), with and
    without temperatures
E2  order 2, reward_param "free", T-support pairs == S-2 (em_train2_naturenet EM trace)
E3  order N (additive, increments 0) at init == order 1 (the history increments start at the S-1 policy)
E4  durations on with b = 0 at init == the geometric (SWIRL) model it was initialised from
E5  hierarchy with identical regimes (init_noise 0) == R = 1
E6  switching-history features with zero weights == no features
E7  every toggle on (no penalties): EM trace non-decreasing
E8  context tree: data contexts are consistent with the VI next-context map
E9  memory: at order N the policy depends on the last N-1 states only (order 2 / S-2: pi(a | x_prev, x) =
    pi(a | x) for any reward; order 3: depends on x_prev)
"""
from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import vmap

from naturenet.models.irl import swirl_scales as sc
from naturenet.models.irl.swirl_data import init_rewards
from naturenet.models.irl.swirl_ext import em_train_ext, swirl_eval
from naturenet.models.irl.swirl_multiagent import sticky_init

TOL = 1e-7


def rand_mdp(S, A, seed=0, branch=2):
    rng = np.random.default_rng(seed)
    T = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            nx = rng.choice(S, size=branch, replace=False)
            T[s, a, nx] = rng.dirichlet(np.ones(branch))
    return T


def rollout(T, N, L, seed=1):
    rng = np.random.default_rng(seed)
    S, A = T.shape[:2]
    rows, acts = np.zeros((N, L), int), np.zeros((N, L - 1), int)
    for n in range(N):
        s = rng.integers(S)
        rows[n, 0] = s
        for t in range(L - 1):
            a = rng.integers(A)
            s = rng.choice(S, p=T[s, a])
            acts[n, t], rows[n, t + 1] = a, s
    return rows, acts[:, 1:]


def _setup(S=10, A=9, K=3, N=16, L=35, seed=0):
    T = rand_mdp(S, A, seed)
    rows, a = rollout(T, N, L, seed + 1)
    base = sticky_init(K, S, seed)
    _, R2 = init_rewards(K, S, A, seed)
    return T, rows, a, base, R2


def _onehots(rows, a, S, A):
    x = rows[:, 1:-1]
    return jnp.asarray(np.eye(S)[x][:, :, None]), jnp.asarray(np.eye(A)[a][:, :, None])


def _model(cfg, T, rows, a, base, R2, S, extra_pairs=None):
    cfg = sc.ScalesConfig.from_dict(cfg)
    tree = sc.ContextTree.build(rows, S, T, order=cfg.order, extra_pairs=extra_pairs)
    d = sc.scales_data(rows, a, tree, cfg.history, S)
    p = sc.init_params(cfg, tree, base, R2, d["phi"].shape[-1])
    return cfg, tree, d, p


def _ll0(cfg, tree, d, p, temps=None):
    r = dict(params=p, cfg=cfg, temps=temps, kin_model=None)
    return sc.scales_eval(r, tree, d)


def test_e1_order1_equals_swirl():
    T, rows, a, base, R2 = _setup()
    S, A = T.shape[:2]
    xo, ao = _onehots(rows, a, S, A)
    for temps in (None, jnp.array([0.5, 1.0, 1.0])):
        r0 = em_train_ext(jnp.asarray(base[0]), jnp.asarray(base[1]), tuple(jnp.asarray(w) for w in base[2]),
                          jnp.asarray(R2)[:, None], jnp.asarray(T), xo, ao, temps=temps, iters=4)
        cfg, tree, d, p = _model({}, T, rows, a, base, R2, S)
        r1 = sc.em_train_scales(p, cfg, tree, d, temps=temps, iters=4)
        assert np.allclose(r0["LL_list"], r1["LL_list"], rtol=0, atol=1e-6), (r0["LL_list"], r1["LL_list"])
        assert np.abs(np.asarray(r0["rewards"])[:, 0] - r1["params"]["theta"]).max() < 1e-6
        e0, e1 = swirl_eval(r0, jnp.asarray(T), xo, ao, paths=False), sc.scales_eval(r1, tree, d)
        assert abs(e0["LL"] - e1["LL"]) < TOL and np.abs(e0["gamma"] - e1["gamma"]).max() < 1e-6
        assert np.abs(e0["step_ll"] - e1["step_ll"]).max() < 1e-6


def test_e2_order2_free_equals_s2():
    from naturenet.models.irl.swirl_training_top_level import em_train2_naturenet
    from naturenet.models.irl.swirl_utils import (compute_prev_state_map, construct_new_trans_probs_limited,
                                                  n_prev_slots, one_hotx2_partial, preprocess_xs_prev_np)
    T, rows, a, base, R2 = _setup(S=8, K=2, N=12, L=30)
    S, A = T.shape[:2]
    K = R2.shape[0]
    x, xp = rows[:, 1:-1], rows[:, :-2]
    xo, ao = _onehots(rows, a, S, A)
    psm = compute_prev_state_map(T, S, A, observed_pairs=[(x, xp)])
    M = n_prev_slots(psm)
    newT, _ = construct_new_trans_probs_limited(T, psm, None, S, A, n_slots=M)
    xs_prev = preprocess_xs_prev_np(x, xp, psm, A, S)
    x2 = vmap(partial(one_hotx2_partial, n_states=S, n_actions=M))(jnp.asarray(x), jnp.asarray(xs_prev))
    R2b = jnp.repeat(jnp.asarray(R2)[:, None, :, None], M, axis=3).reshape(K, 1, S * M)
    r0 = em_train2_naturenet(jnp.asarray(base[0]), jnp.asarray(base[1]), tuple(jnp.asarray(w) for w in base[2]),
                             R2b, jnp.asarray(newT), xo, x2, ao, 4)
    pairs = [(int(q), s) for s in range(S) for q in psm.get(s, [])]
    cfg, tree, d, p = _model(dict(order=2, reward_param="free", reward_tol=1e-4), T, rows, a, base, R2, S, extra_pairs=pairs)
    r1 = sc.em_train_scales(p, cfg, tree, d, iters=4)
    assert np.allclose(np.asarray(r0[4]), r1["LL_list"], rtol=0, atol=1e-6), (r0[4], r1["LL_list"])


def test_e3_orderN_init_equals_order1():
    T, rows, a, base, R2 = _setup()
    S = T.shape[0]
    ref = _ll0(*_model({}, T, rows, a, base, R2, S))
    for N in (2, 3, 4):
        e = _ll0(*_model(dict(order=N), T, rows, a, base, R2, S))
        assert abs(e["LL"] - ref["LL"]) < TOL, (N, e["LL"], ref["LL"])


def test_e4_geometric_durations():
    T, rows, a, base, R2 = _setup()
    S = T.shape[0]
    ref = _ll0(*_model({}, T, rows, a, base, R2, S))
    for D in (1, 4, 12):
        e = _ll0(*_model(dict(durations=dict(max_duration=D)), T, rows, a, base, R2, S))
        assert abs(e["LL"] - ref["LL"]) < TOL, (D, e["LL"], ref["LL"])
    e = _ll0(*_model(dict(durations=dict(max_duration=6, hazard="nonparametric")), T, rows, a, base, R2, S))
    assert abs(e["LL"] - ref["LL"]) < TOL


def test_e5_degenerate_regimes():
    T, rows, a, base, R2 = _setup()
    S = T.shape[0]
    ref = _ll0(*_model({}, T, rows, a, base, R2, S))
    e = _ll0(*_model(dict(hierarchy=dict(n_regimes=3, init_noise=0.0)), T, rows, a, base, R2, S))
    assert abs(e["LL"] - ref["LL"]) < TOL


def test_e6_zero_history_weights():
    T, rows, a, base, R2 = _setup()
    S = T.shape[0]
    ref = _ll0(*_model({}, T, rows, a, base, R2, S))
    h = dict(kind=["lags", "decay", "summary"], lags=2, decay_steps=[3, 10], window=6)
    cfg, tree, d, p = _model(dict(switch_history=h), T, rows, a, base, R2, S)
    assert d["phi"].shape[-1] == 2 * S + 2 * S + 3
    assert abs(_ll0(cfg, tree, d, p)["LL"] - ref["LL"]) < TOL


def test_e7_em_monotone_all_on():
    T, rows, a, base, R2 = _setup(N=20, L=40)
    S = T.shape[0]
    conf = dict(order=2, reward_l2=0.0, switch_history=dict(kind="lags", lags=1),
                durations=dict(max_duration=6, l2=0.0, hazard_features=True),
                hierarchy=dict(n_regimes=2, init_noise=0.3))
    cfg, tree, d, p = _model(conf, T, rows, a, base, R2, S)
    r = sc.em_train_scales(p, cfg, tree, d, temps=jnp.array([0.5, 1.0, 1.0]), iters=6)
    LL = r["LL_list"]
    assert np.all(np.diff(LL) >= -1e-6 * np.abs(LL[:-1])), LL


def test_e8_context_consistency():
    T, rows, a, base, R2 = _setup(N=10, L=30)
    S = T.shape[0]
    for N in (2, 3):
        tree = sc.ContextTree.build(rows, S, T, order=N)
        ctx = tree.contexts(rows)
        assert np.all(tree.last[ctx] == rows[:, 1:-1])
        for n in range(rows.shape[0]):
            for t in range(ctx.shape[1] - 1):
                h, s2, act = ctx[n, t], rows[n, t + 2], a[n, t]
                b = np.flatnonzero((tree.prob[h, act] > 0) & (tree.last[tree.nxt[h, act]] == s2))
                assert len(b) == 1 and tree.nxt[h, act, b[0]] == ctx[n, t + 1]


def test_e9_policy_memory():
    T, rows, a, base, R2 = _setup(N=10, L=40)
    S = T.shape[0]
    rng = np.random.default_rng(3)
    for N, expect_dep in ((2, False), (3, True)):
        cfg, tree, d, p = _model(dict(order=N), T, rows, a, base, R2, S)
        p["theta"] = rng.normal(size=p["theta"].shape)
        pi = sc.policy(dict(params=p, cfg=cfg, temps=None), tree)
        dev = 0.0
        for c in tree.nodes:
            if len(c) == N:
                key = c[-(N - 1):] if N > 2 else c[-1:]
                same = [tree.index[o] for o in tree.nodes if len(o) == N and o[-(N - 1):] == key[-(N - 1):]]
                dev = max(dev, np.abs(pi[:, same] - pi[:, same[:1]]).max())
        assert dev < 1e-10, (N, dev)                       # the oldest state never matters
        last = [tree.index[o] for o in tree.nodes if len(o) == N]
        spread = max(np.abs(pi[:, [h for h in last if tree.last[h] == s]] -
                            pi[:, [h for h in last if tree.last[h] == s][:1]]).max()
                     for s in range(S) if any(tree.last[h] == s for h in last))
        assert (spread > 1e-4) == expect_dep, (N, spread)


def test_e10_k2_durations_regimes_identifiable():
    """K=2 with durations forces Q to the other intention; regimes and history
    must still act through regime-specific hazards (fits differ, EM monotone)."""
    T, rows, a, base, R2 = _setup(K=2, N=20, L=40)
    S = T.shape[0]
    dur = dict(max_duration=6, l2=0.0)
    c0 = dict(reward_l2=0.0, durations=dur)
    c1 = dict(reward_l2=0.0, durations=dur, hierarchy=dict(n_regimes=2, init_noise=0.5),
              switch_history=dict(kind="lags", lags=1))
    out = []
    for conf in (c0, c1):
        cfg, tree, d, p = _model(conf, T, rows, a, base, R2, S)
        r = sc.em_train_scales(p, cfg, tree, d, iters=5)
        LL = np.asarray(r["LL_list"])
        assert np.all(np.diff(LL) >= -1e-6 * np.abs(LL[:-1])), LL
        out.append(LL[-1])
    assert abs(out[1] - out[0]) > 1e-6, out


if __name__ == "__main__":
    import time
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            f()
            print(f"{name}: ok ({time.time() - t:.1f} s)")
