"""
segment_policy: variable (padded + masked sequences). CPU, a few minutes:

    PYTHONPATH=src JAX_PLATFORMS=cpu NATURENET_JAX_PLATFORM=cpu python -m naturenet.models.irl.test_variable_length
    (or pytest src/naturenet/models/irl/test_variable_length.py)

V1  build_dataset: variable keeps every run >= min_seq_len, splits runs > seq_len into near-equal pieces
    (shared boundary state), pads at the end; equal-length runs -> identical to chunk
V2  align_for_swirl / step_mask: padded steps a = -1, (x, x_prev) = first real step
V3  SWIRL (swirl_ext, S-1 + kinematics + partner context): padded batch == every sequence scored alone
    (total LL, step LLs, posteriors, Viterbi paths); learnt_LL1 / learnt_LL2 likewise
V4  SWIRL EM is invariant to the padded content (garbage states / kinematics / context on padded steps)
V5  HIAVI: padded == per sequence (LL, posteriors, paths), EM invariant to padded content
V6  coupled HIAVI: forward_backward_tv / loglik / decode padded == per sequence, EM invariant
V7  decision scales, every toggle on (SWIRL and HIQL emissions, coupling): padded == per sequence, EM invariant
V8  fixed-length data through the variable code paths == the old results (mask all True)
V9  scripts that need fixed lengths reject segment_policy: variable
"""
import numpy as np
import jax.numpy as jnp

from naturenet.models.irl import swirl_scales as sc
from naturenet.models.irl.kinematics import KinematicEmissions, KinFeatures
from naturenet.models.irl.swirl_data import (PAD_ACTION, align_for_swirl, build_dataset, init_rewards,
                                             load_swirl_dataset, step_mask)
from naturenet.models.irl.swirl_ext import em_train_ext, swirl_eval
from naturenet.models.irl.swirl_multiagent import sticky_init

TOL = 1e-6
LENS = [5, 9, 14, 22, 30, 30, 7, 18]          # states per sequence (min 3)


def rand_mdp(S, A, seed=0, branch=2):
    rng = np.random.default_rng(seed)
    T = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            nx = rng.choice(S, size=branch, replace=False)
            T[s, a, nx] = rng.dirichlet(np.ones(branch))
    return T


def rollout_var(T, lens, seed=1):
    """Lists of raw (states, actions) of the given lengths."""
    rng = np.random.default_rng(seed)
    S, A = T.shape[:2]
    out = []
    for L in lens:
        s = rng.integers(S)
        st, ac = [s], []
        for _ in range(L - 1):
            a = rng.integers(A)
            s = rng.choice(S, p=T[s, a])
            ac.append(a)
            st.append(s)
        out.append((np.array(st), np.array(ac)))
    return out


def pad(seqs, garbage=None):
    """Raw padded (N, Lmax) states / (N, Lmax-1) actions as build_dataset makes them (or garbage pads)."""
    lens = np.array([len(s) for s, _ in seqs])
    Lm = lens.max()
    rows = np.zeros((len(seqs), Lm), int)
    acts = np.zeros((len(seqs), Lm - 1), int)
    rng = None if garbage is None else np.random.default_rng(garbage)
    for i, (s, a) in enumerate(seqs):
        rows[i, :len(s)], acts[i, :len(a)] = s, a
        rows[i, len(s):] = s[-1] if rng is None else rng.integers(rows.max() + 1, size=Lm - len(s))
        if rng is not None:
            acts[i, len(a):] = rng.integers(9, size=Lm - 1 - len(a))
    return rows, acts, lens


def aligned(rows, acts, lens):
    x, xp, a = align_for_swirl(rows, acts, lens)
    return x, xp, a


def onehots(x, a, S, A):
    m = (a >= 0)[..., None, None]
    xo = np.eye(S)[x][:, :, None] * m
    ao = np.eye(A)[np.maximum(a, 0)][:, :, None] * m
    return jnp.asarray(xo), jnp.asarray(ao)


def kin_feats(N, T, lens=None, seed=3, garbage=False):
    rng = np.random.default_rng(seed)
    step = rng.gamma(2.0, 1.0, size=(N, T))
    turn = rng.uniform(-np.pi, np.pi, size=(N, T))
    if lens is not None:
        pad_ = np.arange(T)[None] >= (lens - 2)[:, None]
        if garbage:
            step[pad_] = rng.gamma(9.0, 5.0, size=pad_.sum())
        else:
            step[pad_], turn[pad_] = np.nan, np.nan
    return KinFeatures(step, turn)


def context(N, T, K, lens=None, seed=5, garbage=False):
    rng = np.random.default_rng(seed)
    w = rng.uniform(size=(N, T)) * (rng.uniform(size=(N, T)) < 0.6)
    c = rng.dirichlet(np.ones(K), size=(N, T))
    if lens is not None and not garbage:
        w = np.where(np.arange(T)[None] < (lens - 2)[:, None], w, 0.0)
    return w, c


def _setup(S=10, A=9, K=3, seed=0, lens=LENS):
    T = rand_mdp(S, A, seed)
    seqs = rollout_var(T, lens, seed + 1)
    base = sticky_init(K, S, seed)
    _, R2 = init_rewards(K, S, A, seed)
    return T, seqs, base, R2


def _trim(arr, n):
    return np.asarray(arr)[:n]


# ---------------------------------------------------------------------------
# V1 / V2  data layer
# ---------------------------------------------------------------------------

def test_v1_build_dataset():
    rng = np.random.default_rng(0)
    segs = []
    for g, L in (("a", 4), ("a", 23), ("b", 2), ("b", 61), ("c", 12)):
        s = rng.integers(10, size=L)
        segs.append(dict(group=g, key=g, seg=len(segs), states=s, actions=rng.integers(9, size=L - 1)))
    segs[3]["states"][30] = -1                                # invalid state: split into runs 0..29, 31..60
    st, ac, gr, L, meta = build_dataset(segs, 12, "variable", verbose=False, return_meta=True, min_seq_len=4)
    lens = np.array([m["length"] for m in meta])
    assert L == lens.max() <= 12 and lens.min() >= 4
    assert st.shape == (len(lens), L) and ac.shape == (len(lens), L - 1)
    for i, m in enumerate(meta):                              # rows = the original states, padding ff
        src = segs[m["seg"]]
        n = lens[i]
        assert np.array_equal(st[i, :n], src["states"][m["start"]:m["start"] + n])
        assert np.array_equal(ac[i, :n - 1], src["actions"][m["start"]:m["start"] + n - 1])
        assert np.all(st[i, n:] == st[i, n - 1]) and np.all(ac[i, n - 1:] == 0)
    # every transition of every run >= 4 states is kept exactly once (pieces share their boundary state)
    kept = sum(int(n) - 1 for n in lens)
    assert kept == 3 + 22 + 29 + 29 + 11, kept
    # 23 states, max 12: 2 pieces of 12 states (11 + 11 transitions); 30 states: 3 pieces of 10 / 11
    assert sorted(lens[[m["seg"] == 1 for m in meta]]) == [12, 12]
    # equal-length runs (= seq_len) -> identical to chunk
    segs2 = [dict(group="a", key="a", seg=i, states=rng.integers(10, size=9), actions=rng.integers(9, size=8))
             for i in range(4)]
    v = build_dataset(segs2, 9, "variable", verbose=False, return_meta=True)
    c = build_dataset(segs2, 9, "chunk", verbose=False, return_meta=True)
    assert np.array_equal(v[0], c[0]) and np.array_equal(v[1], c[1]) and v[3] == c[3]
    assert [m["start"] for m in v[4]] == [m["start"] for m in c[4]]


def test_v2_align_and_mask():
    T, seqs, _, _ = _setup()
    rows, acts, lens = pad(seqs)
    x, xp, a = aligned(rows, acts, lens)
    d = dict(states=rows, actions=acts, lengths=lens, variable=True)
    m = step_mask(d, 1)
    assert np.array_equal(m, a >= 0) and np.array_equal(m.sum(1), lens - 2)
    for i, (s, ac) in enumerate(seqs):
        n = lens[i] - 2
        assert np.array_equal(x[i, :n], s[1:-1]) and np.array_equal(xp[i, :n], s[:-2])
        assert np.array_equal(a[i, :n], ac[1:])
        assert np.all(a[i, n:] == PAD_ACTION) and np.all(x[i, n:] == x[i, 0]) and np.all(xp[i, n:] == xp[i, 0])


# ---------------------------------------------------------------------------
# V3 / V4  SWIRL
# ---------------------------------------------------------------------------

def _swirl_fit(T, xo, ao, base, R2, kin=None, km=None, ctx=None, iters=3):
    return em_train_ext(jnp.asarray(base[0]), jnp.asarray(base[1]), tuple(jnp.asarray(w) for w in base[2]),
                        jnp.asarray(R2)[:, None], jnp.asarray(T), xo, ao, kin=kin, kin_model=km, ctx=ctx,
                        alpha=0.1, fit_alpha=ctx is not None, iters=iters)


def _km(K, kin):
    km = KinematicEmissions(K)
    km.init_quantiles(kin, np.random.default_rng(0))
    return km


def test_v3_swirl_padded_equals_per_sequence():
    T, seqs, base, R2 = _setup()
    S, A = T.shape[:2]
    K = R2.shape[0]
    rows, acts, lens = pad(seqs)
    x, xp, a = aligned(rows, acts, lens)
    N, Tm = x.shape
    xo, ao = onehots(x, a, S, A)
    kin = kin_feats(N, Tm, lens)
    w, c = context(N, Tm, K, lens)
    r = _swirl_fit(T, xo, ao, base, R2, kin=kin, km=_km(K, kin), ctx=(w, c))
    e = swirl_eval(r, jnp.asarray(T), xo, ao, kin=kin, ctx=(w, c))
    m = a >= 0
    assert e["n_steps"] == int(m.sum())
    tot = 0.0
    for i, (s, ac) in enumerate(seqs):
        n = lens[i] - 2
        xi, ai = s[None, 1:-1], ac[None, 1:]
        xoi, aoi = onehots(xi, ai, S, A)
        ki = KinFeatures(kin.step[i:i + 1, :n], kin.turn[i:i + 1, :n])
        ei = swirl_eval(r, jnp.asarray(T), xoi, aoi, kin=ki, ctx=(w[i:i + 1, :n], c[i:i + 1, :n]))
        tot += ei["LL_total"]
        assert np.abs(ei["step_ll"][0] - e["step_ll"][i, :n]).max() < TOL
        assert np.abs(ei["gamma"][0] - e["gamma"][i, :n]).max() < TOL
        assert np.array_equal(ei["path"][0], e["path"][i, :n]) and np.all(e["path"][i, n:] == -1)
        assert np.all(e["step_ll"][i, n:] == 0) and np.all(e["gamma"][i, n:] == 0)
    assert abs(tot - e["LL_total"]) < 1e-5 and abs(e["LL"] - tot / m.sum()) < 1e-8


def test_v3b_learnt_ll():
    from naturenet.models.irl.swirl_training_top_level import learnt_LL1
    T, seqs, base, R2 = _setup()
    S, A = T.shape[:2]
    K = R2.shape[0]
    rows, acts, lens = pad(seqs)
    x, _, a = aligned(rows, acts, lens)
    xo, ao = onehots(x, a, S, A)
    R = jnp.asarray(R2)[:, None]
    temps = jnp.ones(K)
    ll, path = learnt_LL1(jnp.asarray(base[0]), jnp.asarray(base[1]), base[2], R, temps, S, A, jnp.asarray(T), xo, ao)
    tot, n_tot = 0.0, 0
    for i, (s, ac) in enumerate(seqs):
        n = lens[i] - 2
        xoi, aoi = onehots(s[None, 1:-1], ac[None, 1:], S, A)
        lli, pi = learnt_LL1(jnp.asarray(base[0]), jnp.asarray(base[1]), base[2], R, temps, S, A, jnp.asarray(T),
                             xoi, aoi)
        tot += float(lli) * n
        n_tot += n
        assert np.array_equal(np.asarray(pi)[0], np.asarray(path)[i, :n]) and np.all(np.asarray(path)[i, n:] == -1)
    assert abs(float(ll) - tot / n_tot) < 1e-8


def test_v4_swirl_em_ignores_padding():
    T, seqs, base, R2 = _setup()
    S, A = T.shape[:2]
    K = R2.shape[0]
    outs = []
    for g in (None, 7):
        rows, acts, lens = pad(seqs, garbage=g)
        x, xp, a = aligned(rows, acts, lens)
        if g is not None:                                     # garbage also in the aligned padded steps
            pm = a < 0
            x = np.where(pm, np.random.default_rng(g).integers(S, size=x.shape), x)
        N, Tm = x.shape
        xo, ao = onehots(x, a, S, A)
        kin = kin_feats(N, Tm, lens, garbage=g is not None)
        w, c = context(N, Tm, K, lens, garbage=g is not None)
        km = _km(K, kin_feats(N, Tm, lens))                   # same initial kinematic model
        r = _swirl_fit(T, xo, ao, base, R2, kin=kin, km=km, ctx=(w, c), iters=4)
        outs.append(r)
    assert np.allclose(outs[0]["LL_list"], outs[1]["LL_list"], rtol=0, atol=1e-6), [o["LL_list"] for o in outs]
    assert np.abs(np.asarray(outs[0]["rewards"]) - np.asarray(outs[1]["rewards"])).max() < 1e-6
    assert abs(outs[0]["alpha"] - outs[1]["alpha"]) < 1e-6


# ---------------------------------------------------------------------------
# V5 / V6  HIAVI and coupled HIAVI
# ---------------------------------------------------------------------------

def _hiavi_data(seqs, garbage=None, S=10):
    rows, acts, lens = pad(seqs, garbage=garbage)
    x, _, a = aligned(rows, acts, lens)
    if garbage is not None:
        x = np.where(a < 0, np.random.default_rng(garbage).integers(S, size=x.shape), x)
    return x, a, lens


def test_v5_hiavi():
    from naturenet.models.irl.hiavi_model import HIAVI
    T, seqs, _, _ = _setup()
    K = 3
    x, a, lens = _hiavi_data(seqs)
    hm = HIAVI(K, T, seed=2, tol=0.0, max_iters=5).fit(x, a)
    g, z = hm.posteriors(x, a), hm.decode(x, a)
    tot = 0.0
    for i, (s, ac) in enumerate(seqs):
        n = lens[i] - 2
        xi, ai = s[None, 1:-1], ac[None, 1:]
        tot += hm.loglik(xi, ai) * n
        assert np.abs(hm.posteriors(xi, ai)[0] - g[i, :n]).max() < TOL and np.all(g[i, n:] == 0)
        assert np.array_equal(hm.decode(xi, ai)[0], z[i, :n]) and np.all(z[i, n:] == -1)
    assert abs(hm.loglik(x, a) - tot / (a >= 0).sum()) < 1e-9
    xg, ag, _ = _hiavi_data(seqs, garbage=4)
    hg = HIAVI(K, T, seed=2, tol=0.0, max_iters=5).fit(xg, ag)
    assert np.allclose(hm.ll_history, hg.ll_history, rtol=0, atol=1e-10)
    assert np.abs(hm.pis - hg.pis).max() < 1e-10


def test_v6_coupled_hiavi():
    from naturenet.models.irl.hiavi_model import HIAVI
    from naturenet.models.irl.hiavi_multiagent import CoupledHIAVI
    T, seqs, _, _ = _setup()
    K = 3
    x, a, lens = _hiavi_data(seqs)
    N, Tm = x.shape
    w, c = context(N, Tm, K, lens)
    m0 = HIAVI(K, T, seed=2, tol=0.0, max_iters=3).fit(x, a)
    cm = CoupledHIAVI(K, T, alpha0=0.2, seed=2, tol=0.0, max_iters=4)
    cm.init_from(m0)
    cm.fit_em(x, a, w, c)
    z = cm.decode(x, a, w, c)
    tot = 0.0
    for i, (s, ac) in enumerate(seqs):
        n = lens[i] - 2
        xi, ai = s[None, 1:-1], ac[None, 1:]
        tot += cm.loglik(xi, ai, w[i:i + 1, :n], c[i:i + 1, :n]) * n
        assert np.array_equal(cm.decode(xi, ai, w[i:i + 1, :n], c[i:i + 1, :n])[0], z[i, :n])
        assert np.all(z[i, n:] == -1)
    assert abs(cm.loglik(x, a, w, c) - tot / (a >= 0).sum()) < 1e-9
    # EM invariant to padded content (states, actions, context)
    xg, ag, _ = _hiavi_data(seqs, garbage=4)
    wg, cg = context(N, Tm, K, lens, garbage=True)
    cg_ = CoupledHIAVI(K, T, alpha0=0.2, seed=2, tol=0.0, max_iters=4)
    cg_.init_from(HIAVI(K, T, seed=2, tol=0.0, max_iters=3).fit(xg, ag))
    cg_.fit_em(xg, ag, np.where(a >= 0, w, wg), cg)
    assert abs(cg_.alpha - cm.alpha) < 1e-8 and np.abs(cg_.pis - cm.pis).max() < 1e-8


# ---------------------------------------------------------------------------
# V7  decision scales
# ---------------------------------------------------------------------------

def _all_on(emission="swirl"):
    return dict(order=2, reward_l2=0.0, emission=emission,
                switch_history=dict(kind=["lags", "decay", "summary"], lags=1, decay_steps=[3], window=4),
                durations=dict(max_duration=6, l2=0.0, hazard_features=True),
                hierarchy=dict(n_regimes=2, init_noise=0.3),
                coupling=dict(enabled=True, alpha0=0.2, beta0=0.5))


def _scales_setup(emission, garbage=None):
    from naturenet.models.irl.hiavi_model import HIAVI
    T, seqs, base, R2 = _setup()
    S, A = T.shape[:2]
    K = R2.shape[0]
    rows, acts, lens = pad(seqs, garbage=garbage)
    _, _, a = aligned(rows, acts, lens)
    cfg = sc.ScalesConfig.from_dict(_all_on(emission))
    tree = sc.ContextTree.build(rows, S, T, order=cfg.order, lengths=lens)
    if emission == "hiql":
        lp0, lP, lpis = HIAVI(K, np.ones((S, A, S)) / S, seed=0)._init()
        base, R2 = (lp0, lP, None), lpis
    d = sc.scales_data(rows, a, tree, cfg.history, S)
    p = sc.init_params(cfg, tree, base, R2, d["phi"].shape[-1])
    N, Tm = a.shape
    w, c = context(N, Tm, K, lens, garbage=garbage is not None)
    return T, seqs, rows, a, lens, cfg, tree, d, p, w, c


def test_v7_scales_padded_equals_per_sequence():
    for emission in ("swirl", "hiql"):
        T, seqs, rows, a, lens, cfg, tree, d, p, w, c = _scales_setup(emission)
        S = T.shape[0]
        r = sc.em_train_scales(p, cfg, tree, sc.with_context(d, w, c), iters=3)
        e = sc.scales_eval(r, tree, sc.with_context(d, w, c))
        assert e["n_steps"] == int((a >= 0).sum())
        tot = 0.0
        for i, (s, ac) in enumerate(seqs):
            n = lens[i] - 2
            di = sc.scales_data(s[None], ac[None, 1:], tree, cfg.history, S)
            assert np.array_equal(np.asarray(di["ctx"])[0], np.asarray(d["ctx"])[i, :n])
            assert np.abs(np.asarray(di["phi"])[0] - np.asarray(d["phi"])[i, :n]).max() < 1e-12
            ei = sc.scales_eval(r, tree, sc.with_context(di, w[i:i + 1, :n], c[i:i + 1, :n]))
            tot += ei["LL_total"]
            assert np.abs(ei["step_ll"][0] - e["step_ll"][i, :n]).max() < TOL, emission
            assert np.abs(ei["gamma"][0] - e["gamma"][i, :n]).max() < TOL, emission
            assert np.array_equal(ei["path"][0], e["path"][i, :n]) and np.all(e["path"][i, n:] == -1)
            assert np.all(e["gamma"][i, n:] == 0) and np.all(e["step_ll"][i, n:] == 0)
        assert abs(tot - e["LL_total"]) < 1e-5, (emission, tot, e["LL_total"])
        assert np.all(np.diff(r["LL_list"]) > -1e-6), r["LL_list"]


def test_v7b_scales_em_ignores_padding():
    for emission in ("swirl", "hiql"):
        outs = []
        for g in (None, 9):
            T, seqs, rows, a, lens, cfg, tree, d, p, w, c = _scales_setup(emission, garbage=g)
            if g is not None:                                 # same tree / init as the clean run
                _, _, _, _, _, _, tree, _, p, _, _ = _scales_setup(emission)
                d = sc.scales_data(rows, a, tree, cfg.history, T.shape[0])
            outs.append(sc.em_train_scales(p, cfg, tree, sc.with_context(d, w, c), iters=3))
        assert np.allclose(outs[0]["LL_list"], outs[1]["LL_list"], rtol=0, atol=1e-6), emission


def test_v7c_context_tree_lengths():
    """Windows reaching into the padding are not counted: nodes == union over the unpadded sequences."""
    T, seqs, _, _ = _setup()
    S = T.shape[0]
    rows, _, lens = pad(seqs)
    tv = sc.ContextTree.build(rows, S, T, order=3, lengths=lens)
    nodes = set()
    for s_, _ in seqs:
        nodes |= set(sc.ContextTree.build(s_[None], S, T, order=3).nodes)
    assert set(tv.nodes) == nodes
    tp = sc.ContextTree.build(rows, S, T, order=3)                # unmasked: picks up (s, s, s) pad windows
    assert set(tp.nodes) >= nodes


# ---------------------------------------------------------------------------
# V8 / V9
# ---------------------------------------------------------------------------

def test_v8_fixed_length_unchanged():
    T, seqs, base, R2 = _setup(lens=[20] * 6)
    S, A = T.shape[:2]
    rows, acts, lens = pad(seqs)
    x0, _, a0 = align_for_swirl(rows, acts)
    x1, _, a1 = align_for_swirl(rows, acts, lens)
    assert np.array_equal(x0, x1) and np.array_equal(a0, a1) and (a1 >= 0).all()
    xo = jnp.asarray(np.eye(S)[x0][:, :, None])
    ao = jnp.asarray(np.eye(A)[a0][:, :, None])
    r = _swirl_fit(T, xo, ao, base, R2)
    e = swirl_eval(r, jnp.asarray(T), xo, ao)
    assert e["n_steps"] == x0.size and abs(e["LL"] - e["LL_total"] / x0.size) < 1e-12


def test_v9_fixed_only_scripts_reject_variable():
    try:
        load_swirl_dataset(dict(segment_policy="variable"))
    except ValueError as err:
        assert "segment_policy: variable" in str(err)
    else:
        raise AssertionError("variable data must be rejected without allow_variable")


if __name__ == "__main__":
    import time
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        t0 = time.time()
        fn()
        print(f"{name}: OK ({time.time() - t0:.1f}s)", flush=True)
    print("all variable-length tests passed")
