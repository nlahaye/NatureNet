"""
Synthetic checks of the partner-coupling forms (mixture / align / matrix) in hiavi_multiagent and swirl_ext.
Run: python -m naturenet.models.irl.test_coupling_forms   (CPU; a few seconds; JAX part skipped if JAX missing)
"""
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scipy.special import logsumexp

from naturenet.models.irl import hiavi_multiagent as hm
from naturenet.models.irl import run_hiavi_multiagent as rm

rng = np.random.default_rng(0)
K, N, T = 3, 40, 60


def _sim(logp_tr, B, w, c, rng):
    """latent chains under softmax(log Lambda + w B c) -> one-hot xi (N, T-1, K, K)."""
    lP = hm.loglinear_log_transitions(logp_tr, B, w, c)
    z = np.zeros((N, T), int)
    z[:, 0] = rng.integers(K, size=N)
    for t in range(1, T):
        p = np.exp(lP[np.arange(N), t - 1, z[:, t - 1]])
        z[:, t] = [rng.choice(K, p=pp / pp.sum()) for pp in p]
    xi = np.zeros((N, T - 1, K, K))
    xi[np.arange(N)[:, None], np.arange(T - 1)[None], z[:, :-1], z[:, 1:]] = 1.0
    return xi


L0 = np.log(np.full((K, K), 0.1) + 0.7 * np.eye(K))
L0 = L0 - logsumexp(L0, axis=1, keepdims=True)
w = (rng.random((N, T)) < 0.6) * rng.random((N, T))
c = rng.dirichlet(np.ones(K), size=(N, T))

# 1. B = 0 is the independent model; mixture with alpha = 0 is too
lP = hm.loglinear_log_transitions(L0, np.zeros((K, K)), w, c)
assert np.allclose(lP, L0[None, None]), "B = 0 must give log Lambda"
assert np.allclose(hm.coupled_log_transitions(L0, 0.0, w, c), L0[None, None])
print("1 B=0 / alpha=0 identity OK")

# 2. analytic gradient of the log-linear M-step objective (re-built here, same formula as loglinear_mstep)
xi = _sim(L0, 2.0 * np.eye(K), w, c, rng)


def obj(th, form):
    L = th[:K * K].reshape(K, K)
    Bm = th[-1] * np.eye(K) if form == "align" else th[K * K:].reshape(K, K)
    lPx = hm.loglinear_log_transitions(L, Bm, w, c) - 0.0
    lPx = lPx - logsumexp(lPx, axis=-1, keepdims=True)
    pen = 0.5 * (th[-1] ** 2 if form == "align" else (Bm ** 2).sum())
    return -((xi * lPx).sum() - pen) / xi.sum()


for form in ("align", "matrix"):
    th = rng.normal(size=K * K + (1 if form == "align" else K * K))
    num = lambda x: obj(x, form)
    # numeric gradient of obj vs the analytic gradient inside loglinear_mstep: compare optima instead
    Lh, Bh = hm.loglinear_mstep(L0, np.zeros((K, K)), xi, w, c, l2=1.0, form=form, pseudocount=0.0, maxiter=500)
    thh = np.concatenate([Lh.ravel(), [Bh[0, 0]] if form == "align" else Bh.ravel()])
    g = np.array([(num(thh + e) - num(thh - e)) / 2e-5 for e in 1e-5 * np.eye(thh.size)])
    # Lh is row-normalised and Bh column-centred (invariances), so the gradient at the optimum must vanish
    assert np.abs(g).max() < 2e-3, (form, np.abs(g).max())
print("2 log-linear M-step reaches a stationary point of the penalised objective OK")

# 3. recovery: align beta and matrix B
for beta in (0.0, 1.5, 3.0):
    xi = _sim(L0, beta * np.eye(K), w, c, rng)
    _, Bh = hm.loglinear_mstep(L0, np.zeros((K, K)), xi, w, c, l2=0.1, form="align", maxiter=500)
    print(f"   beta {beta}: fitted {Bh[0, 0]:.2f}")
    assert abs(Bh[0, 0] - beta) < 0.6, (beta, Bh[0, 0])
Btrue = np.array([[2.0, -1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 2.5]])
Btrue = Btrue - Btrue.mean(axis=0, keepdims=True)
xi = _sim(L0, Btrue, w, c, rng)
_, Bh = hm.loglinear_mstep(L0, np.zeros((K, K)), xi, w, c, l2=0.1, form="matrix", maxiter=500)
err = np.abs(Bh - Btrue).max()
print(f"   matrix B max error {err:.2f}")
assert err < 1.2 and np.corrcoef(Bh.ravel(), Btrue.ravel())[0, 1] > 0.8
print("3 coupling recovery OK")

# 4. CoupledHIAVI: log-linear EM on simulated actions improves on the independent fit
S, A = 6, 3
T_env = rng.dirichlet(np.ones(S), size=(S, A))
pis = rng.dirichlet(0.3 * np.ones(A), size=(K, S))
xi = _sim(L0, 3.0 * np.eye(K), w, c, rng)
z = np.concatenate([xi.sum(-1).argmax(-1), xi[:, -1].sum(-2).argmax(-1)[:, None]], axis=1)
s = rng.integers(S, size=(N, T))
a = np.array([[rng.choice(A, p=pis[z[n, t], s[n, t]]) for t in range(T)] for n in range(N)])
base = hm.HIAVI(K, T_env, max_iters=30, seed=1)
base.logp_init, base.logp_tr, base._logpis_em = base._init()
base.pis = np.exp(base._logpis_em)
res = {}
for form in ("mixture", "align", "matrix"):
    m = hm.CoupledHIAVI(K, T_env, alpha0=0.0 if form == "mixture" else 0.1, coupling=form, coupling_l2=0.1,
                        max_iters=60, seed=1).init_from(base)
    m.fit_em(s, a, w, c)
    res[form] = (m.ll_history[-1], m.coupling_stats())
    print("  ", form, round(res[form][0], 4), res[form][1])
    assert np.all(np.diff(m.ll_history) > -1e-6), f"{form}: EM LL must not decrease"
print("4 CoupledHIAVI log-linear EM monotone OK")

# 5. lag_days shifts the kernel
times = np.tile(np.arange(10.0), (2, 1))
pc = hm.PartnerContext(times, np.zeros((2, 10)), np.zeros((2, 10)), np.array(["a", "b"]), np.array(["a", "b"]))
k0 = pc.build(np.array([0]), np.array([1]), h_days=0.5, max_dt_days=0.1)
k2 = pc.build(np.array([0]), np.array([1]), h_days=0.5, max_dt_days=0.1, lag_days=2.0)
M0, M2 = np.asarray(k0["K"].todense()), np.asarray(k2["K"].todense())
assert np.allclose(np.argmax(M0[:, :], axis=1), np.arange(10))
assert all(M2[t, t - 2] > 0 for t in range(2, 10)) and M2[:2].sum() == 0
print("5 lag_days OK")

# 6. variant grid naming / split and the selection-aware test
vs = rm.expand_variant_grid({"kin": {"h_days": 1.0}}, [{"base": "kin", "coupling": ["align", "matrix"],
                                                        "lag_days": [0, 1]}])
print("  ", sorted(vs))
assert len(vs) == 5 and all(k in vs for k in ("kin", "kin__coupling-align__lag_days-1"))
kv, mv = rm.split_variant(vs["kin__coupling-matrix__lag_days-0"])
assert mv == {"coupling": "matrix"} and kv["max_dt_days"] == 3.0
try:
    rm.split_variant({"bogus": 1})
    raise AssertionError("unknown keys must fail")
except ValueError:
    pass
rows = [dict(variant=v, null_id=-1, d_total=g) for v, g in (("a", 5.0), ("b", 1.0))]
rows += [dict(variant=v, null_id=i, d_total=x) for i in range(20) for v, x in (("a", rng.normal()), ("b", rng.normal()))]
st = rm.selection_test(pd.DataFrame(rows), ["a", "b"])
print("  ", st)
assert st["best_variant"] == "a" and st["p_gain_max"] < 0.1
print("6 variant grid / selection test OK")

# 7. SWIRL forms (JAX): zero coupling = independent; numpy and JAX log-linear transitions agree; M-step recovers beta
try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from naturenet.models.irl import swirl_ext as se
except Exception as e:                                                         # pragma: no cover
    print("7 skipped (JAX not available):", e)
else:
    P = np.exp(L0)[None].repeat(T - 1, 0)
    for form in ("mixture", "align", "matrix"):
        th = se.coupling_theta(form, K)
        assert np.allclose(np.asarray(se._couple(jnp.asarray(P), jnp.asarray(w[0]), jnp.asarray(c[0]), th, form)), P)
    Bm = rng.normal(size=(K, K))
    a_ = np.asarray(se._couple(jnp.asarray(P), jnp.asarray(w[0]), jnp.asarray(c[0]), jnp.asarray(Bm.ravel()), "matrix"))
    b_ = np.exp(hm.loglinear_log_transitions(L0, Bm, w[:1], c[:1])[0])
    assert np.allclose(a_, b_, atol=1e-8)
    # M-step with a zero MLP (single state) on xi simulated under beta = 2.5
    Sx, H = 2, 2
    xo = np.zeros((N, T, 1, Sx)); xo[..., 0, 0] = 1.0
    Rs = (jnp.zeros((Sx, H)), jnp.zeros(H), jnp.zeros((H, K)), jnp.zeros(K))
    xi = _sim(L0, 2.5 * np.eye(K), w, c, rng)
    lp, R, th = se.coupled_trans_m_step(jnp.asarray(L0), Rs, 0.0, jnp.asarray(xi), jnp.asarray(xo), jnp.asarray(w),
                                        jnp.asarray(c), False, 500, form="align", theta=jnp.array([0.0]), l2=0.1)
    print(f"   SWIRL align beta fitted {float(th[0]):.2f} (true 2.5)")
    assert abs(float(th[0]) - 2.5) < 0.7
    _, _, thm = se.coupled_trans_m_step(jnp.asarray(L0), Rs, 0.0, jnp.asarray(xi), jnp.asarray(xo), jnp.asarray(w),
                                        jnp.asarray(c), False, 500, form="matrix", theta=jnp.zeros(K * K), l2=0.1)
    Bf = np.asarray(thm).reshape(K, K)
    print(f"   SWIRL matrix diag-mean {np.mean(np.diag(Bf)):.2f}")
    assert np.mean(np.diag(Bf)) > 1.0
    print("7 SWIRL coupling forms OK")
print("OK")

# 8. SWIRL EM end to end (JAX): log-linear coupling with B = 0 and no fit == independent; fitted forms run and
#    swirl_eval reads them back (coupling / B keys)
try:
    from naturenet.models.irl.test_swirl_scales import _context, _onehots, _setup
    from naturenet.models.irl.swirl_ext import em_train_ext, swirl_eval
except Exception as e:                                                         # pragma: no cover
    print("8 skipped:", e)
else:
    Tm, rows, acts, base, R2 = _setup()
    S_, A_ = Tm.shape[:2]
    N_, L_ = rows.shape
    K_ = R2.shape[0]
    xo, ao = _onehots(rows, acts, S_, A_)
    w8, c8 = _context(N_, L_, K_)
    args = (jnp.asarray(base[0]), jnp.asarray(base[1]), tuple(jnp.asarray(v) for v in base[2]),
            jnp.asarray(R2)[:, None], jnp.asarray(Tm), xo, ao)
    r_ind = em_train_ext(*args, iters=4)
    r_b0 = em_train_ext(*args, ctx=(w8, c8), coupling="matrix", fit_coupling=False, iters=4)
    assert np.allclose(r_ind["LL_list"], r_b0["LL_list"], atol=1e-8), (r_ind["LL_list"], r_b0["LL_list"])
    for form in ("align", "matrix"):
        r = em_train_ext(*args, ctx=(w8, c8), coupling=form, coupling_l2=1.0, iters=4)
        e = swirl_eval(r, jnp.asarray(Tm), xo, ao, ctx=(w8, c8), paths=True)
        assert r["coupling"] == form and np.asarray(r["B"]).shape == (K_, K_) and np.isfinite(e["LL"])
        print(f"   SWIRL EM {form}: LL trace {np.round(r['LL_list'], 4)}, align {np.mean(np.diag(r['B'])):.3f}")
    print("8 SWIRL EM log-linear coupling OK")
print("OK (all)")
