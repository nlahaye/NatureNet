"""
ARHMM action re-scoring, ARHMM pseudo-count, and the K = 1 SWIRL temperature default. CPU, < 2 min:

    PYTHONPATH=src JAX_PLATFORMS=cpu NATURENET_JAX_PLATFORM=cpu python -m naturenet.models.irl.test_arhmm_action_ll

A1  action_logemit: rows are distributions over the 9 actions; deterministic T -> p(a|x) = p(x'_a | x) exactly;
    shared next cells split their mass across the actions that reach them; unreachable mass is dropped
A2  ll_from_logemit == the hand-computed forward LL (K = 1: mean log p(a_t | x_t) over the real steps)
A3  default_final_temps: K = 1 -> [1.0]; K >= 2 unchanged ([0.01, 1, ...]); temps_final overrides
A4  ssm ARCategoricalObservations pseudo-count: default 1e-5 unchanged; larger count raises the unseen-move floor;
    pseudo_support restricts it to the allowed moves
A5  run_arhmm end to end on a toy dataset: writes train_ll_act / test_ll_act, re-scoring an existing fold from the
    saved logits leaves the init file untouched and gives the same action LL; get_stats.pseudo_variant suffixes run_uid
"""
import os
import tempfile

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from naturenet.models.irl import arhmm_action_ll as aal  # noqa: E402
from naturenet.models.irl import run_swirl_kfold as rk  # noqa: E402
from naturenet.models.irl.swirl_utils import one_hota_partial, one_hotx_partial  # noqa: E402


def _det_T(S=12, A=9, seed=0):
    """Deterministic T: action a moves x to nb[x, a] (distinct cells)."""
    r = np.random.default_rng(seed)
    T = np.zeros((S, A, S))
    nb = np.stack([r.choice(S, A, replace=False) for _ in range(S)])
    for x in range(S):
        T[x, np.arange(A), nb[x]] = 1.0
    return T, nb


def test_a1_action_logemit():
    S, A, K = 12, 9, 2
    T, nb = _det_T(S, A)
    r = np.random.default_rng(1)
    lg = r.normal(size=(K, S, S))
    lg -= np.logaddexp.reduce(lg, -1, keepdims=True)
    le = aal.action_logemit(lg, T)
    assert le.shape == (K, S, A)
    assert np.allclose(np.exp(le).sum(-1), 1.0)
    P = np.exp(lg)
    for k in range(K):
        for x in range(S):
            want = P[k, x, nb[x]] / P[k, x, nb[x]].sum()       # mass on the 3 unreachable cells dropped
            assert np.allclose(np.exp(le[k, x]), want, atol=1e-9)
    # two actions to the same cell share its mass equally
    T2 = T.copy()
    T2[0, 1] = T2[0, 0]
    le2 = aal.action_logemit(lg, T2)
    p = np.exp(le2[0, 0])
    c0, c1 = P[0, 0, nb[0, 0]], P[0, 0, nb[0, 1]]
    tot = P[0, 0, nb[0]].sum() - c1                            # nb[0, 1] no longer reachable
    assert np.isclose(p[0], c0 / 2 / tot) and np.isclose(p[1], c0 / 2 / tot)
    # nothing reachable -> uniform
    T3 = T.copy()
    T3[5] = 0.0
    assert np.allclose(np.exp(aal.action_logemit(lg, T3)[:, 5]), 1.0 / A)


def _onehots(x, a, m, S, A):
    mj = jnp.asarray(m, dtype=jnp.float64)[:, :, None, None]
    xo = jax.vmap(lambda v: one_hotx_partial(v, n_states=S))(jnp.asarray(x)) * mj
    ao = jax.vmap(lambda v: one_hota_partial(v, n_actions=A))(jnp.asarray(np.maximum(a, 0))) * mj
    return xo, ao


def test_a2_ll_matches_hand():
    S, A, N, L, H = 12, 9, 6, 15, 3
    T, _ = _det_T(S, A)
    r = np.random.default_rng(2)
    lg = r.normal(size=(1, S, S))
    lg -= np.logaddexp.reduce(lg, -1, keepdims=True)
    le = aal.action_logemit(lg, T)
    x, a = r.integers(0, S, (N, L)), r.integers(0, A, (N, L))
    m = np.arange(L)[None] < r.integers(3, L + 1, N)[:, None]
    xo, ao = _onehots(x, a, m, S, A)
    Rs = (jnp.zeros((S, H)), jnp.zeros(H), jnp.zeros((H, 1)), jnp.zeros(1))
    ll, path = aal.ll_from_logemit(jnp.zeros(1), jnp.zeros((1, 1)), Rs, le, xo, ao)
    hand = np.mean(le[0][x[m], a[m]])
    assert np.isclose(float(ll), hand, atol=1e-8), (float(ll), hand)


def test_a3_default_temps():
    assert rk.default_final_temps(1) == [1.0]
    assert rk.default_final_temps(3) == [0.01, 1.0, 1.0]
    assert list(rk._temps({}, "temps_final", 1, rk.default_final_temps(1))) == [1.0]
    assert list(rk._temps({"temps_final": 1.0}, "temps_final", 3, rk.default_final_temps(3))) == [1.0] * 3


def _ssm_fit(pc, support=None, S=6, seed=0):
    import numpy.random as npr
    from ssm.observations import ARCategoricalObservations as AR
    AR.pseudo_count, AR.pseudo_support, AR.pseudo_floor = pc, support, 1e-5
    npr.seed(seed)
    o = AR(1, 1, C=S)
    x = np.array([[0], [1], [2], [0], [1], [2], [0], [1]])         # only 0 -> 1 -> 2 -> 0 observed
    Ez = np.ones((len(x), 1))
    o.m_step([(Ez, None, None)], [x], [None], [np.ones_like(x, bool)], [None])
    AR.pseudo_count, AR.pseudo_support = 1e-5, None
    return np.exp(o.logits[0, 0])


def test_a4_pseudo_count():
    p_def = _ssm_fit(1e-5)
    assert np.isclose(p_def[0, 1], 1.0, atol=1e-4) and p_def[0, 3] < 1e-5   # default = old behaviour
    p1 = _ssm_fit(1.0)
    assert np.isclose(p1[0, 3], 1.0 / (3 + 6))                              # 3 counts + 6 pseudo
    sup = np.zeros((6, 6), bool)
    sup[0, [1, 3]] = True
    p_s = _ssm_fit(1.0, sup)
    assert np.isclose(p_s[0, 3], 0.2, atol=1e-4) and p_s[0, 4] < 1e-4                             # only allowed moves raised
    assert np.allclose(p_s.sum(-1), 1.0)


def _toy_dataset(S=10, A=9, n_animals=5, per=6, L=20, seed=3):
    r = np.random.default_rng(seed)
    T, _ = _det_T(S, A, seed)
    states, actions, groups, segs = [], [], [], []
    for g in range(n_animals):
        pref = r.dirichlet(np.ones(A) * 0.5)
        for j in range(per):
            s = r.integers(S)
            st, ac = [s], []
            for _ in range(L - 1):
                a = r.choice(A, p=pref)
                s = int(np.argmax(T[s, a]))
                st.append(s)
                ac.append(a)
            states.append(st)
            actions.append(ac)
            groups.append(f"an{g}")
            segs.append(dict(group=f"an{g}", key=f"an{g}", seg=len(segs), states=np.array(st), actions=np.array(ac)))
    return dict(states=np.array(states), actions=np.array(actions), groups=np.array(groups), segments=segs)


def test_a5_run_arhmm_end_to_end():
    from naturenet.models.irl import run_arhmm_kfold as ra
    from naturenet.models.irl.get_stats import pseudo_variant
    out = tempfile.mkdtemp()
    conf = dict(out_dir=out, run_uid="toy", seed=7, n_hidden=2, n_hidden_init=2, emission_dim=1, latent_state_dim=1,
                n_states=10, n_folds=5, fold_seed=0, arhmm_epochs=5, per_fold_trans_prob=True,
                require_full_trans_prob=False, arhmm_pseudo_count=0.1)
    data = _toy_dataset()
    ra.run_arhmm(conf, data=data, folds_to_run=[0])
    f_ll = ra.arhmm_ll_fname(out, "toy", 2, 7, 0)
    f_init = ra.arhmm_fname(out, "toy", 2, 7, 0)
    with np.load(f_ll, allow_pickle=True) as f:
        tr, te = float(f["train_ll_act"]), float(f["test_ll_act"])
        assert np.isfinite(tr) and np.isfinite(te) and -np.log(9) - 1.0 < tr < 0.0, (tr, te)
    with np.load(ra.arhmm_logits_fname(out, "toy", 2, 7, 0)) as f:
        assert float(f["pseudo_count"]) == 0.1
    # drop the action LLs -> re-run re-scores from the saved logits; init file untouched, same numbers
    with np.load(f_ll, allow_pickle=True) as f:
        keep = {k: f[k] for k in f.files if not k.endswith("_act")}
    np.savez(f_ll, **keep)
    mtime = os.path.getmtime(f_init)
    ra.run_arhmm(conf, data=_toy_dataset(), folds_to_run=[0])
    assert os.path.getmtime(f_init) == mtime
    with np.load(f_ll, allow_pickle=True) as f:
        assert np.isclose(float(f["train_ll_act"]), tr) and np.isclose(float(f["test_ll_act"]), te)
    # pseudo-count variants get their own run_uid
    v = pseudo_variant(conf, 1.0, "T")
    assert v["run_uid"] == "toy_pc1T" and v["arhmm_pseudo_support"] == "T" and conf["run_uid"] == "toy"
    assert pseudo_variant(conf) is conf


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
