"""
S-2 index emissions (s2_index_emissions: True) must reproduce the dense one-hot S-2 path.
    PYTHONPATH=src python -m naturenet.models.irl.test_s2_index_emissions
"""
import os
import tempfile
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

jax.config.update("jax_enable_x64", True)

from naturenet.models.irl import run_swirl_kfold as rk  # noqa: E402
from naturenet.models.irl import swirl_training as st  # noqa: E402
from naturenet.models.irl.swirl_utils import one_hota_partial, one_hotx_partial, one_hotx2_partial  # noqa: E402


def _toy(seed=1, N=30, L=25, S=8, A=9, K=2, H=3):
    r = np.random.default_rng(seed)
    states, actions = r.integers(0, S, (N, L + 1)), r.integers(0, A, (N, L))
    lengths = r.integers(6, L + 1, N)
    cnt = np.full((S, A, S), 1e-3)
    for n in range(N):
        for t in range(L):
            cnt[states[n, t], actions[n, t], states[n, t + 1]] += 1
    T = cnt / cnt.sum(-1, keepdims=True)
    x, xp, a = states[:, 1:-1], states[:, :-2], actions[:, 1:]
    m = np.arange(x.shape[1])[None] < (lengths[:, None] - 1)
    mj = jnp.asarray(m, dtype=jnp.float64)[:, :, None, None]
    arh = rk.ArhmmInit((jnp.log(jnp.ones(K) / K), jnp.log(jnp.full((K, K), .2) + jnp.eye(K) * .6),
                        (jnp.asarray(r.normal(size=(S, H)) * .1), jnp.zeros(H),
                         jnp.asarray(r.normal(size=(H, K)) * .1), jnp.zeros(K))))

    def data():
        return dict(x=x, x_prev=xp, a=np.where(m, a, -1), n_states=S, n_actions=A, step_mask=m, states=states,
                    xohs=vmap(partial(one_hotx_partial, n_states=S))(jnp.asarray(x)) * mj,
                    aohs=vmap(partial(one_hota_partial, n_actions=A))(jnp.asarray(np.maximum(a, 0))) * mj,
                    groups=np.arange(N), lengths=lengths,
                    meta=[{"key": str(i), "seg": 0, "start": 0, "length": int(lengths[i])} for i in range(N)])
    return T, data, arh, K, S, A


def test_comp_ll_and_mask():
    r = np.random.default_rng(0)
    N, T, K, S, A = 6, 12, 3, 20, 9
    lg = jnp.asarray(r.normal(size=(K, S, A)))
    xi, ai = r.integers(0, S, (N, T)), r.integers(0, A, (N, T))
    m = np.arange(T)[None] < r.integers(3, T + 1, N)[:, None]
    xo = jax.nn.one_hot(xi, S)[:, :, None, :] * m[:, :, None, None]
    ao = jax.nn.one_hot(ai, A)[:, :, None, :] * m[:, :, None, None]
    idx = jnp.asarray(np.where(m, xi, -1).astype(np.int32))[:, :, None]
    d = vmap(partial(st.comp_ll_jax, lg))(xo, ao)
    i = vmap(partial(st.comp_ll_jax, lg))(idx, ao)
    assert float(jnp.abs(d - i).max()) == 0.0
    assert (np.asarray(st.oh_mask(xo)) == np.asarray(st.oh_mask(idx))).all()
    assert st.n_real(xo) == st.n_real(idx) == m.sum()


def test_fit_swirl_final_dense_equals_index():
    T, data, arh, K, S, A = _toy()
    res = {}
    for mode in (False, True):
        out = tempfile.mkdtemp()
        conf = dict(out_dir=out, run_uid="t", seed=0, n_hidden=K, n_hidden_init=K, n_states=S, epochs=3, epochs_2=2,
                    run_s2=True, s2_index_emissions=mode, temps_final=[0.5, 1.0])
        rk.fit_swirl_final(conf, data(), T, np.arange(20), np.arange(20, 30), arh, "_KF_0")
        res[mode] = {f: np.load(os.path.join(out, f)) for f in os.listdir(out) if "LL" in f}
    assert len(res[True]) == 4
    for f in res[True]:
        assert abs(float(res[False][f]["LL"]) - float(res[True][f]["LL"])) < 1e-8, f
        assert (res[False][f]["jax_path_vmap"] == res[True][f]["jax_path_vmap"]).all(), f


if __name__ == "__main__":
    test_comp_ll_and_mask()
    print("test_comp_ll_and_mask: OK")
    test_fit_swirl_final_dense_equals_index()
    print("test_fit_swirl_final_dense_equals_index: OK")
