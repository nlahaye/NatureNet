"""
HIQL emission in the single-animal decision-scale runner (run_swirl_scales_kfold), CPU, < 1 min:

    PYTHONPATH=src JAX_PLATFORMS=cpu python -m naturenet.models.irl.test_scales_hiql_runner

R1  scales.emission hiql: variant tags match the SWIRL run (N1, N2, ...), hiql keys default to hiavi_*
R2  N1 (one restart, no early stop) == HIAVI.fit with the same seed (EM trace per step)
R3  N2 warm-starts from N1 (order-2 contexts, finite LL, IAVI rewards saved), N2 train LL >= N1 at init
R4  restarts: best training LL kept; CSV rows carry emission; mixing emissions in one CSV raises
R5  SWIRL configs are unchanged (tags, emission swirl)
"""
import os
import tempfile

import numpy as np

from naturenet.models.irl import run_swirl_scales_kfold as rk
from naturenet.models.irl import swirl_scales as sc
from naturenet.models.irl.hiavi_model import HIAVI
from naturenet.models.irl.run_hiavi_kfold import _seed
from naturenet.models.irl.test_swirl_scales import rand_mdp, rollout


def _data(S=8, A=9, N=20, L=30, seed=0):
    T = rand_mdp(S, A, seed)
    rows, a = rollout(T, N, L, seed + 1)
    return T, dict(states=rows, a=a, n_states=S, n_actions=A, groups=np.repeat(np.arange(5), N // 5))


def _conf(out, **kw):
    y = dict(seed=3, n_hidden=2, out_dir=out, run_uid="T_HIQL", hiavi_restarts=1, hiavi_max_iters=8,
             hiavi_pseudocount=0.5, hiavi_discount=0.9, hiavi_sticky=0.95,
             scales=dict(emission="hiql", hiql_tol=0.0, grid=dict(order=[1, 2]), hiql=dict(backoff=5.0)))
    y.update(kw)
    return y


def test_r1_variants():
    y = _conf("/tmp")
    v = rk.scales_variants(y)
    assert [t for t, _ in v] == ["N1", "N2"], v
    cfg = sc.ScalesConfig.from_dict(v[0][1])
    assert cfg.emission == "hiql" and cfg.hiql.pseudocount == 0.5 and cfg.hiql.backoff == 5.0
    assert cfg.transition_input == "none"


def test_r2_r3_r4():
    T, data = _data()
    tr, te = np.arange(16), np.arange(16, 20)
    with tempfile.TemporaryDirectory() as out:
        y = _conf(out)
        v = dict(rk.scales_variants(y))
        r1, tree1, row1 = rk.fit_variant_fold(y, data, T, tr, te, "N1", v["N1"], None, None, None, None, None,
                                              None, None, 0)
        x = data["states"][tr][:, 1:-1]
        hm = HIAVI(2, T, seed=_seed(3, 0, 0), sticky=0.95, pseudocount=0.5, tol=0.0, max_iters=8)
        hm.fit(x, data["a"][tr])
        LL = np.asarray(r1["LL_list"]) / x.size
        n = min(len(LL), len(hm.ll_history))
        assert np.allclose(np.asarray(hm.ll_history)[:n], LL[:n], atol=1e-7), (hm.ll_history, LL)
        assert row1["emission"] == "hiql" and np.isfinite(row1["LL_test"])
        z = np.load(os.path.join(out, "T_HIQL_SC_N1_2_3_scales_KF_0.npz"), allow_pickle=True)
        assert "sc_iavi_r" in z.files and "sc_logpi" in z.files
        r2, tree2, row2 = rk.fit_variant_fold(y, data, T, tr, te, "N2", v["N2"], None, None, None, None, None,
                                              None, (r1, tree1), 0)
        assert tree2.order == 2 and np.isfinite(row2["LL_test"])
        assert row2["LL_train"] >= row1["LL_train"] - 1e-9, (row1["LL_train"], row2["LL_train"])
        # restarts: best of 3 >= the single restart
        y3 = _conf(out, hiavi_restarts=3, run_uid="T_HIQL3")
        r3, _, row3 = rk.fit_variant_fold(y3, data, T, tr, te, "N1", v["N1"], None, None, None, None, None,
                                          None, None, 0)
        assert row3["LL_train"] >= row1["LL_train"] - 1e-9
        rk._write_rows(y, [row1, row2])
        ys = dict(y, scales=dict(grid=dict(order=[1])))
        try:
            rk._write_rows(ys, [dict(row1, emission="swirl")])
            raise AssertionError("mixing emissions should raise")
        except ValueError:
            pass


def test_r5_swirl_unchanged():
    y = dict(seed=0, scales=dict(grid=dict(order=[1, 2, 3]), reward_param="additive"))
    v = rk.scales_variants(y)
    assert [t for t, _ in v] == ["N1", "N2", "N3"]
    assert all(sc.ScalesConfig.from_dict(c).emission == "swirl" for _, c in v)


if __name__ == "__main__":
    for f in (test_r1_variants, test_r2_r3_r4, test_r5_swirl_unchanged):
        f()
        print("ok", f.__name__)
