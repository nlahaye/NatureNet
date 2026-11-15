"""
Sufficient-statistic M-steps (mstep_suff_stats: True) must reproduce the original per-step losses, and
subsample_fold must keep folds representative.
    PYTHONPATH=src python -m naturenet.models.irl.test_suff_stats
"""
import os
import tempfile
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

jax.config.update("jax_enable_x64", True)

from naturenet.models.irl import run_swirl_kfold as rk  # noqa: E402
from naturenet.models.irl import swirl_training as st  # noqa: E402
from naturenet.models.irl import swirl_data as sd  # noqa: E402
from naturenet.models.irl.test_s2_index_emissions import _toy  # noqa: E402


def _rand_batch(seed=0, N=7, T=15, K=3, S=11, A=9, H=4):
    r = np.random.default_rng(seed)
    xi, ai = r.integers(0, S, (N, T)), r.integers(0, A, (N, T))
    m = np.arange(T)[None] < r.integers(3, T + 1, N)[:, None]
    xo = jax.nn.one_hot(xi, S)[:, :, None, :] * m[:, :, None, None]
    ao = jax.nn.one_hot(ai, A)[:, :, None, :] * m[:, :, None, None]
    idx = jnp.asarray(np.where(m, xi, -1).astype(np.int32))[:, :, None]
    gam = jax.nn.softmax(jnp.asarray(r.normal(size=(N, T, K))), -1)
    xis = jax.nn.softmax(jnp.asarray(r.normal(size=(N, T - 1, K * K))), -1).reshape(N, T - 1, K, K)
    Tp = r.random((S, A, S)); Tp /= Tp.sum(-1, keepdims=True)
    Rs = (jnp.asarray(r.normal(size=(S, H))), jnp.asarray(r.normal(size=H)),
          jnp.asarray(r.normal(size=(H, K))), jnp.asarray(r.normal(size=K)))
    return dict(xo=xo, ao=ao, idx=idx, gam=gam, xis=xis, T=jnp.asarray(Tp), Rs=Rs,
                logPs=jnp.asarray(r.normal(size=(K, K))), rew=jnp.asarray(r.normal(size=(K, 1, S))), K=K, S=S, A=A)


def test_losses_equal():
    b = _rand_batch()
    n = st.n_real(b["xo"])
    shape = tuple(b["rew"].shape)
    for use_temp in (False, True):
        tm = jnp.asarray([0.5, 1.0, 2.0]) if use_temp else None
        v0, g0 = st._emit_vg(b["rew"].ravel(), b["T"], tm, b["gam"], b["xo"], b["ao"], n, shape, use_temp)
        for x in (b["xo"], b["idx"]):
            C = st.emit_counts(b["gam"], x, b["ao"], n_rows=b["S"])
            v1, g1 = st._emit_vg_ss(b["rew"].ravel(), b["T"], tm, C, n, shape, use_temp)
            assert abs(float(v0 - v1)) < 1e-10 and float(jnp.abs(g0 - g1).max()) < 1e-10
    parts = [b["logPs"], *b["Rs"]]
    shapes = tuple(tuple(int(d) for d in p.shape) for p in parts)
    flat = jnp.concatenate([p.ravel() for p in parts])
    v0, g0 = st._trans_vg(flat, b["xis"], b["xo"], n, shapes)
    v1, g1 = st._trans_vg_ss(flat, st.trans_counts(b["xis"], b["xo"]), n, shapes)
    assert abs(float(v0 - v1)) < 1e-10 and float(jnp.abs(g0 - g1).max()) < 1e-10


def test_fit_swirl_final_equal():
    T, data, arh, K, S, A = _toy()
    res, secs = {}, {}
    for ss in (False, True):
        st.set_mstep_options(suff_stats=ss)
        out = tempfile.mkdtemp()
        conf = dict(out_dir=out, run_uid="t", seed=0, n_hidden=K, n_hidden_init=K, n_states=S, epochs=3,
                    epochs_2=2, run_s2=True, s2_index_emissions=True, temps_final=[0.5, 1.0])
        t0 = time.time()
        rk.fit_swirl_final(conf, data(), T, np.arange(20), np.arange(20, 30), arh, "_KF_0")
        secs[ss] = time.time() - t0
        res[ss] = {f: np.load(os.path.join(out, f)) for f in os.listdir(out) if "LL" in f}
    st.set_mstep_options(suff_stats=True)
    for f in res[True]:
        d = abs(float(res[False][f]["LL"]) - float(res[True][f]["LL"]))
        assert d < 1e-6 * max(1.0, abs(float(res[False][f]["LL"]))), (f, d)
        assert (res[False][f]["jax_path_vmap"] == res[True][f]["jax_path_vmap"]).all(), f
    print("  fit_swirl_final seconds (incl. compile): per-step", round(secs[False], 1), "suff-stats", round(secs[True], 1))


def test_structured_s2_policy():
    T, data, arh, K, S, A = _toy()
    from naturenet.models.irl.swirl_utils import compute_prev_state_map, construct_new_trans_probs_limited, n_prev_slots
    d = data()
    psm = compute_prev_state_map(T, S, A, observed_pairs=[(d["x"], d["x_prev"])])
    M = n_prev_slots(psm)
    dense, _ = construct_new_trans_probs_limited(T, psm, None, S, A, n_slots=M)
    s2 = rk.s2_structured_trans(T, psm, S, M)
    rew = jnp.asarray(np.random.default_rng(0).normal(size=(K, 1, S * M)))
    for use_temp in (False, True):
        tm = jnp.asarray([0.3, 1.0]) if use_temp else None
        a = st._log_policy(rew, jnp.asarray(dense), tm, use_temp)
        b = st._log_policy(rew, st.device_tp(s2), tm, use_temp)
        assert float(jnp.abs(a - b).max()) < 1e-10


def test_fit_swirl_final_structured_equal():
    T, data, arh, K, S, A = _toy()
    res = {}
    for mode in (False, True):
        out = tempfile.mkdtemp()
        conf = dict(out_dir=out, run_uid="t", seed=0, n_hidden=K, n_hidden_init=K, n_states=S, epochs=3,
                    epochs_2=2, run_s2=True, s2_structured_vi=mode, temps_final=[0.5, 1.0])
        rk.fit_swirl_final(conf, data(), T, np.arange(20), np.arange(20, 30), arh, "_KF_0")
        res[mode] = {f: np.load(os.path.join(out, f)) for f in os.listdir(out) if "LL" in f}
    for f in res[True]:
        d = abs(float(res[False][f]["LL"]) - float(res[True][f]["LL"]))
        assert d < 1e-6 * max(1.0, abs(float(res[False][f]["LL"]))), (f, d)
        assert (res[False][f]["jax_path_vmap"] == res[True][f]["jax_path_vmap"]).all(), f


def test_subsample():
    r = np.random.default_rng(3)
    groups = np.repeat(np.array(["a", "b", "c", "d", "e"]), [300, 200, 100, 40, 5])
    lengths = r.integers(10, 61, len(groups))
    idx = np.arange(len(groups))
    sub = sd.subsample_index(idx, groups, lengths, 120, seed=0)
    assert len(sub) == 120 and set(groups[sub]) == set(groups)
    frac = {g: (groups[sub] == g).mean() for g in "abcd"}
    full = {g: (groups == g).mean() for g in "abcd"}
    assert all(abs(frac[g] - full[g]) < 0.03 for g in "abcd"), (frac, full)
    assert abs(np.median(lengths[sub]) - np.median(lengths)) <= 3
    assert (sd.subsample_index(idx, groups, lengths, 120, seed=0) == sub).all()          # deterministic
    assert (sd.subsample_index(idx, groups, lengths, None) == idx).all()
    data = dict(groups=groups, lengths=lengths)
    tr, te = np.where(groups != "c")[0], np.where(groups == "c")[0]
    tr2, te2 = sd.subsample_fold(dict(max_train_seqs=100, max_test_seqs=50, fold_seed=0), data, tr, te, 0)
    assert len(tr2) == 100 and len(te2) == 50 and not set(groups[tr2]) & set(groups[te2])


if __name__ == "__main__":
    test_losses_equal()
    print("test_losses_equal: OK")
    test_fit_swirl_final_equal()
    print("test_fit_swirl_final_equal: OK")
    test_structured_s2_policy()
    print("test_structured_s2_policy: OK")
    test_fit_swirl_final_structured_equal()
    print("test_fit_swirl_final_structured_equal: OK")
    test_subsample()
    print("test_subsample: OK")
