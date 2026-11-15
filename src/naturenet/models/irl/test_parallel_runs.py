"""
Tests for the parallel / resumable pieces (CPU only):

    PYTHONPATH=src JAX_PLATFORMS=cpu NATURENET_JAX_PLATFORM=cpu python -m naturenet.models.irl.test_parallel_runs

- parallel.cpu_map equals the serial loop; locked update_csv keeps every concurrent writer's rows
- multi-agent --part pieces (obs, null chunks) + merge == one full run (CV rows, synchrony, summary, final fits),
  with HIAVI restarts in parallel (hiavi_restart_jobs = 2) in the parts run and serial in the full run
- jax_precision tiers select float32 / float64 in a fresh process
- ARHMM early stop (arhmm_tol_per_step) stops before arhmm_epochs (ssm fork with ARHMMs.initialize(tolerance=))
"""
import copy
import json
import multiprocessing as mp
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from naturenet.models.irl.parallel import cpu_map, update_csv


def _sq(x, y=0):
    return x * x + y


def test_cpu_map():
    items = [(i, 1) for i in range(7)]
    assert cpu_map(_sq, items, 1) == cpu_map(_sq, items, 3) == [i * i + 1 for i in range(7)]
    assert cpu_map(_sq, [3], 4) == [9]


def _writer(path, i):
    for r in range(5):
        update_csv(path, lambda old, r=r: pd.concat([old, pd.DataFrame([dict(w=i, r=r)])]) if old is not None
                   else pd.DataFrame([dict(w=i, r=r)]))


def test_locked_csv():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "x.csv")
    ps = [mp.get_context("spawn").Process(target=_writer, args=(path, i)) for i in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    df = pd.read_csv(path)
    assert len(df) == 20 and set(zip(df.w, df.r)) == {(i, r) for i in range(4) for r in range(5)}
    shutil.rmtree(d)


def _gen(d, seed=0):
    """Toy multi-animal tracks (synchronised latent switching), in the multi-agent input format."""
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    S, A, K = 8, 9, 2
    T = rng.dirichlet(np.ones(S) * 0.5, (S, A))
    pis = np.stack([rng.dirichlet(np.ones(A) * 0.3, S) for _ in range(K)])
    t0 = pd.Timestamp("2023-11-01", tz="UTC")
    g = np.zeros(345 * 24, int)
    z = 0
    for h in range(len(g)):
        if rng.uniform() < 1 / (8 * 24):
            z = 1 - z
        g[h] = z
    spans = {"A0": (20, 240, 3.0), "A1": (100, 240, 1.0), "A2": (5, 230, 0.8)}
    dfs = {}
    for an, (a0, a1, rate) in spans.items():
        n = int((a1 - a0) * rate)
        tt = np.sort(rng.uniform(a0, a1, n))
        s = rng.integers(S)
        S_, A_ = [s], []
        for i in range(n - 1):
            zi = g[int(tt[i] * 24)] if rng.uniform() < 0.9 else 1 - g[int(tt[i] * 24)]
            a = rng.choice(A, p=pis[zi, s])
            s = rng.choice(S, p=T[s, a])
            A_.append(a)
            S_.append(s)
        lat = 24 + np.cumsum(rng.normal(0, 0.02, n))
        lon = -77 + np.cumsum(rng.normal(0, 0.02, n))
        ts = t0 + pd.to_timedelta(tt, unit="D")
        segs, ss, aa, i = [], [], [], 0
        while i < n - 1:
            j = min(i + 49, n)
            segs.append(pd.DataFrame(dict(timestamp=ts[i:j], latitude=lat[i:j], longitude=lon[i:j])))
            ss.append(np.array(S_[i:j]))
            aa.append(np.array(A_[i:j - 1]))
            i = j - 1
        dfs[an] = segs
        pickle.dump(aa, open(f"{d}/hh_{an}_actions.pkl", "wb"))
        pickle.dump(ss, open(f"{d}/hh_{an}_states_abstract_env.pkl", "wb"))
    pickle.dump(dfs, open(f"{d}/hh_dfs.pkl", "wb"))


def _ma_conf(d, out):
    return dict(actions=f"{d}/hh_*_actions.pkl", positions="auto", df_dir=d, df_run_uid="hh", fill_unobserved="self_loop",
                hiavi_max_iters=60, hiavi_pseudocount=0.5, hiavi_restarts=3, hiavi_tol=1e-5, n_hidden=2, n_states=8,
                out_dir=out, per_fold_trans_prob=True, require_full_trans_prob=False, run_uid="MA", seed=1,
                segment_policy="chunk", seq_len=12,
                multiagent=dict(Ks=[2], n_null=3, n_null_sync=10, n_outer=3, window=["2023-11-01", "2024-10-10"],
                                variants=dict(temporal=dict(h_days=1.0, max_dt_days=3.0))))


def test_multiagent_parts_equal_full():
    from naturenet.models.irl.run_hiavi_multiagent import expand_inputs, parse_part, run_multiagent
    assert parse_part(None, 5) is None and parse_part("obs", 5) == (True, [])
    assert parse_part("null:1-3", 5) == (False, [1, 2, 3]) and parse_part("null:all", 2) == (False, [0, 1])
    try:
        parse_part("null:7", 5)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    root = tempfile.mkdtemp()
    _gen(os.path.join(root, "in"))
    full = expand_inputs(_ma_conf(os.path.join(root, "in"), os.path.join(root, "full")))
    parts = copy.deepcopy(full)
    parts["out_dir"] = os.path.join(root, "parts")
    parts["hiavi_restart_jobs"] = 2
    run_multiagent(full, verbose=False)
    run_multiagent(parts, part="null:0-1", verbose=False)       # nulls before obs: base fits cached by this part
    run_multiagent(parts, part="obs", verbose=False)
    run_multiagent(parts, part="null:2", verbose=False)
    import naturenet.models.irl.run_hiavi_multiagent as ram
    run_cv = ram.run_cv
    ram.run_cv = lambda *a, **k: (_ for _ in ()).throw(AssertionError("merge recomputed a CV part"))
    try:
        run_multiagent(parts, verbose=False)                    # merge: every CV pass must come from the parts
    finally:
        ram.run_cv = run_cv
    a, b = os.path.join(root, "full", "multiagent"), os.path.join(root, "parts", "multiagent")
    for f in ("ma_cv.csv", "ma_sync.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(os.path.join(a, f)), pd.read_csv(os.path.join(b, f)),
                                      check_exact=False, rtol=1e-9, atol=1e-12)
    assert json.load(open(os.path.join(a, "ma_summary.json"))) == json.load(open(os.path.join(b, "ma_summary.json")))
    fa, fb = np.load(os.path.join(a, "ma_final_K2.npz"), allow_pickle=True), \
        np.load(os.path.join(b, "ma_final_K2.npz"), allow_pickle=True)
    for k in fa.files:
        if fa[k].dtype.kind in "fiub":
            assert np.allclose(fa[k], fb[k], equal_nan=True), k
    sub = os.listdir(os.path.join(b, "parts"))
    files = sorted(os.listdir(os.path.join(b, "parts", sub[0])))
    assert "cv_K2_obs.csv" in files and "cv_K2_null0002.csv" in files and any(f.startswith("base_K2_") for f in files)
    shutil.rmtree(root)


def test_precision_tiers():
    code = "import naturenet.models.irl.jax_precision as p, jax.numpy as jnp; print(jnp.ones(1).dtype, p.describe())"
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    out64 = subprocess.run([sys.executable, "-c", code], env=dict(env, NATURENET_JAX_X64="1"), capture_output=True,
                           text=True).stdout
    out32 = subprocess.run([sys.executable, "-c", code], env=dict(env, NATURENET_JAX_X64="0",
                                                                  NATURENET_JAX_MATMUL_PRECISION="bfloat16"),
                           capture_output=True, text=True).stdout
    assert out64.startswith("float64"), out64
    assert out32.startswith("float32") and "bfloat16" in out32, out32


def test_arhmm_early_stop():
    import inspect
    from ssm import swirl
    if "tolerance=tolerance" not in inspect.getsource(swirl.ARHMMs):    # initialize is wrapped (signature hidden)
        print("  (skipped: ssm without ARHMMs.initialize(tolerance=))")
        return
    from naturenet.models.irl.run_arhmm_kfold import fit_arhmm_params
    rng = np.random.default_rng(0)
    xs = np.empty(30, dtype=object)
    for i in range(30):
        xs[i] = rng.integers(0, 6, 20)
    np.random.seed(1)
    _, _, ll_all = fit_arhmm_params(xs, 2, 6, epochs=40)
    np.random.seed(1)
    _, _, ll_tol = fit_arhmm_params(xs, 2, 6, epochs=40, tol_per_step=1e-2)
    assert len(ll_tol) < len(ll_all), (len(ll_tol), len(ll_all))
    n = min(len(ll_tol), len(ll_all))
    assert np.allclose(ll_tol[:n], ll_all[:n])           # same path until it stops


def test_select_heldout():
    """Data-driven held-out set: min-context filter, top-N by context steps, explicit list wins."""
    import pandas as pd
    from naturenet.models.irl.run_hiavi_multiagent import select_heldout
    conc = pd.DataFrame(dict(
        variant=["temporal"] * 4 + ["prox"] * 4,
        animal=["a", "b", "c", "d"] * 2,
        steps_with_context=[100, 5, 40, 40, 1, 1, 1, 1]))
    base = dict(sync_variant="temporal", min_context_steps=30)
    assert select_heldout(conc, dict(base)) is None
    assert select_heldout(conc, dict(base, heldout_min_context=True)) == ["a", "c", "d"]
    assert select_heldout(conc, dict(base, heldout_min_context=True, heldout_top_n=2)) == ["a", "c"]
    assert select_heldout(conc, dict(base, heldout_top_n=4)) == ["a", "c", "d", "b"]
    assert select_heldout(conc, dict(base, heldout_min_context=True, heldout_animals=["b"])) is None
    try:
        select_heldout(conc, dict(base, heldout_min_context=True, min_context_steps=1000))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
