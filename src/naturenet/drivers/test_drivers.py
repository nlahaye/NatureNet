"""
Synthetic tests of the environmental-driver tools (no data / GPU needed).

    python -m pytest naturenet/drivers/test_drivers.py -q      (or: python -m naturenet.drivers.test_drivers)
"""
import json
import os
import pickle
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- test 1: nulls
def test_env_null_modes():
    from naturenet.drivers.env_null import null_segments
    K = 4
    rng = np.random.default_rng(0)
    segs = [np.array([0, 1, 2, 3, K, -1, 1, 1]), np.array([2, 2, 3, 0, K])]
    flat = np.concatenate(segs)
    real = (flat >= 0) & (flat < K)
    for mode in ("shift", "shuffle", "marginal", "single"):
        out, info = null_segments(segs, K, mode, rng, pooled_p=np.full(K, 1 / K))
        o = np.concatenate(out)
        assert [len(s) for s in out] == [len(s) for s in segs]
        assert (o[flat == -1] == -1).all()                         # invalid positions untouched
        assert info["n_valid"] == real.sum()
        if mode in ("shift", "shuffle"):                           # same multiset of labels
            assert sorted(o[real]) == sorted(flat[real])
            assert (o[flat == K] == K).all()
        if mode == "single":
            assert (o[real] == 0).all() and (o[flat == K] == 1).all()


# ----------------------------------------------------------------------------- test 3: year folds
def test_year_folds():
    from naturenet.drivers.year_folds import year_fold_groups
    years = {"2015_a": 2015, "2015_b": 2015, "2017_c": 2017, "2017_d": 2017, "2017_e": 2017}
    folds, yg = year_fold_groups(years, [[2015], [2017]])
    assert folds == [["2015_a", "2015_b"], ["2017_c", "2017_d", "2017_e"]] and yg == [[2015], [2017]]
    folds, _ = year_fold_groups(years)                             # default: one fold per year
    assert len(folds) == 2
    try:
        year_fold_groups(years, [[2015]])
        raise AssertionError("unlisted year accepted")
    except ValueError:
        pass


# ----------------------------------------------------------------------------- test 2 / 7: labels
def test_decision_label():
    from naturenet.drivers.env_scale_grid import decision_label
    assert decision_label("HIQL_N1") == "N1"
    assert decision_label("N2_hist-lags+decay") == "N2+hist"
    assert decision_label("N1_hist-env-core") == "N1+env"
    assert decision_label("N1_hist-lags+decay+env-core_dur-logi20") == "N1+hist+env+dur"


# ----------------------------------------------------------------------------- test 4: PPC
class _Model:
    def __init__(self, K=2, S=3, A=2, seed=0):
        r = np.random.default_rng(seed)
        self.pis = r.dirichlet(np.ones(A), size=(K, S))
        self.logp_init = np.log(np.full(K, 1 / K))
        self.logp_tr = np.log(np.array([[0.9, 0.1], [0.2, 0.8]]))
        self.kin_model = None

    def _lls(self, s, a, kin):
        a0 = np.maximum(a, 0)
        return np.log(self.pis[:, s, a0]).transpose(1, 2, 0)           # (N, T, K)


def test_ppc_simulate_and_one_step():
    from naturenet.drivers.ppc import ece, one_step, simulate
    m = _Model()
    S, A = 3, 2
    T_env = np.random.default_rng(1).dirichlet(np.ones(S), size=(S, A))
    lengths = np.array([10, 6, 8])
    s, a, z = simulate(m, T_env, np.array([0, 1, 2]), lengths, np.random.default_rng(2))
    assert s.shape == (3, 10)
    for i, L in enumerate(lengths):
        assert (a[i, :L] >= 0).all() and (a[i, L:] == -1).all() and (z[i, L:] == -1).all()
    r = one_step(m, s, a, None)
    n = int((a >= 0).sum())
    assert len(r["lp"]) == n and np.all(r["lp"] <= 0) and np.all((r["brier"] >= 0) & (r["brier"] <= 2))
    assert 0 <= ece(r["conf"], r["correct"]) <= 1


# ----------------------------------------------------------------------------- test 6: env-matched null
def test_env_matched_null():
    from scipy import sparse as sp
    from naturenet.models.irl.run_hiavi_multiagent import EnvMatchedNull
    N, T = 4, 5
    pc = SimpleNamespace(T=T, times=np.arange(N * T, dtype=float).reshape(N, T) * 3.0,
                         animal=np.array([0, 1, 2, 3]), valid=np.ones((N, T), bool))
    states = np.tile(np.array([0, 1, 0, 1, 2]), (N, 1))
    em = EnvMatchedNull(pc, states, seed=0, rep=1, min_dt_days=0.0)
    tgt, src = np.array([0]), np.array([1, 2, 3])
    K = sp.csr_matrix((np.ones(3), ([0, 1, 2], [0, 1, 4])), shape=(T, len(src) * T))
    out = em({"K": K}, tgt, src, "v")["K"].tocoo()
    s_state = states[src].ravel()
    assert out.nnz + em.stats["dropped"] == 3
    for r, c in zip(out.row, out.col):                                  # same environment state as the original
        orig = K.tocsr()[r].indices[0]
        assert s_state[c] == s_state[orig]


# ----------------------------------------------------------------------------- test 7: env history features
def test_history_env_features():
    from naturenet.models.irl.swirl_scales import HistoryConfig, history_features
    rows = np.array([[0, 1, 1, 2, 0, 1]])
    E = np.arange(12, dtype=float).reshape(1, 6, 2)
    E[0, 3, 1] = np.nan
    h = HistoryConfig(kind=("env",), env_diff=True)
    F, names = history_features(rows, 3, h, env_rows=E, env_names=["a", "b"])
    assert F.shape == (1, 4, 5) and names == ["env_a", "env_b", "denv_a", "denv_b", "env_missing"]
    assert F[0, 0, 0] == E[0, 1, 0] and F[0, 0, 2] == E[0, 1, 0] - E[0, 0, 0]
    assert F[0, 2, 4] == 1.0 and F[0, 2, 1] == 0.0                      # x_t = row 3 has the NaN


def test_env_rows_alignment():
    from naturenet.drivers import env_covariates as ec
    with tempfile.TemporaryDirectory() as d:
        conf = dict(df_dir=d, df_run_uid="RUN")
        with open(os.path.join(d, "RUN_states_input_channels.json"), "w") as f:
            json.dump(["x", "y"], f)
        seg = np.stack([np.arange(8), 10 + np.arange(8)], 1).astype(float)
        orig = ec._input_path
        ec._input_path = lambda c, k: os.path.join(d, f"{k}.pkl")
        try:
            with open(os.path.join(d, "A.pkl"), "wb") as f:
                pickle.dump([seg], f)
            data = dict(meta=[dict(key="A", seg=0, start=2, length=4)], states=np.zeros((1, 5)))
            E, sel = ec.env_rows(conf, data, ["y"], offset=1, standardize=False)
            assert sel == ["y"] and E.shape == (1, 5, 1)
            assert list(E[0, :, 0]) == [13, 14, 15, 16, 17]
            E, _ = ec.env_rows(conf, data, ["x"], offset=4, standardize=False)
            assert np.isnan(E[0, -1, 0]) and E[0, 0, 0] == 6               # past the segment -> NaN
        finally:
            ec._input_path = orig


def test_switch_env_detects_driver():
    from naturenet.drivers.switch_env import analyse
    r = np.random.default_rng(0)
    N, T = 30, 40
    E = r.normal(size=(N, T, 2))
    Z = np.zeros((N, T), int)
    for n in range(N):
        for t in range(1, T):
            Z[n, t] = 1 if E[n, t, 0] > 1.0 else 0                         # channel 0 drives the switch
    d, prof = analyse(E, Z, np.ones((N, T), bool), np.arange(N) % 6, ["drv", "noise"], n_perm=99)
    lv = d[(d["kind"] == "level") & (d["from_z"] == 0)].set_index("channel")
    assert lv.loc["drv", "diff"] > 1 and lv.loc["drv", "p_perm"] < 0.05
    assert abs(lv.loc["noise", "diff"]) < 0.5 and len(prof)


# ----------------------------------------------------------------------------- geometry / synthesis
def test_reptax_bridge_stats():
    from naturenet.drivers import reptax_bridge as rb
    r = np.random.default_rng(0)
    X = np.c_[r.normal(size=(800, 2)), np.zeros((800, 6))] @ r.normal(size=(8, 8))
    gid = rb.twonn_id(X)
    assert 1.3 < gid < 3.0
    lonlat = np.c_[r.uniform(-125, -118, 300), r.uniform(32, 38, 300)]
    v = lonlat[:, 1] + 0.1 * r.normal(size=300)                         # smooth field -> positive autocorrelation
    nbr, _ = rb.haversine_knn(lonlat, 8)
    mi = rb.morans_i(v, nbr, n_perm=99)
    assert mi["moran_I"] > 0.5 and mi["moran_p"] < 0.05
    assert rb.mantel(lonlat, lonlat[:, ::-1] * 1.0, n_perm=49)["mantel_r"] > 0.9
    ranks = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    assert abs(rb.kendall_w(ranks) - 1.0) < 1e-9
    blocks = pd.DataFrame(r.normal(size=(12, 3)) + np.array([0, 1, 3]), columns=["a", "b", "c"])
    summ, mr, pw = rb.friedman_nemenyi(blocks)
    assert mr.index[0] == "c" and summ["p"] < 0.05 and pw.shape == (3, 3)
    df = pd.DataFrame([dict(f1=i, f2=j, v=3 * i + 0.1 * j) for i in range(3) for j in range(4)])
    vd = rb.variance_decomposition(df, ["f1", "f2"], "v")
    assert vd is not None and len(vd)


def test_availability():
    from naturenet.drivers.availability import make_available
    t = pd.date_range("2020-01-01", periods=20, freq="6h")
    seg = pd.DataFrame(dict(timestamp=t, latitude=34 + 0.05 * np.arange(20), longitude=-120 + 0.02 * np.arange(20)))
    out, meta = make_available({"A": [seg], "B": [seg.iloc[:10]]}, m=3, seed=0)
    assert set(out) == {f"{k}__avl{j}" for k in "AB" for j in range(3)}
    p = out["A__avl0"][0]
    assert len(p) == 20 and p["used_fix"].iloc[0] and not p["used_fix"].iloc[1:].any()
    assert p["latitude"].iloc[0] == seg["latitude"].iloc[0] and meta["n_available"] == 3


def test_synthesis_rank_agreement():
    from naturenet.drivers.synthesis import rank_agreement
    rows = []
    for item, base in (("fine", 3.0), ("k12", 2.0), ("null_shift", 0.0)):
        rows.append(dict(test="grid", item=item, perspective="dLL@N1", value=base, higher_better=True))
        rows.append(dict(test="ppc", item=item, perspective="ece", value=1.0 / (1 + base), higher_better=False))
    L = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as d:
        R = rank_agreement(L, {"levels": {"perspectives": ["grid:dLL", "ppc:ece"]}}, d)
        assert abs(R["kendall_w"].iloc[0] - 1.0) < 1e-9
        assert os.path.exists(os.path.join(d, "ranks_levels.csv"))


def test_audit_classify():
    from naturenet.drivers.audit import classify
    lv = ["fine", "k12", "null_shift"]
    assert classify("hiavi.k12-hiavi-kfold-kin", lv) == ("fits", "k12")
    assert classify("scales.null-shift-hiql-scales-kin-k5", lv) == ("fits", "null_shift")
    assert classify("hiavi.year-null-shift-hiavi-kfold-kin", lv) == ("year_transfer", "null_shift")
    assert classify("scales.envswitch-fine-hiql-scales-kin-k5", lv) == ("env_switch", "fine")
    assert classify("ppc.hiql_kin_k5", lv) == ("ppc", "hiql_kin_k5")
    assert classify("synthesis", lv) == ("synthesis", "")


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:                          # noqa: BLE001
                fails += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
