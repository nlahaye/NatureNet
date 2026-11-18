"""
Synthetic tests of the habitat analyses (lagged_env, habitat_tests, reward_env_projection extras). No data / GPU.

    python -m pytest naturenet/drivers/test_habitat.py -q      (or: python -m naturenet.drivers.test_habitat)
"""
import json
import os
import pickle
import tempfile

import numpy as np
import pandas as pd

H, W, DAYS = 20, 20, 60
CH = ["grad_mag_sst", "okubo_weiss_geo", "rossby_geo", "wind_speed", "depth_m"]


def _scenes(rng):
    """(DAYS, C, H, W) scenes: wind = day index (so lagged means are known exactly), others random."""
    X = rng.normal(size=(DAYS, len(CH), H, W)).astype(np.float32)
    X[:, CH.index("grad_mag_sst")] = np.abs(X[:, CH.index("grad_mag_sst")])
    X[:, CH.index("wind_speed")] = np.arange(DAYS, dtype=np.float32)[:, None, None]
    X[:, CH.index("depth_m")] = np.linspace(10, 200, W)[None, None, :]
    return X


def _setup(tmp, n_animals=6, n_fix=40, m=4, seed=0):
    from naturenet.environment.grid_utils import Grid
    rng = np.random.default_rng(seed)
    X = _scenes(rng)
    days = pd.date_range("2020-03-01", periods=DAYS, freq="D")
    sc = os.path.join(tmp, "scenes")
    os.makedirs(sc)
    paths = []
    for i in range(DAYS):
        p = os.path.join(sc, f"t{i}.npy")
        np.save(p, X[i])
        paths.append(p)
    pd.DataFrame(dict(t=range(DAYS), date=days.strftime("%Y-%m-%d"), path=paths, complete=True)).to_csv(
        os.path.join(tmp, "idx.csv"), index=False)
    meta = dict(channels=CH, grid=dict(shape=[H, W], lon_bounds=[0.0, 20.0], lat_bounds=[0.0, 20.0], north_up=True),
                fill_value=-999.0)
    json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"))
    df_dir, avl_dir = os.path.join(tmp, "df"), os.path.join(tmp, "avl")
    os.makedirs(df_dir)
    os.makedirs(avl_dir)
    uid = "RUN"
    grid = Grid(0.0, 0.0, 20.0, 20.0, 1.0, 1.0)
    pickle.dump(grid, open(os.path.join(df_dir, uid + "_grid.pkl"), "wb"))
    pickle.dump(grid, open(os.path.join(avl_dir, uid + "_grid.pkl"), "wb"))
    dfs, avl, series = {}, {}, []
    ts0 = days[35]   # fixes start late enough for 30-day windows
    for a in range(n_animals):
        key = f"A{a}"
        # odd animals in April ("wet" group in the test), even animals in March ("dry")
        start = ts0 if a % 2 else days[15]
        t = start + pd.to_timedelta(np.arange(n_fix) // 4, "D") + pd.Timedelta(hours=1)
        lat = rng.uniform(1, 19, n_fix)
        lon = rng.uniform(1, 19, n_fix)
        z = rng.integers(0, 2, n_fix)
        # mode 0 steps enter deep water (east) -> positive depth selection
        for p in range(n_fix - 1):
            if z[p] == 0:
                lon[p + 1] = rng.uniform(17, 19.9)
        dfs[key] = [pd.DataFrame(dict(timestamp=t, latitude=lat, longitude=lon))]
        for j in range(m):
            avl[f"{key}__avl{j}"] = [pd.DataFrame(dict(timestamp=t, latitude=rng.uniform(1, 19, n_fix),
                                                       longitude=rng.uniform(1, 12, n_fix)))]
        series.append(pd.DataFrame(dict(uid=key, segment=0, seg_pos=np.arange(n_fix), timestamp=t, latent=z,
                                        env=rng.integers(0, 5, n_fix), reward_likelihood=rng.normal(size=n_fix),
                                        speed_out_kmh=np.where(z == 0, 5.0, 1.0), step_out_km=1.0,
                                        turn_angle=rng.normal(size=n_fix), month=t.month)))
    pickle.dump(dfs, open(os.path.join(df_dir, uid + "_dfs.pkl"), "wb"))
    pickle.dump(avl, open(os.path.join(avl_dir, uid + "_dfs.pkl"), "wb"))
    json.dump(dict(n_available=m), open(os.path.join(avl_dir, "availability_meta.json"), "w"))
    sdir = os.path.join(tmp, "series")
    os.makedirs(sdir)
    for s in series:
        s.to_csv(os.path.join(sdir, f"{s['uid'].iloc[0]}_time_series_pd.csv"), index=False)
    sconf = dict(df_dir=df_dir, df_run_uid=uid, scene_index_csv=os.path.join(tmp, "idx.csv"),
                 env_meta_json=os.path.join(tmp, "meta.json"), complete_scenes_only=True, nearest_scene_days=0,
                 input_channels=CH)
    import yaml
    yaml.safe_dump(sconf, open(os.path.join(tmp, "sample.yaml"), "w"))
    return X, days, dfs, avl, sdir


def _reader(path, chan_idx, r0, r1, c0, c1, fill):
    return np.load(path)[list(chan_idx), r0:r1, c0:c1].astype(np.float32)


def _sample_inputs(tmp, df_dir):
    """sample_env_inputs with the npy reader (the real one reads zarr)."""
    import naturenet.environment.sample_env_inputs as se
    orig = se._read_points

    def rp(path, chan, rows, cols, fill):
        a = np.load(path)
        a = a if chan is None else a[chan]
        return a[:, rows, cols].T.astype(np.float32)
    se._read_points = rp
    try:
        import yaml
        c = yaml.safe_load(open(os.path.join(tmp, "sample.yaml")))
        c["df_dir"] = df_dir
        se.sample(c)
    finally:
        se._read_points = orig


def _spec(tmp, sdir, **kw):
    s = dict(out_dir=os.path.join(tmp, "out"), sample_inputs_yaml=os.path.join(tmp, "sample.yaml"),
             availability=dict(avl_dir=os.path.join(tmp, "avl")), tools=[dict(label="t", series_dir=sdir)],
             n_boot=200, seed=0,
             features=dict(fronts=dict(sst_front=dict(channel="grad_mag_sst", percentile=90)),
                           eddy=dict(channel="okubo_weiss_geo", threshold=0.2, polarity_channel="rossby_geo"),
                           threshold_source="scene"),
             lagged=dict(channels=["wind_speed"], windows=dict(d0=[0, 0], m8=[0, 7], lag1w=[7, 13]), min_frac=1.0),
             lags=dict(base="d0"),
             seasons=dict(groups=dict(wet=[4, 5], dry=[3]), contrast=["wet", "dry"], channels=["depth_m"]))
    s.update(kw)
    return s


def test_lagged_means_exact():
    from naturenet.drivers import lagged_env as le
    with tempfile.TemporaryDirectory() as tmp:
        X, days, dfs, _, sdir = _setup(tmp)
        spec = _spec(tmp, sdir)
        le.write_lagged(spec, reader=_reader)
        uid = "RUN"
        names = json.load(open(os.path.join(spec["out_dir"], "lagged_inputs", uid + "_lagged_input_channels.json")))
        assert names == ["wind_speed__d0", "wind_speed__m8", "wind_speed__lag1w"]
        v = pickle.load(open(os.path.join(spec["out_dir"], "lagged_inputs", f"{uid}_A1_lagged_input.pkl"), "rb"))[0]
        d = (pd.to_datetime(dfs["A1"][0]["timestamp"]).dt.normalize() - days[0]).dt.days.to_numpy()
        assert np.allclose(v[:, 0], d)                     # wind = day index
        assert np.allclose(v[:, 1], d - 3.5)               # mean of d-7 .. d
        assert np.allclose(v[:, 2], d - 10.0)              # mean of d-13 .. d-7
        # a gap in the calendar with min_frac 1 -> NaN for windows containing it
        idx = pd.read_csv(os.path.join(tmp, "idx.csv"))
        idx.loc[idx["t"] == int(d[0]) - 1, "complete"] = False
        idx.to_csv(os.path.join(tmp, "idx.csv"), index=False)
        le.write_lagged(spec, reader=_reader)
        v = pickle.load(open(os.path.join(spec["out_dir"], "lagged_inputs", f"{uid}_A1_lagged_input.pkl"), "rb"))[0]
        assert np.isfinite(v[0, 0]) and np.isnan(v[0, 1])


def test_scene_stats_and_features():
    from naturenet.drivers import habitat_tests as ht, lagged_env as le
    with tempfile.TemporaryDirectory() as tmp:
        X, days, dfs, avl, sdir = _setup(tmp)
        _sample_inputs(tmp, os.path.join(tmp, "df"))
        _sample_inputs(tmp, os.path.join(tmp, "avl"))
        spec = _spec(tmp, sdir)
        csv = le.scene_stats(spec, reader=_reader)
        st = pd.read_csv(csv)
        d0 = int((pd.Timestamp(st["date"].iloc[0]) - days[0]).days)
        assert np.isclose(st["grad_mag_sst_p90"].iloc[0], np.percentile(X[d0, 0], 90), rtol=1e-5)
        assert np.isclose(st["okubo_weiss_geo_sd"].iloc[0], np.std(X[d0, 1]), rtol=1e-4)
        tabs = ht.run(spec, ("features",))
        fu = tabs["feature_use"]
        assert set(fu["feature"]) == {"sst_front", "eddy_core", "eddy_core_cyclonic", "eddy_core_anticyclonic"}
        allf = fu[(fu["mode"] == "all") & (fu["feature"] == "sst_front")].iloc[0]
        assert 0.03 < allf["p_available"] < 0.2            # ~10 % of pixels are fronts
        assert abs(allf["log_or"]) < 1.0                   # random scenes: no selection
        assert {"diff", "ci_low", "q_bh"} <= set(tabs["feature_mode_contrast"].columns)


def test_mh_odds_ratio():
    from naturenet.drivers.habitat_tests import mh_terms
    e = np.array([1, 1, 0, 1.0])
    X = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, np.nan]], float)
    num, den, eu, pa = mh_terms(e, X)
    orr = np.nansum(num) / np.nansum(den)
    # strata: (1, x=1, m=4), (1, 1, 4), (0, 2, 4), (1, 0, 3) -> num 3/5 + 3/5 + 0 + 3/4, den 0 + 0 + 2/5 + 0
    assert np.isclose(orr, (0.6 + 0.6 + 0.75) / 0.4)
    assert np.isclose(pa[3], 0.0)


def test_lags_and_seasons():
    from naturenet.drivers import habitat_tests as ht, lagged_env as le
    with tempfile.TemporaryDirectory() as tmp:
        X, days, dfs, avl, sdir = _setup(tmp)
        _sample_inputs(tmp, os.path.join(tmp, "df"))
        _sample_inputs(tmp, os.path.join(tmp, "avl"))
        spec = _spec(tmp, sdir)
        le.write_lagged(spec, reader=_reader)
        le.write_lagged(spec, avl=True, reader=_reader)
        tabs = ht.run(spec, ("lags", "seasons", "lag_profiles"))
        ua = tabs["lag_use_availability"]
        assert np.allclose(ua["diff"], 0, atol=1e-9)       # wind is spatially uniform: used = available
        assert {"lag_mode_association", "lag_reward_association", "lag_reward_joint"} <= set(tabs)
        sua = tabs["season_use_availability"]
        m0 = sua[(sua["mode"] == 0) & (sua["channel"] == "depth_m")]
        assert (m0["diff"] > 0).all() and m0["ci_excludes_0"].all()   # mode 0 enters deep water in every group
        m1 = sua[(sua["mode"] == 1) & (sua["channel"] == "depth_m")]
        assert (m0["diff"].mean() > m1["diff"].mean())
        occ = tabs["season_occupancy"]
        assert set(occ["group"]) == {"wet", "dry", "wet-dry"}
        prof = pd.read_csv(os.path.join(spec["out_dir"], "lagged_state_profiles.csv"))
        assert {"cluster", "n", "wind_speed__m8"} <= set(prof.columns)
        k = pd.read_csv(os.path.join(spec["out_dir"], "mode_kinematics_t.csv"))
        assert k.set_index("mode").loc[0, "speed_rank"] == 1


def test_projection_extras():
    from naturenet.models.irl import reward_env_projection as rep
    rng = np.random.default_rng(0)
    S, K = 30, 2
    P = rng.normal(size=(S, 3))
    extra = pd.DataFrame(dict(cluster=np.arange(S), n=10, lagw=rng.normal(size=S)))
    with tempfile.TemporaryDirectory() as tmp:
        extra.to_csv(os.path.join(tmp, "x.csv"), index=False)
        conf = dict(rep.DEFAULTS)
        conf.update(extra_profiles=[dict(csv=os.path.join(tmp, "x.csv"), index="state", use_in_projection=True)],
                    month_groups=dict(wet=[5, 6], dry=[1, 2]), n_boot=20, n_boot_animals=20)
        Pe = rep.extra_state_profiles(conf, S)
        assert Pe[1] == ["lagw"] and Pe[2].all()
        R = np.vstack([P[:, 0] + Pe[0][:, 0], -P[:, 1]])
        series = {}
        for a in range(4):
            n = 50
            series[f"a{a}"] = pd.DataFrame(dict(uid=f"a{a}", segment=0, seg_pos=np.arange(n),
                                                env=rng.integers(0, S, n).astype(float),
                                                latent=rng.integers(0, K, n).astype(float),
                                                month=np.where(np.arange(n) < 25, 5, 1)))
        T = rng.dirichlet(np.ones(S), size=(S, 3))
        out = rep.run(conf, R, P, ["a", "b", "c"], series, T, tmp, None, Pe)
        co = out["coefficients"]
        assert "lagw" in set(co["variable"])
        assert co[(co["mode"] == 0) & (co["variable"] == "lagw")]["coef"].iloc[0] > 0.3
        g = pd.read_csv(os.path.join(tmp, "use_availability_by_group.csv"))
        assert set(g["group"]) == {"wet", "dry"} and "lagw" in set(g["variable"])


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
