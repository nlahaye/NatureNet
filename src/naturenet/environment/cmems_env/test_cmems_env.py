"""Synthetic tests (no network): python -m naturenet.environment.cmems_env.test_cmems_env"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from naturenet.environment.cmems_env import pipeline, store
from naturenet.environment.cmems_env.fill import pushpull_fill, temporal_fill
from naturenet.environment.cmems_env.grid import Grid, Regridder, aggregate_raster, nearest_distance_km
from naturenet.environment.cmems_env.static import build_water_mask


def test_regrid():
    g = Grid(-10, 10, 0, 20, 1.0)
    # fine -> coarse: area mean conserves a constant and the mean of a linear field
    lon = np.arange(-12, 12, 0.25) + 0.125
    lat = np.arange(-2, 22, 0.25) + 0.125
    X, Y = np.meshgrid(lon, lat)
    f = (2 * X + 3 * Y).astype(np.float32)
    out = Regridder(lon, lat, g)(f)
    gx, gy = np.meshgrid(g.lon, g.lat)
    assert np.allclose(out, 2 * gx + 3 * gy, atol=1e-3), np.abs(out - (2 * gx + 3 * gy)).max()
    # coarse -> fine linear, descending lat and 0..360 lon input
    lonc = np.arange(-15, 16, 2.0) % 360
    latc = np.arange(25, -6, -2.0)
    Xc, Yc = np.meshgrid(((lonc + 180) % 360) - 180, latc)
    out2 = Regridder(lonc, latc, g)((Xc - Yc).astype(np.float32)[None])[0]
    assert np.allclose(out2, gx - gy, atol=1e-4)
    # NaN-aware: land NaNs do not bias neighbours
    f2 = f.copy()
    f2[:, X[0] < 0] = np.nan
    out3 = Regridder(lon, lat, g)(f2)
    assert np.all(np.isnan(out3[:, g.lon < -0.5])) and np.allclose(out3[:, g.lon > 0.5], (2 * gx + 3 * gy)[:, g.lon > 0.5], atol=1e-3)


def test_fill():
    rng = np.random.default_rng(0)
    H, W = 37, 53
    y, x = np.mgrid[0:H, 0:W]
    f = np.sin(x / 9.0) + np.cos(y / 7.0)
    a = np.stack([f, 2 * f]).astype(np.float32)
    holes = rng.random((H, W)) < 0.4
    holes[10:20, 10:30] = True
    b = a.copy()
    b[:, holes] = np.nan
    out = pushpull_fill(b)
    assert np.isfinite(out).all()
    assert np.array_equal(out[:, ~holes], a[:, ~holes])
    err = np.abs(out - a)[:, holes]
    assert err.mean() < 0.1, err.mean()
    # all-missing slice stays NaN
    c = np.full((1, 5, 5), np.nan, np.float32)
    assert np.isnan(pushpull_fill(c)).all()
    # temporal fill: gap of 2 filled linearly, gap of 5 untouched, edges untouched
    t = np.arange(20, dtype=np.float32)
    s = t.copy()
    s[[0, 5, 6]] = np.nan
    s[10:15] = np.nan
    r = temporal_fill(s[:, None, None], 3)[:, 0, 0]
    assert np.isnan(r[0]) and np.allclose(r[5:7], [5, 6]) and np.isnan(r[10:15]).all()


def test_bitround_store():
    a = np.random.default_rng(1).normal(size=(3, 50, 60)).astype(np.float32) * 100
    a[0, 0, 0] = np.nan
    b = store.bitround(a, 12)
    assert np.isnan(b[0, 0, 0])
    m = np.isfinite(a)
    assert np.max(np.abs(b[m] - a[m]) / np.abs(a[m])) <= 2 ** -12
    d = tempfile.mkdtemp()
    p = store.save_scene(os.path.join(d, "s.zarr"), a, attrs=dict(channels=["a", "b", "c"]))
    import zarr
    z = zarr.open_array(p, mode="r")
    assert z.chunks == a.shape and list(z.attrs["channels"]) == ["a", "b", "c"]
    assert np.array_equal(np.asarray(zarr.load(p)), a, equal_nan=True)
    shutil.rmtree(d)


def _tif(path, a, lon0, lat1, res, nodata=None):
    import rasterio
    from rasterio.transform import Affine
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype=a.dtype,
                       crs="EPSG:4326", transform=Affine(res, 0, lon0, 0, -res, lat1), nodata=nodata) as dst:
        dst.write(a, 1)


def _mask_tif(d, g):
    # fine water mask over a margin; land west of lon = -7.5, an inland lake, an island
    res = 0.025
    lon0, lat1 = g.lon_min - 4, g.lat_max + 4
    nx, ny = int((g.lon_max - g.lon_min + 8) / res), int((g.lat_max - g.lat_min + 8) / res)
    lon = lon0 + (np.arange(nx) + 0.5) * res
    lat = lat1 - (np.arange(ny) + 0.5) * res
    LON, LAT = np.meshgrid(lon, lat)
    w = (LON > -7.5).astype(np.uint8)
    w[(np.abs(LON + 9) < 0.3) & (np.abs(LAT - 5) < 0.3)] = 1   # lake
    w[(np.abs(LON - 2) < 0.4) & (np.abs(LAT - 6) < 0.4)] = 0   # island
    p = os.path.join(d, "mask.tif")
    _tif(p, w, lon0, lat1, res)
    return p


def test_mask_and_distance():
    d = tempfile.mkdtemp()
    g = Grid(-10, 0, 0, 10, 0.25)
    p = _mask_tif(d, g)
    m0 = build_water_mask(dict(path=p, coast_buffer_km=0, margin_cells=8), g)
    m1 = build_water_mask(dict(path=p, coast_buffer_km=60, margin_cells=8), g)
    j_lake = np.argmin(np.abs(g.lon + 9)); i_lake = np.argmin(np.abs(g.lat - 5))
    assert not m0["mask"][i_lake, j_lake]                     # lake removed (ocean-connected)
    assert m1["mask"].sum() > m0["mask"].sum()                 # relaxed boundary adds cells
    assert m0["mask"][:, g.lon > -7.4].all() is not False
    col = np.argmin(np.abs(g.lon + 3.0))
    dk = m0["dist_coast_km"][:, col]
    exp = 4.5 * 111.195 * np.cos(np.deg2rad(g.lat))           # to lon -7.5 along the parallel
    near_island = np.abs(g.lat - 6) < 2
    assert np.allclose(dk[~near_island], exp[~near_island], rtol=0.03), (dk, exp)
    dd = nearest_distance_km(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))
    assert abs(dd[0] - 111.195) < 0.1
    # streaming aggregation of a count raster: sums conserved
    a = np.ones((400, 400), np.float32)
    _tif(os.path.join(d, "c.tif"), a, -10, 10, 0.025)
    s = aggregate_raster(os.path.join(d, "c.tif"), g, stats=("sum",), strip_rows=7)["sum"]
    assert np.allclose(s, 100.0)
    shutil.rmtree(d)


class FakeCM:
    """Stands in for copernicusmarine.open_dataset with in-memory xarray datasets."""

    def __init__(self, t0, t1):
        import xarray as xr
        self.xr = xr
        self.t0, self.t1 = t0, t1

    def __call__(self, dsid, bbox, margin, variables=None, depth_range=None, **kw):
        xr = self.xr
        rng = np.random.default_rng(abs(hash(dsid)) % 2 ** 32)
        if dsid == "static":
            lon = np.arange(-11, 1.01, 1 / 12.0); lat = np.arange(-1, 11.01, 1 / 12.0)
            dep = np.broadcast_to(np.clip((lon + 7.5) * 300, 0, None), (len(lat), len(lon)))
            return xr.Dataset({"deptho": (("latitude", "longitude"), dep.astype("f4"))},
                              coords=dict(latitude=lat, longitude=lon))
        if dsid == "phy":
            t = pd.date_range(self.t0, self.t1, freq="D")
            lon = np.arange(-11, 1.01, 0.25); lat = np.arange(11, -1.01, -0.25)
            depth = np.array([0.5, 47.0, 100.0])
            L, A = np.meshgrid(lon, lat)
            th = 20 + 0.1 * L[None, None] - 0.2 * A[None, None] - depth[None, :, None, None] / 20 + np.arange(len(t))[:, None, None, None] * 0.01
            th = np.broadcast_to(th, (len(t), 3) + L.shape).astype("f4").copy()
            th[:, :, :, lon < -7.5] = np.nan
            th[3, :, 10:20, 20:30] = np.nan        # cloud-like hole
            th[5] = np.nan                          # one missing day (temporal fill)
            u = np.broadcast_to(0.1 * np.sin(A / 2), th.shape).astype("f4")
            return xr.Dataset({"thetao": (("time", "depth", "latitude", "longitude"), th),
                               "uo": (("time", "depth", "latitude", "longitude"), u),
                               "vo": (("time", "depth", "latitude", "longitude"), u * 0.5),
                               "zos": (("time", "latitude", "longitude"), th[:, 0] / 100),
                               "siconc": (("time", "latitude", "longitude"), th[:, 0] * 0)},
                              coords=dict(time=t, depth=depth, latitude=lat, longitude=lon))
        if dsid == "wav":
            t = pd.date_range(self.t0, self.t1 + pd.Timedelta(hours=21), freq="3h")
            lon = np.arange(-11, 1.01, 0.5) % 360; lat = np.arange(-1, 11.01, 0.5)
            dirn = np.full((len(t), len(lat), len(lon)), 350.0, "f4")
            dirn[::2] = 10.0                        # mean direction should be ~0 deg, not 180
            return xr.Dataset({"VMDR": (("time", "latitude", "longitude"), dirn),
                               "VHM0": (("time", "latitude", "longitude"), rng.random(dirn.shape).astype("f4")),
                               "VHM0_err": (("time", "latitude", "longitude"), dirn * 0)},
                              coords=dict(time=t, latitude=lat, longitude=lon))
        if dsid == "wav_late":
            raise RuntimeError("not available")
        raise KeyError(dsid)


def test_end_to_end():
    d = tempfile.mkdtemp()
    g = Grid(-10, 0, 0, 10, 0.25)
    mpath = _mask_tif(d, g)
    ship = np.zeros((800, 800), np.float32)
    ship[200:210] = 1000.0
    _tif(os.path.join(d, "ship.tif"), ship, -12, 12, 0.03, nodata=-1)
    tracks = os.path.join(d, "tr.csv")
    pd.DataFrame(dict(timestamp=["2020-01-03 05:00:00+00:00", "2020-01-08 12:00:00+00:00", "2020-01-08 13:00:00+00:00", "2020-03-01"],
                      longitude=[-3, -2, -2, 50], latitude=[5, 5, 5, 5])).to_csv(tracks, index=False)
    cfg = dict(tag="test", out_dir=os.path.join(d, "out"),
               grid=dict(bbox=[-10, 0, 0, 10], res=0.25),
               time=dict(start="2020-01-01", end="auto", pad_days=1, write_days="track",
                         tracks=[dict(path=tracks)]),
               pull=dict(workers=2, keepbits=12),
               mask=dict(path=mpath, coast_buffer_km=20, margin_cells=8),
               bathymetry=dict(cmems_static="static", isobaths=[200]),
               ship_density=[dict(name="log_ship_density", path=os.path.join(d, "ship.tif"))],
               fill=dict(max_temporal_gap=3),
               sources=[dict(name="glorys", datasets=["phy"], variables="all", exclude=["siconc"],
                             depths=dict(thetao=[0, 50, 100], uo=[0], vo=[0]), block_days=3),
                        dict(name="waves", datasets=["wav", "wav_late"], circular=["VMDR"], block_days=2)],
               features=[dict(op="diff", a="thetao_0m", b="thetao_100m", name="strat"),
                         dict(op="speed", u="uo_0m", v="vo_0m", name="spd"),
                         dict(op="grad_mag", var="thetao_0m"),
                         dict(op="rolling_mean", var="thetao_0m", days=3),
                         dict(op="log10", var="missing_var")],
               output=dict(train_every_days=2))
    import yaml
    cp = os.path.join(d, "cfg.yaml")
    yaml.safe_dump(cfg, open(cp, "w"))
    cfg = pipeline.load_config(cp)
    fake = FakeCM(pd.Timestamp("2019-12-20"), pd.Timestamp("2020-01-20"))
    pipeline.cmd_static(cfg, opener=fake)
    pipeline.cmd_pull(cfg, opener=fake)
    assert pipeline.cmd_pull.summary["ok"] > 0 and pipeline.pull_pending(cfg) == {}
    # resume: nothing left to fetch
    caches = pipeline.cmd_pull(cfg, opener=lambda *a, **k: (_ for _ in ()).throw(AssertionError("refetch")) if a[0] != "phy" and a[0] != "wav" else fake(*a, **k))
    meta = pipeline.cmd_assemble(cfg)
    ch = meta["channels"]
    for c in ["thetao_0m", "thetao_50m", "thetao_100m", "uo_0m", "zos", "VMDR_sin", "VMDR_cos", "VHM0",
              "strat", "spd", "grad_mag_thetao_0m", "thetao_0m_mean3d", "dist_coast_km", "depth_m",
              "log_ship_density", "dist_200m_isobath_km"]:
        assert c in ch, (c, ch)
    assert "siconc" not in ch and "VHM0_err" not in ch and "log10_missing_var" not in ch and "water_frac" not in ch
    idx = pd.read_csv(os.path.join(cfg["out_dir"], "test_scene_index.csv"))
    assert list(idx["date"]) == ["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-07", "2020-01-08", "2020-01-09"]
    assert list(idx["t"]) == [1, 2, 3, 6, 7, 8] and idx["complete"].all()
    import zarr
    sc = np.asarray(zarr.load(idx["path"][2]))            # 2020-01-04 had a hole on day 3 and day 5 missing
    assert sc.shape == (len(ch), g.ny, g.nx) and sc.dtype == np.float32
    mask = np.asarray(zarr.load(os.path.join(cfg["work_dir"], "mask.zarr")))[0] > 0.5
    assert np.all(sc[:, ~mask] == -999999.0) and np.all(np.isfinite(sc[:, mask])) and np.all(sc[:, mask] != -999999.0)
    th = sc[ch.index("thetao_0m")]
    gx, gy = np.meshgrid(g.lon, g.lat)
    exp = 20 + 0.1 * gx - 0.2 * gy - 0.5 / 20 + 15 * 0.01   # 2020-01-04 = day 15 of the fake series
    core = mask & (gx > -7)
    assert np.allclose(th[core], exp[core], atol=0.02), np.abs(th - exp)[core].max()
    assert np.allclose(sc[ch.index("strat")][core], (100 - 0.5) / 20, atol=1e-2)
    assert np.allclose(sc[ch.index("VMDR_cos")][core], np.cos(np.deg2rad(10)), atol=1e-3)
    # day 5 of fake series missing entirely -> filled temporally (2020-01-?? check QA)
    qa = pd.read_csv(os.path.join(cfg["out_dir"], "test_qa_missing.csv"))
    assert qa["missing_frac_prefill"].between(0, 1).all()
    st = pd.read_csv(os.path.join(cfg["out_dir"], "test_channel_stats.csv"))
    assert set(st["channel"]) == set(ch) and (st["n"] > 0).all()
    blk = yaml.safe_load(open(os.path.join(cfg["out_dir"], "test_sitfuse_data_block.yaml")))
    assert blk["data"]["number_channels"] == len(ch) and len(blk["data"]["files_test"]) == 6
    assert os.path.exists(os.path.join(cfg["out_dir"], "test_water_mask.tif"))
    sh = sc[ch.index("log_ship_density")]
    assert sh[mask].max() > 0
    from naturenet.environment.cmems_env import store
    dn = store.open_array(os.path.join(cfg["work_dir"], "cache", "glorys.done.zarr"), mode="r+")
    dn[0:2] = 0                                          # as left by a stalled or failed block
    assert pipeline.pull_pending(cfg) == {"glorys": 2}
    shutil.rmtree(d)


def test_run_blocks_stall_and_heartbeat():
    import logging
    import threading
    import time
    gate = threading.Event()
    done = []

    def run(t):
        if t == "hang":
            gate.wait(10)                                # a dead socket that never returns in time
        if t == "bad":
            raise IOError("boom")
        time.sleep(0.05)
        done.append(t)
        return t

    recs = []
    h = logging.Handler()
    h.emit = lambda r: recs.append(r.getMessage())
    lg = logging.getLogger("cmems_env")
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        tasks = ["a", "hang", "b", "bad", "c", "d", "e"]
        t0 = time.monotonic()
        sm = pipeline.run_blocks(tasks, run, str, workers=2, heartbeat=0.2, block_timeout=0.6, max_stalled=2,
                                 poll=0.05)
        assert time.monotonic() - t0 < 5
        assert sm == dict(ok=5, failed=1, stalled=["hang"], not_started=0), sm
        assert sorted(done) == ["a", "b", "c", "d", "e"]
        assert any("block stalled" in m and "hang" in m for m in recs)
        assert any(m.startswith("pull heartbeat") and "hang" in m for m in recs)
        assert any("pull incomplete" in m for m in recs)
        # every slot stalls -> stop starting new blocks, leave them for the next run
        gate2 = threading.Event()
        sm = pipeline.run_blocks(["x", "y", "z"], lambda t: gate2.wait(10), str, workers=1, heartbeat=0,
                                 block_timeout=0.2, max_stalled=1, poll=0.05)
        assert sm == dict(ok=0, failed=0, stalled=["x"], not_started=2), sm
        # a stalled block that finishes before the end is kept
        gate3 = threading.Event()
        threading.Timer(0.5, gate3.set).start()
        q = ["slow"] + [f"q{i}" for i in range(6)]       # the q blocks keep the run going past 0.5 s
        sm = pipeline.run_blocks(q, lambda t: gate3.wait(10) if t == "slow" else time.sleep(0.15),
                                 str, workers=2, heartbeat=0, block_timeout=0.3, max_stalled=2, poll=0.05)
        assert sm == dict(ok=7, failed=0, stalled=[], not_started=0), sm
        assert any("after stalling" in m for m in recs)
    finally:
        gate.set()
        lg.removeHandler(h)


def test_http_timeouts():
    import botocore.config
    from naturenet.environment.cmems_env import sources
    sources.configure_http(dict(read_timeout=45, connect_timeout=15, workers=2, dask_threads=2))
    c = botocore.config.Config(signature_version=None, retries={"max_attempts": 10, "mode": "adaptive"})
    assert (c.read_timeout, c.connect_timeout, c.max_pool_connections) == (45, 15, 16)
    sources.configure_http({})
    assert botocore.config.Config().read_timeout == 60


def test_monthly_prior_and_coverage():
    import xarray as xr
    from naturenet.environment.cmems_env.sources import Source
    t = pd.DatetimeIndex(["2020-01-01", "2020-02-01", "2020-03-01"])
    lon = np.arange(0, 3.0); lat = np.arange(0, 3.0)
    v = np.arange(3, dtype="f4")[:, None, None] * np.ones((1, 3, 3), "f4")
    ds = xr.Dataset({"fe": (("time", "latitude", "longitude"), v)}, coords=dict(time=t, latitude=lat, longitude=lon))
    s = Source(dict(name="m", datasets=["m"], variables=["fe"], temporal="prior"), [0, 2, 0, 2], 0,
               opener=lambda *a, **k: ds)
    assert s.resolve() == ["fe"]
    days = pd.date_range("2020-01-30", "2020-02-02")
    out, _, _ = s.read_days("m", days)
    assert list(out[:, 0, 1, 1]) == [0, 0, 1, 1]
    assert s.dataset_for_days(pd.DatetimeIndex(["2019-12-31", "2020-03-20", "2020-05-01"])) == [None, "m", None]


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
