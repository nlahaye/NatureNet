"""
env_window in swirl_track_series: fixes outside the scene calendar are dropped from the series, fixes inside it
with the absorbing (no env data) state are kept.

    python -m naturenet.models.irl.test_env_window
"""
import os
import pickle
import tempfile

import numpy as np
import pandas as pd

import naturenet.models.irl.swirl_track_series as sts


class _R:
    def lookup(self, k, s, p, a):
        return np.where(k >= 0, s.astype(float), np.nan)


def _setup(tmp, absorbing=74):
    # one animal, one segment: 10 daily fixes before the calendar, 10 inside (2 absorbing), 5 after
    t = pd.date_range("2023-10-22 06:00", periods=10 + 10 + 5, freq="D")
    n = len(t)
    df = pd.DataFrame(dict(timestamp=t, latitude=np.linspace(25, 26, n), longitude=np.linspace(-80, -79, n)))
    with open(os.path.join(tmp, "uid_dfs.pkl"), "wb") as f:
        pickle.dump({"183623": [df]}, f)
    states = np.full(n, absorbing)
    states[10:20] = np.arange(10)
    states[13] = states[16] = absorbing           # small in-window patches (land / nodata)
    segs = [dict(key="183623", seg=0, states=states, actions=np.zeros(n - 1, int))]
    lat = dict(zs=[np.zeros(n - 1, int)], seq_key=np.array(["183623"]), seq_seg=np.array([0]),
               seq_start=np.array([0]), seq_offset=np.array([0]), split=np.array(["train"]))
    return segs, lat, t


def _run(conf, segs, lat):
    orig_seg, orig_lay = sts.load_segments, sts.load_layers
    sts.load_segments, sts.load_layers = (lambda c: segs), (lambda c: {})
    try:
        return sts.build_track_series(conf, rewards=_R(), latents=lat)
    finally:
        sts.load_segments, sts.load_layers = orig_seg, orig_lay


def test_w1_bounds():
    lo, hi = sts.env_window_bounds({"env_window": {"start_time": "2023-11-01", "n_days": 345}})
    assert lo == pd.Timestamp("2023-11-01") and hi == pd.Timestamp("2024-10-11"), (lo, hi)
    lo, hi = sts.env_window_bounds({"env_window": {"start_time": "2023-11-01T00:00:00+00:00", "n_days": 10,
                                                   "nearest_days": 1}})
    assert lo == pd.Timestamp("2023-10-31") and hi == pd.Timestamp("2023-11-12")
    assert sts.env_window_bounds({}) is None


def test_w2_series_filter():
    with tempfile.TemporaryDirectory() as tmp:
        segs, lat, t = _setup(tmp)
        base = dict(df_run_uid="uid", df_dir=tmp + "/")
        full = _run(base, segs, lat)["183623"]
        assert len(full) == 24                    # last fix has no latent (offset 0, n-1 steps)
        win = _run(dict(base, env_window={"start_time": "2023-11-01", "n_days": 10}), segs, lat)["183623"]
        assert len(win) == 10, len(win)
        assert win["timestamp"].min() >= pd.Timestamp("2023-11-01")
        assert win["timestamp"].max() < pd.Timestamp("2023-11-11")
        assert (win["env"] == 74).sum() == 2      # in-window absorbing patches kept
        assert win["Time"].iloc[0] == 0.0
        try:
            _run(dict(base, env_window={"start_time": "2025-01-01", "n_days": 10}), segs, lat)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_w3_scene_index_csv():
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "idx.csv")
        days = pd.date_range("2023-11-01", periods=12, freq="D")
        pd.DataFrame(dict(t=range(12), date=days.strftime("%Y-%m-%d"), path="x",
                          complete=[False] + [True] * 10 + [False])).to_csv(csv, index=False)
        lo, hi = sts.env_window_bounds({"env_window": {"scene_index_csv": csv}})
        assert lo == pd.Timestamp("2023-11-02") and hi == pd.Timestamp("2023-11-12"), (lo, hi)
        lo, hi = sts.env_window_bounds({"env_window": {"scene_index_csv": csv, "complete_scenes_only": False}})
        assert lo == pd.Timestamp("2023-11-01") and hi == pd.Timestamp("2023-11-13")
        segs, lat, t = _setup(tmp)
        win = _run(dict(df_run_uid="uid", df_dir=tmp + "/", env_window={"scene_index_csv": csv}), segs, lat)["183623"]
        assert win["timestamp"].min() >= pd.Timestamp("2023-11-02") and win["timestamp"].max() < pd.Timestamp("2023-11-12")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
