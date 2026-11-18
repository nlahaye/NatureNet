"""
Used-vs-available points (step-selection design) for the environment-geometry analysis.

For every real fix t >= 1 of every original track, m available points are drawn from the fix t-1 with the empirical
step-length and turning-angle distributions pooled over all original tracks (heading of t-2 -> t-1; uniform heading
at t = 1). Available point j of fix t is row t of segment s of the pseudo-track "<key>__avl<j>", so after
sample_env_inputs its environment aligns position-wise with the used fix (row 0 = the used fix 0).

Writes into avl_dir:
  <df_run_uid>_dfs.pkl                       {<key>__avl<j>: [segment DataFrames (timestamp, latitude, longitude, used_fix)]}
  <df_run_uid>[_<key>__avl<j>]_grid.pkl      links to the movement run's grid file(s)
  sample_inputs_avl.yaml                     sample_env_inputs config (same channels as the movement run)
  availability_meta.json
Then: python -m naturenet.environment.sample_env_inputs -y <avl_dir>/sample_inputs_avl.yaml

YAML (or the "availability" block of the geometry spec or of a habitat_tests spec):
  sample_inputs_yaml: config/v2/environment/whales_sample_inputs_v2.yaml   # df_dir, df_run_uid, channels, scenes
  avl_dir: /data/nlahaye/NatureNet/Blue_Whale_v2/drivers/availability/
  n_available: 10
  seed: 0
  max_step_km: null            # optional cap on the sampled step lengths

    python -m naturenet.drivers.availability -y config/v2/drivers/generated/whales/specs/geometry.yaml
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from naturenet.drivers.common import abs_cfg, atomic_json, is_original, read_yaml, write_yaml

R = 6371.0088


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


def _dist_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _dest(lat, lon, brg, d_km):
    p1, l1, dr = np.deg2rad(lat), np.deg2rad(lon), d_km / R
    p2 = np.arcsin(np.sin(p1) * np.cos(dr) + np.cos(p1) * np.sin(dr) * np.cos(brg))
    l2 = l1 + np.arctan2(np.sin(brg) * np.sin(dr) * np.cos(p1), np.cos(dr) - np.sin(p1) * np.sin(p2))
    return np.rad2deg(p2), (np.rad2deg(l2) + 540) % 360 - 180


def step_turn(seg):
    lat, lon = seg["latitude"].to_numpy(float), seg["longitude"].to_numpy(float)
    st = _dist_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    b = _bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
    tu = (np.diff(b) + np.pi) % (2 * np.pi) - np.pi
    return st, tu, b


def make_available(dfs, m=10, seed=0, max_step_km=None):
    rng = np.random.default_rng(seed)
    steps, turns = [], []
    for k, segs in dfs.items():
        if not is_original(k):
            continue
        for seg in segs:
            if len(seg) >= 3:
                s, t, _ = step_turn(seg)
                steps.append(s[np.isfinite(s)])
                turns.append(t[np.isfinite(t)])
    steps, turns = np.concatenate(steps), np.concatenate(turns)
    if max_step_km:
        steps = steps[steps <= float(max_step_km)]
    out = {}
    for k, segs in dfs.items():
        if not is_original(k):
            continue
        for j in range(m):
            out[f"{k}__avl{j}"] = []
        for seg in segs:
            n = len(seg)
            lat, lon = seg["latitude"].to_numpy(float), seg["longitude"].to_numpy(float)
            if n >= 2:
                _, _, b = step_turn(seg)
                head = np.concatenate([[np.nan], b[:-1]])          # heading into fix t-1, for t = 1 .. n-1
            for j in range(m):
                la, lo = lat.copy(), lon.copy()
                used = np.zeros(n, bool)
                used[:1] = True
                if n >= 2:
                    d = rng.choice(steps, n - 1)
                    h = np.where(np.isfinite(head), head + rng.choice(turns, n - 1), rng.uniform(-np.pi, np.pi, n - 1))
                    la[1:], lo[1:] = _dest(lat[:-1], lon[:-1], h, d)
                out[f"{k}__avl{j}"].append(pd.DataFrame(dict(timestamp=seg["timestamp"].to_numpy(), latitude=la,
                                                             longitude=lo, used_fix=used)))
    return out, dict(n_steps=int(len(steps)), step_km_median=float(np.median(steps)), n_available=int(m),
                     seed=int(seed))


def _link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def build(spec):
    spec = spec.get("habitat_tests", spec)       # habitat_tests specs carry the same availability block
    av = spec.get("availability", spec)
    sconf = read_yaml(abs_cfg(av["sample_inputs_yaml"]))
    df_dir, uid = sconf["df_dir"], sconf["df_run_uid"]
    out_dir = av["avl_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(df_dir, uid + "_dfs.pkl"), "rb") as f:
        dfs = pd.read_pickle(f)
    avl, meta = make_available(dfs, int(av.get("n_available", 10)), int(av.get("seed", 0)), av.get("max_step_km"))
    tmp = os.path.join(out_dir, uid + "_dfs.pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(avl, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, os.path.join(out_dir, uid + "_dfs.pkl"))
    shared = os.path.join(df_dir, uid + "_grid.pkl")
    if os.path.exists(shared):
        _link(shared, os.path.join(out_dir, uid + "_grid.pkl"))
    for k in avl:
        src = os.path.join(df_dir, f"{uid}_{k.split('__avl')[0]}_grid.pkl")
        if os.path.exists(src):
            _link(src, os.path.join(out_dir, f"{uid}_{k}_grid.pkl"))
    sc = dict(sconf)
    sc["df_dir"] = out_dir
    sc["uids"] = None
    sc.pop("channels_out", None)
    sc.pop("input_channels_request", None)
    write_yaml(sc, os.path.join(out_dir, "sample_inputs_avl.yaml"),
               header="# generated by naturenet.drivers.availability (available points of the step-selection design)\n")
    meta.update(source_df_dir=df_dir, df_run_uid=uid, n_keys=len(avl))
    atomic_json(meta, os.path.join(out_dir, "availability_meta.json"))
    print(f"availability: {len(avl)} pseudo-tracks -> {out_dir}; next: python -m naturenet.environment.sample_env_inputs"
          f" -y {os.path.join(out_dir, 'sample_inputs_avl.yaml')}")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    build(read_yaml(abs_cfg(ap.parse_args(argv).yaml)))


if __name__ == "__main__":
    main()
