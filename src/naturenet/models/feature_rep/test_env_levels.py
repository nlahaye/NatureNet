"""
env_data_format environment levels (cluster_level fine / parent / coarse) on SIT-FUSE-style rasters
(value = (num_child * parent + child) / 1000, no data = -0.001), plus make_level_configs on the shipped specs.

    python -m naturenet.models.feature_rep.test_env_levels
"""
import glob
import json
import os
import pickle
import tempfile

import numpy as np
import yaml

import naturenet.models.feature_rep.env_data_format as edf
from naturenet.models.feature_rep import test_env_data_format_v2 as base

NC = 5


def _sitfuse_rasters(tmp, rng):
    """Rewrite the label tifs of the base fixture as heir rasters; returns {t: fine ids} (-1 = invalid)."""
    ids = {}
    for p in sorted(glob.glob(os.path.join(tmp, "cop_env_x_t*.zarr.clust.data_*clusters.no_geo.tif"))):
        t = int(os.path.basename(p).split("_t")[1].split(".")[0])
        par = rng.integers(0, 6, size=(base.H, base.W))
        par[0, 5] = 0                                   # make sure id 0 occurs
        child = rng.integers(0, NC, size=par.shape)
        child[0, 5] = 0
        fine = NC * par + child
        fine[1, 4] = -1                                 # SIT-FUSE no data
        v = (fine / 1000.0).astype(np.float32)
        base._tif(p, v)
        ids[t] = fine
    return ids


def _conf(tmp, out, **kw):
    c = dict(df_uid="uid", df_dir=tmp, out_dir=out, output_uid="final_env_maps_",
             scene_index_csv=os.path.join(tmp, "x_scene_index.csv"), allow_missing_scenes=True,
             scene_file_template=os.path.join(tmp, "cop_env_x_t{i}.zarr.clust.data_*clusters.no_geo.tif"),
             mask_path=os.path.join(tmp, "mask.tif"), land_value=0, land_fill=-1, sit_fuse_output=True,
             cluster_scale=1000, zero_label_valid=True, num_child=NC)
    c.update(kw)
    os.makedirs(out, exist_ok=True)
    return c


def _run(conf):
    edf.run(conf)
    with open(os.path.join(conf["out_dir"], "final_env_maps_uid.pkl"), "rb") as f:
        maps = pickle.load(f)
    with open(os.path.join(conf["out_dir"], "final_env_maps__class_mapper.pkl"), "rb") as f:
        mapper = pickle.load(f)
    return maps, mapper


def test_l1_level_ids():
    v = np.array([-0.001, 0.0, 0.004, 0.005, 0.029, np.nan, -1.0])
    fine = edf.level_ids(v, dict(level="fine", scale=1000, zero_valid=True))
    assert fine.tolist() == [-1, 0, 4, 5, 29, -1, -1]
    legacy = edf.level_ids(v, dict(level="fine", scale=1000, zero_valid=False))
    assert legacy.tolist() == [-1, -1, 4, 5, 29, -1, -1]
    sp = dict(level="parent", scale=1000, zero_valid=True, num_child=NC, label_input="fine")
    assert edf.level_ids(v, sp).tolist() == [-1, 0, 0, 1, 5, -1, -1]
    co = dict(sp, level="coarse", keys=np.array([0, 1, 5]), vals=np.array([1, 0, 1]), unmapped="error",
              hierarchy_json="h")
    assert edf.level_ids(v, co).tolist() == [-1, 1, 1, 0, 1, -1, -1]
    try:
        edf.level_ids(np.array([0.030]), co)
        raise AssertionError("expected unmapped error")
    except ValueError:
        pass
    assert edf.level_ids(np.array([0.030]), dict(co, unmapped="invalid")).tolist() == [-1]


def test_l2_levels_nest():
    with tempfile.TemporaryDirectory() as tmp:
        base._setup(tmp)
        ids = _sitfuse_rasters(tmp, np.random.default_rng(1))
        cut = {p: (0 if p in (0, 3) else 1 if p in (1, 4) else 2) for p in range(6)}
        hj = os.path.join(tmp, "cluster_hierarchy.json")
        with open(hj, "w") as f:
            json.dump(dict(num_child=NC, label_scale=1000, cuts={"3": {str(p): c for p, c in cut.items()}}), f)
        mf, kf = _run(_conf(tmp, os.path.join(tmp, "fine")))
        mp, kp = _run(_conf(tmp, os.path.join(tmp, "parent"), cluster_level="parent"))
        mc, kc = _run(_conf(tmp, os.path.join(tmp, "coarse"), cluster_level="coarse", hierarchy_json=hj, level_k=3))
        assert 0.0 in kf, "fine id 0 kept with zero_label_valid"
        assert len([k for k in kc if k >= 0]) == 3 and len([k for k in kp if k >= 0]) == 6
        inv = lambda m: {v: k for k, v in m.items() if v >= 0}
        if_, ip, ic = inv(kf), inv(kp), inv(kc)
        n = 0
        for seg_f, seg_p, seg_c in zip(mf["A"], mp["A"], mc["A"]):
            for a, b, c in zip(seg_f, seg_p, seg_c):
                if a is None:
                    assert b is None and c is None
                    continue
                assert ((a < 0) == (b < 0)).all() and ((a < 0) == (c < 0)).all()
                ok = a >= 0
                fine = np.vectorize(if_.get)(a[ok]).astype(int)
                par = np.vectorize(ip.get)(b[ok]).astype(int)
                co = np.vectorize(ic.get)(c[ok]).astype(int)
                assert (fine // NC == par).all() and (np.vectorize(cut.get)(par) == co).all()
                n += ok.sum()
        assert n > 0
        with open(os.path.join(tmp, "coarse", "final_env_maps__level.json")) as f:
            info = json.load(f)
        assert info["level"] == "coarse" and info["level_k"] == 3 and info["n_clusters"] == 3
        try:
            _run(_conf(tmp, os.path.join(tmp, "bad"), cluster_level="coarse", hierarchy_json=hj, level_k=4))
            raise AssertionError("expected missing cut")
        except ValueError:
            pass


def test_l3_make_level_configs():
    from naturenet.environment import make_level_configs as mlc
    from naturenet.environment.sync_v2_configs import patch_text
    pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for sp in ("sharks", "whales"):
        spec_path = os.path.join(pkg, "config", "v2", "levels", f"{sp}_levels_v2.yaml")
        with open(spec_path) as f:
            spec = yaml.safe_load(f)
        with tempfile.TemporaryDirectory() as tmp:
            spec["out_config_dir"] = tmp
            w = mlc.generate(spec, spec_path, pkg)
            for p, t in w.items():
                c = yaml.safe_load(t)
                name = os.path.basename(p)
                if name in ("hiavi_kfold.yaml", "iavi_kfold.yaml", "swirl_kfold.yaml", "hiql_final.yaml"):
                    lv = os.path.basename(os.path.dirname(p))
                    assert c["n_states"] is None
                    assert c["df_dir"] == spec["levels_dir"] + lv + "/"
                    assert c["actions"].startswith(spec["levels_dir"] + lv + "/")
                    assert c["folds_dir"] == spec["base_dir"] + "folds/"          # shared folds
                    assert c["run_uid"].endswith("_" + lv.upper())
                    assert c["out_dir"].startswith(spec["levels_dir"] + lv + "/")
                    new, n = patch_text(t, dict(n_states="21"))
                    assert n >= 1 and yaml.safe_load(new)["n_states"] == 21
                if name == "env_format.yaml":
                    assert c["cluster_level"] in ("parent", "coarse") and c["zero_label_valid"] is True
                    if c["cluster_level"] == "coarse":
                        assert c["level_k"] in [lv["k"] for lv in spec["levels"] if lv["level"] == "coarse"]
                if name == "reward_env_projection_hiql.yaml":
                    r = c["reward_env_projection"]
                    assert "/levels/" in r["class_mapper"] and "/xai/interpret_" in r["profiles_csv"]
                if name == "ll_comparison_levels.yaml":
                    aliases = [m.get("alias", m["name"]) for m in c["methods"]]
                    assert len(aliases) == len(set(aliases))
                    for m in c["methods"]:
                        if m["yaml"].startswith("../"):      # fine baseline, relative to the real out_config_dir
                            y = os.path.normpath(os.path.join(pkg, "config", "v2", "levels", sp, m["yaml"]))
                            assert os.path.exists(y), y
                        else:
                            assert os.path.join(tmp, m["yaml"]) in w, m["yaml"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
