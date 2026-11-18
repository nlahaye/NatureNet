"""
Synthetic checks for partial-run outputs (no data, no GPU):
    python -m naturenet.models.irl.test_auto_animation_jobs
  - auto_animation_jobs: finals series discovery, scales variants with partial folds, N1 as the reference panel
  - summarize_experiments.t2_scales: variants with fewer folds than expected are marked "partial"
"""
import os
import tempfile

import numpy as np
import pandas as pd
import yaml

from naturenet.models.irl import auto_animation_jobs as aj


def _touch(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()


def test_jobs():
    with tempfile.TemporaryDirectory() as d:
        fin = os.path.join(d, "final", "hiql_kin")
        _touch(os.path.join(fin, "plots", "series_hiavi_s1", "a_time_series_pd.csv"))
        np.save(os.path.join(fin, "plots", "state_reward_hiavi_s1.npy"), np.zeros(3))
        sw = os.path.join(d, "final", "swirl_kin")
        for o in ("s1", "s2"):
            _touch(os.path.join(sw, "plots", f"series_linear_{o}", "a_time_series_pd.csv"))
        os.makedirs(os.path.join(sw, "plots", "series_empty"))            # no CSVs: ignored
        sc = os.path.join(d, "scales_kin")
        conf = dict(out_dir=sc, run_uid="RU", n_hidden=5, seed=12345)
        folds = {"N1": [0, 1, 2], "N2": [0, 1], "N1_R2": [0], "N3": [1], "N1_dur-logi12": [0, 2], "N2_hist": [0]}
        for v, ks in folds.items():
            for k in ks:
                _touch(os.path.join(sc, f"RU_SC_{v}_5_12345{aj.TEST_SUFFIX}_KF_{k}.npz"))
        _touch(os.path.join(sc, "RU_SC_N1_5_12345_naturenet_S1_iter2_LL1_train_KF_0.npz"))   # not a test file
        y = os.path.join(d, "scales.yaml")
        yaml.safe_dump(conf, open(y, "w"))
        base = dict(root=d, select=dict(n=3), animations=[dict(name="x")])
        out = aj.build(base, finals=[("hiql_kin", fin), ("swirl_kin", sw + "/")], scales=[("swirl", y)], max_panels=4)
        names = [j["name"] for j in out["animations"]]
        assert "animations" in out and out["root"] == d and "x" not in names, names
        assert "auto_final_hiql_kin_hiavi_s1_by_mode" in names and "auto_final_swirl_kin_compare" in names, names
        bm = [j for j in out["animations"] if j["name"] == "auto_final_hiql_kin_hiavi_s1_by_mode"][0]
        assert bm["background"] == "reward" and bm["panels"][0]["state_reward"].endswith("state_reward_hiavi_s1.npy")
        nob = [j for j in out["animations"] if j["name"] == "auto_final_swirl_kin_linear_s1_by_mode"][0]
        assert nob["background"] == "env" and "state_reward" not in nob["panels"][0]
        assert [len(j["panels"]) for j in out["animations"] if j["name"] == "auto_finals_compare"] == [2]
        scj = [j for j in out["animations"] if j["name"].startswith("auto_scales_swirl_")]
        assert len(scj) == 2, scj                                       # 5 non-N1 variants, 3 per job
        for j in scj:
            assert j["panels"][0]["name"].startswith("swirl N1 (3 folds)"), j["panels"][0]
            assert len(j["panels"]) <= 4
        allp = [p for j in scj for p in j["panels"][1:]]
        assert sorted(p["name"].split()[1] for p in allp) == sorted(v for v in folds if v != "N1")
        assert all(p["regime"] == ("_R2" in p["path"]) for p in allp)
        assert aj.scales_variants_on_disk(conf, min_folds=2) == [("N1", 3), ("N1_dur-logi12", 2), ("N2", 2)]
        # cv-table order is kept (N1 first)
        pd.DataFrame(dict(variant=["N1", "N3", "N2", "N1_R2"], fold=0)).to_csv(os.path.join(sc, "RU_scales_cv.csv"))
        assert [v for v, _ in aj.scales_variants_on_disk(conf)][:4] == ["N1", "N3", "N2", "N1_R2"]
        o = os.path.join(d, "auto.yaml")
        yaml.safe_dump(base, open(os.path.join(d, "base.yaml"), "w"))
        aj.main(["-b", os.path.join(d, "base.yaml"), "--scales", f"swirl={y}", "--out", o])
        assert yaml.safe_load(open(o))["animations"][0]["name"] == "auto_scales_swirl_0"
    print("ok test_jobs")


def test_t2_partial():
    from naturenet.models.irl.summarize_experiments import t2_scales
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.csv")
        pd.DataFrame(dict(variant=["N1", "N2", "N3", "N1_R2"], gain=[0, 0.5, 0.5, 0.01], gain_se=[0, 0.1, 0.1, 0.1],
                          folds_won=[0, 5, 2, 1], n_folds=[5, 5, 2, 5])).to_csv(p, index=False)
        s = t2_scales([("a", p)], expected_folds=5).set_index("variant")
        assert s.loc["N1", "keep"] == "ref" and s.loc["N2", "keep"] == "yes"
        assert s.loc["N3", "keep"] == "partial" and s.loc["N1_R2", "keep"] == "no", s["keep"].to_dict()
    print("ok test_t2_partial")


if __name__ == "__main__":
    test_jobs()
    test_t2_partial()
