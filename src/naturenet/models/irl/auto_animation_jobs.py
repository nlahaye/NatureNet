"""
Animation jobs for whatever fitted outputs exist: writes an animate_tracks YAML with one job group per final run
(every exported series_<tag>/ dir) and per decision-scales run (every variant with at least one out-of-fold test
file). Generic: any SIT-FUSE / NatureNet use case with plot_* series exports or scales test files.

    python -m naturenet.models.irl.auto_animation_jobs -b config/v2/plots/whales_animations.yaml \
        --finals hiql_kin=/data/.../final/hiql_kin swirl_kin=/data/.../final/swirl_kin \
        --scales swirl_kin=config/v2/swirl/whales_swirl_scales_kin.yaml \
        --out /data/.../animations/auto_animations.yaml [--max_panels 4] [--min_folds 1]
    python -m naturenet.models.irl.animate_tracks -y /data/.../animations/auto_animations.yaml

Jobs written (top-level keys of the base YAML are copied: root, dfs_pkl, env_maps, select, ...):
  auto_final_<label>_<tag>_by_mode   one animal per decoded mode, the mode's state reward as background
                                     (env background when state_reward_<tag>.npy is missing)
  auto_final_<label>_compare         every series of the run side by side (up to max_panels; S-1 / S-2 views)
  auto_finals_compare                the first series of every final run side by side (up to max_panels)
  auto_scales_<label>_<i>            N1 (reference panel) + up to max_panels - 1 other variants; regime strips for
                                     the _R<n> variants. With partial CV, only folds already written are decoded, and
                                     animate_tracks chooses among the animals every panel has (common_animals).
Test-file names follow run_swirl_scales_kfold: <out_dir>/<run_uid>_SC_<variant>_<K>_<seed>_naturenet_S1_iter2_LL1_test_KF_<k>.npz.
"""
import argparse
import glob
import os
import re

import yaml

TEST_SUFFIX = "_naturenet_S1_iter2_LL1_test"


def _read(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _kv(items):
    out = []
    for s in items or []:
        lab, _, val = s.partition("=")
        if not val:
            lab, val = os.path.basename(os.path.normpath(s)), s
        out.append((lab, val))
    return out


def scales_variants_on_disk(conf, min_folds=1):
    """[(variant, n_test_folds)] for one scales config; N1 first, then the cv-table order (or name order)."""
    out_dir, uid = conf["out_dir"], conf["run_uid"]
    K, seed = int(conf["n_hidden"]), int(conf["seed"])
    pat = os.path.join(out_dir, f"{uid}_SC_*_{K}_{seed}{TEST_SUFFIX}_KF_*.npz")
    rx = re.compile(re.escape(f"{uid}_SC_") + r"(.+)_" + f"{K}_{seed}" + re.escape(TEST_SUFFIX) + r"_KF_(\d+)\.npz$")
    folds = {}
    for f in glob.glob(pat):
        m = rx.match(os.path.basename(f))
        if m:
            folds.setdefault(m.group(1), set()).add(int(m.group(2)))
    order = list(folds)
    cv = os.path.join(out_dir, uid + "_scales_cv.csv")
    if os.path.exists(cv):
        try:
            import pandas as pd
            seen = list(dict.fromkeys(pd.read_csv(cv)["variant"].astype(str)))
            order = [v for v in seen if v in folds] + sorted(v for v in folds if v not in seen)
        except Exception:  # noqa: BLE001
            order = sorted(folds)
    else:
        order = sorted(folds)
    order = sorted(order, key=lambda v: v != "N1")
    return [(v, len(folds[v])) for v in order if len(folds[v]) >= min_folds]


def scales_jobs(label, conf, max_panels=4, min_folds=1):
    K, seed, uid = int(conf["n_hidden"]), int(conf["seed"]), conf["run_uid"]
    vs = scales_variants_on_disk(conf, min_folds)
    if not vs:
        return []
    panel = lambda v, n: dict(  # noqa: E731
        name=f"{label} {v} ({n} fold{'s' if n != 1 else ''})", kind="latents",
        path=os.path.join(conf["out_dir"], f"{uid}_SC_{v}_{K}_{seed}{TEST_SUFFIX}_KF_*.npz"),
        regime=bool(re.search(r"_R\d+", v)))
    ref = [x for x in vs if x[0] == "N1"]
    rest = [x for x in vs if x[0] != "N1"]
    if not ref:                                   # no N1 yet: the first variant is the reference
        ref, rest = vs[:1], vs[1:]
    step = max(int(max_panels) - 1, 1)
    chunks = [rest[i:i + step] for i in range(0, len(rest), step)] or [[]]
    return [dict(name=f"auto_scales_{label}_{i}", panels=[panel(*ref[0])] + [panel(*x) for x in ch])
            for i, ch in enumerate(chunks)]


def final_series(final_dir):
    """[(tag, series_dir, state_reward or None)] for every plots/series_<tag>/ dir with time-series CSVs."""
    out = []
    for d in sorted(glob.glob(os.path.join(final_dir.rstrip("/"), "plots", "series_*"))):
        if not os.path.isdir(d) or not glob.glob(os.path.join(d, "*_time_series_pd.csv")):
            continue
        tag = os.path.basename(d)[len("series_"):]
        sr = os.path.join(final_dir, "plots", f"state_reward_{tag}.npy")
        out.append((tag, d, sr if os.path.exists(sr) else None))
    return out


def final_jobs(label, final_dir, max_panels=4):
    ser = final_series(final_dir)
    jobs = []
    pan = lambda tag, d, sr: dict(name=f"{label} {tag}", kind="series", path=d,  # noqa: E731
                                  **({"state_reward": sr} if sr else {}))
    for tag, d, sr in ser:
        jobs.append(dict(name=f"auto_final_{label}_{tag}_by_mode", background="reward" if sr else "env",
                         select={"strategy": "by_mode", "max_frames": 160}, panels=[pan(tag, d, sr)]))
    if len(ser) > 1:
        jobs.append(dict(name=f"auto_final_{label}_compare", panels=[pan(*x) for x in ser[:max_panels]]))
    return jobs, ([pan(*ser[0])] if ser else [])


def build(base, finals=(), scales=(), max_panels=4, min_folds=1):
    conf = {k: v for k, v in base.items() if k != "animations"}
    jobs, firsts = [], []
    for lab, d in finals:
        j, f = final_jobs(lab, d, max_panels)
        jobs += j
        firsts += f
        if not j:
            print(f"auto_animation_jobs: no series under {d.rstrip('/')}/plots (skipped {lab})")
    if len(firsts) > 1:
        jobs.append(dict(name="auto_finals_compare", panels=firsts[:max_panels]))
    for lab, y in scales:
        c = _read(y)
        j = scales_jobs(lab, c, max_panels, min_folds)
        print(f"auto_animation_jobs: {lab}: {sum(len(x['panels']) for x in j)} scales panels in {len(j)} jobs"
              if j else f"auto_animation_jobs: {lab}: no test files yet")
        jobs += j
    conf["animations"] = jobs
    return conf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-b", "--base", required=True, help="species animations YAML (top-level defaults)")
    ap.add_argument("--finals", nargs="*", default=[], help="label=final run dir (with plots/series_*)")
    ap.add_argument("--scales", nargs="*", default=[], help="label=scales config YAML")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_panels", type=int, default=4)
    ap.add_argument("--min_folds", type=int, default=1)
    a = ap.parse_args(argv)
    conf = build(_read(a.base), _kv(a.finals), _kv(a.scales), a.max_panels, a.min_folds)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("# generated by naturenet.models.irl.auto_animation_jobs; regenerate as runs finish\n")
        yaml.safe_dump(conf, f, sort_keys=False, width=140)
    print(f"auto_animation_jobs: {len(conf['animations'])} jobs -> {a.out}")
    return conf


if __name__ == "__main__":
    main()
