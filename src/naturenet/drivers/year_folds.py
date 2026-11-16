"""
Test 3 - leave-year-out transfer folds.

Writes <folds_dir>/folds_<n>_seed<fold_seed>.json with one fold per year group (fold k test = the animals of year
group k), in the format get_folds reuses (it keeps a file whose animal set matches the data). Every K-fold runner
(run_hiavi_kfold, get_stats_hiavi, run_swirl_scales_kfold, run_swirl_kfold) then fits on one year and tests on the
other with no code change: point the run YAML at this folds_dir with n_folds = number of year groups.

Year of an animal: `year_regex` on the group id (whale ids are year-prefixed, e.g. 2015_xxx / Bmu_2017_...),
else the first timestamp of its track in <df_dir>/<df_run_uid>_dfs.pkl. `year_groups` can merge years
(e.g. sharks [[2020, 2021, 2022], [2023, 2024]]); years not listed are dropped with a warning (get_folds would then
rebuild random folds, so the run refuses instead).

    python -m naturenet.drivers.year_folds -y config/v2/swirl/whales_hiavi_kfold_kin.yaml \
        --folds_dir /data/nlahaye/NatureNet/Blue_Whale_v2/folds_year/ [--year_groups "2015;2017"]
"""
import argparse
import json
import os

import pandas as pd

from naturenet.drivers.common import atomic_json, base_id, read_yaml, year_of


def animal_years(conf, groups, year_regex=None):
    years, missing = {}, []
    for g in sorted(set(map(str, groups))):
        y = year_of(g, year_regex) if year_regex else year_of(g)
        if y is None:
            missing.append(g)
        else:
            years[g] = y
    if missing:
        path = os.path.join(conf["df_dir"], conf["df_run_uid"] + "_dfs.pkl")
        dfs = pd.read_pickle(path)
        pre = conf["df_run_uid"] + "_"
        for g in missing:
            k = g[len(pre):] if g.startswith(pre) else g
            segs = dfs.get(k) or dfs.get(base_id(k)) or []
            ts = [pd.to_datetime(s["timestamp"], errors="coerce", utc=True).min() for s in segs if len(s)]
            ts = [t for t in ts if pd.notna(t)]
            if not ts:
                raise ValueError(f"year_folds: no year for animal {g} (regex and dfs.pkl both failed)")
            years[g] = int(min(ts).year)
    return years


def year_fold_groups(years, year_groups=None):
    ys = sorted(set(years.values()))
    year_groups = [[y] for y in ys] if not year_groups else [[int(v) for v in grp] for grp in year_groups]
    folds = [[g for g, y in sorted(years.items()) if y in grp] for grp in year_groups]
    dropped = sorted(g for g, y in years.items() if not any(y in grp for grp in year_groups))
    if dropped:
        raise ValueError(f"year_folds: animals of unlisted years {sorted({years[g] for g in dropped})}: {dropped[:5]}")
    if any(len(f) == 0 for f in folds):
        raise ValueError(f"year_folds: empty year group in {year_groups} (years present: {ys})")
    return folds, year_groups


def run(conf, folds_dir, year_groups=None, year_regex=None, groups=None):
    if groups is None:
        from naturenet.models.irl.swirl_data import load_swirl_dataset
        groups = load_swirl_dataset(conf, allow_variable=True, verbose=False)["groups"]
    years = animal_years(conf, groups, year_regex)
    folds, yg = year_fold_groups(years, year_groups)
    os.makedirs(folds_dir, exist_ok=True)
    path = os.path.join(folds_dir, f"folds_{len(folds)}_seed{int(conf.get('fold_seed', 0))}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(folds, f, indent=1)
    os.replace(path + ".tmp", path)
    atomic_json(dict(year_groups=yg, n_animals=[len(f) for f in folds], years=years, folds_file=path, folds=folds),
                os.path.join(folds_dir, "year_folds_meta.json"))
    print(f"year folds {yg}: {[len(f) for f in folds]} animals -> {path}")
    return path


def _parse_groups(s):
    if not s:
        return None
    return [[int(v) for v in part.split(",") if v.strip()] for part in s.split(";") if part.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True, help="any run YAML of the species (actions / df_dir / df_run_uid)")
    ap.add_argument("--folds_dir", required=True)
    ap.add_argument("--year_groups", default=None, help='e.g. "2015;2017" or "2020,2021,2022;2023,2024"')
    ap.add_argument("--year_regex", default=None)
    a = ap.parse_args(argv)
    run(read_yaml(a.yaml), a.folds_dir, _parse_groups(a.year_groups), a.year_regex)


if __name__ == "__main__":
    main()
