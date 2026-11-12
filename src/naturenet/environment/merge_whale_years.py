"""
One-time merge of per-year blue whale track CSVs into one movement CSV.

    python -m naturenet.environment.merge_whale_years \
        --inputs /data/nlahaye/NatureNet/Blue_Whale_In/Bmu_2015CA.augmented.land_jittered.csv \
                 /data/nlahaye/NatureNet/Blue_Whale_In/Bmu_2017CA.augmented.land_jittered.csv \
        --out /data/nlahaye/NatureNet/Blue_Whale_In/Bmu_2015_2017CA.augmented.land_jittered.csv

uids are kept as they are: a uid seen in both years is the same animal, so its 2015 and 2017 fixes form
one track (split into separate segments at the 2016 gap by max_gap_days) and stay in one CV fold.
Adds:
  year             - calendar year of the fix
  gen_sex_animal   - one sex per uid: the known value if the years agree or only one year is known,
  gen_haplo_animal   "Unknown" / "Missing" if the years give different known values (listed in the summary)
Writes <out>.summary.json (rows, animals, years per uid, conflicts, sex slices for the correlation config).
"""
import argparse
import json
import os

import pandas as pd

UNKNOWN = {"", "unknown", "missing", "nan", "none", "na"}


def _known(v):
    return str(v).strip().lower() not in UNKNOWN


def _harmonize(df, col, fill):
    """Per-uid value from the original (non-augmented) rows; returns (mapping, conflicts)."""
    orig = df[df["is_augmented"].astype(int) == 0] if "is_augmented" in df else df
    out, conflicts = {}, {}
    for uid, g in orig.groupby("uid"):
        vals = sorted({str(v).strip() for v in g[col] if _known(v)})
        if len(vals) == 1:
            out[uid] = vals[0]
        else:
            out[uid] = fill
            if len(vals) > 1:
                conflicts[uid] = {str(y): sorted({str(v) for v in gy[col]}) for y, gy in g.groupby("year")}
    return out, conflicts


def merge(inputs, out, uid_col="uid", time_col="timestamp"):
    dfs = []
    for p in inputs:
        d = pd.read_csv(p, dtype={uid_col: str}, low_memory=False)
        dfs.append(d)
    cols = list(dfs[0].columns)
    for p, d in zip(inputs, dfs):
        if list(d.columns) != cols:
            raise ValueError(f"{p}: columns differ from {inputs[0]}")
    df = pd.concat(dfs, ignore_index=True)
    df[uid_col] = df[uid_col].astype(str).str.strip()
    t = pd.to_datetime(df[time_col], utc=True, format="mixed")
    df["year"] = t.dt.year
    df["_t"] = t
    df = df.sort_values([uid_col, "_t"], kind="stable").drop(columns="_t").reset_index(drop=True)
    t = pd.to_datetime(df[time_col], utc=True, format="mixed")
    dup = df.duplicated([uid_col, time_col]).sum()

    base = df[uid_col].str.replace(r"__aug\d+$", "", regex=True)
    tmp = df.assign(uid=base)
    sex, sex_c = _harmonize(tmp, "gen_sex", "Unknown")
    hap, hap_c = _harmonize(tmp, "gen_haplo", "Missing")
    df["gen_sex_animal"] = base.map(sex).fillna("Unknown")
    df["gen_haplo_animal"] = base.map(hap).fillna("Missing")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)

    years = {u: sorted(int(y) for y in g.unique()) for u, g in df.assign(b=base).groupby("b")["year"]}
    per_year = {str(y): sorted({b for b, ys in years.items() if y in ys}, key=_num)
                for y in sorted(df["year"].unique())}
    slices = {"full": sorted(years, key=_num)}
    for key, lab in (("males", "male"), ("females", "female")):
        slices[key] = [u for u in slices["full"] if sex.get(u, "").lower() == lab]
    slices["unknown"] = [u for u in slices["full"] if sex.get(u, "Unknown").lower() not in ("male", "female")]
    summary = {
        "inputs": list(inputs), "out": out, "rows": int(len(df)), "rows_per_input": [int(len(d)) for d in dfs],
        "duplicate_uid_time_rows": int(dup),
        "duplicate_uid_time_rows_per_input": [int(d.duplicated([uid_col, time_col]).sum()) for d in dfs],
        "time_range": [str(t.min()), str(t.max())],
        "time_range_per_year": {str(y): [str(t[df["year"] == y].min()), str(t[df["year"] == y].max())]
                                for y in sorted(df["year"].unique())},
        "n_animals": len(years), "animals_per_year": per_year,
        "multi_year_animals": {u: ys for u, ys in years.items() if len(ys) > 1},
        "sex_conflicts": sex_c, "haplo_conflicts": hap_c,
        "sex_slices": slices,
    }
    with open(os.path.splitext(out)[0] + ".summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    return df, summary


def _num(u):
    try:
        return (0, int(u))
    except ValueError:
        return (1, u)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, s = merge(a.inputs, a.out)
    print(f"{s['rows']} rows ({' + '.join(map(str, s['rows_per_input']))}), {s['n_animals']} animals, "
          f"{s['time_range'][0]} .. {s['time_range'][1]}")
    for y, us in s["animals_per_year"].items():
        print(f"  {y}: {len(us)} animals  {s['time_range_per_year'][y][0][:10]} .. {s['time_range_per_year'][y][1][:10]}")
    print(f"  in both years: {sorted(s['multi_year_animals'], key=_num)}")
    if s["duplicate_uid_time_rows"]:
        print(f"note: {s['duplicate_uid_time_rows']} rows repeat a (uid, timestamp); per input "
              f"{s['duplicate_uid_time_rows_per_input']} (already in the inputs, kept as is)")
    for name, c in (("sex", s["sex_conflicts"]), ("haplotype", s["haplo_conflicts"])):
        for u, v in c.items():
            print(f"WARNING: {name} differs between years for {u}: {v} -> set to unknown in *_animal columns")
    print("sex slices:", {k: len(v) for k, v in s["sex_slices"].items()})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
