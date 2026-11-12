"""
One-time merge of per-year blue whale track CSVs into one movement CSV.

    python -m naturenet.environment.merge_whale_years \
        --inputs /data/nlahaye/NatureNet/Blue_Whale_In/Bmu_2015CA.augmented.land_jittered.csv \
                 /data/nlahaye/NatureNet/Blue_Whale_In/Bmu_2017CA.augmented.land_jittered.csv \
        --out /data/nlahaye/NatureNet/Blue_Whale_In/Bmu_2015_2017CA.augmented.land_jittered.csv

The same uid in two years is not assumed to be the same animal, so every uid gets its deployment year as a
prefix: "825" -> "2015_825", "825__aug3" -> "2015_825__aug3" (the __aug<N> suffix stays last, so
base_uid / fold grouping keep each animal's augmented copies together). Each input file is one year (the
year of its fixes; pass --years to set it). Adds:
  orig_uid  - uid as in the input file
  year      - deployment year (the uid prefix)
Writes <out>.summary.json (rows, animals per year, ids repeated across years with their metadata, and the
sex slices for the correlation config, from gen_sex).
"""
import argparse
import json
import os

import pandas as pd


def _num(u):
    """Sort key: year, then numeric id."""
    y, _, rest = str(u).partition("_")
    try:
        return (y, 0, int(rest))
    except ValueError:
        return (y, 1, rest)


def _year_of(d, time_col, path):
    ys = pd.to_datetime(d[time_col], utc=True, format="mixed").dt.year
    vc = ys.value_counts()
    if len(vc) > 1:
        print(f"WARNING: {path} spans years {sorted(vc.index)}; prefixing with the most common, {vc.index[0]}")
    return int(vc.index[0])


def merge(inputs, out, years=None, uid_col="uid", time_col="timestamp", sex_col="gen_sex"):
    dfs = [pd.read_csv(p, dtype={uid_col: str}, low_memory=False) for p in inputs]
    cols = list(dfs[0].columns)
    for p, d in zip(inputs, dfs):
        if list(d.columns) != cols:
            raise ValueError(f"{p}: columns differ from {inputs[0]}")
    years = [int(y) for y in years] if years else [_year_of(d, time_col, p) for p, d in zip(inputs, dfs)]
    if len(set(years)) != len(years):
        raise ValueError(f"inputs must be one year each, got {years}")
    for d, y in zip(dfs, years):
        d[uid_col] = d[uid_col].astype(str).str.strip()
        d["orig_uid"] = d[uid_col]
        d["year"] = y
        d[uid_col] = str(y) + "_" + d[uid_col]
    df = pd.concat(dfs, ignore_index=True)
    df["_t"] = pd.to_datetime(df[time_col], utc=True, format="mixed")
    df = df.sort_values([uid_col, "_t"], kind="stable").reset_index(drop=True)
    t = df.pop("_t")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)

    base = df[uid_col].str.replace(r"__aug\d+$", "", regex=True)
    orig = df[df["is_augmented"].astype(int) == 0] if "is_augmented" in df else df
    obase = base[orig.index]
    meta_cols = [c for c in (sex_col, "gen_haplo", "ptt", "tag_id", "animal_name") if c in df]
    meta = orig.assign(b=obase).groupby("b")[meta_cols].agg(lambda s: "/".join(sorted({str(v) for v in s})))
    animals = sorted(base.unique(), key=_num)
    per_year = {str(y): [a for a in animals if a.startswith(f"{y}_")] for y in years}
    ob = pd.Series([a.split("_", 1)[1] for a in animals], index=animals)
    repeated = {o: {a: meta.loc[a].to_dict() for a in g.index if a in meta.index}
                for o, g in ob.groupby(ob) if len(g) > 1}
    sex = {a: str(meta.loc[a, sex_col]).lower() if a in meta.index and sex_col in meta else "unknown" for a in animals}
    slices = {"full": animals,
              "males": [a for a in animals if sex[a] == "male"],
              "females": [a for a in animals if sex[a] == "female"]}
    slices["unknown"] = [a for a in animals if sex[a] not in ("male", "female")]
    summary = {
        "inputs": list(inputs), "years": years, "out": out, "rows": int(len(df)),
        "rows_per_input": [int(len(d)) for d in dfs],
        "duplicate_uid_time_rows": int(df.duplicated([uid_col, time_col]).sum()),
        "duplicate_uid_time_rows_per_input": [int(d.duplicated([uid_col, time_col]).sum()) for d in dfs],
        "time_range": [str(t.min()), str(t.max())],
        "time_range_per_year": {str(y): [str(t[df["year"] == y].min()), str(t[df["year"] == y].max())] for y in years},
        "n_animals": len(animals), "animals_per_year": per_year,
        "ids_in_several_years": repeated,
        "sex_slices": slices,
    }
    with open(os.path.splitext(out)[0] + ".summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    return df, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=None, help="uid prefix per input (default: year of its fixes)")
    a = ap.parse_args()
    _, s = merge(a.inputs, a.out, a.years)
    print(f"{s['rows']} rows ({' + '.join(map(str, s['rows_per_input']))}), {s['n_animals']} animals, "
          f"{s['time_range'][0][:10]} .. {s['time_range'][1][:10]}")
    for y, us in s["animals_per_year"].items():
        r = s["time_range_per_year"][y]
        print(f"  {y}: {len(us)} animals  {r[0][:10]} .. {r[1][:10]}")
    if s["ids_in_several_years"]:
        print(f"  ids in several years (kept as separate animals): {sorted(s['ids_in_several_years'], key=lambda u: int(u) if u.isdigit() else u)}")
    if s["duplicate_uid_time_rows"]:
        print(f"note: {s['duplicate_uid_time_rows']} rows repeat a (uid, timestamp); per input "
              f"{s['duplicate_uid_time_rows_per_input']} (already in the inputs, kept as is)")
    print("sex slices:", {k: len(v) for k, v in s["sex_slices"].items()})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
