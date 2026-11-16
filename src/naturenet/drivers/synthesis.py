"""
Cross-perspective synthesis of the environmental-driver tests (Rep_Taxonomy synthesis protocol).

The tools are complementary views, so the synthesis reports rank AGREEMENT across perspectives, not a winner:

  metrics_long.csv        every scalar result of the driver tests in one long table
                          (species, test, item, perspective, block, value, higher_better, source)
  rank_agreement.csv      per rank set: Kendall W across perspectives, n items / perspectives
  rank_tau_<set>.csv      pairwise Kendall tau between perspectives;  ranks_<set>.csv  item ranks per perspective
  friedman.csv            Friedman chi2 / p and Nemenyi critical difference of the level x decision grid
                          (blocks = folds, and animals when available);  mean_ranks.csv, nemenyi_p_<unit>.csv
  wilcoxon_vs_ref.csv     paired Wilcoxon of every grid cell vs the reference cell, BH q
  variance_decomposition.csv   SS shares of level / decision / fold (and interactions) of the held-out LL
  ll_ladder.csv           additive held-out LL ladder: no-environment null -> environment (reference level, N1)
                          -> best decision scale; social coupling (multi-agent LOAO) appended and flagged
                          protocol = "loao" (a different hold-out, not additive with the K-fold rungs)

YAML (written by plan to config/v2/drivers/generated/<sp>/specs/synthesis.yaml):
  out_dir: .../drivers/synthesis/
  species: whales
  inputs:
    grid: .../drivers/env_scale_grid/            # grid_folds.csv, grid_animals.csv, grid_gains.csv
    grid_year: .../drivers/env_scale_grid_year/  # same, year-blocked folds (test 3)
    ppc: .../drivers/ppc/                        # ppc_summary.csv
    consistency: .../drivers/driver_consistency/ # fold_stability.csv, cross_agreement.csv
    coupling: .../drivers/coupling_env/          # coupling_nulls.csv
    switch: .../drivers/switch_env/              # switch_env_<label>.csv
    geometry: .../drivers/geometry/              # level_purity.csv, projection_summary.csv
  reference: {level: fine, decision: N1}
  no_env_level: null_single
  rank_sets:
    levels: {item: level, perspectives: [grid:test_ll@N1, grid_year:test_ll@N1, geometry:eta2_mean,
                                         geometry:silhouette, projection:native, projection:pca]}
    models: {item: model, perspectives: [ppc:log_score, ppc:-ece, ppc:-occupancy_js, ppc:-pit_ks]}

    python -m naturenet.drivers.synthesis -y config/v2/drivers/generated/whales/specs/synthesis.yaml
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from naturenet.drivers import reptax_bridge as rb
from naturenet.drivers.common import abs_cfg, atomic_csv, read_yaml

LOWER_BETTER = {"ece", "brier", "occupancy_js", "action_js", "step_ks", "turn_ks", "pit_ks", "p_time_shift",
                "p_env_matched"}


def _read(path):
    return pd.read_csv(path) if path and os.path.exists(path) else pd.DataFrame()


def collect(spec):
    sp, inp = spec.get("species", ""), spec.get("inputs") or {}
    rows = []

    def add(test, item, persp, block, value, src, hb=True):
        if value is None or not np.isfinite(value):
            return
        rows.append(dict(species=sp, test=test, item=str(item), perspective=persp, block=str(block),
                         value=float(value), higher_better=bool(hb), source=src))

    for key in ("grid", "grid_year"):
        p = os.path.join(inp.get(key) or "", "grid_folds.csv")
        d = _read(p if inp.get(key) else None)
        for r in d.itertuples():
            add(key, r.level, f"test_ll@{r.decision}", f"fold{r.fold}", r.test_ll, p)
        pa = os.path.join(inp.get(key) or "", "grid_animals.csv")
        a = _read(pa if inp.get(key) else None)
        for r in a.itertuples():
            add(key + "_animal", r.level, f"ll_per_step@{r.decision}", f"{r.animal}|f{r.fold}", r.ll_per_step, pa)
    p = os.path.join(inp.get("ppc") or "", "ppc_summary.csv")
    d = _read(p if inp.get("ppc") else None)
    for r in d.to_dict("records"):
        for m in ("log_score", "brier", "ece", "acc", "pit_ks", "occupancy_js", "action_js", "step_ks", "turn_ks",
                  "selection_rho", "ppp_dwell_mean", "ppp_state_run_mean"):
            if m in r:
                add("ppc", r["model"], m, f"fold{r.get('fold', 0)}", r[m], p, m not in LOWER_BETTER)
    if inp.get("geometry"):
        p = os.path.join(inp["geometry"], "level_purity.csv")
        for r in _read(p).to_dict("records"):
            for m in ("eta2_mean", "silhouette"):
                add("geometry", r["level"], m, "all", r.get(m, np.nan), p)
        for p in glob.glob(os.path.join(inp["geometry"], "projection_summary*.csv")):
            for r in _read(p).itertuples():
                add("projection", r.level, r.projection, "all", r.mean, p)
    if inp.get("coupling"):
        p = os.path.join(inp["coupling"], "coupling_nulls.csv")
        for r in _read(p).to_dict("records"):
            item = f"{r.get('run')}|K{r.get('K')}|{r.get('variant')}"
            for m in ("observed", "p_time_shift", "p_env_matched", "env_share"):
                if m in r:
                    add("coupling", item, m, "loao", r[m], p, m not in LOWER_BETTER)
    if inp.get("consistency"):
        p = os.path.join(inp["consistency"], "fold_stability.csv")
        d = _read(p)
        num = [c for c in d.columns if d[c].dtype.kind in "fi" and c not in ("mode",)]
        if len(d) and "perspective" in d:
            for pl, g in d.groupby("perspective"):
                for c in num:
                    add("consistency", pl, c, "all", float(g[c].mean()), p)
    if inp.get("switch"):
        for p in glob.glob(os.path.join(inp["switch"], "switch_env_*.csv")):
            lab = os.path.basename(p)[len("switch_env_"):-4]
            d = _read(p)
            if len(d) and "q_bh" in d:
                add("switch", lab, "n_sig_q05", "all", float((d["q_bh"] < 0.05).sum()), p)
                add("switch", lab, "share_sig_q05", "all", float((d["q_bh"] < 0.05).mean()), p)
    return pd.DataFrame(rows)


def _persp_values(L, ps):
    """'test:metric[@decision]' (prefix '-' or a known lower-is-better metric is negated) -> Series item -> mean value (higher=better)."""
    test, metric = ps.split(":", 1)
    minus = metric.startswith("-")
    metric = metric.lstrip("-")
    if test in ("grid", "grid_year") and "@" not in metric:
        metric = metric + "@N1"
    g = L[(L["test"] == test) & (L["perspective"] == metric)]
    if not len(g):
        return pd.Series(dtype=float)
    s = g.groupby("item")["value"].mean()
    return -s if (minus or not bool(g["higher_better"].iloc[0])) else s   # oriented so that higher = better


def rank_agreement(L, rank_sets, out):
    rows = []
    for name, rs in (rank_sets or {}).items():
        cols = {ps: _persp_values(L, ps) for ps in rs["perspectives"]}
        M = pd.DataFrame({k: v for k, v in cols.items() if len(v)})
        if M.shape[1] < 2:
            rows.append(dict(rank_set=name, n_items=int(len(M)), n_perspectives=int(M.shape[1]), kendall_w=np.nan,
                             note="fewer than 2 perspectives with results"))
            continue
        excl = list(rs.get("exclude_items") or [])
        M = M.drop(index=[i for i in excl if i in M.index]).dropna()
        R = M.rank(ascending=False)
        atomic_csv(R.reset_index().rename(columns={"index": "item"}), os.path.join(out, f"ranks_{name}.csv"))
        tau = rb.kendall_tau_matrix(R, list(R.columns))
        atomic_csv(tau.reset_index().rename(columns={"index": "perspective"}), os.path.join(out, f"rank_tau_{name}.csv"))
        rows.append(dict(rank_set=name, n_items=int(len(R)), n_perspectives=int(R.shape[1]),
                         kendall_w=rb.kendall_w(R.to_numpy().T), note=""))
    return pd.DataFrame(rows)


def grid_stats(spec, out):
    inp = spec.get("inputs") or {}
    ref = spec.get("reference") or {"level": "parent", "decision": "N1"}
    fr, wx, vd = [], [], []
    for key in ("grid", "grid_year"):
        if not inp.get(key):
            continue
        for unit, fname, val, blk in (("fold", "grid_folds.csv", "test_ll", ["fold"]),
                                      ("animal", "grid_animals.csv", "ll_per_step", ["fold", "animal"])):
            d = _read(os.path.join(inp[key], fname))
            if not len(d):
                continue
            d["cell"] = d["level"].astype(str) + "|" + d["decision"].astype(str)
            B = d.pivot_table(index=blk, columns="cell", values=val)
            summ, mr, pw = rb.friedman_nemenyi(B)
            fr.append(dict(grid=key, unit=unit, **summ))
            if len(mr):
                atomic_csv(mr.rename("mean_rank").reset_index().rename(columns={"index": "cell"}),
                           os.path.join(out, f"mean_ranks_{key}_{unit}.csv"))
                atomic_csv(pw.reset_index().rename(columns={"index": "cell"}),
                           os.path.join(out, f"nemenyi_p_{key}_{unit}.csv"))
            rc = f"{ref['level']}|{ref['decision']}"
            if rc in B:
                wx.append(rb.wilcoxon_bh(B, rc).assign(grid=key, unit=unit))
            if unit == "fold":
                vd.append(rb.variance_decomposition(d, ["level", "decision", "fold"], val).assign(grid=key))
    atomic_csv(pd.DataFrame(fr), os.path.join(out, "friedman.csv"))
    if wx:
        atomic_csv(pd.concat(wx, ignore_index=True), os.path.join(out, "wilcoxon_vs_ref.csv"))
    if vd:
        atomic_csv(pd.concat(vd, ignore_index=True), os.path.join(out, "variance_decomposition.csv"))


def ll_ladder(spec, L):
    ref = spec.get("reference") or {"level": "parent", "decision": "N1"}
    g = L[L["test"] == "grid"]
    rows = []
    if len(g):
        cell = g.groupby(["item", "perspective"])["value"].mean()

        def get(level, dec):
            return cell.get((level, f"test_ll@{dec}"), np.nan)

        no_env = get(spec.get("no_env_level", "null_single"), ref["decision"])
        env = get(ref["level"], ref["decision"])
        dec = g[g["item"] == ref["level"]].groupby("perspective")["value"].mean()
        best_dec = dec.idxmax() if len(dec) else None
        rungs = [("no environment (" + spec.get("no_env_level", "null_single") + ")", no_env),
                 (f"+ environment ({ref['level']}, {ref['decision']})", env)]
        if best_dec is not None:
            rungs.append((f"+ decision scale ({best_dec.split('@')[-1]})", float(dec.max())))
        prev = np.nan
        for name, v in rungs:
            rows.append(dict(rung=name, test_ll=v, increment=v - prev if np.isfinite(prev) else np.nan,
                             protocol="kfold"))
            prev = v if np.isfinite(v) else prev
    c = L[(L["test"] == "coupling") & (L["perspective"] == "observed")]
    for r in c.itertuples():
        rows.append(dict(rung=f"+ social coupling ({r.item})", test_ll=np.nan, increment=r.value, protocol="loao"))
    return pd.DataFrame(rows)


def run(spec):
    out = spec["out_dir"]
    os.makedirs(out, exist_ok=True)
    L = collect(spec)
    atomic_csv(L, os.path.join(out, "metrics_long.csv"))
    if not len(L):
        print("synthesis: no driver results found yet")
        return L
    atomic_csv(rank_agreement(L, spec.get("rank_sets"), out), os.path.join(out, "rank_agreement.csv"))
    grid_stats(spec, out)
    atomic_csv(ll_ladder(spec, L), os.path.join(out, "ll_ladder.csv"))
    print(f"synthesis: {len(L)} metrics -> {out}")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    run(read_yaml(abs_cfg(ap.parse_args(argv).yaml)))


if __name__ == "__main__":
    main()
