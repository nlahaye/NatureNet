"""
Test 6 summary - is the social coupling gain explained by shared environment?

Reads the multi-agent CV tables of one run (run_hiavi_multiagent / run_scales_multiagent):
  ma_cv.csv              observed rows (null_id < 0) + time-shift null replicates
  ma_cv_env_matched.csv  observed rows + environment-matched null replicates (multiagent.null_kind: env_matched)
and reports, per K and kernel variant, the observed pooled held-out gain against both nulls:

  p_time_shift   P(null >= observed), calendars shifted (no concurrency)
  p_env_matched  P(null >= observed), partner steps replaced by non-concurrent steps in the same environment state
  env_share      (env-matched null mean - time-shift null mean) / (observed - time-shift null mean): the share of the
                 coupling gain reproduced by environment co-occupancy alone (0 = none, 1 = all)
  env_matched_dropped  mean share of partner kernel entries with no same-state, non-concurrent candidate (dropped)

plus the per-environment-state gain of the observed rows (d_by_state; steps with partner context), optionally
joined to the state profiles (Spearman of the per-state gain with each environmental variable).

YAML (one entry per multi-agent run):
  out_dir: /data/nlahaye/NatureNet/Blue_Whale_v2/drivers/coupling_env/
  runs:
    - {label: whales_ma_kin, yaml: config/v2/swirl/whales_multiagent_kin.yaml,
       projection: config/v2/swirl/whales_reward_env_projection_hiql.yaml}
Outputs: coupling_nulls.csv, coupling_by_state.csv, coupling_state_drivers.csv

    python -m naturenet.drivers.coupling_env -y config/v2/drivers/generated/whales/specs/coupling_env.yaml
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from naturenet.drivers.common import abs_cfg, atomic_csv, read_yaml, run_conf


def _null_dist(df, vname):
    nu = df[(df["variant"] == vname) & (df["null_id"] >= 0)]
    return nu.groupby("null_id")["d_total"].sum().to_numpy() if len(nu) else np.zeros(0)


def null_table(label, ts, em):
    rows = []
    base = ts if ts is not None else em
    for K in sorted(base["K"].unique()):
        b = base[base["K"] == K]
        for vname in sorted(b["variant"].unique()):
            ob = b[(b["variant"] == vname) & (b["null_id"] < 0)]
            obs = float(ob["d_total"].sum())
            r = dict(run=label, K=int(K), variant=vname, observed=obs, n_steps=int(ob["n_test_steps"].sum()),
                     observed_per_step=obs / max(int(ob["n_test_steps"].sum()), 1))
            for name, df in (("time_shift", ts), ("env_matched", em)):
                if df is None:
                    continue
                nd = _null_dist(df[df["K"] == K], vname)
                if len(nd):
                    r.update({f"{name}_mean": float(nd.mean()), f"{name}_sd": float(nd.std()),
                              f"p_{name}": float((1 + (nd >= obs).sum()) / (1 + len(nd))), f"n_{name}": int(len(nd))})
            if em is not None and "env_null_dropped" in em.columns:      # kernel entries without a matched step
                dz = em[(em["K"] == K) & (em["variant"] == vname) & (em["null_id"] >= 0)]["env_null_dropped"]
                r["env_matched_dropped"] = float(dz.mean()) if len(dz) else np.nan
            if "time_shift_mean" in r and "env_matched_mean" in r:
                den = obs - r["time_shift_mean"]
                r["env_share"] = float((r["env_matched_mean"] - r["time_shift_mean"]) / den) if den else np.nan
            rows.append(r)
    return rows


def by_state(label, df):
    rows = []
    if df is None or "d_by_state" not in df:
        return rows
    ob = df[df["null_id"] < 0]
    for (K, vname), g in ob.groupby(["K", "variant"]):
        acc = {}
        for js in g["d_by_state"].dropna():
            for s, (tot, n) in json.loads(js).items():
                a = acc.setdefault(int(s), [0.0, 0])
                a[0] += tot
                a[1] += n
        for s, (tot, n) in sorted(acc.items()):
            rows.append(dict(run=label, K=int(K), variant=vname, state=s, d_total=tot, n_steps=n,
                             d_per_step=tot / max(n, 1)))
    return rows


def state_drivers(label, bs, p):
    if not len(bs) or not p.get("projection"):
        return []
    from naturenet.drivers.driver_consistency import profiles_for
    conf = run_conf(p["yaml"])
    P, names = profiles_for(p, conf)
    rows = []
    for (K, vname), g in bs.groupby(["K", "variant"]):
        g = g[g["state"] < P.shape[0]]
        x = P[g["state"].to_numpy()]
        for j, n in enumerate(names):
            ok = np.isfinite(x[:, j])
            if ok.sum() >= 5:
                rho = spearmanr(x[ok, j], g["d_per_step"].to_numpy()[ok]).correlation
                rows.append(dict(run=label, K=int(K), variant=vname, variable=n, spearman=float(rho),
                                 n_states=int(ok.sum())))
    return rows


def run(spec):
    from naturenet.models.irl.run_hiavi_multiagent import ma_conf
    out = spec["out_dir"]
    nulls, states, drivers = [], [], []
    for p in spec["runs"]:
        conf = run_conf(p["yaml"])
        d = os.path.join(conf["out_dir"], p.get("ma_subdir") or ma_conf(conf)["ma_subdir"])
        rd = lambda f: pd.read_csv(os.path.join(d, f)) if os.path.exists(os.path.join(d, f)) else None  # noqa: E731
        ts, em = rd("ma_cv.csv"), rd("ma_cv_env_matched.csv")
        if ts is None and em is None:
            print(f"coupling_env: no ma_cv*.csv in {d}")
            continue
        nulls += null_table(p["label"], ts, em)
        bs = pd.DataFrame(by_state(p["label"], em if em is not None and "d_by_state" in em else ts))
        states.append(bs)
        try:
            drivers += state_drivers(p["label"], bs, p)
        except Exception as ex:
            print(f"coupling_env: state drivers skipped for {p['label']} ({ex})")
    atomic_csv(pd.DataFrame(nulls), os.path.join(out, "coupling_nulls.csv"))
    if states:
        atomic_csv(pd.concat(states, ignore_index=True), os.path.join(out, "coupling_by_state.csv"))
    atomic_csv(pd.DataFrame(drivers), os.path.join(out, "coupling_state_drivers.csv"))
    print(pd.DataFrame(nulls).to_string(index=False) if nulls else "coupling_env: nothing yet")
    return pd.DataFrame(nulls)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    a = ap.parse_args(argv)
    run(read_yaml(abs_cfg(a.yaml)))


if __name__ == "__main__":
    main()
