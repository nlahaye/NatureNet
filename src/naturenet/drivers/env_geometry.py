"""
Environment geometry along the tracks (Rep_Taxonomy geo-intrinsic-dimension tools applied to the movement data).

Points are the original fixes (no __aug copies) with their raw environment (sample_env_inputs states_input files,
z-scored; rows with a missing selected channel dropped). Analyses:

  id            global intrinsic dimension (TwoNN, MLE; skdim estimators if installed) of the used environment,
                of the available environment (availability.py) and of their union
  lid           pointwise MLE local ID of the used environment; Moran's I on the geographic kNN graph, variogram,
                density control (LID vs sampling density), LID by animal / year / level state (Kruskal-Wallis)
  geo_env       geographic-vs-environment kNN Jaccard and Mantel r (how much environment space mirrors geography)
  selection     used vs available: per-channel standardised difference and KS, and a grouped-CV logistic
                classifier AUC (animals held out) = multivariate selection strength
  levels        purity of each environment level in the raw environment: per-channel eta^2 (between-state share of
                variance), silhouette on a subsample, n states
YAML:
  out_dir: /data/nlahaye/NatureNet/Blue_Whale_v2/drivers/geometry/
  sample_inputs_yaml: config/v2/environment/whales_sample_inputs_v2.yaml
  channels: [sst, grad_mag_sst, log10_CHL, adt, eke, depth_m]   # null = all channels of the run
  env_transforms: {}
  availability: {avl_dir: ..., n_available: 10}
  levels: {parent: /data/nlahaye/NatureNet/Blue_Whale_v2/, k30: /data/nlahaye/NatureNet/Blue_Whale_v2/levels/k30/, ...}
  group_regex: {year: "(?:^|_)((?:19|20)\\d\\d)(?:_|$)"}
  k: 20
  max_points: 20000
  n_perm: 199
    python -m naturenet.drivers.env_geometry -y config/v2/drivers/generated/whales/specs/geometry.yaml [--part id|lid|geo_env|selection|levels]
Outputs: global_id.csv, lid_*.csv, selection_channels.csv, level_purity.csv, geometry_summary_<parts>.json
"""
import argparse
import json
import os
import pickle
import re

import numpy as np
import pandas as pd

from naturenet.drivers import reptax_bridge as rb
from naturenet.drivers.common import abs_cfg, atomic_csv, atomic_json, base_id, is_original, read_yaml
from naturenet.drivers.env_covariates import TRANSFORMS

PARTS = ("id", "lid", "geo_env", "selection", "levels")


def _channels(df_dir, uid):
    with open(os.path.join(df_dir, uid + "_states_input_channels.json")) as f:
        return list(json.load(f))


def load_points(df_dir, uid, channels=None, keys=None, transforms=None):
    """DataFrame: key, seg, pos, lon, lat, <channels> for every fix of the selected keys (raw, transformed)."""
    names = _channels(df_dir, uid)
    sel = list(channels) if channels else names
    cols = [names.index(c) for c in sel]
    with open(os.path.join(df_dir, uid + "_dfs.pkl"), "rb") as f:
        dfs = pd.read_pickle(f)
    frames = []
    for k, segs in dfs.items():
        if keys is not None and not keys(k):
            continue
        p = os.path.join(df_dir, f"{uid}_{k}_states_input.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            inp = pickle.load(f)
        for si, (seg, x) in enumerate(zip(segs, inp)):
            if not len(seg):
                continue
            d = pd.DataFrame(np.asarray(x)[:, cols], columns=sel)
            d.insert(0, "lat", seg["latitude"].to_numpy(float))
            d.insert(0, "lon", seg["longitude"].to_numpy(float))
            d.insert(0, "pos", np.arange(len(seg)))
            d.insert(0, "seg", si)
            d.insert(0, "key", k)
            frames.append(d)
    P = pd.concat(frames, ignore_index=True)
    for c, t in (transforms or {}).items():
        if c in P:
            P[c] = TRANSFORMS[t](P[c].to_numpy(float))
    return P, sel


def zscore(P, sel, ref=None):
    ref = P if ref is None else ref
    mu, sd = ref[sel].mean(), ref[sel].std().replace(0, 1.0)
    return ((P[sel] - mu) / sd).to_numpy(float)


def _sub(n, m, seed):
    return np.sort(np.random.default_rng(seed).choice(n, m, replace=False)) if n > m else np.arange(n)


def level_labels(level_dir, uid, P):
    """Per-fix state of a level (states_abstract_env pkl, list of per-segment arrays), aligned with P."""
    lab = np.full(len(P), -1, np.int64)
    for k, idx in P.groupby("key").groups.items():
        p = os.path.join(level_dir, f"{uid}_{k}_states_abstract_env.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            segs = pickle.load(f)
        sub = P.loc[idx]
        for i, s, q in zip(idx, sub["seg"].to_numpy(), sub["pos"].to_numpy()):
            arr = np.asarray(segs[s]) if s < len(segs) else ()
            lab[i] = int(arr[q]) if q < len(arr) else -1
    return lab


def run(spec, parts=PARTS):
    sconf = read_yaml(abs_cfg(spec["sample_inputs_yaml"]))
    df_dir, uid = sconf["df_dir"], sconf["df_run_uid"]
    out = spec["out_dir"]
    os.makedirs(out, exist_ok=True)
    k, M, nperm, seed = int(spec.get("k", 20)), int(spec.get("max_points", 20000)), int(spec.get("n_perm", 199)), \
        int(spec.get("seed", 0))
    P, sel = load_points(df_dir, uid, spec.get("channels"), is_original, spec.get("env_transforms"))
    P = P.dropna(subset=sel + ["lon", "lat"]).reset_index(drop=True)
    P["animal"] = [base_id(x) for x in P["key"]]
    for g, rx in (spec.get("group_regex") or {}).items():
        P[g] = [(re.search(rx, a).group(1) if re.search(rx, a) else "na") for a in P["animal"]]
    Pi = P.iloc[_sub(len(P), M, seed)].reset_index(drop=True)
    X = zscore(Pi, sel, P)
    ll = Pi[["lon", "lat"]].to_numpy(float)
    summary = dict(n_fixes=int(len(P)), n_used=int(len(Pi)), channels=sel, backend=rb.backend())

    A = None
    avl = (spec.get("availability") or {}).get("avl_dir")
    if avl and os.path.exists(os.path.join(avl, uid + "_dfs.pkl")) and ({"id", "selection"} & set(parts)):
        A, _ = load_points(avl, uid, sel, None, spec.get("env_transforms"))
        A = A[A["pos"] > 0].dropna(subset=sel).reset_index(drop=True)
        A["animal"] = [base_id(x.split("__avl")[0]) for x in A["key"]]
        summary["n_available"] = int(len(A))

    if "id" in parts:
        rows = [rb.global_id(X, spec.get("id_methods", ("TwoNN", "MLE")), k, seed=seed).assign(set="used")]
        if A is not None:
            Ai = A.iloc[_sub(len(A), M, seed)]
            XA = zscore(Ai, sel, P)
            rows.append(rb.global_id(XA, spec.get("id_methods", ("TwoNN", "MLE")), k, seed=seed).assign(set="available"))
            rows.append(rb.global_id(np.vstack([X, XA]), spec.get("id_methods", ("TwoNN", "MLE")), k,
                                     seed=seed).assign(set="union"))
        g = pd.concat(rows, ignore_index=True)
        atomic_csv(g, os.path.join(out, "global_id.csv"))
        summary["global_id"] = g.groupby(["set", "method"])["global_id"].mean().reset_index().to_dict("records")

    if "lid" in parts:
        lid = rb.mle_local_id(X, k)
        nbr, _ = rb.haversine_knn(ll, k)
        mi = rb.morans_i(lid, nbr, nperm, seed)
        dc = rb.density_control(lid, ll, k)
        vg = rb.variogram(lid, ll, spec.get("variogram_bins_km", [0, 10, 25, 50, 100, 200, 400, 800, 1600]),
                          seed=seed)
        atomic_csv(vg, os.path.join(out, "lid_variogram.csv"))
        Pi["lid"] = lid
        rows = []
        groups = ["animal"] + list((spec.get("group_regex") or {}).keys())
        for lv, d in (spec.get("levels") or {}).items():
            Pi[f"state_{lv}"] = level_labels(d, uid, Pi)
            groups.append(f"state_{lv}")
        for gname in groups:
            gs = Pi[gname] if not gname.startswith("state_") else Pi[gname].where(Pi[gname] >= 0)
            s, kw = rb.region_summary(lid, gs)
            if len(s):
                atomic_csv(s.assign(group_by=gname), os.path.join(out, f"lid_by_{gname}.csv"))
            rows.append(dict(group_by=gname, **kw))
        atomic_csv(pd.DataFrame(rows), os.path.join(out, "lid_group_tests.csv"))
        atomic_csv(Pi[["key", "seg", "pos", "lon", "lat", "lid"]], os.path.join(out, "lid_points.csv"))
        summary["lid"] = dict(median=float(np.nanmedian(lid)), **mi, **dc)

    if "geo_env" in parts:
        summary["geo_env"] = dict(knn_jaccard=rb.knn_jaccard(ll, X, k),
                                  **rb.mantel(ll, X, int(spec.get("mantel_max_n", 1500)), nperm, seed))

    if "selection" in parts and A is not None:
        from scipy.stats import ks_2samp
        rows = []
        for c in sel:
            u, a = P[c].to_numpy(float), A[c].to_numpy(float)
            rows.append(dict(channel=c, mean_used=float(u.mean()), mean_avail=float(a.mean()),
                             smd=float((u.mean() - a.mean()) / np.sqrt(0.5 * (u.var() + a.var()) + 1e-12)),
                             ks=float(ks_2samp(u, a).statistic), ks_p=float(ks_2samp(u, a).pvalue)))
        atomic_csv(pd.DataFrame(rows), os.path.join(out, "selection_channels.csv"))
        summary["selection_auc"] = selection_auc(P, A, sel, seed, M)

    if "levels" in parts and spec.get("levels"):
        rows = []
        Pl = P.iloc[_sub(len(P), M, seed + 1)].reset_index(drop=True)
        XL = zscore(Pl, sel, P)
        for lv, d in spec["levels"].items():
            lab = level_labels(d, uid, Pl)
            ok = lab >= 0
            if ok.sum() < 10 or len(np.unique(lab[ok])) < 2:
                rows.append(dict(level=lv, n_states=int(len(np.unique(lab[ok]))), n=int(ok.sum())))
                continue
            r = dict(level=lv, n_states=int(len(np.unique(lab[ok]))), n=int(ok.sum()))
            for j, c in enumerate(sel):
                x = XL[ok, j]
                gm = pd.Series(x).groupby(lab[ok]).transform("mean").to_numpy()
                r[f"eta2_{c}"] = float(((gm - x.mean()) ** 2).sum() / max(((x - x.mean()) ** 2).sum(), 1e-12))
            r["eta2_mean"] = float(np.mean([r[f"eta2_{c}"] for c in sel]))
            from sklearn.metrics import silhouette_score
            ii = _sub(int(ok.sum()), int(spec.get("silhouette_n", 5000)), seed)
            r["silhouette"] = float(silhouette_score(XL[ok][ii], lab[ok][ii])) if len(np.unique(lab[ok][ii])) > 1 \
                else np.nan
            rows.append(r)
        atomic_csv(pd.DataFrame(rows), os.path.join(out, "level_purity.csv"))
    # one summary per part set, so parallel --part jobs never write the same file
    name = "all" if set(parts) == set(PARTS) else "_".join(parts)
    atomic_json(summary, os.path.join(out, f"geometry_summary_{name}.json"))
    print(f"env_geometry: {parts} -> {out}")
    return summary


def selection_auc(P, A, sel, seed=0, max_n=20000, n_splits=5):
    """Grouped (animal) CV AUC of a logistic used-vs-available classifier; 0.5 = no multivariate selection."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    U = P.iloc[_sub(len(P), max_n, seed)]
    V = A.iloc[_sub(len(A), max_n, seed)]
    X = np.vstack([zscore(U, sel, P), zscore(V, sel, P)])
    y = np.r_[np.ones(len(U)), np.zeros(len(V))]
    g = np.r_[U["animal"].to_numpy(), V["animal"].to_numpy()]
    ns = min(n_splits, len(np.unique(g)))
    if ns < 2:
        return np.nan
    pr = np.zeros(len(y))
    for tr, te in GroupKFold(ns).split(X, y, g):
        pr[te] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, pr))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--part", action="append", choices=PARTS)
    a = ap.parse_args(argv)
    run(read_yaml(abs_cfg(a.yaml)), tuple(a.part or PARTS))


if __name__ == "__main__":
    main()
