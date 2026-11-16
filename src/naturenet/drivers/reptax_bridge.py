"""
Bridge to the Rep_Taxonomy analysis code (geo intrinsic dimension, spatial statistics, synthesis statistics).

The functions below are self-contained ports (same definitions, numpy / scipy / sklearn only), so NatureNet does
not depend on a Rep_Taxonomy install; backend() reports whether Rep_Taxonomy (REPTAXONOMY_SRC=<dir containing
reptaxonomy/>) and skdim are importable. With skdim, global_id() also accepts the skdim estimators (e.g. "ESS",
"FisherS"); without it the TwoNN and Levina-Bickel MLE estimators here are used.

Ported from reptaxonomy.robustness.geo_intrinsic_dimension (haversine_knn, great_circle_km, morans_i, variogram,
knn_jaccard, mantel) and reptaxonomy.summarize.synthesis_stats (kendall_w, kendall_tau_matrix, variance
decomposition, generalised here to any set of crossed factors).
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0088

if os.environ.get("REPTAXONOMY_SRC") and os.environ["REPTAXONOMY_SRC"] not in sys.path:
    sys.path.insert(0, os.environ["REPTAXONOMY_SRC"])
try:
    from reptaxonomy.robustness import geo_intrinsic_dimension as _gid      # noqa: F401
    HAVE_REPTAX = True
except Exception:                                                          # pragma: no cover - optional
    _gid = None
    HAVE_REPTAX = False
import importlib.util

HAVE_SKDIM = importlib.util.find_spec("skdim") is not None


def backend():
    return dict(reptaxonomy=HAVE_REPTAX, skdim=HAVE_SKDIM)


# ----------------------------------------------------------------------------- intrinsic dimension
def clean_for_id(X, seed=0):
    X = np.asarray(X, dtype=np.float64)
    sd = float(np.nanstd(X)) or 1.0
    _, inv, cnt = np.unique(np.round(X / sd, 8), axis=0, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        rng = np.random.default_rng(seed)
        dup = cnt[inv.ravel()] > 1
        X = X.copy()
        X[dup] += rng.normal(scale=1e-6 * sd, size=(int(dup.sum()), X.shape[1]))
    return X


def _knn_dist(X, k):
    from sklearn.neighbors import NearestNeighbors
    d, i = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return d[:, 1:], i[:, 1:]


def twonn_id(X, discard=0.1):
    """Facco et al. (2017) TwoNN global ID (linear fit through the origin of -log(1-F) vs log mu)."""
    d, _ = _knn_dist(clean_for_id(X), 2)
    mu = d[:, 1] / np.maximum(d[:, 0], 1e-300)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1)])
    n = len(mu)
    if n < 10:
        return np.nan
    keep = int(n * (1 - discard))
    F = np.arange(1, n + 1) / n
    x, y = np.log(mu[:keep]), -np.log(1 - F[:keep])
    return float((x * y).sum() / (x * x).sum())


def mle_local_id(X, k=20):
    """Levina-Bickel pointwise MLE ID (MacKay-Ghahramani averaging of the inverse for the global value)."""
    d, _ = _knn_dist(clean_for_id(X), k)
    d = np.maximum(d, 1e-300)
    lr = np.log(d[:, -1:] / d[:, :-1])
    inv = lr.sum(1) / (k - 1)
    return 1.0 / np.maximum(inv, 1e-12)


def global_id(X, methods=("TwoNN", "MLE"), k=20, n_runs=3, max_points=5000, seed=0):
    rows = []
    for r in range(int(n_runs)):
        rng = np.random.default_rng(seed + r)
        idx = rng.choice(len(X), min(len(X), max_points), replace=False)
        Xr = np.asarray(X)[idx]
        for m in methods:
            try:
                if HAVE_SKDIM and m not in ("TwoNN", "MLE"):
                    import skdim
                    v = float(getattr(skdim.id, m)().fit(clean_for_id(Xr, seed + r)).dimension_)
                elif m == "TwoNN":
                    v = twonn_id(Xr)
                elif m == "MLE":
                    v = float(1.0 / np.mean(1.0 / mle_local_id(Xr, k)))
                else:
                    raise ValueError(f"{m} needs skdim")
                err = ""
            except Exception as e:
                v, err = np.nan, str(e)[:200]
            rows.append(dict(method=m, k=k, run=r, n=len(Xr), global_id=v, error=err))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- spatial statistics
def _rad(lonlat):
    return np.deg2rad(np.asarray(lonlat, dtype=np.float64)[:, [1, 0]])


def haversine_knn(lonlat, k):
    from sklearn.neighbors import BallTree
    k = int(min(k, len(lonlat) - 1))
    d, i = BallTree(_rad(lonlat), metric="haversine").query(_rad(lonlat), k=k + 1)
    return i[:, 1:], d[:, 1:] * EARTH_R_KM


def great_circle_km(a, b):
    a, b = _rad(a), _rad(b)
    dlat, dlon = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def morans_i(values, nbr, n_perm=499, seed=0):
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return {"moran_I": np.nan, "moran_z": np.nan, "moran_p": np.nan, "moran_EI": np.nan}
    v = np.where(ok, v, np.nanmean(v))
    n = len(v)

    def _I(x):
        z = x - x.mean()
        return float((z * z[nbr].mean(1)).sum() / max((z ** 2).sum(), 1e-300))

    I = _I(v)
    rng = np.random.default_rng(seed)
    perm = np.array([_I(rng.permutation(v)) for _ in range(int(n_perm))])
    z = (I - perm.mean()) / (perm.std() + 1e-12) if len(perm) else np.nan
    p = (1 + (np.abs(perm - perm.mean()) >= abs(I - perm.mean())).sum()) / (len(perm) + 1) if len(perm) else np.nan
    return {"moran_I": I, "moran_z": float(z), "moran_p": float(p), "moran_EI": -1.0 / (n - 1)}


def variogram(values, lonlat, bins_km, max_pairs=200000, seed=0):
    v = np.asarray(values, float)
    ok = np.where(np.isfinite(v))[0]
    if len(ok) < 10:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    m = min(max_pairs, len(ok) * (len(ok) - 1) // 2)
    a, b = rng.choice(ok, m), rng.choice(ok, m)
    keep = a != b
    a, b = a[keep], b[keep]
    d = great_circle_km(lonlat[a], lonlat[b])
    g = 0.5 * (v[a] - v[b]) ** 2
    cut = pd.cut(d, bins=list(bins_km), include_lowest=True)
    df = pd.DataFrame({"bin": cut, "d": d, "gamma": g}).groupby("bin", observed=True).agg(
        dist_km=("d", "mean"), semivariance=("gamma", "mean"), n_pairs=("gamma", "size")).reset_index()
    df["bin"] = df["bin"].astype(str)
    df["sill_ratio"] = df["semivariance"] / (np.nanvar(v[ok]) + 1e-12)
    return df


def density_control(values, lonlat, k):
    """Spearman(value, log distance to the k-th geographic neighbour): values that only track sampling density."""
    from scipy.stats import spearmanr
    _, d = haversine_knn(lonlat, k)
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return {"density_rho": np.nan, "density_p": np.nan}
    r = spearmanr(v[ok], np.log1p(d[ok, -1]))
    return {"density_rho": float(r.statistic), "density_p": float(r.pvalue)}


def knn_jaccard(lonlat, X, k):
    """Mean Jaccard overlap of geographic and feature-space kNN sets (1 = env space mirrors geography)."""
    from sklearn.neighbors import NearestNeighbors
    k = int(min(k, len(X) - 1))
    gi, _ = haversine_knn(lonlat, k)
    ei = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(gi, ei)]))


def mantel(lonlat, X, max_n=1500, n_perm=199, seed=0, metric="euclidean"):
    from scipy.stats import rankdata
    from sklearn.metrics.pairwise import haversine_distances, pairwise_distances
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), max_n, replace=False) if len(X) > max_n else np.arange(len(X))
    G = haversine_distances(_rad(lonlat[idx])) * EARTH_R_KM
    E = pairwise_distances(np.asarray(X)[idx], metric=metric)
    iu = np.triu_indices(len(idx), 1)
    g, e = rankdata(G[iu]), rankdata(E[iu])
    g = (g - g.mean()) / g.std()
    e = (e - e.mean()) / e.std()
    r = float((g * e).mean())
    Rg = np.zeros_like(G)
    Rg[iu] = g
    Rg = Rg + Rg.T
    cnt = 0
    for _ in range(int(n_perm)):
        p = rng.permutation(len(idx))
        cnt += float((Rg[np.ix_(p, p)][iu] * e).mean()) >= r
    return {"mantel_r": r, "mantel_p": (1 + cnt) / (1 + n_perm) if n_perm else np.nan}


def region_summary(values, groups):
    from scipy.stats import kruskal
    df = pd.DataFrame({"v": values, "group": pd.Series(groups).astype(str).values}).dropna()
    if df.empty:
        return pd.DataFrame(), {"kw_H": np.nan, "kw_p": np.nan}
    s = df.groupby("group")["v"].agg(n="size", mean="mean", median="median", std="std").reset_index()
    samples = [g["v"].values for _, g in df.groupby("group") if len(g) >= 3]
    if len(samples) >= 2:
        r = kruskal(*samples)
        return s, {"kw_H": float(r.statistic), "kw_p": float(r.pvalue)}
    return s, {"kw_H": np.nan, "kw_p": np.nan}


# ----------------------------------------------------------------------------- synthesis statistics
def kendall_w(rank_matrix):
    r = np.asarray(rank_matrix, dtype=float)
    m, n = r.shape
    if m < 2 or n < 2:
        return np.nan
    rs = r.sum(axis=0)
    s = float(((rs - rs.mean()) ** 2).sum())
    ties = sum(float((c ** 3 - c).sum()) for c in (np.unique(row, return_counts=True)[1] for row in r))
    denom = m ** 2 * (n ** 3 - n) - m * ties
    return 12.0 * s / denom if denom > 0 else np.nan


def kendall_tau_matrix(df, cols):
    from scipy.stats import kendalltau
    out = pd.DataFrame(np.nan, index=cols, columns=cols)
    for a in cols:
        for b in cols:
            ok = df[a].notna() & df[b].notna()
            out.loc[a, b] = kendalltau(df.loc[ok, a], df.loc[ok, b]).statistic if ok.sum() >= 3 else np.nan
    return out


def friedman_nemenyi(block_by_item, alpha=0.05):
    """Friedman test + Nemenyi critical difference. block_by_item: DataFrame (blocks x items), higher = better.
    Returns (summary dict, mean ranks Series, pairwise p DataFrame)."""
    from scipy.stats import friedmanchisquare, rankdata, studentized_range
    X = block_by_item.dropna(axis=0, how="any")
    n, k = X.shape
    if n < 2 or k < 2:
        return dict(n_blocks=n, n_items=k, chi2=np.nan, p=np.nan, cd=np.nan), pd.Series(dtype=float), pd.DataFrame()
    R = np.vstack([rankdata(-row) for row in X.to_numpy()])          # rank 1 = best
    mr = pd.Series(R.mean(0), index=X.columns).sort_values()
    chi2, p = friedmanchisquare(*X.to_numpy().T) if k >= 3 else (np.nan, np.nan)
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * n)))
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    pw = pd.DataFrame(np.nan, index=X.columns, columns=X.columns)
    for a, b in itertools.combinations(X.columns, 2):
        z = abs(mr[a] - mr[b]) / se
        pv = float(studentized_range.sf(z * np.sqrt(2), k, np.inf))
        pw.loc[a, b] = pw.loc[b, a] = pv
    return dict(n_blocks=n, n_items=k, chi2=float(chi2), p=float(p), cd=cd), mr, pw


def wilcoxon_bh(block_by_item, reference):
    from scipy.stats import wilcoxon
    from naturenet.drivers.common import bh
    rows = []
    for c in block_by_item.columns:
        if c == reference:
            continue
        d = (block_by_item[c] - block_by_item[reference]).dropna()
        p = float(wilcoxon(d).pvalue) if len(d) >= 5 and (d != 0).any() else np.nan
        rows.append(dict(item=c, reference=reference, n=len(d), median_diff=float(d.median()) if len(d) else np.nan,
                         p_wilcoxon=p))
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = bh(out["p_wilcoxon"].to_numpy())
    return out


def variance_decomposition(df, factors, value):
    """Sum-of-squares shares of the main effects and two-way interactions of crossed factors (cell means; the
    highest-order interaction is the residual). Incomplete designs are reduced to the complete crossed subset."""
    cell = df.groupby(list(factors), as_index=False)[value].mean().dropna()
    full = cell.set_index(list(factors))[value]
    if len(factors) > 1:
        lev = [full.index.get_level_values(f).unique() for f in factors]
        idx = pd.MultiIndex.from_product(lev, names=factors)
        full = full.reindex(idx)
        for f in factors:            # drop levels with missing cells until complete
            bad = full[full.isna()].index.get_level_values(f).unique() if full.isna().any() else []
            if len(bad):
                full = full[~full.index.get_level_values(f).isin(bad)]
        full = full.dropna()
    y = full
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    eff, ss = {}, {}
    for f in factors:
        m = y.groupby(level=f).mean() - grand
        eff[f] = m
        ss[f] = float((y.groupby(level=f).size() * m ** 2).sum())
    if len(factors) > 2:
        for f1, f2 in itertools.combinations(factors, 2):
            cm = y.groupby(level=[f1, f2]).mean()
            npc = y.groupby(level=[f1, f2]).size()
            ix = cm.index
            inter = (cm.to_numpy() - eff[f1].reindex(ix.get_level_values(f1)).to_numpy()
                     - eff[f2].reindex(ix.get_level_values(f2)).to_numpy() - grand)
            ss[f"{f1} x {f2}"] = float((npc.to_numpy() * inter ** 2).sum())
    ss["residual"] = ss_total - sum(ss.values())
    rows = [dict(source=k, sum_sq=v, share=v / ss_total if ss_total > 0 else np.nan) for k, v in ss.items()]
    rows.append(dict(source="total", sum_sq=ss_total, share=1.0))
    out = pd.DataFrame(rows)
    out["n_cells"] = len(y)
    return out
