"""
Pairwise relationships between the per-animal SWIRL time series (plot_swirl_new / swirl_track_series
export: <uid>_time_series_pd.csv / .xlsx). Adapted from the TRACES_ES library.

Rewritten. Fixes vs. the old version:
- statistics are computed per series (one animal, one segment, one contiguous run of fixes) and pooled
  (Fisher z, weighted by effective N); the old code concatenated all animals, so rolling windows, CCF and
  lags crossed from one animal / segment to the next;
- categorical columns (latent, env, season) use Cramer's V / mutual information / eta^2 instead of
  Pearson / Spearman / Kendall on category ids;
- CCF normalised by N (in [-1, 1]); ONE lag convention: lag k > 0 means series 1 leads series 2
  (corr(s1[t], s2[t + k])); the old CCF (k > 0 = s1 leads) and delayed correlation (k > 0 = s2 leads)
  disagreed. Lags are in fixes; median dt_h is reported to convert to hours;
- p-values use the effective sample size n (1 - r1 r2) / (1 + r1 r2) (lag-1 autocorrelations) and are
  Benjamini-Hochberg corrected across pairs;
- unique p-value columns (the summary dict repeated 'P-Value', keeping only Kendall's); constant series
  skipped with a count instead of raising; month as sin / cos; 'seasaon' / 'season' dropped only if present;
  each pair analysed once; main() with arguments instead of module-level code and hard-coded paths.

All inputs come from a YAML config (python correlation_analysis_shark.py -y config.yaml); example:
config/time_series/correlation_whale.yaml. Runs every (second_dir, slice) under main_dir:

    main_dir: /data/.../Blue_Whale_v1/      # root
    second_dir: [whale]                     # sub-directories (e.g. s1_complex, s2_simple); [""] = main_dir
    data_slices: {full: [826_time_series_pd.xlsx, ...], males: [...], ...}   # file names per slice;
                                            # omitted -> one slice "full" = every *_time_series_pd.(csv|xlsx)
    keys: [full, males]                     # slices to run (default: all data_slices)
    out_dir: null                           # default <main_dir>/<second_dir>/<key>; else <out_dir>/<second_dir>/<key>
    pair_filter: [reward, latent]           # keep pairs where either column contains one of these ([] = all)
    parent_child: {season: [month]}         # pairs never compared
    exclude: [], categorical: [], drop_columns: [seasaon]
    min_rows: 4                             # files with fewer rows skipped (as in the whale script)
    missing_files: warn                     # warn | error
    top_n: null                             # keep the n strongest pairs in the saved table / plots (whale: 24)
    rolling_window: 20, max_lag: 8, significance_level: 0.05, min_correlation: 0.3, min_len: 8, n_bins: 8
"""
import argparse
import glob
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from naturenet.environment.pipeline_checks import check_single_series  # noqa: E402

CONFIG = {
    "rolling_window": 20,
    "max_lag": 8,
    "significance_level": 0.05,
    "min_correlation": 0.3,
    "min_len": 8,       # shortest series used for a statistic
    "n_bins": 8,        # quantile bins for continuous columns in categorical comparisons
}

ID_COLUMNS = ["uid", "segment", "seg_pos", "split", "timestamp", "Time", "month", "seasaon",
              "distance_traveled", "latitude", "longitude"]
CATEGORICAL = ["latent", "env", "season"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read(path):
    return pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path, header=0)


def load_series(files: List[str], drop=("seasaon",), min_rows=0) -> List[pd.DataFrame]:
    """One DataFrame per contiguous run (uid, segment, consecutive seg_pos); files < min_rows skipped."""
    runs = []
    for f in files:
        df = _read(f).drop(columns=[c for c in drop if c], errors="ignore")
        if len(df) < min_rows:
            print(f"Skipping {os.path.basename(f)}: {len(df)} rows < min_rows={min_rows}")
            continue
        if "uid" not in df:
            df["uid"] = os.path.basename(f).replace("_time_series_pd", "").rsplit(".", 1)[0]
        if "segment" not in df:
            df["segment"] = 0
        if "seg_pos" not in df:
            df["seg_pos"] = np.arange(len(df))
        if "month" in df and "month_sin" not in df:
            df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
            df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
        for _, g in df.groupby(["uid", "segment"], sort=False):
            g = g.sort_values("seg_pos")
            brk = np.r_[0, np.cumsum(np.diff(g["seg_pos"].to_numpy()) != 1)]
            for _, r in g.groupby(brk):
                check_single_series(r)
                runs.append(r.reset_index(drop=True))
    if not runs:
        raise ValueError("No series loaded")
    return runs


def column_types(runs, exclude=(), categorical=()):
    cols = [c for c in runs[0].columns if c not in set(ID_COLUMNS) | set(exclude)]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(runs[0][c])]
    cat = set(CATEGORICAL) | set(categorical)
    return [c for c in cols if c not in cat], [c for c in cols if c in cat]


# ---------------------------------------------------------------------------
# Per-series statistics
# ---------------------------------------------------------------------------

def _lag1(x):
    x = x - x.mean()
    d = np.dot(x, x)
    return float(np.dot(x[:-1], x[1:]) / d) if d > 0 and len(x) > 2 else 0.0


def n_effective(x, y):
    r1, r2 = _lag1(x), _lag1(y)
    f = (1 - r1 * r2) / (1 + r1 * r2) if (1 + r1 * r2) > 0 else 1.0
    return float(np.clip(len(x) * f, 3.0 + 1e-9, len(x)))


def ccf(x, y, max_lag):
    """Normalised CCF, c[k] = corr(x[t], y[t + k]) / N-normalised, k = -max_lag..max_lag (k > 0: x leads)."""
    n = len(x)
    xs, ys = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    out = np.full(2 * max_lag + 1, np.nan)
    for i, k in enumerate(range(-max_lag, max_lag + 1)):
        if abs(k) < n:
            out[i] = np.dot(xs[:n - k], ys[k:]) / n if k >= 0 else np.dot(xs[-k:], ys[:n + k]) / n
    return out


def _pair_runs(runs, a, b, min_len):
    for r in runs:
        x, y = r[a].to_numpy(dtype=float), r[b].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_len:
            continue
        # keep the longest contiguous finite stretch so lags stay meaningful
        idx = np.flatnonzero(ok)
        brk = np.flatnonzero(np.diff(idx) != 1)
        pieces = np.split(idx, brk + 1)
        for p in pieces:
            if len(p) >= min_len:
                yield x[p], y[p], r


def continuous_pair(runs, a, b, cfg=CONFIG):
    zs = {"pearson": [], "spearman": [], "kendall": []}
    w, n_obs, n_const, ccfs, ccf_w, roll = [], 0, 0, [], [], []
    for x, y, _ in _pair_runs(runs, a, b, cfg["min_len"]):
        if x.std() == 0 or y.std() == 0:
            n_const += 1
            continue
        ne = n_effective(x, y)
        zs["pearson"].append(np.arctanh(np.clip(stats.pearsonr(x, y)[0], -0.999999, 0.999999)))
        zs["spearman"].append(np.arctanh(np.clip(stats.spearmanr(x, y)[0], -0.999999, 0.999999)))
        zs["kendall"].append(np.arctanh(np.clip(stats.kendalltau(x, y)[0], -0.999999, 0.999999)))
        w.append(ne - 3)
        n_obs += len(x)
        ccfs.append(ccf(x, y, cfg["max_lag"]))
        ccf_w.append(len(x))
        if len(x) >= cfg["rolling_window"]:
            rc = pd.Series(x).rolling(cfg["rolling_window"]).corr(pd.Series(y)).dropna()
            roll.extend(rc[np.isfinite(rc)].tolist())
    out = {"Series 1": a, "Series 2": b, "Kind": "continuous", "n_series": len(w), "n_obs": n_obs,
           "n_constant_skipped": n_const}
    if not w:
        return out
    w = np.asarray(w)
    se = 1 / np.sqrt(w.sum())
    for m, z in zs.items():
        zp = float(np.dot(w, z) / w.sum())
        out[m.capitalize()] = float(np.tanh(zp))
        out[f"{m.capitalize()} p"] = float(2 * stats.norm.sf(abs(zp) / se))
    c = np.nansum(np.asarray(ccfs) * np.asarray(ccf_w)[:, None], axis=0) / np.sum(ccf_w)
    lags = np.arange(-cfg["max_lag"], cfg["max_lag"] + 1)
    i = int(np.nanargmax(np.abs(c)))
    out.update({"Max CCF": float(c[i]), "Optimal Lag": int(lags[i]), "Zero-lag CCF": float(c[cfg["max_lag"]]),
                "Lag strength ratio": float(abs(c[i]) / max(abs(c[cfg["max_lag"]]), 1e-12)),
                "Rolling Mean": float(np.mean(roll)) if roll else np.nan,
                "Rolling Std": float(np.std(roll)) if roll else np.nan, "_ccf": c})
    return out


def _binned(v, n_bins):
    q = np.unique(np.nanquantile(v, np.linspace(0, 1, n_bins + 1)))
    return np.digitize(v, q[1:-1]) if len(q) > 2 else np.zeros(len(v), dtype=int)


def categorical_pair(runs, a, b, a_cat, b_cat, cfg=CONFIG):
    xs, ys = [], []
    for x, y, _ in _pair_runs(runs, a, b, 2):
        xs.append(x)
        ys.append(y)
    out = {"Series 1": a, "Series 2": b, "Kind": "categorical", "n_series": len(xs),
           "n_obs": int(sum(len(x) for x in xs))}
    if not xs:
        return out
    x, y = np.concatenate(xs), np.concatenate(ys)
    if not a_cat or not b_cat:  # eta^2: share of the continuous variance explained by the categories
        c, v = (x, y) if a_cat else (y, x)
        groups = [v[c == k] for k in np.unique(c)]
        ss_t = ((v - v.mean()) ** 2).sum()
        out["Eta2"] = float(sum(len(g) * (g.mean() - v.mean()) ** 2 for g in groups) / ss_t) if ss_t > 0 else np.nan
    xb = x.astype(int) if a_cat else _binned(x, cfg["n_bins"])
    yb = y.astype(int) if b_cat else _binned(y, cfg["n_bins"])
    tab = pd.crosstab(xb, yb).to_numpy()
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        out["n_constant_skipped"] = 1
        return out
    chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
    n = tab.sum()
    phi2 = chi2 / n
    r, k = tab.shape  # bias-corrected Cramer's V (Bergsma 2013)
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    out["Cramers V"] = float(np.sqrt(phi2c / max(min(kc - 1, rc - 1), 1e-12)))
    pxy = tab / n
    px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    nz = pxy > 0
    out["Mutual information"] = float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())
    out["Chi2 p"] = float(p)  # NB: assumes independent fixes (no autocorrelation correction)
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        qq = np.empty_like(r)
        qq[o] = np.minimum(r, 1)
        q[ok] = qq
    return q


def classify(row, cfg=CONFIG):
    if row.get("Kind") != "continuous" or not np.isfinite(row.get("Pearson", np.nan)):
        return "categorical" if row.get("Kind") == "categorical" else "insufficient", ""
    diff = abs(row["Pearson"] - row["Spearman"])
    if diff < 0.1 and (not np.isfinite(row["Rolling Std"]) or row["Rolling Std"] < 0.2):
        return "linear", "pearson"
    if diff > 0.2:
        return "non_linear", "spearman, kendall"
    if row["Lag strength ratio"] > 1.2 and row["Optimal Lag"] != 0:
        return "lagged", "ccf"
    return "complex", "ccf, spearman"


def valid_pairs(cols, pair_filter=(), parent_child=None):
    """All pairs, minus parent / child pairs, keeping only pairs that touch a pair_filter substring."""
    parent_child = parent_child or {}
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if any((a == p and b in ch) or (b == p and a in ch) for p, ch in parent_child.items()):
                continue
            if pair_filter and not any(f in a or f in b for f in pair_filter):
                continue
            out.append((a, b))
    return out


def analyze(runs, exclude=(), categorical=(), cfg=CONFIG, pair_filter=(), parent_child=None) -> pd.DataFrame:
    cont, cat = column_types(runs, exclude, categorical)
    pairs = valid_pairs(cont + cat, pair_filter, parent_child)
    if not pairs:
        raise ValueError(f"No column pairs left (columns {cont + cat}, pair_filter {list(pair_filter)})")
    rows = []
    for a, b in pairs:
        if a in cont and b in cont:
            rows.append(continuous_pair(runs, a, b, cfg))
        else:
            rows.append(categorical_pair(runs, a, b, a in cat, b in cat, cfg))
    df = pd.DataFrame(rows)
    pcol = np.where(df["Kind"] == "continuous", df.get("Pearson p", np.nan), df.get("Chi2 p", np.nan))
    df["q (BH)"] = bh_fdr(pcol)
    df["significant"] = df["q (BH)"] < cfg["significance_level"]
    rel = [classify(r, cfg) for _, r in df.iterrows()]
    df["Relationship Type"] = [r[0] for r in rel]
    df["Best Method"] = [r[1] for r in rel]
    strength = np.where(df["Kind"] == "continuous",
                        df[[c for c in ("Pearson", "Spearman", "Kendall") if c in df]].abs().max(axis=1),
                        df.get("Cramers V", np.nan))
    df["Abs_Max_Corr"] = strength
    df["Confidence"] = np.where(df["significant"], strength, 0.0)
    dts = [r["dt_h"].median() for r in runs if "dt_h" in r]
    df.attrs["median_dt_h"] = float(np.nanmedian(dts)) if dts else np.nan
    return df.sort_values("Abs_Max_Corr", ascending=False, na_position="last").reset_index(drop=True)


def plot_results(df, out_dir, max_lag):
    os.makedirs(out_dir, exist_ok=True)
    cont = df[(df["Kind"] == "continuous") & df["Pearson"].notna()] if "Pearson" in df else df.iloc[:0]
    if len(cont):
        top = cont.head(20)
        fig, ax = plt.subplots(figsize=(10, 0.35 * len(top) + 1.5))
        ypos = np.arange(len(top))
        for j, m in enumerate(["Pearson", "Spearman", "Kendall"]):
            ax.barh(ypos + (j - 1) * 0.27, top[m], height=0.27, label=m)
        ax.set_yticks(ypos)
        ax.set_yticklabels(top["Series 1"] + " - " + top["Series 2"], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("pooled correlation")
        ax.legend()
        fig.savefig(os.path.join(out_dir, "correlation_comparison.png"), bbox_inches="tight", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        lags = np.arange(-max_lag, max_lag + 1)
        for _, r in top.head(8).iterrows():
            ax.plot(lags, r["_ccf"], marker=".", label=f'{r["Series 1"]} - {r["Series 2"]}')
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("lag (fixes; > 0: series 1 leads)")
        ax.set_ylabel("CCF")
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(out_dir, "CCF_analysis.png"), bbox_inches="tight", dpi=200)
        plt.close(fig)

    names = sorted(set(df["Series 1"]) | set(df["Series 2"]))
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    for _, r in df.iterrows():
        mat.loc[r["Series 1"], r["Series 2"]] = mat.loc[r["Series 2"], r["Series 1"]] = r["Abs_Max_Corr"]
    fig, ax = plt.subplots(figsize=(0.5 * len(names) + 3, 0.5 * len(names) + 2))
    im = ax.imshow(mat.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(im, ax=ax, label="|corr| (continuous) / Cramer's V (categorical)")
    fig.savefig(os.path.join(out_dir, "matrix_analysis.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def run(files, out_dir, exclude=(), categorical=(), cfg=CONFIG, pair_filter=(), parent_child=None,
        drop_columns=("seasaon",), min_rows=0, top_n=None, label=""):
    runs = load_series(files, drop_columns, min_rows)
    df = analyze(runs, exclude, categorical, cfg, pair_filter, parent_child)
    if top_n:
        df = df.head(int(top_n))  # already sorted by strength; q-values computed over all pairs first
    os.makedirs(out_dir, exist_ok=True)
    df.drop(columns=["_ccf"], errors="ignore").to_csv(os.path.join(out_dir, "correlation_results.csv"), index=False)
    plot_results(df, out_dir, cfg["max_lag"])
    print(f"{label}{len(runs)} series, {len(df)} pairs, {int(df['significant'].sum())} significant (BH q < "
          f"{cfg['significance_level']}); lags in fixes, median dt = {df.attrs.get('median_dt_h', np.nan):.2f} h")
    return df


def _read_yaml(path):
    try:
        from sit_fuse.utils import read_yaml
        return read_yaml(path)
    except ImportError:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)


def _as_list(v):
    return [] if v is None else (list(v) if isinstance(v, (list, tuple)) else [v])


def run_from_config(conf):
    """Loop over second_dir x slices (see module docstring for the keys). Returns {(second_dir, key): df}."""
    main_dir = conf["main_dir"]
    second_dirs = _as_list(conf.get("second_dir")) or [""]
    slices = conf.get("data_slices")
    keys = _as_list(conf.get("keys")) or (list(slices) if slices else ["full"])
    cfg = dict(CONFIG, **{k: conf[k] for k in ("rolling_window", "max_lag", "significance_level",
                                                "min_correlation", "min_len", "n_bins") if k in conf})
    pair_filter = _as_list(conf.get("pair_filter", ["reward", "latent"]))
    common = dict(exclude=_as_list(conf.get("exclude")), categorical=_as_list(conf.get("categorical")), cfg=cfg,
                  pair_filter=pair_filter, parent_child=conf.get("parent_child") or {},
                  drop_columns=_as_list(conf.get("drop_columns", ["seasaon"])), min_rows=int(conf.get("min_rows", 4)),
                  top_n=conf.get("top_n"))
    missing = conf.get("missing_files", "warn")
    results = {}
    for sd in second_dirs:
        data_dir = os.path.join(main_dir, sd)
        for key in keys:
            if slices:
                if key not in slices:
                    raise KeyError(f"keys entry '{key}' not in data_slices ({list(slices)})")
                files = [os.path.join(data_dir, f) for f in _as_list(slices[key])]
            else:
                files = sorted(glob.glob(os.path.join(data_dir, "*_time_series_pd.csv"))) or \
                    sorted(glob.glob(os.path.join(data_dir, "*_time_series_pd.xlsx")))
            absent = [f for f in files if not os.path.exists(f)]
            if absent:
                msg = f"{sd}/{key}: {len(absent)} missing files, e.g. {absent[:3]}"
                if missing == "error":
                    raise FileNotFoundError(msg)
                print("WARNING:", msg)
                files = [f for f in files if os.path.exists(f)]
            if not files:
                print(f"WARNING: {sd}/{key}: no files; skipped")
                continue
            out_dir = os.path.join(conf.get("out_dir") or main_dir, sd, key)
            results[(sd, key)] = run(files, out_dir, label=f"[{sd or '.'}/{key}] ", **common)
    return results


def main():
    p = argparse.ArgumentParser(description="Pairwise relationships between SWIRL time series (YAML-driven)")
    p.add_argument("-y", "--yaml", required=True, help="YAML config (see module docstring)")
    a = p.parse_args()
    run_from_config(_read_yaml(a.yaml))


if __name__ == "__main__":
    main()
