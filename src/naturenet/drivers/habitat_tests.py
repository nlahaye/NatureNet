"""
Habitat tests by behavioural mode (literature comparison analyses 1-3; generic for any NatureNet / SIT-FUSE run).

All three use the step-selection design of naturenet.drivers.availability: for every decoded step (decision fix
p -> next fix p + 1) of a final tool, the environment at the fix actually entered (used) is compared with the
environment at the m available points drawn from fix p (same date, empirical step / turn distributions). The step is
labelled with the mode decoded at the decision fix (mode_at: step, default; mode_at: fix uses the latent of the
entered fix). Uncertainty is a bootstrap over animals (multinomial animal weights shared by every row, so contrasts
between modes / groups are paired); p_boot is the two-sided bootstrap sign p, q_bh is Benjamini-Hochberg over each
output table.

  parts
  features   analysis 1: binary feature masks (fronts = per-scene upper decile of a gradient magnitude, e.g. SST or
             log10 CHL;
             eddy cores = Okubo-Weiss W < -0.2 std(W) per scene, split cyclonic / anticyclonic by the sign of the
             Rossby number; optional threshold masks). Per mode: used and available prevalence, Mantel-Haenszel
             odds ratio over the matched sets (1 used : m available), and the mode-vs-rest log odds-ratio contrast.
             Thresholds come from the scene_feature_stats.csv of lagged_env --part scene (threshold_source: scene)
             or from the pooled used + available values (threshold_source: pooled; no scene reads, cruder).
             -> feature_use.csv, feature_mode_contrast.csv, feature_use.png
  lags       analysis 2: lagged / rolling covariates (lagged_env --part lagged, used and --avl). Per mode and window:
             use vs availability of the lagged value (lag_use_availability.csv); mode association = mean
             within-animal-centred z value at the decision fixes of mode k minus the other modes
             (lag_mode_association.csv); reward association = rank correlation between the decoded mode's
             reward at the decision fix (series reward_likelihood) and each window, plus the joint standardised
             regression reward ~ same-day (base window) + lagged window (lag_reward_association.csv,
             lag_reward_joint.csv). A calm-wind reward that becomes a lagged-upwelling effect shows as a negative
             same-day coefficient that shrinks while the lagged wind / Ekman coefficient turns positive.
             -> lag_*.csv, lag_profiles.png
  seasons    analysis 3: use vs availability per mode and month group (groups: {wet: [5..10], dry: [11..4]} or any
             month lists), the paired contrast between two groups of both the selection (diff in sd units) and the
             used mean, and mode occupancy by group. -> season_use_availability.csv, season_contrast.csv,
             season_occupancy.csv, season_contrast.png
  lag_profiles  state-level companion of analysis 2: per environment state, the mean of each lagged column over
             the tracked fixes in that state (n >= lag_profiles.min_fixes) -> lagged_state_profiles.csv, indexed by
             state id (profile_index: state). reward_env_projection reads it through extra_profiles to rerun the
             reward projection / use vs availability with lagged wind / Ekman. Occupancy-conditioned (only visited
             states, fix-weighted), unlike the scene-wide SIT-FUSE cluster medians; report it as such.
  Every part also writes mode_kinematics_<tool>.csv (speed, step, |turn|, exploit / explore class per mode) so modes
  can be named (fast tortuous, slow, ...).

YAML (top-level or under habitat_tests:), see config/v2/drivers/<species>_habitat_tests.yaml
  out_dir, sample_inputs_yaml, availability {sample_inputs_yaml, avl_dir, n_available, seed}
  tools: [{label, series_dir}]          exported final-run series (plots/series_<model>_<order>)
  mode_at: step | fix,  n_boot, seed
  features: {fronts: {name: {channel, percentile}}, eddy: {channel, threshold, polarity_channel}, threshold_source,
             masks: {name: {channel, op: ">" | "<" | ">=" | "<=", value}}}
  scene_stats / lagged: see naturenet.drivers.lagged_env
  lags: {base: d0, channels: [...] (null = every lagged channel)}
  seasons: {groups: {name: [months]}, contrast: [a, b], channels: [...]}

    python -m naturenet.drivers.habitat_tests -y config/v2/drivers/whales_habitat_tests.yaml --part features
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

from naturenet.drivers.common import abs_cfg, atomic_csv, atomic_json, bh, read_yaml

PARTS = ("features", "lags", "seasons", "lag_profiles")
OPS = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal}


# ============================================================================ inputs
def load_series(series_dir):
    files = sorted(glob.glob(os.path.join(series_dir, "*_time_series_pd.csv")))
    if not files:
        raise FileNotFoundError(f"no *_time_series_pd.csv in {series_dir} (run plot_hiavi / plot_swirl_new first)")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, format="mixed").dt.tz_convert(None)
    d = d[np.isfinite(d["latent"])].copy()
    d["latent"] = d["latent"].astype(int)
    return d


def mode_kinematics(series):
    g = series.assign(abs_turn=np.abs(series.get("turn_angle", np.nan))).groupby("latent")
    out = pd.DataFrame(dict(n_rows=g.size(), share=g.size() / len(series),
                            speed_out_kmh_median=g["speed_out_kmh"].median() if "speed_out_kmh" in series else np.nan,
                            step_out_km_median=g["step_out_km"].median() if "step_out_km" in series else np.nan,
                            abs_turn_mean=g["abs_turn"].mean()))
    if "latent_class" in series:
        out["latent_class"] = g["latent_class"].agg(lambda x: x.dropna().astype(str).mode().iloc[0]
                                                    if len(x.dropna()) else "")
    out["speed_rank"] = out["speed_out_kmh_median"].rank(ascending=False)
    return out.reset_index().rename(columns={"latent": "mode"})


def _pickles(d, uid, keys, suffix):
    out = {}
    for k in keys:
        p = os.path.join(d, f"{uid}_{k}_{suffix}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[k] = [np.asarray(x, np.float64) for x in pickle.load(f)]
    return out


def gather(store, keys, segs, pos, n_col):
    """(n, n_col) values at (key, seg, pos); NaN where the file / row / columns are missing."""
    out = np.full((len(keys), n_col), np.nan)
    for i, (k, s, p) in enumerate(zip(keys, segs, pos)):
        a = store.get(k)
        if a is None or s >= len(a):
            continue
        x = a[s]
        if x.ndim == 2 and p < len(x) and x.shape[1] >= n_col:
            out[i] = x[p, :n_col]
    return out


def _channels(path):
    with open(path) as f:
        return list(json.load(f))


class Design:
    """Matched used / available values for the steps of one tool."""

    def __init__(self, spec, series):
        self.spec = spec
        sconf = read_yaml(abs_cfg(spec["sample_inputs_yaml"]))
        self.df_dir, self.uid = sconf["df_dir"], sconf["df_run_uid"]
        av = spec.get("availability") or {}
        self.avl_dir = av.get("avl_dir")
        meta = os.path.join(self.avl_dir or "", "availability_meta.json")
        self.m = int(json.load(open(meta))["n_available"]) if os.path.exists(meta) else int(av.get("n_available", 10))
        with open(os.path.join(self.df_dir, self.uid + "_dfs.pkl"), "rb") as f:
            self.dfs = pd.read_pickle(f)
        at = spec.get("mode_at", "step")
        if at not in ("step", "fix"):
            raise ValueError("mode_at must be step or fix")
        s = series.rename(columns={"uid": "animal", "segment": "seg", "seg_pos": "pos"})
        s = s[s["animal"].isin(self.dfs.keys())].copy()
        s["tpos"] = s["pos"] + 1 if at == "step" else s["pos"]
        n = np.array([len(self.dfs[a][g]) if g < len(self.dfs[a]) else 0 for a, g in zip(s["animal"], s["seg"])])
        s = s[(s["tpos"] >= 1) & (s["tpos"] < n)].reset_index(drop=True)
        from naturenet.environment.sample_env_inputs import _fix_dates
        dates = {}
        for a, g in s[["animal", "seg"]].drop_duplicates().itertuples(index=False):
            dates[(a, g)] = _fix_dates(self.dfs[a][g])
        s["date"] = pd.to_datetime([dates[(a, g)][p] for a, g, p in zip(s["animal"], s["seg"], s["tpos"])])
        s["month"] = s["date"].dt.month.where(s["date"].notna(), s["timestamp"].dt.month)
        s["mode"] = s["latent"].astype(int)
        self.steps = s
        self.animals = np.array(sorted(s["animal"].unique()))
        self.aidx = np.searchsorted(self.animals, s["animal"].to_numpy())

    def source(self, kind):
        """(used dir, available dir, file suffix, channel json name) of a value source."""
        if kind == "inputs":
            return self.df_dir, self.avl_dir, "states_input", "_states_input_channels.json"
        lg = self.spec.get("lagged") or {}
        u = lg.get("out_dir") or os.path.join(self.spec["out_dir"], "lagged_inputs")
        a = lg.get("avl_out_dir") or os.path.join(self.spec["out_dir"], "lagged_inputs_avl")
        return u, a, "lagged_input", "_lagged_input_channels.json"

    def values(self, kind, channels=None, at="used"):
        """at = used: (n, C) at the entered fix; decision: (n, C) at the decision fix; avail: (n, m, C)."""
        du, da, suf, cj = self.source(kind)
        d = da if at == "avail" else du
        if d is None:
            raise FileNotFoundError(f"no directory for {kind} / {at}")
        names = _channels(os.path.join(d, self.uid + cj))
        sel = list(channels) if channels else names
        miss = [c for c in sel if c not in names]
        if miss:
            print(f"habitat_tests: {kind} channels {miss} not available - dropped")
            sel = [c for c in sel if c in names]
        cols = [names.index(c) for c in sel]
        s = self.steps
        if at == "avail":
            keys = [f"{a}__avl{j}" for a in self.animals for j in range(self.m)]
            st = _pickles(d, self.uid, keys, suf)
            if not st:
                raise FileNotFoundError(f"no {suf} files in {d} (sample the available points first)")
            out = np.full((len(s), self.m, len(sel)), np.nan)
            for j in range(self.m):
                v = gather(st, [f"{a}__avl{j}" for a in s["animal"]], s["seg"].to_numpy(), s["tpos"].to_numpy(),
                           len(names))
                out[:, j] = v[:, cols]
            return out, sel
        st = _pickles(d, self.uid, list(self.animals), suf)
        pos = s["tpos"] if at == "used" else s["pos"]
        v = gather(st, list(s["animal"]), s["seg"].to_numpy(), pos.to_numpy(), len(names))
        return v[:, cols], sel


# ============================================================================ bootstrap statistics
def boot_weights(n_animals, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_animals, np.full(n_animals, 1.0 / n_animals), size=n_boot).astype(float)


def asum(aidx, n_animals, v):
    """Per-animal sums of v (n, C) ignoring NaN, and the per-animal finite counts."""
    v = np.asarray(v, float)
    v2 = v.reshape(len(v), -1)
    ok = np.isfinite(v2)
    S = np.zeros((n_animals, v2.shape[1]))
    N = np.zeros_like(S)
    np.add.at(S, aidx, np.where(ok, v2, 0.0))
    np.add.at(N, aidx, ok.astype(float))
    return S, N


def p_sign(b, point=None):
    b = b[np.isfinite(b)]
    if not len(b):
        return np.nan
    return float(min(1.0, 2 * min((np.sum(b <= 0) + 1) / (len(b) + 1), (np.sum(b >= 0) + 1) / (len(b) + 1))))


def _ci(b):
    b = b[np.isfinite(b)]
    return (np.percentile(b, 2.5), np.percentile(b, 97.5)) if len(b) else (np.nan, np.nan)


def _ratio(Wb, S, N):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Wb @ S) / (Wb @ N)


def ua_continuous(d, U, A, names, by, Wb):
    """Per group of `by` and channel: used - available (mean over steps), sd units, animal-bootstrap CI.
    Returns the table and the bootstrap draws {group: (n_boot, C) sd-unit diffs, used means}."""
    av = np.nanmean(A, 1) if A.ndim == 3 else A
    diff = U - av
    pooled = np.concatenate([U, A.reshape(-1, A.shape[-1])]) if A.ndim == 3 else np.concatenate([U, A])
    sd = np.nanstd(pooled, 0)
    sd = np.where(sd > 0, sd, np.nan)
    rows, draws = [], {}
    nA = Wb.shape[1]
    for gkey, g in d.groupby(by, sort=True):
        ii = g.index.to_numpy()
        a = g["_aidx"].to_numpy()
        Sd, Nd = asum(a, nA, diff[ii])
        Su, Nu = asum(a, nA, U[ii])
        Sa, Na = asum(a, nA, av[ii])
        bd = _ratio(Wb, Sd, Nd) / sd
        bu = _ratio(Wb, Su, Nu)
        draws[gkey] = (bd, bu)
        gk = gkey if isinstance(gkey, tuple) else (gkey,)
        for j, c in enumerate(names):
            m = Sd[:, j].sum() / max(Nd[:, j].sum(), 1)
            lo, hi = _ci(bd[:, j])
            rows.append(dict(**dict(zip(by, gk)), channel=c, used_mean=Su[:, j].sum() / max(Nu[:, j].sum(), 1),
                             available_mean=Sa[:, j].sum() / max(Na[:, j].sum(), 1), diff=m, diff_sd_units=m / sd[j],
                             ci_low=lo, ci_high=hi, p_boot=p_sign(bd[:, j]), n_steps=int(Nd[:, j].sum()),
                             n_animals=int((Nd[:, j] > 0).sum())))
    t = pd.DataFrame(rows)
    if len(t):
        t["q_bh"] = bh(t["p_boot"].to_numpy())
        t["ci_excludes_0"] = (t["ci_low"] > 0) | (t["ci_high"] < 0)
    return t, draws


def mh_terms(e, X):
    """Matched sets (1 used : m available) for a binary feature: per step Mantel-Haenszel numerator / denominator,
    used exposure and available prevalence (NaN where the step has no information)."""
    ok_c = np.isfinite(X)
    m = ok_c.sum(1).astype(float)
    x = np.where(ok_c, X, 0).sum(1)
    good = np.isfinite(e) & (m > 0)
    n = m + 1
    ee = np.where(good, e, 0.0)
    num = np.where(good, ee * (m - x) / n, np.nan)
    den = np.where(good, x * (1 - ee) / n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        pa = np.where(good, x / m, np.nan)
    return num, den, np.where(good, e, np.nan), pa


def feature_table(d, E, X, names, Wb):
    """Per mode (and pooled 'all') and feature: prevalences, MH odds ratio, animal-bootstrap CI; plus draws."""
    nA = Wb.shape[1]
    rows, draws = [], {}
    groups = [("all", d.index.to_numpy())] + [(int(k), g.index.to_numpy()) for k, g in d.groupby("mode")]
    for k, ii in groups:
        a = d.loc[ii, "_aidx"].to_numpy()
        for j, f in enumerate(names):
            num, den, eu, pa = mh_terms(E[ii, j], X[ii, :, j])
            Sn, _ = asum(a, nA, num[:, None])
            Sd, _ = asum(a, nA, den[:, None])
            Se, Ne = asum(a, nA, eu[:, None])
            Sp, Np = asum(a, nA, pa[:, None])
            with np.errstate(invalid="ignore", divide="ignore"):
                orr = Sn.sum() / Sd.sum()
                b = np.log((Wb @ Sn[:, 0]) / (Wb @ Sd[:, 0]))
            b[~np.isfinite(b)] = np.nan
            draws[(k, f)] = b
            lo, hi = _ci(b)
            rows.append(dict(mode=k, feature=f, p_used=Se.sum() / max(Ne.sum(), 1),
                             p_available=Sp.sum() / max(Np.sum(), 1), or_mh=orr,
                             log_or=np.log(orr) if orr > 0 else np.nan, ci_low=lo, ci_high=hi, p_boot=p_sign(b),
                             n_steps=int(Ne.sum()), n_used_in_feature=int(Se.sum()),
                             n_animals=int((Ne[:, 0] > 0).sum())))
    t = pd.DataFrame(rows)
    t["q_bh"] = bh(t["p_boot"].to_numpy())
    t["ci_excludes_0"] = (t["ci_low"] > 0) | (t["ci_high"] < 0)
    return t, draws


def mode_vs_rest(d, E, X, names, Wb):
    """log OR_MH(mode k) - log OR_MH(all other modes), paired over the same animal draws."""
    nA = Wb.shape[1]
    rows = []
    for k in sorted(d["mode"].unique()):
        for j, f in enumerate(names):
            out = []
            for sel in (d["mode"] == k, d["mode"] != k):
                ii = d.index[sel].to_numpy()
                num, den, _, _ = mh_terms(E[ii, j], X[ii, :, j])
                a = d.loc[ii, "_aidx"].to_numpy()
                Sn, _ = asum(a, nA, num[:, None])
                Sd, _ = asum(a, nA, den[:, None])
                with np.errstate(invalid="ignore", divide="ignore"):
                    out.append((np.log(Sn.sum() / Sd.sum()), np.log((Wb @ Sn[:, 0]) / (Wb @ Sd[:, 0]))))
            b = out[0][1] - out[1][1]
            b[~np.isfinite(b)] = np.nan
            lo, hi = _ci(b)
            rows.append(dict(mode=int(k), feature=f, log_or_mode=out[0][0], log_or_rest=out[1][0],
                             diff=out[0][0] - out[1][0], ci_low=lo, ci_high=hi, p_boot=p_sign(b)))
    t = pd.DataFrame(rows)
    t["q_bh"] = bh(t["p_boot"].to_numpy())
    return t


# ============================================================================ analysis 1: features
def _thresholds(spec, dates, U, A, names):
    """Per step: front threshold and eddy std (scene: per day from scene_feature_stats.csv; pooled: global)."""
    ft = spec["features"]
    src = ft.get("threshold_source", "scene")
    out = {}
    pooled = lambda c: np.concatenate([U[:, names.index(c)], A[:, :, names.index(c)].ravel()])
    if src == "scene":
        csv = (spec.get("scene_stats") or {}).get("csv") or os.path.join(spec["out_dir"], "scene_feature_stats.csv")
        if not os.path.exists(csv):
            raise FileNotFoundError(f"{csv} (run lagged_env --part scene, or set features.threshold_source: pooled)")
        st = pd.read_csv(csv)
        st["date"] = pd.to_datetime(st["date"])
        st = st.set_index("date")
        di = pd.DatetimeIndex(dates).normalize()
        for fname, f in (ft.get("fronts") or {}).items():
            c, p = f["channel"], float(f.get("percentile", 90))
            out[fname] = st[f"{c}_p{p:g}"].reindex(di).to_numpy(float)
        if ft.get("eddy"):
            out["eddy_sd"] = st[f"{ft['eddy']['channel']}_sd"].reindex(di).to_numpy(float)
    elif src == "pooled":
        n = len(dates)
        for fname, f in (ft.get("fronts") or {}).items():
            out[fname] = np.full(n, np.nanpercentile(pooled(f["channel"]), float(f.get("percentile", 90))))
        if ft.get("eddy"):
            out["eddy_sd"] = np.full(n, np.nanstd(pooled(ft["eddy"]["channel"])))
    else:
        raise ValueError("features.threshold_source must be scene or pooled")
    return out


def feature_masks(spec, V, names, thr):
    """Binary (float, NaN = missing) feature values for an (n, ..., C) value array; thresholds broadcast per step."""
    ft = spec["features"]
    out, fn = [], []
    ex = (slice(None),) + (None,) * (V.ndim - 2)

    def col(c):
        return V[..., names.index(c)]

    def b(mask, valid):
        return np.where(valid, mask.astype(float), np.nan)

    for fname, f in (ft.get("fronts") or {}).items():
        if f["channel"] not in names:
            continue
        g = col(f["channel"])
        t = thr[fname][ex]
        out.append(b(g >= t, np.isfinite(g) & np.isfinite(t)))
        fn.append(fname)
    if ft.get("eddy") and ft["eddy"]["channel"] in names:
        w = col(ft["eddy"]["channel"])
        t = -float(ft["eddy"].get("threshold", 0.2)) * thr["eddy_sd"][ex]
        valid = np.isfinite(w) & np.isfinite(t)
        core = w < t
        out.append(b(core, valid))
        fn.append("eddy_core")
        pc = ft["eddy"].get("polarity_channel")
        if pc and pc in names:
            r = col(pc)
            vr = valid & np.isfinite(r)
            out += [b(core & (r > 0), vr), b(core & (r < 0), vr)]
            fn += ["eddy_core_cyclonic", "eddy_core_anticyclonic"]
    for name, mk in (ft.get("masks") or {}).items():
        if mk["channel"] in names:
            x = col(mk["channel"])
            out.append(b(OPS[mk.get("op", ">")](x, float(mk["value"])), np.isfinite(x)))
            fn.append(name)
    if not out:
        raise ValueError("no feature channel of features.* is in the sampled inputs")
    return np.stack(out, -1), fn


def feature_channels(spec):
    ft = spec["features"]
    ch = []
    ch += [f["channel"] for f in (ft.get("fronts") or {}).values()]
    if ft.get("eddy"):
        ch.append(ft["eddy"]["channel"])
        if ft["eddy"].get("polarity_channel"):
            ch.append(ft["eddy"]["polarity_channel"])
    ch += [m["channel"] for m in (ft.get("masks") or {}).values()]
    return list(dict.fromkeys(ch))


def run_features(spec, design, label, Wb, out):
    U, names = design.values("inputs", feature_channels(spec), "used")
    A, _ = design.values("inputs", names, "avail")
    thr = _thresholds(spec, design.steps["date"], U, A, names)
    E, fn = feature_masks(spec, U, names, thr)
    X, _ = feature_masks(spec, A, names, thr)
    d = design.steps.assign(_aidx=design.aidx)
    t, _ = feature_table(d, E, X, fn, Wb)
    c = mode_vs_rest(d, E, X, fn, Wb)
    return t.assign(tool=label), c.assign(tool=label)


# ============================================================================ analysis 2: lags
def _wsplit(names):
    return [n.rsplit("__", 1) if "__" in n else (n, "") for n in names]


def _zc(v, aidx):
    """z-score per column, then centre within animal."""
    z = (v - np.nanmean(v, 0)) / np.where(np.nanstd(v, 0) > 0, np.nanstd(v, 0), 1.0)
    out = z.copy()
    for a in np.unique(aidx):
        m = aidx == a
        out[m] -= np.nanmean(z[m], 0)
    return out


def _rank(x):
    s = pd.Series(x)
    return s.rank().to_numpy() if s.notna().any() else x


def _ols_boot(aidx, nA, X, y, Wb):
    """Standardised OLS (intercept) from per-animal sufficient statistics; point estimate and bootstrap draws."""
    ok = np.all(np.isfinite(X), 1) & np.isfinite(y)
    X, y, a = X[ok], y[ok], aidx[ok]
    if len(y) < X.shape[1] + 3:
        return None, None
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    y = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
    Z = np.c_[np.ones(len(y)), X]
    p = Z.shape[1]
    G = np.zeros((nA, p, p))
    h = np.zeros((nA, p))
    np.add.at(G, a, Z[:, :, None] * Z[:, None, :])
    np.add.at(h, a, Z * y[:, None])
    point = np.linalg.lstsq(G.sum(0), h.sum(0), rcond=None)[0][1:]
    Gb = np.einsum("ba,aij->bij", Wb, G) + 1e-9 * np.eye(p)
    hb = Wb @ h
    draws = np.linalg.solve(Gb, hb[..., None])[..., 0][:, 1:]
    return point, draws


def run_lags(spec, design, label, Wb):
    lg = spec.get("lags") or {}
    lag_names = _channels(os.path.join(design.source("lagged")[0], design.uid + "_lagged_input_channels.json"))
    want = lg.get("channels")
    sel = [n for n in lag_names if not want or n.rsplit("__", 1)[0] in want]
    d = design.steps.assign(_aidx=design.aidx)
    nA = Wb.shape[1]
    out = {}
    U, sel = design.values("lagged", sel, "used")
    try:
        A, _ = design.values("lagged", sel, "avail")
        ua, _ = ua_continuous(d, U, A, sel, ["mode"], Wb)
        cw = _wsplit(ua["channel"])
        out["lag_use_availability"] = ua.assign(variable=[c for c, _ in cw], window=[w for _, w in cw], tool=label)
    except FileNotFoundError as e:
        print(f"habitat_tests lags: no available-point lagged values ({e}); use vs availability skipped")
    V, _ = design.values("lagged", sel, "decision")
    Z = _zc(V, design.aidx)
    rows = []
    for k in sorted(d["mode"].unique()):
        ink = (d["mode"] == k).to_numpy()
        S1, N1 = asum(design.aidx[ink], nA, Z[ink])
        S0, N0 = asum(design.aidx[~ink], nA, Z[~ink])
        b = _ratio(Wb, S1, N1) - _ratio(Wb, S0, N0)
        for j, n in enumerate(sel):
            lo, hi = _ci(b[:, j])
            v, w = _wsplit([n])[0]
            rows.append(dict(tool=label, mode=int(k), variable=v, window=w,
                             diff_z=S1[:, j].sum() / max(N1[:, j].sum(), 1) - S0[:, j].sum() / max(N0[:, j].sum(), 1),
                             ci_low=lo, ci_high=hi, p_boot=p_sign(b[:, j]), n_mode=int(N1[:, j].sum())))
    t = pd.DataFrame(rows)
    t["q_bh"] = bh(t["p_boot"].to_numpy())
    out["lag_mode_association"] = t
    rew = d["reward_likelihood"].to_numpy(float) if "reward_likelihood" in d else np.full(len(d), np.nan)
    if np.isfinite(rew).sum() < 10:
        print("habitat_tests lags: no reward_likelihood in the series; reward association skipped")
        return out
    base = lg.get("base", "d0")
    rows, jrows = [], []
    for k in sorted(d["mode"].unique()):
        ink = (d["mode"] == k).to_numpy()
        a = design.aidx[ink]
        y = rew[ink]
        for j, n in enumerate(sel):
            pt, dr = _ols_boot(a, nA, _rank(V[ink, j])[:, None], _rank(y), Wb)
            if pt is None:
                continue
            v, w = _wsplit([n])[0]
            lo, hi = _ci(dr[:, 0])
            rows.append(dict(tool=label, mode=int(k), variable=v, window=w, rank_corr=pt[0], ci_low=lo, ci_high=hi,
                             p_boot=p_sign(dr[:, 0]), n=int(np.sum(np.isfinite(V[ink, j]) & np.isfinite(y)))))
            bn = f"{v}__{base}"
            if w == base or bn not in sel:
                continue
            jb = sel.index(bn)
            pt, dr = _ols_boot(a, nA, np.c_[V[ink, jb], V[ink, j]], y, Wb)
            if pt is None:
                continue
            lb, hb = _ci(dr[:, 0])
            ll, hl = _ci(dr[:, 1])
            jrows.append(dict(tool=label, mode=int(k), variable=v, window=w, base=base, coef_base=pt[0],
                              base_ci_low=lb, base_ci_high=hb, coef_lag=pt[1], lag_ci_low=ll, lag_ci_high=hl,
                              p_lag=p_sign(dr[:, 1])))
    if rows:
        t = pd.DataFrame(rows)
        t["q_bh"] = bh(t["p_boot"].to_numpy())
        out["lag_reward_association"] = t
    if jrows:
        t = pd.DataFrame(jrows)
        t["q_bh_lag"] = bh(t["p_lag"].to_numpy())
        out["lag_reward_joint"] = t
    return out


# ============================================================================ analysis 3: seasons
def month_groups(groups):
    m2g = {}
    for g, months in groups.items():
        for m in months:
            if int(m) in m2g:
                raise ValueError(f"month {m} in two groups ({m2g[int(m)]}, {g})")
            m2g[int(m)] = str(g)
    return m2g


def run_seasons(spec, design, label, Wb):
    se = spec["seasons"]
    m2g = month_groups(se["groups"])
    U, names = design.values("inputs", se.get("channels"), "used")
    A, _ = design.values("inputs", names, "avail")
    d = design.steps.assign(_aidx=design.aidx, group=design.steps["month"].map(m2g))
    keep = d["group"].notna().to_numpy()
    d, U, A = d[keep].reset_index(drop=True), U[keep], A[keep]
    ua, draws = ua_continuous(d, U, A, names, ["mode", "group"], Wb)
    out = dict(season_use_availability=ua.assign(tool=label))
    ga, gb = (se.get("contrast") or list(se["groups"])[:2])[:2]
    rows = []
    for k in sorted(d["mode"].unique()):
        if (k, ga) not in draws or (k, gb) not in draws:
            continue
        (da, ua_), (db, ub) = draws[(k, ga)], draws[(k, gb)]
        ta = ua[(ua["mode"] == k) & (ua["group"] == ga)].set_index("channel")
        tb = ua[(ua["mode"] == k) & (ua["group"] == gb)].set_index("channel")
        for j, c in enumerate(names):
            bs, bu = da[:, j] - db[:, j], ua_[:, j] - ub[:, j]
            (l1, h1), (l2, h2) = _ci(bs), _ci(bu)
            rows.append(dict(tool=label, mode=int(k), channel=c, group_a=ga, group_b=gb,
                             selection_diff=ta.loc[c, "diff_sd_units"] - tb.loc[c, "diff_sd_units"],
                             sel_ci_low=l1, sel_ci_high=h1, p_selection=p_sign(bs),
                             used_mean_a=ta.loc[c, "used_mean"], used_mean_b=tb.loc[c, "used_mean"],
                             used_diff=ta.loc[c, "used_mean"] - tb.loc[c, "used_mean"], used_ci_low=l2,
                             used_ci_high=h2, p_used=p_sign(bu)))
    t = pd.DataFrame(rows)
    if len(t):
        t["q_selection"] = bh(t["p_selection"].to_numpy())
        t["q_used"] = bh(t["p_used"].to_numpy())
    out["season_contrast"] = t
    nA = Wb.shape[1]
    occ = []
    modes = sorted(d["mode"].unique())
    bo = {}
    for g in (ga, gb):
        gm = (d["group"] == g).to_numpy()
        _, Ng = asum(d["_aidx"].to_numpy()[gm], nA, np.ones((gm.sum(), 1)))
        for k in modes:
            km = gm & (d["mode"] == k).to_numpy()
            _, Nk = asum(d["_aidx"].to_numpy()[km], nA, np.ones((km.sum(), 1)))
            b = _ratio(Wb, Nk, Ng)[:, 0]
            bo[(g, k)] = b
            lo, hi = _ci(b)
            occ.append(dict(tool=label, group=g, mode=int(k), share=Nk.sum() / max(Ng.sum(), 1), ci_low=lo, ci_high=hi,
                            n_steps=int(Nk.sum())))
    for k in modes:
        b = bo[(ga, k)] - bo[(gb, k)]
        lo, hi = _ci(b)
        sa = [r["share"] for r in occ if r["group"] == ga and r["mode"] == k][0]
        sb = [r["share"] for r in occ if r["group"] == gb and r["mode"] == k][0]
        occ.append(dict(tool=label, group=f"{ga}-{gb}", mode=int(k), share=sa - sb, ci_low=lo, ci_high=hi,
                        p_boot=p_sign(b)))
    out["season_occupancy"] = pd.DataFrame(occ)
    return out


# ============================================================================ state-level lagged profiles
def run_lag_profiles(spec, series, n_states=None):
    lp = spec.get("lag_profiles") or {}
    sconf = read_yaml(abs_cfg(spec["sample_inputs_yaml"]))
    uid = sconf["df_run_uid"]
    lag_dir = (spec.get("lagged") or {}).get("out_dir") or os.path.join(spec["out_dir"], "lagged_inputs")
    names = _channels(os.path.join(lag_dir, uid + "_lagged_input_channels.json"))
    s = series[np.isfinite(series["env"])]
    st = _pickles(lag_dir, uid, sorted(s["uid"].unique()), "lagged_input")
    V = gather(st, list(s["uid"]), s["segment"].to_numpy(int), s["seg_pos"].to_numpy(int), len(names))
    df = pd.DataFrame(V, columns=names)
    df["state"] = s["env"].to_numpy(int)
    g = df.groupby("state")
    prof = g[names].mean()
    prof.insert(0, "n", g.size())
    prof = prof[prof["n"] >= int(lp.get("min_fixes", 5))]
    prof = prof.reset_index().rename(columns={"state": "cluster"})
    out = lp.get("csv") or os.path.join(spec["out_dir"], "lagged_state_profiles.csv")
    atomic_csv(prof, out)
    print(f"habitat_tests lag_profiles: {len(prof)} states with >= {lp.get('min_fixes', 5)} fixes -> {out}")
    return prof


# ============================================================================ plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_forest(t, x, lo, hi, row, panel, out_png, xlabel):
    if not len(t):
        return
    plt = _plt()
    panels = list(dict.fromkeys(t[panel]))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 0.32 * t[row].nunique() * 1.0
                                                      * max(1, t["_series"].nunique()) + 1.6),
                             sharey=True, squeeze=False)
    for ax, p in zip(axes[0], panels):
        q = t[t[panel] == p]
        labels = list(dict.fromkeys(q[row].astype(str)))
        ser = list(dict.fromkeys(q["_series"]))
        for si, s in enumerate(ser):
            r = q[q["_series"] == s]
            y = np.array([labels.index(str(v)) for v in r[row]]) + (si - (len(ser) - 1) / 2) * 0.18
            ax.errorbar(r[x], y, xerr=[np.abs(r[x] - r[lo]), np.abs(r[hi] - r[x])], fmt="o", ms=3.5, lw=1,
                        label=str(s))
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(str(p), fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
    axes[0][0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_lag_profiles(t, out_png, value="diff_z"):
    if t is None or not len(t):
        return
    plt = _plt()
    tools, variables = list(dict.fromkeys(t["tool"])), list(dict.fromkeys(t["variable"]))
    wins = list(dict.fromkeys(t["window"]))
    fig, axes = plt.subplots(len(tools), len(variables), figsize=(3.2 * len(variables), 2.6 * len(tools)),
                             squeeze=False, sharex=True)
    for i, tl in enumerate(tools):
        for j, v in enumerate(variables):
            ax = axes[i][j]
            q = t[(t["tool"] == tl) & (t["variable"] == v)]
            for k, r in q.groupby("mode"):
                r = r.set_index("window").reindex(wins)
                ax.errorbar(range(len(wins)), r[value], yerr=[np.abs(r[value] - r["ci_low"]),
                                                              np.abs(r["ci_high"] - r[value])],
                            marker="o", ms=3, lw=1, capsize=2, label=f"mode {k}")
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_xticks(range(len(wins)), wins, rotation=45, fontsize=7)
            ax.set_title(f"{tl}: {v}", fontsize=8)
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ============================================================================ driver
def run(spec, parts=PARTS):
    spec = spec.get("habitat_tests", spec)
    out = spec["out_dir"]
    os.makedirs(out, exist_ok=True)
    n_boot, seed = int(spec.get("n_boot", 1000)), int(spec.get("seed", 0))
    res = {p: {} for p in parts if p != "lag_profiles"}
    summary = dict(parts=list(parts), tools=[], mode_at=spec.get("mode_at", "step"))
    if "lag_profiles" in parts:
        t0 = next((t for t in spec["tools"] if os.path.isdir(t["series_dir"])), None)
        if t0 is None:
            raise FileNotFoundError("lag_profiles: no tool series_dir exists")
        run_lag_profiles(spec, load_series(t0["series_dir"]).reset_index(drop=True))
        parts = tuple(p for p in parts if p != "lag_profiles")
        if not parts:
            return {}
    for tool in spec["tools"]:
        label = tool["label"]
        if not os.path.isdir(tool["series_dir"]):
            print(f"habitat_tests: SKIP {label} (missing {tool['series_dir']})")
            continue
        series = load_series(tool["series_dir"])
        atomic_csv(mode_kinematics(series), os.path.join(out, f"mode_kinematics_{label}.csv"))
        design = Design(spec, series)
        Wb = boot_weights(len(design.animals), n_boot, seed)
        summary["tools"].append(dict(label=label, n_steps=int(len(design.steps)), n_animals=int(len(design.animals))))
        print(f"habitat_tests {label}: {len(design.steps)} steps, {len(design.animals)} animals, m = {design.m}")
        if "features" in parts and spec.get("features"):
            t, c = run_features(spec, design, label, Wb, out)
            res["features"].setdefault("feature_use", []).append(t)
            res["features"].setdefault("feature_mode_contrast", []).append(c)
        if "lags" in parts and spec.get("lagged"):
            for k, v in run_lags(spec, design, label, Wb).items():
                res["lags"].setdefault(k, []).append(v)
        if "seasons" in parts and spec.get("seasons"):
            for k, v in run_seasons(spec, design, label, Wb).items():
                res["seasons"].setdefault(k, []).append(v)
    tables = {}
    for p, dd in res.items():
        for name, lst in dd.items():
            t = pd.concat(lst, ignore_index=True)
            tables[name] = t
            atomic_csv(t, os.path.join(out, f"{name}.csv"))
    if "feature_use" in tables:
        t = tables["feature_use"]
        plot_forest(t.assign(_series=t["tool"], mode=t["mode"].astype(str)), "log_or", "ci_low", "ci_high", "mode",
                    "feature", os.path.join(out, "feature_use.png"), "log odds ratio (used vs available)")
    if "lag_mode_association" in tables:
        plot_lag_profiles(tables["lag_mode_association"], os.path.join(out, "lag_mode_association.png"))
    if "lag_use_availability" in tables:
        plot_lag_profiles(tables["lag_use_availability"], os.path.join(out, "lag_use_availability.png"),
                          "diff_sd_units")
    if "lag_reward_association" in tables:
        plot_lag_profiles(tables["lag_reward_association"], os.path.join(out, "lag_reward_association.png"),
                          "rank_corr")
    if "season_contrast" in tables and len(tables["season_contrast"]):
        t = tables["season_contrast"]
        plot_forest(t.assign(_series=t["tool"], mode=t["mode"].astype(str), lo=t["used_ci_low"],
                             hi=t["used_ci_high"]), "used_diff", "lo", "hi", "mode", "channel",
                    os.path.join(out, "season_contrast_used.png"), "used mean, group a - group b")
        plot_forest(t.assign(_series=t["tool"], mode=t["mode"].astype(str), lo=t["sel_ci_low"],
                             hi=t["sel_ci_high"]), "selection_diff", "lo", "hi", "mode", "channel",
                    os.path.join(out, "season_contrast_selection.png"), "selection (sd units), a - b")
    name = "all" if set(parts) == set(PARTS) else "_".join(parts)
    atomic_json(summary, os.path.join(out, f"habitat_summary_{name}.json"))
    print(f"habitat_tests: {parts} -> {out}")
    return tables


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--part", action="append", choices=PARTS)
    a = ap.parse_args(argv)
    run(read_yaml(abs_cfg(a.yaml)), tuple(a.part or PARTS))


if __name__ == "__main__":
    main()
