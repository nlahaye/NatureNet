"""
Per-intention kinematic emissions (step length / speed + turning angle) for HIAVI, coupled HIAVI and SWIRL.

The movement models emit the discrete action a_t from the (abstract) state x_t. With kinematics enabled,
each intention z also emits the continuous movement of the same step:

    p(a_t, l_t, theta_t | x_t, z_t = k) = pi_k(a_t | x_t) * f_k(l_t) * vM(theta_t; mu_k, kappa_k)

    l_t      step feature of the move fix p -> p+1 (step_km | speed_kmh | diffusive = km / sqrt(h))
    theta_t  turning angle at fix p: bearing(p -> p+1) - bearing(p-1 -> p), wrapped to (-pi, pi]
    f_k      zero-inflated gamma (default) or log-normal; zero = step below zero_km (stationary)
    vM       von Mises (mu_k free, or fixed at 0 with fix_mu0)

Missing values (NaN: first fix of a segment, stationary neighbours for theta, dt < min_dt_h) contribute
nothing, so the emission reduces to the action-only model on those steps. M-steps are closed-form /
1-D Newton on the posterior weights, so EM stays monotone.

For model comparison the joint LL is not comparable with action-only LLs (it contains continuous densities).
`pooled()` fits the same kinematic model with ONE intention on the training data; the "comparable LL" is

    LL_cmp = LL_joint - LL_kin_pooled    (= action LL if the kinematics do not depend on the intention)

which is what the training scripts save as LL (LL_joint and LL_kin_null are saved too).

Exploit / explore labels come straight from the fitted parameters (`intention_table`, `kinematic_labels`):
slow + tortuous (low expected speed, low persistence A1(kappa) cos(mu)) = exploit (area-restricted search),
fast + directed = explore (transit); conflicting = ambiguous. Per-step P(exploit) = sum of the posterior
over exploit intentions.
"""
import os

import numpy as np
import pandas as pd
from scipy.special import gammaln, i0e, i1e, polygamma, psi

STEP_FEATURES = ("step_km", "speed_kmh", "diffusive")
TINY = 1e-300
LOG2PI = np.log(2 * np.pi)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

class KinFeatures:
    """step (N, T) >= 0 or NaN (0 = stationary), turn (N, T) in (-pi, pi] or NaN; indexable by sequence."""

    def __init__(self, step, turn, info=None):
        self.step = np.asarray(step, dtype=float)
        self.turn = np.asarray(turn, dtype=float)
        if self.step.shape != self.turn.shape or self.step.ndim != 2:
            raise ValueError(f"step {self.step.shape} / turn {self.turn.shape} must both be (N, T)")
        fin = np.isfinite(self.step)
        if np.any(self.step[fin] < 0):
            raise ValueError("negative step values")
        ft = np.isfinite(self.turn)
        if np.any(np.abs(self.turn[ft]) > np.pi + 1e-9):
            raise ValueError("turn angles must be in (-pi, pi]")
        self.info = dict(info or {})

    @property
    def shape(self):
        return self.step.shape

    def __getitem__(self, idx):
        return KinFeatures(self.step[idx], self.turn[idx], self.info)

    def coverage(self):
        return dict(step_finite=float(np.isfinite(self.step).mean()), turn_finite=float(np.isfinite(self.turn).mean()),
                    step_zero=float((self.step == 0).mean()))


def kin_conf(yml_conf):
    """`kinematics` YAML block with defaults (None if absent or disabled)."""
    k = yml_conf.get("kinematics")
    if not k or not k.get("enabled", True):
        return None
    k = dict(k)
    k.setdefault("step_feature", "speed_kmh")
    k.setdefault("step_dist", "gamma")
    k.setdefault("turn_dist", "vonmises")
    k.setdefault("use_step", True)
    k.setdefault("use_turn", True)
    k.setdefault("fix_mu0", False)
    k.setdefault("zero_km", 0.05)
    k.setdefault("min_dt_h", 0.5)
    k.setdefault("max_speed_kmh", None)
    k.setdefault("weight", 1.0)
    k.setdefault("kappa_max", 200.0)
    k.setdefault("min_weight", 2.0)
    k.setdefault("label_speed_threshold", None)
    k.setdefault("label_persistence_threshold", None)
    k.setdefault("label_margin", 0.5)
    if k["step_feature"] not in STEP_FEATURES:
        raise ValueError(f"kinematics.step_feature must be one of {STEP_FEATURES}")
    return k


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


def wrap_angle(x):
    """Wrap to (-pi, pi]."""
    y = np.mod(np.asarray(x) + np.pi, 2 * np.pi) - np.pi
    return np.where(y == -np.pi, np.pi, y)


def track_kinematics(lat, lon, t_hours, cfg):
    """
    Per-fix kinematics of ONE segment (length n): step / dt / turn for the move fix p -> p+1 (NaN at p = n-1).
    Returns dict(step_km, dt_h, feature, turn) arrays of length n.
    """
    lat, lon, t = (np.asarray(v, dtype=float) for v in (lat, lon, t_hours))
    n = len(lat)
    step = np.full(n, np.nan)
    dt = np.full(n, np.nan)
    brg = np.full(n, np.nan)
    if n >= 2:
        step[:-1] = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dt[:-1] = t[1:] - t[:-1]
        brg[:-1] = _bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
    zero = step < float(cfg["zero_km"])
    bad_dt = ~(dt >= float(cfg["min_dt_h"]))
    moving = np.isfinite(step) & ~zero
    brg = np.where(moving, brg, np.nan)
    turn = np.full(n, np.nan)
    if n >= 2:
        turn[1:] = wrap_angle(brg[1:] - brg[:-1])     # NaN propagates when either step is stationary / missing
    sf = cfg["step_feature"]
    with np.errstate(divide="ignore", invalid="ignore"):
        if sf == "step_km":
            feat = step.copy()
        elif sf == "speed_kmh":
            feat = step / dt
        else:
            feat = step / np.sqrt(dt)
    feat = np.where(zero, 0.0, feat)
    feat[bad_dt] = np.nan
    if cfg.get("max_speed_kmh"):
        with np.errstate(invalid="ignore"):
            feat[(step / dt) > float(cfg["max_speed_kmh"])] = np.nan
    turn[bad_dt] = np.nan
    if n >= 2:
        turn[1:][bad_dt[:-1]] = np.nan                 # incoming step too short in time as well
    return dict(step_km=step, dt_h=dt, feature=feat, turn=turn)


def step_kinematics(yml_conf, data, cfg, dfs=None, offset=None, shape=None):
    """
    KinFeatures aligned with the model steps: sequence i, step t <-> fix p = meta[i].start + offset + t of
    segment meta[i].seg of <df_dir>/<df_run_uid>_dfs.pkl (the same mapping as run_hiavi_multiagent.step_metadata).
    offset / shape default to the HIAVI arrays (data["hiavi_offset"], data["hiavi_s"].shape) or the SWIRL x.
    Turn uses fix p-1 when it exists in the segment (also before the sequence start).
    """
    from naturenet.models.irl.swirl_track_series import _df_key

    if dfs is None:
        path = (yml_conf.get("multiagent") or {}).get("dfs_pkl") or (yml_conf.get("kinematics") or {}).get("dfs_pkl") \
            or os.path.join(yml_conf["df_dir"], yml_conf["df_run_uid"] + "_dfs.pkl")
        dfs = pd.read_pickle(path)
    if shape is None:
        shape = np.asarray(data["hiavi_s"] if "hiavi_s" in data else data["x"]).shape
    if offset is None:
        offset = int(data.get("hiavi_offset", 1))
    N, T = shape
    if len(data["meta"]) != N:
        raise ValueError(f"{len(data['meta'])} sequence ids for {N} sequences")
    step = np.full((N, T), np.nan)
    turn = np.full((N, T), np.nan)
    cache = {}
    for i, m in enumerate(data["meta"]):
        dkey = _df_key(m["key"], yml_conf["df_run_uid"])
        ck = (dkey, int(m["seg"]))
        if ck not in cache:
            if dkey not in dfs:
                raise KeyError(f"sequence {i}: no movement DataFrame for {dkey} in dfs.pkl")
            df = dfs[dkey][m["seg"]]
            ts = pd.to_datetime(pd.Series(df["timestamp"].to_numpy()), utc=True, errors="coerce", format="mixed")
            th = (ts.dt.tz_convert(None) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy() / 3600.0
            cache[ck] = track_kinematics(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float), th, cfg)
        kin = cache[ck]
        pos = int(m["start"]) + int(offset) + np.arange(T)
        if pos.max() >= len(kin["feature"]):
            raise ValueError(f"{dkey} segment {m['seg']}: step position {pos.max()} >= {len(kin['feature'])} fixes")
        step[i] = kin["feature"][pos]
        turn[i] = kin["turn"][pos]
    out = KinFeatures(step, turn, info=dict(step_feature=cfg["step_feature"], offset=int(offset)))
    cov = out.coverage()
    if cov["step_finite"] < 0.5:
        print(f"WARNING: only {100 * cov['step_finite']:.0f}% of steps have a usable step feature "
              f"(min_dt_h={cfg['min_dt_h']}); check timestamps / min_dt_h")
    return out


def cached_kinematics(yml_conf, data, cfg, offset=None, shape=None):
    key = ("kin", cfg["step_feature"], float(cfg["zero_km"]), float(cfg["min_dt_h"]), offset,
           None if shape is None else tuple(shape))
    cache = data.setdefault("_cache", {})
    if key not in cache:
        cache[key] = step_kinematics(yml_conf, data, cfg, offset=offset, shape=shape)
    return cache[key]


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def a1(kappa):
    """Mean resultant length of a von Mises: I1(k) / I0(k)."""
    kappa = np.asarray(kappa, dtype=float)
    return np.where(kappa > 1e-8, i1e(kappa) / i0e(kappa), kappa / 2)


def a1inv(R, kappa_max=200.0):
    """kappa with A1(kappa) = R (Best & Fisher start + 3 Newton steps), clipped to [0, kappa_max]."""
    R = float(np.clip(R, 0.0, 1 - 1e-9))
    if R < 0.53:
        k = 2 * R + R ** 3 + 5 * R ** 5 / 6
    elif R < 0.85:
        k = -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        k = 1 / (R ** 3 - 4 * R ** 2 + 3 * R)
    for _ in range(3):
        if k <= 1e-8:
            break
        A = float(a1(k))
        dA = 1 - A / k - A ** 2
        if dA <= 1e-12:
            break
        k = max(k - (A - R) / dA, 1e-8)
    return float(np.clip(k, 0.0, kappa_max))


def vm_logpdf(theta, mu, kappa):
    kappa = np.asarray(kappa, dtype=float)
    return kappa * np.cos(theta - mu) - LOG2PI - (np.log(i0e(kappa)) + kappa)


def gamma_logpdf(x, shape, scale):
    return (shape - 1) * np.log(x) - x / scale - shape * np.log(scale) - gammaln(shape)


def lognorm_logpdf(x, mu, sd):
    lx = np.log(x)
    return -lx - 0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * ((lx - mu) / sd) ** 2


def gamma_wmle(x, w, shape_bounds=(0.05, 500.0)):
    """Weighted gamma MLE (Minka's generalized Newton on the shape)."""
    W = w.sum()
    xbar = (w * x).sum() / W
    lbar = (w * np.log(x)).sum() / W
    s = max(np.log(xbar) - lbar, 1e-10)
    a = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(20):
        g = np.log(a) - psi(a) - s
        h = 1 / a - polygamma(1, a)
        a_new = 1 / (1 / a + g / (a ** 2 * h))
        if not np.isfinite(a_new) or a_new <= 0:
            break
        if abs(a_new - a) < 1e-8 * a:
            a = a_new
            break
        a = a_new
    a = float(np.clip(a, *shape_bounds))
    return a, float(xbar / a)


# ---------------------------------------------------------------------------
# Emission model
# ---------------------------------------------------------------------------

class KinematicEmissions:
    """
    K per-intention kinematic emission factors. loglik(kin) -> (N, T, K) (already multiplied by `weight`),
    m_step(gamma, kin) closed-form. Parameters: p0 (P(stationary)), step (gamma shape/scale or lognormal
    mu/sd), turn (mu, kappa).
    """

    def __init__(self, K, step_dist="gamma", turn_dist="vonmises", use_step=True, use_turn=True, fix_mu0=False,
                 weight=1.0, kappa_max=200.0, min_weight=2.0, p0_pseudocount=0.5, **_):
        if step_dist not in ("gamma", "lognormal"):
            raise ValueError("step_dist must be gamma | lognormal")
        if turn_dist != "vonmises":
            raise ValueError("turn_dist must be vonmises")
        self.K = int(K)
        self.step_dist, self.turn_dist = step_dist, turn_dist
        self.use_step, self.use_turn, self.fix_mu0 = bool(use_step), bool(use_turn), bool(fix_mu0)
        self.weight, self.kappa_max, self.min_weight = float(weight), float(kappa_max), float(min_weight)
        self.p0_pseudocount = float(p0_pseudocount)
        self.zero_inflated = False
        self.p0 = np.zeros(self.K)
        self.s1 = np.ones(self.K)      # gamma shape | lognormal mu
        self.s2 = np.ones(self.K)      # gamma scale | lognormal sd
        self.mu = np.zeros(self.K)
        self.kappa = np.zeros(self.K)

    @classmethod
    def from_conf(cls, K, cfg):
        return cls(K, **{k: v for k, v in cfg.items() if k in ("step_dist", "turn_dist", "use_step", "use_turn",
                                                                 "fix_mu0", "weight", "kappa_max", "min_weight")})

    # -- fitting --
    def _fit_one(self, k, x, wx, z, wz, th, wt):
        """Weighted updates of intention k; arrays are the valid entries only."""
        if self.use_step:
            Wz = wz.sum()
            if self.zero_inflated and Wz >= self.min_weight:
                # Beta(0.01, p0_pseudocount) prior keeps p0 off 0 / 1 for intentions with few stationary steps
                self.p0[k] = (np.sum(wz * z) + 0.01) / (Wz + 0.01 + self.p0_pseudocount)
            if wx.sum() >= self.min_weight:
                if self.step_dist == "gamma":
                    self.s1[k], self.s2[k] = gamma_wmle(x, wx)
                else:
                    lx = np.log(x)
                    m = (wx * lx).sum() / wx.sum()
                    self.s1[k] = m
                    self.s2[k] = max(np.sqrt((wx * (lx - m) ** 2).sum() / wx.sum()), 1e-3)
        if self.use_turn and wt.sum() >= self.min_weight:
            C = (wt * np.cos(th)).sum() / wt.sum()
            S = (wt * np.sin(th)).sum() / wt.sum()
            if self.fix_mu0:
                self.mu[k], R = 0.0, max(C, 0.0)
            else:
                self.mu[k], R = float(np.arctan2(S, C)), float(np.hypot(C, S))
            self.kappa[k] = a1inv(R, self.kappa_max)

    def _valid(self, kin):
        st, tu = kin.step, kin.turn
        has_s = np.isfinite(st)
        return has_s, has_s & (st > 0), np.isfinite(tu)

    def init_quantiles(self, kin, rng=None):
        """Intention k <- k-th quantile bin of the step feature (k = 0 slowest); rng permutes the bins."""
        has_s, pos, has_t = self._valid(kin)
        self.zero_inflated = bool(np.any(kin.step[has_s] == 0))
        x = kin.step[pos]
        if x.size < self.K:
            raise ValueError(f"only {x.size} positive step values for K={self.K} intentions")
        edges = np.quantile(x, np.linspace(0, 1, self.K + 1))
        bins = np.clip(np.searchsorted(edges, kin.step, side="right") - 1, 0, self.K - 1)
        order = np.arange(self.K) if rng is None else rng.permutation(self.K)
        p0 = float(np.mean(kin.step[has_s] == 0)) if self.zero_inflated else 0.0
        for j in range(self.K):
            k = order[j]
            sel = pos & (bins == j)
            selt = has_t & (bins == j)
            self.p0[k] = p0
            self._fit_one(k, kin.step[sel], np.ones(sel.sum()), np.zeros(0), np.zeros(0),
                          kin.turn[selt], np.ones(selt.sum()))
        return self

    def m_step(self, gamma, kin):
        gamma = np.asarray(gamma, dtype=float)
        if gamma.shape != kin.shape + (self.K,):
            raise ValueError(f"gamma {gamma.shape} does not match kinematics {kin.shape} and K={self.K}")
        has_s, pos, has_t = self._valid(kin)
        x, th = kin.step[pos], kin.turn[has_t]
        z = (kin.step[has_s] == 0).astype(float)
        for k in range(self.K):
            g = gamma[..., k]
            self._fit_one(k, x, g[pos], z, g[has_s], th, g[has_t])
        return self

    # -- evaluation --
    def loglik(self, kin, weighted=True):
        """(N, T, K) log f_k(step) + log vM(turn); missing -> 0."""
        has_s, pos, has_t = self._valid(kin)
        out = np.zeros(kin.shape + (self.K,))
        if self.use_step:
            x = np.where(pos, kin.step, 1.0)[..., None]
            if self.step_dist == "gamma":
                lp = gamma_logpdf(x, self.s1, self.s2)
            else:
                lp = lognorm_logpdf(x, self.s1, self.s2)
            if self.zero_inflated:
                lp = lp + np.log(np.maximum(1 - self.p0, TINY))
                lz = np.log(np.maximum(self.p0, TINY))
                out += np.where(pos[..., None], lp, np.where((has_s & ~pos)[..., None], lz, 0.0))
            else:
                out += np.where(pos[..., None], lp, 0.0)
        if self.use_turn:
            th = np.where(has_t, kin.turn, 0.0)[..., None]
            out += np.where(has_t[..., None], vm_logpdf(th, self.mu, self.kappa), 0.0)
        return out * (self.weight if weighted else 1.0)

    def total_loglik(self, kin, gamma=None):
        """Sum of the K=1 (pooled) LL, or the posterior-weighted LL under gamma."""
        ll = self.loglik(kin, weighted=False)
        return float(ll[..., 0].sum()) if gamma is None else float((ll * gamma).sum())

    # -- summaries --
    def mean_step(self):
        m = self.s1 * self.s2 if self.step_dist == "gamma" else np.exp(self.s1 + self.s2 ** 2 / 2)
        return (1 - self.p0) * m

    def median_step(self):
        if self.step_dist == "lognormal":
            return np.exp(self.s1)
        from scipy.stats import gamma as _g
        return _g.ppf(0.5, self.s1, scale=self.s2)

    def persistence(self):
        return a1(self.kappa) * np.cos(self.mu)

    def params(self, prefix="kin_"):
        return {prefix + "p0": self.p0, prefix + "s1": self.s1, prefix + "s2": self.s2, prefix + "mu": self.mu,
                prefix + "kappa": self.kappa, prefix + "zero_inflated": self.zero_inflated,
                prefix + "step_dist": self.step_dist, prefix + "use_step": self.use_step,
                prefix + "use_turn": self.use_turn, prefix + "weight": self.weight}

    @classmethod
    def from_params(cls, p, prefix="kin_", **kw):
        p0 = np.asarray(p[prefix + "p0"], dtype=float)
        m = cls(len(p0), step_dist=str(p[prefix + "step_dist"]), use_step=bool(p[prefix + "use_step"]),
                use_turn=bool(p[prefix + "use_turn"]), weight=float(p[prefix + "weight"]), **kw)
        m.p0, m.s1, m.s2 = p0, np.asarray(p[prefix + "s1"], float), np.asarray(p[prefix + "s2"], float)
        m.mu, m.kappa = np.asarray(p[prefix + "mu"], float), np.asarray(p[prefix + "kappa"], float)
        m.zero_inflated = bool(p[prefix + "zero_inflated"])
        return m

    def permute(self, perm):
        perm = np.asarray(perm)
        for k in ("p0", "s1", "s2", "mu", "kappa"):
            setattr(self, k, getattr(self, k)[perm])
        return self

    def copy(self):
        return KinematicEmissions.from_params(self.params(), kappa_max=self.kappa_max, min_weight=self.min_weight,
                                              fix_mu0=self.fix_mu0)

    @staticmethod
    def has_params(p, prefix="kin_"):
        return (prefix + "p0") in p


def pooled(kin, cfg):
    """One-intention kinematic model on kin (the null for the comparable LL)."""
    m = KinematicEmissions.from_conf(1, cfg)
    m.init_quantiles(kin)
    return m.m_step(np.ones(kin.shape + (1,)), kin)


def comparable_ll(ll_joint_total, kin_null, kin, n_steps):
    """(LL_joint - LL_kin_pooled) / n_steps, plus the two parts (per step)."""
    ln = kin_null.total_loglik(kin) * kin_null.weight
    return (ll_joint_total - ln) / n_steps, ll_joint_total / n_steps, ln / n_steps


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def intention_table(km, step_feature="speed_kmh"):
    """One row per intention: p_stationary, mean / median step feature, turn mu / kappa / MRL / persistence."""
    return pd.DataFrame(dict(state=np.arange(km.K), p_stationary=km.p0, step_mean=km.mean_step(),
                             step_median=km.median_step(), step_s1=km.s1, step_s2=km.s2,
                             turn_mu_deg=np.degrees(km.mu), turn_kappa=km.kappa, mrl=a1(km.kappa),
                             persistence=km.persistence(), step_feature=step_feature))


def kinematic_labels(km, step_feature="speed_kmh", speed_threshold=None, persistence_threshold=None, margin=0.5):
    """
    exploit / explore / ambiguous per intention from the fitted kinematics. Uses the same rule as
    exploit_explore_init.classify_exploit_explore with speed_median := fitted median step feature and
    mrl := persistence (A1(kappa) cos mu: tortuous / reversing < directed).
    """
    from naturenet.models.irl.exploit_explore_init import classify_exploit_explore

    tab = intention_table(km, step_feature)
    m = tab.assign(speed_median=tab["step_median"], mrl=tab["persistence"])
    st = speed_threshold if step_feature == "speed_kmh" else None
    if speed_threshold is not None and st is None:
        print("NOTE: label_speed_threshold is in km/h; ignored for step_feature", step_feature)
    m = classify_exploit_explore(m, st, persistence_threshold if st is not None else None, margin)
    return m.drop(columns=["speed_median"]).rename(columns={"class": "label"})


def labels_from_conf(km, cfg):
    return kinematic_labels(km, cfg["step_feature"], cfg.get("label_speed_threshold"),
                            cfg.get("label_persistence_threshold"), float(cfg.get("label_margin", 0.5)))


def p_exploit(gamma, labels):
    """Per-step P(exploit), P(explore) from posteriors (N, T, K) and the label table."""
    lab = np.asarray(labels["label"])
    g = np.asarray(gamma)
    return g[..., lab == "exploit"].sum(-1), g[..., lab == "explore"].sum(-1)
