"""
Link per-mode IRL rewards to the physical variables behind the SIT-FUSE environmental states.

Inputs
  * per-mode state reward R_k(s) from the final SWIRL / NN-SWIRL / HIAVI run (swirl_track_series.RewardModel);
  * per-state physical profile P(s, v): cluster medians written by SIT-FUSE xai/cluster_interpret.py
    (cluster_profiles_wide.csv), mapped to NatureNet state ids with the <output_uid>_class_mapper.pkl of
    env_data_format.compress_clusters (keys = rint(cluster value * cluster_scale));
  * decoded latent paths and fixes (swirl_track_series.build_track_series).

Analyses (written to <reward_env_projection.out_dir>)
  1. reward_projection_coefficients.csv   ridge regression of the z-scored reward of each mode on z-scored state
     profiles (state-frequency weights optional): standardised coefficients with bootstrap CIs over states,
     cross-validated R^2, per-variable Spearman rho.
  2. reward_partial_dependence.csv        binned (quantile) mean reward per variable and mode, weighted by state use.
  3. reward_coefficient_stability.csv     Spearman agreement of coefficients across K-fold reward files (tags).
  4. use_availability.csv                 per mode: mean variable value of the next state actually entered minus its
     expectation under the available next states (T averaged over actions), with animal-bootstrap CIs.
  5. selection_ratios.csv                 per mode x state: used / available (Manly-type selection ratio).

Rewards are identified only up to per-mode shift and scale (and IAVI only up to a per-state shift handled in
RewardModel), so each mode's reward is z-scored across states before regression; coefficients are comparable in
sign and relative size, not in absolute units.

Usage
  python -m naturenet.models.irl.reward_env_projection -y config/swirl/full_plan/whales_swirl_kfold.yaml \\
      -p config/swirl/full_plan/whales_reward_env_projection.yaml --n_hidden <K*>
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from naturenet.models.irl.swirl_track_series import build_track_series, load_reward_model

DEFAULTS = dict(model="linear", order="s1", tag="", fold_tags=[], cluster_scale=1000, profile_index="cluster",
                variables=None, weight_by_use=True, alphas=[0.01, 0.1, 1.0, 10.0, 100.0], n_boot=500, n_bins=5,
                n_boot_animals=500, availability="uniform_actions", min_state_visits=1, seed=0,
                splits=["train", "test"])


# ============================== MAPPING ===============================

def state_profiles(profiles, n_states, mapper=None, scale=1000, index="cluster", variables=None):
    """(S, V) profile matrix (NaN rows for states without a profile) and the variable names.

    index = "cluster": profiles.cluster holds SIT-FUSE label values, mapped through the class mapper;
    index = "state":   profiles.cluster already holds NatureNet state ids.
    """
    df = profiles.copy()
    skip = {"cluster", "cluster_id", "n", "frac", "state"}
    variables = list(variables) if variables else [c for c in df.columns
                                                   if c not in skip and np.issubdtype(df[c].dtype, np.number)]
    P = np.full((n_states, len(variables)), np.nan)
    if index == "state":
        states = df["cluster"].astype(int).to_numpy()
    else:
        if mapper is None:
            raise ValueError("profile_index 'cluster' needs the class mapper")
        keys = np.rint(df["cluster"].to_numpy(float) * scale)
        states = np.array([mapper.get(float(k), -1) for k in keys], dtype=int)
    ok = (states >= 0) & (states < n_states)
    if not ok.any():
        raise ValueError("no profile row maps to a valid state; check cluster_scale / profile_index")
    if len(np.unique(states[ok])) != ok.sum():
        raise ValueError("several profile rows map to the same state")
    P[states[ok]] = df.loc[ok, variables].to_numpy(float)
    return P, variables


def zscore(a, w=None, axis=0):
    a = np.asarray(a, float)
    if w is None:
        mu, sd = np.nanmean(a, axis, keepdims=True), np.nanstd(a, axis, keepdims=True)
    else:
        w = np.asarray(w, float)
        shape = [1] * a.ndim
        shape[axis] = -1
        ww = w.reshape(shape) * np.isfinite(a)
        mu = np.nansum(a * ww, axis, keepdims=True) / ww.sum(axis, keepdims=True)
        sd = np.sqrt(np.nansum(ww * (a - mu) ** 2, axis, keepdims=True) / ww.sum(axis, keepdims=True))
    return (a - mu) / np.where(sd > 0, sd, 1.0)


# ============================== REWARD PROJECTION ===============================

def ridge_fit(Z, y, w, alpha):
    """Weighted ridge with intercept on standardised Z. Returns coefficients (V,)."""
    sw = np.sqrt(w / w.sum())
    Zc = Z - np.average(Z, 0, w)
    yc = y - np.average(y, weights=w)
    A = (Zc * sw[:, None]).T @ (Zc * sw[:, None]) + alpha * np.eye(Z.shape[1])
    return np.linalg.solve(A, (Zc * sw[:, None]).T @ (yc * sw))


def cv_r2(Z, y, w, alpha, n_splits=5, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % n_splits
    pred = np.empty_like(y)
    for f in range(n_splits):
        tr, te = fold != f, fold == f
        b = ridge_fit(Z[tr], y[tr], w[tr], alpha)
        pred[te] = np.average(y[tr], weights=w[tr]) + (Z[te] - np.average(Z[tr], 0, w[tr])) @ b
    ss_res = np.sum(w * (y - pred) ** 2)
    ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def project_rewards(R_state, P, names, use_counts=None, alphas=(0.1, 1.0, 10.0), n_boot=500, seed=0,
                    min_visits=1):
    """Per-mode ridge of z-scored reward on z-scored profiles over states with a profile (and >= min_visits)."""
    from scipy.stats import spearmanr
    K, S = R_state.shape
    ok = np.all(np.isfinite(P), 1) & np.all(np.isfinite(R_state), 0)
    if use_counts is not None:
        ok &= use_counts >= min_visits
    idx = np.flatnonzero(ok)
    if len(idx) < P.shape[1] + 2:
        raise ValueError(f"only {len(idx)} states with profiles and rewards for {P.shape[1]} variables")
    w = (use_counts[idx].astype(float) if use_counts is not None else np.ones(len(idx)))
    Z = zscore(P[idx], w)
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(K):
        y = zscore(R_state[k, idx], w)
        r2 = {a: cv_r2(Z, y, w, a, min(5, len(idx)), seed) for a in alphas}
        alpha = max(r2, key=lambda a: -np.inf if np.isnan(r2[a]) else r2[a])
        b = ridge_fit(Z, y, w, alpha)
        boots = np.empty((n_boot, len(b)))
        for i in range(n_boot):
            s = rng.integers(0, len(idx), len(idx))
            boots[i] = ridge_fit(Z[s], y[s], w[s], alpha)
        if n_boot:
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        else:
            lo = hi = np.full(len(b), np.nan)
        for j, n in enumerate(names):
            rho = spearmanr(P[idx, j], R_state[k, idx]).correlation
            rows.append(dict(mode=k, variable=n, coef=b[j], ci_low=lo[j], ci_high=hi[j],
                             ci_excludes_0=bool(lo[j] > 0 or hi[j] < 0), spearman=rho,
                             alpha=alpha, cv_r2=r2[alpha], n_states=len(idx)))
    return pd.DataFrame(rows)


def partial_dependence(R_state, P, names, use_counts=None, n_bins=5):
    rows = []
    K = R_state.shape[0]
    w_all = use_counts.astype(float) if use_counts is not None else np.ones(P.shape[0])
    for j, n in enumerate(names):
        ok = np.isfinite(P[:, j]) & (w_all > 0)
        if ok.sum() < n_bins:
            continue
        edges = np.unique(np.quantile(P[ok, j], np.linspace(0, 1, n_bins + 1)))
        b = np.clip(np.searchsorted(edges, P[:, j], side="right") - 1, 0, len(edges) - 2)
        for k in range(K):
            yk = zscore(R_state[k][ok], w_all[ok])
            full = np.full(P.shape[0], np.nan)
            full[ok] = yk
            for q in range(len(edges) - 1):
                sel = ok & (b == q) & np.isfinite(full)
                if sel.any():
                    rows.append(dict(mode=k, variable=n, bin=q, lo=edges[q], hi=edges[q + 1],
                                     centre=np.average(P[sel, j], weights=w_all[sel]),
                                     reward_z=np.average(full[sel], weights=w_all[sel]), n_states=int(sel.sum()),
                                     weight=float(w_all[sel].sum())))
    return pd.DataFrame(rows)


def coefficient_stability(coef_tables):
    """Spearman between coefficient vectors of the same mode across fold reward files (modes matched by index)."""
    from scipy.stats import spearmanr
    rows = []
    tags = list(coef_tables)
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = coef_tables[tags[i]], coef_tables[tags[j]]
            for k in sorted(set(a["mode"]) & set(b["mode"])):
                ca = a[a["mode"] == k].set_index("variable")["coef"]
                cb = b[b["mode"] == k].set_index("variable")["coef"].reindex(ca.index)
                rows.append(dict(tag_a=tags[i], tag_b=tags[j], mode=k, spearman=spearmanr(ca, cb).correlation,
                                 sign_agreement=float(np.mean(np.sign(ca) == np.sign(cb)))))
    return pd.DataFrame(rows)


def match_modes(ref, other):
    """Hungarian match of mode indices by reward correlation (fold runs permute labels)."""
    from scipy.optimize import linear_sum_assignment
    ok = np.all(np.isfinite(ref), 0) & np.all(np.isfinite(other), 0)
    c = np.corrcoef(np.vstack([ref[:, ok], other[:, ok]]))[:len(ref), len(ref):]
    r, col = linear_sum_assignment(-np.nan_to_num(c))
    perm = np.empty(len(ref), dtype=int)
    perm[r] = col
    return other[perm]


# ============================== USE VS AVAILABILITY ===============================

def available_next(T, action_weights=None):
    """(S, S) availability: mean over actions of T[s, a, :] (uniform or given action weights)."""
    T = np.asarray(T.todense() if hasattr(T, "todense") else T, float)
    if T.ndim != 3:
        raise ValueError(f"T must be (S, A, S), got {T.shape}")
    w = np.full(T.shape[1], 1.0 / T.shape[1]) if action_weights is None else np.asarray(action_weights, float)
    w = w / w.sum()
    Av = np.einsum("sat,a->st", T, w)
    assert np.allclose(Av.sum(1)[Av.sum(1) > 0], 1.0, atol=1e-6), "availability rows must sum to 1"
    return Av


def transitions_by_mode(series):
    """(uid, mode, s, s_next) for every fix with a decoded mode and a valid next state in the same segment."""
    rows = []
    for _, df in series.items():
        for _, g in df.groupby("segment", sort=False):
            g = g.sort_values("seg_pos")
            s = g["env"].to_numpy(float)
            z = g["latent"].to_numpy(float)
            pos = g["seg_pos"].to_numpy(int)
            consec = np.r_[np.diff(pos) == 1, False]
            nxt = np.r_[s[1:], np.nan]
            ok = consec & np.isfinite(s) & np.isfinite(nxt) & np.isfinite(z)
            uid = g["uid"].iloc[0]
            rows.append(pd.DataFrame(dict(uid=uid, mode=z[ok].astype(int), s=s[ok].astype(int),
                                          s_next=nxt[ok].astype(int))))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["uid", "mode", "s", "s_next"])


def use_availability(tr, Av, P, names, n_boot=500, seed=0):
    """Per mode and variable: mean P[s_next] - sum_s' Av[s, s'] P[s'] with an animal bootstrap CI."""
    rng = np.random.default_rng(seed)
    Pf = np.where(np.isfinite(P), P, np.nan)
    rows = []
    for k, g in tr.groupby("mode"):
        used = Pf[g["s_next"].to_numpy()]
        A = Av[g["s"].to_numpy()]
        avail = np.stack([np.nansum(A * np.where(np.isfinite(Pf[:, j]), Pf[:, j], 0)[None], 1)
                          / np.maximum((A * np.isfinite(Pf[:, j])[None]).sum(1), 1e-12) for j in range(P.shape[1])], 1)
        diff = used - avail
        uids = g["uid"].to_numpy()
        uu = np.unique(uids)
        per_animal = {u: np.flatnonzero(uids == u) for u in uu}
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(uu, len(uu), replace=True)
            ii = np.concatenate([per_animal[u] for u in pick])
            boots.append(np.nanmean(diff[ii], 0))
        boots = np.array(boots)
        sd = np.nanstd(Pf, 0)
        for j, n in enumerate(names):
            m = np.nanmean(diff[:, j])
            lo, hi = np.nanpercentile(boots[:, j], [2.5, 97.5]) if len(boots) else (np.nan, np.nan)
            rows.append(dict(mode=int(k), variable=n, used_mean=np.nanmean(used[:, j]),
                             available_mean=np.nanmean(avail[:, j]), diff=m, diff_sd_units=m / sd[j] if sd[j] else np.nan,
                             ci_low=lo, ci_high=hi, ci_excludes_0=bool(lo > 0 or hi < 0),
                             n_steps=len(g), n_animals=len(uu)))
    return pd.DataFrame(rows)


def selection_ratios(tr, Av, n_states):
    rows = []
    for k, g in tr.groupby("mode"):
        used = np.bincount(g["s_next"], minlength=n_states).astype(float)
        avail = Av[g["s"].to_numpy()].sum(0)
        for s in np.flatnonzero((used > 0) | (avail > 0)):
            rows.append(dict(mode=int(k), state=int(s), used=used[s], available=avail[s],
                             ratio=used[s] / avail[s] if avail[s] > 0 else np.inf))
    return pd.DataFrame(rows)


# ============================== DRIVER ===============================

def load_transition(yml_conf, n_states):
    """Global T from yml 'trans_probs' if given, else estimated from all segments (as the final run does)."""
    if yml_conf.get("trans_probs"):
        import sparse
        return sparse.load_npz(yml_conf["trans_probs"])
    from naturenet.models.irl.swirl_data import load_segments, trans_prob_from_segments
    n_actions = int(yml_conf.get("n_actions", 9))
    return trans_prob_from_segments(load_segments(yml_conf), n_states, n_actions,
                                    fill_unobserved=yml_conf.get("fill_unobserved", "self_loop"))


def run(conf, R_state, P, names, series, T, out_dir, fold_rewards=None):
    os.makedirs(out_dir, exist_ok=True)
    S = P.shape[0]
    assert R_state.shape[1] == S, f"reward has {R_state.shape[1]} states, profiles {S}"
    tr = transitions_by_mode(series)
    use = np.bincount(np.concatenate([df["env"].dropna().astype(int).to_numpy() for df in series.values()]),
                      minlength=S)[:S] if series else None
    w = use if conf["weight_by_use"] else None
    coefs = project_rewards(R_state, P, names, w, conf["alphas"], conf["n_boot"], conf["seed"],
                            conf["min_state_visits"])
    coefs.to_csv(os.path.join(out_dir, "reward_projection_coefficients.csv"), index=False)
    partial_dependence(R_state, P, names, use, conf["n_bins"]).to_csv(
        os.path.join(out_dir, "reward_partial_dependence.csv"), index=False)
    out = dict(coefficients=coefs)
    if fold_rewards:
        tables = {"final": coefs}
        for tag, Rf in fold_rewards.items():
            tables[tag] = project_rewards(match_modes(R_state, Rf), P, names, w, conf["alphas"], 0, conf["seed"],
                                          conf["min_state_visits"])
        stab = coefficient_stability(tables)
        stab.to_csv(os.path.join(out_dir, "reward_coefficient_stability.csv"), index=False)
        out["stability"] = stab
    if T is not None and len(tr):
        Av = available_next(T)
        ua = use_availability(tr, Av, P, names, conf["n_boot_animals"], conf["seed"])
        ua.to_csv(os.path.join(out_dir, "use_availability.csv"), index=False)
        selection_ratios(tr, Av, S).to_csv(os.path.join(out_dir, "selection_ratios.csv"), index=False)
        out["use_availability"] = ua
    plot_coefficients(coefs, os.path.join(out_dir, "reward_projection_coefficients.png"))
    return out


def plot_coefficients(coefs, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = sorted(coefs["mode"].unique())
    fig, axes = plt.subplots(1, len(modes), figsize=(3.2 * len(modes), 0.35 * coefs["variable"].nunique() + 1.5),
                             sharey=True, squeeze=False)
    for ax, k in zip(axes[0], modes):
        c = coefs[coefs["mode"] == k]
        y = np.arange(len(c))
        ax.errorbar(c["coef"], y, xerr=[c["coef"] - c["ci_low"], c["ci_high"] - c["coef"]], fmt="o", ms=4)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(y, c["variable"])
        ax.set_title(f"mode {k} (CV R2 {c['cv_r2'].iloc[0]:.2f})", fontsize=9)
        ax.set_xlabel("standardised coefficient")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main(path, proj_path=None, n_hidden=None):
    """path: the yml of the final SWIRL / HIAVI run; proj_path: optional yml with the reward_env_projection block."""
    from sit_fuse.utils import read_yaml
    yml = read_yaml(path)
    if n_hidden is not None:
        yml["n_hidden"] = yml["n_hidden_init"] = int(n_hidden)
    conf = dict(DEFAULTS)
    conf.update(yml.get("reward_env_projection", {}) or {})
    if proj_path:
        conf.update(read_yaml(proj_path).get("reward_env_projection", {}) or {})
    S = int(yml["n_states"])
    with open(conf["class_mapper"], "rb") as f:
        mapper = pickle.load(f)
    P, names = state_profiles(pd.read_csv(conf["profiles_csv"]), S, mapper, conf["cluster_scale"],
                              conf["profile_index"], conf["variables"])
    rm = load_reward_model(yml, conf["model"], conf["order"], conf["tag"])
    series = build_track_series(yml, conf["model"], conf["order"], tuple(conf["splits"]), conf["tag"], rewards=rm)
    fold = {t: load_reward_model(yml, conf["model"], conf["order"], t).state_reward for t in conf["fold_tags"]}
    T = load_transition(yml, S)
    out_dir = conf.get("out_dir") or os.path.join(yml["out_dir"], "reward_env_projection")
    run(conf, rm.state_reward, P, names, series, T, out_dir, fold or None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-y", "--yaml", required=True, help="yml of the final SWIRL / HIAVI run")
    p.add_argument("-p", "--projection", default=None, help="yml with the reward_env_projection block")
    p.add_argument("--n_hidden", type=int, default=None, help="K of the final run to analyse")
    a = p.parse_args()
    main(a.yaml, a.projection, a.n_hidden)
