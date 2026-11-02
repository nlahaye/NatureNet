"""
Minimum-viable multi-agent IRL test: do concurrently tracked animals share intention dynamics?

    python -m naturenet.models.irl.run_hiavi_multiagent -y config/swirl/hiavi_multiagent.yaml [--K 2] [--skip_null]
    python -m naturenet.models.irl.plot_hiavi_multiagent -y config/swirl/hiavi_multiagent.yaml

Uses the HIAVI data pipeline (same YAML keys as hiavi_kfold.yaml: actions / positions / n_states / seq_len /
segment_policy / hiavi_*), plus a `multiagent` block (see the YAML). Steps:

1. Step metadata: every aligned HIAVI step is mapped to its fix (timestamp, lat, lon) through the sequence
   ids (key, segment, start) and <df_dir>/<df_run_uid>_dfs.pkl (step t <-> fix start + seq_offset + t).
2. Concurrency report (ma_concurrency.csv): per animal, original-track steps with another animal tracked
   within the kernel window, and within the proximity cut.
3. Leave-one-animal-out CV for each K and coupling variant (default: temporal; proximity-weighted):
     independent HIAVI (restarts, best train LL)  vs  coupled HIAVI (warm start, mean-field coupling).
   The held-out animal's context comes from the TRAINING animals' posteriors at the same calendar time;
   the per-fold transition model uses the training animals only.
   Metric: held-out one-step predictive LL difference (coupled - independent), summed / per step over the
   held-out animal's original-track steps (augmented copies excluded unless eval_include_aug), overall and
   on the steps that have partner context (w > 0).
4. Time-shift null: each animal's calendar is circularly shifted by an independent random offset
   (>= null_min_shift_days) inside the window, which keeps every track and its autocorrelation but breaks
   co-timing. The whole CV is re-run per replicate; p = (1 + #{null >= observed}) / (1 + n_null) on the
   pooled held-out gain.
5. Synchrony index on a full-data independent HIAVI fit (no coupling in the model): mean expected
   intention agreement sum_k gamma_i(k) c_i(k) over steps with partner context, vs the time-shift null and
   vs chance sum_k pbar_k^2.
6. Final full-data fits (independent and every coupled variant) with posteriors / times for plotting.

Outputs in <out_dir>/<ma_subdir>/: ma_concurrency.csv, ma_pairs.npz, ma_cv.csv, ma_sync.csv, ma_summary.json,
ma_final_K<K>.npz.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import yaml

from naturenet.models.irl.hiavi_model import HIAVI
from naturenet.models.irl.hiavi_multiagent import (
    CoupledHIAVI,
    PartnerContext,
    fit_coupled,
    hiavi_step_loglik,
)
from naturenet.models.irl.run_hiavi_kfold import _seed, fold_trans_prob, hiavi_settings, prepare_hiavi_data
from naturenet.models.irl.swirl_track_series import _df_key

DEFAULT_VARIANTS = {"temporal": {"h_days": 1.0, "max_dt_days": 3.0},
                    "proximity": {"h_days": 1.0, "max_dt_days": 3.0, "lambda_km": 50.0, "max_km": 150.0}}


# ---------------------------------------------------------------------------
# Config / data
# ---------------------------------------------------------------------------

def read_conf(path):
    with open(path) as f:
        return expand_inputs(yaml.safe_load(f))


def expand_inputs(yml_conf):
    """
    actions may be a glob ("<df_dir>/hammerhead_v1_*_actions.pkl"); positions "auto" pairs each actions file
    with its <...>_states_abstract_env.pkl. Optional multiagent.animals keeps only those base ids.
    """
    acts = yml_conf["actions"]
    if isinstance(acts, str) and any(ch in acts for ch in "*?["):
        acts = sorted(glob.glob(acts))
        if not acts:
            raise FileNotFoundError(f"no files match {yml_conf['actions']}")
    acts = [acts] if isinstance(acts, str) else list(acts)
    keep = (yml_conf.get("multiagent") or {}).get("animals")
    if keep:
        keep = {str(k) for k in keep}
        def _animal(f):
            k = re.sub(r"__aug\d+$", "", os.path.basename(f)[:-len("_actions.pkl")])
            return k.split(yml_conf["df_run_uid"] + "_", 1)[-1]
        acts = [f for f in acts if any(_animal(f) == k or _animal(f).startswith(k + "_") for k in keep)]
        if not acts:
            raise ValueError(f"multiagent.animals {sorted(keep)} matched no actions files")
    pos = yml_conf.get("positions", "auto")
    if pos == "auto" or pos is None:
        pos = [f.replace("_actions.pkl", "_states_abstract_env.pkl") for f in acts]
    yml_conf = dict(yml_conf, actions=acts, positions=pos)
    return yml_conf


def ma_conf(yml_conf):
    m = dict(yml_conf.get("multiagent") or {})
    m.setdefault("Ks", [int(yml_conf.get("n_hidden", 2))])
    m.setdefault("variants", DEFAULT_VARIANTS)
    m.setdefault("n_outer", 5)
    m.setdefault("alpha0", 0.1)
    m.setdefault("n_null", 20)
    m.setdefault("null_variants", ["temporal"])
    m.setdefault("null_min_shift_days", 30)
    m.setdefault("n_null_sync", 200)
    m.setdefault("sync_variant", "temporal")
    m.setdefault("eval_include_aug", False)
    m.setdefault("min_context_steps", 30)
    m.setdefault("ma_subdir", "multiagent")
    m.setdefault("seed", int(yml_conf.get("seed", 0)))
    return m


def _to_days(ts):
    ts = pd.to_datetime(pd.Series(ts), utc=True, errors="coerce", format="mixed").dt.tz_convert(None)
    return (ts - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy() / 86400.0


def step_metadata(yml_conf, data, dfs=None):
    """(N, T) times (days since 1970), lat, lon for the aligned HIAVI steps; animal (N,), key (N,)."""
    if dfs is None:
        path = yml_conf.get("multiagent", {}).get("dfs_pkl") or \
            os.path.join(yml_conf["df_dir"], yml_conf["df_run_uid"] + "_dfs.pkl")
        dfs = pd.read_pickle(path)
    s = data["hiavi_s"]
    N, T = s.shape
    off = int(data["hiavi_offset"])
    times, lat, lon = np.full((N, T), np.nan), np.full((N, T), np.nan), np.full((N, T), np.nan)
    keys = []
    for i, m in enumerate(data["meta"]):
        dkey = _df_key(m["key"], yml_conf["df_run_uid"])
        if dkey not in dfs:
            raise KeyError(f"sequence {i}: no movement DataFrame for {dkey} in dfs.pkl")
        df = dfs[dkey][m["seg"]]
        pos = m["start"] + off + np.arange(T)
        if pos.max() >= len(df):
            raise ValueError(f"{dkey} segment {m['seg']}: step position {pos.max()} >= {len(df)} fixes "
                             "(dfs.pkl and actions / positions out of sync?)")
        sub = df.iloc[pos]
        times[i] = _to_days(sub["timestamp"].to_numpy())
        lat[i] = sub["latitude"].to_numpy(dtype=float)
        lon[i] = sub["longitude"].to_numpy(dtype=float)
        keys.append(str(m["key"]))
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{int((~np.isfinite(times)).sum())} steps have no valid timestamp")
    return PartnerContext(times, lat, lon, np.asarray(data["groups"]).astype(str), np.asarray(keys))


def original_mask(pc):
    return np.array([re.search(r"__aug\d+", k) is None for k in pc.key])


def window_days(conf, pc):
    w = conf.get("window")
    if w:
        return float(_to_days([w[0]])[0]), float(_to_days([w[1]])[0]) + 1.0
    return float(pc.times.min()), float(pc.times.max()) + 1.0


def draw_shifts(animals, rng, window, min_shift):
    L = window[1] - window[0]
    if L <= 2 * min_shift:
        raise ValueError(f"window ({L:.0f} d) too short for null_min_shift_days={min_shift}")
    return {a: float(rng.uniform(min_shift, L - min_shift)) for a in animals}


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def concurrency_report(pc, variants, out_dir):
    allidx = np.arange(pc.N)
    orig = original_mask(pc)
    rows, pairs = [], {}
    for vname, v in variants.items():
        kern = pc.build(allidx, allidx, **v)
        mass = np.asarray(kern["K"].sum(axis=1)).ravel().reshape(pc.N, pc.T)
        pairs[vname + "_dist_km"] = pc._last_pairs["dist_km"]
        pairs[vname + "_dt_days"] = pc._last_pairs["dt_days"]
        for an in np.unique(pc.animal):
            m = (pc.animal == an) & orig
            rows.append(dict(variant=vname, animal=an, seqs=int(m.sum()), steps=int(m.sum() * pc.T),
                             steps_with_context=int((mass[m] > 0).sum()),
                             frac_with_context=float((mass[m] > 0).mean()) if m.any() else 0.0,
                             mean_w=float((1 - np.exp(-mass[m])).mean()) if m.any() else 0.0))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ma_concurrency.csv"), index=False)
    np.savez(os.path.join(out_dir, "ma_pairs.npz"), **pairs)
    print("Concurrency (original tracks):")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

def fit_independent(yml_conf, K, T_env, s, a, idx, kf_ind, settings):
    best, best_ll = None, -np.inf
    for rs in range(int(yml_conf.get("hiavi_restarts", 10))):
        m = HIAVI(K, T_env, seed=_seed(yml_conf["seed"], kf_ind, rs), **settings).fit(s[idx], a[idx])
        ll = m.loglik(s[idx], a[idx])
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError(f"all HIAVI restarts failed (K={K}, fold {kf_ind})")
    return best, best_ll


def heldout_scores(pc, model0, models, s, a, tr, te, kerns_te, eval_mask):
    """Per-step predictive LLs on te for the independent and each coupled model; context from train."""
    out = {}
    l0 = hiavi_step_loglik(model0, s[te], a[te])
    for vname, (m, (w_tr, c_tr)) in models.items():
        g_tr = m.posteriors(s[tr], a[tr], w_tr, c_tr)
        w_te, c_te = PartnerContext.context(kerns_te[vname], g_tr)
        l1 = m.step_loglik(s[te], a[te], w_te, c_te)
        msk = eval_mask[te][:, None] & np.ones_like(w_te, dtype=bool)
        conc = msk & (w_te > 0)
        d = l1 - l0
        out[vname] = dict(alpha=m.alpha, test_ll_indep=float(l0[msk].mean()), test_ll_coupled=float(l1[msk].mean()),
                          d_total=float(d[msk].sum()), d_per_step=float(d[msk].mean()),
                          d_conc=float(d[conc].sum()) if conc.any() else 0.0,
                          d_conc_per_step=float(d[conc].mean()) if conc.any() else np.nan,
                          n_test_steps=int(msk.sum()), n_conc_steps=int(conc.sum()))
    return out


def run_cv(yml_conf, conf, data, pc, K, variants, times=None, null_id=-1, base_cache=None, verbose=True):
    """One leave-one-animal-out pass (observed calendar if times is None, else shifted). Returns rows."""
    st = hiavi_settings(yml_conf)
    s, a = data["hiavi_s"], data["hiavi_a"]
    groups = np.asarray(data["groups"]).astype(str)
    eval_mask = np.ones(len(groups), bool) if conf["eval_include_aug"] else original_mask(pc)
    animals = conf.get("heldout_animals") or sorted(np.unique(groups))
    rows = []
    for kf, an in enumerate(animals):
        te = np.where(groups == str(an))[0]
        tr = np.where(groups != str(an))[0]
        T_env = fold_trans_prob(yml_conf, data, tr)
        ck = (K, an)
        if base_cache is not None and ck in base_cache:
            m0, ll0 = base_cache[ck]
        else:
            m0, ll0 = fit_independent(yml_conf, K, T_env, s, a, tr, kf, st)
            if base_cache is not None:
                base_cache[ck] = (m0, ll0)
        models, kerns_te = {}, {}
        for vname, v in variants.items():
            kern_tr = pc.build(tr, tr, times=times, **v)
            kerns_te[vname] = pc.build(te, tr, times=times, **v)
            models[vname] = fit_coupled(m0, T_env, s, a, pc, tr, st, kern_tr, n_outer=conf["n_outer"],
                                        alpha0=conf["alpha0"])
        sc = heldout_scores(pc, m0, models, s, a, tr, te, kerns_te, eval_mask)
        for vname, r in sc.items():
            m, (w_tr, c_tr) = models[vname]
            rows.append(dict(K=K, fold=kf, heldout=an, variant=vname, null_id=null_id, train_ll_indep=ll0,
                             train_ll_coupled=m.loglik(s[tr], a[tr], w_tr, c_tr), **r))
            if verbose:
                print(f"K={K} hold {an} {vname:9s} null {null_id:3d}: alpha {r['alpha']:.3f} "
                      f"dLL/step {r['d_per_step']:+.4f} (context steps {r['n_conc_steps']}, "
                      f"dLL there {r['d_conc_per_step']:+.4f})")
    return rows


def synchrony(pc, model, s, a, variant, times=None, mask=None):
    """Mean expected agreement sum_k gamma_i c_i over steps with context (full-data independent posteriors)."""
    idx = np.arange(pc.N)
    g = model.posteriors(s, a)
    kern = pc.build(idx, idx, times=times, **variant)
    w, c = PartnerContext.context(kern, g)
    m = (w > 0) if mask is None else (w > 0) & mask[:, None]
    if not m.any():
        return np.nan, 0, g
    agree = (g * c).sum(-1)
    return float(agree[m].mean()), int(m.sum()), g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_multiagent(yml_conf, Ks=None, skip_null=False, verbose=True):
    yml_conf = expand_inputs(yml_conf)
    conf = ma_conf(yml_conf)
    out_dir = os.path.join(yml_conf["out_dir"], conf["ma_subdir"])
    os.makedirs(out_dir, exist_ok=True)
    data = prepare_hiavi_data(yml_conf)
    pc = step_metadata(yml_conf, data)
    variants = conf["variants"]
    win = window_days(conf, pc)
    rng = np.random.default_rng(conf["seed"])
    animals = sorted(np.unique(pc.animal))
    orig = original_mask(pc)

    conc = concurrency_report(pc, variants, out_dir)
    ok = conc[(conc["variant"] == conf["sync_variant"]) & (conc["steps_with_context"] >= conf["min_context_steps"])]
    if len(ok) < 2:
        print(f"WARNING: fewer than 2 animals have >= {conf['min_context_steps']} steps with partner context; "
              "the multi-agent test has no power on this window")

    s, a = data["hiavi_s"], data["hiavi_a"]
    st = hiavi_settings(yml_conf)
    null_shifts = [draw_shifts(animals, rng, win, conf["null_min_shift_days"]) for _ in range(int(conf["n_null"]))]
    sync_shifts = [draw_shifts(animals, rng, win, conf["null_min_shift_days"])
                   for _ in range(int(conf["n_null_sync"]))]
    rows, sync_rows, summary = [], [], {"window_days": win, "animals": animals, "K": {}}
    for K in (Ks or conf["Ks"]):
        K = int(K)
        cache = {}
        rows_k = run_cv(yml_conf, conf, data, pc, K, variants, base_cache=cache, verbose=verbose)
        if not skip_null:
            nv = {v: variants[v] for v in conf["null_variants"] if v in variants}
            for r, sh in enumerate(null_shifts):
                rows_k += run_cv(yml_conf, conf, data, pc, K, nv, times=pc.shifted_times(sh, win), null_id=r,
                                 base_cache=cache, verbose=False)
                if verbose:
                    print(f"K={K}: null replicate {r + 1}/{len(null_shifts)} done")
        rows += rows_k

        # full-data fits
        all_idx = np.arange(pc.N)
        T_all = fold_trans_prob(yml_conf, data, all_idx)
        m0, ll0 = fit_independent(yml_conf, K, T_all, s, a, all_idx, None, st)
        sv = variants[conf["sync_variant"]]
        a_obs, n_obs, g0 = synchrony(pc, m0, s, a, sv, mask=orig)
        a_null = [synchrony(pc, m0, s, a, sv, times=pc.shifted_times(sh, win), mask=orig)[0] for sh in sync_shifts]
        pbar = g0[orig].reshape(-1, K).mean(0)
        chance = float((pbar ** 2).sum())
        a_null = np.asarray([x for x in a_null if np.isfinite(x)])
        p_sync = float((1 + np.sum(a_null >= a_obs)) / (1 + len(a_null))) if len(a_null) else np.nan
        sync_rows += [dict(K=K, kind="observed", agreement=a_obs, n_steps=n_obs)] + \
                     [dict(K=K, kind="time_shift", agreement=x, n_steps=np.nan) for x in a_null]
        final = dict(times=pc.times, lat=pc.lat, lon=pc.lon, animal=pc.animal, key=pc.key, original=orig,
                     gamma_indep=g0, logp_tr_indep=m0.logp_tr, pis_indep=m0.pis, r_indep=m0.r, train_ll_indep=ll0)
        for vname, v in variants.items():
            kern = pc.build(all_idx, all_idx, **v)
            m, (w, c) = fit_coupled(m0, T_all, s, a, pc, all_idx, st, kern, n_outer=conf["n_outer"],
                                    alpha0=conf["alpha0"])
            final.update({f"gamma_{vname}": m.posteriors(s, a, w, c), f"w_{vname}": w, f"alpha_{vname}": m.alpha,
                          f"logp_tr_{vname}": m.logp_tr, f"r_{vname}": m.r,
                          f"train_ll_{vname}": m.loglik(s, a, w, c)})
        np.savez(os.path.join(out_dir, f"ma_final_K{K}.npz"), **final)

        # summary
        df = pd.DataFrame(rows_k)
        sk = {"sync": dict(observed=a_obs, n_steps=n_obs, null_mean=float(np.mean(a_null)) if len(a_null) else None,
                           null_sd=float(np.std(a_null)) if len(a_null) else None, chance=chance, p=p_sync),
              "variants": {}}
        for vname in variants:
            ob = df[(df["variant"] == vname) & (df["null_id"] < 0)]
            nu = df[(df["variant"] == vname) & (df["null_id"] >= 0)]
            pooled = float(ob["d_total"].sum())
            n_steps = int(ob["n_test_steps"].sum())
            ent = dict(pooled_d_total=pooled, pooled_d_per_step=pooled / max(n_steps, 1),
                       pooled_d_conc=float(ob["d_conc"].sum()), n_conc_steps=int(ob["n_conc_steps"].sum()),
                       alpha_median=float(ob["alpha"].median()),
                       animals_improved=int((ob["d_total"] > 0).sum()), n_animals=int(len(ob)),
                       full_data_alpha=float(final[f"alpha_{vname}"]),
                       # alpha * mean w on steps with context: the mixing weight actually applied (alpha can sit
                       # at its bound when w is small, e.g. under the proximity kernel)
                       effective_coupling=float(final[f"alpha_{vname}"] *
                                                final[f"w_{vname}"][(final[f"w_{vname}"] > 0) & orig[:, None]].mean())
                       if np.any((final[f"w_{vname}"] > 0) & orig[:, None]) else 0.0)
            if len(nu):
                null_pooled = nu.groupby("null_id")["d_total"].sum().to_numpy()
                null_alpha = nu.groupby("null_id")["alpha"].median().to_numpy()
                ent.update(null_pooled_mean=float(null_pooled.mean()), null_pooled_sd=float(null_pooled.std()),
                           p_gain=float((1 + np.sum(null_pooled >= pooled)) / (1 + len(null_pooled))),
                           p_alpha=float((1 + np.sum(null_alpha >= ent["alpha_median"])) / (1 + len(null_alpha))),
                           n_null=int(len(null_pooled)))
            sk["variants"][vname] = ent
        summary["K"][str(K)] = sk
        print(json.dumps(sk, indent=1))

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ma_cv.csv"), index=False)
    pd.DataFrame(sync_rows).to_csv(os.path.join(out_dir, "ma_sync.csv"), index=False)
    with open(os.path.join(out_dir, "ma_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", out_dir)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--K", type=int, nargs="*", default=None)
    ap.add_argument("--skip_null", action="store_true")
    args = ap.parse_args()
    run_multiagent(read_conf(args.yaml), Ks=args.K, skip_null=args.skip_null)
