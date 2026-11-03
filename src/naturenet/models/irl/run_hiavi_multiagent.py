"""
Minimum-viable multi-agent IRL test: do concurrently tracked animals share intention dynamics?

    python -m naturenet.models.irl.run_hiavi_multiagent -y config/swirl/hiavi_multiagent.yaml [--K 2] [--skip_null]
    python -m naturenet.models.irl.run_swirl_multiagent -y config/swirl/swirl_multiagent.yaml   # SWIRL backend
    python -m naturenet.models.irl.run_hiavi_multiagent -y <yaml> --concurrency_only            # data check only
    python -m naturenet.models.irl.plot_hiavi_multiagent -y <yaml> [--subdir multiagent_swirl]

The CV / null / synchrony logic is model-agnostic: a backend (HiaviBackend here, swirl_multiagent.SwirlBackend,
scales_multiagent.ScalesSwirlBackend / ScalesHiqlBackend for the decision-scale toggles) fits the independent
and the partner-coupled model and scores them (multiagent.backend: hiavi | swirl | swirl_scales | hiql_scales;
run_scales_multiagent loops the scale variants).
With a `kinematics` block both models also emit per-intention step / turn densities (kinematics.py); the
coupled-vs-independent comparison then isolates the coupling with kinematics held in both models, and
ma_final_K<K>.npz / ma_kinematics.csv carry the kinematic parameters, exploit / explore labels and P(exploit).

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
ma_final_K<K>.npz (+ ma_kinematics.csv with kinematics).
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
from naturenet.models.irl.kinematics import kin_conf, labels_from_conf, p_exploit
from naturenet.models.irl.run_hiavi_kfold import (
    _seed,
    fold_trans_prob,
    hiavi_kinematics,
    hiavi_settings,
    prepare_hiavi_data,
)
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
    with its <...>_states_abstract_env.pkl. Optional multiagent.animals keeps only those base ids;
    multiagent.max_aug keeps only augmented copies __aug1..__aug<max_aug> (0 = original tracks only; null = all).
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
    max_aug = (yml_conf.get("multiagent") or {}).get("max_aug")
    if max_aug is not None:
        def _aug(f):
            m = re.search(r"__aug(\d+)_actions\.pkl$", os.path.basename(f))
            return int(m.group(1)) if m else 0
        acts = [f for f in acts if _aug(f) <= int(max_aug)]
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
    valid = np.asarray(data.get("hiavi_mask", np.ones((N, T), bool)), bool)   # padded steps: NaN metadata
    times, lat, lon = np.full((N, T), np.nan), np.full((N, T), np.nan), np.full((N, T), np.nan)
    keys = []
    for i, m in enumerate(data["meta"]):
        dkey = _df_key(m["key"], yml_conf["df_run_uid"])
        if dkey not in dfs:
            raise KeyError(f"sequence {i}: no movement DataFrame for {dkey} in dfs.pkl")
        df = dfs[dkey][m["seg"]]
        nt = int(valid[i].sum())
        assert valid[i, :nt].all(), "padding must be at the end of a sequence"
        pos = m["start"] + off + np.arange(nt)
        if pos.max() >= len(df):
            raise ValueError(f"{dkey} segment {m['seg']}: step position {pos.max()} >= {len(df)} fixes "
                             "(dfs.pkl and actions / positions out of sync?)")
        sub = df.iloc[pos]
        times[i, :nt] = _to_days(sub["timestamp"].to_numpy())
        lat[i, :nt] = sub["latitude"].to_numpy(dtype=float)
        lon[i, :nt] = sub["longitude"].to_numpy(dtype=float)
        keys.append(str(m["key"]))
    if not np.all(np.isfinite(times[valid])):
        raise ValueError(f"{int((~np.isfinite(times[valid])).sum())} steps have no valid timestamp")
    return PartnerContext(times, lat, lon, np.asarray(data["groups"]).astype(str), np.asarray(keys), valid=valid)


def original_mask(pc):
    return np.array([re.search(r"__aug\d+", k) is None for k in pc.key])


def window_days(conf, pc):
    w = conf.get("window")
    if w:
        return float(_to_days([w[0]])[0]), float(_to_days([w[1]])[0]) + 1.0
    return float(np.nanmin(pc.times)), float(np.nanmax(pc.times)) + 1.0


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
            v_ = pc.valid[m]                                  # real (unpadded) steps
            mm = mass[m][v_]
            rows.append(dict(variant=vname, animal=an, seqs=int(m.sum()), steps=int(v_.sum()),
                             steps_with_context=int((mm > 0).sum()),
                             frac_with_context=float((mm > 0).mean()) if mm.size else 0.0,
                             mean_w=float((1 - np.exp(-mm)).mean()) if mm.size else 0.0))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ma_concurrency.csv"), index=False)
    np.savez(os.path.join(out_dir, "ma_pairs.npz"), **pairs)
    print("Concurrency (original tracks):")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Backends: the CV / null / synchrony logic below is model-agnostic. A backend fits the independent and the
# partner-coupled model and scores / decodes them; HIAVI here, SWIRL in swirl_multiagent.SwirlBackend.
# ---------------------------------------------------------------------------

class HiaviBackend:
    """Independent HIAVI vs CoupledHIAVI (optionally with per-intention kinematic emissions)."""
    name = "HIAVI"

    def __init__(self, yml_conf, conf):
        self.yml_conf, self.conf = yml_conf, conf
        self.st = hiavi_settings(yml_conf)

    def prepare(self):
        self.data = prepare_hiavi_data(self.yml_conf)
        self.s, self.a = self.data["hiavi_s"], self.data["hiavi_a"]
        self.kin = hiavi_kinematics(self.yml_conf, self.data)
        return self.data

    def _k(self, idx):
        return None if self.kin is None else self.kin[idx]

    def fold_env(self, tr):
        return fold_trans_prob(self.yml_conf, self.data, tr)

    def fit_independent(self, K, T_env, idx, kf_ind):
        best, best_ll = None, -np.inf
        for rs in range(int(self.yml_conf.get("hiavi_restarts", 10))):
            m = HIAVI(K, T_env, seed=_seed(self.yml_conf["seed"], kf_ind, rs), **self.st).fit(
                self.s[idx], self.a[idx], self._k(idx))
            ll = m.loglik(self.s[idx], self.a[idx], self._k(idx))
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
        if best is None:
            raise RuntimeError(f"all HIAVI restarts failed (K={K}, fold {kf_ind})")
        return best, best_ll

    def fit_coupled(self, m0, T_env, idx, kern):
        return fit_coupled(m0, T_env, self.s, self.a, None, idx, self.st, kern, n_outer=self.conf["n_outer"],
                           alpha0=self.conf["alpha0"], kin=self.kin)

    def step_ll_indep(self, m0, T_env, idx):
        return hiavi_step_loglik(m0, self.s[idx], self.a[idx], self._k(idx))

    def posteriors_indep(self, m0, T_env, idx):
        return m0.posteriors(self.s[idx], self.a[idx], self._k(idx))

    def posteriors(self, m, T_env, idx, w, c):
        return m.posteriors(self.s[idx], self.a[idx], w, c, self._k(idx))

    def step_ll(self, m, T_env, idx, w, c):
        return m.step_loglik(self.s[idx], self.a[idx], w, c, self._k(idx))

    def loglik(self, m, T_env, idx, w, c):
        return m.loglik(self.s[idx], self.a[idx], w, c, self._k(idx))

    @staticmethod
    def alpha(m):
        return float(m.alpha)

    def kin_model(self, m):
        return getattr(m, "kin_model", None)

    def arrays(self, m, tag):
        return {f"logp_tr_{tag}": m.logp_tr, f"pis_{tag}": m.pis, f"r_{tag}": m.r}


def heldout_scores(bk, T_env, model0, models, tr, te, kerns_te, eval_mask, valid=None):
    """
    Per-step predictive LLs on te for the independent and each coupled model; context from train.
    valid (N, T): real steps (padded steps of segment_policy: variable are excluded from every score).
    """
    out = {}
    l0 = bk.step_ll_indep(model0, T_env, te)
    for vname, (m, (w_tr, c_tr)) in models.items():
        post = getattr(bk, "context_posteriors", bk.posteriors)            # scales backends: ctx_source
        g_tr = post(m, T_env, tr, w_tr, c_tr)
        w_te, c_te = PartnerContext.context(kerns_te[vname], g_tr)
        l1 = bk.step_ll(m, T_env, te, w_te, c_te)
        msk = eval_mask[te][:, None] & (np.ones_like(w_te, dtype=bool) if valid is None else valid[te])
        conc = msk & (w_te > 0)
        d = l1 - l0
        ex = bk.extra(m) if hasattr(bk, "extra") else {}
        out[vname] = dict(alpha=bk.alpha(m), **ex, test_ll_indep=float(l0[msk].mean()),
                          test_ll_coupled=float(l1[msk].mean()),
                          d_total=float(d[msk].sum()), d_per_step=float(d[msk].mean()),
                          d_conc=float(d[conc].sum()) if conc.any() else 0.0,
                          d_conc_per_step=float(d[conc].mean()) if conc.any() else np.nan,
                          n_test_steps=int(msk.sum()), n_conc_steps=int(conc.sum()))
    return out


def run_cv(bk, conf, pc, K, variants, times=None, null_id=-1, base_cache=None, verbose=True):
    """One leave-one-animal-out pass (observed calendar if times is None, else shifted). Returns rows."""
    groups = np.asarray(bk.data["groups"]).astype(str)
    eval_mask = np.ones(len(groups), bool) if conf["eval_include_aug"] else original_mask(pc)
    animals = sorted(np.unique(groups))
    if conf.get("heldout_animals"):  # short ids ("10826") or full group names ("SWIRL_WHALE_v1_10826")
        want = [str(k) for k in conf["heldout_animals"]]
        sel = [g for g in animals if any(g == k or g.endswith("_" + k) or ("_" + k + "_") in g for k in want)]
        miss = [k for k in want if not any(g == k or g.endswith("_" + k) or ("_" + k + "_") in g for g in animals)]
        if miss:
            print(f"WARNING: heldout_animals not found in the data: {miss}")
        if not sel:
            raise ValueError(f"heldout_animals {want} matched none of {animals}")
        animals = sel
    rows = []
    for kf, an in enumerate(animals):
        te = np.where(groups == str(an))[0]
        tr = np.where(groups != str(an))[0]
        ck = (K, an)
        if base_cache is not None and ck in base_cache:
            T_env, m0, ll0 = base_cache[ck]
        else:
            T_env = bk.fold_env(tr)
            m0, ll0 = bk.fit_independent(K, T_env, tr, kf)
            if base_cache is not None:
                base_cache[ck] = (T_env, m0, ll0)
        models, kerns_te = {}, {}
        for vname, v in variants.items():
            kern_tr = pc.build(tr, tr, times=times, **v)
            kerns_te[vname] = pc.build(te, tr, times=times, **v)
            models[vname] = bk.fit_coupled(m0, T_env, tr, kern_tr)
        sc = heldout_scores(bk, T_env, m0, models, tr, te, kerns_te, eval_mask, valid=pc.valid)
        for vname, r in sc.items():
            m, (w_tr, c_tr) = models[vname]
            rows.append(dict(model=bk.name, K=K, fold=kf, heldout=an, variant=vname, null_id=null_id,
                             train_ll_indep=ll0, train_ll_coupled=bk.loglik(m, T_env, tr, w_tr, c_tr), **r))
            if verbose:
                print(f"{bk.name} K={K} hold {an} {vname:9s} null {null_id:3d}: alpha {r['alpha']:.3f} "
                      f"dLL/step {r['d_per_step']:+.4f} (context steps {r['n_conc_steps']}, "
                      f"dLL there {r['d_conc_per_step']:+.4f})")
    return rows


def synchrony(pc, gamma, variant, times=None, mask=None):
    """Mean expected agreement sum_k gamma_i c_i over steps with context (full-data independent posteriors)."""
    idx = np.arange(pc.N)
    kern = pc.build(idx, idx, times=times, **variant)
    w, c = PartnerContext.context(kern, gamma)
    m = (w > 0) & pc.valid if mask is None else (w > 0) & pc.valid & mask[:, None]
    if not m.any():
        return np.nan, 0
    agree = (gamma * c).sum(-1)
    return float(agree[m].mean()), int(m.sum())


def _kin_final(bk, m, gamma, tag):
    """Kinematic parameters, labels and per-step P(exploit) of a fitted model (empty without kinematics)."""
    km = bk.kin_model(m)
    if km is None:
        return {}, None
    cfg = kin_conf(bk.yml_conf)
    lab = labels_from_conf(km, cfg)
    pe, px = p_exploit(gamma, lab)
    out = {f"p_exploit_{tag}": pe, f"p_explore_{tag}": px, f"kin_labels_{tag}": np.asarray(lab["label"])}
    out.update(km.params(prefix=f"kin_{tag}_"))
    return out, lab


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_backend(yml_conf, conf):
    b = str(conf.get("backend", "hiavi")).lower()
    if b == "hiavi":
        return HiaviBackend(yml_conf, conf)
    if b == "swirl":
        from naturenet.models.irl.swirl_multiagent import SwirlBackend
        return SwirlBackend(yml_conf, conf)
    if b in ("swirl_scales", "hiql_scales"):          # decision-scale toggles + coupling (scales_multiagent)
        from naturenet.models.irl.scales_multiagent import ScalesHiqlBackend, ScalesSwirlBackend
        return (ScalesSwirlBackend if b == "swirl_scales" else ScalesHiqlBackend)(yml_conf, conf)
    raise ValueError(f"multiagent.backend must be hiavi | swirl | swirl_scales | hiql_scales, got {b}")


def run_multiagent(yml_conf, Ks=None, skip_null=False, verbose=True, backend=None, concurrency_only=False):
    yml_conf = expand_inputs(yml_conf)
    conf = ma_conf(yml_conf)
    if backend is not None:
        conf["backend"] = backend
        if not (yml_conf.get("multiagent") or {}).get("ma_subdir"):
            conf["ma_subdir"] = "multiagent" if backend == "hiavi" else f"multiagent_{backend}"
    bk = make_backend(yml_conf, conf)
    out_dir = os.path.join(yml_conf["out_dir"], conf["ma_subdir"])
    os.makedirs(out_dir, exist_ok=True)
    data = bk.prepare()
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
    if concurrency_only:
        return {"concurrency": conc}

    null_shifts = [draw_shifts(animals, rng, win, conf["null_min_shift_days"]) for _ in range(int(conf["n_null"]))]
    sync_shifts = [draw_shifts(animals, rng, win, conf["null_min_shift_days"])
                   for _ in range(int(conf["n_null_sync"]))]
    rows, sync_rows = [], []
    summary = {"model": bk.name, "kinematics": bk.kin is not None, "window_days": win, "animals": animals, "K": {}}
    kin_tables = []
    for K in (Ks or conf["Ks"]):
        K = int(K)
        cache = {}
        rows_k = run_cv(bk, conf, pc, K, variants, base_cache=cache, verbose=verbose)
        if not skip_null:
            nv = {v: variants[v] for v in conf["null_variants"] if v in variants}
            for r, sh in enumerate(null_shifts):
                rows_k += run_cv(bk, conf, pc, K, nv, times=pc.shifted_times(sh, win), null_id=r,
                                 base_cache=cache, verbose=False)
                if verbose:
                    print(f"K={K}: null replicate {r + 1}/{len(null_shifts)} done")
        rows += rows_k

        # full-data fits
        all_idx = np.arange(pc.N)
        T_all = bk.fold_env(all_idx)
        m0, ll0 = bk.fit_independent(K, T_all, all_idx, None)
        g0 = bk.posteriors_indep(m0, T_all, all_idx)
        sv = variants[conf["sync_variant"]]
        a_obs, n_obs = synchrony(pc, g0, sv, mask=orig)
        a_null = [synchrony(pc, g0, sv, times=pc.shifted_times(sh, win), mask=orig)[0] for sh in sync_shifts]
        pbar = g0[orig[:, None] & pc.valid].reshape(-1, K).mean(0)
        chance = float((pbar ** 2).sum())
        a_null = np.asarray([x for x in a_null if np.isfinite(x)])
        p_sync = float((1 + np.sum(a_null >= a_obs)) / (1 + len(a_null))) if len(a_null) else np.nan
        sync_rows += [dict(K=K, kind="observed", agreement=a_obs, n_steps=n_obs)] + \
                     [dict(K=K, kind="time_shift", agreement=x, n_steps=np.nan) for x in a_null]
        final = dict(times=pc.times, lat=pc.lat, lon=pc.lon, animal=pc.animal, key=pc.key, original=orig,
                     valid=pc.valid,
                     gamma_indep=g0, train_ll_indep=ll0, model=bk.name, **bk.arrays(m0, "indep"))
        kf0, lab0 = _kin_final(bk, m0, g0, "indep")
        final.update(kf0)
        k_all = bk._k(all_idx)
        if k_all is not None:
            final.update(kin_step=k_all.step, kin_turn=k_all.turn)
        if lab0 is not None:
            kin_tables.append(lab0.assign(K=K, model=bk.name, fit="indep"))
        for vname, v in variants.items():
            kern = pc.build(all_idx, all_idx, **v)
            m, (w, c) = bk.fit_coupled(m0, T_all, all_idx, kern)
            g = bk.posteriors(m, T_all, all_idx, w, c)
            if hasattr(bk, "extra"):
                final.update({f"{k}_{vname}": v for k, v in bk.extra(m).items()})
            final.update({f"gamma_{vname}": g, f"w_{vname}": w, f"alpha_{vname}": bk.alpha(m),
                          f"train_ll_{vname}": bk.loglik(m, T_all, all_idx, w, c), **bk.arrays(m, vname)})
            kf, lab = _kin_final(bk, m, g, vname)
            final.update(kf)
            if lab is not None:
                kin_tables.append(lab.assign(K=K, model=bk.name, fit=vname))
        np.savez(os.path.join(out_dir, f"ma_final_K{K}.npz"), **final)

        # summary
        df = pd.DataFrame(rows_k)
        sk = {"sync": dict(observed=a_obs, n_steps=n_obs, null_mean=float(np.mean(a_null)) if len(a_null) else None,
                           null_sd=float(np.std(a_null)) if len(a_null) else None, chance=chance, p=p_sync),
              "variants": {}}
        for vname in variants:
            ob = df[(df["variant"] == vname) & (df["null_id"] < 0)]
            nu = df[(df["variant"] == vname) & (df["null_id"] >= 0)]
            pooled_ = float(ob["d_total"].sum())
            n_steps = int(ob["n_test_steps"].sum())
            wv = final[f"w_{vname}"]
            sel = (wv > 0) & orig[:, None] & pc.valid
            ent = dict(pooled_d_total=pooled_, pooled_d_per_step=pooled_ / max(n_steps, 1),
                       pooled_d_conc=float(ob["d_conc"].sum()), n_conc_steps=int(ob["n_conc_steps"].sum()),
                       alpha_median=float(ob["alpha"].median()),
                       **({"beta_median": float(ob["beta"].median())} if "beta" in ob else {}),
                       animals_improved=int((ob["d_total"] > 0).sum()), n_animals=int(len(ob)),
                       full_data_alpha=float(final[f"alpha_{vname}"]),
                       # alpha * mean w on steps with context: the mixing weight actually applied (alpha can sit
                       # at its bound when w is small, e.g. under the proximity kernel)
                       effective_coupling=float(final[f"alpha_{vname}"] * wv[sel].mean()) if sel.any() else 0.0)
            if len(nu):
                null_pooled = nu.groupby("null_id")["d_total"].sum().to_numpy()
                null_alpha = nu.groupby("null_id")["alpha"].median().to_numpy()
                ent.update(null_pooled_mean=float(null_pooled.mean()), null_pooled_sd=float(null_pooled.std()),
                           p_gain=float((1 + np.sum(null_pooled >= pooled_)) / (1 + len(null_pooled))),
                           p_alpha=float((1 + np.sum(null_alpha >= ent["alpha_median"])) / (1 + len(null_alpha))),
                           n_null=int(len(null_pooled)))
                if "beta" in nu and "beta_median" in ent:     # partner hazard term (decision-scale backends)
                    null_beta = nu.groupby("null_id")["beta"].median().to_numpy()
                    ent.update(null_beta_mean=float(null_beta.mean()),
                               p_beta=float((1 + np.sum(null_beta >= ent["beta_median"])) / (1 + len(null_beta))))
            if "beta" in ob:
                ent["full_data_beta"] = float(final.get(f"beta_{vname}", np.nan))
            sk["variants"][vname] = ent
        if lab0 is not None:
            sk["kinematic_labels_indep"] = list(map(str, lab0["label"]))
        summary["K"][str(K)] = sk
        print(json.dumps(sk, indent=1))

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ma_cv.csv"), index=False)
    pd.DataFrame(sync_rows).to_csv(os.path.join(out_dir, "ma_sync.csv"), index=False)
    if kin_tables:
        pd.concat(kin_tables).to_csv(os.path.join(out_dir, "ma_kinematics.csv"), index=False)
    with open(os.path.join(out_dir, "ma_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", out_dir)
    return summary


def cli(default_backend="hiavi"):
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--K", type=int, nargs="*", default=None)
    ap.add_argument("--skip_null", action="store_true")
    ap.add_argument("--concurrency_only", action="store_true",
                    help="only write ma_concurrency.csv / ma_pairs.npz (data-sufficiency check, e.g. new species)")
    ap.add_argument("--backend", default=None, help="hiavi | swirl | swirl_scales | hiql_scales (default: "
                    "multiagent.backend or " +
                    default_backend + ")")
    args = ap.parse_args()
    conf = read_conf(args.yaml)
    b = args.backend or (conf.get("multiagent") or {}).get("backend") or default_backend
    run_multiagent(conf, Ks=args.K, skip_null=args.skip_null, backend=b, concurrency_only=args.concurrency_only)


if __name__ == "__main__":
    cli("hiavi")
