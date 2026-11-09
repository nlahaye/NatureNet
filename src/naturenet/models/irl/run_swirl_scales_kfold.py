"""
K-fold runner for the SWIRL decision-scale variants (swirl_scales): order-N policies, switching history,
intention durations and hierarchical regimes, each a toggle (default off).

    python -m naturenet.models.irl.run_swirl_scales_kfold -y config/swirl/swirl_scales.yaml [--fold 0 1]
                                                          [--variant N2 N1_dur-logi20] [--check]

Uses the run_swirl_kfold data preparation, grouped folds, per-fold T (train animals) and the per-fold ARHMM
init (scales.init: arhmm | sticky | auto), then the S-1 schedule per variant:
    stage 1: no temperature, `epochs`      stage 2: temps_final, `epochs_2`
Order-N variants warm-start from the fitted order-(N-1) variant with the same latent toggles (the new
history increments start at 0, so the child starts at the parent's policy) and run stage 2 only
(scales.warm_epochs, default epochs_2). With a `kinematics` block the per-intention kinematic emissions are
fitted too and LL is the comparable LL (joint - pooled kinematic null), as in run_swirl_kfold.

Outputs in out_dir (run_uid_v = <run_uid>_SC_<variant tag>):
  <run_uid_v>_<K>_<seed>_scales_KF_<k>.npz                         params, context tree, diagnostics
  <run_uid_v>_<K>_<seed>_naturenet_S1_iter2_LL1[_test]_KF_<k>.npz   LL (per step), step_ll, gamma, gamma_r,
                                                                   jax_path_vmap (posterior-argmax z), seq meta
  <run_uid>_scales_cv.csv                                          one row per (variant, fold)
--check prints the segment-length / context-coverage report and exits (use it to choose order, D and R).

HIQL / HIAVI emission (scales.emission: hiql): the same variants with tabular HIAVI policies per context
(closed-form M-step, Dirichlet back-off scales.hiql.backoff towards the parent context; order 1 + all off =
HIAVI), state-free intention switching (transition_input none, as HIAVI) and no temperatures. Init per fold:
hiavi_restarts random HIAVI inits (HIAVI._init: random policies, sticky Lambda), best training LL kept; no ARHMM.
Schedule: one EM run of hiavi_max_iters iterations, relative tol scales.hiql_tol (default 1e-7, as the
multi-agent hiql_scales backend); order-N children warm-start from the fitted parent (one fit). Unset
scales.hiql keys default to hiavi_pseudocount / hiavi_discount / hiavi_iavi_tol / hiavi_iavi_max_iters, so N1
matches run_hiavi_kfold (hiavi_align: swirl). IAVI rewards on the context MDP (sc_iavi_r / sc_iavi_q) are added
after the fit (scales.hiql_iavi: False skips). Variant tags and CSV rows are the same as for SWIRL (N1, N2, ...)
with an `emission` column; use a separate run_uid (or out_dir) for HIQL runs - the runner refuses to mix
emissions in one CSV.
Multi-agent (partner-coupled) fits of the same variants: run_scales_multiagent / scales_multiagent.
"""
import argparse
import copy
import itertools
import os
import time

import numpy as np
import pandas as pd
import jax.numpy as jnp

from naturenet.models.irl import swirl_scales as sc
from naturenet.models.irl.run_swirl_kfold import (
    _fold_iter, _init_kin_model, _temps, default_final_temps, fold_trans_prob, load_arhmm, prepare_data,
    swirl_kinematics)
from naturenet.models.irl.swirl_data import init_rewards, load_swirl_dataset, seq_lengths, seq_meta_arrays, step_mask
from naturenet.models.irl.kinematics import comparable_ll, kin_conf, pooled
from naturenet.models.irl.swirl_multiagent import sticky_init
from naturenet.models.irl.hiavi_model import HIAVI
from naturenet.models.irl.run_hiavi_kfold import _seed, hiavi_settings

from sit_fuse.utils import read_yaml

LATENT_KEYS = ("switch_history", "durations", "hierarchy")
SHARED_KEYS = ("reward_param", "reward_l2", "context_min_count", "context_source", "context_t_support",
               "reward_optimizer", "reward_tol", "emission", "transition_input", "hiql")
EMISSIONS = ("swirl", "hiql")


def scales_emission(yml_conf):
    """Run-wide emission: scales.emission (swirl | hiql; default swirl)."""
    e = str((yml_conf.get("scales") or {}).get("emission", "swirl")).lower()
    if e in ("hiavi",):
        e = "hiql"
    if e not in EMISSIONS:
        raise ValueError(f"scales.emission must be swirl | hiql, got {e!r}")
    return e


def _hiql_block(yml_conf):
    """scales.hiql with unset keys taken from the hiavi_* settings (as the multi-agent hiql_scales backend)."""
    st = hiavi_settings(yml_conf)
    if st["mstep"] != "soft":
        raise ValueError("scales.emission hiql supports hiavi_mstep: soft only")
    hq = dict((yml_conf.get("scales") or {}).get("hiql") or {})
    for k in ("pseudocount", "discount", "iavi_tol", "iavi_max_iters"):
        hq.setdefault(k, st[k])
    return hq


def _tag(cfg):
    """Variant tag without the emission prefix (one emission per run, so N1 is the baseline for either)."""
    t = cfg.tag()
    pre = cfg.emission.upper() + "_"
    return t[len(pre):] if cfg.emission != "swirl" and t.startswith(pre) else t


def _np_T(T):
    T = T.todense() if hasattr(T, "todense") else T
    return np.asarray(T, dtype=float)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def _toggle(value, default_block):
    if value in (None, False, "off", "none"):
        return None
    if value in (True, "on"):
        return copy.deepcopy(default_block) if default_block else True
    if isinstance(value, dict):
        out = copy.deepcopy(default_block or {})
        out.update(value)
        return out
    return value                      # e.g. "lags+decay" for switch_history


def scales_variants(yml_conf):
    """[(tag, conf dict)] from scales.variants (explicit) and / or scales.grid; the baseline (N1) is first."""
    s = yml_conf.get("scales") or {}
    defaults = s.get("defaults") or {}
    shared = {k: s[k] for k in SHARED_KEYS if k in s}
    emission = scales_emission(yml_conf)
    shared["emission"] = emission
    if emission == "hiql":
        shared["hiql"] = _hiql_block(yml_conf)
    confs = [dict(order=1)]
    grid = s.get("grid")
    if grid:
        orders = list(np.atleast_1d(grid.get("order", [1])))
        axes = [list(grid.get(k, ["off"])) if isinstance(grid.get(k, ["off"]), (list, tuple)) else [grid[k]]
                for k in LATENT_KEYS]
        for N, *vals in itertools.product(orders, *axes):
            c = dict(order=int(N))
            for k, v in zip(LATENT_KEYS, vals):
                t = _toggle(v, defaults.get(k))
                if t is not None:
                    c[k] = t
            confs.append(c)
    for v in s.get("variants") or []:
        v = dict(v)
        c = dict(order=int(v.pop("order", 1)))
        for k in LATENT_KEYS:
            if k in v:
                t = _toggle(v.pop(k), defaults.get(k))
                if t is not None:
                    c[k] = t
        c.update({k: v[k] for k in v if k != "name"})
        if str(c.get("emission", emission)).lower().replace("hiavi", "hiql") != emission:
            raise ValueError(f"variant {v.get('name', c)} sets emission {c['emission']!r}; this run is {emission!r} "
                             "(scales.emission). Use a separate config / run_uid per emission.")
        c.pop("emission", None)
        confs.append(c)
    out, seen = [], set()
    for c in confs:
        c = {**shared, **c, "seed": int(yml_conf.get("seed", 0))}
        tag = _tag(sc.ScalesConfig.from_dict(c))
        if tag not in seen:
            seen.add(tag)
            out.append((tag, c))
    return sorted(out, key=lambda tc: (sc.ScalesConfig.from_dict(tc[1]).order, tc[0] != "N1"))


def _latent_key(cfg):
    t = _tag(cfg)
    return t.split("_", 1)[1] if "_" in t else ""


def _parent_tag(cfg):
    if cfg.order <= 1:
        return None
    rest = _latent_key(cfg)
    return f"N{cfg.order - 1}" + ("_" + rest if rest else "")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def segment_report(data, variants, S):
    """Sequence-length / context-coverage report for choosing order, max_duration and n_regimes."""
    rows = np.asarray(data["states"])
    N, L = rows.shape
    T = L - 2
    lens = seq_lengths(data)
    msk = step_mask(data, 1)
    n_steps = int(msk.sum())
    print(f"{N} sequences x {T} steps = {N * T} steps; {len(set(np.asarray(data['groups'])))} animals; S = {S}"
          if n_steps == N * T else
          f"{N} sequences, {n_steps} real steps (variable lengths: {int(msk.sum(1).min())}-{T} steps, "
          f"median {np.median(msk.sum(1)):.0f}; {1 - n_steps / (N * T):.1%} padding); "
          f"{len(set(np.asarray(data['groups'])))} animals; S = {S}")
    real = np.arange(L - 1)[None] < (lens - 1)[:, None]
    stay = float(np.mean((rows[:, 1:] == rows[:, :-1])[real]))
    print(f"state self-transition fraction {stay:.3f}")
    for tag, c in variants:
        cfg = sc.ScalesConfig.from_dict(c)
        msg = [tag]
        if cfg.order > 1:
            tree = sc.ContextTree.build(rows, S, np.ones((S, 1, S)) / S, order=cfg.order,
                                        min_count=cfg.context_min_count, lengths=lens)
            ctx = tree.contexts(rows)
            dep = tree.depth[ctx]
            full = np.minimum(cfg.order, np.arange(T)[None] + 2)
            msg.append(f"contexts H = {tree.H} (deep {tree.H - S}); steps at full depth "
                       f"{np.mean((dep == full)[msk]):.2f}")
            msg.append(f"steps / deep context {n_steps / max(tree.H - S, 1):.1f}")
        Tm = int(np.median(msk.sum(1)))                               # typical sequence length (= T if fixed)
        if cfg.durations.enabled:
            D = cfg.D
            msg.append(f"D = {D}: {'OK' if D <= Tm // 2 else 'WARNING D > T/2'}")
        if cfg.hierarchy.enabled:
            msg.append(f"R = {cfg.R}: steps per sequence {Tm} ({'short for regimes' if Tm < 50 else 'OK'})")
        if cfg.history.enabled:
            msg.append(f"history {'+'.join(cfg.history.kind)}")
        print("  " + "; ".join(msg))


# ---------------------------------------------------------------------------
# Fit one fold
# ---------------------------------------------------------------------------

def _base_init(yml_conf, scales_conf, K, S, kf_ind):
    if scales_emission(yml_conf) == "hiql":
        return None                                                  # HIAVI random inits per restart
    mode = scales_conf.get("init", "auto")
    out_dir, run_uid, seed = yml_conf["out_dir"], yml_conf["run_uid"], int(yml_conf["seed"])
    if mode in ("arhmm", "auto"):
        try:
            return load_arhmm(out_dir, run_uid, int(yml_conf.get("n_hidden_init", K)), seed, kf_ind)
        except FileNotFoundError:
            if mode == "arhmm":
                raise
            print("No ARHMM init for fold", kf_ind, "- using sticky_init")
    b = sticky_init(K, S, seed, sticky=float(scales_conf.get("sticky", 0.95)))
    return (jnp.asarray(b[0]), jnp.asarray(b[1]), tuple(jnp.asarray(w) for w in b[2]))


def _names(yml_conf, tag, kf_ind):
    out_dir, run_uid, seed, K = yml_conf["out_dir"], yml_conf["run_uid"], yml_conf["seed"], int(yml_conf["n_hidden"])
    uid = f"{run_uid}_SC_{tag}"
    f = lambda s: os.path.join(out_dir, f"{uid}_{K}_{seed}{s}_KF_{kf_ind}.npz")  # noqa: E731
    return uid, f("_scales"), f("_naturenet_S1_iter2_LL1"), f("_naturenet_S1_iter2_LL1_test")


def _tree(cfg, data, T, tr, S):
    rows = np.asarray(data["states"])
    lens = seq_lengths(data)
    src, src_len = (rows, lens) if cfg.context_source == "all" else (rows[tr], lens[tr])
    extra = None
    if cfg.context_t_support and cfg.order >= 2:
        from naturenet.models.irl.swirl_utils import compute_prev_state_map
        psm = compute_prev_state_map(T, S, data["n_actions"], observed_pairs=[(data["x"], data["x_prev"])])
        extra = [(int(p), s) for s in range(S) for p in psm.get(s, [])]
    return sc.ContextTree.build(src, S, T, order=cfg.order, min_count=cfg.context_min_count, extra_pairs=extra,
                                lengths=src_len)


def fit_variant_fold(yml_conf, data, T, tr, te, tag, conf, base, R_start, temps, kin, kin_null, km0, parent,
                     kf_ind):
    s = yml_conf.get("scales") or {}
    cfg = sc.ScalesConfig.from_dict(conf)
    S = data["n_states"]
    rows, a = np.asarray(data["states"]), np.asarray(data["a"])
    tol = yml_conf.get("em_tol")
    epochs, epochs_2 = int(yml_conf.get("epochs", 50)), int(yml_conf.get("epochs_2", 30))
    t0 = time.time()
    tree = _tree(cfg, data, T, tr, S)
    dtr = sc.scales_data(rows, a, tree, cfg.history, S, kin=kin, idx=tr)
    dte = None if te is None or len(te) == 0 else sc.scales_data(rows, a, tree, cfg.history, S, kin=kin, idx=te)
    if cfg.emission == "hiql":
        r = _fit_hiql(yml_conf, s, cfg, tree, dtr, T, tr, kin, parent, kf_ind)
    else:
        r = _fit_swirl(yml_conf, s, cfg, tree, dtr, base, R_start, temps, km0, parent, tol, epochs, epochs_2)
    sec = time.time() - t0
    return _finish_fold(yml_conf, data, cfg, tree, r, dtr, dte, tr, te, tag, kin, kin_null, kf_ind, sec)


def _fit_hiql(yml_conf, s, cfg, tree, dtr, T, tr, kin, parent, kf_ind):
    """HIQL emission: best of hiavi_restarts HIAVI inits (training LL), or one warm-started fit for order N."""
    from naturenet.models.irl.kinematics import KinematicEmissions
    st = hiavi_settings(yml_conf)
    K = int(yml_conf["n_hidden"])
    tol = float(s.get("hiql_tol", 1e-7))
    iters = int(st["max_iters"])
    kcfg = kin_conf(yml_conf)
    if parent is not None and s.get("warm_start", True):
        r_par, tree_par = parent
        p = sc.init_params(cfg, tree, (r_par["params"]["logpi0"][0], r_par["params"]["log_Ps"][0], None),
                           np.zeros((K, tree.S, tree.A)), dtr["phi"].shape[-1])
        p = sc.warm_start(p, tree, r_par, tree_par)
        print("  warm start from the fitted parent (HIQL)")
        r = sc.em_train_scales(p, cfg, tree, dtr, temps=None, kin_model=r_par.get("kin_model"),
                               iters=int(s.get("warm_epochs", iters)), tol=tol)
    else:
        T_np = _np_T(T)
        best, best_ll = None, -np.inf
        for rs in range(int(yml_conf.get("hiavi_restarts", 10))):
            ss = _seed(yml_conf["seed"], kf_ind, rs)
            hm = HIAVI(K, T_np, seed=ss, sticky=st["sticky"])
            lp0, lP, lpis = hm._init()
            km = None
            if kin is not None and kcfg is not None:
                km = KinematicEmissions.from_conf(K, kcfg).init_quantiles(kin[tr], rng=hm.rng)
            seed = int(np.asarray(ss.generate_state(1))[0]) % (2 ** 31 - 1)
            p = sc.init_params(cfg, tree, (lp0, lP, None), lpis, dtr["phi"].shape[-1], seed=seed)
            try:
                rr = sc.em_train_scales(p, cfg, tree, dtr, temps=None, kin_model=km, iters=iters, tol=tol)
            except FloatingPointError as e:
                print(f"  restart {rs} failed: {e}")
                continue
            ll = sc.scales_eval(rr, tree, dtr)["LL"]
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = rr, ll
        if best is None:
            raise RuntimeError(f"all HIQL restarts failed (fold {kf_ind})")
        r = best
    if s.get("hiql_iavi", True):
        r = sc.hiql_finalize(r, tree, dtr)
    return r


def _fit_swirl(yml_conf, s, cfg, tree, dtr, base, R_start, temps, km0, parent, tol, epochs, epochs_2):
    p = sc.init_params(cfg, tree, base, R_start, dtr["phi"].shape[-1])
    km = km0
    if parent is not None and s.get("warm_start", True):
        r_par, tree_par = parent
        p = sc.warm_start(p, tree, r_par, tree_par)
        km = r_par.get("kin_model") or km0
        print(f"  warm start from the fitted parent; stage 2 only")
        r = sc.em_train_scales(p, cfg, tree, dtr, temps=temps, kin_model=km,
                               iters=int(s.get("warm_epochs", epochs_2)), tol=tol)
    else:
        r = sc.em_train_scales(p, cfg, tree, dtr, temps=None, kin_model=km, iters=epochs, tol=tol)
        r = sc.em_train_scales(r["params"], cfg, tree, dtr, temps=temps, kin_model=r["kin_model"], iters=epochs_2,
                               tol=tol)
    return r


def _finish_fold(yml_conf, data, cfg, tree, r, dtr, dte, tr, te, tag, kin, kin_null, kf_ind, sec):
    uid, f_par, f_ll, f_llt = _names(yml_conf, tag, kf_ind)
    m_tr = np.asarray(sc.data_mask(dtr))                              # real (unpadded) steps
    counts = np.bincount(np.asarray(dtr["ctx"])[m_tr], minlength=tree.H)
    diag = dict(history_kl=sc.history_kl(r, tree, counts), reward_depth_share=sc.reward_depth_share(r, tree, counts),
                ctx_counts=counts, phi_names=np.asarray(dtr["phi_names"], dtype=object),
                phi_mean=np.asarray(dtr["phi"])[m_tr].mean(0)
                if dtr["phi"].shape[-1] else np.zeros(0), variant=tag, fit_sec=sec)
    np.savez(f_par, **sc.params_to_save(r, tree), **diag)
    row = dict(variant=tag, fold=kf_ind, emission=cfg.emission, order=cfg.order, latent=_latent_key(cfg) or "none",
               history="+".join(cfg.history.kind) or "off",
               durations=cfg.durations.hazard if cfg.durations.enabled else "off", D=cfg.D, R=cfg.R,
               reward_param=cfg.reward_param, H=tree.H, n_params=sc.n_params(r["params"], cfg, tree),
               n_iter=len(r["LL_list"]), fit_sec=round(sec, 1))
    for split, idx, d, path in (("train", tr, dtr, f_ll), ("test", te, dte, f_llt)):
        if d is None:
            continue
        ev = sc.scales_eval(r, tree, d)
        ll = ev["LL"]
        extra = {}
        if kin is not None:
            eva = sc.scales_eval(r, tree, d, use_kin=False)
            ll, joint, null = comparable_ll(ev["LL_total"], kin_null, kin[idx], ev["n_steps"])
            extra = dict(LL_joint=joint, LL_kin_null=null, LL_action=eva["LL"])
            row[f"LL_{split}_action"] = eva["LL"]
        dep = np.where(ev["mask"], tree.depth[np.asarray(d["ctx"])], 0)  # 0 on padded steps
        np.savez(path, LL=ll, step_ll=ev["step_ll"], gamma=ev["gamma"], gamma_r=ev["gamma_r"],
                 jax_path_vmap=ev["path"], path_kind="posterior_argmax", ctx_depth=dep, **extra,
                 **seq_meta_arrays(data, idx))
        row[f"LL_{split}"] = ll
        row[f"n_{split}_steps"] = ev["n_steps"]
        row[f"ctx_depth_{split}"] = float(dep[ev["mask"]].mean())
    print(f"  {tag} fold {kf_ind}: train {row['LL_train']:.4f}" +
          (f"  test {row['LL_test']:.4f}" if "LL_test" in row else "") + f"  ({sec:.0f} s, {row['n_params']} params)")
    return r, tree, row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _csv_path(yml_conf):
    return os.path.join(yml_conf["out_dir"], yml_conf["run_uid"] + "_scales_cv.csv")


def _write_rows(yml_conf, rows):
    path = _csv_path(yml_conf)
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        e_old = set(old["emission"].dropna()) if "emission" in old else {"swirl"}
        e_new = set(new["emission"]) if "emission" in new else {"swirl"}
        if e_old and e_old != e_new:
            raise ValueError(f"{path} holds {sorted(e_old)} rows; this run is {sorted(e_new)}. Use a separate "
                             "run_uid (or out_dir) per emission so N1 baselines and files do not collide.")
        key = set(zip(new["variant"], new["fold"]))
        old = old[[(v, f) not in key for v, f in zip(old["variant"], old["fold"])]]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["order", "variant", "fold"]).to_csv(path, index=False)
    return path


def run_scales(yml_conf, folds_to_run=None, only=None, check=False):
    data = prepare_data(yml_conf, load_swirl_dataset(yml_conf, allow_variable=True))
    s = yml_conf.get("scales") or {}
    if (yml_conf.get("multiagent") or {}).get("enabled", False):
        raise NotImplementedError("this runner is single-agent (k-fold over sequences); multi-agent decision-scale "
                                  "runs use run_scales_multiagent (leave-one-animal-out with partner context)")
    K, S, A, seed = int(yml_conf["n_hidden"]), data["n_states"], data["n_actions"], int(yml_conf["seed"])
    variants = scales_variants(yml_conf)
    if only:
        keep = set(only)
        variants = [v for v in variants if v[0] in keep]
    print("Variants:", [v[0] for v in variants])
    segment_report(data, variants, S)
    if check:
        return None
    os.makedirs(yml_conf["out_dir"], exist_ok=True)
    emission = scales_emission(yml_conf)
    if emission == "hiql":
        temps, R_start = None, None
        print("Emission: HIQL (HIAVI policies per context);", int(yml_conf.get("hiavi_restarts", 10)),
              "restarts per fold; no ARHMM init, no temperatures")
    else:
        temps = _temps(yml_conf, "temps_final", K, default_final_temps(K))
        _, R_start = init_rewards(K, S, A, seed)
    kcfg = kin_conf(yml_conf)
    kin = swirl_kinematics(yml_conf, data, kcfg) if kcfg is not None else None
    rows_out = []
    for kf_ind, tr, te in _fold_iter(yml_conf, data, folds_to_run):
        print("Scales fold", kf_ind, ":", len(tr), "train /", len(te), "test sequences")
        T = fold_trans_prob(yml_conf, data, tr)
        base = _base_init(yml_conf, s, K, S, kf_ind)
        kin_null = km0 = None
        if kin is not None:
            kin_null = pooled(kin[tr], kcfg)
            km0 = _init_kin_model(yml_conf, kcfg, K, kin[tr], getattr(base, "kin", None), seed)
        fitted = {}
        for tag, conf in variants:
            cfg = sc.ScalesConfig.from_dict(conf)
            _, f_par, _, f_llt = _names(yml_conf, tag, kf_ind)
            if os.path.exists(f_par) and os.path.exists(f_llt) and not yml_conf.get("overwrite", False):
                print(f"  {tag} fold {kf_ind} exists; loading")
                fitted[tag] = sc.load_scales(f_par, T)
                continue
            parent = fitted.get(_parent_tag(cfg))
            r, tree, row = fit_variant_fold(yml_conf, data, T, tr, te, tag, conf, base, R_start, temps, kin,
                                            kin_null, km0, parent, kf_ind)
            fitted[tag] = (r, tree)
            rows_out.append(row)
            _write_rows(yml_conf, rows_out)
    path = _write_rows(yml_conf, rows_out) if rows_out else _csv_path(yml_conf)
    print("CV table:", path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--fold", type=int, nargs="*", default=None)
    ap.add_argument("--variant", nargs="*", default=None, help="only these variant tags (parents must exist)")
    ap.add_argument("--check", action="store_true", help="segment / context report only")
    args = ap.parse_args()
    run_scales(read_yaml(args.yaml), folds_to_run=args.fold, only=args.variant, check=args.check)
