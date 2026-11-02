"""
HIAVI (multi-intention inverse Q-learning, Zhu et al. 2024; https://github.com/nrgrp/hiql) K-fold CV and
final (holdout) run on the SWIRL data pipeline, so HIAVI and SWIRL use the same sequences, the same
animal-grouped folds and the same LL definition, and the outputs feed the same plotting / time-series code.

Per fold: T (S, A, S') from the TRAIN animals, `hiavi_restarts` random restarts, best restart chosen by
TRAIN LL (the original picked it by TEST LL, which biases the CV score), then train / test LL and Viterbi
paths. LL = forward log p(a | s) / (N * T) (per step; comparable to the SWIRL LL files).

Alignment (hiavi_align):
  swirl (default): s = states[:, 1:-1], a = actions[:, 1:] (the SWIRL emission pairs; latent t <-> fix
                   seq_start + 1 + t, so LLs and paths line up with SWIRL);
  full:            s = states[:, :-1], a = actions (one more step per sequence; latent t <-> seq_start + t).

Outputs (base = <out_dir>/<run_uid>_<K>_<seed>, tag = _KF_<k> or "" for the final run):
  base_hiavi<tag>.npz          logp_init, logp_tr, pis, r, q (K, S, A), ll_history, restart_train_ll, ...
  base_hiavi_LL<tag>.npz       LL, learnt_zs, seq_key / seq_group / seq_seg / seq_start, seq_offset
  base_hiavi_LL_test<tag>.npz  same for the test animals
  <out_dir>/ll_hiavi.csv       one row per (K, run, fold): K, seed, run_uid, fold, train_ll, test_ll

Config keys (defaults): hiavi_discount 0.9, hiavi_restarts 10, hiavi_mstep soft (| sample),
hiavi_pseudocount 0, hiavi_max_iters 200, hiavi_tol 1e-6, hiavi_sticky 0.95, hiavi_iavi_tol 1e-8,
hiavi_iavi_max_iters 10000, hiavi_align swirl, plus the usual n_states, n_hidden, seed, out_dir, run_uid,
n_folds, fold_seed, folds_to_run, per_fold_trans_prob (True), fill_unobserved, holdout_*, overwrite.

Optional `kinematics` block (see kinematics.py / config/swirl/kinematics_example.yaml): per-intention step
(gamma / log-normal, zero-inflated) and turning-angle (von Mises) emissions fitted inside the EM. The LL files
then hold LL = LL_joint - LL_kin_null (comparable with action-only runs; kin null = one-intention kinematic
fit on the TRAIN animals), LL_joint, LL_kin_null, LL_action (kinematics marginalised), p_exploit / p_explore
(N, T) and kin_labels (K,); the model file holds the kinematic parameters and <model>_kinematics.csv the
per-intention table with exploit / explore / ambiguous labels.
"""
import argparse
import os

import numpy as np
import pandas as pd
import sparse

from naturenet.environment.pipeline_checks import check_folds, check_swirl_inputs, check_trans_prob
from naturenet.models.irl.hiavi_model import HIAVI
from naturenet.models.irl.kinematics import cached_kinematics, comparable_ll, kin_conf, labels_from_conf, p_exploit, pooled
from naturenet.models.irl.swirl_data import (
    align_for_swirl,
    as_list,
    get_folds,
    holdout_split,
    load_swirl_dataset,
    seq_meta_arrays,
    trans_prob_from_segments,
)

N_ACTIONS = 9


# ---------------------------------------------------------------------------
# Data / settings
# ---------------------------------------------------------------------------

def prepare_hiavi_data(yml_conf, data=None):
    """Aligned (s, a) arrays, computed once and reused across folds and (K, run)."""
    data = data if data is not None else load_swirl_dataset(yml_conf)
    align = yml_conf.get("hiavi_align", "swirl")
    if data.get("hiavi_align") == align:
        return data
    S = int(yml_conf["n_states"])
    check_swirl_inputs(data["states"], data["actions"], S, N_ACTIONS)
    if align == "swirl":
        s, _, a = align_for_swirl(data["states"], data["actions"])
        offset = 1
    elif align == "full":
        s, a, offset = np.asarray(data["states"])[:, :-1], np.asarray(data["actions"]), 0
    else:
        raise ValueError(f"hiavi_align must be swirl | full, got {align!r}")
    s, a = np.asarray(s, dtype=np.int64), np.asarray(a, dtype=np.int64)
    if s.min() < 0 or s.max() >= S or a.min() < 0 or a.max() >= N_ACTIONS:
        raise ValueError("HIAVI inputs contain invalid states / actions after alignment")
    data.update(n_states=S, n_actions=N_ACTIONS, hiavi_s=s, hiavi_a=a, hiavi_offset=offset, hiavi_align=align)
    return data


def hiavi_settings(yml_conf):
    return dict(discount=float(yml_conf.get("hiavi_discount", 0.9)),
                mstep=yml_conf.get("hiavi_mstep", "soft"),
                pseudocount=float(yml_conf.get("hiavi_pseudocount", 0.0)),
                max_iters=int(yml_conf.get("hiavi_max_iters", 200)),
                tol=float(yml_conf.get("hiavi_tol", 1e-6)),
                sticky=float(yml_conf.get("hiavi_sticky", 0.95)),
                iavi_tol=float(yml_conf.get("hiavi_iavi_tol", 1e-8)),
                iavi_max_iters=int(yml_conf.get("hiavi_iavi_max_iters", 10000)),
                kinematics=kin_conf(yml_conf))


def hiavi_kinematics(yml_conf, data):
    """KinFeatures aligned with hiavi_s (None if the kinematics block is absent / disabled)."""
    cfg = kin_conf(yml_conf)
    if cfg is None:
        return None
    return cached_kinematics(yml_conf, data, cfg, offset=int(data["hiavi_offset"]), shape=data["hiavi_s"].shape)


def kin_ll_fields(model, s, a, kin, kin_null, ll_joint):
    """LL (comparable with action-only models), LL_joint, LL_kin_null, LL_action, P(exploit) per step."""
    if kin is None:
        return {}
    cmp_, joint, null = comparable_ll(ll_joint * s.size, kin_null, kin, s.size)
    cfg = model.kin_cfg
    labels = labels_from_conf(model.kin_model, cfg)
    g = model.posteriors(s, a, kin)
    pe, px = p_exploit(g, labels)
    return dict(LL=cmp_, LL_joint=joint, LL_kin_null=null, LL_action=model.loglik(s, a),
                p_exploit=pe, p_explore=px, kin_labels=np.asarray(labels["label"]), gamma=g,
                kin_step=kin.step, kin_turn=kin.turn)


def fold_trans_prob(yml_conf, data, train_index):
    """Dense (S, A, S') T from the train animals (cached per split, shared by every (K, run))."""
    S, A = data["n_states"], data["n_actions"]
    cache = data.setdefault("_cache", {})
    per_fold = yml_conf.get("per_fold_trans_prob", True) and train_index is not None
    key = ("T_hiavi", None if not per_fold else hash(np.asarray(train_index, dtype=np.int64).tobytes()))
    if key not in cache:
        if per_fold:
            T = trans_prob_from_segments(data["segments"], S, A, keep_groups=set(data["groups"][train_index]),
                                         fill_unobserved=yml_conf.get("fill_unobserved", "self_loop"))
        else:
            T = sparse.load_npz(yml_conf["trans_probs"])
        T = np.asarray(T.todense() if hasattr(T, "todense") else T, dtype=float)
        if T.shape != (S, A, S):
            raise ValueError(f"trans_prob has shape {T.shape}, expected {(S, A, S)}")
        check_trans_prob(T, n_actions=A, require_full=yml_conf.get("require_full_trans_prob", True))
        cache[key] = T
    return cache[key]


def hiavi_names(yml_conf, tag=""):
    base = os.path.join(yml_conf["out_dir"], f'{yml_conf["run_uid"]}_{yml_conf["n_hidden"]}_{yml_conf["seed"]}')
    return {"model": base + "_hiavi" + tag + ".npz",
            "ll": base + "_hiavi_LL" + tag + ".npz",
            "ll_test": base + "_hiavi_LL_test" + tag + ".npz"}


def _seed(seed, kf_ind, restart):
    return np.random.SeedSequence([int(seed), 0 if kf_ind is None else int(kf_ind) + 1, int(restart)])


# ---------------------------------------------------------------------------
# One fit (K-fold and final runs)
# ---------------------------------------------------------------------------

def fit_hiavi(yml_conf, data, train_index, test_index, tag, kf_ind=None, verbose=True):
    """Restarts on the train split, best by TRAIN LL; saves model + train / test LL / latent files."""
    K = int(yml_conf["n_hidden"])
    st = hiavi_settings(yml_conf)
    names = hiavi_names(yml_conf, tag)
    T = fold_trans_prob(yml_conf, data, train_index)
    s, a = data["hiavi_s"], data["hiavi_a"]
    tr = np.asarray(train_index)
    n_restarts = int(yml_conf.get("hiavi_restarts", 10))
    kin = hiavi_kinematics(yml_conf, data)
    k_tr = None if kin is None else kin[tr]
    kin_null = None if kin is None else pooled(k_tr, st["kinematics"])   # train-only null for the comparable LL

    best, restart_ll = None, []
    for rs in range(n_restarts):
        m = HIAVI(K, T, seed=_seed(yml_conf["seed"], kf_ind, rs), **st).fit(s[tr], a[tr], k_tr)
        ll = m.loglik(s[tr], a[tr], k_tr)
        restart_ll.append(ll)
        if not np.isfinite(ll):
            continue
        if best is None or ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError(f"all {n_restarts} HIAVI restarts gave non-finite LL ({names['model']})")
    if verbose:
        print(f"HIAVI K={K} {tag or 'final'}: train LL{' (joint)' if kin is not None else ''} {best_ll:.4f} "
              f"(restarts {np.min(restart_ll):.4f}..{np.max(restart_ll):.4f})")

    # stage checks
    p = best.params()
    for k in ("pis", "r", "q"):
        if p[k].shape != (K, data["n_states"], data["n_actions"]) or not np.all(np.isfinite(p[k])):
            raise ValueError(f"HIAVI {k} has shape {p[k].shape} or non-finite values")
    if not np.allclose(np.exp(p["logp_tr"]).sum(1), 1, atol=1e-6) or not np.isclose(np.exp(p["logp_init"]).sum(), 1):
        raise ValueError("HIAVI latent initial / transition probabilities do not sum to 1")

    if kin is not None:
        lab = labels_from_conf(best.kin_model, st["kinematics"])
        p["kin_labels"] = np.asarray(lab["label"])
        p["kin_step_feature"] = st["kinematics"]["step_feature"]
        lab.to_csv(names["model"].replace(".npz", "_kinematics.csv"), index=False)
        if verbose:
            print(lab[["state", "label", "step_median", "persistence", "p_stationary"]].to_string(index=False))
    np.savez(names["model"], **p, restart_train_ll=np.asarray(restart_ll), best_restart=int(np.argmax(restart_ll)),
             discount=st["discount"], mstep=st["mstep"], align=data["hiavi_align"],
             seq_offset=data["hiavi_offset"])
    tr_f = dict(LL=best_ll)
    tr_f.update(kin_ll_fields(best, s[tr], a[tr], k_tr, kin_null, best_ll))
    out = {"train_ll": float(tr_f["LL"]), "test_ll": np.nan}
    if verbose and kin is not None:
        print(f"HIAVI K={K} {tag or 'final'}: train LL (comparable) {out['train_ll']:.4f} (joint {tr_f['LL_joint']:.4f}, "
              f"kinematic null {tr_f['LL_kin_null']:.4f}, action-only {float(tr_f['LL_action']):.4f})")
    np.savez(names["ll"], **tr_f, learnt_zs=best.decode(s[tr], a[tr], k_tr), seq_offset=data["hiavi_offset"],
             **seq_meta_arrays(data, tr))
    if test_index is not None and len(test_index):
        te = np.asarray(test_index)
        k_te = None if kin is None else kin[te]
        te_f = dict(LL=best.loglik(s[te], a[te], k_te))
        te_f.update(kin_ll_fields(best, s[te], a[te], k_te, kin_null, te_f["LL"]))
        out["test_ll"] = float(te_f["LL"])
        if verbose:
            print(f"HIAVI K={K} {tag or 'final'}: test LL {out['test_ll']:.4f}" +
                  (f" (comparable; joint {te_f['LL_joint']:.4f}, kinematic null {te_f['LL_kin_null']:.4f}, "
                   f"action-only {float(te_f['LL_action']):.4f})" if kin is not None else ""))
        np.savez(names["ll_test"], **te_f, learnt_zs=best.decode(s[te], a[te], k_te),
                 seq_offset=data["hiavi_offset"], **seq_meta_arrays(data, te))
    return best, out


def _append_summary(yml_conf, fold, out):
    path = os.path.join(yml_conf["out_dir"], "ll_hiavi.csv")
    row = pd.DataFrame([{"K": int(yml_conf["n_hidden"]), "seed": int(yml_conf["seed"]),
                         "run_uid": yml_conf["run_uid"], "fold": fold,
                         "train_ll": out["train_ll"], "test_ll": out["test_ll"]}])
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~((old["run_uid"] == yml_conf["run_uid"]) & (old["K"] == int(yml_conf["n_hidden"]))
                    & (old["seed"] == int(yml_conf["seed"])) & (old["fold"].astype(str) == str(fold)))]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(path, index=False)


def summary_from_files(yml_conf, n_folds=None):
    """Rebuild the CV rows for one (K, run) from the LL files (e.g. after array jobs wrote them in parallel)."""
    rows = []
    for k in range(int(n_folds if n_folds is not None else yml_conf.get("n_folds", 5))):
        nm = hiavi_names(yml_conf, "_KF_" + str(k))
        if os.path.exists(nm["ll"]):
            rows.append({"K": int(yml_conf["n_hidden"]), "seed": int(yml_conf["seed"]), "run_uid": yml_conf["run_uid"],
                         "fold": k, "train_ll": float(np.load(nm["ll"])["LL"]),
                         "test_ll": float(np.load(nm["ll_test"])["LL"]) if os.path.exists(nm["ll_test"]) else np.nan})
    return pd.DataFrame(rows)


def run_hiavi_kfold(yml_conf, data=None, folds_to_run=None, write_summary=True):
    data = prepare_hiavi_data(yml_conf, data)
    os.makedirs(yml_conf["out_dir"], exist_ok=True)
    sel = folds_to_run if folds_to_run is not None else yml_conf.get("folds_to_run")
    sel = None if sel is None else set(int(k) for k in as_list(sel))
    for kf_ind, (tr, te) in enumerate(get_folds(yml_conf, data["groups"])):
        if sel is not None and kf_ind not in sel:
            continue
        check_folds(data["groups"], tr, te, name=f"fold {kf_ind}")
        tag = "_KF_" + str(kf_ind)
        if os.path.exists(hiavi_names(yml_conf, tag)["ll_test"]) and not yml_conf.get("overwrite", False):
            print("HIAVI fold", kf_ind, "exists; skipping")
            continue
        _, out = fit_hiavi(yml_conf, data, tr, te, tag, kf_ind)
        if write_summary:
            _append_summary(yml_conf, kf_ind, out)
    return data


def run_hiavi_final(yml_conf, data=None):
    """Final run: train on all but the holdout animals (holdout_groups / holdout_fraction), tag ""."""
    data = prepare_hiavi_data(yml_conf, data)
    os.makedirs(yml_conf["out_dir"], exist_ok=True)
    tr, te, hold = holdout_split(yml_conf, data["groups"])
    print("HIAVI final run; holdout animals:", hold)
    if os.path.exists(hiavi_names(yml_conf)["ll"]) and not yml_conf.get("overwrite", False):
        print("HIAVI final run exists; skipping")
        return data
    _, out = fit_hiavi(yml_conf, data, tr, te, "", None)
    _append_summary(yml_conf, "holdout", out)
    return data


if __name__ == '__main__':
    from sit_fuse.utils import read_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    parser.add_argument("--fold", type=int, nargs="*", default=None, help="Only these fold indices.")
    parser.add_argument("--holdout", action="store_true", help="Final (holdout) run instead of K-fold.")
    args = parser.parse_args()

    conf = read_yaml(args.yaml)
    conf.setdefault("n_hidden", conf.get("n_hidden_init"))
    if args.holdout:
        run_hiavi_final(conf)
    else:
        run_hiavi_kfold(conf, folds_to_run=args.fold)
