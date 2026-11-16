"""
Test 5 - environmental driver attribution at every level / model / fold, with sign and rank agreement.

For every perspective (environment level x model kind x fold) the state reward of each intention, R_k(s), is
regressed on the SIT-FUSE cluster profiles of the states (reward_env_projection.project_rewards: weighted ridge on
z-scored profiles, bootstrap CI, CV R^2). Then:

  fold stability     within a perspective: intentions matched across folds (Hungarian on R_k(s) correlation),
                     Spearman / sign agreement of the coefficient vectors between folds
  cross-perspective  between levels / models (different state spaces): intentions matched by coefficient-vector
                     correlation (plus kinematic label agreement when both carry exploit / explore labels),
                     sign agreement and Spearman of the |coef| variable ranks for every matched pair
  consensus          per variable and matched intention: share of perspectives with the consensus sign, mean rank

Reward sources (kind):
  hiavi       <..>_hiavi[_KF_k].npz: R(s) = sum_a pi(a|s) r(s, a) (HIQL / IAVI)
  scales      <run_uid>_SC_<tag>_<K>_<seed>_scales_KF_k.npz: sc_iavi_r (K, H, A) on the context nodes, policy
              softmax(sc_iavi_q) (or sc_logpi), averaged over the contexts ending in s (weights ctx_counts)
  multiagent  <out_dir>/<ma_subdir>/ma_final_K<K>.npz: r_<fit> / pis_<fit> (fit = indep | a coupling variant)
  swirl_npz   any npz with new_reward (K, 1, S) or (K, S) (path given directly)

YAML:
  out_dir: /data/nlahaye/NatureNet/Blue_Whale_v2/drivers/driver_consistency/
  n_boot: 200
  variables: null            # default: the variables common to every perspective's profiles
  perspectives:
    - {label: fine_hiql_kin, level: fine, kind: hiavi, yaml: config/v2/swirl/whales_hiavi_kfold_kin.yaml, K: 5,
       projection: config/v2/swirl/whales_reward_env_projection_hiql.yaml, folds: all, final: True}
    - {label: k12_hiql_kin, level: k12, kind: hiavi, yaml: config/v2/levels/whales/k12/hiavi_kfold_kin.yaml, K: 5,
       projection: config/v2/levels/whales/k12/reward_env_projection_hiql.yaml}
    - {label: fine_scales_N2, level: fine, kind: scales, yaml: config/v2/swirl/whales_hiql_scales_kin_k5.yaml,
       variant: HIQL_N2, projection: ...}
    - {label: fine_ma_temporal, level: fine, kind: multiagent, yaml: config/v2/swirl/whales_multiagent_kin.yaml,
       K: 5, fit: temporal, projection: ...}
Outputs: coefficients.csv, fold_stability.csv, cross_agreement.csv, consensus.csv, agreement_heatmap.png

    python -m naturenet.drivers.driver_consistency -y config/v2/drivers/generated/whales/specs/driver_consistency.yaml \
        [--perspective fine_hiql_kin]      # one perspective (parallel), then --collect
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from scipy.stats import spearmanr

from naturenet.drivers.common import (abs_cfg, atomic_csv, best_run, hiavi_files, read_yaml, run_conf)


# ---------------------------------------------------------------------------- rewards
def state_reward_hiavi(z):
    r, pis = np.asarray(z["r"], float), np.asarray(z["pis"], float)
    return (pis * r).sum(-1) if r.ndim == 3 else r


def state_reward_scales(z, S):
    from naturenet.models.irl.swirl_scales import ContextTree
    r = np.asarray(z["sc_iavi_r"], float)                       # (K, H, A)
    if "sc_logpi" in z.files:
        pol = np.exp(np.asarray(z["sc_logpi"], float))
        pol = pol / pol.sum(-1, keepdims=True)
    else:
        pol = softmax(np.asarray(z["sc_iavi_q"], float), axis=-1)
    Rh = (pol * r).sum(-1)                                      # (K, H)
    nodes = ContextTree.nodes_from_save(z)
    last = np.array([n[-1] for n in nodes])
    w = np.asarray(z["ctx_counts"], float) if "ctx_counts" in z.files else np.ones(len(nodes))
    K = Rh.shape[0]
    R = np.full((K, S), np.nan)
    for s in range(S):
        m = last == s
        if m.any():
            ww = w[m] if w[m].sum() > 0 else np.ones(m.sum())
            R[:, s] = (Rh[:, m] * ww).sum(1) / ww.sum()
    return R


def kin_labels(z, prefix=""):
    k = prefix + "kin_labels"
    return np.asarray(z[k]).astype(str) if k in z.files else None


def load_rewards(p):
    """[(fold or 'final', R_state (K, S), labels or None)] for one perspective."""
    conf = run_conf(dict(yaml=p["yaml"], set=p.get("set") or {}))
    S = int(conf["n_states"])
    kind = p["kind"]
    out = []
    n_folds = int(conf.get("n_folds", 5))
    folds = range(n_folds) if p.get("folds", "all") in ("all", None) else p["folds"]
    if kind == "hiavi":
        K = int(p.get("K", conf.get("n_hidden", 5)))
        sweep = bool(p.get("sweep", True))
        run = best_run(conf, K, n_folds) if sweep else 0
        for k in folds:
            f = hiavi_files(conf, K, run, k, sweep=sweep)["model"]
            if os.path.exists(f):
                z = np.load(f, allow_pickle=True)
                out.append((k, state_reward_hiavi(z), kin_labels(z)))
        if p.get("final"):
            fc = run_conf(dict(yaml=p["final_yaml"])) if p.get("final_yaml") else conf
            f = hiavi_files(fc, K, 0, None, sweep=False)["model"]
            if os.path.exists(f):
                z = np.load(f, allow_pickle=True)
                out.append(("final", state_reward_hiavi(z), kin_labels(z)))
    elif kind == "scales":
        from naturenet.models.irl.run_swirl_scales_kfold import _names
        for k in folds:
            f = _names(conf, p["variant"], k)[1]
            if os.path.exists(f):
                z = np.load(f, allow_pickle=True)
                out.append((k, state_reward_scales(z, S), kin_labels(z)))
    elif kind == "multiagent":
        from naturenet.models.irl.run_hiavi_multiagent import ma_conf
        m = ma_conf(conf)
        f = os.path.join(conf["out_dir"], m["ma_subdir"], f"ma_final_K{int(p.get('K', 5))}.npz")
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            fit = p.get("fit", "indep")
            r, pis = np.asarray(z[f"r_{fit}"], float), np.asarray(z[f"pis_{fit}"], float)
            R = (pis * r).sum(-1) if r.ndim == 3 else r
            lk = f"kin_labels_{fit}"
            out.append(("final", R, np.asarray(z[lk]).astype(str) if lk in z.files else None))
    elif kind == "swirl_npz":
        for f in sorted(glob.glob(p["path"])):
            z = np.load(f, allow_pickle=True)
            R = np.asarray(z["new_reward"], float)
            out.append((os.path.basename(f), R[:, 0, :] if R.ndim == 3 else R, kin_labels(z)))
    else:
        raise ValueError(f"unknown kind {kind}")
    return conf, out


def profiles_for(p, conf):
    from naturenet.models.irl.reward_env_projection import DEFAULTS, state_profiles
    pc = dict(DEFAULTS)
    pc.update(conf.get("reward_env_projection") or {})
    if p.get("projection"):
        pc.update(read_yaml(abs_cfg(p["projection"])).get("reward_env_projection") or {})
    with open(pc["class_mapper"], "rb") as f:
        mapper = pickle.load(f)
    P, names = state_profiles(pd.read_csv(pc["profiles_csv"]), int(conf["n_states"]), mapper,
                              pc["cluster_scale"], pc["profile_index"], pc.get("variables"))
    return P, names


def use_counts(conf):
    from naturenet.drivers.common import hiavi_data
    d = hiavi_data(conf)
    s = d["hiavi_s"][d["hiavi_mask"]]
    return np.bincount(s, minlength=int(conf["n_states"]))


# ---------------------------------------------------------------------------- matching
def hungarian(C):
    """perm with perm[i] = column matched to row i (maximising C)."""
    r, c = linear_sum_assignment(-np.nan_to_num(C, nan=-1.0))
    perm = np.full(C.shape[0], -1)
    perm[r] = c
    return perm


def corr_rows(A, B):
    ok = np.all(np.isfinite(A), 0) & np.all(np.isfinite(B), 0)
    if ok.sum() < 3:
        return np.zeros((len(A), len(B)))
    return np.corrcoef(np.vstack([A[:, ok], B[:, ok]]))[:len(A), len(A):]


def coef_matrix(df, names):
    K = int(df["mode"].max()) + 1
    M = np.full((K, len(names)), np.nan)
    for _, r in df.iterrows():
        if r["variable"] in names:
            M[int(r["mode"]), names.index(r["variable"])] = r["coef"]
    return M


# ---------------------------------------------------------------------------- per perspective
def run_perspective(spec, p):
    from naturenet.models.irl.reward_env_projection import project_rewards
    conf, rewards = load_rewards(p)
    if not rewards:
        print(f"driver_consistency: no reward files for {p['label']}")
        return None
    P, names = profiles_for(p, conf)
    if spec.get("variables"):
        keep = [n for n in spec["variables"] if n in names]
        P, names = P[:, [names.index(n) for n in keep]], keep
    try:
        uc = use_counts(conf)
    except Exception as ex:
        print(f"driver_consistency: use counts unavailable for {p['label']} ({ex}); unweighted")
        uc = None
    ref = rewards[0][1]
    rows = []
    for tag, R, labs in rewards:
        perm = hungarian(corr_rows(ref, R)) if R.shape == ref.shape and tag != rewards[0][0] else np.arange(len(R))
        Rm = R[perm]
        df = project_rewards(Rm, P, names, use_counts=uc, n_boot=int(spec.get("n_boot", 200)),
                             seed=int(spec.get("seed", 0)), min_visits=int(spec.get("min_state_visits", 1)))
        df["fold"] = str(tag)
        df["mode_orig"] = [int(perm[m]) for m in df["mode"]]
        df["kin_label"] = [labs[perm[m]] if labs is not None and perm[m] < len(labs) else "" for m in df["mode"]]
        rows.append(df)
    d = pd.concat(rows, ignore_index=True)
    d.insert(0, "perspective", p["label"])
    d.insert(1, "level", p.get("level", ""))
    d.insert(2, "kind", p["kind"])
    out = os.path.join(spec["out_dir"], "parts", f"coef_{p['label']}.csv")
    atomic_csv(d, out)
    return d


# ---------------------------------------------------------------------------- synthesis
def fold_stability(d):
    rows = []
    for pl, g in d.groupby("perspective"):
        tags = [t for t in g["fold"].unique()]
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a, b = g[g["fold"] == tags[i]], g[g["fold"] == tags[j]]
                for k in sorted(set(a["mode"]) & set(b["mode"])):
                    ca = a[a["mode"] == k].set_index("variable")["coef"]
                    cb = b[b["mode"] == k].set_index("variable")["coef"].reindex(ca.index)
                    rows.append(dict(perspective=pl, fold_a=tags[i], fold_b=tags[j], mode=k,
                                     spearman=spearmanr(ca, cb).correlation,
                                     sign_agreement=float(np.mean(np.sign(ca) == np.sign(cb)))))
    return pd.DataFrame(rows)


def consensus_coefs(d, names):
    """Per perspective: mean coefficient over its folds (modes already matched to its first fold)."""
    out = {}
    for pl, g in d.groupby("perspective"):
        gg = g[g["fold"] != "final"] if (g["fold"] != "final").any() else g
        m = gg.groupby(["mode", "variable"])["coef"].mean().reset_index()
        lab = gg.groupby("mode")["kin_label"].agg(lambda x: x.mode().iat[0] if len(x.mode()) else "")
        out[pl] = (coef_matrix(m, names), lab.to_dict())
    return out


def cross_agreement(cons, names, label_bonus=0.5):
    rows = []
    pls = list(cons)
    for i in range(len(pls)):
        for j in range(i + 1, len(pls)):
            A, la = cons[pls[i]]
            B, lb = cons[pls[j]]
            C = corr_rows(A, B)
            if label_bonus:
                for x in range(C.shape[0]):
                    for y in range(C.shape[1]):
                        if la.get(x) and la.get(x) == lb.get(y) and la.get(x) != "ambiguous":
                            C[x, y] += label_bonus
            perm = hungarian(C)
            for x, y in enumerate(perm):
                if y < 0:
                    continue
                a, b = A[x], B[y]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 3:
                    continue
                rows.append(dict(perspective_a=pls[i], perspective_b=pls[j], mode_a=x, mode_b=int(y),
                                 label_a=la.get(x, ""), label_b=lb.get(int(y), ""),
                                 coef_corr=float(np.corrcoef(a[ok], b[ok])[0, 1]),
                                 sign_agreement=float(np.mean(np.sign(a[ok]) == np.sign(b[ok]))),
                                 rank_spearman=float(spearmanr(np.abs(a[ok]), np.abs(b[ok])).correlation),
                                 n_variables=int(ok.sum())))
    return pd.DataFrame(rows)


def consensus_table(cons, names, ref=None):
    """Align every perspective's intentions to the reference perspective and summarise per variable."""
    pls = list(cons)
    ref = ref or pls[0]
    A, _ = cons[ref]
    aligned = {}
    for pl in pls:
        B, _ = cons[pl]
        perm = hungarian(corr_rows(A, B)) if pl != ref else np.arange(len(A))
        aligned[pl] = np.array([B[y] if y >= 0 else np.full(len(names), np.nan) for y in perm])
    stack = np.stack([aligned[pl] for pl in pls])               # (P, K, V)
    rows = []
    for k in range(stack.shape[1]):
        for v, n in enumerate(names):
            x = stack[:, k, v]
            ok = np.isfinite(x)
            if not ok.any():
                continue
            sgn = np.sign(np.nanmedian(x))
            ranks = np.array([pd.Series(np.abs(stack[p, k])).rank(ascending=False).iat[v] for p in range(len(pls))])
            rows.append(dict(reference=ref, mode=k, variable=n, median_coef=float(np.nanmedian(x)),
                             consensus_sign=int(sgn), sign_share=float(np.mean(np.sign(x[ok]) == sgn)),
                             mean_rank=float(np.nanmean(ranks[ok])), n_perspectives=int(ok.sum())))
    return pd.DataFrame(rows)


def heatmap(ag, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not len(ag):
        return
    m = ag.groupby(["perspective_a", "perspective_b"])["sign_agreement"].mean()
    pls = sorted(set(ag["perspective_a"]) | set(ag["perspective_b"]))
    M = np.full((len(pls), len(pls)), np.nan)
    for (a, b), v in m.items():
        M[pls.index(a), pls.index(b)] = M[pls.index(b), pls.index(a)] = v
    np.fill_diagonal(M, 1.0)
    fig, ax = plt.subplots(figsize=(0.5 * len(pls) + 3, 0.5 * len(pls) + 2.5))
    im = ax.imshow(M, vmin=0.5, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(pls)), pls, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(pls)), pls, fontsize=7)
    fig.colorbar(im, ax=ax, label="mean sign agreement (matched intentions)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def collect(spec):
    parts = sorted(glob.glob(os.path.join(spec["out_dir"], "parts", "coef_*.csv")))
    if not parts:
        print("driver_consistency: nothing to collect")
        return None
    d = pd.concat([pd.read_csv(p, dtype={"fold": str}) for p in parts], ignore_index=True)
    d["kin_label"] = d["kin_label"].fillna("")
    sets = [set(g["variable"]) for _, g in d.groupby("perspective")]
    names = [n for n in d["variable"].unique() if all(n in s for s in sets)]
    d = d[d["variable"].isin(names)]
    out = spec["out_dir"]
    atomic_csv(d, os.path.join(out, "coefficients.csv"))
    atomic_csv(fold_stability(d), os.path.join(out, "fold_stability.csv"))
    cons = consensus_coefs(d, names)
    ag = cross_agreement(cons, names, float(spec.get("label_bonus", 0.5)))
    atomic_csv(ag, os.path.join(out, "cross_agreement.csv"))
    atomic_csv(consensus_table(cons, names, spec.get("reference")), os.path.join(out, "consensus.csv"))
    heatmap(ag, os.path.join(out, "agreement_heatmap.png"))
    print(f"driver_consistency: {d['perspective'].nunique()} perspectives, {len(names)} common variables -> {out}")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--perspective", default=None)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args(argv)
    spec = read_yaml(abs_cfg(a.yaml))
    if not a.collect:
        for p in spec["perspectives"]:
            if a.perspective and p["label"] != a.perspective:
                continue
            run_perspective(spec, p)
    if a.collect or not a.perspective:
        collect(spec)


if __name__ == "__main__":
    main()
