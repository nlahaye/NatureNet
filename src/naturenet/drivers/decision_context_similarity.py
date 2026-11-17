"""
Decision-context similarity between behaviour models fitted to the same tracks (HIQL vs SWIRL, K vs K', S-1 vs S-2,
coupled vs independent, ...). Model-agnostic: every model only has to provide one intention / mode label per step.

    python -m naturenet.drivers.decision_context_similarity -y config/v2/drivers/whales_decision_context.yaml

YAML ("{root}" in any path is replaced by `root`; a path with * / ? is globbed and must match exactly one file):
    root: /data/nlahaye/NatureNet/Blue_Whale_v2
    out_dir: "{root}/decision_context"
    models:                                     # any number; names are free
      hiql_k5:
        kind: series                            # dir of <uid>_time_series_pd.csv (plot_hiavi / plot_swirl_new export)
        dir: /data/nlahaye/.../plots/series_hiavi_s1
        label_col: latent                       # default latent
        class_col: latent_class                 # optional (exploit / explore / ambiguous per step)
        p_exploit_col: p_exploit                # optional
        exploit_explore: .../full_exploit_explore.csv   # optional (state, class): mode classes; overrides class_col
        state_reward: .../state_reward_hiavi_s1.npy     # optional (K, S) reward per mode and env state
      swirl_cv:
        kind: latents                           # K-fold npz with decoded paths
        npz: /data/.../SWIRL_..._KF_{fold}.npz  # {fold} filled from folds
        folds: [0, 1, 2, 3, 4]
        path_field: jax_path_vmap               # (N, T) decoded modes ([learnt_zs] also accepted; argmax if (N, T, K))
        key_field: seq_key                      # (N,) sequence key ("<df_run_uid>_<uid>[__augX]")
        seg_field: seq_seg                      # (N,) segment
        start_field: seq_start                  # (N,) first seg_pos of the sequence
        seq_offset: 1                           # seg_pos of step t = start + seq_offset + t
        df_run_uid: SWIRL_WHALE_v2              # stripped from keys (swirl_track_series._df_key)
        p_exploit_field: null                   # optional (N, T)
        env_field: null                         # optional (N, T) env states
    pairs: [[hiql_k5, swirl_s2_k5]]             # default: every pair
    env_col: env                                # env state column (series) for occupancy similarity
    keep_aug: false                             # drop augmented copies (__aug) from latents inputs
    n_boot: 1000                                # animal bootstrap
    n_null: 50                                  # within-animal circular-shift null of the second model's labels
    seed: 0

Steps are joined on (uid, segment, seg_pos). Per pair (pair_summary.csv):
  n_steps, n_animals, K_a, K_b, AMI, ARI, matched_agreement (Hungarian one-to-one mode matching), purity_a_in_b /
  purity_b_in_a (how much each mode of one model sits inside a single mode of the other), class agreement and
  Cohen's kappa on exploit / explore labels, Spearman of p_exploit, the shift-null mean of AMI and its p, animal
  bootstrap 95% CIs (AMI, matched agreement, kappa, Spearman). class_* uses steps both models call exploit or
  explore (do they agree when both commit?); class3_* adds "ambiguous" as a third class (all steps).
Also: crosstab_<a>__<b>.csv (+ row-normalised heatmap .png), mode_alignment_<a>__<b>.csv (for every mode of each
model: n steps, class, Hungarian match in the other model and P(match | mode), best-overlap mode, class agreement,
Jensen-Shannon divergence of env-state occupancy and Spearman of state reward between matched modes),
per_animal.csv (AMI / matched agreement / kappa per animal) and per_fold.csv when an input has folds.
Interpretation: AMI near the shift null = the models carve the tracks into unrelated contexts; high purity in one
direction only = one model's modes are a refinement (split) of the other's.
"""
import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, cohen_kappa_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KEY = ["uid", "segment", "seg_pos"]
CLASSES = ("exploit", "explore")


# ---------------------------------------------------------------------------------------------- loading
def load_series_model(m):
    files = sorted(glob.glob(os.path.join(m["dir"], "*_time_series_pd.csv")))
    if not files:
        raise FileNotFoundError(f"no *_time_series_pd.csv in {m['dir']}")
    lab, cc, pc, ec = m.get("label_col", "latent"), m.get("class_col"), m.get("p_exploit_col"), m.get("env_col", "env")
    cols = KEY + [lab] + [c for c in (cc, pc, ec) if c]
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols) for f in files], ignore_index=True)
    out = df[KEY].copy()
    out["label"] = df[lab].astype(int)
    if cc and cc in df:
        out["cls"] = df[cc].astype(str)
    if pc and pc in df:
        out["p_exploit"] = df[pc].astype(float)
    if ec and ec in df:
        out["env"] = df[ec]
    out["uid"] = out["uid"].astype(str)
    return out


def _df_key(k, uid):
    pre = str(uid) + "_" if uid else ""
    k = str(k)
    return k[len(pre):] if pre and k.startswith(pre) else k


def load_latents_model(m, keep_aug=False):
    frames = []
    for fold in m.get("folds", [None]):
        f = m["npz"].format(fold=fold) if fold is not None else m["npz"]
        z = np.load(f, allow_pickle=True)
        pf = m.get("path_field", "jax_path_vmap")
        if pf not in z and "learnt_zs" in z:
            pf = "learnt_zs"
        P = np.asarray(z[pf])
        if P.ndim == 3:
            P = P.argmax(-1)
        keys = [_df_key(k, m.get("df_run_uid")) for k in np.asarray(z[m.get("key_field", "seq_key")])]
        seg = np.asarray(z[m.get("seg_field", "seq_seg")])
        st = np.asarray(z[m.get("start_field", "seq_start")]).astype(int)
        off = int(m.get("seq_offset", 1))
        pe = np.asarray(z[m["p_exploit_field"]]) if m.get("p_exploit_field") else None
        ev = np.asarray(z[m["env_field"]]) if m.get("env_field") else None
        N, T = P.shape
        for n in range(N):
            if not keep_aug and "__aug" in keys[n]:
                continue
            ok = P[n] >= 0
            d = pd.DataFrame(dict(uid=keys[n].split("__aug")[0], segment=seg[n], seg_pos=st[n] + off + np.arange(T),
                                  label=P[n]))
            if pe is not None:
                d["p_exploit"] = pe[n]
            if ev is not None:
                d["env"] = ev[n]
            if fold is not None:
                d["fold"] = fold
            frames.append(d[ok])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(KEY)
    out["uid"] = out["uid"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def load_model(m, keep_aug=False):
    df = load_series_model(m) if m.get("kind", "series") == "series" else load_latents_model(m, keep_aug)
    if m.get("exploit_explore"):                       # mode classes from the exploit / explore table
        t = pd.read_csv(m["exploit_explore"])
        df["cls"] = df["label"].map(dict(zip(t["state"].astype(int), t["class"].astype(str))))
    sr = np.load(m["state_reward"]) if m.get("state_reward") else None
    return df, sr


# ---------------------------------------------------------------------------------------------- metrics
def matched(a, b):
    """Hungarian one-to-one matching of a's modes to b's (max overlap). Returns (agreement, {a_mode: b_mode}, ct)."""
    ct = pd.crosstab(a, b)
    r, c = linear_sum_assignment(-ct.to_numpy())
    mp = {ct.index[i]: ct.columns[j] for i, j in zip(r, c)}
    return float(ct.to_numpy()[r, c].sum() / max(len(a), 1)), mp, ct


def purity(a, b):
    """fraction of steps whose b-mode is the majority b-mode of their a-mode."""
    ct = pd.crosstab(a, b).to_numpy()
    return float(ct.max(axis=1).sum() / max(ct.sum(), 1))


def class_stats(ca, cb, classes=CLASSES):
    """agreement / Cohen's kappa on steps where both models give one of `classes`."""
    if ca is None or cb is None:
        return np.nan, np.nan, 0
    ok = ca.isin(classes) & cb.isin(classes)
    if ok.sum() < 2:
        return np.nan, np.nan, int(ok.sum())
    x, y = ca[ok].to_numpy(), cb[ok].to_numpy()
    kap = cohen_kappa_score(x, y) if len(set(x)) > 1 or len(set(y)) > 1 else np.nan
    return float(np.mean(x == y)), float(kap), int(ok.sum())


def _spear(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok])[0]) if ok.sum() > 2 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0 else np.nan


def core(j):
    a, b = j["label_a"].to_numpy(), j["label_b"].to_numpy()
    agr, _, _ = matched(a, b)
    ca = j["cls_a"] if "cls_a" in j else None
    cb = j["cls_b"] if "cls_b" in j else None
    cag, kap, ncl = class_stats(ca, cb)
    cag3, kap3, ncl3 = class_stats(ca, cb, CLASSES + ("ambiguous",))
    sp = _spear(j["p_exploit_a"].to_numpy(float), j["p_exploit_b"].to_numpy(float)) \
        if "p_exploit_a" in j and "p_exploit_b" in j else np.nan
    return dict(AMI=float(adjusted_mutual_info_score(a, b)), ARI=float(adjusted_rand_score(a, b)),
                matched_agreement=agr, purity_a_in_b=purity(a, b), purity_b_in_a=purity(b, a),
                class_agreement=cag, class_kappa=kap, n_class_steps=ncl,
                class3_agreement=cag3, class3_kappa=kap3, n_class3_steps=ncl3, spearman_p_exploit=sp)


def shift_null(j, rng, n_null):
    """AMI with the second model's labels circularly shifted within every animal sequence (keeps dwell structure)."""
    if not n_null:
        return np.array([])
    j = j.sort_values(KEY)
    grp = j.groupby(["uid", "segment"]).indices
    a, b = j["label_a"].to_numpy(), j["label_b"].to_numpy()
    out = []
    for _ in range(int(n_null)):
        bs = b.copy()
        for idx in grp.values():
            if len(idx) > 1:
                bs[idx] = np.roll(b[idx], rng.integers(1, len(idx)))
        out.append(adjusted_mutual_info_score(a, bs))
    return np.asarray(out)


def boot(j, rng, n_boot, keys=("AMI", "matched_agreement", "class_kappa", "class3_kappa", "spearman_p_exploit")):
    if not n_boot:
        return {}
    an = j["uid"].unique()
    by = {u: g for u, g in j.groupby("uid")}
    res = {k: [] for k in keys}
    for _ in range(int(n_boot)):
        s = pd.concat([by[u] for u in rng.choice(an, size=len(an), replace=True)], ignore_index=True)
        c = core(s)
        for k in keys:
            res[k].append(c[k])
    out = {}
    for k, v in res.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        out[f"{k}_lo"], out[f"{k}_hi"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) \
            if len(v) else (np.nan, np.nan)
    return out


def _mode_class(j, side):
    c = f"cls_{side}"
    if c not in j:
        return {}
    return j.groupby(f"label_{side}")[c].agg(lambda s: s.value_counts().index[0]).to_dict()


def mode_alignment(j, na, nb, sra, srb):
    rows = []
    for x, y, nx, ny, srx, sry in (("a", "b", na, nb, sra, srb), ("b", "a", nb, na, srb, sra)):
        _, mp, ct = matched(j[f"label_{x}"].to_numpy(), j[f"label_{y}"].to_numpy())
        rn = ct.div(ct.sum(axis=1), axis=0)
        clx, cly = _mode_class(j, x), _mode_class(j, y)
        envs = sorted(j["env_a"].dropna().unique()) if "env_a" in j else None
        for k in ct.index:
            m = mp.get(k)
            r = dict(model=nx, mode=int(k), n_steps=int(ct.loc[k].sum()), cls=clx.get(k), other=ny,
                     matched_mode=None if m is None else int(m),
                     p_matched=float(rn.loc[k, m]) if m is not None else np.nan,
                     best_mode=int(rn.loc[k].idxmax()), p_best=float(rn.loc[k].max()),
                     matched_cls=cly.get(m) if m is not None else None)
            r["cls_agree"] = (r["cls"] == r["matched_cls"]) if r["cls"] is not None and r["matched_cls"] else None
            if envs is not None and m is not None:
                ex = j.loc[j[f"label_{x}"] == k, "env_a"].value_counts().reindex(envs, fill_value=0).to_numpy(float)
                ey = j.loc[j[f"label_{y}"] == m, "env_a"].value_counts().reindex(envs, fill_value=0).to_numpy(float)
                r["js_env_occupancy"] = float(jensenshannon(ex / ex.sum(), ey / ey.sum(), base=2) ** 2) \
                    if ex.sum() and ey.sum() else np.nan
            if srx is not None and sry is not None and m is not None and k < len(srx) and m < len(sry) \
                    and np.shape(srx)[-1] == np.shape(sry)[-1]:
                r["spearman_state_reward"] = _spear(np.asarray(srx[k], float), np.asarray(sry[m], float))
            rows.append(r)
    return pd.DataFrame(rows)


def heatmap(ct, na, nb, path):
    rn = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * ct.shape[1], 1.0 + 0.5 * ct.shape[0]))
    ax.imshow(rn.to_numpy(), cmap="viridis", vmin=0, vmax=1)
    for (i, k), v in np.ndenumerate(rn.to_numpy()):
        ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w" if v < 0.6 else "k")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels(ct.columns)
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels(ct.index)
    ax.set_xlabel(f"{nb} mode")
    ax.set_ylabel(f"{na} mode")
    ax.set_title(f"P({nb} mode | {na} mode)", fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def join(da, db):
    ca = {c: f"{c}_a" for c in da.columns if c not in KEY}
    cb = {c: f"{c}_b" for c in db.columns if c not in KEY}
    return da.rename(columns=ca).merge(db.rename(columns=cb), on=KEY, how="inner")


def compare(na, nb, da, db, sra, srb, conf, rng, out):
    j = join(da, db)
    if j.empty:
        raise ValueError(f"{na} vs {nb}: no shared (uid, segment, seg_pos) steps; check keys / seq_offset")
    if "env_a" not in j and "env_b" in j:
        j["env_a"] = j["env_b"]
    c = core(j)
    nul = shift_null(j, rng, conf.get("n_null", 50))
    row = dict(model_a=na, model_b=nb, n_steps=len(j), n_animals=int(j["uid"].nunique()),
               frac_a_joined=len(j) / len(da), frac_b_joined=len(j) / len(db),
               K_a=int(j["label_a"].nunique()), K_b=int(j["label_b"].nunique()), **c)
    if len(nul):
        row.update(AMI_null_mean=float(nul.mean()), AMI_null_sd=float(nul.std()),
                   p_AMI=float((1 + np.sum(nul >= c["AMI"])) / (1 + len(nul))))
    row.update(boot(j, rng, conf.get("n_boot", 1000)))
    tag = f"{na}__{nb}"
    _, _, ct = matched(j["label_a"].to_numpy(), j["label_b"].to_numpy())
    ct.to_csv(os.path.join(out, f"crosstab_{tag}.csv"))
    heatmap(ct, na, nb, os.path.join(out, f"crosstab_{tag}.png"))
    mode_alignment(j, na, nb, sra, srb).to_csv(os.path.join(out, f"mode_alignment_{tag}.csv"), index=False)
    pa = []
    for u, g in j.groupby("uid"):
        if len(g) > 5:
            pa.append(dict(model_a=na, model_b=nb, uid=u, n_steps=len(g), **core(g)))
    pf = []
    for fc in ("fold_a", "fold_b"):
        if fc in j:
            for f, g in j.groupby(fc):
                pf.append(dict(model_a=na, model_b=nb, fold_by=fc, fold=f, n_steps=len(g), **core(g)))
            fr = pd.DataFrame([r for r in pf if r["fold_by"] == fc])
            for k in ("AMI", "matched_agreement", "class_kappa"):
                row[f"{k}_fold_mean"] = float(fr[k].mean())
                row[f"{k}_fold_se"] = float(fr[k].std(ddof=1) / np.sqrt(len(fr))) if len(fr) > 1 else np.nan
            break
    return row, pa, pf


def _resolve(conf):
    """fill {root} and expand single-match globs in every model path."""
    root = conf.get("root", "")
    fx = lambda p: p.replace("{root}", root) if isinstance(p, str) else p
    conf = dict(conf, out_dir=fx(conf["out_dir"]))
    models = {}
    for n, m in conf["models"].items():
        m = {k: fx(v) for k, v in m.items()}
        for k in ("dir", "exploit_explore", "state_reward"):
            if isinstance(m.get(k), str) and any(c in m[k] for c in "*?["):
                g = sorted(glob.glob(m[k]))
                if len(g) != 1:
                    raise FileNotFoundError(f"{n}.{k}: {m[k]} matched {len(g)} paths ({g[:4]}); make it specific")
                m[k] = g[0]
        models[n] = m
    conf["models"] = models
    return conf


def run(conf):
    conf = _resolve(conf)
    out = conf["out_dir"]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(int(conf.get("seed", 0)))
    models = {n: load_model(m, bool(conf.get("keep_aug", False))) for n, m in conf["models"].items()}
    for n, (d, _) in models.items():
        print(f"{n}: {len(d)} steps, {d['uid'].nunique()} animals, K={d['label'].nunique()}")
    pairs = conf.get("pairs") or list(itertools.combinations(models, 2))
    rows, pa, pf = [], [], []
    for na, nb in pairs:
        (da, sra), (db, srb) = models[na], models[nb]
        r, a_, f_ = compare(na, nb, da, db, sra, srb, conf, rng, out)
        rows.append(r)
        pa += a_
        pf += f_
        print(f"{na} vs {nb}: AMI {r['AMI']:.3f} (null {r.get('AMI_null_mean', np.nan):.3f}), matched "
              f"{r['matched_agreement']:.3f}, kappa {r['class_kappa']:.3f} (3-class {r['class3_kappa']:.3f}), rho(p_exploit) {r['spearman_p_exploit']:.3f}")
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out, "pair_summary.csv"), index=False)
    pd.DataFrame(pa).to_csv(os.path.join(out, "per_animal.csv"), index=False)
    if pf:
        pd.DataFrame(pf).to_csv(os.path.join(out, "per_fold.csv"), index=False)
    print("wrote", out)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--n-null", type=int, default=None)
    a = ap.parse_args()
    with open(a.yaml) as f:
        conf = yaml.safe_load(f)
    if a.n_boot is not None:
        conf["n_boot"] = a.n_boot
    if a.n_null is not None:
        conf["n_null"] = a.n_null
    run(conf)


if __name__ == "__main__":
    main()
