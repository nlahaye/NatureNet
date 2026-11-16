"""
Tests 1 + 2 summary - environment level x decision scale grid with paired held-out gains.

Every entry is one K-fold run on the SAME folds / sequences / actions (only the environment states differ between
levels, only the decision layer differs between decision scales), so per-fold (and per-animal) test-LL differences
are paired:

  gain vs reference   entry - reference (default: fine level, plain HIQL = N1)
  environment effect  level - matched null level at the same decision scale (test 1: real - null_shift, ...)
  decision effect     decision scale - N1 at the same level (test 2: does deeper history help at every level?)

YAML:
  out_dir: /data/nlahaye/NatureNet/Blue_Whale_v2/drivers/env_scale_grid/
  reference: {level: fine, decision: N1}
  nulls: [null_shift, null_shuffle, null_marginal, null_single]   # environment effect vs each null present
  per_animal: True          # also pair per animal (held-out animal = unit; scales step_ll / HIAVI seq_loglik)
  entries:
    - {level: fine, kind: hiavi, yaml: config/v2/swirl/whales_hiavi_kfold_kin.yaml, K: 5}       # decision N1
    - {level: fine, kind: scales, yaml: config/v2/swirl/whales_hiql_scales_kin_k5.yaml}         # every variant
    - {level: k12, kind: hiavi, yaml: config/v2/levels/whales/k12/hiavi_kfold_kin.yaml, K: 5, sweep: True}
  (kind hiavi: sweep True = get_stats_hiavi names, best run by mean TRAIN LL; scales: every variant in
   <run_uid>_scales_cv.csv, decision = variant tag shortened, e.g. N2, N1+hist, N1+dur)
Outputs: grid_folds.csv, grid_animals.csv, grid_gains.csv (paired summaries), grid_heatmap.png.

    python -m naturenet.drivers.env_scale_grid -y config/v2/drivers/generated/whales/specs/env_scale_grid.yaml
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from naturenet.drivers.common import (abs_cfg, atomic_csv, base_id, best_run, hiavi_files, is_original,
                                      paired_summary, read_yaml, run_conf)


def decision_label(tag):
    """Scales variant tag -> short decision-scale label (HIQL_N2_hist-lags_dur-logi20 -> N2+hist+dur)."""
    t = str(tag)
    m = re.search(r"(?:^|_)N(\d+)", t)
    lab = f"N{m.group(1)}" if m else t
    h = re.search(r"(?:^|_)hist-([A-Za-z0-9+]+)", t)
    if h:
        kinds = h.group(1).split("+")
        if any(k != "env" for k in kinds):
            lab += "+hist"
        if "env" in kinds:                       # environment-driven switching (test 7)
            lab += "+env"
    for key, short in (("dur", "dur"), ("hier", "hier"), ("reg", "hier")):
        if re.search(rf"(^|_){key}", t):
            lab += f"+{short}"
    return lab


def _hiavi_rows(e, conf):
    K = int(e.get("K", conf.get("n_hidden", 5)))
    csv = os.path.join(conf["out_dir"], "ll_hiavi.csv")
    rows = []
    if os.path.exists(csv):
        d = pd.read_csv(csv)
        d = d[d["K"] == K]
        if len(d):
            tr = d.groupby("run_uid")["train_ll"].mean()
            uid = tr.idxmax()
            for _, r in d[d["run_uid"] == uid].iterrows():
                rows.append(dict(fold=int(r["fold"]), test_ll=float(r["test_ll"]), train_ll=float(r["train_ll"]),
                                 run_uid=uid, K=K))
            return rows
    n_folds = int(conf.get("n_folds", 5))
    sweep = bool(e.get("sweep", True))
    run = best_run(conf, K, n_folds) if sweep else 0
    for k in range(n_folds):
        f = hiavi_files(conf, K, run, k, sweep=sweep)
        if os.path.exists(f["ll_test"]):
            rows.append(dict(fold=k, test_ll=float(np.load(f["ll_test"])["LL"]),
                             train_ll=float(np.load(f["ll"])["LL"]) if os.path.exists(f["ll"]) else np.nan,
                             run_uid=os.path.basename(f["model"]), K=K, run=run))
    return rows


def _scales_rows(e, conf):
    path = os.path.join(conf["out_dir"], conf["run_uid"] + "_scales_cv.csv")
    if not os.path.exists(path):
        return []
    d = pd.read_csv(path)
    keep = e.get("variants")
    rows = []
    for _, r in d.iterrows():
        if keep and r["variant"] not in keep:
            continue
        rows.append(dict(fold=int(r["fold"]), test_ll=float(r["LL_test"]), train_ll=float(r.get("LL_train", np.nan)),
                         variant=r["variant"], decision=decision_label(r["variant"])))
    return rows


def fold_table(spec):
    out = []
    for e in spec["entries"]:
        conf = run_conf(dict(yaml=e["yaml"], set=e.get("set") or {}))
        rows = _hiavi_rows(e, conf) if e["kind"] == "hiavi" else _scales_rows(e, conf)
        if not rows:
            print(f"env_scale_grid: no results yet for {e['level']} {e['kind']} ({e['yaml']})")
        for r in rows:
            r.setdefault("decision", e.get("decision", "N1"))
            out.append(dict(level=e["level"], kind=e["kind"], yaml=e["yaml"], **r))
    d = pd.DataFrame(out)
    if len(d):
        # the scales N1 variant duplicates the HIAVI N1 entry: keep the HIAVI one (the fitted baseline)
        d = d.sort_values("kind").drop_duplicates(["level", "decision", "fold"], keep="first")
    return d


# ---------------------------------------------------------------------------- per animal
def _per_animal_scales(conf, variant):
    from naturenet.models.irl.run_swirl_scales_kfold import _names
    rows = []
    for k in range(int(conf.get("n_folds", 5))):
        f = _names(conf, variant, k)[3]
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        sl = np.asarray(z["step_ll"], float)
        m = np.isfinite(sl) & (sl != 0)
        keys = np.asarray(z["seq_key"]).astype(str)
        tot = np.where(m, sl, 0).sum(1)
        n = m.sum(1)
        df = pd.DataFrame(dict(key=keys, ll=tot, n=n))
        df = df[[is_original(x) for x in df["key"]]]
        df["animal"] = [base_id(x) for x in df["key"]]
        g = df.groupby("animal")[["ll", "n"]].sum()
        for an, r in g.iterrows():
            rows.append(dict(fold=k, animal=an, ll_per_step=float(r["ll"] / max(r["n"], 1)), n_steps=int(r["n"])))
    return rows


def _per_animal_hiavi(conf, e):
    from naturenet.drivers.common import fold_T, folds, hiavi_data, kinematics, load_hiavi
    K = int(e.get("K", conf.get("n_hidden", 5)))
    data = hiavi_data(conf)
    fl = folds(conf, data)
    kin = kinematics(conf, data)
    sweep = bool(e.get("sweep", True))
    run = best_run(conf, K, len(fl)) if sweep else 0
    s, a = data["hiavi_s"], data["hiavi_a"]
    rows = []
    for k, (tr, te) in enumerate(fl):
        f = hiavi_files(conf, K, run, k, sweep=sweep)["model"]
        if not os.path.exists(f):
            continue
        m = load_hiavi(f, fold_T(conf, data, tr), conf)
        ll = m.seq_loglik(s[te], a[te], None if kin is None else kin[te])
        n = (np.asarray(a[te]) >= 0).sum(1)
        keys = np.array([data["meta"][i]["key"] for i in te]).astype(str)
        df = pd.DataFrame(dict(key=keys, ll=ll, n=n))
        df = df[[is_original(x) for x in df["key"]]]
        df["animal"] = [base_id(x) for x in df["key"]]
        for an, r in df.groupby("animal")[["ll", "n"]].sum().iterrows():
            rows.append(dict(fold=k, animal=an, ll_per_step=float(r["ll"] / max(r["n"], 1)), n_steps=int(r["n"])))
    return rows


def animal_table(spec, folds_df):
    out = []
    for e in spec["entries"]:
        conf = run_conf(dict(yaml=e["yaml"], set=e.get("set") or {}))
        if e["kind"] == "hiavi":
            try:
                rows = _per_animal_hiavi(conf, e)
            except Exception as ex:              # missing inputs at this level: skip, reported
                print(f"env_scale_grid: per-animal {e['level']} hiavi skipped ({type(ex).__name__}: {ex})")
                rows = []
            for r in rows:
                out.append(dict(level=e["level"], decision=e.get("decision", "N1"), **r))
        else:
            sub = folds_df[(folds_df["level"] == e["level"]) & (folds_df["kind"] == "scales")]
            for v in sub["variant"].dropna().unique():
                for r in _per_animal_scales(conf, v):
                    out.append(dict(level=e["level"], decision=decision_label(v), **r))
    d = pd.DataFrame(out)
    return d.drop_duplicates(["level", "decision", "fold", "animal"]) if len(d) else d


# ---------------------------------------------------------------------------- gains
def paired_gains(d, unit_cols, value, ref_level, ref_dec, nulls):
    rows = []
    if not len(d):
        return pd.DataFrame()
    piv = d.pivot_table(index=unit_cols, columns=["level", "decision"], values=value)
    cols = list(piv.columns)

    def add(kind, a, b):
        if a not in piv or b not in piv:
            return
        diff = (piv[a] - piv[b]).dropna()
        s = paired_summary(diff.to_numpy())
        rows.append(dict(contrast=kind, level=a[0], decision=a[1], vs_level=b[0], vs_decision=b[1], **s))

    ref = (ref_level, ref_dec)
    for c in cols:
        if c != ref:
            add("vs_reference", c, ref)
        for nl in nulls:
            if c[0] not in nulls and (nl, c[1]) in cols:
                add(f"env_vs_{nl}", c, (nl, c[1]))
        if c[1] != ref_dec and (c[0], ref_dec) in cols:
            add("decision_vs_N1", c, (c[0], ref_dec))
    return pd.DataFrame(rows)


def heatmap(gains, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = gains[gains["contrast"] == "vs_reference"]
    if not len(g):
        return
    piv = g.pivot_table(index="level", columns="decision", values="mean")
    sig = g.pivot_table(index="level", columns="decision", values="p_signflip")
    fig, ax = plt.subplots(figsize=(1.2 * piv.shape[1] + 3, 0.5 * piv.shape[0] + 2))
    v = np.nanmax(np.abs(piv.to_numpy())) or 1.0
    im = ax.imshow(piv.to_numpy(), cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            x = piv.iat[i, j]
            if np.isfinite(x):
                p = sig.iat[i, j]
                ax.text(j, i, f"{x:+.3f}" + ("*" if np.isfinite(p) and p < 0.05 else ""), ha="center",
                        va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="paired test LL gain vs reference (per step)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(spec):
    out = spec["out_dir"]
    os.makedirs(out, exist_ok=True)
    ref = spec.get("reference") or {"level": "fine", "decision": "N1"}
    nulls = list(spec.get("nulls") or [])
    fd = fold_table(spec)
    atomic_csv(fd, os.path.join(out, "grid_folds.csv"))
    gains = paired_gains(fd, ["fold"], "test_ll", ref["level"], ref["decision"], nulls)
    if len(gains):
        gains.insert(0, "unit", "fold")
    parts = [gains]
    if spec.get("per_animal", True) and len(fd):
        ad = animal_table(spec, fd)
        atomic_csv(ad, os.path.join(out, "grid_animals.csv"))
        ga = paired_gains(ad, ["fold", "animal"], "ll_per_step", ref["level"], ref["decision"], nulls)
        if len(ga):
            ga.insert(0, "unit", "animal")
            parts.append(ga)
    g = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
    if len(g):
        from naturenet.drivers.common import bh
        g["q_bh"] = np.nan
        for u in g["unit"].unique():
            m = g["unit"] == u
            g.loc[m, "q_bh"] = bh(g.loc[m, "p_signflip"].to_numpy())
        atomic_csv(g, os.path.join(out, "grid_gains.csv"))
        heatmap(g[g["unit"] == "fold"], os.path.join(out, "grid_heatmap.png"),
                spec.get("title", "Environment level x decision scale (paired held-out LL)"))
    print(f"env_scale_grid: {len(fd)} fold rows, {len(g)} contrasts -> {out}")
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    a = ap.parse_args(argv)
    run(read_yaml(abs_cfg(a.yaml)))


if __name__ == "__main__":
    main()
