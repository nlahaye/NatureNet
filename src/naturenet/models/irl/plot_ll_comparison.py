"""
Train vs test log-likelihood comparison across the K-fold runs:
  ARHMM (linear-SWIRL init), ARHMM (NN-SWIRL init), SWIRL S-1 / S-2, NN-SWIRL S-1 / S-2, HIAVI, and the
  hiql single-intention IAVI baseline (ll_iavi.csv), whichever are listed in the YAML and found on disk.

    python -m naturenet.models.irl.plot_ll_comparison -y config/swirl/ll_comparison.yaml [--methods swirl_s1 hiavi]

Each method entry points at the YAML its runs used (out_dir, run_uid, seed, n_hidden or the get_stats sweep
min_latent / max_latent / num_runs, n_folds), so the file names are rebuilt exactly as the run scripts write them:
  arhmm        <run_uid>_<K>_hidden_<seed>_seed_arhmm_s_LLs_KF_<k>.npz          train_ll / test_ll
  arhmm_act    <run_uid>_<K>_hidden_<seed>_seed_arhmm_s_LLs_KF_<k>.npz          train_ll_act / test_ll_act
  arhmm_nn     <run_uid>_<K>_hidden_<seed>_seed_arhmm_nn_s_LLs_KF_<k>.npz       train_ll / test_ll
  swirl_s1     <run_uid>_<K>_<seed>_naturenet_S1_iter2_LL1[_test]_KF_<k>.npz    LL
  swirl_s2     <run_uid>_<K>_<seed>_naturenet_S2_iter2_LL2[_test]_KF_<k>.npz    LL
  swirl_nn_s1  <run_uid>_<K>_<seed>_naturenet_MLP_S1_new1_LL[_test]_KF_<k>.npz  LL
  swirl_nn_s2  <run_uid>_<K>_<seed>_naturenet_MLP_S2_new2_LL[_test]_KF_<k>.npz  LL
  hiavi        <run_uid>_<K>_<seed>_hiavi_LL[_test]_KF_<k>.npz                  LL
  hiql_iavi    csv: ll_iavi.csv (num_trajs, fold, train_ll, test_ll); K = 1 reference line
With a sweep, run_uid = <base>_loop_<i>_latent_<K>_states and seed = base seed + i (swirl_data.iter_loop_configs).

All LLs are per step (total / (N * T)). The ARHMM files need train_ll / test_ll, written by the updated
run_arhmm_kfold(_nn).py. Older files only hold the EM trace (a total, not per step), so they are skipped with a
warning. The ARHMM scores p(x_t | x_{t-1}) over states, while SWIRL / HIAVI score p(a_t | s_t) over actions;
on the grid the action fixes the next state, which is the comparison the SWIRL paper makes, but the event spaces
differ. hiql_iavi uses leave-one-sequence-out folds and averages per-sequence means (run_hiql_iavi.py), so it is
drawn only as a reference line. For a matched single-intention baseline, include K = 1 in the HIAVI sweep.

Outputs (<out_dir>/<out_prefix>...):
  _ll_vs_k.png       train LL, test LL, and train - test gap vs K (mean +/- t CI over folds x runs)
  _ll_folds.png      per-fold train -> test pairs for each method at compare_k (default: best test K per method)
  _ll_paired.png     paired test-LL difference vs `reference` (same K / run / fold), if reference is set
  _ll_rows.csv       one row per (method, K, run, fold, split)
  _ll_summary.csv    mean / sd / sem / CI per (method, K, split), plus best and parsimonious K per method
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# name -> (label, colour, file pattern(s) relative to <out_dir>/<run_uid>_, LL key(s))
METHODS = {
    "arhmm": dict(label="ARHMM", color="#d62728", kind="arhmm",
                  pattern="{K}_hidden_{seed}_seed_arhmm_s_LLs_KF_{k}.npz"),
    # The ARHMM re-scored as an action model on the SWIRL steps (train_ll_act / test_ll_act in the same LL file;
    # arhmm_action_ll.py): same event space and steps as SWIRL / HIAVI, unlike "arhmm" (next-state LL).
    "arhmm_act": dict(label="ARHMM (actions)", color="#8c564b", kind="arhmm", key="{sp}_ll_act",
                      pattern="{K}_hidden_{seed}_seed_arhmm_s_LLs_KF_{k}.npz"),
    "arhmm_nn": dict(label="ARHMM (NN init)", color="#ff9896", kind="arhmm",
                     pattern="{K}_hidden_{seed}_seed_arhmm_nn_s_LLs_KF_{k}.npz"),
    "swirl_s1": dict(label="SWIRL S-1", color="#1f77b4", kind="ll",
                     train="{K}_{seed}_naturenet_S1_iter2_LL1_KF_{k}.npz",
                     test="{K}_{seed}_naturenet_S1_iter2_LL1_test_KF_{k}.npz"),
    "swirl_s2": dict(label="SWIRL S-2", color="#17becf", kind="ll",
                     train="{K}_{seed}_naturenet_S2_iter2_LL2_KF_{k}.npz",
                     test="{K}_{seed}_naturenet_S2_iter2_LL2_test_KF_{k}.npz"),
    "swirl_nn_s1": dict(label="NN-SWIRL S-1", color="#2ca02c", kind="ll",
                        train="{K}_{seed}_naturenet_MLP_S1_new1_LL_KF_{k}.npz",
                        test="{K}_{seed}_naturenet_MLP_S1_new1_LL_test_KF_{k}.npz"),
    "swirl_nn_s2": dict(label="NN-SWIRL S-2", color="#98df8a", kind="ll",
                        train="{K}_{seed}_naturenet_MLP_S2_new2_LL_KF_{k}.npz",
                        test="{K}_{seed}_naturenet_MLP_S2_new2_LL_test_KF_{k}.npz"),
    "hiavi": dict(label="HIAVI", color="#9467bd", kind="ll",
                  train="{K}_{seed}_hiavi_LL_KF_{k}.npz",
                  test="{K}_{seed}_hiavi_LL_test_KF_{k}.npz"),
    "hiql_iavi": dict(label="IAVI (hiql, K=1)", color="#7f7f7f", kind="csv"),
}
SPLITS = ("train", "test")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _read_yaml(path, rel_to=None):
    if rel_to is not None and not os.path.isabs(path):
        cand = os.path.join(rel_to, path)
        path = cand if os.path.exists(cand) else path
    with open(path) as f:
        return yaml.safe_load(f) or {}


def method_runs(entry, base_dir=None):
    """
    (K, run, seed, run_uid, out_dir, n_folds) for one method entry. Keys in the entry override the run YAML.
    sweep: True -> get_stats naming (K in min..max_latent, num_runs); False -> the single run_uid / n_hidden.
    """
    conf = dict(_read_yaml(entry["yaml"], base_dir)) if entry.get("yaml") else {}
    conf.update({k: v for k, v in entry.items() if k not in ("name", "yaml", "label", "color")})
    out_dir, base_uid, base_seed = conf["out_dir"], conf["run_uid"], int(conf["seed"])
    n_folds = int(conf.get("n_folds", 5))
    sweep = conf.get("sweep")
    if sweep is None:
        sweep = all(k in conf for k in ("min_latent", "max_latent", "num_runs"))
    if sweep:
        Ks = conf.get("Ks") or list(range(int(conf["min_latent"]), int(conf["max_latent"]) + 1))
        for K in Ks:
            for i in range(int(conf.get("num_runs", 1))):
                yield (int(K), i, base_seed + i, f"{base_uid}_loop_{i}_latent_{int(K)}_states", out_dir, n_folds)
    else:
        K = int(conf.get("n_hidden", conf.get("n_hidden_init")))
        yield (K, 0, base_seed, base_uid, out_dir, n_folds)


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def _load_ll(path, key="LL"):
    with np.load(path, allow_pickle=True) as f:
        if key not in f.files:
            return None
        v = float(np.asarray(f[key]).squeeze())
    return v if np.isfinite(v) else np.nan


def collect_method(entry, base_dir=None):
    """Rows (method, label, K, run, seed, fold, split, ll, path) for one method entry; logs missing files."""
    name = entry["name"]
    spec = METHODS[name]
    rows, missing, old = [], 0, 0
    if spec["kind"] == "csv":
        path = entry["csv"]
        if base_dir is not None and not os.path.isabs(path) and os.path.exists(os.path.join(base_dir, path)):
            path = os.path.join(base_dir, path)
        if not os.path.exists(path):
            warnings.warn(f"{name}: {path} not found; skipped")
            return rows
        df = pd.read_csv(path)
        if "num_trajs" in df:  # the original loop repeated the same CV for num_trajs 0..7; keep the last
            df = df[df["num_trajs"] == df["num_trajs"].max()]
        for _, r in df.iterrows():
            for sp in SPLITS:
                rows.append(dict(method=name, K=1, run=0, seed=np.nan, fold=int(r["fold"]), split=sp,
                                 ll=float(r[f"{sp}_ll"]), path=path))
        return rows

    for K, run, seed, uid, out_dir, n_folds in method_runs(entry, base_dir):
        for k in range(n_folds):
            fmt = dict(K=K, seed=seed, k=k)
            if spec["kind"] == "arhmm":
                path = os.path.join(out_dir, uid + "_" + spec["pattern"].format(**fmt))
                if not os.path.exists(path):
                    missing += 1
                    continue
                vals = {sp: _load_ll(path, spec.get("key", "{sp}_ll").format(sp=sp)) for sp in SPLITS}
                if vals["train"] is None:
                    old += 1
                    continue
                for sp in SPLITS:
                    rows.append(dict(method=name, K=K, run=run, seed=seed, fold=k, split=sp, ll=vals[sp], path=path))
            else:
                for sp in SPLITS:
                    path = os.path.join(out_dir, uid + "_" + spec[sp].format(**fmt))
                    if not os.path.exists(path):
                        missing += 1
                        continue
                    rows.append(dict(method=name, K=K, run=run, seed=seed, fold=k, split=sp,
                                     ll=_load_ll(path), path=path))
    if missing:
        print(f"{name}: {missing} expected LL file(s) missing")
    if old:
        if name == "arhmm_act":
            warnings.warn(f"{name}: {old} ARHMM LL file(s) have no train_ll_act / test_ll_act; re-run get_stats.py / "
                          f"run_arhmm_kfold.py (it adds them; SWIRL init files are kept)")
        else:
            warnings.warn(f"{name}: {old} ARHMM LL file(s) have no train_ll / test_ll (written before the update); "
                          f"re-run run_arhmm_kfold{'_nn' if name == 'arhmm_nn' else ''}.py (it re-fits those folds)")
    return rows


def register_aliases(conf):
    """
    Entries may set alias: <unique name> to include one method several times (e.g. ARHMM pseudo-count variants,
    each with its own run_uid / arhmm_pseudo_count). The alias is registered as a copy of the method spec and
    replaces entry["name"], so rows, summaries and plots keep the variants apart. Idempotent.
    """
    for e in conf["methods"]:
        alias = e.get("alias")
        if alias and e["name"] != alias:
            if e["name"] not in METHODS:
                raise ValueError(f"unknown method {e['name']}; options: {sorted(METHODS)}")
            METHODS[alias] = dict(METHODS[e["name"]], label=e.get("label", alias))
            e["base"], e["name"] = e["name"], alias
    return conf


def collect_lls(conf, base_dir=None, methods=None):
    register_aliases(conf)
    rows = []
    for entry in conf["methods"]:
        if entry["name"] not in METHODS:
            raise ValueError(f"unknown method {entry['name']}; options: {sorted(METHODS)}")
        if methods and entry["name"] not in methods:
            continue
        rows += collect_method(entry, base_dir)
    df = pd.DataFrame(rows, columns=["method", "K", "run", "seed", "fold", "split", "ll", "path"])
    labels = {e["name"]: e.get("label", METHODS[e["name"]]["label"]) for e in conf["methods"]}
    df["label"] = df["method"].map(labels)
    n_bad = int(df["ll"].isna().sum())
    if n_bad:
        warnings.warn(f"{n_bad} non-finite LL value(s) dropped")
    return df.dropna(subset=["ll"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _ci(x, cl):
    x = np.asarray(x, dtype=float)
    n, m = len(x), float(np.mean(x))
    if n < 2:
        return m, np.nan, np.nan, np.nan, np.nan
    sd, se = float(np.std(x, ddof=1)), float(stats.sem(x))
    lo, hi = stats.t.interval(cl, df=n - 1, loc=m, scale=se) if se > 0 else (m, m)
    return m, sd, se, lo, hi


def summarize(df, cl=0.95):
    """mean / sd / sem / CI per (method, K, split) over folds x runs, plus the train - test gap."""
    out = []
    for (meth, K, sp), g in df.groupby(["method", "K", "split"]):
        m, sd, se, lo, hi = _ci(g["ll"], cl)
        out.append(dict(method=meth, label=g["label"].iloc[0], K=K, split=sp, n=len(g), mean=m, sd=sd, sem=se,
                        ci_lo=lo, ci_hi=hi))
    piv = df.pivot_table(index=["method", "label", "K", "run", "fold"], columns="split", values="ll").reset_index()
    if {"train", "test"} <= set(piv.columns):
        piv["gap"] = piv["train"] - piv["test"]
        for (meth, K), g in piv.dropna(subset=["gap"]).groupby(["method", "K"]):
            m, sd, se, lo, hi = _ci(g["gap"], cl)
            out.append(dict(method=meth, label=g["label"].iloc[0], K=K, split="gap", n=len(g), mean=m, sd=sd,
                            sem=se, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(out).sort_values(["method", "split", "K"]).reset_index(drop=True)


def select_k(summary):
    """
    Per method: best_k = highest mean test LL; parsimonious_k = smallest K whose test CI upper bound reaches
    the best K's CI lower bound (statistically indistinguishable from the best; same idea as plot_lls_kf_shark).
    """
    sel = {}
    for meth, g in summary[summary["split"] == "test"].groupby("method"):
        g = g.sort_values("K")
        best = g.loc[g["mean"].idxmax()]
        lo = best["ci_lo"] if np.isfinite(best["ci_lo"]) else best["mean"]
        ok = g[(g["ci_hi"].fillna(g["mean"]) >= lo)]
        sel[meth] = dict(best_k=int(best["K"]), best_test=float(best["mean"]),
                         parsimonious_k=int(ok["K"].min()) if len(ok) else int(best["K"]))
    return sel


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style(conf, meth):
    e = next((e for e in conf["methods"] if e["name"] == meth), {})
    return e.get("label", METHODS[meth]["label"]), e.get("color", METHODS[meth]["color"])


def _hline(ax, row, label, color):
    ax.axhline(row["mean"], color=color, ls=":", lw=1.5, label=label)
    if np.isfinite(row["ci_lo"]):
        ax.axhspan(row["ci_lo"], row["ci_hi"], color=color, alpha=0.12)


def plot_ll_vs_k(summary, sel, conf, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    titles = {"train": "Train LL (per step)", "test": "Test LL (per step)", "gap": "Train - test LL"}
    all_k = sorted(summary.loc[summary["method"] != "hiql_iavi", "K"].unique())
    order = [e["name"] for e in conf["methods"]]
    for ax, sp in zip(axes, ("train", "test", "gap")):
        for meth in order:
            g = summary[(summary["split"] == sp) & (summary["method"] == meth)]
            if g.empty:
                continue
            label, color = _style(conf, meth)
            g = g.sort_values("K")
            if METHODS[meth]["kind"] == "csv":
                _hline(ax, g.iloc[0], label, color)
                continue
            ls = "--" if meth.startswith("arhmm") else "-"
            ax.plot(g["K"], g["mean"], ls, marker="o", ms=4, color=color, label=label)
            ax.fill_between(g["K"], g["ci_lo"], g["ci_hi"], color=color, alpha=0.15, lw=0)
            if sp == "test" and meth in sel:
                b = g[g["K"] == sel[meth]["best_k"]]
                ax.plot(b["K"], b["mean"], marker="*", ms=13, color=color, mec="k", mew=0.6, ls="none")
        if sp == "gap":
            ax.axhline(0, color="k", lw=0.6)
        ax.set_title(titles[sp])
        ax.set_xlabel("K (hidden modes)")
        if all_k:
            ax.set_xticks(all_k)
        ax.grid(alpha=0.3)
    # Common y-range for train and test (including CI bands) so the two panels are directly comparable.
    tt = summary[summary["split"].isin(SPLITS)]
    lo = np.nanmin(np.r_[tt["mean"].values, tt["ci_lo"].values])
    hi = np.nanmax(np.r_[tt["mean"].values, tt["ci_hi"].values])
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    for ax in axes[:2]:
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("log-likelihood per step")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 8), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{conf.get('title', 'Log-likelihood by model')}  (mean, {int(100 * conf.get('ci', 0.95))}% CI over "
                 f"folds x runs; star = best test K)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_folds_at_k(df, sel, conf, path):
    """Per-fold train -> test pairs at compare_k (or each method's best test K)."""
    ck = conf.get("compare_k")
    meths = [e["name"] for e in conf["methods"] if e["name"] in df["method"].unique()]
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(meths) + 2), 4.8))
    ticks = []
    rng = np.random.default_rng(0)
    for i, meth in enumerate(meths):
        label, color = _style(conf, meth)
        K = 1 if METHODS[meth]["kind"] == "csv" else (ck if ck is not None else sel.get(meth, {}).get("best_k"))
        g = df[(df["method"] == meth) & (df["K"] == K)]
        if g.empty:
            continue
        piv = g.pivot_table(index=["run", "fold"], columns="split", values="ll")
        x0, x1 = i - 0.17, i + 0.17
        for _, r in piv.iterrows():
            j = rng.uniform(-0.04, 0.04)
            if {"train", "test"} <= set(piv.columns) and np.isfinite(r.get("train", np.nan)) and \
                    np.isfinite(r.get("test", np.nan)):
                ax.plot([x0 + j, x1 + j], [r["train"], r["test"]], color=color, alpha=0.35, lw=0.8)
        for sp, x, mk in (("train", x0, "o"), ("test", x1, "s")):
            if sp in piv:
                v = piv[sp].dropna()
                ax.scatter(np.full(len(v), x) + rng.uniform(-0.04, 0.04, len(v)), v, color=color, marker=mk, s=18,
                           edgecolor="k", lw=0.3, zorder=3)
                ax.hlines(v.mean(), x - 0.1, x + 0.1, color="k", lw=2, zorder=4)
        ticks.append((i, f"{label}\nK={K}"))
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], fontsize=8)
    ax.set_ylabel("log-likelihood per step")
    ax.set_title("Per-fold train (circle, left) vs test (square, right); bar = mean", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def paired_differences(df, reference, cl=0.95):
    """Test-LL difference (method - reference) over matching (K, run, fold); valid only with shared folds."""
    test = df[df["split"] == "test"]
    ref = test[test["method"] == reference].set_index(["K", "run", "fold"])["ll"]
    out = []
    for meth, g in test[test["method"] != reference].groupby("method"):
        if METHODS[meth]["kind"] == "csv":
            continue
        d = g.set_index(["K", "run", "fold"])["ll"] - ref
        d = d.dropna()
        for K, dk in d.groupby(level="K"):
            m, sd, se, lo, hi = _ci(dk.values, cl)
            out.append(dict(method=meth, K=K, n=len(dk), mean=m, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(out)


def plot_paired(diff, conf, reference, path):
    if diff.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for meth in [e["name"] for e in conf["methods"]]:
        g = diff[diff["method"] == meth]
        if g.empty:
            continue
        label, color = _style(conf, meth)
        g = g.sort_values("K")
        ax.plot(g["K"], g["mean"], "--" if meth.startswith("arhmm") else "-", marker="o", ms=4, color=color,
                label=label)
        ax.fill_between(g["K"], g["ci_lo"], g["ci_hi"], color=color, alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("K (hidden modes)")
    ax.set_ylabel(f"test LL - {_style(conf, reference)[0]} test LL")
    ax.set_title("Paired test-LL difference (same K / run / fold)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def plot_ll_comparison(conf, base_dir=None, methods=None):
    """
    Collect, summarise and plot train / test LLs for every method in conf["methods"].
    Returns (rows DataFrame, summary DataFrame, K selection dict).
    """
    cl = float(conf.get("ci", 0.95))
    out_dir = conf["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, conf.get("out_prefix", "ll_comparison"))

    df = collect_lls(conf, base_dir, methods)
    if df.empty:
        raise FileNotFoundError("no LL files found for any method; check out_dir / run_uid / seed / n_folds")
    # Every method that is plotted should have test LLs (train-only usually means test_index was empty).
    for meth, g in df.groupby("method"):
        if "test" not in set(g["split"]):
            warnings.warn(f"{meth}: no test LLs found")
    summary = summarize(df, cl)
    sel = select_k(summary)

    df.to_csv(prefix + "_ll_rows.csv", index=False)
    sel_df = pd.DataFrame([dict(method=m, **v) for m, v in sel.items()])
    summary.merge(sel_df, on="method", how="left").to_csv(prefix + "_ll_summary.csv", index=False)

    plot_ll_vs_k(summary, sel, conf, prefix + "_ll_vs_k.png")
    plot_folds_at_k(df, sel, conf, prefix + "_ll_folds.png")
    ref = conf.get("reference")
    if ref:
        if ref in set(df["method"]):
            plot_paired(paired_differences(df, ref, cl), conf, ref, prefix + "_ll_paired.png")
        else:
            warnings.warn(f"reference {ref} has no LL files; paired plot skipped")

    for m, v in sel.items():
        if METHODS[m]["kind"] == "csv":
            continue
        print(f"{m:12s} best test K = {v['best_k']} (LL {v['best_test']:.4f}), parsimonious K = {v['parsimonious_k']}")
    print("wrote", prefix + "_ll_*.png / .csv")
    return df, summary, sel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True, help="LL comparison YAML")
    parser.add_argument("--methods", nargs="*", default=None, help="Only these method names")
    args = parser.parse_args()
    plot_ll_comparison(_read_yaml(args.yaml), base_dir=os.path.dirname(os.path.abspath(args.yaml)),
                       methods=args.methods)
