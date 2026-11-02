"""
Manuscript summary across the three experiment blocks (baselines, decision scales, multi-agent).

    python -m naturenet.models.irl.summarize_experiments --out <dir>/manuscript \
        [--ll_summary <ll_comparison out>/<prefix>_ll_summary.csv] \
        [--scales_summary sharks=<scales out>/<run_uid>_scales_summary.csv whales=...] \
        [--ma_compare <dir>/ma_compare.csv] [--ma_scales <out>/ma_scales_summary.csv ...] \
        [--alpha 0.05]

Inputs are the CSVs the per-block tools already write (any subset):
  ll_summary      plot_ll_comparison   (method, K, split, mean, sem, best_k, parsimonious_k)
  scales_summary  plot_swirl_scales    (variant, order, latent, LL_test(_se), gain(_se), folds_won, N_star)
  ma_compare      plot_multiagent_compare  (HIAVI / SWIRL coupled vs independent, with / without kinematics)
  ma_scales       run_scales_multiagent    (decision-scale variants, steps 1-3)
Writes <out>_T1_baselines.csv, <out>_T2_scales.csv, <out>_T3_multiagent.csv, <out>_tables.md (all three as
Markdown) and <out>_overview.png (test LL vs K; decision-scale gains; multi-agent gains).
Decision rules (the test plan):
  T2 keep    gain > 2 SE and folds_won >= 0.8 x n_folds (4 of 5)
  T3 effect  gain_per_step > 0, animals improved > half, and p_gain (and p_alpha / p_beta when present) < alpha;
             rows without a null are marked "screen" (positive gain) or "-" instead.
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _cols(df, want):
    return df[[c for c in want if c in df.columns]]


def _md(df, floatfmt="{:.4g}"):
    if df is None or df.empty:
        return "_none_\n"
    cells = [[floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
              ("" if isinstance(v, (float, np.floating)) else str(v)) for v in row] for row in df.itertuples(index=False)]
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(r) + " |\n" for r in cells)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def t1_baselines(path):
    """Test LL per step (mean +/- SEM over folds x runs) per method and K; best and parsimonious K."""
    s = pd.read_csv(path)
    te = s[s["split"] == "test"].copy()
    te["test_ll"] = te.apply(lambda r: f"{r['mean']:.4f} +/- {r['sem']:.4f}", axis=1)
    wide = te.pivot_table(index=["method", "label"] if "label" in te else ["method"], columns="K",
                          values="test_ll", aggfunc="first")
    wide.columns = [f"K={k}" for k in wide.columns]
    sel = s.drop_duplicates("method").set_index("method")
    out = wide.reset_index()
    for c in ("best_k", "parsimonious_k", "best_test"):
        if c in sel:
            out[c] = out["method"].map(sel[c])
    return out, te


def t2_scales(paths):
    rows = []
    for lab, p in paths:
        s = pd.read_csv(p).assign(dataset=lab)
        if {"gain", "gain_se", "folds_won", "n_folds"} <= set(s.columns):
            k = (s["gain"] > 2 * s["gain_se"]) & (s["folds_won"] >= np.ceil(0.8 * s["n_folds"]))
            s["keep"] = np.where(s["variant"] == "N1", "ref", np.where(k, "yes", "no"))
        rows.append(s)
    if not rows:
        return pd.DataFrame()
    s = pd.concat(rows, ignore_index=True)
    return _cols(s, ["dataset", "variant", "order", "latent", "n_folds", "n_params", "LL_test", "LL_test_se",
                     "gain", "gain_se", "folds_won", "N_star", "keep", "fit_sec"])


def _verdict(r, alpha):
    ps = [r.get(k) for k in ("p_gain", "p_alpha", "p_beta")]
    ps = [p for p in ps if p is not None and np.isfinite(p)]
    imp, n = r.get("n_improved"), r.get("n_animals")
    pos = r["gain_per_step"] > 0 and (not np.isfinite(n) or imp > n / 2)
    if not ps:
        return "screen" if pos else "-"
    return "effect" if pos and max(ps) < alpha else "null"


def t3_multiagent(compare_paths, scales_paths, alpha):
    rows = []
    for p in compare_paths:
        d = pd.read_csv(p)
        imp = d["animals_improved"].astype(str).str.split("/", expand=True).astype(float)
        rows.append(pd.DataFrame(dict(
            source="coupled_base", model=d["run"], step="", scale_variant="base", K=d["K"],
            coupling_variant=d["variant"], gain_per_step=d["gain_per_step"], pooled_gain=d["pooled_gain_nats"],
            n_improved=imp[0], n_animals=imp[1], alpha_median=d["alpha_median"], beta_median=np.nan,
            p_gain=d.get("p_gain"), p_alpha=np.nan, p_beta=np.nan, sync_obs=d.get("sync_obs"),
            p_sync=d.get("p_sync"))))
    for p in scales_paths:
        d = pd.read_csv(p)
        rows.append(pd.DataFrame(dict(
            source="scales", model=d["backend"] + ":" + d["model"], step=d["step"], scale_variant=d["scale_variant"],
            K=d["K"], coupling_variant=d["coupling_variant"], gain_per_step=d["gain_per_step"],
            pooled_gain=d["pooled_gain"], n_improved=d["animals_improved"], n_animals=d["n_animals"],
            alpha_median=d["alpha_median"], beta_median=d.get("beta_median"), p_gain=d.get("p_gain"),
            p_alpha=d.get("p_alpha"), p_beta=d.get("p_beta"), sync_obs=d.get("sync_observed"),
            p_sync=d.get("sync_p"))))
    if not rows:
        return pd.DataFrame()
    t = pd.concat(rows, ignore_index=True)
    for c in ("p_gain", "p_alpha", "p_beta", "beta_median", "sync_obs", "p_sync"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["verdict"] = [_verdict(r, alpha) for r in t.to_dict("records")]
    t["animals_improved"] = [f"{int(a)}/{int(b)}" if np.isfinite(a) and np.isfinite(b) else ""
                             for a, b in zip(t["n_improved"], t["n_animals"])]
    return t.drop(columns=["n_improved", "n_animals"])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def overview(te, t2, t3, path):
    panels = [x for x in (("ll", te), ("scales", t2), ("ma", t3)) if x[1] is not None and not x[1].empty]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2), squeeze=False)
    for ax, (kind, d) in zip(axes[0], panels):
        if kind == "ll":
            for (m, g) in d.groupby("label" if "label" in d else "method"):
                g = g.sort_values("K")
                ax.errorbar(g["K"], g["mean"], yerr=g["sem"], marker="o", capsize=3, label=m)
            ax.set_xlabel("K (intentions)")
            ax.set_ylabel("held-out LL per step")
            ax.set_title("A. Baselines", fontsize=10)
            ax.legend(fontsize=6)
        elif kind == "scales":
            d = d[d["variant"] != "N1"].dropna(subset=["gain"])
            lab = (d["dataset"] + ":" + d["variant"]).tolist()
            col = ["C2" if k == "yes" else "0.6" for k in d.get("keep", ["no"] * len(d))]
            ax.barh(range(len(d)), d["gain"], xerr=2 * d["gain_se"], color=col)
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels(lab, fontsize=6)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("held-out gain vs N1 (per step; bars = 2 SE)")
            ax.set_title("B. Decision scales (green = keep)", fontsize=10)
        else:
            lab = (d["model"].astype(str).str.split(":").str[0] + " " + d["scale_variant"].astype(str) + " " +
                   d["coupling_variant"].astype(str) + " K" + d["K"].astype(str)).tolist()
            cmap = {"effect": "C3", "screen": "C1", "null": "0.6", "-": "0.85"}
            ax.barh(range(len(d)), d["gain_per_step"], color=[cmap[v] for v in d["verdict"]])
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels(lab, fontsize=5)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("held-out gain, coupled - independent (per step)")
            ax.set_title("C. Multi-agent (red effect, orange screen)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--ll_summary", default=None)
    ap.add_argument("--scales_summary", nargs="*", default=[], help="label=path (or path)")
    ap.add_argument("--ma_compare", nargs="*", default=[])
    ap.add_argument("--ma_scales", nargs="*", default=[])
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t1, te = t1_baselines(a.ll_summary) if a.ll_summary else (pd.DataFrame(), None)
    sp = [tuple(s.split("=", 1)) if "=" in s else (os.path.basename(s).split("_")[0], s) for s in a.scales_summary]
    t2 = t2_scales(sp)
    t3 = t3_multiagent(a.ma_compare, a.ma_scales, a.alpha)
    md = []
    for name, t, title in (("T1_baselines", t1, "T1. Single-agent baselines: held-out LL per step"),
                           ("T2_scales", t2, "T2. Decision scales (SWIRL-scales K-fold)"),
                           ("T3_multiagent", t3, "T3. Multi-agent coupling (leave-one-animal-out)")):
        if not t.empty:
            t.to_csv(f"{a.out}_{name}.csv", index=False)
        md.append(f"## {title}\n\n" + _md(t))
    with open(f"{a.out}_tables.md", "w") as f:
        f.write("\n".join(md))
    overview(te, t2, t3, f"{a.out}_overview.png")
    print("wrote", a.out + "_{T1_baselines,T2_scales,T3_multiagent}.csv, _tables.md, _overview.png")


if __name__ == "__main__":
    main()
