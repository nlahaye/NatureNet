"""
Per-intention kinematic emission figures (step length + turning angle) for any kinematic model:
HIAVI / SWIRL K-fold (model npz + LL file), kinematic ARHMM, or the multi-agent ma_final_K<K>.npz.

    # SWIRL / HIAVI K-fold: parameters from the model npz, posteriors + step/turn from the LL file
    python -m naturenet.models.irl.plot_kinematics --params <..._S1_temp_2_KF_0.npz> \
        --data <..._S1_iter2_LL1_test_KF_0.npz> --out kin_KF0 [--ll <LL files ...>]
    # multi-agent (one npz holds both; choose the fit tag)
    python -m naturenet.models.irl.plot_kinematics --params ma_final_K2.npz --tag indep --out kin_ma

Writes <out>.png (rows = intentions: posterior-weighted step histogram + fitted density, turn histogram +
fitted von Mises; title = label, median step, persistence), <out>_labels.csv, and with --ll
<out>_ll.png (held-out comparable LL vs action-only LL per file).
"""
import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from naturenet.models.irl.kinematics import (  # noqa: E402
    KinematicEmissions,
    gamma_logpdf,
    intention_table,
    kinematic_labels,
    lognorm_logpdf,
    vm_logpdf,
)

LCOL = {"exploit": "#d62728", "explore": "#1f77b4", "ambiguous": "0.5"}


def _prefix(p, tag=None):
    if tag:
        return f"kin_{tag}_"
    if "kin_p0" in p:
        return "kin_"
    c = sorted(k[:-2] for k in p if k.startswith("kin_") and k.endswith("_p0"))
    if not c:
        raise KeyError("no kinematic parameters (kin_*_p0) in the params npz")
    return c[0]


def load(params_path, data_path=None, tag=None):
    p = dict(np.load(params_path, allow_pickle=True))
    pre = _prefix(p, tag)
    km = KinematicEmissions.from_params(p, prefix=pre)
    d = p if data_path is None else dict(np.load(data_path, allow_pickle=True))
    gk = f"gamma_{tag}" if tag and f"gamma_{tag}" in d else "gamma"
    if "kin_step" not in d or gk not in d:
        raise KeyError(f"{data_path or params_path} lacks kin_step / kin_turn / {gk} (re-run with kinematics)")
    return km, np.asarray(d["kin_step"], float), np.asarray(d["kin_turn"], float), np.asarray(d[gk], float)


def step_pdf(km, k, x):
    if km.step_dist == "gamma":
        return np.exp(gamma_logpdf(x, km.s1[k], km.s2[k]))
    return np.exp(lognorm_logpdf(x, km.s1[k], km.s2[k]))


def plot(km, step, turn, gamma, path, step_feature="speed_kmh", labels=None, title=""):
    K = km.K
    labels = labels if labels is not None else kinematic_labels(km, step_feature)
    g = gamma.reshape(-1, K)
    x, th = step.ravel(), turn.ravel()
    pos = np.isfinite(x) & (x > 0)
    hi = np.nanpercentile(x[pos], 99) if pos.any() else 1.0
    xs = np.linspace(hi / 400, hi, 300)
    ts = np.linspace(-np.pi, np.pi, 300)
    unit = {"speed_kmh": "speed (km/h)", "step_km": "step length (km)", "diffusive": "step / sqrt(dt) (km/sqrt(h))"}
    fig, axes = plt.subplots(K, 2, figsize=(10, 2.6 * K + 0.6), squeeze=False)
    for k in range(K):
        r = labels.iloc[k]
        col = LCOL.get(r["label"], "k")
        ax = axes[k, 0]
        w = g[pos, k]
        if w.sum() > 0:
            ax.hist(x[pos], bins=np.linspace(0, hi, 40), weights=w, density=True, color=col, alpha=0.35)
        ax.plot(xs, step_pdf(km, k, xs), color=col, lw=2)
        ax.set_xlim(0, hi)
        ax.set_ylabel("density")
        ax.set_title(f"z={k} [{r['label']}]  median {r['step_median']:.2f}, P(stationary) {r['p_stationary']:.3f}, "
                     f"posterior mass {g[:, k].sum() / max(g.sum(), 1e-12):.2f}", fontsize=9)
        ax = axes[k, 1]
        ok = np.isfinite(th)
        if g[ok, k].sum() > 0:
            ax.hist(th[ok], bins=np.linspace(-np.pi, np.pi, 37), weights=g[ok, k], density=True, color=col, alpha=0.35)
        ax.plot(ts, np.exp(vm_logpdf(ts, km.mu[k], km.kappa[k])), color=col, lw=2)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels(["-180", "-90", "0", "90", "180"])
        ax.set_title(f"turn mu {np.degrees(km.mu[k]):.0f} deg, kappa {km.kappa[k]:.2f}, "
                     f"persistence {r['persistence']:.2f}", fontsize=9)
    axes[-1, 0].set_xlabel(unit.get(step_feature, step_feature))
    axes[-1, 1].set_xlabel("turning angle (deg)")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return labels


def plot_ll(files, path):
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "LL_action" not in d.files:
            continue
        rows.append(dict(file=os.path.basename(f), LL_comparable=float(d["LL"]), LL_action=float(d["LL_action"]),
                         LL_joint=float(d["LL_joint"]), LL_kin_null=float(d["LL_kin_null"])))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(df)), 3.8))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["LL_action"], 0.4, color="0.6", label="action-only (kinematics marginalised)")
    ax.bar(x + 0.2, df["LL_comparable"], 0.4, color="#2ca02c", label="kinematic, comparable (joint - pooled kin)")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(".npz", "")[-28:] for f in df["file"]], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("LL per step (nats)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(path.replace(".png", ".csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="npz with kin_* (or kin_<tag>_*) parameters")
    ap.add_argument("--data", default=None, help="npz with gamma, kin_step, kin_turn (default: --params)")
    ap.add_argument("--tag", default=None, help="multi-agent fit tag (indep | temporal | proximity)")
    ap.add_argument("--step_feature", default="speed_kmh")
    ap.add_argument("--speed_threshold", type=float, default=None)
    ap.add_argument("--persistence_threshold", type=float, default=None)
    ap.add_argument("--ll", nargs="*", default=None, help="LL files for the comparable-vs-action-only bar chart")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    km, step, turn, gamma = load(a.params, a.data, a.tag)
    labels = kinematic_labels(km, a.step_feature, a.speed_threshold, a.persistence_threshold)
    plot(km, step, turn, gamma, a.out + ".png", a.step_feature, labels,
         title=os.path.basename(a.params) + (f" [{a.tag}]" if a.tag else ""))
    labels.to_csv(a.out + "_labels.csv", index=False)
    print(labels.to_string(index=False))
    print(intention_table(km, a.step_feature).to_string(index=False))
    if a.ll:
        df = plot_ll(a.ll, a.out + "_ll.png")
        if df is not None:
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
