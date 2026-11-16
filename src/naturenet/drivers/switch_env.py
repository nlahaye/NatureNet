"""
Test 7 (descriptive part) - switch-triggered environmental averages of the decoded intentions.

For every held-out sequence (K-fold test files, so the decoder never saw the animal) the MAP intentions z_t
(learnt_zs of the HIAVI LL_test files) give switch times t (z_t != z_t-1). For each switch type (from -> to) and
environmental channel (z-scored covariates at the steps, env_covariates.env_steps):

  level      mean covariate at the switch step vs at matched non-switch steps (same "from" intention, same
             animal, z_t = z_t-1); difference and an animal-block permutation p (switch labels shuffled within animal)
  change     mean covariate change over the `window` steps before the switch (x_t - x_t-w) vs matched steps
  profile    triggered average at lags -window .. +window (for plots)

This is descriptive (which conditions coincide with switching); the predictive test is the decision-scale variant
with switch_history kind "env" (held-out LL gain over the same model without env covariates, env_scale_grid).

YAML:
  out_dir: /data/nlahaye/NatureNet/Blue_Whale_v2/drivers/switch_env/
  channels: [sst, grad_mag_sst, log10_CHL, adt, eke, depth_m]     # null = all
  window: 3
  n_perm: 500
  models:
    - {label: hiql_kin_k5, yaml: config/v2/swirl/whales_hiavi_kfold_kin.yaml, K: 5, sweep: True}
Outputs: switch_env_<label>.csv, switch_profile_<label>.csv, switch_profile_<label>.png

    python -m naturenet.drivers.switch_env -y config/v2/drivers/generated/whales/specs/switch_env.yaml [--model hiql_kin_k5]
"""
import argparse
import os

import numpy as np
import pandas as pd

from naturenet.drivers.common import (abs_cfg, atomic_csv, base_id, best_run, bh, folds, hiavi_data, hiavi_files,
                                      read_yaml, run_conf)


def decoded(conf, data, K, sweep=True):
    """(N, T) held-out MAP intentions (-1 where not decoded / padded), from the K-fold LL_test files."""
    fl = folds(conf, data)
    run = best_run(conf, K, len(fl)) if sweep else 0
    Z = np.full(data["hiavi_s"].shape, -1, np.int64)
    labels = None
    for k, (_, te) in enumerate(fl):
        f = hiavi_files(conf, K, run, k, sweep=sweep)
        if not os.path.exists(f["ll_test"]):
            continue
        z = np.load(f["ll_test"], allow_pickle=True)
        zs = np.asarray(z["learnt_zs"])
        Z[np.asarray(te), :zs.shape[1]] = zs
        if labels is None and os.path.exists(f["model"]):
            m = np.load(f["model"], allow_pickle=True)
            labels = np.asarray(m["kin_labels"]).astype(str) if "kin_labels" in m.files else None
    return Z, labels


def events(Z, mask):
    """Switch (n, t) indices and non-switch candidates; t >= 1, both steps real and decoded."""
    prev = np.concatenate([np.full((Z.shape[0], 1), -1), Z[:, :-1]], 1)
    ok = mask & (Z >= 0) & (prev >= 0)
    ok[:, 0] = False
    return ok & (Z != prev), ok & (Z == prev), prev


def analyse(E, Z, mask, animals, names, window=3, n_perm=500, seed=0):
    rng = np.random.default_rng(seed)
    sw, st, prev = events(Z, mask)
    rows, prof = [], []
    K = int(Z.max()) + 1
    an = np.repeat(np.asarray(animals)[:, None], Z.shape[1], 1)
    dE = np.full_like(E, np.nan)
    dE[:, window:] = E[:, window:] - E[:, :-window]
    for a in range(K):
        base = st & (prev == a)
        for b in range(K):
            if a == b:
                continue
            ev = sw & (prev == a) & (Z == b)
            if ev.sum() < 3 or base.sum() < 3:
                continue
            cand = ev | base
            lab = ev[cand]
            grp = an[cand]
            for j, n in enumerate(names):
                for kind, X in (("level", E[..., j]), ("change", dE[..., j])):
                    x = X[cand]
                    fin = np.isfinite(x)
                    if fin.sum() < 6 or lab[fin].sum() < 3 or (~lab[fin]).sum() < 3:
                        continue
                    xx, ll, gg = x[fin], lab[fin], grp[fin]
                    obs = xx[ll].mean() - xx[~ll].mean()
                    null = np.empty(n_perm)
                    ug = np.unique(gg)
                    idx = [np.flatnonzero(gg == g) for g in ug]
                    for r in range(n_perm):
                        perm = ll.copy()
                        for ii in idx:
                            perm[ii] = rng.permutation(ll[ii])
                        null[r] = xx[perm].mean() - xx[~perm].mean() if perm.any() and (~perm).any() else 0.0
                    p = float((1 + (np.abs(null) >= abs(obs)).sum()) / (1 + n_perm))
                    rows.append(dict(from_z=a, to_z=b, channel=n, kind=kind, n_switch=int(ll.sum()),
                                     n_matched=int((~ll).sum()), n_animals=int(len(np.unique(gg[ll]))),
                                     mean_switch=float(xx[ll].mean()), mean_matched=float(xx[~ll].mean()),
                                     diff=float(obs), p_perm=p))
            # triggered profile
            nn, tt = np.nonzero(ev)
            for lag in range(-window, window + 1):
                t2 = tt + lag
                okk = (t2 >= 0) & (t2 < Z.shape[1])
                vals = E[nn[okk], t2[okk]]
                for j, n in enumerate(names):
                    v = vals[:, j]
                    v = v[np.isfinite(v)]
                    if len(v):
                        prof.append(dict(from_z=a, to_z=b, channel=n, lag=lag, mean=float(v.mean()),
                                         se=float(v.std() / np.sqrt(len(v))), n=int(len(v))))
    d = pd.DataFrame(rows)
    if len(d):
        d["q_bh"] = bh(d["p_perm"].to_numpy())
    return d, pd.DataFrame(prof)


def plot_profile(prof, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not len(prof):
        return
    chans = list(prof["channel"].unique())
    types = prof[["from_z", "to_z"]].drop_duplicates().to_numpy()
    fig, axes = plt.subplots(1, len(chans), figsize=(2.6 * len(chans), 2.6), squeeze=False)
    for ax, c in zip(axes[0], chans):
        for a, b in types:
            g = prof[(prof["channel"] == c) & (prof["from_z"] == a) & (prof["to_z"] == b)].sort_values("lag")
            ax.plot(g["lag"], g["mean"], marker=".", lw=1, label=f"{a}->{b}")
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_title(c, fontsize=8)
        ax.set_xlabel("steps from switch", fontsize=7)
    axes[0][0].set_ylabel("z-scored covariate", fontsize=7)
    axes[0][-1].legend(fontsize=6, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_model(spec, ms):
    from naturenet.drivers.env_covariates import env_steps
    conf = run_conf(dict(yaml=ms["yaml"], set=ms.get("set") or {}))
    data = hiavi_data(conf)
    K = int(ms.get("K", conf.get("n_hidden", 5)))
    Z, labels = decoded(conf, data, K, bool(ms.get("sweep", True)))
    E, names = env_steps(conf, data, spec.get("channels") or ())
    keys = [m["key"] for m in data["meta"]]
    orig = np.array([base_id(k) == k for k in keys])
    mask = np.asarray(data["hiavi_mask"]) & orig[:, None]
    d, prof = analyse(E, Z, mask, [base_id(g) for g in data["groups"]], names, int(spec.get("window", 3)),
                      int(spec.get("n_perm", 500)), int(spec.get("seed", 0)))
    if labels is not None and len(d):
        d["from_label"] = [labels[i] if i < len(labels) else "" for i in d["from_z"]]
        d["to_label"] = [labels[i] if i < len(labels) else "" for i in d["to_z"]]
    out = spec["out_dir"]
    atomic_csv(d, os.path.join(out, f"switch_env_{ms['label']}.csv"))
    atomic_csv(prof, os.path.join(out, f"switch_profile_{ms['label']}.csv"))
    plot_profile(prof, os.path.join(out, f"switch_profile_{ms['label']}.png"),
                 f"{ms['label']}: switch-triggered environment (held-out decoding; fold intentions may permute)")
    print(f"switch_env {ms['label']}: {len(d)} tests -> {out}")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--model", default=None)
    a = ap.parse_args(argv)
    spec = read_yaml(abs_cfg(a.yaml))
    for ms in spec["models"]:
        if a.model is None or ms["label"] == a.model:
            run_model(spec, ms)


if __name__ == "__main__":
    main()
