"""
Projection robustness of the environment levels (Rep_Taxonomy projection / silhouette protocol).

For every level (parent, k30, k20, k12, seascape, ...) the silhouette of its state labels at the used fixes is
computed in the native z-scored environment space and in 2-D projections (PCA, UMAP, t-SNE; UMAP / t-SNE are
skipped with a note if umap-learn / sklearn TSNE are unavailable), over n_runs subsamples. A level whose state
separation holds across projections is a robust partition of the raw environment; one that only separates in one
projection is projection-dependent. Uses the same spec as env_geometry (sample_inputs_yaml, channels, levels).

  projections: [native, pca, umap, tsne]
  projection_n: 3000
  n_runs: 3
Outputs: projection_silhouette.csv (level, projection, run, silhouette, n), projection_summary.csv

    python -m naturenet.drivers.projection_robustness -y config/v2/drivers/generated/whales/specs/geometry.yaml [--level k30]
"""
import argparse
import os

import numpy as np
import pandas as pd

from naturenet.drivers.common import abs_cfg, atomic_csv, is_original, read_yaml
from naturenet.drivers.env_geometry import _sub, level_labels, load_points, zscore


def project(X, how, seed=0):
    if how == "native":
        return X
    if how == "pca":
        from sklearn.decomposition import PCA
        return PCA(2, random_state=seed).fit_transform(X)
    if how == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(2, random_state=seed, init="pca", perplexity=min(30, max(5, len(X) // 100))).fit_transform(X)
    if how == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
    raise ValueError(how)


def run(spec, only=None):
    from sklearn.metrics import silhouette_score
    sconf = read_yaml(abs_cfg(spec["sample_inputs_yaml"]))
    df_dir, uid = sconf["df_dir"], sconf["df_run_uid"]
    out = spec["out_dir"]
    P, sel = load_points(df_dir, uid, spec.get("channels"), is_original, spec.get("env_transforms"))
    P = P.dropna(subset=sel).reset_index(drop=True)
    n = int(spec.get("projection_n", 3000))
    rows = []
    for lv, d in (spec.get("levels") or {}).items():
        if only and lv not in only:
            continue
        for r in range(int(spec.get("n_runs", 3))):
            S = P.iloc[_sub(len(P), n, int(spec.get("seed", 0)) + r)].reset_index(drop=True)
            lab = level_labels(d, uid, S)
            ok = lab >= 0
            if ok.sum() < 20 or len(np.unique(lab[ok])) < 2:
                continue
            X = zscore(S[ok], sel, P)
            for how in spec.get("projections", ["native", "pca", "umap", "tsne"]):
                try:
                    Y = project(X, how, r)
                    s, note = float(silhouette_score(Y, lab[ok])), ""
                except Exception as e:
                    s, note = np.nan, f"{type(e).__name__}: {str(e)[:120]}"
                rows.append(dict(level=lv, projection=how, run=r, silhouette=s, n=int(ok.sum()),
                                 n_states=int(len(np.unique(lab[ok]))), note=note))
    d = pd.DataFrame(rows)
    tag = "" if not only else "_" + "_".join(only)
    atomic_csv(d, os.path.join(out, f"projection_silhouette{tag}.csv"))
    if len(d):
        s = d.groupby(["level", "projection"])["silhouette"].agg(["mean", "std", "count"]).reset_index()
        atomic_csv(s, os.path.join(out, f"projection_summary{tag}.csv"))
    print(f"projection_robustness: {len(d)} rows -> {out}")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--level", action="append")
    a = ap.parse_args(argv)
    run(read_yaml(abs_cfg(a.yaml)), a.level)


if __name__ == "__main__":
    main()
