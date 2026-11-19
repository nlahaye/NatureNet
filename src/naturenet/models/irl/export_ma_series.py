"""
Export the full-data fits of a multi-agent run (ma_final_K<K>.npz of run_hiavi_multiagent / run_swirl_multiagent /
run_scales_multiagent) as per-animal series CSVs, so every single-animal downstream tool (decision-context similarity,
habitat tests, exploit / explore, time-series correlation, animations with kind: series) reads multi-agent fits
unchanged. Generic: any multi-agent config / species.

    python -m naturenet.models.irl.export_ma_series --npz <out_dir>/<ma_subdir>/ma_final_K5.npz \
        --ref_series <root>/final/hiql_kin/plots/series_hiavi_s1 --out <out_dir>/<ma_subdir>/series_K5 \
        [--fits indep temporal best] [--df_run_uid hammerhead_v2] [--summary <out_dir>/<ma_subdir>/ma_summary.json]

Writes <out>/<fit>/<uid>_time_series_pd.csv (fit = indep, every coupled variant in the npz, or "best" = the variant
with the largest pooled held-out gain in ma_summary.json, selection.best_variant; written under its own name and
linked as <out>/best) plus <out>/export_summary.csv (rows per animal and fit, steps matched to the reference series).

Rows: the reference series rows (the single-animal export of the same tracks: uid, segment, seg_pos, timestamp,
latitude, longitude, env, environment columns, kinematics) of the steps the multi-agent model decoded, with
  latent        argmax of gamma_<fit>
  p_exploit     p_exploit_<fit> (kinematic multi-agent fits; NaN otherwise)
  latent_class  kin_labels_<fit>[latent] (exploit / explore / ambiguous; empty without kinematics)
  ma_fit, ma_K  provenance
reward_likelihood is set to NaN (the reference value belongs to the single-animal model). Steps are joined on
(uid, timestamp rounded to --round_s seconds); augmented copies (original = False) are dropped. Steps the
multi-agent model decoded but the reference does not hold are counted (unmatched) and dropped.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from naturenet.models.irl.animate_tracks import _naive, _short

DROP_REF = ("latent", "latent_class", "p_exploit", "reward_likelihood", "ma_fit", "ma_K")


def fits_in(z):
    return sorted(k[len("gamma_"):] for k in z.files if k.startswith("gamma_"))


def best_variant(summary, K):
    if not summary or not os.path.exists(summary):
        return None
    with open(summary) as f:
        s = json.load(f)
    sel = ((s.get("K") or {}).get(str(K)) or {}).get("selection") or {}
    return sel.get("best_variant")


def ma_steps(z, fit, df_run_uid=None):
    """Tidy (uid, t_key, latent, p_exploit, latent_class) of the original-track steps of one fit."""
    gam, valid = np.asarray(z[f"gamma_{fit}"]), np.asarray(z["valid"], bool)
    keep = np.asarray(z["original"], bool) if "original" in z.files else np.ones(len(gam), bool)
    m = valid & keep[:, None]
    an = np.broadcast_to(np.asarray(z["animal"]).astype(str)[:, None], valid.shape)[m]
    t = pd.Timestamp("1970-01-01") + pd.to_timedelta(np.asarray(z["times"])[m], unit="D")
    lat = np.argmax(gam, -1)[m]
    d = pd.DataFrame(dict(uid=[_short(a, df_run_uid) for a in an], timestamp=t.to_numpy(), latent=lat))
    pe = f"p_exploit_{fit}"
    d["p_exploit"] = np.asarray(z[pe], float)[m] if pe in z.files else np.nan
    kl = f"kin_labels_{fit}"
    if kl in z.files:
        labs = np.asarray(z[kl]).astype(str)
        d["latent_class"] = [labs[k] if 0 <= k < len(labs) else "" for k in d["latent"]]
    else:
        d["latent_class"] = ""
    return d


def _tkey(ts, round_s):
    t = pd.to_datetime(pd.Series(_naive(ts)))                  # unit-safe (datetime64[s / us / ns])
    return np.round((t - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=round_s)).astype("int64").to_numpy()


def load_ref(ref_dir):
    files = sorted(glob.glob(os.path.join(ref_dir, "*_time_series_pd.csv")))
    if not files:
        raise FileNotFoundError(f"no *_time_series_pd.csv in {ref_dir}")
    return {os.path.basename(f)[:-len("_time_series_pd.csv")]: f for f in files}


def export(npz, ref_series, out, fits=("all",), df_run_uid=None, summary=None, round_s=60, K=None):
    z = np.load(npz, allow_pickle=True)
    have = fits_in(z)
    if K is None:
        b = os.path.basename(npz)
        K = int(b.split("_K")[1].split(".")[0].split("_")[0]) if "_K" in b else None
    want, best = [], None
    for f in fits:
        if f == "all":
            want += have
        elif f == "best":
            best = best_variant(summary, K)
            if best is None:
                print(f"WARNING: no selection.best_variant for K={K} in {summary}; 'best' skipped")
            elif best not in have:
                print(f"WARNING: best variant {best} not in {npz} (fits: {have}); 'best' skipped")
                best = None
            else:
                want.append(best)
        elif f in have:
            want.append(f)
        else:
            raise KeyError(f"{npz}: no gamma_{f} (fits: {have})")
    want = list(dict.fromkeys(want))
    refs = load_ref(ref_series)
    rows = []
    for fit in want:
        d = ma_steps(z, fit, df_run_uid)
        d["tk"] = _tkey(d["timestamp"], round_s)
        d = d.drop_duplicates(["uid", "tk"])
        od = os.path.join(out, fit)
        os.makedirs(od, exist_ok=True)
        for uid, g in d.groupby("uid"):
            if uid not in refs:
                rows.append(dict(fit=fit, uid=uid, n_ma=len(g), n_matched=0, n_ref=0, note="no reference series"))
                continue
            r = pd.read_csv(refs[uid], dtype={"uid": str})
            r["tk"] = _tkey(r["timestamp"], round_s)
            r = r.drop(columns=[c for c in DROP_REF if c in r])
            j = r.merge(g[["tk", "latent", "p_exploit", "latent_class"]], on="tk", how="inner")
            j["reward_likelihood"] = np.nan
            j["ma_fit"], j["ma_K"] = fit, K
            j.drop(columns=["tk"]).to_csv(os.path.join(od, f"{uid}_time_series_pd.csv"), index=False)
            rows.append(dict(fit=fit, uid=uid, n_ma=len(g), n_matched=len(j), n_ref=len(r), note=""))
    if best is not None:
        link = os.path.join(out, "best")
        if os.path.islink(link):
            os.remove(link)
        if not os.path.exists(link):
            os.symlink(best, link)
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(out, "export_summary.csv"), index=False)
    if len(rep):
        t = rep.groupby("fit")[["n_ma", "n_matched"]].sum()
        for fit, r in t.iterrows():
            frac = r.n_matched / max(r.n_ma, 1)
            print(f"{fit}: {int(r.n_matched)} / {int(r.n_ma)} multi-agent steps matched ({frac:.1%})"
                  + ("  WARNING: < 95% matched (reference series from another run / time rounding?)"
                     if frac < 0.95 else ""))
    if best is not None:
        print(f"best = {best} -> {os.path.join(out, 'best')}")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ref_series", required=True, help="single-animal series dir of the same tracks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fits", nargs="+", default=["all"], help="indep, variant names, all, best")
    ap.add_argument("--summary", default=None, help="ma_summary.json (default: next to --npz)")
    ap.add_argument("--df_run_uid", default=None, help="prefix stripped from the multi-agent animal ids")
    ap.add_argument("--K", type=int, default=None, help="default: from the npz name")
    ap.add_argument("--round_s", type=int, default=60)
    a = ap.parse_args(argv)
    summary = a.summary or os.path.join(os.path.dirname(a.npz), "ma_summary.json")
    export(a.npz, a.ref_series, a.out, a.fits, a.df_run_uid, summary, a.round_s, a.K)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
