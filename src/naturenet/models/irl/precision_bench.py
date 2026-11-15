"""
Precision trade-off test for the JAX fits (SWIRL get_stats_2, or one decision-scale variant): runs the same
(K, run, fold) once per precision tier (jax_precision.TIERS: fp64 = current default, fp32, tf32, bf16) in separate
processes, side by side on different GPUs, and compares every output array with the fp64 run.

    python -m naturenet.models.irl.precision_bench -y <swirl_kfold.yaml> --K 3 --run 0 --fold 0 \
        --out /data/nlahaye/NatureNet/precision/sharks --gpus 0 1 2 3
    python -m naturenet.models.irl.precision_bench -y <swirl_scales.yaml> --runner scales --variant N2 --fold 0 ...

Each tier writes to <out>/<tier>/ (a copy of the YAML with out_dir changed, folds_dir = the original folds, and
symlinks to the ARHMM inits *_arhmm_s*_KF_<fold>.npz of the original out_dir), so the inputs are identical.
Outputs: <out>/precision_compare.csv (tier, file, key: abs / relative difference vs fp64, integer agreement),
<out>/precision_summary.md (wall time, speed-up, log-likelihood differences per tier) and <out>/<tier>/run.log.

Reading the results: SWIRL's M-step uses L-BFGS with line searches and convergence tolerances set for float64;
log-sum-exp recursions over long sequences and value iteration accumulate rounding. With float32 expect small LL
differences (|dLL| per step << the K / model differences you compare), occasional different L-BFGS paths (then
parameters differ but LLs stay close) and, at bf16 / tf32, larger drift in value iteration. A tier is usable when its
held-out LL differences are well below the gaps between the models being compared and the decoded states agree.
Wall times include JIT compilation (the same for every tier, within noise).
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from naturenet.models.irl.jax_precision import TIERS


def _tier_yaml(conf, out, tier, fold):
    d = os.path.join(out, tier)
    os.makedirs(d, exist_ok=True)
    c = dict(conf)
    src = conf["out_dir"]
    c["out_dir"] = d + "/"
    c.setdefault("folds_dir", src)
    c["overwrite"] = True
    for f in glob.glob(os.path.join(src, f"*arhmm_s*_KF_{fold}.npz")):
        link = os.path.join(d, os.path.basename(f))
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(f), link)
    p = os.path.join(d, "config.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(c, f, sort_keys=False)
    return d, p


def _cmd(a, yml):
    if a.runner == "swirl":
        return [sys.executable, "-m", "naturenet.models.irl.get_stats_2", "-y", yml, "--n_hidden", str(a.K),
                "--run", str(a.run), "--fold", str(a.fold)]
    return [sys.executable, "-m", "naturenet.models.irl.run_swirl_scales_kfold", "-y", yml, "--fold", str(a.fold),
            "--variant", *a.variant]


def run_tiers(a, conf):
    gpus = [str(g) for g in (a.gpus or [""])]
    procs, t0, info = {}, {}, {}
    queue = list(a.tiers)
    free = list(gpus)
    while queue or procs:
        while queue and free:
            tier = queue.pop(0)
            gpu = free.pop(0)
            d, yml = _tier_yaml(conf, a.out, tier, a.fold)
            env = dict(os.environ, **TIERS[tier], XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1")
            env.pop("JAX_COMPILATION_CACHE_DIR", None)      # same JIT cost for every tier
            if gpu != "":
                env["CUDA_VISIBLE_DEVICES"] = gpu
            log = open(os.path.join(d, "run.log"), "w")
            procs[tier] = (subprocess.Popen(_cmd(a, yml), env=env, stdout=log, stderr=subprocess.STDOUT), log, gpu, d)
            t0[tier] = time.time()
            print(f"started {tier} on gpu {gpu or '-'}")
        for tier, (p, log, gpu, d) in list(procs.items()):
            if p.poll() is not None:
                log.close()
                info[tier] = dict(tier=tier, rc=p.returncode, wall_sec=time.time() - t0[tier], gpu=gpu, dir=d)
                print(f"finished {tier} rc={p.returncode} in {info[tier]['wall_sec']:.0f} s")
                free.append(gpu)
                del procs[tier]
        time.sleep(2)
    return info


def _outputs(d):
    return {os.path.basename(f): f for f in glob.glob(os.path.join(d, "*.npz")) if not os.path.islink(f)}


def compare_arrays(ref, x):
    """Difference record of one array against the fp64 reference."""
    ref, x = np.asarray(ref), np.asarray(x)
    rec = dict(shape=str(ref.shape), dtype=str(ref.dtype))
    if ref.shape != x.shape:
        return dict(rec, kind="shape_mismatch")
    if ref.dtype.kind in "iub":
        return dict(rec, kind="int", agree=float(np.mean(ref == x)) if ref.size else np.nan)
    if ref.dtype.kind not in "fc":
        return dict(rec, kind="other")
    r, v = ref.astype(np.float64), x.astype(np.float64)
    fin = np.isfinite(r) & np.isfinite(v)
    if not fin.any():
        return dict(rec, kind="float", abs_diff=np.nan, rel_diff=np.nan, nonfinite_mismatch=int(np.sum(~fin)))
    dif = np.abs(r[fin] - v[fin])
    scale = max(float(np.max(np.abs(r[fin]))), 1e-300)
    out = dict(rec, kind="scalar" if r.size == 1 else "float", abs_diff=float(dif.max()),
               rel_diff=float(dif.max() / scale), mean_abs_diff=float(dif.mean()),
               nonfinite_mismatch=int(np.sum(np.isfinite(r) != np.isfinite(v))))
    if r.size == 1:
        out.update(ref=float(r.ravel()[0]), value=float(v.ravel()[0]))
    return out


def compare(info, ref_tier="fp64"):
    ref = _outputs(info[ref_tier]["dir"])
    rows = []
    for tier, inf in info.items():
        if tier == ref_tier or inf["rc"] != 0:
            continue
        out = _outputs(inf["dir"])
        for name, fr in sorted(ref.items()):
            if name not in out:
                rows.append(dict(tier=tier, file=name, key="", kind="missing"))
                continue
            a, b = np.load(fr, allow_pickle=True), np.load(out[name], allow_pickle=True)
            for k in a.files:
                if k not in b.files:
                    rows.append(dict(tier=tier, file=name, key=k, kind="missing"))
                    continue
                try:
                    rows.append(dict(tier=tier, file=name, key=k, **compare_arrays(a[k], b[k])))
                except Exception as e:  # noqa: BLE001  (object arrays etc.)
                    rows.append(dict(tier=tier, file=name, key=k, kind=f"skipped: {type(e).__name__}"))
    return pd.DataFrame(rows)


def summarize(info, df, path, ref_tier="fp64"):
    lines = ["# Precision tiers vs fp64", "", "| tier | rc | wall (s) | speed-up |", "|---|---|---|---|"]
    w0 = info.get(ref_tier, {}).get("wall_sec")
    for tier, inf in info.items():
        sp = f"{w0 / inf['wall_sec']:.2f}x" if w0 and inf["rc"] == 0 and np.isfinite(inf["wall_sec"]) else ""
        lines.append(f"| {tier} | {inf['rc']} | {inf['wall_sec']:.0f} | {sp} |")
    if len(df):
        ll = df[df["key"].str.contains("ll|LL|loglik", regex=True, na=False) & df["kind"].isin(["scalar", "float"])]
        if len(ll):
            lines += ["", "## Log-likelihood outputs (max |difference| vs fp64)", "",
                      "| tier | file | key | fp64 | tier value | abs diff | rel diff |", "|---|---|---|---|---|---|---|"]
            for r in ll.itertuples():
                sc_ = r.kind == "scalar"
                lines.append(f"| {r.tier} | {r.file} | {r.key} | {f'{r.ref:.6g}' if sc_ else 'array'} | "
                             f"{f'{r.value:.6g}' if sc_ else 'array'} | {r.abs_diff:.3g} | {r.rel_diff:.3g} |")
        ints = df[df["kind"] == "int"]
        if len(ints):
            lines += ["", "## Integer outputs (fraction equal to fp64)", "", "| tier | file | key | agree |",
                      "|---|---|---|---|"]
            lines += [f"| {r.tier} | {r.file} | {r.key} | {r.agree:.4f} |" for r in ints.itertuples()]
        fl = df[df["kind"] == "float"].groupby("tier")[["rel_diff"]].agg(["median", "max"])
        if len(fl):
            fl.columns = ["median_rel_diff", "max_rel_diff"]
            lines += ["", "## All float arrays: relative difference (max |d| / max |fp64|)", "",
                      "| tier | median | max |", "|---|---|---|"]
            lines += [f"| {t} | {r.median_rel_diff:.3g} | {r.max_rel_diff:.3g} |" for t, r in fl.iterrows()]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--runner", choices=["swirl", "scales"], default="swirl")
    ap.add_argument("--K", type=int, default=None, help="swirl: number of hidden modes")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--variant", nargs="*", default=["N1"], help="scales: variant tags (parents first)")
    ap.add_argument("--tiers", nargs="*", default=["fp64", "fp32", "tf32", "bf16"])
    ap.add_argument("--gpus", nargs="*", default=None, help="GPU ids for the tiers (run in parallel)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare_only", action="store_true", help="re-read <out>/<tier>/ (no runs)")
    a = ap.parse_args(argv)
    if "fp64" not in a.tiers:
        a.tiers = ["fp64"] + list(a.tiers)
    with open(a.yaml) as f:
        conf = yaml.safe_load(f)
    if a.runner == "swirl":
        if a.K is None:
            ap.error("--K is required for --runner swirl")
        conf = dict(conf, min_latent=a.K, max_latent=a.K)
    os.makedirs(a.out, exist_ok=True)
    if a.compare_only:
        info = {t: dict(tier=t, rc=0, wall_sec=np.nan, gpu="", dir=os.path.join(a.out, t)) for t in a.tiers}
        runs = os.path.join(a.out, "precision_runs.csv")
        if os.path.exists(runs):
            for r in pd.read_csv(runs).to_dict("records"):
                if r["tier"] in info:
                    info[r["tier"]].update(rc=int(r["rc"]), wall_sec=float(r["wall_sec"]))
    else:
        info = run_tiers(a, conf)
        pd.DataFrame(info.values()).to_csv(os.path.join(a.out, "precision_runs.csv"), index=False)
    if info["fp64"]["rc"] != 0:
        sys.exit("fp64 run failed: see " + os.path.join(info["fp64"]["dir"], "run.log"))
    df = compare(info)
    df.to_csv(os.path.join(a.out, "precision_compare.csv"), index=False)
    print("wrote", summarize(info, df, os.path.join(a.out, "precision_summary.md")))


if __name__ == "__main__":
    main()
