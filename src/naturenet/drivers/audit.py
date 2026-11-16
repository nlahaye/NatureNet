"""
Experiment auditor for the environmental-driver study (read-only).

Two parts, per species:

  progress   every job of the launcher plan: done | stale (done marker of a different command) | failed | running
             (claimed) | blocked (a dependency failed) | todo; rolled up per test x level x fold with a "partial"
             state for groups that are neither all done nor untouched
  integrity  design checks that make the comparisons valid:
               v2_only          every run config is under config/v2, data paths are v2 roots, no /mnt/data
               sequences        seq_len 60, segment_policy variable, min_seq_len 10 in every run config
               folds            all K-fold runs share folds_dir / n_folds / fold_seed (year runs share the year folds);
                                the fold files exist; year folds unchanged since year_folds wrote them
               cv_grid          fine HIQL K-fold sweep starts at min_latent 1
               synced           n_states filled in every level / generated config (setup stage ran)
               nulls            null_meta.json of every null level, mode as configured
               channels         env_switch / geometry channels present in the states_input channel list
               fine_fits        fine-level baselines present for every fold (ll_hiavi.csv K rows; scales N1 rows)
               freshness        analysis outputs newer than the fit files they summarise (else "stale analysis")
Writes <drivers_dir>/audit/audit_progress.csv, audit_groups.csv, audit_checks.csv, audit_report.md.
Exit code 1 if work remains (todo / failed / stale / blocked / running) or an integrity check fails (--strict:
also on warnings), so it can gate the next stage of a script.

    python -m naturenet.drivers.audit -y config/v2/drivers/whales_drivers.yaml [--plan <plan.yaml>] [--strict]

Pipeline manifest mode (what has run / what is left for any staged pipeline, e.g. steps 1-15; see manifest_audit):
    python -m naturenet.drivers.audit --manifest config/v2/audit/whales_pipeline.yaml [--steps 1-15] [--strict]
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

from naturenet.drivers.common import abs_cfg, atomic_csv, pkg_dir, read_yaml

SEQ = dict(seq_len=60, segment_policy="variable", min_seq_len=10)
OPEN = ("todo", "failed", "stale", "blocked", "running")


# ----------------------------------------------------------------------------- progress
def job_status(plan_path, levels=()):
    from sit_fuse.local_launcher import load_plan
    st, jobs = load_plan(plan_path)
    sd = st["state_dir"]
    stat = {}
    for j in jobs:                                    # toposorted: dependencies first
        done = os.path.join(sd, "markers", j.id + ".done")
        fail = os.path.join(sd, "markers", j.id + ".failed")
        s = "todo"
        if os.path.exists(done):
            try:
                with open(done) as f:
                    s = "done" if json.load(f).get("hash") == j.hash else "stale"
            except Exception:
                s = "stale"
        elif j.creates and all(os.path.exists(c) for c in j.creates):
            s = "done"
        elif os.path.exists(fail):
            s = "failed"
        elif os.path.exists(os.path.join(sd, "claims", j.id)):
            s = "running"
        if s == "todo" and any(stat.get(d.id) in ("failed", "blocked") for d in j.deps):
            s = "blocked"
        stat[j.id] = s
    rows = []
    for j in jobs:
        test, level = classify(j.name, levels)
        rows.append(dict(job=j.id, name=j.name, test=test, level=level, fold=j.params.get("fold", ""),
                         status=stat[j.id], log=os.path.join(sd, "logs", j.id + ".log")))
    return pd.DataFrame(rows), sd


def classify(name, levels=()):
    """job group name -> (test, level). Tags are plan_jobs._tag: <config dir>-<config stem>, "_" -> "-"."""
    head, _, tag = name.partition(".")
    tests = {"hiavi": "fits", "scales": "fits", "ma_obs": "coupling_fit", "ma_null": "coupling_fit",
             "ma_merge": "coupling_fit", "year_folds": "year_transfer", "availability": "geometry",
             "sample_avl": "geometry", "geometry": "geometry", "geometry_avl": "geometry",
             "projection": "projection", "grid": "env_scale_grid", "grid_year": "year_transfer",
             "ppc": "ppc", "ppc_collect": "ppc", "consistency": "consistency", "consistency_collect": "consistency",
             "coupling_env": "coupling", "switch": "env_switch", "correlation": "env_switch", "synthesis": "synthesis"}
    test = tests.get(head, head)
    if head in ("ppc", "switch"):
        return test, tag
    t = tag
    for pre, tst in (("year-", "year_transfer"), ("envswitch-", "env_switch")):
        if t.startswith(pre):
            test, t = tst, t[len(pre):]
    if not t or test not in ("fits", "year_transfer", "env_switch"):
        return test, ""
    for lv in sorted(levels, key=len, reverse=True):
        if t.startswith(lv.replace("_", "-") + "-"):
            return test, lv
    return test, "fine" if t.startswith("swirl-") else t.split("-")[0]


def roll_up(P):
    def state(s):
        u = set(s)
        if u == {"done"}:
            return "done"
        if u == {"todo"}:
            return "todo"
        for k in ("failed", "blocked", "stale", "running"):
            if k in u and u <= {k, "done"}:
                return k
        return "partial"
    g = P.groupby(["test", "level", "name"])["status"].agg(
        n="size", done=lambda s: int((s == "done").sum()), failed=lambda s: int((s == "failed").sum()),
        state=state).reset_index()
    return g


# ----------------------------------------------------------------------------- integrity checks
class Checks:
    def __init__(self):
        self.rows = []

    def add(self, check, ok, item, detail="", level="error"):
        self.rows.append(dict(check=check, status="ok" if ok else level, item=str(item), detail=detail))


def run_configs(st):
    """[(level, run, path, conf)] of every run config the study uses (level + generated)."""
    out = []
    runs = list((st.d.get("env_levels") or {}).get("runs") or [])
    for lv in st.all_levels():
        for r in runs:
            p = st.run_yaml(lv, r)
            out.append((lv, r, p, read_yaml(p) if os.path.exists(p) else None))
    for p in sorted(glob.glob(st.gpath("year", "*.yaml")) + glob.glob(st.gpath("envswitch", "*.yaml"))):
        out.append(("generated", os.path.basename(p), p, read_yaml(p)))
    return out


def integrity(st):
    C = Checks()
    cfgs = run_configs(st)
    v2 = os.path.join(pkg_dir(), "config", "v2")
    for lv, r, p, c in cfgs:
        if c is None:
            C.add("synced", False, p, "config missing (run STAGE=setup / plan)")
            continue
        C.add("v2_only", os.path.abspath(p).startswith(v2), p, "config outside config/v2")
        txt = json.dumps(c)
        C.add("v2_only", "/mnt/data" not in txt, p, "a path uses /mnt/data (base is /data/nlahaye)")
        C.add("v2_only", "_v2" in str(c.get("df_dir", "")), p, f"df_dir {c.get('df_dir')} is not a v2 root")
        bad = {k: c.get(k) for k, v in SEQ.items() if c.get(k) != v}
        C.add("sequences", not bad, p, f"differs from {SEQ}: {bad}" if bad else "")
        C.add("synced", c.get("n_states") not in (None, "null"), p, "n_states is null (sync not run)")
    # folds
    groups = {}
    for lv, r, p, c in cfgs:
        if not c:
            continue
        key = (c.get("folds_dir"), int(c.get("n_folds", 5)), int(c.get("fold_seed", 0)))
        groups.setdefault("year" if str(c.get("run_uid", "")).endswith("_YEAR") else "main", set()).add(key)
    for g, keys in groups.items():
        C.add("folds", len(keys) == 1, g, f"{len(keys)} different fold settings: {sorted(map(str, keys))}")
        for fd, n, s in keys:
            f = os.path.join(fd or "", f"folds_{n}_seed{s}.json")
            C.add("folds", os.path.exists(f), f, "fold file missing (created by the first fit / year_folds job)",
                  level="warning")
    yt = st.d.get("year_transfer") or {}
    if yt:
        meta = os.path.join(yt["folds_dir"], "year_folds_meta.json")
        if os.path.exists(meta):
            with open(meta) as f:
                m = json.load(f)
            fp = m.get("folds_file")
            if fp and os.path.exists(fp):
                with open(fp) as f:
                    cur = json.load(f)
                C.add("folds", cur == m.get("folds", cur), fp, "year folds changed after year_folds wrote them "
                      "(get_folds rebuilt them: animal set mismatch?)")
        else:
            C.add("folds", False, meta, "year folds not written yet", level="warning")
    # CV grid (user setting: min_latent 1 for every CV test at the fine level)
    fh = st.run_yaml("fine", "hiavi_kfold_kin")
    if os.path.exists(fh):
        c = read_yaml(fh)
        C.add("cv_grid", int(c.get("min_latent", 1)) == 1, fh, f"min_latent {c.get('min_latent')} (expected 1)")
    # nulls
    for nl in st.d.get("nulls") or []:
        meta = os.path.join(st.level_dir(nl), "null_meta.json")
        mode = next((x.get("mode") for x in st.lspec.get("levels", []) if x.get("name") == nl), None)
        if os.path.exists(meta):
            with open(meta) as f:
                m = json.load(f)
            C.add("nulls", m.get("mode") == mode, meta, f"mode {m.get('mode')} != {mode}")
        else:
            C.add("nulls", False, meta, "null level not built (STAGE=setup)")
    # channels
    g = st.d.get("geometry") or {}
    if g:
        sc = read_yaml(abs_cfg(g["sample_inputs_yaml"]))
        cj = os.path.join(sc["df_dir"], sc["df_run_uid"] + "_states_input_channels.json")
        if os.path.exists(cj):
            with open(cj) as f:
                have = set(json.load(f))
            for blk in ("geometry", "env_switch"):
                miss = [x for x in (st.d.get(blk) or {}).get("channels") or [] if x not in have]
                C.add("channels", not miss, f"{blk}.channels", f"not in {cj}: {miss}")
        else:
            C.add("channels", False, cj, "states_input channels missing (run sample_env_inputs)")
    # fine fits
    for r in ("hiavi_kfold_kin", "hiql_scales_kin_k5"):
        p = st.run_yaml("fine", r) if r in st.d["fine"] else None
        if not p or not os.path.exists(p):
            continue
        c = read_yaml(p)
        n = int(c.get("n_folds", 5))
        if "hiavi" in r:
            f = os.path.join(c["out_dir"], "ll_hiavi.csv")
            ok = os.path.exists(f) and set(pd.read_csv(f).query(f"K == {st.K}")["fold"]) >= set(range(n))
        else:
            f = os.path.join(c["out_dir"], f"{c['run_uid']}_scales_cv.csv")
            ok = False
            if os.path.exists(f):
                d = pd.read_csv(f)
                ok = set(d[d["variant"].astype(str).str.fullmatch(r"(HIQL_)?N1")]["fold"]) >= set(range(n))
        C.add("fine_fits", ok, f, f"fine {r} missing folds (run the kinematic baseline / scales plans)")
    # freshness: analysis outputs vs the newest fit summary they read
    fits = []
    for lv, r, p, c in cfgs:
        if c:
            fits += glob.glob(os.path.join(c.get("out_dir", ""), "ll_hiavi.csv"))
            fits += glob.glob(os.path.join(c.get("out_dir", ""), f"{c.get('run_uid')}_scales_cv.csv"))
    newest = max((os.path.getmtime(f) for f in fits), default=0)
    for name, rel in (("env_scale_grid", "grid_folds.csv"), ("synthesis", "metrics_long.csv")):
        f = os.path.join(st.odir(name), rel)
        if os.path.exists(f):
            C.add("freshness", os.path.getmtime(f) >= newest, f, "older than a fit summary: rerun the analysis",
                  level="warning")
    return pd.DataFrame(C.rows)


def report(st, P, G, K, path):
    L = [f"# Environmental-driver audit: {st.sp}", ""]
    if P is not None and len(P):
        c = P["status"].value_counts().to_dict()
        L += ["## Progress", "", ", ".join(f"{k} {v}" for k, v in sorted(c.items())), "",
              "| test | level | group | jobs | done | failed | state |", "|---|---|---|---|---|---|---|"]
        for r in G.itertuples():
            L.append(f"| {r.test} | {r.level} | {r.name} | {r.n} | {r.done} | {r.failed} | {r.state} |")
        f = P[P["status"] == "failed"]
        if len(f):
            L += ["", "Failed job logs:", ""] + [f"- {r.job}: {r.log}" for r in f.head(30).itertuples()]
    L += ["", "## Integrity", ""]
    bad = K[K["status"] != "ok"] if len(K) else K
    L.append(f"{int((K['status'] == 'ok').sum()) if len(K) else 0} ok, "
             f"{int((K['status'] == 'error').sum()) if len(K) else 0} errors, "
             f"{int((K['status'] == 'warning').sum()) if len(K) else 0} warnings")
    if len(bad):
        L += ["", "| check | status | item | detail |", "|---|---|---|---|"]
        L += [f"| {r.check} | {r.status} | {r.item} | {r.detail} |" for r in bad.itertuples()]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--manifest" in argv or "-m" in argv:           # pipeline manifest mode (steps 1-15, any use case)
        from naturenet.drivers import manifest_audit
        return manifest_audit.main(argv)
    from naturenet.drivers.plan import Study
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--plan", default=None, help="launcher plan (default <state_dir>/../plan_drivers_<species>.yaml)")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--out", default=None, help="report dir (default <drivers_dir>/audit/)")
    a = ap.parse_args(argv)
    st = Study(a.yaml)
    out = a.out or os.path.join(st.out, "audit")
    os.makedirs(out, exist_ok=True)
    plan = a.plan or os.path.join(os.path.dirname((st.d.get("launcher") or {}).get("state_dir", st.out).rstrip("/")),
                                  f"plan_drivers_{st.sp}.yaml")
    P = G = None
    if os.path.exists(plan):
        P, _ = job_status(plan, st.all_levels())
        G = roll_up(P)
        atomic_csv(P, os.path.join(out, "audit_progress.csv"))
        atomic_csv(G, os.path.join(out, "audit_groups.csv"))
    else:
        print(f"audit: no plan at {plan} (progress skipped)")
    K = integrity(st)
    atomic_csv(K, os.path.join(out, "audit_checks.csv"))
    report(st, P, G, K, os.path.join(out, "audit_report.md"))
    n_open = int(P["status"].isin(OPEN).sum()) if P is not None else 0
    n_err = int((K["status"] == "error").sum()) if len(K) else 0
    n_warn = int((K["status"] == "warning").sum()) if len(K) else 0
    print(f"audit {st.sp}: {0 if P is None else len(P)} jobs, {n_open} open; checks {n_err} errors, {n_warn} warnings "
          f"-> {out}/audit_report.md")
    if G is not None:
        print(G[G["state"] != "done"].to_string(index=False) if (G["state"] != "done").any() else "all job groups done")
    if len(K) and (K["status"] != "ok").any():
        print(K[K["status"] != "ok"].to_string(index=False))
    sys.exit(1 if (n_open or n_err or (a.strict and n_warn)) else 0)


if __name__ == "__main__":
    main()
