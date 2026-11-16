"""
Compute-resource accounting for launcher plans (CPU-hours, GPU-hours, wall time per test / job family).

Reads the launcher state (markers/<id>.done|.failed JSON with wall_sec, status.csv) and the plan (job cpus and GPU
share), and writes <out>/resources_jobs.csv (one row per finished job) and resources_summary.csv (per job name:
n done / failed, wall hours, CPU-hours = wall x cpus, GPU-hours = wall x GPU share, p50 / max wall).

    python -m naturenet.drivers.resources --plan <state_dir>/../plan_drivers_whales.yaml --out <dir>
    python -m naturenet.drivers.resources --profile <out.json> -- python -m naturenet.drivers.ppc -y ...

--profile runs one command and records wall time, CPU time and peak RSS (resource.getrusage of the children), for
sizing the cpus / gpu share of a job before putting it in a plan.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

from naturenet.drivers.common import atomic_csv, atomic_json


def _load_plan(path):
    from sit_fuse.local_launcher import load_plan
    return load_plan(path)


def account(plan, out):
    st, jobs = _load_plan(plan)
    sd = st["state_dir"]
    rows = []
    for j in jobs:
        for kind in ("done", "failed"):
            p = os.path.join(sd, "markers", f"{j.id}.{kind}")
            if os.path.exists(p):
                with open(p) as f:
                    m = json.load(f)
                w = float(m.get("wall_sec", 0) or 0)
                rows.append(dict(job=j.id, name=j.name, status=kind, current=m.get("hash") == j.hash, wall_h=w / 3600,
                                 cpus=j.cpus, gpu=j.gpu, cpu_h=w / 3600 * j.cpus, gpu_h=w / 3600 * j.gpu,
                                 **{f"p_{k}": v for k, v in j.params.items()}))
    d = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    atomic_csv(d, os.path.join(out, "resources_jobs.csv"))
    if len(d):
        s = d.groupby("name").agg(n_done=("status", lambda x: int((x == "done").sum())),
                                  n_failed=("status", lambda x: int((x == "failed").sum())),
                                  wall_h=("wall_h", "sum"), cpu_h=("cpu_h", "sum"), gpu_h=("gpu_h", "sum"),
                                  wall_h_p50=("wall_h", "median"), wall_h_max=("wall_h", "max")).reset_index()
        tot = s[["wall_h", "cpu_h", "gpu_h"]].sum()
        s = pd.concat([s, pd.DataFrame([dict(name="TOTAL", n_done=int(s["n_done"].sum()),
                                             n_failed=int(s["n_failed"].sum()), **tot.to_dict())])], ignore_index=True)
        atomic_csv(s, os.path.join(out, "resources_summary.csv"))
        print(s.to_string(index=False))
    return d


def profile_cmd(cmd, out_json):
    import resource
    t0, c0 = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    rc = subprocess.call(cmd)
    c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_kb = c1.ru_maxrss   # KB on Linux (max over waited children)
    r = dict(cmd=" ".join(cmd), rc=rc, wall_sec=time.time() - t0,
             cpu_sec=(c1.ru_utime - c0.ru_utime) + (c1.ru_stime - c0.ru_stime), peak_rss_gb=peak_kb / 1024 ** 2)
    r["mean_cores"] = r["cpu_sec"] / max(r["wall_sec"], 1e-9)
    atomic_json(r, out_json)
    print(json.dumps(r, indent=1))
    return rc


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="append", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", default=None, help="JSON path; profile the command after --")
    a = ap.parse_args(argv)
    if a.profile:
        sys.exit(profile_cmd(cmd, a.profile))
    for p in a.plan:
        account(p, a.out or os.path.join(os.path.dirname(os.path.abspath(p)), "resources",
                                         os.path.splitext(os.path.basename(p))[0]))


if __name__ == "__main__":
    main()
