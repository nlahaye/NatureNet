"""
Which animation jobs need re-rendering? Compares every input of each job of an animate_tracks YAML with the files the
job wrote, so only the jobs whose inputs changed are rerun (generic: any NatureNet / SIT-FUSE animation config).

    python -m naturenet.models.irl.animation_staleness -y config/v2/plots/sharks_animations.yaml
    python -m naturenet.models.irl.animation_staleness -y <base>.yaml:animate <auto>.yaml:animate_auto --print only
    python -m naturenet.models.irl.animation_staleness -y <yaml> --snapshot <dir>     # before rerunning step 10
    python -m naturenet.models.irl.animation_staleness -y <yaml> --since <dir>        # after: content-aware

Inputs per job (after the top-level defaults and {root} are applied, as animate_tracks does):
  panel path       series dir (newest *_time_series_pd.csv), latents glob (newest match) or multi-agent npz
  labels_csv       per-mode class table (exploit_explore_init / kinematic labels / mode_alignment)
  state_reward     (K, S) reward npy (reward backgrounds)
Outputs: <out_dir>/<job>/*.gif|*.mp4|*.png (the oldest media file is the render time; stills when animate: False).

Status per job
  current          every input is older than the render
  stale            an input is newer than the render (reasons list which)
  missing          inputs exist but nothing was rendered (e.g. the job was skipped while an input was missing)
  no_inputs        a panel input does not exist yet (the job would be skipped again)
--snapshot DIR saves the {mode: class} map of every labels_csv; with --since DIR a labels_csv whose map is unchanged
does not make a job stale, even if the file was rewritten (step 10 reruns usually reproduce the same classes).
P(exploit) traces come from the panel series / npz (step 8 export or ma_final), so they follow the panel path.

--print only    space-separated launcher --only globs (<prefix>.job<name>_nf*) for the stale + missing jobs
--print names   the same as animate_tracks --only job names
--csv FILE      the full table
Exit code 0 always (a report, not a check).
"""
import argparse
import glob
import json
import os

import pandas as pd

from naturenet.models.irl.animate_tracks import job_confs, read_conf, read_labels_csv

MEDIA = (".gif", ".mp4")


def _newest(paths):
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return None, None
    p = max(paths, key=os.path.getmtime)
    return p, os.path.getmtime(p)


def panel_input(p):
    """(path used for the mtime, mtime) of a panel's main input; (None, None) when it does not exist."""
    path = p.get("path")
    if not path:
        return None, None
    if os.path.isdir(path):
        return _newest(glob.glob(os.path.join(path, "*_time_series_pd.csv")))
    if any(ch in path for ch in "*?["):
        return _newest(glob.glob(path))
    return _newest([path])


def job_inputs(c):
    """[(role, path, mtime or None)] for every input of a job."""
    panels = c.get("panels") or [c]
    out = []
    for j, p in enumerate(panels):
        tag = p.get("name", f"panel{j}")
        f, t = panel_input(p)
        out.append((f"{tag}:path", f or p.get("path"), t))
        lcsv = p.get("labels_csv", c.get("labels_csv") if len(panels) <= 1 else None)
        if lcsv:
            out.append((f"{tag}:labels_csv", lcsv, os.path.getmtime(lcsv) if os.path.isfile(lcsv) else None))
        if p.get("state_reward"):
            sr = p["state_reward"]
            out.append((f"{tag}:state_reward", sr, os.path.getmtime(sr) if os.path.isfile(sr) else None))
    return out


def job_render(c):
    """(oldest media mtime, n files) in <out_dir>/<job>/ (stills when there are no gif / mp4 files)."""
    od = os.path.join(c.get("out_dir", "."), c["name"])
    files = [f for f in glob.glob(os.path.join(od, "*")) if os.path.isfile(f)]
    media = [f for f in files if f.endswith(MEDIA)] or [f for f in files if f.endswith(".png")]
    if not media:
        return None, 0
    return min(os.path.getmtime(f) for f in media), len(media)


def label_maps(confs):
    out = {}
    for c in confs:
        for role, path, t in job_inputs(c):
            if role.endswith(":labels_csv") and t is not None and path not in out:
                try:
                    out[path] = {str(k): v for k, v in read_labels_csv(path).items()}
                except Exception as e:                      # unreadable table: compare by mtime only
                    print(f"WARNING: {path}: {e}")
    return out


def assess(confs, prefix, since=None):
    rows = []
    for c in confs:
        ins = job_inputs(c)
        rt, n = job_render(c)
        missing_in = [r for r, p, t in ins if t is None and r.endswith(":path")]
        reasons = []
        if missing_in:
            status = "no_inputs"
            reasons = [f"missing {r}" for r in missing_in]
        elif rt is None:
            status = "missing"
        else:
            for r, p, t in ins:
                if t is None or t <= rt:
                    continue
                if r.endswith(":labels_csv") and since is not None and p in since:
                    try:
                        now = {str(k): v for k, v in read_labels_csv(p).items()}
                    except Exception:
                        now = None
                    if now == since[p]:
                        continue                            # rewritten, same classes
                reasons.append(f"{r} newer ({os.path.basename(str(p))})")
            status = "stale" if reasons else "current"
        newest_in = max((t for _, _, t in ins if t is not None), default=None)
        fmt = lambda t: pd.Timestamp(t, unit="s").strftime("%Y-%m-%d %H:%M") if t else ""
        rows.append(dict(prefix=prefix, job=c["name"], status=status, rendered=fmt(rt), n_files=n,
                         newest_input=fmt(newest_in), uses_labels=any(r.endswith(":labels_csv") for r, _, _ in ins),
                         colour_by=c.get("colour_by", "mode"), reasons="; ".join(reasons)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", nargs="+", required=True,
                    help="animation YAMLs, optionally <yaml>:<launcher job prefix> (default prefix animate)")
    ap.add_argument("--snapshot", default=None, help="write the labels_csv class maps to DIR/labels_snapshot.json")
    ap.add_argument("--since", default=None, help="DIR with a labels_snapshot.json: content-aware labels_csv check")
    ap.add_argument("--print", dest="print_", choices=["table", "only", "names"], default="table")
    ap.add_argument("--include_missing", type=int, default=1, help="1: list 'missing' jobs with the stale ones")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    specs = []
    for s in a.yaml:
        path, _, prefix = s.rpartition(":") if not os.path.exists(s) and ":" in s else (s, "", "")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found; skipped")
            continue
        specs.append((path, prefix or "animate", job_confs(read_conf(path))))
    if a.snapshot:
        os.makedirs(a.snapshot, exist_ok=True)
        maps = {}
        for _, _, confs in specs:
            maps.update(label_maps(confs))
        with open(os.path.join(a.snapshot, "labels_snapshot.json"), "w") as f:
            json.dump(maps, f, indent=1)
        print(f"wrote {os.path.join(a.snapshot, 'labels_snapshot.json')} ({len(maps)} labels tables)")
        return 0
    since = None
    if a.since:
        with open(os.path.join(a.since, "labels_snapshot.json")) as f:
            since = json.load(f)
    rows = []
    for _, prefix, confs in specs:
        rows += assess(confs, prefix, since)
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    want = {"stale", "missing"} if a.include_missing else {"stale"}
    sel = df[df["status"].isin(want)] if len(df) else df
    if a.print_ == "only":
        print(" ".join(f"{r.prefix}.job{r.job}_nf*" for r in sel.itertuples()))
    elif a.print_ == "names":
        print(" ".join(sel["job"]))
    else:
        with pd.option_context("display.max_rows", None, "display.width", 220, "display.max_colwidth", 80):
            print(df.to_string(index=False) if len(df) else "no jobs")
        print("\n" + ", ".join(f"{k}: {v}" for k, v in df["status"].value_counts().items()) if len(df) else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
