"""
Manifest auditor: what has run and what still needs to run, for any staged pipeline (read-only).

A manifest YAML lists the steps of a pipeline and, for each step, the files it writes. The auditor resolves the
paths (variables + values read from the run configs), checks that every output exists (and, optionally, that
CSV / JSON contents are complete), folds in launcher-plan progress for steps that run under the SIT-FUSE local
launcher, flags outputs that are older than the step outputs they depend on, and runs config-level checks.
Nothing in the engine is NatureNet-specific: the same code audits any SIT-FUSE use case given its own manifest.

    python -m naturenet.drivers.audit --manifest config/v2/audit/whales_pipeline.yaml [--steps 1-15] [--strict]
    python -m naturenet.drivers.manifest_audit -m <manifest.yaml> [--steps 5b,6b,7-13] [--set NAME=value ...]

Manifest
    name: whales v2 pipeline
    out: /data/me/audit/whales                 # report dir (--out overrides)
    vars:  {R: /data/me/run, UID: run1}        # {R} in any string; --set R=... overrides
    configs: {final: config/v2/swirl/x.yaml}   # {cfg.final.out_dir}, {cfg.final.multiagent.ma_subdir}
                                               # relative paths are taken from the package dir (or --root)
    checks:  [<check>, ...]                    # config checks run once (see below)
    steps:
      - id: 5b                                 # ids sort by their leading number (--steps 1-15 selects 1 .. 15x)
        title: Kinematic baselines
        after: [3e]                            # upstream steps: waiting-on and staleness
        optional: false                        # optional steps are reported but never fail the audit
        manual: false                          # manual steps: outputs are listed, state is "manual"
        note: free text shown in the report
        plan: {path: "{LS}/kin_plan.yaml", jobs: ["hiavi.*"], optional: true}   # launcher progress
        checks: [<check>, ...]                 # step-level config checks (errors make the step partial)
        outputs:
          - path: "{R}/out/*.npz"              # glob (or a list of globs, matched together); ** recurses
            min: 25                            # matched-file count; an expression is allowed ("5 * 5")
            optional: false                    # reported, not required
            level: error                       # content-check level (error | warning)
            ref: true                          # false: do not use this output as an upstream timestamp
            csv:  {where: "run_uid == 'A'", cover: {K: "range(1, 6)", fold: "range(5)"}, per: variant,
                   expr: "df['x'].notna().all()"}      # every (K, fold) present [within each variant]
            json: {expr: "'K' in d"}           # d = the parsed JSON
Checks (config files must exist; a missing file is an error)
    - {name: sequences, files: [globs], expect: {seq_len: 60}, when: df_dir, level: error}
    - {name: synced, files: [...], not_null: [n_states]}
    - {name: v2_paths, files: [...], absent: ["/mnt/data"]}

Step states: done | partial | missing | failed | running | manual, plus "stale" when a done step has an output
older than the newest output of a step in `after`. Writes <out>/manifest_steps.csv, manifest_outputs.csv,
manifest_checks.csv and manifest_report.md. Exit 1 if a required step in the selection is not done or a check
errors (--strict: also stale steps and warnings), so the audit can gate the next stage of a script.
"""
import argparse
import fnmatch
import glob
import json
import os
import re
import sys

import pandas as pd
import yaml

_REF = re.compile(r"\{([A-Za-z_][\w.\-]*)\}")
_SAFE = {"range": range, "len": len, "list": list, "set": set, "min": min, "max": max, "int": int, "float": float,
         "str": str, "sorted": sorted, "any": any, "all": all, "sum": sum, "abs": abs, "isinstance": isinstance,
         "dict": dict, "bool": bool}


# ----------------------------------------------------------------------------- resolution
class Resolver:
    def __init__(self, man, root, overrides=None):
        self.vars = {str(k): str(v) for k, v in (man.get("vars") or {}).items()}
        self.vars.update(overrides or {})
        self.root = root
        self.cfg_paths = {k: self.path(v) for k, v in (man.get("configs") or {}).items()}
        self._cfg = {}

    def path(self, p):
        p = self.sub(p)
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def cfg(self, alias):
        if alias not in self._cfg:
            p = self.cfg_paths.get(alias)
            if p is None:
                raise KeyError(f"unknown config alias {alias!r} (add it under configs:)")
            if not os.path.exists(p):
                raise FileNotFoundError(f"config {alias}: {p} not found")
            with open(p) as f:
                self._cfg[alias] = yaml.safe_load(f) or {}
        return self._cfg[alias]

    def lookup(self, name):
        if name.startswith("cfg."):
            alias, *keys = name[4:].split(".")
            v = self.cfg(alias)
            for k in keys:
                if not isinstance(v, dict) or k not in v:
                    raise KeyError(f"{alias}: key {'.'.join(keys)} not in {self.cfg_paths[alias]}")
                v = v[k]
            return "" if v is None else str(v)
        if name in self.vars:
            return self.sub(self.vars[name])
        raise KeyError(f"unknown variable {{{name}}} (add it under vars: or pass --set {name}=...)")

    def sub(self, s):
        if not isinstance(s, str):
            return s
        for _ in range(10):                                       # nested variables
            t = _REF.sub(lambda m: self.lookup(m.group(1)), s)
            if t == s:
                break
            s = t
        return s

    def value(self, v):
        """number / list / expression string -> python value."""
        if isinstance(v, str):
            s = self.sub(v)
            try:
                return eval(s, {"__builtins__": {}}, dict(_SAFE))
            except Exception:
                return s
        if isinstance(v, list):
            return [self.value(x) for x in v]
        return v


def _globs(pats):
    files = set()
    for p in pats:
        p = os.path.normpath(p) if "**" not in p else p
        files.update(glob.glob(p, recursive="**" in p))
    return sorted(files)


def _mtime(files):
    t = [os.path.getmtime(f) for f in files if os.path.exists(f)]
    return (min(t), max(t)) if t else (None, None)


# ----------------------------------------------------------------------------- checks
def run_checks(checks, R, scope):
    rows = []
    for c in checks or []:
        name, level = c.get("name", "check"), c.get("level", "error")
        try:
            files = _globs([R.path(f) for f in c.get("files") or []])
        except Exception as e:                                     # noqa: BLE001
            rows.append(dict(scope=scope, check=name, status="error", item="", detail=str(e)))
            continue
        if not files:
            rows.append(dict(scope=scope, check=name, status="error", item=str(c.get("files")),
                             detail="no config files matched"))
            continue
        for f in files:
            try:
                with open(f) as fh:
                    txt = fh.read()
                conf = yaml.safe_load(txt) or {}
            except Exception as e:                                 # noqa: BLE001
                rows.append(dict(scope=scope, check=name, status="error", item=f, detail=f"unreadable: {e}"))
                continue
            if c.get("when") and c["when"] not in conf:
                continue
            bad = []
            for k, v in (c.get("expect") or {}).items():
                if conf.get(k) != R.value(v):
                    bad.append(f"{k}={conf.get(k)!r} (expected {R.value(v)!r})")
            for k in c.get("not_null") or []:
                if k in conf and conf[k] in (None, "null", ""):
                    bad.append(f"{k} is null")
            for s in c.get("absent") or []:
                if s in txt:
                    bad.append(f"contains {s!r}")
            rows.append(dict(scope=scope, check=name, status=level if bad else "ok", item=f, detail="; ".join(bad)))
    return rows


# ----------------------------------------------------------------------------- outputs
def check_output(o, R):
    pats = o["path"] if isinstance(o["path"], list) else [o["path"]]
    row = dict(path=" | ".join(map(str, pats)), optional=bool(o.get("optional")), n=0, min=1, status="missing",
               detail="", mtime_min=None, mtime_max=None, ref=o.get("ref", True))
    try:
        pats = [R.path(p) for p in pats]
        pats = [p if "**" in p else os.path.normpath(p) for p in pats]
        row["path"] = " | ".join(pats)
        n_min = int(R.value(o.get("min", 1)))
        row["min"] = n_min
    except Exception as e:                                         # noqa: BLE001
        row.update(status="error", detail=f"cannot resolve: {e}")
        return row, []
    files = _globs(pats)
    row["n"] = len(files)
    row["mtime_min"], row["mtime_max"] = _mtime(files)
    if not files:
        return row, files
    if len(files) < n_min:
        row.update(status="short", detail=f"{len(files)} of {n_min} files")
        return row, files
    row["status"] = "ok"
    level = o.get("level", "error")
    msg = []
    try:
        if o.get("csv"):
            msg += _csv_check(o["csv"], files, R)
        if o.get("json"):
            msg += _json_check(o["json"], files)
    except Exception as e:                                         # noqa: BLE001
        msg.append(f"content check failed: {e}")
    if msg:
        row.update(status="incomplete" if level == "error" else "warning", detail="; ".join(msg))
    return row, files


def _csv_check(spec, files, R):
    msg = []
    for f in files:
        df = pd.read_csv(f)
        if spec.get("where"):
            df = df.query(R.sub(spec["where"]))
        tag = os.path.basename(f) if len(files) > 1 else ""
        cover = {k: list(R.value(v)) for k, v in (spec.get("cover") or {}).items()}
        if cover:
            miss = [k for k in cover if k not in df.columns]
            if miss:
                msg.append(f"{tag} no column(s) {miss}".strip())
                continue
            groups = df.groupby(spec["per"]) if spec.get("per") and spec["per"] in df else [("", df)]
            if spec.get("per") and len(df) == 0:
                msg.append(f"{tag} no rows".strip())
            for g, d in groups:
                have = set(map(tuple, d[list(cover)].astype(str).drop_duplicates().to_numpy()))
                want = {()}
                for k, vals in cover.items():
                    want = {w + (str(v),) for w in want for v in vals}
                lack = sorted(want - have)
                if lack:
                    lab = f"{spec['per']}={g} " if spec.get("per") else ""
                    msg.append(f"{tag} {lab}missing {len(lack)}/{len(want)} {tuple(cover)}: "
                               f"{', '.join(map(str, lack[:6]))}{' ...' if len(lack) > 6 else ''}".strip())
        if spec.get("expr"):
            ok = eval(R.sub(spec["expr"]), {"__builtins__": {}}, dict(_SAFE, df=df, pd=pd))
            if not bool(ok):
                msg.append(f"{tag} fails {spec['expr']}".strip())
    return msg


def _json_check(spec, files):
    msg = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if not bool(eval(spec["expr"], {"__builtins__": {}}, dict(_SAFE, d=d))):
            msg.append(f"{os.path.basename(f)} fails {spec['expr']}")
    return msg


# ----------------------------------------------------------------------------- launcher plans
def plan_progress(spec, R):
    """{todo, done, failed, running, stale, blocked} counts of the plan jobs matching spec['jobs']."""
    path = R.path(spec["path"])
    if not os.path.exists(path):
        return None, path
    from naturenet.drivers.audit import job_status
    P, _ = job_status(path)
    pats = spec.get("jobs") or ["*"]
    P = P[[any(fnmatch.fnmatch(n, p) for p in pats) for n in P["name"]]]
    return P, path


# ----------------------------------------------------------------------------- steps
def _num(sid):
    m = re.match(r"\d+", str(sid))
    return int(m.group()) if m else 10 ** 6


def select(steps, spec):
    if not spec:
        return steps
    keep = set()
    for part in str(spec).split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        for s in steps:
            sid = str(s["id"])
            if (m and int(m.group(1)) <= _num(sid) <= int(m.group(2))) or sid == part or \
                    (part.isdigit() and _num(sid) == int(part)):
                keep.add(sid)
    return [s for s in steps if str(s["id"]) in keep]


def audit_step(s, R):
    outs, files_by_ref = [], []
    for o in s.get("outputs") or []:
        row, files = check_output(o, R)
        row["step"] = str(s["id"])
        outs.append(row)
        if row["ref"] and not row["optional"]:
            files_by_ref += files
    checks = run_checks(s.get("checks"), R, str(s["id"]))
    plan = s.get("plan")
    pc, ppath, plan_note = {}, "", ""
    if plan:
        try:
            P, ppath = plan_progress(plan, R)
            if P is None:
                plan_note = "plan not built" + (" (optional)" if plan.get("optional", True) else "")
            else:
                pc = P["status"].value_counts().to_dict()
                f = P[P["status"] == "failed"]
                if len(f):
                    plan_note = "failed logs: " + ", ".join(f["log"].head(5))
        except Exception as e:                                     # noqa: BLE001
            plan_note = f"plan unreadable: {e}"
    req = [o for o in outs if not o["optional"]]
    st_out = {o["status"] for o in req}
    open_jobs = sum(pc.get(k, 0) for k in ("todo", "failed", "stale", "blocked", "running"))
    chk_err = any(c["status"] == "error" for c in checks)
    if s.get("manual"):
        state = "manual"
    elif not req and not checks and not plan:            # only optional outputs: report whether any exist
        state = "done" if any(o["status"] == "ok" for o in outs) else "missing"
    elif pc.get("failed"):
        state = "failed"
    elif pc.get("running"):
        state = "running"
    elif (not req or st_out == {"ok"}) and not open_jobs and not chk_err and \
            not (plan and not pc and not plan.get("optional", True)):
        state = "done"
    elif req and st_out == {"missing"} and not pc.get("done") and not chk_err:
        state = "missing"
    else:
        state = "partial"
    row = dict(step=str(s["id"]), title=s.get("title", ""), state=state, optional=bool(s.get("optional")),
               outputs_ok=sum(o["status"] == "ok" for o in req), outputs=len(req),
               jobs_done=pc.get("done", 0), jobs=sum(pc.values()) if pc else 0, plan=ppath, plan_note=plan_note,
               after=",".join(map(str, s.get("after") or [])), note=s.get("note", ""))
    t_min, t_max = _mtime(files_by_ref)
    return row, outs, checks, (t_min, t_max)


def run(man, R, steps_sel=None):
    steps = select(man.get("steps") or [], steps_sel)
    gl = run_checks(man.get("checks"), R, "global")
    rows, outs, checks, times = [], [], list(gl), {}
    all_steps = {str(s["id"]): s for s in man.get("steps") or []}
    cache = {}
    for s in man.get("steps") or []:                     # upstream steps outside the selection still give times
        sid = str(s["id"])
        if s in steps or any(sid in map(str, t.get("after") or []) for t in steps):
            cache[sid] = audit_step(s, R)
    for s in steps:
        r, o, c, t = cache[str(s["id"])]
        rows.append(r)
        outs += o
        checks += c
        times[r["step"]] = t
    S = pd.DataFrame(rows)
    stale, waiting = [], []
    for r in rows:
        up = [str(a) for a in (all_steps[r["step"]].get("after") or [])]
        newest = [(a, cache[a][3][1]) for a in up if a in cache and cache[a][3][1]]
        mine = times[r["step"]][0]
        late = [a for a, t in newest if mine and t > mine + 1.0]
        stale.append(",".join(late))
        waiting.append(",".join(a for a in up if a in cache and cache[a][0]["state"] not in ("done", "manual")
                                and r["state"] != "done"))
    if len(S):
        S["older_than"] = stale
        S["waiting_on"] = waiting
        S.loc[(S["state"] == "done") & (S["older_than"] != ""), "state"] = "stale"
    O = pd.DataFrame(outs)
    if len(O):
        for k in ("mtime_min", "mtime_max"):
            O[k] = pd.to_datetime(O[k], unit="s").dt.strftime("%Y-%m-%d %H:%M")
    return S, O, pd.DataFrame(checks, columns=["scope", "check", "status", "item", "detail"])


def report(name, S, O, K, path):
    L = [f"# Pipeline audit: {name}", ""]
    if len(S):
        c = S["state"].value_counts().to_dict()
        L += [", ".join(f"{k} {v}" for k, v in sorted(c.items())), "",
              "| step | title | state | outputs | jobs | waiting on | older than | note |",
              "|---|---|---|---|---|---|---|---|"]
        for r in S.itertuples():
            jobs = f"{r.jobs_done}/{r.jobs}" if r.jobs else (r.plan_note or "")
            opt = " (optional)" if r.optional else ""
            L.append(f"| {r.step} | {r.title}{opt} | {r.state} | {r.outputs_ok}/{r.outputs} | {jobs} | "
                     f"{r.waiting_on} | {r.older_than} | {r.note} |")
        bad = O[O["status"] != "ok"] if len(O) else O
        if len(bad):
            L += ["", "## Outputs not complete", "", "| step | status | files | path | detail |",
                  "|---|---|---|---|---|"]
            L += [f"| {r.step} | {r.status}{' (optional)' if r.optional else ''} | {r.n}/{r.min} | `{r.path}` | "
                  f"{r.detail} |" for r in bad.itertuples()]
        f = S[S["plan_note"].str.startswith("failed")] if len(S) else S
        if len(f):
            L += ["", "## Failed launcher jobs", ""] + [f"- {r.step}: {r.plan_note}" for r in f.itertuples()]
    L += ["", "## Config checks", ""]
    bad = K[K["status"] != "ok"] if len(K) else K
    L.append(f"{int((K['status'] == 'ok').sum()) if len(K) else 0} ok, "
             f"{int((K['status'] == 'error').sum()) if len(K) else 0} errors, "
             f"{int((K['status'] == 'warning').sum()) if len(K) else 0} warnings")
    if len(bad):
        L += ["", "| scope | check | status | item | detail |", "|---|---|---|---|---|"]
        L += [f"| {r.scope} | {r.check} | {r.status} | `{r.item}` | {r.detail} |" for r in bad.itertuples()]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None, root=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--manifest", required=True)
    ap.add_argument("--steps", default=None, help="e.g. 1-15, or 5b,6b,7-13 (default: every step)")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="override manifest vars")
    ap.add_argument("--root", default=None, help="base dir of relative config paths (default: package dir)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true", help="also fail on stale steps and warnings")
    a = ap.parse_args(argv)
    with open(a.manifest) as f:
        man = yaml.safe_load(f)
    over = dict(kv.split("=", 1) for kv in a.set)
    if a.root is None and root is None:
        from naturenet.drivers.common import pkg_dir
        root = pkg_dir()
    R = Resolver(man, a.root or root, over)
    out = a.out or R.sub(man.get("out") or os.path.join(os.path.dirname(os.path.abspath(a.manifest)), "audit"))
    os.makedirs(out, exist_ok=True)
    S, O, K = run(man, R, a.steps)
    S.to_csv(os.path.join(out, "manifest_steps.csv"), index=False)
    O.to_csv(os.path.join(out, "manifest_outputs.csv"), index=False)
    K.to_csv(os.path.join(out, "manifest_checks.csv"), index=False)
    name = man.get("name", os.path.basename(a.manifest))
    report(name, S, O, K, os.path.join(out, "manifest_report.md"))
    req = S[~S["optional"] & (S["state"] != "manual")] if len(S) else S
    n_open = int((~req["state"].isin(["done", "stale"])).sum()) if len(req) else 0
    n_stale = int((req["state"] == "stale").sum()) if len(req) else 0
    n_err = int((K["status"] == "error").sum()) if len(K) else 0
    n_warn = int((K["status"] == "warning").sum()) + (int((O["status"] == "warning").sum()) if len(O) else 0)
    print(f"audit {name}: {len(S)} steps, {n_open} open, {n_stale} stale; checks {n_err} errors, {n_warn} warnings "
          f"-> {out}/manifest_report.md")
    if len(S):
        cols = ["step", "title", "state", "outputs_ok", "outputs", "jobs_done", "jobs", "waiting_on", "older_than"]
        print(S[cols].to_string(index=False))
    if len(K) and (K["status"] != "ok").any():
        bad = K[K["status"] != "ok"]
        print(bad.groupby(["scope", "check", "status"]).size().rename("n").reset_index().to_string(index=False))
        print(f"(details: {out}/manifest_report.md)")
    fail = n_open or n_err or (a.strict and (n_stale or n_warn))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
