"""
Gather run outputs (figures, tables, animations, logs) into per-group zips plus a manifest, driven by a YAML spec.

Generic: nothing here is NatureNet-specific. A spec lists groups; each group has root dirs, include globs (relative to
the root, ``**`` allowed), optional exclude globs, and an optional extension allow-list / per-file size cap. Model
weights and large arrays are left out by the extension list, not by name, so new output types are picked up when
their extension is allowed.

    python -m naturenet.misc.gather_outputs -y spec.yaml [--dry_run] [--only g1,g2] [--skip g3] [--out DIR]
        [--vars R=/path S=/path] [--split_mb 1900] [--max_file_mb 200] [--no_zip]

Spec (all keys but groups optional):
    vars: {R: /data/run, S: /data/launcher}       # {R} etc. substituted in every string; --vars overrides
    out_dir: "{R}/gathered/"                       # zips + manifest go to <out_dir>/<name>_<stamp>/
    name: whales_available_outputs
    defaults:
      exts: [png, pdf, svg, gif, mp4, csv, json, md, txt, yaml, log]   # case-insensitive; [] = any extension
      max_file_mb: 200                             # larger files are listed in the manifest as skipped
      follow_symlinks: false                       # symlinked inputs (e.g. level runs) are not copied
      exclude: ["**/__pycache__/**"]
    split_mb: 1900                                 # roll a group's zip into parts (0 = one zip per group)
    groups:
      - name: finals
        roots: ["{R}/final"]
        include: ["**/plots/*", "**/reward_env_projection/**", "*/*.csv"]
        exclude: ["**/series_*/**"]
        exts: [...]                                # overrides defaults
        enabled: true

Outputs: <group>[_partN].zip (paths inside are <group>/<path relative to root's parent>), MANIFEST.csv (group, archive
path, source, bytes, mtime, status), SUMMARY.md (per-group counts, sizes, newest file, missing roots, skipped files).
Already-compressed formats (png, gif, mp4, zip, ...) are stored, the rest deflated.
"""
import argparse
import csv
import os
import re
import sys
import time
import zipfile

import yaml

STORED = {"png", "jpg", "jpeg", "gif", "mp4", "webm", "mov", "zip", "gz", "bz2", "xz", "npz", "pdf", "tif", "tiff"}
DEFAULTS = {"exts": ["png", "pdf", "svg", "jpg", "jpeg", "gif", "mp4", "webm", "html", "csv", "tsv", "json", "md",
                     "txt", "yaml", "yml", "log"],
            "max_file_mb": 200, "follow_symlinks": False, "exclude": ["**/__pycache__/**", "**/.ipynb_checkpoints/**"]}


def subst(x, v):
    """Recursively substitute {KEY} for vars (unknown keys are left as they are)."""
    if isinstance(x, str):
        return re.sub(r"\{(\w+)\}", lambda m: str(v[m.group(1)]) if m.group(1) in v else m.group(0), x)
    if isinstance(x, list):
        return [subst(i, v) for i in x]
    if isinstance(x, dict):
        return {k: subst(i, v) for k, i in x.items()}
    return x


def glob_re(pat):
    """Glob -> regex on '/'-separated relative paths: ** spans dirs, * and ? stay within one component."""
    out, i = "", 0
    while i < len(pat):
        if pat.startswith("**/", i):
            out += "(?:.*/)?"; i += 3
        elif pat.startswith("**", i):
            out += ".*"; i += 2
        elif pat[i] == "*":
            out += "[^/]*"; i += 1
        elif pat[i] == "?":
            out += "[^/]"; i += 1
        else:
            out += re.escape(pat[i]); i += 1
    return re.compile(out + r"\Z")


def match_any(rel, regs):
    return any(r.match(rel) for r in regs)


def walk(root, follow):
    for d, dirs, files in os.walk(root, followlinks=follow):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(d, f)
            yield p, os.path.relpath(p, root).replace(os.sep, "/")


def collect_group(g, d):
    """-> (selected [(src, arc, bytes, mtime)], skipped [(src, reason, bytes)], missing roots)."""
    exts = {e.lower().lstrip(".") for e in g.get("exts", d["exts"])}
    cap = float(g.get("max_file_mb", d["max_file_mb"])) * 2 ** 20
    follow = bool(g.get("follow_symlinks", d["follow_symlinks"]))
    inc = [glob_re(p) for p in g.get("include", ["**"])]
    exc = [glob_re(p) for p in list(d["exclude"]) + list(g.get("exclude", []))]
    sel, skip, missing, seen = [], [], [], set()
    for root in g["roots"]:
        root = os.path.normpath(root)
        if not os.path.isdir(root):
            if os.path.isfile(root):          # a single file root (e.g. a plan yaml)
                st = os.stat(root)
                arc = f"{g['name']}/{os.path.basename(root)}"
                if root not in seen:
                    seen.add(root); sel.append((root, arc, st.st_size, st.st_mtime))
            else:
                missing.append(root)
            continue
        base = os.path.basename(root)
        for p, rel in walk(root, follow):
            if not match_any(rel, inc) or match_any(rel, exc):
                continue
            ext = rel.rsplit(".", 1)[-1].lower() if "." in os.path.basename(rel) else ""
            if exts and ext not in exts:
                continue
            if os.path.islink(p) and not follow:
                skip.append((p, "symlink", 0)); continue
            try:
                st = os.stat(p)
            except OSError as e:
                skip.append((p, f"stat failed: {e}", 0)); continue
            if st.st_size > cap:
                skip.append((p, f"> max_file_mb ({st.st_size / 2 ** 20:.0f} MB)", st.st_size)); continue
            rp = os.path.realpath(p)
            if rp in seen:
                continue
            seen.add(rp)
            sel.append((p, f"{g['name']}/{base}/{rel}", st.st_size, st.st_mtime))
    return sel, skip, missing


def write_zips(name, files, out, split_bytes):
    """Write files into <out>/<name>.zip, rolling into _partN when split_bytes is reached. -> list of zip paths."""
    zips, zf, size, part = [], None, 0, 1
    multi = split_bytes > 0 and sum(f[2] for f in files) > split_bytes

    def open_part(n):
        p = os.path.join(out, f"{name}_part{n}.zip" if multi else f"{name}.zip")
        zips.append(p)
        return zipfile.ZipFile(p, "w", allowZip64=True)

    for src, arc, nb, _ in files:
        if zf is None:
            zf = open_part(part)
        elif split_bytes > 0 and size > 0 and size + nb > split_bytes:
            zf.close(); part += 1; size = 0; zf = open_part(part)
        ext = arc.rsplit(".", 1)[-1].lower()
        zf.write(src, arc, compress_type=zipfile.ZIP_STORED if ext in STORED else zipfile.ZIP_DEFLATED)
        size += nb
    if zf is not None:
        zf.close()
    return zips


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--vars", nargs="*", default=[], help="KEY=VALUE overrides of the spec vars")
    ap.add_argument("--out", help="override out_dir")
    ap.add_argument("--only", help="comma-separated groups")
    ap.add_argument("--skip", help="comma-separated groups to leave out")
    ap.add_argument("--split_mb", type=float, help="override split_mb (0 = one zip per group)")
    ap.add_argument("--max_file_mb", type=float, help="override defaults.max_file_mb")
    ap.add_argument("--stamp", help="output subdir suffix (default YYYYmmdd_HHMM)")
    ap.add_argument("--dry_run", action="store_true", help="list counts / sizes, write nothing")
    ap.add_argument("--no_zip", action="store_true", help="write MANIFEST.csv + SUMMARY.md only")
    a = ap.parse_args(argv)

    spec = yaml.safe_load(open(a.yaml)) or {}
    v = {k: str(x) for k, x in (spec.get("vars") or {}).items()}
    for kv in a.vars:
        k, _, x = kv.partition("=")
        v[k] = x
    v = {k: subst(x, v) for k, x in v.items()}            # vars may reference earlier vars
    spec = subst({k: x for k, x in spec.items() if k != "vars"}, v)
    d = dict(DEFAULTS); d.update(spec.get("defaults") or {})
    if a.max_file_mb is not None:
        d["max_file_mb"] = a.max_file_mb
    only = set(a.only.split(",")) if a.only else None
    skipg = set(a.skip.split(",")) if a.skip else set()
    groups = [g for g in spec.get("groups") or [] if g.get("enabled", True)
              and (only is None or g["name"] in only) and g["name"] not in skipg]
    if only:
        unknown = only - {g["name"] for g in spec.get("groups") or []}
        if unknown:
            sys.exit(f"unknown groups: {sorted(unknown)}")
    name = spec.get("name", os.path.splitext(os.path.basename(a.yaml))[0])
    stamp = a.stamp or time.strftime("%Y%m%d_%H%M")
    out = os.path.join(a.out or spec.get("out_dir", "."), f"{name}_{stamp}")
    split = float(a.split_mb if a.split_mb is not None else spec.get("split_mb", 0)) * 2 ** 20

    rows, summ, total = [], [], 0
    if not (a.dry_run or a.no_zip):
        os.makedirs(out, exist_ok=True)
    for g in groups:
        sel, skip, missing = collect_group(g, d)
        nb = sum(f[2] for f in sel)
        total += nb
        newest = time.strftime("%Y-%m-%d %H:%M", time.localtime(max(f[3] for f in sel))) if sel else "-"
        zips = []
        if sel and not (a.dry_run or a.no_zip):
            zips = write_zips(g["name"], sel, out, int(split))
        rows += [(g["name"], arc, src, nb_, time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)), "included")
                 for src, arc, nb_, mt in sel]
        rows += [(g["name"], "", src, nb_, "", f"skipped: {why}") for src, why, nb_ in skip]
        rows += [(g["name"], "", r, 0, "", "missing root") for r in missing]
        summ.append((g["name"], len(sel), nb, newest, len(skip), missing, [os.path.basename(z) for z in zips],
                     g.get("note", "")))
        print(f"{g['name']:<22} {len(sel):>6} files {human(nb):>10}  newest {newest}  skipped {len(skip)}"
              + (f"  MISSING {', '.join(missing)}" if missing else ""))
    print(f"{'total':<22} {sum(s[1] for s in summ):>6} files {human(total):>10}")
    if a.dry_run:
        return 0
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "archive_path", "source", "bytes", "mtime", "status"])
        w.writerows(rows)
    with open(os.path.join(out, "SUMMARY.md"), "w") as f:
        f.write(f"# {name} ({stamp})\n\nspec: {os.path.abspath(a.yaml)}\nvars: {v}\n\n")
        f.write("| group | files | size | newest | skipped | zips | missing roots | note |\n|---|---|---|---|---|---|---|---|\n")
        for gname, n, nb, newest, ns, miss, zl, note in summ:
            f.write(f"| {gname} | {n} | {human(nb)} | {newest} | {ns} | {', '.join(zl) or '-'} | "
                    f"{', '.join(miss) or '-'} | {note} |\n")
        f.write(f"\nTotal: {sum(s[1] for s in summ)} files, {human(total)}. Skipped files and missing roots are "
                "listed in MANIFEST.csv (status column).\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
