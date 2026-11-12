"""
Reuse a movement preprocessing run in a new output folder (e.g. v1 -> v2 environment).

    python -m naturenet.environment.link_movement_run -y config/v2/environment/sharks_link_v2.yaml

Symlinks every file in src_dir whose name starts with <df_run_uid> (tracks, segments, actions, grids, paths,
distances, params) into dst_dir, except names matching `exclude` (env-derived outputs that the new run rewrites).
Directories in copy_dirs (e.g. folds) are copied, not linked, so the new run cannot modify the old one. Existing
regular files in dst_dir are never overwritten; existing symlinks are refreshed.

Keys: src_dir, dst_dir, df_run_uid, exclude (glob patterns), copy_dirs, dry_run.
"""
import argparse
import fnmatch
import os
import shutil

import yaml


def link_run(conf):
    src, dst, uid = conf["src_dir"], conf["dst_dir"], conf["df_run_uid"]
    excl = list(conf.get("exclude") or [])
    dry = bool(conf.get("dry_run", False))
    if not os.path.isdir(src):
        raise FileNotFoundError(src)
    if not dry:
        os.makedirs(dst, exist_ok=True)
    n_link = n_skip = 0
    for name in sorted(os.listdir(src)):
        p = os.path.join(src, name)
        if not os.path.isfile(p) or not name.startswith(uid) or any(fnmatch.fnmatch(name, e) for e in excl):
            continue
        q = os.path.join(dst, name)
        if os.path.exists(q) and not os.path.islink(q):
            n_skip += 1
            continue
        if not dry:
            if os.path.islink(q):
                os.remove(q)
            os.symlink(os.path.abspath(p), q)
        n_link += 1
    for d in conf.get("copy_dirs") or []:
        s, t = os.path.join(src, d), os.path.join(dst, d)
        if os.path.isdir(s) and not os.path.exists(t):
            if not dry:
                shutil.copytree(s, t)
            print(("would copy " if dry else "copied ") + s + " -> " + t)
        elif os.path.exists(t):
            print("kept existing", t)
    if not os.path.exists(os.path.join(src, uid + "_dfs.pkl")):
        print(f"WARNING: {src}{uid}_dfs.pkl not found; check df_run_uid")
    print(f"{'would link' if dry else 'linked'} {n_link} files, skipped {n_skip} existing regular files -> {dst}")
    return n_link


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("-y", "--yaml", required=True)
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args(argv)
    with open(a.yaml) as f:
        conf = yaml.safe_load(f)
    if a.dry_run:
        conf["dry_run"] = True
    link_run(conf)


if __name__ == "__main__":
    main()
