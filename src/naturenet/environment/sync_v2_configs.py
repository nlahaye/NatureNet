"""
Fill values that are only known after the v2 environment is built into the v2 configs (in place, comments kept).

    # NatureNet configs: n_states / n_clusters from the class mapper, input channels from the env meta JSON
    python -m naturenet.environment.sync_v2_configs patch -y config/v2/environment/sharks_sync_v2.yaml [--dry_run]

    # SIT-FUSE model config: files_train / files_test / number_channels / fill_value / valid_min from the
    # cmems_env <tag>_sitfuse_data_block.yaml
    python -m naturenet.environment.sync_v2_configs sitfuse --model <SIT-FUSE>/config/model/copernicus_shark_v2.yaml \
        --block /data/nlahaye/NatureNet_Env/Copernicus/v2/sharks/sharks_sitfuse_data_block.yaml

patch rewrites these keys wherever they appear (any indentation), one line each:
  n_states            K + 1 (invalid_state: absorbing) or K, K = valid clusters in the env_data_format class mapper
  n_clusters          K
  input_channels      resolved channel list (sample config input_channels_request: names / shell patterns,
                      matched against env_meta_json["channels"] in request order, absent ones dropped with a note)
  input_keys          the same list, only where it is not [] (i.e. configs with plot_inputs)
  variables           the same list (reward_env_projection; must be columns of cluster_profiles_wide.csv, i.e.
                      SIT-FUSE xai.feature_names = the full channel list, so any subset is valid)
  n_input_features    len(list)
"""
import argparse
import fnmatch
import glob
import json
import os
import pickle
import re

import yaml

SYNC_TAG = "# synced by sync_v2_configs"


def n_valid_clusters(mapper_path):
    with open(mapper_path, "rb") as f:
        mapper = pickle.load(f)
    vals = sorted({int(v) for v in mapper.values() if int(v) >= 0})
    if vals != list(range(len(vals))):
        raise ValueError(f"{mapper_path}: mapper values are not 0..K-1: {vals[:10]}...")
    return len(vals)


def resolve_channels(request, channels):
    if not request:
        return list(channels)
    out, missing = [], []
    for pat in request:
        hit = [c for c in channels if fnmatch.fnmatchcase(c, pat) and c not in out]
        if not hit:
            missing.append(pat)
        out += hit
    if missing:
        print(f"NOTE: {len(missing)} requested channels not in the v2 scenes, dropped: {missing}")
    if not out:
        raise ValueError("no requested channel exists in the scenes")
    return out


def _flow(lst):
    return "[" + ", ".join(lst) + "]"


def patch_text(text, values, keep_empty=("input_keys",)):
    """Replace `key: <value>` lines (any indentation). values: {key: rendered value}. Returns (text, n changes)."""
    n = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)([A-Za-z_]+):(\s*)(.*)$", line)
        if not m or m.group(2) not in values:
            continue
        key, rest = m.group(2), m.group(4)
        cur = rest.split(" #")[0].strip()
        if key in keep_empty and cur in ("[]", "null", ""):
            continue
        new = f"{m.group(1)}{key}: {values[key]}   {SYNC_TAG}"
        if new != line:
            lines[i] = new
            n += 1
    return "\n".join(lines), n


def _abs(path, base):
    return path if os.path.isabs(path) else os.path.join(base, path)


def patch(conf, base_dir, dry_run=False):
    with open(conf["env_meta_json"]) as f:
        meta = json.load(f)
    K = n_valid_clusters(conf["class_mapper"])
    absorbing = conf.get("invalid_state", "absorbing") == "absorbing"
    n_states = K + 1 if absorbing else K
    sample_cfg = _abs(conf["sample_config"], base_dir)
    with open(sample_cfg) as f:
        req = (yaml.safe_load(f) or {}).get("input_channels_request")
    names = resolve_channels(req, meta["channels"])
    values = dict(n_states=str(n_states), n_clusters=str(K), input_channels=_flow(names),
                  input_keys=_flow(names), variables=_flow(names), n_input_features=str(len(names)))
    print(f"K = {K} clusters -> n_states = {n_states}; {len(names)} input channels of {len(meta['channels'])}")
    files = []
    for pat in conf["files"]:
        files += sorted(glob.glob(_abs(pat, base_dir)))
    for fn in dict.fromkeys(files):
        with open(fn) as f:
            text = f.read()
        new, n = patch_text(text, values)
        yaml.safe_load(new)                      # still valid YAML
        if n:
            print(f"{'would patch' if dry_run else 'patched'} {n:2d} line(s): {fn}")
            if not dry_run:
                with open(fn, "w") as f:
                    f.write(new)
    left = [fn for fn in files if "__run_sync_v2_configs__" in open(fn).read()
            or re.search(r"(?m)^\s*n_states:\s*null", open(fn).read())]
    if left and not dry_run:
        print("WARNING: placeholders left in", left)
    return dict(K=K, n_states=n_states, channels=names)


def sync_sitfuse(model_path, block_path, out=None):
    """Replace the data-block keys of a SIT-FUSE model YAML (the rest of the file, incl. comments, is kept)."""
    with open(block_path) as f:
        block = yaml.safe_load(f)["data"]
    with open(model_path) as f:
        text = f.read()
    for key in ("number_channels", "fill_value", "valid_min", "valid_max", "reader_type", "chan_dim"):
        if key in block:
            text, n = re.subn(rf"(?m)^(\s+){key}:.*$", lambda m: f"{m.group(1)}{key}: {block[key]}", text, count=1)
            if not n:
                raise ValueError(f"{model_path}: no data.{key} line")
    for key in ("files_train", "files_test"):
        files = block.get(key) or []
        body = "[\n" + ",\n".join(f'"{p}"' for p in files) + "\n]"
        text, n = re.subn(rf"(?ms)^(\s+){key}:\s*\[.*?\]", lambda m: f"{m.group(1)}{key}: {body}", text, count=1)
        if not n:
            raise ValueError(f"{model_path}: no data.{key}: [...] block")
    yaml.safe_load(text)
    with open(out or model_path, "w") as f:
        f.write(text)
    print(f"wrote {out or model_path}: {len(block.get('files_train') or [])} train / "
          f"{len(block.get('files_test') or [])} test scenes, {block.get('number_channels')} channels")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("patch")
    a.add_argument("-y", "--yaml", required=True)
    a.add_argument("--base_dir", default=None, help="root for relative paths (default: the naturenet package dir)")
    a.add_argument("--dry_run", action="store_true")
    b = sub.add_parser("sitfuse")
    b.add_argument("--model", required=True)
    b.add_argument("--block", required=True)
    b.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if args.cmd == "patch":
        with open(args.yaml) as f:
            conf = yaml.safe_load(f)
        base = args.base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        patch(conf, base, args.dry_run)
    else:
        sync_sitfuse(args.model, args.block, args.out)


if __name__ == "__main__":
    main()
