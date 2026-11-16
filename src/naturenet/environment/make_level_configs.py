"""
Generate the per-level NatureNet configs for the multi-scale SIT-FUSE environment (parent / coarse levels) and
for external categorical environments (categorical level, e.g. MBON Seascapes), plus environment-ablation nulls
(level: null, mode: shift | shuffle | marginal | single; naturenet.drivers.env_null writes the states files).

    python -m naturenet.environment.make_level_configs -y config/v2/levels/sharks_levels_v2.yaml [--dry_run]

The fine (heir) level is the v2 baseline (config/v2/environment + config/v2/swirl). Every other level gets its own
data folder <levels_dir>/<name>/ (movement files symlinked from base_dir, env-derived files rebuilt) and a config
folder <out_config_dir>/<name>/ with:
  link.yaml          link_movement_run: dfs / actions / grids / inputs from base_dir (folds stay in base_dir/folds)
  env_format.yaml    env_data_format with cluster_level / level_k / hierarchy_json -> <level dir>/env_clusters/
  abstract_env.yaml  movement_preprocess_pipeline -> <level dir>/<df_run_uid>_*_states_abstract_env.pkl, trans_prob
  sync.yaml          sync_v2_configs patch: n_states = K_level + 1 in this level's configs only
  <run>.yaml         one per spec `runs` entry (HIQL / IAVI K-fold, SWIRL, final fit, projection): paths under
                     base_dir -> level dir (folds_dir kept, so every level uses the same folds and the paired
                     test-LL differences are valid), run_uid + "_<NAME>", n_states: null, spec `overrides`
Level entries: {name, level: parent} | {name, level: coarse, k} | {name, level: categorical,
scene_file_template (with {i} = scene t), cluster_scale: 1, zero_label_valid: False, exclude_classes: [],
interpret_subdir (SIT-FUSE cluster_interpret --out_subdir of the class profiles; default interpret_<name>)}.
plus <out_config_dir>/ll_comparison_levels.yaml (fine baseline + every level; missing runs are skipped).
Templates are edited line by line, so their comments are kept. Rerun after changing the spec or the templates
(generated files are overwritten), then run sync.yaml for each level after its env_format step.
"""
import argparse
import os
import re

import yaml

PALETTE = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd",
           "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]   # tab20
KEY_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*):(.*)$")


def _split_comment(rest):
    """value part and comment part of a YAML line remainder (comment = first ' #' outside quotes)."""
    q = None
    for i, ch in enumerate(rest):
        if ch in "\"'":
            q = None if q == ch else (ch if q is None else q)
        elif ch == "#" and q is None and (i == 0 or rest[i - 1] in " \t"):
            return rest[:i], rest[i:]
    return rest, ""


def replace_prefix(text, old, new, skip_keys=()):
    """Replace path prefix `old` by `new` in the values of non-comment lines (keys in skip_keys untouched)."""
    out = []
    for line in text.split("\n"):
        m = KEY_RE.match(line)
        if m and m.group(2) not in skip_keys:
            val, com = _split_comment(m.group(3))
            if old in val:
                line = f"{m.group(1)}{m.group(2)}:{val.replace(old, new)}{com}"
        elif not m and not line.lstrip().startswith("#") and old in line:      # list items
            val, com = _split_comment(line)
            line = val.replace(old, new) + com
        out.append(line)
    return "\n".join(out)


def set_key(text, key, value, note="level", indent=None, append=True):
    """Set `key: value` on every matching line (any indentation, or only `indent`); append at the end if absent."""
    lines, hit = text.split("\n"), False
    for i, line in enumerate(lines):
        m = KEY_RE.match(line)
        if m and m.group(2) == key and (indent is None or m.group(1) == indent):
            lines[i] = f"{m.group(1)}{key}: {value}" + (f"   # {note}" if note else "")
            hit = True
    if not hit and append:
        while lines and lines[-1] == "":
            lines.pop()
        lines += [f"{key}: {value}" + (f"   # {note}" if note else ""), ""]
    return "\n".join(lines)


def _y(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        return yaml.safe_dump(v, default_flow_style=True, width=1000).strip()
    return f'"{v}"'


def _abs(path, base):
    return path if os.path.isabs(path) else os.path.join(base, path)


def _rel(path, start):
    return os.path.relpath(path, start)


def level_tag(lv):
    return lv["name"].upper()


def interpret_subdir(lv):
    if lv["level"] == "null":                    # shift / shuffle / marginal keep the source label meanings
        return lv.get("interpret_subdir") or "interpret"
    if lv["level"] == "categorical":
        return lv.get("interpret_subdir") or f"interpret_{lv['name']}"
    return "interpret_parent" if lv["level"] == "parent" else f"interpret_k{int(lv['k'])}"


def _categorical_env_format(t, lv):
    for k in ("scene_file_template",):
        if k not in lv:
            raise ValueError(f"{lv['name']}: categorical level needs {k}")
    t = set_key(t, "scene_file_template", _y(lv["scene_file_template"]), "categorical class rasters, {i} = scene t")
    t = set_key(t, "cluster_scale", _y(lv.get("cluster_scale", 1)), "class id = raster value")
    t = set_key(t, "zero_label_valid", _y(bool(lv.get("zero_label_valid", False))), "0 / -1 = no class")
    t = set_key(t, "label_input", '"fine"', "rasters read as they are")
    t = set_key(t, "exclude_classes", _y(list(lv.get("exclude_classes") or [])), "set invalid (absorbing)")
    for k, v in (lv.get("env_format") or {}).items():
        t = set_key(t, k, _y(v), "level override")
    return t


def env_format_text(tmpl, spec, lv, ldir):
    t = replace_prefix(tmpl, spec["base_dir"], ldir)
    t = set_key(t, "cluster_level", _y(lv["level"]))
    if lv["level"] == "categorical":
        return _categorical_env_format(t, lv)
    t = set_key(t, "label_input", _y(lv.get("label_input", spec.get("label_input", "fine"))))
    t = set_key(t, "num_child", _y(int(spec.get("num_child", 5))), "SIT-FUSE cluster.heir.num_classes")
    t = set_key(t, "zero_label_valid", "True", "id 0 is a real cluster")
    t = set_key(t, "unmapped_parent", _y(spec.get("unmapped_parent", "error")))
    if lv["level"] == "coarse":
        t = set_key(t, "hierarchy_json", _y(spec["hierarchy_json"]), "sit_fuse cluster_hierarchy build")
        t = set_key(t, "level_k", _y(int(lv["k"])), "cut of the parent hierarchy")
    if lv.get("label_input", spec.get("label_input", "fine")) == "parent":
        t = t.replace(".zarr.clust.data_", ".zarr.no_heir.clust.data_")       # first-stage rasters
    return t


def run_text(tmpl, spec, lv, ldir, overrides):
    t = replace_prefix(tmpl, spec["base_dir"], ldir, skip_keys=("folds_dir",))
    t = set_key(t, "n_states", "null", "synced by naturenet.environment.sync_v2_configs (level K + 1 absorbing)",
                indent="", append=False)
    m = re.search(r'(?m)^run_uid:\s*"?([^"\s#]+)"?', t)
    if m:
        t = set_key(t, "run_uid", _y(f"{m.group(1)}_{level_tag(lv)}"), "level run", indent="")
    for k, v in (overrides or {}).items():
        t = set_key(t, k, _y(v), "level override", indent="")
    if "reward_env_projection:" in t:
        prof = os.path.join(spec["sitfuse_out_dir"], "xai", interpret_subdir(lv), "cluster_profiles_wide.csv")
        note = (f"cluster_interpret --label_pattern <class rasters> --label_scale 1 --out_subdir {interpret_subdir(lv)}"
                if lv["level"] == "categorical" else f"cluster_interpret --level {lv['level']}")
        t = set_key(t, "profiles_csv", _y(prof), note, indent="  ")
        t = set_key(t, "class_mapper", _y(mapper_path(spec, ldir)), "level class mapper", indent="  ")
        if lv["level"] == "categorical":
            t = set_key(t, "cluster_scale", _y(lv.get("cluster_scale", 1)), "categorical ids", indent="  ")
    return t


def null_text(spec, lv, ldir, cdir, base):
    """naturenet.drivers.env_null config of a null level (environment ablation, test 1)."""
    src = lv.get("source_dir") or spec["base_dir"]
    conf = dict(source_dir=src, dst_dir=ldir, df_run_uid=spec["df_run_uid"], mode=lv.get("mode", "shift"),
                seed=int(lv.get("seed", 0)), output_uid=spec.get("output_uid", "final_env_maps_"),
                source_mapper=lv.get("source_mapper") or os.path.join(src, "env_clusters", spec.get(
                    "output_uid", "final_env_maps_") + "_class_mapper.pkl"), link=True, write_trans_prob=True)
    return (f"# Environment level {lv['name']}: {conf['mode']} null of {src} (movement unchanged, labels only).\n"
            f"#   python -m naturenet.drivers.env_null -y {_rel(os.path.join(cdir, 'null.yaml'), base)}\n"
            + yaml.safe_dump(conf, sort_keys=False))


def mapper_path(spec, ldir):
    return os.path.join(ldir, "env_clusters", spec.get("output_uid", "final_env_maps_") + "_class_mapper.pkl")


def header(spec_path, tmpl_path, lv):
    what = {"parent": "parent (first-stage) clusters", "categorical": "external categorical classes",
            "null": f"environment null ({lv.get('mode', 'shift')})"}.get(
        lv["level"]) or f"coarse level K = {lv['k']}"
    return (f"# Environment level {lv['name']}: {what}. Generated by naturenet.environment.make_level_configs from\n"
            f"# {tmpl_path} ({os.path.basename(spec_path)}); edit the spec or template and regenerate.\n")


def generate(spec, spec_path, base, dry_run=False):
    out_cfg = _abs(spec["out_config_dir"], base)
    written = {}
    tm = spec["templates"]
    read = lambda p: open(_abs(p, base)).read()
    for lv in spec["levels"]:
        if lv["level"] not in ("parent", "coarse", "categorical", "null"):
            raise ValueError(f"{lv['name']}: level must be parent | coarse | categorical | null "
                             "(fine = the v2 baseline)")
        is_null = lv["level"] == "null"
        ldir = os.path.join(spec["levels_dir"], lv["name"]) + "/"
        cdir = os.path.join(out_cfg, lv["name"])
        files = {}
        files["link.yaml"] = (
            f"# Environment level {lv['name']}: movement files from the v2 run (symlinks); folds stay in "
            f"{spec['base_dir']}folds/.\n"
            f"#   python -m naturenet.environment.link_movement_run -y {_rel(os.path.join(cdir, 'link.yaml'), base)}\n"
            + yaml.safe_dump(dict(src_dir=spec["base_dir"], dst_dir=ldir, df_run_uid=spec["df_run_uid"],
                                  exclude=["*_states_abstract_env.pkl", "*_trans_prob_abstract_env.npz",
                                           "*_total_count_abstract_env.pkl", "*_points_abstract_env.pkl"],
                                  copy_dirs=[], dry_run=False), sort_keys=False))
        if is_null:
            files["null.yaml"] = null_text(spec, lv, ldir, cdir, base)
        else:
            files["env_format.yaml"] = header(spec_path, tm["env_format"], lv) + env_format_text(
                read(tm["env_format"]), spec, lv, ldir)
            ab = replace_prefix(read(tm["abstract_env"]), spec["base_dir"], ldir)
            ab = set_key(ab, "n_clusters", "null", "synced by sync_v2_configs (level K)", indent="", append=False)
            files["abstract_env.yaml"] = header(spec_path, tm["abstract_env"], lv) + ab
        base_sync = yaml.safe_load(read(tm["sync"]))
        run_names = [r for r in spec["runs"] if not is_null or r in (lv.get("runs") or spec.get("null_runs")
                                                                       or spec["runs"])]
        run_files = [os.path.join(_rel(cdir, base), f"{r}.yaml") for r in run_names]
        files["sync.yaml"] = (
            f"# Environment level {lv['name']}: n_states = K_level + 1 in this level's configs only.\n"
            f"#   python -m naturenet.environment.sync_v2_configs patch -y {_rel(os.path.join(cdir, 'sync.yaml'), base)}\n"
            + yaml.safe_dump(dict(env_meta_json=base_sync["env_meta_json"], class_mapper=mapper_path(spec, ldir),
                                  invalid_state=base_sync.get("invalid_state", "absorbing"),
                                  sample_config=base_sync["sample_config"],
                                  files=([] if is_null else [os.path.join(_rel(cdir, base), "abstract_env.yaml")])
                                  + run_files),
                             sort_keys=False))
        for r, tp in spec["runs"].items():
            if is_null and r not in (lv.get("runs") or spec.get("null_runs") or spec["runs"]):
                continue
            files[f"{r}.yaml"] = header(spec_path, tp, lv) + run_text(read(tp), spec, lv, ldir,
                                                                      (spec.get("overrides") or {}).get(r))
        for name, text in files.items():
            yaml.safe_load(text)
            written[os.path.join(cdir, name)] = text
    # LL comparison across levels: fine entries as given (yaml relative to out_config_dir), then per_level
    # templates ({lv} = level name, run = generated config) for every level.
    llc = spec["ll_comparison"]
    methods = [dict(e) for e in (llc.get("fine") or [])]
    for i, lv in enumerate(spec["levels"]):
        for e in llc.get("per_level") or []:
            if not os.path.join(out_cfg, lv["name"], f"{e['run']}.yaml") in written:
                continue                                   # run not generated at this level (e.g. nulls)
            d = {k: (v.format(lv=lv["name"]) if isinstance(v, str) else v) for k, v in e.items() if k != "run"}
            d["yaml"] = os.path.join(lv["name"], f"{e['run']}.yaml")
            d.setdefault("color", PALETTE[len(methods) % len(PALETTE)])
            methods.append(d)
    ll = dict(out_dir=os.path.join(spec["levels_dir"], "ll_comparison") + "/", out_prefix=llc["out_prefix"],
              title=llc["title"], ci=0.95, compare_k=None, reference=llc.get("reference"), methods=methods)
    txt = ("# LL comparison across environment levels (fine = v2 baseline). Generated by make_level_configs.\n"
           f"#   python -m naturenet.models.irl.plot_ll_comparison -y {_rel(os.path.join(out_cfg, 'll_comparison_levels.yaml'), base)}\n"
           "# Same folds / sequences / actions at every level, so paired test-LL differences vs `reference` are valid.\n"
           "# Entries whose runs are missing (e.g. SWIRL at levels where it was not run) are reported and skipped.\n"
           + yaml.safe_dump(ll, sort_keys=False, width=200))
    written[os.path.join(out_cfg, "ll_comparison_levels.yaml")] = txt
    for p, t in written.items():
        print(("would write " if dry_run else "wrote ") + _rel(p, base))
        if not dry_run:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(t)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-y", "--yaml", required=True)
    p.add_argument("--base_dir", default=None, help="root for relative paths (default: the naturenet package dir)")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args(argv)
    with open(a.yaml) as f:
        spec = yaml.safe_load(f)
    base = a.base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate(spec, a.yaml, base, a.dry_run)


if __name__ == "__main__":
    main()
