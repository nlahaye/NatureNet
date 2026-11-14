"""
Write a sit_fuse.local_launcher plan (YAML) for the NatureNet IRL runs of one or more configs, so the independent
pieces (K x run x fold, scale variant x fold, multi-agent null replicates) run side by side on the local GPUs / CPUs
with the right order between them:

    python -m naturenet.models.irl.plan_jobs --out plan.yaml --state_dir /data/nlahaye/launch/sharks_v2 \
        --swirl   <swirl_kfold.yaml>          # get_stats (ARHMM, CPU) -> get_stats_2 (SWIRL, GPU) per (K, run, fold)
        --hiavi   <hiavi_kfold.yaml>          # get_stats_hiavi per (K, run, fold) (CPU; restarts in parallel)
        --iavi    <iavi_kfold.yaml>           # run_hiavi_kfold per fold (one K from the YAML; e.g. IAVI K = 1)
        --nn_swirl <swirl_nn_kfold.yaml>      # get_stats_nn (NN ARHMM init, CPU) -> get_stats_3 (NN-SWIRL S-1, GPU)
                                              # and get_stats_3_s2 (S-2, GPU; run_s2_nn: False skips) per (K, run, fold)
        --scales  <swirl_scales.yaml>         # run_swirl_scales_kfold per (variant, fold); N2 after N1, ... (the
                                              # same fold); swirl emission waits for the ARHMM init it loads
        --multiagent <ma.yaml>[@opt...]       # parts: obs, null chunks -> merge (summary, final fits)
        --scales_multiagent <ma_scales.yaml>[@opt...]   # per (backend, scale variant): parts -> merge
    python -m sit_fuse.local_launcher run plan.yaml        # status / --dry-run / --only 'swirl*' ...

Every option may be repeated (several configs in one plan). Multi-agent entries take per-entry options after the
YAML path, separated by "@" (they override the YAML for that entry only):
    @backend=hiavi|swirl                  --multiagent: model of the coupled fits (default multiagent.backend;
                                          the global --ma_backend applies to entries without one)
    @backends=hiql_scales,swirl_scales    --scales_multiagent: backends (default multiagent.scales_backends)
    @variants=S1_N1,S1_N2                 --scales_multiagent: scale variants (default every scales_variants entry)
    @K=2,5                                Ks (default multiagent.Ks)
    @skip_null                            observed CV only (screen): no null jobs, merge with --skip_null; a later
                                          entry without @skip_null reuses the observed parts and adds the nulls
e.g. --multiagent whales_multiagent.yaml@K=2  --scales_multiagent whales_scales_multiagent.yaml@backends=hiql_scales@skip_null Jobs that are already finished are skipped by the runners
themselves (and by the launcher's markers), so the plan can be regenerated and rerun at any time.

Resources (defaults sized for A100s; adjust): --gpu_share 0.25 (four SWIRL / scales fits share one GPU;
XLA_PYTHON_CLIENT_PREALLOCATE=false and MEM_FRACTION = share are set per job), --arhmm_cpus 1, --swirl_cpus 2,
--hiavi_cpus 4 (= hiavi_restart_jobs of that job), --ma_cpus 4, --null_chunk 2 (null replicates per job).
Kinematic emissions need no stage of their own: give a --swirl / --hiavi / --iavi / --scales / --multiagent YAML with
a kinematics: block (and its own run_uid); the ARHMM job then fits the kinematic ARHMM that the SWIRL job loads.
--precision fp64|fp32|tf32|bf16 sets the JAX precision tier of every job (jax_precision.py; fp64 = as before).
A persistent JAX compilation cache is shared by all jobs (<state_dir>/jax_cache) unless --no_jax_cache.
"""
import argparse
import os
import shlex
import sys

import yaml

STAGES = ("swirl", "hiavi", "iavi", "nn_swirl", "scales", "multiagent", "scales_multiagent")


def _read(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _tag(path):
    b = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return f"{parent}-{b}".replace("_", "-")


def _folds(conf):
    n = int(conf.get("n_folds", 5))
    sel = conf.get("folds_to_run")
    if sel is None:
        return list(range(n))
    return [int(k) for k in (sel if isinstance(sel, (list, tuple)) else [sel])]


def _loops(conf):
    """[(K, run, run_uid, seed)] exactly as swirl_data.iter_loop_configs."""
    out = []
    for K in range(int(conf["min_latent"]), int(conf["max_latent"]) + 1):
        for i in range(int(conf["num_runs"])):
            out.append((K, i, f"{conf['run_uid']}_loop_{i}_latent_{K}_states", int(conf["seed"]) + i))
    return out


def _arhmm_path(out_dir, run_uid, K, seed, fold):
    return os.path.join(out_dir, f"{run_uid}_{K}_hidden_{seed}_seed_arhmm_s_KF_{fold}.npz")


def _arhmm_nn_path(out_dir, run_uid, K, seed, fold):
    return os.path.join(out_dir, f"{run_uid}_{K}_hidden_{seed}_seed_arhmm_nn_s_KF_{fold}.npz")


def _cases(rows):
    return [{k: v for k, v in r.items()} for r in rows]


def swirl_jobs(cfg, a):
    conf, t, q = _read(cfg), _tag(cfg), shlex.quote(cfg)
    rows = [dict(K=K, run=i, fold=k, arhmm=_arhmm_path(conf["out_dir"], uid, K, seed, k))
            for K, i, uid, seed in _loops(conf) for k in _folds(conf)]
    jobs = []
    if not a.no_arhmm:
        jobs.append(dict(name=f"arhmm.{t}", cpus=a.arhmm_cpus, gpu=0, env=_cpu_env(a.arhmm_cpus),
                         cmd=f"{{py}} -m naturenet.models.irl.get_stats -y {q} --n_hidden {{K}} --run {{run}} "
                             f"--fold {{fold}}",
                         cases=_cases(rows), provides=["{arhmm}"], id_keys=["K", "run", "fold"]))
    jobs.append(dict(name=f"swirl.{t}", cpus=a.swirl_cpus, gpu=a.gpu_share,
                     cmd=f"{{py}} -m naturenet.models.irl.get_stats_2 -y {q} --n_hidden {{K}} --run {{run}} "
                         f"--fold {{fold}}",
                     cases=_cases(rows), needs=["{arhmm}"], id_keys=["K", "run", "fold"]))
    return jobs


def hiavi_jobs(cfg, a):
    conf, t, q = _read(cfg), _tag(cfg), shlex.quote(cfg)
    rows = [dict(K=K, run=i, fold=k) for K, i, _, _ in _loops(conf) for k in _folds(conf)]
    return [dict(name=f"hiavi.{t}", cpus=a.hiavi_cpus, gpu=0, env=dict(_cpu_env(a.hiavi_cpus),
                                                                       NATURENET_HIAVI_RESTART_JOBS=str(a.hiavi_cpus)),
                 cmd=f"{{py}} -m naturenet.models.irl.get_stats_hiavi -y {q} --n_hidden {{K}} --run {{run}} "
                     f"--fold {{fold}}",
                 cases=_cases(rows))]


def iavi_jobs(cfg, a):
    conf, t, q = _read(cfg), _tag(cfg), shlex.quote(cfg)
    return [dict(name=f"iavi.{t}", cpus=a.hiavi_cpus, gpu=0, env=dict(_cpu_env(), NATURENET_HIAVI_RESTART_JOBS=str(
        a.hiavi_cpus)), cmd=f"{{py}} -m naturenet.models.irl.run_hiavi_kfold -y {q} --fold {{fold}}",
        cases=[dict(fold=k) for k in _folds(conf)])]


def nn_swirl_jobs(cfg, a):
    """NN-SWIRL K-fold: NN ARHMM init (ssm, CPU) -> S-1 and S-2 NN fits (GPU share), per (K, run, fold)."""
    conf, t, q = _read(cfg), _tag(cfg), shlex.quote(cfg)
    if (conf.get("kinematics") or {}).get("enabled", False):
        print(f"note: {cfg}: NN-SWIRL has no kinematic emissions; the kinematics block is ignored")
    rows = [dict(K=K, run=i, fold=k, arhmm_nn=_arhmm_nn_path(conf["out_dir"], uid, K, seed, k))
            for K, i, uid, seed in _loops(conf) for k in _folds(conf)]
    args = "--n_hidden {K} --run {run} --fold {fold}"
    jobs = []
    if not a.no_arhmm:
        jobs.append(dict(name=f"arhmm_nn.{t}", cpus=a.arhmm_cpus, gpu=0, env=_cpu_env(a.arhmm_cpus),
                         cmd=f"{{py}} -m naturenet.models.irl.get_stats_nn -y {q} {args}",
                         cases=_cases(rows), provides=["{arhmm_nn}"], id_keys=["K", "run", "fold"]))
    stages = [("nn_s1", "get_stats_3")] + ([("nn_s2", "get_stats_3_s2")] if conf.get("run_s2_nn", True) else [])
    for name, mod in stages:
        jobs.append(dict(name=f"{name}.{t}", cpus=a.swirl_cpus, gpu=a.gpu_share,
                         cmd=f"{{py}} -m naturenet.models.irl.{mod} -y {q} {args}",
                         cases=_cases(rows), needs=["{arhmm_nn}"], id_keys=["K", "run", "fold"]))
    return jobs


def scales_jobs(cfg, a):
    from naturenet.models.irl import run_swirl_scales_kfold as rs
    from naturenet.models.irl import swirl_scales as sc
    conf, t, q = _read(cfg), _tag(cfg), shlex.quote(cfg)
    emission = rs.scales_emission(conf)
    init = (conf.get("scales") or {}).get("init", "auto")
    K = int(conf["n_hidden"])
    rows = []
    for tag, vconf in rs.scales_variants(conf):
        cfg_v = sc.ScalesConfig.from_dict(vconf)
        ptag = rs._parent_tag(cfg_v)
        for k in _folds(conf):
            r = dict(variant=tag, fold=k, par=rs._names(conf, tag, k)[1],
                     parent=rs._names(conf, ptag, k)[1] if ptag else "")
            if emission == "swirl" and init in ("auto", "arhmm"):
                r["arhmm"] = _arhmm_path(conf["out_dir"], conf["run_uid"], int(conf.get("n_hidden_init", K)),
                                         int(conf["seed"]), k)
            rows.append(r)
    needs = [n for n in ("{parent}", "{arhmm}") if any(n[1:-1] in r and r[n[1:-1]] for r in rows)]
    for r in rows:
        r.setdefault("arhmm", "")
    gpu = a.gpu_share if emission == "swirl" else 0
    cpus = a.swirl_cpus if emission == "swirl" else a.hiavi_cpus
    env = {} if gpu else dict(_cpu_env(cpus), NATURENET_HIAVI_RESTART_JOBS=str(cpus))
    return [dict(name=f"scales.{t}", cpus=cpus, gpu=gpu, env=env, cases=_cases(rows),
                 cmd=f"{{py}} -m naturenet.models.irl.run_swirl_scales_kfold -y {q} --fold {{fold}} "
                     f"--variant {{variant}}",
                 provides=["{par}"], needs=needs, id_keys=["variant", "fold"], auto_init=init == "auto")]


def _entry(spec):
    """'path[@key=val[@flag]...]' -> (abs path, options dict). Values with commas -> lists."""
    path, *opts = spec.split("@")
    out = {}
    for o in opts:
        k, _, v = o.partition("=")
        k = k.strip()
        if k not in ("backend", "backends", "variants", "K", "skip_null"):
            raise SystemExit(f"{spec}: unknown option @{k} (backend, backends, variants, K, skip_null)")
        if k == "skip_null":
            out[k] = v.strip().lower() not in ("0", "false", "no") if v else True
        elif k == "backend":
            out[k] = v.strip()
        else:
            out[k] = [x.strip() for x in v.split(",") if x.strip()]
            if k == "K":
                out[k] = [int(x) for x in out[k]]
    return os.path.abspath(path), out


def _opt_tag(o):
    t = ""
    if o.get("K"):
        t += ".K" + "-".join(str(k) for k in o["K"])
    if o.get("variants"):
        t += "." + "-".join(o["variants"])
    if o.get("skip_null"):
        t += ".screen"
    return t.replace("_", "-")


def _null_chunks(n, size):
    size = max(1, int(size))
    return [(i, min(n, i + size) - 1) for i in range(0, n, size)]


def _ma_res(backend, a):
    if backend in ("swirl", "swirl_scales"):
        return a.ma_cpus, a.gpu_share, {}
    return a.ma_cpus, 0, dict(_cpu_env(a.ma_cpus), NATURENET_HIAVI_RESTART_JOBS=str(a.ma_cpus))


def multiagent_jobs(spec, a):
    from naturenet.models.irl.run_hiavi_multiagent import ma_conf
    cfg, o = _entry(spec)
    conf, q = _read(cfg), shlex.quote(cfg)
    m = ma_conf(conf)
    backend = o.get("backend") or a.ma_backend or m.get("backend") or "hiavi"
    Ks = o.get("K") or [int(K) for K in m["Ks"]]
    t = f"{_tag(cfg)}.{backend}{_opt_tag(o)}".replace("_", "-")
    cpus, gpu, env = _ma_res(backend, a)
    base = f"{{py}} -m naturenet.models.irl.run_hiavi_multiagent -y {q} --backend {backend}"
    merge = base + " --K " + " ".join(str(K) for K in Ks) + (" --skip_null" if o.get("skip_null") else "")
    jobs = [dict(name=f"ma_obs.{t}", cpus=cpus, gpu=gpu, env=env, cases=[dict(K=K, part="obs") for K in Ks],
                 cmd=base + " --K {K} --part {part}")]
    after = [f"ma_obs.{t}"]
    if not o.get("skip_null"):
        nulls = [dict(K=K, part=f"null:{i}-{j}") for K in Ks for i, j in _null_chunks(int(m["n_null"]), a.null_chunk)]
        jobs.append(dict(name=f"ma_null.{t}", cpus=cpus, gpu=gpu, env=env, cases=nulls, after=[f"ma_obs.{t}"],
                         cmd=base + " --K {K} --part {part}"))
        after.append(f"ma_null.{t}")
    jobs.append(dict(name=f"ma_merge.{t}", cpus=cpus, gpu=gpu, env=env, after=after, cmd=merge))
    return jobs


def scales_multiagent_jobs(spec, a):
    from naturenet.models.irl.run_hiavi_multiagent import ma_conf
    from naturenet.models.irl.run_scales_multiagent import BACKENDS, scale_variants
    cfg, o = _entry(spec)
    conf = _read(cfg)          # (input globs are expanded by the jobs themselves)
    t, q = _tag(cfg), shlex.quote(cfg)
    m = ma_conf(conf)
    backends = o.get("backends") or list((conf.get("multiagent") or {}).get("scales_backends") or BACKENDS)
    bad = [b for b in backends if b not in BACKENDS]
    if bad:
        raise SystemExit(f"{spec}: unknown scales backend(s) {bad}; use {list(BACKENDS)}")
    all_v = [v[0] for v in scale_variants(conf)]
    variants = o.get("variants") or all_v
    bad = [v for v in variants if v not in all_v]
    if bad:
        raise SystemExit(f"{spec}: variant(s) {bad} not in multiagent.scales_variants {all_v}")
    Ks = o.get("K") or [int(K) for K in m["Ks"]]
    ks = " ".join(str(K) for K in Ks)
    skip = " --skip_null" if o.get("skip_null") else ""
    jobs = []
    for backend in backends:
        cpus, gpu, env = _ma_res(backend, a)
        base = f"{{py}} -m naturenet.models.irl.run_scales_multiagent -y {q} --backend {backend} --variant {{variant}}"
        nb = f"{t}.{backend}{_opt_tag(o)}".replace("_", "-")
        jobs.append(dict(name=f"msc_obs.{nb}", cpus=cpus, gpu=gpu, env=env,
                         cases=[dict(variant=v, K=K, part="obs") for v in variants for K in Ks],
                         cmd=base + " --K {K} --part {part}"))
        after = [dict(name=f"msc_obs.{nb}", on=["variant"])]
        if not o.get("skip_null"):
            jobs.append(dict(name=f"msc_null.{nb}", cpus=cpus, gpu=gpu, env=env,
                             after=[dict(name=f"msc_obs.{nb}", on=["variant", "K"])],
                             cases=[dict(variant=v, K=K, part=f"null:{i}-{j}") for v in variants for K in Ks
                                    for i, j in _null_chunks(int(m["n_null"]), a.null_chunk)],
                             cmd=base + " --K {K} --part {part}"))
            after.append(dict(name=f"msc_null.{nb}", on=["variant"]))
        jobs.append(dict(name=f"msc_merge.{nb}", cpus=cpus, gpu=gpu, env=env, cases=[dict(variant=v) for v in variants],
                         after=after, cmd=base + f" --K {ks}{skip}"))
    return jobs


def _cpu_env(n=None):
    return {"NATURENET_JAX_PLATFORM": "cpu", "JAX_PLATFORMS": "cpu"}


def build_plan(a):
    from naturenet.models.irl.jax_precision import TIERS
    env = {"PYTHONUNBUFFERED": "1"}
    if a.precision:
        env.update({k: v for k, v in TIERS[a.precision].items()})
    if not a.no_jax_cache:
        env["JAX_COMPILATION_CACHE_DIR"] = os.path.join(a.state_dir, "jax_cache")
        env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "2"
    launcher = dict(state_dir=a.state_dir, gpus=a.gpus, cpus=a.cpus, retries=a.retries, poll_sec=5, env=env)
    jobs = []
    for stage, fn in (("swirl", swirl_jobs), ("hiavi", hiavi_jobs), ("iavi", iavi_jobs), ("nn_swirl", nn_swirl_jobs),
                      ("scales", scales_jobs),
                      ("multiagent", multiagent_jobs), ("scales_multiagent", scales_multiagent_jobs)):
        for cfg in getattr(a, stage) or []:
            jobs += fn(cfg if stage in ("multiagent", "scales_multiagent") else os.path.abspath(cfg), a)
    names = [j["name"] for j in jobs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"duplicate job groups {dup}: the same YAML / options are listed twice")
    # scales with init: auto use the ARHMM init only if it exists or a planned job writes it (else sticky_init)
    provided = {os.path.realpath(c["arhmm"]) for j in jobs if j["name"].startswith("arhmm.") for c in j["cases"]}
    for j in jobs:
        if j["name"].startswith("scales.") and j.get("auto_init"):
            gone = [c for c in j["cases"] if c["arhmm"] and os.path.realpath(c["arhmm"]) not in provided
                    and not os.path.exists(c["arhmm"])]
            for c in gone:
                c["arhmm"] = ""
            if gone:
                print(f"note: {j['name']}: no ARHMM init for {len(gone)} (variant, fold) jobs; "
                      f"scales.init auto -> sticky_init")
        j.pop("auto_init", None)
    if a.cpu_only:                      # e.g. a machine without GPUs, or a CPU smoke test of the plan
        for j in jobs:
            j["gpu"] = 0
            j["env"] = dict(j.get("env") or {}, **_cpu_env())
    return dict(launcher=launcher, vars=dict(py=a.python), jobs=jobs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="plan YAML to write")
    ap.add_argument("--state_dir", required=True, help="launcher logs / markers")
    for s in STAGES:
        ap.add_argument("--" + s, action="append", default=[], metavar="YAML")
    ap.add_argument("--ma_backend", default=None, help="--multiagent entries without @backend= (hiavi | swirl)")
    ap.add_argument("--no_arhmm", action="store_true", help="--swirl / --nn_swirl: ARHMM inits exist already (no get_stats / "
                    "get_stats_nn jobs)")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--cpus", default="auto")
    ap.add_argument("--gpu_share", type=float, default=0.25)
    ap.add_argument("--arhmm_cpus", type=int, default=1)
    ap.add_argument("--swirl_cpus", type=int, default=2)
    ap.add_argument("--hiavi_cpus", type=int, default=4)
    ap.add_argument("--ma_cpus", type=int, default=4)
    ap.add_argument("--null_chunk", type=int, default=2)
    ap.add_argument("--retries", type=int, default=0)
    ap.add_argument("--precision", choices=["fp64", "fp32", "tf32", "bf16"], default=None)
    ap.add_argument("--no_jax_cache", action="store_true")
    ap.add_argument("--cpu_only", action="store_true", help="every job on CPU (JAX platform cpu, no GPU share)")
    ap.add_argument("--python", default=shlex.quote(sys.executable))
    a = ap.parse_args(argv)
    if not any(getattr(a, s) for s in STAGES):
        ap.error("give at least one of --" + " / --".join(STAGES))
    plan = build_plan(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        yaml.safe_dump(plan, f, sort_keys=False, width=140)
    n = sum(max(1, len(j.get("cases", []))) for j in plan["jobs"])
    print(f"wrote {a.out}: {len(plan['jobs'])} job groups, {n} jobs")
    for j in plan["jobs"]:
        print(f"  {j['name']:48s} {max(1, len(j.get('cases', []))):5d}  cpus={j['cpus']} gpu={j['gpu']}")


if __name__ == "__main__":
    main()
