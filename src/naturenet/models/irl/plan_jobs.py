"""
Write a sit_fuse.local_launcher plan (YAML) for the NatureNet IRL runs of one or more configs, so the independent
pieces (K x run x fold, scale variant x fold, multi-agent null replicates) run side by side on the local GPUs / CPUs
with the right order between them:

    python -m naturenet.models.irl.plan_jobs --out plan.yaml --state_dir /data/nlahaye/launch/sharks_v2 \
        --swirl   <swirl_kfold.yaml>          # get_stats (ARHMM, CPU) -> get_stats_2 (SWIRL, GPU) per (K, run, fold)
        --hiavi   <hiavi_kfold.yaml>          # get_stats_hiavi per (K, run, fold) (CPU; restarts in parallel)
        --iavi    <iavi_kfold.yaml>           # run_hiavi_kfold per fold (one K from the YAML; e.g. IAVI K = 1)
        --scales  <swirl_scales.yaml>         # run_swirl_scales_kfold per (variant, fold); N2 after N1, ... (the
                                              # same fold); swirl emission waits for the ARHMM init it loads
        --multiagent <ma.yaml> [--ma_backend hiavi|swirl]   # parts: obs, null chunks -> merge (summary, final fits)
        --scales_multiagent <ma_scales.yaml>  # per (backend, scale variant): parts -> merge
    python -m sit_fuse.local_launcher run plan.yaml        # status / --dry-run / --only 'swirl*' ...

Every option may be repeated (several configs in one plan). Jobs that are already finished are skipped by the runners
themselves (and by the launcher's markers), so the plan can be regenerated and rerun at any time.

Resources (defaults sized for A100s; adjust): --gpu_share 0.25 (four SWIRL / scales fits share one GPU;
XLA_PYTHON_CLIENT_PREALLOCATE=false and MEM_FRACTION = share are set per job), --arhmm_cpus 1, --swirl_cpus 2,
--hiavi_cpus 4 (= hiavi_restart_jobs of that job), --ma_cpus 4, --null_chunk 2 (null replicates per job).
--precision fp64|fp32|tf32|bf16 sets the JAX precision tier of every job (jax_precision.py; fp64 = as before).
A persistent JAX compilation cache is shared by all jobs (<state_dir>/jax_cache) unless --no_jax_cache.
"""
import argparse
import os
import shlex
import sys

import yaml

STAGES = ("swirl", "hiavi", "iavi", "scales", "multiagent", "scales_multiagent")


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


def _null_chunks(n, size):
    size = max(1, int(size))
    return [(i, min(n, i + size) - 1) for i in range(0, n, size)]


def _ma_res(backend, a):
    if backend in ("swirl", "swirl_scales"):
        return a.ma_cpus, a.gpu_share, {}
    return a.ma_cpus, 0, dict(_cpu_env(a.ma_cpus), NATURENET_HIAVI_RESTART_JOBS=str(a.ma_cpus))


def multiagent_jobs(cfg, a):
    from naturenet.models.irl.run_hiavi_multiagent import ma_conf
    conf, t, q = _read(cfg), _tag(cfg), shlex.quote(cfg)
    m = ma_conf(conf)
    backend = a.ma_backend or m.get("backend") or "hiavi"
    cpus, gpu, env = _ma_res(backend, a)
    base = f"{{py}} -m naturenet.models.irl.run_hiavi_multiagent -y {q} --backend {backend}"
    parts = [dict(K=int(K), part="obs") for K in m["Ks"]]
    nulls = [dict(K=int(K), part=f"null:{i}-{j}") for K in m["Ks"] for i, j in _null_chunks(int(m["n_null"]),
                                                                                          a.null_chunk)]
    return [dict(name=f"ma_obs.{t}", cpus=cpus, gpu=gpu, env=env, cases=parts,
                 cmd=base + " --K {K} --part {part}"),
            dict(name=f"ma_null.{t}", cpus=cpus, gpu=gpu, env=env, cases=nulls, after=[f"ma_obs.{t}"],
                 cmd=base + " --K {K} --part {part}"),
            dict(name=f"ma_merge.{t}", cpus=cpus, gpu=gpu, env=env, after=[f"ma_obs.{t}", f"ma_null.{t}"],
                 cmd=base)]


def scales_multiagent_jobs(cfg, a):
    from naturenet.models.irl.run_hiavi_multiagent import ma_conf
    from naturenet.models.irl.run_scales_multiagent import BACKENDS, scale_variants
    conf = _read(cfg)          # (input globs are expanded by the jobs themselves)
    t, q = _tag(cfg), shlex.quote(cfg)
    m = ma_conf(conf)
    backends = list((conf.get("multiagent") or {}).get("scales_backends") or BACKENDS)
    jobs = []
    for backend in backends:
        cpus, gpu, env = _ma_res(backend, a)
        variants = [v[0] for v in scale_variants(conf)]
        base = f"{{py}} -m naturenet.models.irl.run_scales_multiagent -y {q} --backend {backend} --variant {{variant}}"
        nb = f"{t}.{backend}".replace("_", "-")
        jobs += [dict(name=f"msc_obs.{nb}", cpus=cpus, gpu=gpu, env=env,
                      cases=[dict(variant=v, K=int(K), part="obs") for v in variants for K in m["Ks"]],
                      cmd=base + " --K {K} --part {part}"),
                 dict(name=f"msc_null.{nb}", cpus=cpus, gpu=gpu, env=env, after=[dict(name=f"msc_obs.{nb}",
                                                                                     on=["variant", "K"])],
                      cases=[dict(variant=v, K=int(K), part=f"null:{i}-{j}") for v in variants for K in m["Ks"]
                             for i, j in _null_chunks(int(m["n_null"]), a.null_chunk)],
                      cmd=base + " --K {K} --part {part}"),
                 dict(name=f"msc_merge.{nb}", cpus=cpus, gpu=gpu, env=env, cases=[dict(variant=v) for v in variants],
                      after=[dict(name=f"msc_obs.{nb}", on=["variant"]), dict(name=f"msc_null.{nb}", on=["variant"])],
                      cmd=base)]
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
    for stage, fn in (("swirl", swirl_jobs), ("hiavi", hiavi_jobs), ("iavi", iavi_jobs), ("scales", scales_jobs),
                      ("multiagent", multiagent_jobs), ("scales_multiagent", scales_multiagent_jobs)):
        for cfg in getattr(a, stage) or []:
            jobs += fn(os.path.abspath(cfg), a)
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
    ap.add_argument("--ma_backend", default=None)
    ap.add_argument("--no_arhmm", action="store_true", help="--swirl: ARHMM inits exist already (no get_stats jobs)")
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
