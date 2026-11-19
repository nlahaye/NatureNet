#!/usr/bin/env bash
# Environment-level analysis with the kinematic tools (HIQL + kin, SWIRL + kin) on sit_fuse.local_launcher.
# Same folds, sequences and kinematic block as the fine kinematic baselines, so paired test-LL differences against the
# fine HIQL + kin run are valid. Stages (in order; STAGE=all runs every one):
#   gen     make_level_configs (spec <sp>_levels_v2.yaml) -> set the K ranges below in the level CV configs -> sync
#           n_states of every built level (make_level_configs resets them to null)
#   cv      HIQL + kin K sweep (get_stats_hiavi) and SWIRL + kin CV (get_stats -> get_stats_2) per ready level
#           (plan_jobs; finished folds are skipped, and the K = 5 HIQL folds of hiavi_kfold_kin are reused)
#   ll      plot_ll_comparison -y <L>/<sp>/ll_comparison_kin.yaml (reference = fine HIQL + kin)
#   finals  per level: K = best mean test LL (or the overrides below) for HIQL + kin and SWIRL + kin, copies of
#           hiql_final_kin.yaml / swirl_final_kin.yaml with n_hidden = K, then fit -> plot -> reward projection
#           (cluster_interpret first for levels without xai/interpret_<lv>/; Seascapes needs interpret_seascape, 17c)
#   audit   manifest audit (run_available_outputs_<sp>.sh STAGE=audit)
# A level is skipped (with the reason) while its env is not built / synced or its CV is incomplete.
#   bash scripts/run_levels_kin_v2.sh                       # STAGE=all, both species
#   bash scripts/run_levels_kin_v2.sh cv whales             # one stage, one species
#   DRY=1 bash scripts/run_levels_kin_v2.sh                 # configs + plans + launcher dry runs only
#   python -m sit_fuse.local_launcher status /data/nlahaye/NatureNet/launcher/levels_kin_cv_sharks.yaml
# Generic: the stage logic only reads the spec, run names and labels below; another SIT-FUSE use case needs its own
# <spec>_levels_v2.yaml (runs / overrides / ll_comparisons) and the species table at the bottom of the settings.
set -euo pipefail

# ----------------------------------------------------------------------------- settings
NN_SRC=/home/nlahaye/NatureNet/src
PY=/home/nlahaye/SIT_FUSE/.pixi/envs/default/bin/python
S=/data/nlahaye/NatureNet/launcher           # plans + launcher state
SF_ROOT=/data/nlahaye/output/Learnergy       # SIT-FUSE outputs (xai/interpret_<lv>, hierarchy)
LEVELS="parent k30 k20 k12 seascape"
# resources: multi-agent (run_multiagent_v2.sh) holds GPUs 0-3 + 20 CPUs; this leaves GPUs 6-7 and 8 CPUs free
GPUS="4,5"
CPUS=20
GPU_SHARE=0.25                               # four SWIRL fits per A100
HIAVI_CPUS=4                                 # per HIQL job (= restart workers)
SWIRL_CPUS=2
ARHMM_CPUS=1
JOB_CPUS=4                                   # finals: HIQL fit / plots / projection / cluster_interpret
# CV design (all CV tests: min_latent = 1 for HIQL; SWIRL + kin at K = 5 as at the fine level's tool choice)
HIQL_KMIN=1
HIQL_KMAX=5
SWIRL_KMIN=5                                 # set 1 for a full SWIRL K sweep (5x the GPU work)
SWIRL_KMAX=5
NUM_RUNS=3                                   # seeds per (K, fold)
EXPECTED_N=15                                # rows per K in the LL summary (NUM_RUNS x 5 folds)
# run names in the spec (config/v2/levels/<sp>_levels_v2.yaml)
RUN_HIQL_CV=hiql_kfold_kin
RUN_SWIRL_CV=swirl_kfold_kin
RUN_HIQL_FINAL=hiql_final_kin
RUN_SWIRL_FINAL=swirl_final_kin
PROJ_HIQL=reward_env_projection_hiql
LL_NAME=kin                                  # -> ll_comparison_kin.yaml
# per species: SWIRL order used to pick K, projection runs of the SWIRL final, K overrides ("lv=K ...", empty = best)
SWIRL_ORDER_sharks="S-1"
SWIRL_ORDER_whales="S-2"
PROJ_SWIRL_sharks="reward_env_projection_swirl_kin"
PROJ_SWIRL_whales="reward_env_projection_swirl_kin reward_env_projection_swirl_kin_s1"
NH_HIQL_sharks=""
NH_HIQL_whales=""
NH_SWIRL_sharks=""
NH_SWIRL_whales=""
STAGE="${1:-all}"
SPECIES="${2:-sharks whales}"
DRY="${DRY:-0}"
# ----------------------------------------------------------------------------- end settings

L=naturenet/config/v2/levels
SFC="$("$PY" -c 'import os, sit_fuse; print(os.path.join(os.path.dirname(sit_fuse.__file__), "config"))')"
CPUENV="{JAX_PLATFORMS: cpu, NATURENET_JAX_PLATFORM: cpu, CUDA_VISIBLE_DEVICES: \"\", OMP_NUM_THREADS: \"$JOB_CPUS\", MKL_NUM_THREADS: \"$JOB_CPUS\"}"
cpu() { env JAX_PLATFORMS=cpu NATURENET_JAX_PLATFORM=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$JOB_CPUS MKL_NUM_THREADS=$JOB_CPUS "$@"; }
want() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }
launch() {
  "$PY" -m sit_fuse.local_launcher run "$1" --dry-run
  [ "$DRY" = "1" ] || "$PY" -m sit_fuse.local_launcher run "$1" \
    || echo "WARNING: some jobs failed ($PY -m sit_fuse.local_launcher status $1)"
}
cd "$NN_SRC"
eval "$("$PY" -m naturenet.environment.proj_env --export 2>/dev/null | grep "^export PROJ_" || true)"
mkdir -p "$S"

# n_states filled in every run config of the level the stages use
level_ready() {
  local sp="$1" lv="$2" f
  for f in "$L/$sp/$lv/$RUN_HIQL_CV.yaml" "$L/$sp/$lv/$RUN_SWIRL_CV.yaml"; do
    [ -f "$f" ] || { echo "SKIP $lv: no $f (rerun STAGE=gen)"; return 1; }
    grep -Eq '^n_states:[[:space:]]*[0-9]+' "$f" || { echo "SKIP $lv: $f not synced (level env not built)"; return 1; }
  done
}

for sp in $SPECIES; do
  case "$sp" in whales) sg=whale ;; sharks) sg=shark ;; *) echo "unknown species $sp"; exit 1 ;; esac
  echo "================ $sp ($STAGE)"

  # ------------------------------------------------------------------- gen
  if want gen; then
    if [ "$DRY" = "1" ]; then "$PY" -m naturenet.environment.make_level_configs -y "$L/${sp}_levels_v2.yaml" --dry_run | tail -3
    else
      "$PY" -m naturenet.environment.make_level_configs -y "$L/${sp}_levels_v2.yaml" > /dev/null
      # K ranges / seeds from the settings (comments kept)
      "$PY" - "$L/$sp" "$LEVELS" "$RUN_HIQL_CV" "$HIQL_KMIN" "$HIQL_KMAX" "$RUN_SWIRL_CV" "$SWIRL_KMIN" "$SWIRL_KMAX" \
              "$NUM_RUNS" <<'PYEOF'
import os, re, sys
d, levels, rh, hmin, hmax, rs, smin, smax, nr = sys.argv[1:]
def put(path, kv):
    t = open(path).read()
    for k, v in kv.items():
        t, n = re.subn(rf"(?m)^{k}:\s*[^#\n]*", f"{k}: {v}   ", t, count=1)
        if not n: t += f"\n{k}: {v}   # run_levels_kin_v2.sh\n"
    open(path, "w").write(t)
for lv in levels.split():
    for run, kmin, kmax, extra in ((rh, hmin, hmax, {}), (rs, smin, smax, {"n_hidden": smax, "n_hidden_init": smax})):
        p = os.path.join(d, lv, f"{run}.yaml")
        if os.path.exists(p):
            put(p, dict(min_latent=kmin, max_latent=kmax, num_runs=nr, **extra))
print(f"K ranges: {rh} {hmin}-{hmax}, {rs} {smin}-{smax}, {nr} runs")
PYEOF
      for lv in $LEVELS; do
        m=$(grep -E '^class_mapper:' "$L/$sp/$lv/sync.yaml" 2>/dev/null | sed -E 's/^class_mapper:[[:space:]]*"?([^"#]*)"?.*/\1/' | xargs || true)
        if [ -n "$m" ] && [ ! -f "$m" ]; then echo "NOT BUILT $lv: no $m (run run_available_outputs_${sp}.sh STAGE=levels / seascape)"; continue; fi
        "$PY" -m naturenet.environment.sync_v2_configs patch -y "$L/$sp/$lv/sync.yaml" > /dev/null \
          && echo "synced $lv" || echo "WARNING: sync failed for $lv"
      done
    fi
  fi

  # ------------------------------------------------------------------- cv
  if want cv; then
    a=()
    for lv in $LEVELS; do
      level_ready "$sp" "$lv" || continue
      a+=(--hiavi "$L/$sp/$lv/$RUN_HIQL_CV.yaml" --swirl "$L/$sp/$lv/$RUN_SWIRL_CV.yaml")
    done
    if [ ${#a[@]} -gt 0 ]; then
      plan="$S/levels_kin_cv_${sp}.yaml"
      "$PY" -m naturenet.models.irl.plan_jobs --out "$plan" --state_dir "$S/levels_kin_${sp}" "${a[@]}" \
        --gpus "$GPUS" --cpus "$CPUS" --gpu_share "$GPU_SHARE" --hiavi_cpus "$HIAVI_CPUS" \
        --swirl_cpus "$SWIRL_CPUS" --arhmm_cpus "$ARHMM_CPUS"
      launch "$plan"
    else echo "SKIP cv: no ready level"; fi
  fi

  # ------------------------------------------------------------------- ll
  if want ll || want finals; then
    if [ "$DRY" = "1" ]; then echo "would run plot_ll_comparison -y $L/$sp/ll_comparison_$LL_NAME.yaml"
    else cpu "$PY" -m naturenet.models.irl.plot_ll_comparison -y "$L/$sp/ll_comparison_$LL_NAME.yaml" \
           || echo "WARNING: plot_ll_comparison failed (finals use the existing summary)"; fi
  fi

  # ------------------------------------------------------------------- finals
  if want finals; then
    plan="$S/levels_kin_finals_${sp}.yaml"
    so="SWIRL_ORDER_$sp"; ps="PROJ_SWIRL_$sp"; nhh="NH_HIQL_$sp"; nhs="NH_SWIRL_$sp"
    "$PY" - "$sp" "$sg" "$L" "$LEVELS" "$EXPECTED_N" "$plan" "$S" "$PY" "$SFC" "$SF_ROOT" "$CPUS" "$JOB_CPUS" \
            "$CPUENV" "$GPUS" "$GPU_SHARE" "$LL_NAME" "$RUN_HIQL_CV" "$RUN_SWIRL_CV" "$RUN_HIQL_FINAL" \
            "$RUN_SWIRL_FINAL" "$PROJ_HIQL" "${!ps}" "${!so}" "${!nhh}" "${!nhs}" <<'PYEOF'
import os, re, sys, yaml
import pandas as pd
(sp, sg, L, levels, exp_n, plan, S, py, sfc, sfroot, cpus, jcpus, cpuenv, gpus, share, llname, rhcv, rscv,
 rhf, rsf, projh, projs, sorder, nhh, nhs) = sys.argv[1:]
exp_n, jcpus, share = int(exp_n), int(jcpus), float(share)
ovr = lambda s: dict(x.split("=") for x in s.split()) if s.strip() else {}
over = dict(hiql=ovr(nhh), swirl=ovr(nhs))
llc = yaml.safe_load(open(f"{L}/{sp}/ll_comparison_{llname}.yaml"))
summ = os.path.join(llc["out_dir"], f'{llc["out_prefix"]}_ll_summary.csv')
d = pd.read_csv(summ) if os.path.exists(summ) else pd.DataFrame()
xd, hj = f"{sfroot}/COP_{sg.upper()}_V2/xai", f"{sfroot}/COP_{sg.upper()}_V2/hierarchy/cluster_hierarchy.json"
X = f"{sfc}/model/copernicus_{sg}_v2_xai.yaml"
def ks(cfg):
    c = yaml.safe_load(open(cfg)); return list(range(int(c["min_latent"]), int(c["max_latent"]) + 1))
def pick(kind, lv, label, cv):
    if lv in over[kind]:
        return int(over[kind][lv]), "override"
    t = d[(d["label"] == label) & (d["split"] == "test")] if len(d) else d
    if len(t) == 0:
        print(f"SKIP {kind} {lv}: no '{label}' rows in {summ} (CV not run)"); return None, None
    miss = [k for k in ks(cv) if k not in set(t.K) or int(t[t.K == k].n.iloc[0]) < exp_n]
    if miss:
        print(f"SKIP {kind} {lv}: CV incomplete for K = {miss} (< {exp_n} runs)"); return None, None
    t = t[t.K.isin(ks(cv))]
    return int(t.loc[t["mean"].idxmax(), "K"]), "best test LL"
def copy_nh(src, nh):
    txt = open(src).read()
    if re.search(r"^n_states:\s*(null|~)?\s*(#.*)?$", txt, re.M):
        return None
    new = re.sub(r"^n_hidden:\s*\S+", f"n_hidden: {nh}", txt, flags=re.M)
    new = re.sub(r"^n_hidden_init:\s*\S+", f"n_hidden_init: {nh}", new, flags=re.M)
    new = re.sub(r'^run_uid:\s*"?([^"\s#]+)"?', lambda m: f'run_uid: "{m.group(1)}_NH{nh}"', new, flags=re.M)
    out = src.replace(".yaml", f"_nh{nh}.yaml")
    open(out, "w").write(f"# generated by scripts/run_levels_kin_v2.sh from {os.path.basename(src)} with n_hidden = {nh}\n" + new)
    return out
hq, sw, interp, need_xai = [], [], [], set()
for lv in levels.split():
    ld = f"{L}/{sp}/{lv}"
    for kind, cv, fin, label, projs_lv, rows in (
            ("hiql", rhcv, rhf, f"HIQL+kin {lv}", [projh], hq),
            ("swirl", rscv, rsf, f"SWIRL {sorder}+kin {lv}", projs.split(), sw)):
        if not os.path.exists(f"{ld}/{fin}.yaml"):
            print(f"SKIP {kind} {lv}: no {ld}/{fin}.yaml (STAGE=gen)"); continue
        nh, why = pick(kind, lv, label, f"{ld}/{cv}.yaml")
        if nh is None: continue
        cfg = copy_nh(f"{ld}/{fin}.yaml", nh)
        if cfg is None:
            print(f"SKIP {kind} {lv}: {fin}.yaml not synced"); continue
        pj = [p for p in (f"{ld}/{x}.yaml" for x in projs_lv) if os.path.exists(p)]
        print(f"FINAL {kind} {lv}: K = {nh} ({why}) -> {cfg}; projections {[os.path.basename(p) for p in pj]}")
        for p in pj:
            rows.append(dict(lv=lv, nh=nh, cfg=cfg, proj=p, pt=os.path.basename(p)[:-5]))
        if not pj:
            rows.append(dict(lv=lv, nh=nh, cfg=cfg, proj="", pt="none"))
        need_xai.add(lv)
for lv in sorted(need_xai):
    if lv == "seascape":
        if not os.path.exists(f"{xd}/interpret_seascape/cluster_profiles_wide.csv"):
            print("  WARNING: interpret_seascape profiles missing (17c, STAGE=seascape_xai): seascape projections will fail")
        continue
    sub = "interpret_parent" if lv == "parent" else f"interpret_{lv}"
    if not os.path.exists(f"{xd}/{sub}/cluster_profiles_wide.csv"):
        args = "--level parent" if lv == "parent" else f"--level coarse --k {lv[1:]} --hierarchy_json {hj}"
        interp.append(dict(lv=lv, x=X, args=args))
if not hq and not sw:
    print("nothing to fit"); open(plan + ".none", "w").close(); sys.exit(0)
env = yaml.safe_load(cpuenv)
gpu_env = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": str(share)}
def uniq(rows):
    return [dict(lv=a, nh=b, cfg=c) for (a, b, c) in dict.fromkeys((r["lv"], r["nh"], r["cfg"]) for r in rows)]
jobs = []
if interp:
    jobs.append(dict(name="level_xai", cmd="{py} -m sit_fuse.xai.cluster_interpret -y {x} --no_reference {args}",
                     cases=interp, id_keys=["lv"], cpus=6, env=env))
xafter = [dict(name="level_xai", on=[])] if interp else []
for kind, rows, fitcmd, plotcmd, gpu in (
        ("hiql", hq, "{py} -m naturenet.models.irl.run_hiavi_kfold -y {cfg} --holdout",
         "{py} -m naturenet.models.irl.plot_hiavi -y {cfg}", 0),
        ("swirl", sw, "{py} -m naturenet.models.irl.run_swirl -y {cfg}",
         "{py} -m naturenet.models.irl.plot_swirl_new -y {cfg}", share)):
    if not rows: continue
    e = env if not gpu else gpu_env
    fits = uniq(rows)
    jobs += [dict(name=f"{kind}_fit", cmd=fitcmd, cases=fits, id_keys=["lv", "nh"], cpus=jcpus, gpu=gpu, env=e),
             dict(name=f"{kind}_plot", cmd=plotcmd, cases=fits, id_keys=["lv", "nh"], cpus=jcpus, gpu=gpu, env=e,
                  after=[dict(name=f"{kind}_fit", on=["lv", "nh"])])]
    pc = [r for r in rows if r["proj"]]
    if pc:
        jobs.append(dict(name=f"{kind}_project",
                         cmd="{py} -m naturenet.models.irl.reward_env_projection -y {cfg} -p {proj} --n_hidden {nh}",
                         cases=pc, id_keys=["lv", "nh", "pt"], cpus=jcpus, env=env,
                         after=[dict(name=f"{kind}_plot", on=["lv", "nh"])] + xafter))
class NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data): return True
gl = [int(g) for g in gpus.strip(",").split(",") if g.strip()]
with open(plan, "w") as f:
    f.write("# generated by scripts/run_levels_kin_v2.sh; edit the script, not this file\n")
    yaml.dump(dict(launcher=dict(state_dir=f"{S}/levels_kin_finals_{sp}", gpus=gl, cpus=int(cpus), retries=0),
                   vars=dict(py=py), jobs=jobs), f, Dumper=NoAlias, sort_keys=False, width=200)
if os.path.exists(plan + ".none"): os.remove(plan + ".none")
print(f"wrote {plan}")
PYEOF
    if [ -f "$plan.none" ]; then rm -f "$plan.none"; else launch "$plan"; fi
  fi

  # ------------------------------------------------------------------- audit
  if want audit && [ "$DRY" != "1" ]; then
    STAGE=audit bash scripts/run_available_outputs_${sp}.sh || true
  fi
done
echo "next: bash scripts/gather_needed_outputs_v2.sh   (levels group: ll_comparison/*_levels_kin_*, <lv>/final/{hiql_kin,swirl_kin})"
