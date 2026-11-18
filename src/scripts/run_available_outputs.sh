#!/usr/bin/env bash
# Every plot, analysis, animation and still that the runs already on disk support (v2 only; finished fits are not
# redone; no step-9 xai). Also runs steps 17b / 17c (Seascapes xai), 20 (shark secondary switch) and 21 (habitat
# tests + whale pre-scales rerun) through the species scripts. Built for partial decision-scales CV: whatever folds exist are plotted, tabled (T2 rows marked "partial")
# and animated; rerunning the script later picks up the folds that finished in the meantime.
#   Species wrappers: run_available_outputs_whales.sh, run_available_outputs_sharks.sh (SP=whales | sharks).
#
#   SP=whales bash run_available_outputs.sh                 # STAGE=all (order below)
#   SP=whales STAGE=<stage> bash run_available_outputs.sh   # one stage:
#     pre        steps 20 / 21 sync + final (species script; finished fits are skipped, then that script's plots /
#                projection / exploit-explore / series / decision context / animations): whales ->
#                run_whale_prescales.sh STAGE=sync, final; sharks -> run_shark_swirl_kin.sh STAGE=sync, final
#     outputs    launcher plan (no fits; every job is gated on its inputs existing when the plan is written):
#                  step 8   plot_hiavi / plot_swirl_new for every final with a fitted npz (kinematic + action-only)
#                  step 11  reward-to-environment projection (needs the step-9 xai/interpret profiles on disk)
#                  step 10  exploit / explore (series + fitted kinematics; kinematic tools)
#                  step 12  time-series correlation; 14b.4 decision-context similarity
#                  step 6b  plot_swirl_scales for every scales run with a cv table (partial folds ok)
#                  step 8b  animations + stills: every job of config/v2/plots/<sp>_animations.yaml (missing inputs
#                           skipped) + auto jobs for every final series and every scales variant on disk
#                Scales-dependent jobs carry the current scales file count in their id, so a later rerun redoes them.
#     habitat    step 11h (new habitat tests): whales -> run_whale_prescales.sh STAGE=habitat,
#                sharks -> run_shark_swirl_kin.sh STAGE=habitat
#     ll         step 5 / 5b LL comparison plots (T1 inputs)
#     ma         multi-agent figures (14, 14b compare, multi-agent scales) for the summaries that exist
#     tables     step 13: T1, T2 (partial rows marked, --expected_folds 5), T3
#     levels     step 16 (no xai): 16a hierarchy verify/build -> 16b level configs -> 16c env per level ->
#                16d HIQL / IAVI sweep (plan_jobs) -> level LL plot; 16e SWIRL if LEVELS_SWIRL is set; 16f HIQL
#                final at LEVEL_FINAL (with NH) -> plot_hiavi -> projection if xai/interpret_<lv> profiles exist
#     seascape   step 17: 17a download / regrid / report -> 17d seascape level (as 16c-16d)
#     seascape_xai  17b agreement of the SIT-FUSE clusters with Seascapes (fine, parent and every coarse K of LEVELS;
#                coarse needs the 16a hierarchy) -> reference_summary; 17c Seascape class profiles (interpret_seascape).
#                Launcher plan, CPU. Needs the 17a rasters and a sit_fuse whose cluster_interpret has --reference_only
#     step20_post  sharks only (step 20): run_shark_swirl_kin.sh STAGE=scales, ma, plots, tables (scales / multi-agent
#                CV of SWIRL S-1 + kin K = 5; same launcher state dirs as run_scales_kfold.sh / run_multiagent_kfold.sh,
#                so folds already claimed by another launcher are not repeated). STEP20_FITS=0 -> plots, tables only
#   Step aliases: STAGE=step20 = pre + step20_post (sharks); STAGE=step21 = pre + habitat; STAGE=step17 = seascape +
#   seascape_xai.
#     audit      manifest audit (5b, 6b, 7-14, incl. 11h) + launcher status
#   STAGE=all = pre, outputs, habitat, ll, ma, tables, levels, seascape, seascape_xai, step20_post, audit. levels /
#   seascape fit the level sweeps and step20_post the shark scales / multi-agent CV (runners skip finished work);
#   SKIP_LEVELS=1 leaves levels / seascape / seascape_xai out of all; SKIP_STEP20=1 leaves step20_post out.
#   Env: SP, GPUS (default 6; scales use 4,5 and multi-agent 0-3 by default), CPUS (default 16 of 48), S (launcher
#        root), NN_SRC, PY, HPY (python for the small yaml / glob helpers; default PY), SFC (SIT-FUSE config dir;
#        default from the installed sit_fuse), LEVELS (default "parent k30 k20 k12"), LEVELS_SWIRL, LEVEL_FINAL, NH,
#        EXPECTED_FOLDS (default 5), DRY=1 (write plans + launcher dry runs only), STEP20_FITS (default 1), MA_STAGE
#        (step 20 multi-agent: screen | full; default screen), REMAKE_XAI=1 (regenerate the SIT-FUSE xai yaml),
#        RERUN_FAILED=1 (retry launcher jobs marked failed by an earlier run; otherwise they and their dependents
#        are skipped, e.g. a level whose env_format failed never gets its class mapper).
set -euo pipefail
SP="${SP:?set SP=whales or SP=sharks (or use run_available_outputs_<sp>.sh)}"
STAGE="${STAGE:-all}"
NN_SRC="${NN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPTS="$NN_SRC/scripts"
S="${S:-/data/nlahaye/NatureNet/launcher}"
PY="${PY:-python}"
HPY="${HPY:-$PY}"
GPUS="${GPUS:-6}"
CPUS="${CPUS:-16}"
case "$GPUS" in *,*|auto|none) ;; *) GPUS="$GPUS," ;; esac   # one id needs a trailing comma (a bare N = ids 0..N-1)
GPUS_YAML="[${GPUS%,}]"; case "$GPUS" in auto|none) GPUS_YAML="$GPUS" ;; esac
EXPECTED_FOLDS="${EXPECTED_FOLDS:-5}"
LEVELS="${LEVELS:-parent k30 k20 k12}"
RERUN=""; [ "${RERUN_FAILED:-0}" = "1" ] && RERUN="--rerun-failed"   # retry jobs a previous launcher run marked failed
cd "$NN_SRC"
# PROJ_DATA / PROJ_LIB without proj.db (pixi / conda activation) breaks rasterio EPSG lookups; point it at a valid
# proj.db for every stage and launcher job (inherited env). No-op when the current value is fine.
eval "$("$PY" -m naturenet.environment.proj_env --export 2>/dev/null | grep "^export PROJ_" || true)"
C=naturenet/config/v2/swirl
T=naturenet/config/v2/time_series
D=naturenet/config/v2/drivers
E=naturenet/config/v2/environment
P=naturenet/config/v2/plots
L=naturenet/config/v2/levels
N="$PY -m naturenet.models"
mkdir -p "$S"

# ----------------------------------------------------------------------------- species tables
# PLOTS:  final config | plotter (hiavi | swirl)
# VIEWS:  label | final config | K | series tag | projection yaml (- = none) | fitted npz for ee (- = no ee)
# SCALES: label | scales config      MA: name | config | K list | subdir (- = default)    MAS: scales config | K list
case "$SP" in
  whales)
    R=/data/nlahaye/NatureNet/Blue_Whale_v2; SG=whale; RUID=SWIRL_WHALE_1517; LLP=whale_v2_kin_nn
    PLOTS=(whales_hiql_final_kin_k5:hiavi whales_swirl_s2_final_kin_k5:swirl
           whales_hiql_final:hiavi whales_iavi_final:hiavi whales_swirl_final:swirl)
    VIEWS=(
      "whales_hiql_kin_k5:whales_hiql_final_kin_k5:5:hiavi_s1:whales_reward_env_projection_hiql:WHALE_HIQL_KIN_FINAL_5_12345_hiavi.npz"
      "whales_swirl_kin_s2_k5:whales_swirl_s2_final_kin_k5:5:linear_s2:whales_reward_env_projection_s2_kin:WHALE_SWIRL_KIN_FINAL_5_12345_S2_naturenet_iter2.npz"
      "whales_swirl_kin_s1_k5:whales_swirl_s2_final_kin_k5:5:linear_s1:whales_reward_env_projection_s1_kin:WHALE_SWIRL_KIN_FINAL_5_12345_S1_naturenet_iter2_temp_2.npz"
      "whales_hiql_k2:whales_hiql_final:2:hiavi_s1:whales_reward_env_projection_hiql:-"
      "whales_swirl_s1_k5:whales_swirl_final:5:linear_s1:whales_reward_env_projection:-"
      "whales_swirl_s2_k5:whales_swirl_final:5:linear_s2:whales_reward_env_projection_s2:-")
    SCALES=(hiql_kin_k5:whales_hiql_scales_kin_k5 swirl_kin_k5:whales_swirl_scales_kin
            hiql_k2:whales_hiql_scales swirl_k5:whales_swirl_scales)
    T2=(hiql_kin_k5:WHALE_HIQL_SCALES_KIN_K5:scales_kin swirl_kin_k5:WHALE_SWIRL_SCALES_KIN:scales_kin)
    MA=("multiagent_kin:whales_multiagent_kin:5:-" "multiagent_swirl_kin:whales_multiagent_swirl_kin:5:multiagent_swirl_kin"
        "multiagent_kin_coupling:whales_multiagent_kin_coupling:5:multiagent_kin_coupling"
        "multiagent_swirl_kin_coupling:whales_multiagent_swirl_kin_coupling:5:multiagent_swirl_kin_coupling")
    MAS=("whales_scales_multiagent_kin:5" "whales_scales_multiagent_swirl_kin:5")
    T3=("multiagent_kin:HIQL+kin:5")
    HABITAT_SCRIPT=run_whale_prescales.sh; PRE_STAGES="sync final"; STEP20_POST=""
    ;;
  sharks)
    R=/data/nlahaye/NatureNet/Hammerhead_Out_v2; SG=shark; RUID=hammerhead_v2; LLP=hammerhead_v2_kin_nn
    PLOTS=(sharks_hiql_final_kin_k5:hiavi sharks_hiql_final_kin_k2:hiavi sharks_swirl_final_kin_k5:swirl
           sharks_hiql_final:hiavi sharks_iavi_final:hiavi sharks_swirl_final:swirl)
    VIEWS=(
      "sharks_hiql_kin_k5:sharks_hiql_final_kin_k5:5:hiavi_s1:sharks_reward_env_projection_hiql:SHARK_HIQL_KIN_FINAL_5_12345_hiavi.npz"
      "sharks_swirl_kin_s1_k5:sharks_swirl_final_kin_k5:5:linear_s1:sharks_reward_env_projection_s1_kin:SHARK_SWIRL_KIN_FINAL_5_12345_S1_naturenet_iter2_temp_2.npz"
      "sharks_swirl_kin_s2_k5:sharks_swirl_final_kin_k5:5:linear_s2:sharks_reward_env_projection_s2_kin:SHARK_SWIRL_KIN_FINAL_5_12345_S2_naturenet_iter2.npz"
      "sharks_hiql_kin_k2:sharks_hiql_final_kin_k2:2:hiavi_s1:sharks_reward_env_projection_hiql:SHARK_HIQL_KIN_K2_FINAL_2_12345_hiavi.npz"
      "sharks_hiql_k2:sharks_hiql_final:2:hiavi_s1:sharks_reward_env_projection_hiql:-"
      "sharks_swirl_s1_k3:sharks_swirl_final:3:linear_s1:sharks_reward_env_projection:-")
    SCALES=(hiql_kin_k5:sharks_hiql_scales_kin_k5 hiql_kin_k2:sharks_hiql_scales_kin_k2
            swirl_kin_k5:sharks_swirl_scales_kin hiql_k2:sharks_hiql_scales swirl_k3:sharks_swirl_scales)
    T2=(hiql_kin_k5:HIQL_SCALES_KIN_K5:scales_kin swirl_kin_k5:SHARK_SWIRL_SCALES_KIN:scales_kin
        hiql_kin_k2:HIQL_SCALES_KIN_K2:scales_kin)
    MA=("multiagent_kin:sharks_multiagent_kin:5 2:-" "multiagent_swirl_kin:sharks_multiagent_swirl_kin:5:multiagent_swirl_kin"
        "multiagent_kin_coupling:sharks_multiagent_kin_coupling:5 2:multiagent_kin_coupling"
        "multiagent_swirl_kin_coupling:sharks_multiagent_swirl_kin_coupling:5:multiagent_swirl_kin_coupling")
    MAS=("sharks_scales_multiagent_kin:5 2" "sharks_scales_multiagent_swirl_kin:5")
    T3=("multiagent_kin:HIQL+kin:5 2" "multiagent_swirl_kin:SWIRL+kin:5")
    HABITAT_SCRIPT=run_shark_swirl_kin.sh; PRE_STAGES="sync final"; STEP20_POST="scales ma plots tables"
    ;;
  *) echo "SP must be whales or sharks" >&2; exit 1 ;;
esac
SGU=$(echo "$SG" | tr '[:lower:]' '[:upper:]')
SF_OUT=/data/nlahaye/output/Learnergy/COP_${SGU}_V2

declare -A ALIAS=([step20]="pre step20_post" [step21]="pre habitat" [step17]="seascape seascape_xai")
want() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ] || [[ " ${ALIAS[$STAGE]:-} " == *" $1 "* ]]; }
explicit() { [ "$STAGE" = "$1" ] || [[ " ${ALIAS[$STAGE]:-} " == *" $1 "* ]]; }   # asked for by name / alias
sub() {      # species script stage (steps 20 / 21) with the same GPUs / CPUs / launcher root / python
  env STAGE="$1" GPUS="$GPUS" CPUS="$CPUS" S="$S" PY="$PY" NN_SRC="$NN_SRC" DRY="${DRY:-0}" MA_STAGE="${MA_STAGE:-screen}" \
    bash "$SCRIPTS/$HABITAT_SCRIPT" || echo "WARNING: $HABITAT_SCRIPT STAGE=$1 failed (see above)"
}
sfc_dir() { echo "${SFC:-$("$HPY" -c 'import os, sit_fuse; print(os.path.join(os.path.dirname(sit_fuse.__file__), "config"))' 2>/dev/null || true)}"; }
cpu() { env JAX_PLATFORMS=cpu NATURENET_JAX_PLATFORM=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 "$@"; }
launch() {   # plan [extra launcher args]: dry run summary, then run unless DRY=1
  local plan="$1"; shift
  "$PY" -m sit_fuse.local_launcher run "$plan" --dry-run --gpus "$GPUS" --cpus "$CPUS" "$@" \
    | awk '/WARNING/ && w++ < 5; /^\[/ {n++} END {print n " jobs in the launcher dry run"}'
  [ "${DRY:-0}" = "1" ] && return 0
  "$PY" -m sit_fuse.local_launcher run "$plan" --gpus "$GPUS" --cpus "$CPUS" $RERUN "$@" \
    || echo "WARNING: $(basename "$plan"): some jobs failed or were blocked (status: python -m sit_fuse.local_launcher status $plan)"
}
ykey() {     # yaml dotted.key -> value ("" if missing)
  "$HPY" -c 'import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
for k in sys.argv[2].split("."):
    c = c.get(k) if isinstance(c, dict) else None
print("" if c is None else c)' "$1" "$2"
}
anyfile() { compgen -G "$1" > /dev/null; }
nfiles() { local n=0 f; for f in $1; do [ -e "$f" ] && n=$((n + 1)); done; echo $n; }
GPU_ENV="{XLA_PYTHON_CLIENT_PREALLOCATE: \"false\"}"
CPU_ENV="{JAX_PLATFORMS: cpu, NATURENET_JAX_PLATFORM: cpu}"
# scales stamp: number of out-of-fold test files + multi-agent scale outputs (changes as the CV progresses)
SC_N=$(( $(nfiles "$R/scales_kin/*_test_KF_*.npz") + $(nfiles "$R/scales/*_test_KF_*.npz") \
       + $(nfiles "$R/multiagent_scales_kin/*/*.npz") ))
echo "== $SP: root $R, scales stamp $SC_N"

# ----------------------------------------------------------------------------- outputs (launcher plan)
PO="$S/available_outputs_${SP}.yaml"
AUTO="$R/animations/auto_animations.yaml"
write_outputs_plan() {
  local e cfg plotter dir uid lab K rep proj npz prof lines
  local ph=() psw=() prh=() prs=() ees=() een=() scl=() ab=() aa=()
  declare -A plotted=()
  for e in "${PLOTS[@]}"; do
    cfg=${e%%:*}; plotter=${e#*:}
    dir=$(ykey "$C/$cfg.yaml" out_dir); uid=$(ykey "$C/$cfg.yaml" run_uid)
    if ! anyfile "${dir%/}/${uid}_*.npz"; then echo "SKIP plots $cfg (no fitted npz in $dir)"; continue; fi
    plotted[$cfg]=1
    [ "$plotter" = hiavi ] && ph+=("      - {cfg: $cfg}") || psw+=("      - {cfg: $cfg}")
  done
  for e in "${VIEWS[@]}"; do
    IFS=: read -r lab cfg K rep proj npz <<< "$e"
    [ -n "${plotted[$cfg]:-}" ] || continue
    dir=$(ykey "$C/$cfg.yaml" out_dir); dir=${dir%/}
    if [ "$proj" != "-" ]; then
      prof=$(ykey "$C/$proj.yaml" reward_env_projection.profiles_csv)
      if [ -n "$prof" ] && [ ! -f "$prof" ]; then echo "SKIP projection $lab (no xai profiles $prof; xai not rerun)"
      elif [[ $rep == hiavi* ]]; then prh+=("      - {label: $lab, cfg: $cfg, proj: $proj, k: $K}")
      else prs+=("      - {label: $lab, cfg: $cfg, proj: $proj, k: $K}"); fi
    fi
    if [ "$npz" != "-" ]; then
      ees+=("      - {label: $lab, dir: \"$dir\", rep: $rep}")
      [ -f "$dir/$npz" ] && een+=("      - {label: $lab, npz: \"$dir/$npz\"}") || echo "SKIP ee_npz $lab (no $dir/$npz)"
    fi
  done
  for e in "${SCALES[@]}"; do
    lab=${e%%:*}; cfg=${e#*:}
    dir=$(ykey "$C/$cfg.yaml" out_dir); uid=$(ykey "$C/$cfg.yaml" run_uid)
    if [ -f "${dir%/}/${uid}_scales_cv.csv" ]; then
      scl+=("      - {label: $lab, cfg: $cfg, nf: \"$SC_N\"}")
    else echo "SKIP scales plots $lab (no ${dir%/}/${uid}_scales_cv.csv yet)"; fi
  done
  # animations: base yaml jobs (scales / multi-agent jobs stamped) + generated jobs
  for j in $("$HPY" -c 'import sys, yaml; print(" ".join(a["name"] for a in (yaml.safe_load(open(sys.argv[1])) or {}).get("animations") or []))' "$P/${SP}_animations.yaml"); do
    case "$j" in scales_*|ma_*) ab+=("      - {job: $j, nf: \"$SC_N\"}") ;; *) ab+=("      - {job: $j, nf: \"0\"}") ;; esac
  done
  local fin=() sc=()
  for e in "${PLOTS[@]}"; do cfg=${e%%:*}; [ -n "${plotted[$cfg]:-}" ] || continue
    fin+=("${cfg#${SP}_}=$(ykey "$C/$cfg.yaml" out_dir)"); done
  for e in "${SCALES[@]}"; do sc+=("${e%%:*}=$C/${e#*:}.yaml"); done
  if [ "${DRY:-0}" = "1" ]; then AUTO="$S/available_outputs_${SP}_auto_animations.yaml"; fi
  "$PY" -m naturenet.models.irl.auto_animation_jobs -b "$P/${SP}_animations.yaml" --out "$AUTO" \
      ${fin:+--finals "${fin[@]}"} --scales "${sc[@]}" || echo "WARNING: auto_animation_jobs failed (auto jobs skipped)"
  if [ -f "$AUTO" ]; then
    for j in $("$HPY" -c 'import sys, yaml; print(" ".join(a["name"] for a in (yaml.safe_load(open(sys.argv[1])) or {}).get("animations") or []))' "$AUTO"); do
      case "$j" in auto_scales_*) aa+=("      - {job: $j, nf: \"$SC_N\"}") ;; *) aa+=("      - {job: $j, nf: \"0\"}") ;; esac
    done
  fi

  {
    echo "# generated by scripts/run_available_outputs.sh (SP=$SP, STAGE=outputs); edit the script, not this file"
    echo "launcher: {state_dir: \"$S/available_outputs_${SP}\", gpus: $GPUS_YAML, cpus: $CPUS, retries: 0}"
    echo "vars: {py: \"$PY\", c: \"$C\", t: \"$T\", ee: \"$R/final/exploit_explore\"}"
    echo "jobs:"
    if [ ${#ph[@]} -gt 0 ]; then cat <<EOF
  - name: plots_hiavi               # step 8: maps, tracks, series_hiavi_s1/, state_reward_hiavi_s1.npy
    cmd: "{py} -m naturenet.models.irl.plot_hiavi -y {c}/{cfg}.yaml"
    cases:
$(printf '%s\n' "${ph[@]}")
    cpus: 4
    env: $CPU_ENV
EOF
    fi
    if [ ${#psw[@]} -gt 0 ]; then cat <<EOF
  - name: plots_swirl               # step 8: series_linear_s1 / s2, state_reward_linear_s{1,2}.npy
    cmd: "{py} -m naturenet.models.irl.plot_swirl_new -y {c}/{cfg}.yaml"
    cases:
$(printf '%s\n' "${psw[@]}")
    gpu: 0.25
    cpus: 4
    env: $GPU_ENV
EOF
    fi
    if [ ${#prh[@]} -gt 0 ]; then cat <<EOF
  - name: project_hiavi             # step 11: reward-to-environment projection + use vs availability
    cmd: "{py} -m naturenet.models.irl.reward_env_projection -y {c}/{cfg}.yaml -p {c}/{proj}.yaml --n_hidden {k}"
    cases:
$(printf '%s\n' "${prh[@]}")
    id_keys: [label]
    cpus: 2
    env: $CPU_ENV
    after: [{name: plots_hiavi, on: [cfg]}]
EOF
    fi
    if [ ${#prs[@]} -gt 0 ]; then cat <<EOF
  - name: project_swirl
    cmd: "{py} -m naturenet.models.irl.reward_env_projection -y {c}/{cfg}.yaml -p {c}/{proj}.yaml --n_hidden {k}"
    cases:
$(printf '%s\n' "${prs[@]}")
    id_keys: [label]
    gpu: 0.25
    cpus: 2
    env: $GPU_ENV
    after: [{name: plots_swirl, on: [cfg]}]
EOF
    fi
    if [ ${#ees[@]} -gt 0 ]; then cat <<EOF
  - name: ee_series                 # step 10: exploit / explore from the exported series (+ state reward)
    cmd: "{py} -m naturenet.models.irl.exploit_explore_init --data_dir {dir}/plots/series_{rep} --slices {t}/${SP}_correlation_final_kin_v2.yaml --state_reward {dir}/plots/state_reward_{rep}.npy --out_dir {ee}/{label}"
    cases:
$(printf '%s\n' "${ees[@]}")
    id_keys: [label]
    cpus: 2
    env: $CPU_ENV
    after: [{name: plots_hiavi, on: []}, {name: plots_swirl, on: []}]
EOF
    fi
    if [ ${#een[@]} -gt 0 ]; then cat <<EOF
  - name: ee_npz                    # step 10: from the fitted kinematic parameters (same out_dir: after ee_series)
    cmd: "{py} -m naturenet.models.irl.exploit_explore_init --model_npz {npz} --out_dir {ee}/{label}"
    cases:
$(printf '%s\n' "${een[@]}")
    id_keys: [label]
    cpus: 1
    env: $CPU_ENV
    after: [{name: ee_series, on: [label]}]
EOF
    fi
    if [ ${#ees[@]} -gt 0 ]; then cat <<EOF
  - name: series                    # step 12: time-series correlation, all slices
    cmd: "{py} -m naturenet.models.time_series_analysis.correlation_analysis_shark -y {t}/${SP}_correlation_final_kin_v2.yaml"
    cpus: 4
    env: $CPU_ENV
    after: [{name: plots_hiavi, on: []}, {name: plots_swirl, on: []}]
  - name: decision_context          # step 14b.4: decision-context similarity between the kinematic tools
    cmd: "{py} -m naturenet.drivers.decision_context_similarity -y $D/${SP}_decision_context.yaml"
    cpus: 4
    env: $CPU_ENV
    after: [{name: ee_series, on: []}, {name: ee_npz, on: []}]
EOF
    fi
    if [ ${#scl[@]} -gt 0 ]; then cat <<EOF
  - name: scales_plots              # step 6b: scales figures + <run_uid>_scales_summary.csv from the folds on disk
    cmd: "{py} -m naturenet.models.irl.plot_swirl_scales -y {c}/{cfg}.yaml"
    cases:
$(printf '%s\n' "${scl[@]}")
    id_keys: [label, nf]
    gpu: 0.25
    cpus: 2
    env: $GPU_ENV
EOF
    fi
    if [ ${#ab[@]} -gt 0 ]; then cat <<EOF
  - name: animate                   # step 8b: $P/${SP}_animations.yaml (jobs without inputs are skipped)
    cmd: "{py} -m naturenet.models.irl.animate_tracks -y $P/${SP}_animations.yaml --only {job}"
    cases:
$(printf '%s\n' "${ab[@]}")
    id_keys: [job, nf]
    cpus: 2
    env: $CPU_ENV
    after: [{name: plots_hiavi, on: []}, {name: plots_swirl, on: []}]
EOF
    fi
    if [ ${#aa[@]} -gt 0 ]; then cat <<EOF
  - name: animate_auto              # step 8b: generated jobs (every final series, every scales variant on disk)
    cmd: "{py} -m naturenet.models.irl.animate_tracks -y $AUTO --only {job}"
    cases:
$(printf '%s\n' "${aa[@]}")
    id_keys: [job, nf]
    cpus: 2
    env: $CPU_ENV
    after: [{name: plots_hiavi, on: []}, {name: plots_swirl, on: []}]
EOF
    fi
  } > "$PO"
  # drop after-references to jobs that were not written (nothing to plot for that plotter)
  "$HPY" - "$PO" <<'PYEOF'
import sys, yaml
p = sys.argv[1]
txt = open(p).read()
plan = yaml.safe_load(txt) or {}
names = {j["name"] for j in plan.get("jobs") or []}
for j in plan.get("jobs") or []:
    if "after" in j:
        j["after"] = [a for a in j["after"] if (a["name"] if isinstance(a, dict) else a) in names]
        for a in j["after"]:            # unquoted `on:` loads as True (YAML 1.1); write it back as the string key
            if isinstance(a, dict) and True in a:
                a["on"] = a.pop(True)
        if not j["after"]:
            del j["after"]
with open(p, "w") as f:
    f.write(txt.splitlines()[0] + "\n")
    yaml.safe_dump(plan, f, sort_keys=False, width=200)
PYEOF
  echo "wrote $PO"
}
# ----------------------------------------------------------------------------- pre (steps 20 / 21: sync + final)
if want pre; then
  echo "== pre: $HABITAT_SCRIPT STAGE=$PRE_STAGES (step $([ "$SP" = sharks ] && echo 20 || echo 21) finals; finished fits skipped)"
  for st in $PRE_STAGES; do sub "$st"; done
fi

if want outputs; then write_outputs_plan; launch "$PO"; fi

# ----------------------------------------------------------------------------- habitat (step 11h)
if want habitat; then
  sub habitat
fi

[ "${DRY:-0}" = "1" ] && [ "$STAGE" = "all" ] && echo "DRY=1: the cheap CPU stages (ll, ma, tables) are only listed"

# ----------------------------------------------------------------------------- ll (steps 5, 5b)
if want ll; then
  for y in "$C/${SP}_ll_comparison.yaml" "$C/${SP}_ll_comparison_kin_nn.yaml"; do
    if [ "${DRY:-0}" = "1" ]; then echo "would run plot_ll_comparison -y $y"; continue; fi
    cpu $N.irl.plot_ll_comparison -y "$y" || echo "SKIP $y (failed)"
  done
fi

# ----------------------------------------------------------------------------- ma (steps 14, 14b, multi-agent scales)
if want ma; then
  MD=$R/multiagent
  for e in "${MA[@]}"; do
    IFS=: read -r name cfg ks sub <<< "$e"
    [ -f "$MD/$name/ma_summary.json" ] || { echo "SKIP multi-agent plots $name (no $MD/$name/ma_summary.json)"; continue; }
    sargs=(); [ "$sub" != "-" ] && sargs=(--subdir "$sub")
    if [ "${DRY:-0}" = "1" ]; then echo "would run plot_hiavi_multiagent -y $C/$cfg.yaml --K $ks ${sargs[*]}"; continue; fi
    # shellcheck disable=SC2086
    cpu $N.irl.plot_hiavi_multiagent -y "$C/$cfg.yaml" --K $ks "${sargs[@]}" || echo "WARNING: $name plots failed"
  done
  if [ "${DRY:-0}" != "1" ]; then
    for b in kin swirl_kin; do
      [ -f "$MD/multiagent_$b/ma_summary.json" ] && [ -f "$MD/multiagent_${b}_coupling/ma_summary.json" ] && \
        cpu $N.irl.plot_multiagent_compare --dirs "$MD/multiagent_$b" "$MD/multiagent_${b}_coupling" \
          --labels initial coupling --K 5 --out "$MD/multiagent_${b}_coupling/compare_initial" --B || true
    done
    [ -f "$MD/multiagent_kin/ma_summary.json" ] && [ -f "$MD/multiagent_swirl_kin/ma_summary.json" ] && \
      cpu $N.irl.plot_multiagent_compare --dirs "$MD/multiagent_kin" "$MD/multiagent_swirl_kin" \
        --labels "HIQL+kin" "SWIRL+kin" --K 5 --out "$MD/ma_compare_hiql_swirl_kin" || true
  fi
  if [ -f "$R/multiagent_scales_kin/ma_scales_summary.csv" ]; then
    for e in "${MAS[@]}"; do
      cfg=${e%%:*}; ks=${e#*:}
      for k in $ks; do
        if [ "${DRY:-0}" = "1" ]; then echo "would run plot_scales_multiagent -y $C/$cfg.yaml --K $k"; continue; fi
        cpu $N.irl.plot_scales_multiagent -y "$C/$cfg.yaml" --K "$k" || echo "SKIP $cfg K=$k (no rows yet)"
      done
    done
  else echo "SKIP multi-agent scales plots (no $R/multiagent_scales_kin/ma_scales_summary.csv)"; fi
fi

# ----------------------------------------------------------------------------- tables (step 13)
if want tables; then
  M=$R/manuscript_kin
  args=(--out "$M/ms_${SP}_kin" --expected_folds "$EXPECTED_FOLDS")
  LLS=$R/ll_comparison_kin_nn/${LLP}_ll_summary.csv
  [ -f "$LLS" ] && args+=(--ll_summary "$LLS") || echo "SKIP T1 (missing $LLS)"
  sc=()
  for e in "${T2[@]}"; do
    IFS=: read -r lab uid sub <<< "$e"
    f="$R/$sub/${uid}_scales_summary.csv"
    [ -f "$f" ] && sc+=("$lab=$f") || echo "SKIP T2 row $lab (no $f; run STAGE=outputs once the first folds exist)"
  done
  [ ${#sc[@]} -gt 0 ] && args+=(--scales_summary "${sc[@]}")
  mc=()
  for e in "${T3[@]}"; do
    IFS=: read -r name label ks <<< "$e"
    [ -f "$R/multiagent/$name/ma_summary.json" ] || continue
    if [ "${DRY:-0}" = "1" ]; then mc+=("$M/ma_compare_$name.csv"); continue; fi
    mkdir -p "$M"
    # shellcheck disable=SC2086
    cpu $N.irl.plot_multiagent_compare --dirs "$R/multiagent/$name" --labels "$label" --K $ks \
      --out "$M/ma_compare_$name" && mc+=("$M/ma_compare_$name.csv")
  done
  [ ${#mc[@]} -gt 0 ] && args+=(--ma_compare "${mc[@]}") || echo "SKIP T3 coupling (no multi-agent summaries)"
  [ -f "$R/multiagent_scales_kin/ma_scales_summary.csv" ] && args+=(--ma_scales "$R/multiagent_scales_kin/ma_scales_summary.csv") \
    || echo "SKIP T3 scales (no ma_scales_summary.csv)"
  if [ "${DRY:-0}" = "1" ]; then echo "would run summarize_experiments ${args[*]}"
  else mkdir -p "$M"; cpu $N.irl.summarize_experiments "${args[@]}"; fi
fi

# ----------------------------------------------------------------------------- levels (step 16) / seascape (17)
# one plan per species and level set: per level link -> env_format -> sync -> abstract_env, then the HIQL / IAVI sweep
level_mapper() { ykey "$L/$SP/$1/sync.yaml" class_mapper 2>/dev/null || true; }
level_env_plan() {   # plan state_name levels...
  local plan="$1" st="$2"; shift 2
  local cases=() lv
  local m j
  for lv in "$@"; do
    m=$(level_mapper "$lv")
    if [ -n "$m" ] && [ ! -f "$m" ] && [ "${DRY:-0}" != "1" ]; then   # done-marked but no mapper: stale, redo
      for j in env_format sync abstract_env; do rm -f "$S/$st/markers/$j.lv$lv.done"; done
    fi
    if [ -f "$L/$SP/$lv/link.yaml" ]; then cases+=("      - {lv: $lv}")
    else echo "SKIP level $lv (no $L/$SP/$lv/link.yaml; rerun make_level_configs after editing the spec)"; fi
  done
  [ ${#cases[@]} -gt 0 ] || return 1
  local cs; cs=$(printf '%s\n' "${cases[@]}")
  cat > "$plan" <<EOF
# generated by scripts/run_available_outputs.sh (SP=$SP, $st); edit the script, not this file
launcher: {state_dir: "$S/$st", gpus: $GPUS_YAML, cpus: $CPUS, retries: 0}
vars: {py: "$PY", l: "$L/$SP"}
jobs:
  - name: link                      # symlinks the fine movement + raw-input files; copies nothing else
    cmd: "{py} -m naturenet.environment.link_movement_run -y {l}/{lv}/link.yaml"
    cases:
$cs
    cpus: 1
    env: $CPU_ENV
  - name: env_format                # level labels along the tracks (+ final_env_maps__level.json)
    cmd: "{py} -m naturenet.models.feature_rep.env_data_format -y {l}/{lv}/env_format.yaml"
    cases:
$cs
    cpus: 4
    env: $CPU_ENV
    after: [{name: link, on: [lv]}]
  - name: sync                      # n_states = K_level + 1 in this level's configs only
    cmd: "{py} -m naturenet.environment.sync_v2_configs patch -y {l}/{lv}/sync.yaml"
    cases:
$cs
    cpus: 1
    env: $CPU_ENV
    after: [{name: env_format, on: [lv]}]
  - name: abstract_env              # abstract-state trajectories and transition model for the level
    cmd: "{py} -m naturenet.environment.movement_preprocess_pipeline -y {l}/{lv}/abstract_env.yaml"
    cases:
$cs
    cpus: 4
    env: $CPU_ENV
    after: [{name: sync, on: [lv]}]
EOF
  echo "wrote $plan"
}
level_sweep() {      # tag levels...: HIQL / IAVI sweep (plan_jobs; runners skip finished folds), then the LL plot
  local tag="$1"; shift
  local a=() lv
  for lv in "$@"; do
    [ -f "$L/$SP/$lv/hiavi_kfold.yaml" ] && a+=(--hiavi "$L/$SP/$lv/hiavi_kfold.yaml")
    [ -f "$L/$SP/$lv/iavi_kfold.yaml" ]  && a+=(--iavi  "$L/$SP/$lv/iavi_kfold.yaml")
    if [ -n "${LEVELS_SWIRL:-}" ] && [[ " $LEVELS_SWIRL " == *" $lv "* ]] && [ -f "$L/$SP/$lv/swirl_kfold.yaml" ]; then
      a+=(--swirl "$L/$SP/$lv/swirl_kfold.yaml")                                  # 16e
    fi
  done
  [ ${#a[@]} -gt 0 ] || { echo "SKIP $tag sweep (no level CV configs)"; return 0; }
  local plan="$S/available_${SP}_${tag}_sweep.yaml"
  "$PY" -m naturenet.models.irl.plan_jobs --out "$plan" --state_dir "$S/levels_${SP}" "${a[@]}" --gpus "$GPUS" --cpus "$CPUS"
  launch "$plan"
  if [ "${DRY:-0}" = "1" ]; then echo "would run plot_ll_comparison -y $L/$SP/ll_comparison_levels.yaml"
  else cpu $N.irl.plot_ll_comparison -y "$L/$SP/ll_comparison_levels.yaml" || echo "SKIP level LL plot"; fi
}
level_synced() {     # every generated run config of the level has n_states filled
  local f n
  for f in "$L/$SP/$1"/*.yaml; do
    case "$(basename "$f")" in link.yaml|env_format.yaml|abstract_env.yaml|sync.yaml|null.yaml) continue ;; esac
    grep -q '^n_states:' "$f" || continue
    n=$(ykey "$f" n_states); [ -n "$n" ] || return 1
  done
}
sync_levels() {      # state_name levels...: sync every level whose class mapper exists. make_level_configs rewrites
  # n_states: null on every run and the launcher's sync job stays done-marked, so this always runs (cheap, idempotent)
  local st="$1" lv m f; shift
  for lv in "$@"; do
    [ -f "$L/$SP/$lv/sync.yaml" ] || continue
    m=$(level_mapper "$lv")
    if [ -n "$m" ] && [ -f "$m" ]; then
      if [ "${DRY:-0}" = "1" ]; then echo "would sync $lv"
      else "$PY" -m naturenet.environment.sync_v2_configs patch -y "$L/$SP/$lv/sync.yaml" > /dev/null \
             && echo "synced $lv" || echo "WARNING: sync failed for $lv"; fi
    else
      echo "NOT BUILT $lv: no $m (env_format has not finished for this level)"
      for f in "$S/$st/markers/link.lv$lv.failed" "$S/$st/markers/env_format.lv$lv.failed"; do
        [ -e "$f" ] || continue
        echo "     $(basename "${f%%.*}") failed earlier: $S/$st/logs/$(basename "${f%.failed}").log"
        echo "     fix the cause, then rerun with RERUN_FAILED=1 (failed jobs are otherwise skipped)"
      done
    fi
  done
}
level_ready() {      # class mapper exists and the level's run configs are synced: only then the sweep is planned
  local m; m=$(level_mapper "$1")
  { [ -z "$m" ] || [ -f "$m" ]; } && level_synced "$1"
}

if want levels && [ "${SKIP_LEVELS:-0}" != "1" -o "$STAGE" = "levels" ]; then
  echo "== step 16: environment levels ($LEVELS)"
  SFC="$(sfc_dir)"
  HY="$SFC/hierarchy/${SG}_hierarchy_v2.yaml"
  if [ -f "$SF_OUT/hierarchy/cluster_hierarchy.json" ]; then echo "16a: hierarchy exists ($SF_OUT/hierarchy)"
  elif [ -f "$HY" ]; then
    if [ "${DRY:-0}" = "1" ]; then echo "would run cluster_hierarchy verify + build -y $HY"
    else
      cpu "$PY" -m sit_fuse.postprocessing.cluster_hierarchy verify -y "$HY"
      cpu "$PY" -m sit_fuse.postprocessing.cluster_hierarchy build -y "$HY"
    fi
  else echo "SKIP 16a (no $HY; set SFC to the SIT-FUSE config dir)"; fi
  if [ "${DRY:-0}" = "1" ]; then "$PY" -m naturenet.environment.make_level_configs -y "$L/${SP}_levels_v2.yaml" --dry_run
  else "$PY" -m naturenet.environment.make_level_configs -y "$L/${SP}_levels_v2.yaml"; fi       # 16b
  # shellcheck disable=SC2086
  sync_levels "levels_env_${SP}" $LEVELS > /dev/null      # restore n_states of built levels (16b reset them)
  # shellcheck disable=SC2086
  if level_env_plan "$S/available_${SP}_levels_env.yaml" "levels_env_${SP}" $LEVELS; then
    launch "$S/available_${SP}_levels_env.yaml"                                                    # 16c
  fi
  # shellcheck disable=SC2086
  sync_levels "levels_env_${SP}" $LEVELS                  # levels built just now + report the ones that are not
  ready=(); for lv in $LEVELS; do level_ready "$lv" && ready+=("$lv") || echo "SKIP sweep $lv (level env not built or not synced)"; done
  [ ${#ready[@]} -gt 0 ] && level_sweep levels "${ready[@]}"                                       # 16d / 16e
  if [ -n "${LEVEL_FINAL:-}" ]; then                                                               # 16f (no xai)
    lv=$LEVEL_FINAL; F="$L/$SP/$lv/hiql_final.yaml"; PJ="$L/$SP/$lv/reward_env_projection_hiql.yaml"
    NH="${NH:?set NH (HIQL K chosen in 16d) with LEVEL_FINAL}"
    [ "$(ykey "$F" n_hidden)" = "$NH" ] || echo "WARNING: $F n_hidden != NH=$NH (edit n_hidden / n_hidden_init first)"
    if [ "${DRY:-0}" = "1" ]; then echo "would run run_hiavi_kfold --holdout, plot_hiavi, projection for $lv (NH=$NH)"
    else
      cpu $N.irl.run_hiavi_kfold -y "$F" --holdout
      cpu $N.irl.plot_hiavi -y "$F"
      prof=$(ykey "$PJ" reward_env_projection.profiles_csv)
      if [ -z "$prof" ] || [ -f "$prof" ]; then
        cpu $N.irl.reward_env_projection -y "$F" -p "$PJ" --n_hidden "$NH"
      else echo "SKIP 16f projection (no $prof: needs xai/interpret for level $lv, not rerun here)"; fi
    fi
  fi
fi

if want seascape && [ "${SKIP_LEVELS:-0}" != "1" -o "$STAGE" = "seascape" ]; then
  echo "== step 17: Seascapes (17a, 17d; 17b / 17c: STAGE=seascape_xai)"
  if [ "${DRY:-0}" = "1" ]; then echo "would run naturenet.environment.seascapes all -y $E/${SP}_seascapes_v2.yaml"
  else cpu "$PY" -m naturenet.environment.seascapes all -y "$E/${SP}_seascapes_v2.yaml"; fi       # 17a (resumable)
  if level_env_plan "$S/available_${SP}_seascape_env.yaml" "seascape_env_${SP}" seascape; then
    launch "$S/available_${SP}_seascape_env.yaml"
  fi
  sync_levels "seascape_env_${SP}" seascape
  if level_ready seascape; then level_sweep seascape seascape; else echo "SKIP seascape sweep (level env not built)"; fi
fi

# ----------------------------------------------------------------------------- seascape_xai (17b, 17c)
seascape_xai_plan() {   # plan xai_yaml: 17b reference agreement per level + 17c class profiles (one CPU job each)
  local plan="$1" X="$2" HJ="$SF_OUT/hierarchy/cluster_hierarchy.json" lv cases=()
  local pat="/data/nlahaye/NatureNet_Env/Seascapes/v2/$SP/seascape_${SP}_t{{t}}.tif"   # {{t}} -> {t} after the launcher format
  cases+=("      - {lv: fine, args: \"\"}")
  for lv in $LEVELS; do
    case "$lv" in
      parent) cases+=("      - {lv: parent, args: \"--level parent\"}") ;;
      k[0-9]*) if [ -f "$HJ" ]; then cases+=("      - {lv: $lv, args: \"--level coarse --k ${lv#k} --hierarchy_json $HJ\"}")
               else echo "SKIP 17b $lv (no $HJ: run the levels stage / 16a first)"; fi ;;
    esac
  done
  cat > "$plan" <<EOF
# generated by scripts/run_available_outputs.sh (SP=$SP, STAGE=seascape_xai); edit the script, not this file
launcher: {state_dir: "$S/available_${SP}_seascape_xai", gpus: $GPUS_YAML, cpus: $CPUS, retries: 0}
vars: {py: "$PY", x: "$X"}
jobs:
  - name: reference                 # 17b: agreement tables (NMI, ARI, homogeneity, completeness, Cramer's V) per level
    cmd: "{py} -m sit_fuse.xai.cluster_interpret -y {x} --reference_only {args}"
    cases:
$(printf '%s\n' "${cases[@]}")
    id_keys: [lv]
    cpus: 4
    env: $CPU_ENV
  - name: seascape_profiles         # 17c: CMEMS channel profiles within each Seascape class -> xai/interpret_seascape/
    cmd: "{py} -m sit_fuse.xai.cluster_interpret -y {x} --label_pattern '$pat' --label_scale 1 --out_subdir interpret_seascape --no_reference"
    cpus: 8
    env: $CPU_ENV
EOF
  echo "wrote $plan"
}
if { want seascape_xai && [ "${SKIP_LEVELS:-0}" != "1" ]; } || explicit seascape_xai; then
  echo "== steps 17b / 17c: Seascapes agreement and class profiles (SIT-FUSE xai)"
  SFC="$(sfc_dir)"
  XB="$SFC/model/copernicus_${SG}_v2.yaml"; XX="$SFC/xai/${SG}_xai_block_v2.yaml"; X="$SFC/model/copernicus_${SG}_v2_xai.yaml"
  SEA="/data/nlahaye/NatureNet_Env/Seascapes/v2/$SP"
  if ! compgen -G "$SEA/seascape_${SP}_t*.tif" > /dev/null; then
    echo "SKIP 17b / 17c (no Seascapes rasters in $SEA: run STAGE=seascape first)"
  elif ! "$HPY" -c 'import inspect, sys, sit_fuse.xai.cluster_interpret as m; sys.exit(0 if "reference_only" in inspect.getsource(m) else 1)' 2>/dev/null; then
    echo "SKIP 17b / 17c: the installed sit_fuse.xai.cluster_interpret has no --reference_only / --label_pattern"
    echo "     (copy SIT-FUSE/src/sit_fuse/xai/cluster_interpret.py from the 17b / 17c delivery into SIT-FUSE)"
  else
    if [ ! -f "$X" ] || [ "${REMAKE_XAI:-0}" = "1" ] || [ -z "$(ykey "$X" xai.interpret.reference)" ]; then
      if [ "${DRY:-0}" = "1" ]; then echo "would run make_xai_config -b $XB -x $XX -o $X"
      else "$PY" -m sit_fuse.xai.make_xai_config -b "$XB" -x "$XX" -o "$X"; fi      # adds interpret.reference
    fi
    seascape_xai_plan "$S/available_${SP}_seascape_xai.yaml" "$X"
    launch "$S/available_${SP}_seascape_xai.yaml"
    XD="$SF_OUT/xai"
    if [ "${DRY:-0}" = "1" ]; then echo "would run reference_summary -d $XD"
    elif compgen -G "$XD/interpret*/reference_*_agreement.csv" > /dev/null; then
      cpu "$PY" -m sit_fuse.xai.reference_summary -d "$XD" || echo "WARNING: reference_summary failed"   # -> reference_agreement_levels.{csv,md}
    else echo "SKIP reference_summary (no agreement tables yet)"; fi
  fi
fi

# ----------------------------------------------------------------------------- step20_post (sharks: scales / ma CV)
if { want step20_post && [ "${SKIP_STEP20:-0}" != "1" ]; } || explicit step20_post; then
  if [ -z "$STEP20_POST" ]; then
    if explicit step20_post; then echo "SKIP step 20 (shark secondary switch; nothing to do for $SP)"; fi
  else
    st20="$STEP20_POST"; [ "${STEP20_FITS:-1}" = "0" ] && st20="plots tables"
    echo "== step 20 (post): $HABITAT_SCRIPT STAGE=$st20 (finished folds skipped; folds claimed by another launcher not repeated)"
    for st in $st20; do sub "$st"; done
  fi
fi

# ----------------------------------------------------------------------------- audit
if want audit; then
  rc=0
  "$PY" -m naturenet.drivers.manifest_audit -m "naturenet/config/v2/audit/${SP}_pipeline.yaml" --steps 5b,6b,7-14 || rc=1
  for p in "$PO" "$S/available_${SP}_levels_env.yaml" "$S/available_${SP}_levels_sweep.yaml" \
           "$S/available_${SP}_seascape_env.yaml" "$S/available_${SP}_seascape_sweep.yaml" \
           "$S/available_${SP}_seascape_xai.yaml" "$S/whale_prescales_final.yaml" "$S/whale_prescales_habitat.yaml" \
           "$S/shark_swirl_kin_final.yaml" "$S/shark_habitat_tests.yaml" "$S/shark_swirl_kin_scales.yaml" \
           "$S/shark_swirl_kin_ma_${MA_STAGE:-screen}.yaml"; do
    [ -f "$p" ] && "$PY" -m sit_fuse.local_launcher status "$p" | tail -n 5
  done
  [ "$STAGE" = "audit" ] && exit $rc
fi
echo "done (SP=$SP, STAGE=$STAGE)"
