#!/usr/bin/env bash
# Shark secondary tool switch: SWIRL S-1 + kin K = 5 replaces HIQL + kin K = 2 (v2 only). Reruns every fit, plot and
# analysis that depends on the secondary tool, on the local launcher where the work is a job graph.
#   K* = 5 for shark S-1 + kin and S-2 + kin (fold-paired 1-SE rule on SWIRL_BASE_pc1T_kin); S-2 - S-1 at K = 5 is
#   +0.004 +/- 0.031 (4/5 folds), a tie -> S-1 is the tool; the final fit also writes S-2 (extra view, as for whales).
#
#   bash run_shark_swirl_kin.sh                 # STAGE=all: sync, final, scales, ma (screen), plots, tables, audit
#   STAGE=<stage> bash run_shark_swirl_kin.sh   # one stage:
#     sync     fill n_states / input_keys / variables into the new shark configs (sync_v2_configs; rerun-safe)
#     check    launcher dry runs of every plan + scales segment report + manifest audit (no fits)
#     final    launcher plan: final fit (GPU) -> plot_swirl_new -> reward projection S-1 / S-2 -> exploit / explore
#              (series + fitted kinematics) -> time-series correlation -> decision-context similarity -> animations
#     scales   single-animal decision scales CV at K = 5 (plan_jobs --scales; ARHMM init = kinematic baseline, linked)
#     ma       multi-agent SWIRL CV at K = 5: coupling (mixture + align copy) and decision scales (plan_jobs);
#              MA_STAGE=screen (default, observed CV only) | full (adds the time-shift nulls, reuses the screen parts)
#     plots    scales + multi-agent figures and summaries
#     tables   manuscript tables (ms_sharks_kin_*; T2 / T3 rows for HIQL K = 5, SWIRL K = 5, HIQL K = 2 if present)
#     drivers  re-plan the shark driver study with the new secondary (sharks_drivers.yaml) and force the jobs whose
#              inputs changed: correlation (test 7 series), ppc_collect, coupling_env, synthesis. Run after the
#              driver fits are complete (run_env_drivers.sh STAGE=audit).
#     habitat  explicit stage (step 11h; literature-comparison analyses 1 and 3): available points (shared with the
#              driver geometry stage) -> env at available points -> scene front / eddy stats -> habitat_tests
#              features | seasons -> wet / dry season reward projections (HIQL K = 5 / K = 2, SWIRL S-1). Needs final.
#     audit    pipeline manifest audit (steps 5b-14) + launcher status of the plans
#   Env: GPUS (default 6,7), CPUS (default 16 of the 48 cores), S (launcher root, default /data/nlahaye/NatureNet/launcher),
#        NN_SRC (src/ dir), PY, DRY=1 (plan + dry run only), MA_STAGE, N_ANIMALS (audit series count).
#   final, scales and ma are independent once sync is done: they can run at the same time in separate shells with
#   disjoint GPUS (e.g. GPUS=6 STAGE=final, GPUS=7 STAGE=scales). Every runner skips finished work and the launcher
#   keeps done markers, so any stage can be repeated.
set -euo pipefail
STAGE="${STAGE:-all}"
NN_SRC="${NN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
S="${S:-/data/nlahaye/NatureNet/launcher}"
PY="${PY:-python}"
GPUS="${GPUS:-6,7}"
CPUS="${CPUS:-16}"
case "$GPUS" in *,*|auto|none) ;; *) GPUS="$GPUS," ;; esac   # one id needs a trailing comma (a bare N = ids 0..N-1)
GPUS_YAML="[${GPUS%,}]"; case "$GPUS" in auto|none) GPUS_YAML="$GPUS" ;; esac
MA_STAGE="${MA_STAGE:-screen}"
cd "$NN_SRC"
C=naturenet/config/v2/swirl
T=naturenet/config/v2/time_series
D=naturenet/config/v2/drivers
E=naturenet/config/v2/environment
H=/data/nlahaye/NatureNet/Hammerhead_Out_v2
N="$PY -m naturenet.models"
K=5
FIN="$C/sharks_swirl_final_kin_k5.yaml"
SCL="$C/sharks_swirl_scales_kin.yaml"
MA="$C/sharks_multiagent_swirl_kin.yaml"
MAC="$C/sharks_multiagent_swirl_kin_coupling.yaml"
MAS="$C/sharks_scales_multiagent_swirl_kin.yaml"
FD="$H/final/swirl_kin"
UID_F=SHARK_SWIRL_KIN_FINAL
mkdir -p "$S"
want() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }
cpu() { env JAX_PLATFORMS=cpu NATURENET_JAX_PLATFORM=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 "$@"; }
launch() {   # plan [extra launcher args]: dry run summary, then run unless DRY=1
  local plan="$1"; shift
  "$PY" -m sit_fuse.local_launcher run "$plan" --dry-run --gpus "$GPUS" --cpus "$CPUS" "$@" \
    | awk '/WARNING/ && w++ < 5; /^\[/ {n++} END {print n " jobs in the launcher dry run"}'
  [ "${DRY:-0}" = "1" ] && return 0
  "$PY" -m sit_fuse.local_launcher run "$plan" --gpus "$GPUS" --cpus "$CPUS" "$@"
}

# ----------------------------------------------------------------------------- sync
if want sync || [ "$STAGE" = "check" ]; then
  dry=""; [ "$STAGE" = "check" ] || [ "${DRY:-0}" = "1" ] && dry="--dry_run"
  "$PY" -m naturenet.environment.sync_v2_configs patch -y "$E/sharks_sync_v2.yaml" $dry
fi

# ----------------------------------------------------------------------------- final (hand-written launcher plan)
PF="$S/shark_swirl_kin_final.yaml"
write_final_plan() {
  local gpu_env="{XLA_PYTHON_CLIENT_PREALLOCATE: \"false\"}"
  local cpu_env="{JAX_PLATFORMS: cpu, NATURENET_JAX_PLATFORM: cpu}"
  cat > "$PF" <<EOF
# generated by scripts/run_shark_swirl_kin.sh (STAGE=final); edit the script, not this file
launcher: {state_dir: "$S/shark_swirl_kin_final", gpus: $GPUS_YAML, cpus: $CPUS, retries: 0}
vars:
  py: "$PY"
  fin: "$FIN"
  fd: "$FD"
  ee: "$H/final/exploit_explore"
jobs:
  - name: fit                       # kinematic ARHMM (pc1T) -> S-1 -> S-2 at K = 5, all animals
    cmd: "{py} -m naturenet.models.irl.run_swirl -y {fin}"
    gpu: 0.5
    cpus: 4
    env: $gpu_env
    creates: ["{fd}/${UID_F}_${K}_12345_S2_naturenet_iter2.npz"]
  - name: plots                     # maps, tracks, series_linear_s1 / s2, state_reward_linear_s{1,2}.npy
    cmd: "{py} -m naturenet.models.irl.plot_swirl_new -y {fin}"
    gpu: 0.25
    cpus: 4
    env: $gpu_env
    after: [fit]
  - name: project                   # reward-to-environment projection + use vs availability
    cmd: "{py} -m naturenet.models.irl.reward_env_projection -y {fin} -p $C/sharks_reward_env_projection_{order}_kin.yaml --n_hidden $K"
    cases: [{order: s1}, {order: s2}]
    gpu: 0.25
    cpus: 2
    env: $gpu_env
    after: [{name: plots, on: []}]
  - name: ee_series                 # exploit / explore from the exported series (+ state reward)
    cmd: "{py} -m naturenet.models.irl.exploit_explore_init --data_dir {fd}/plots/series_linear_{order} --slices $T/sharks_correlation_final_kin_v2.yaml --state_reward {fd}/plots/state_reward_linear_{order}.npy --out_dir {ee}/sharks_swirl_kin_{order}_k$K"
    cases: [{order: s1}, {order: s2}]
    cpus: 2
    env: $cpu_env
    after: [{name: plots, on: []}]
  - name: ee_npz                    # exploit / explore from the fitted kinematic parameters
    cmd: "{py} -m naturenet.models.irl.exploit_explore_init --model_npz {fd}/{npz} --out_dir {ee}/sharks_swirl_kin_{order}_k$K"
    cases:
      - {order: s1, npz: "${UID_F}_${K}_12345_S1_naturenet_iter2_temp_2.npz"}
      - {order: s2, npz: "${UID_F}_${K}_12345_S2_naturenet_iter2.npz"}
    id_keys: [order]
    cpus: 1
    env: $cpu_env
    after: [{name: fit, on: []}, {name: ee_series, on: [order]}]   # same out_dir as ee_series: run after it
  - name: series                    # time-series correlation (HIQL K = 5, SWIRL S-1 / S-2; sharks_correlation_final_kin_v2)
    cmd: "{py} -m naturenet.models.time_series_analysis.correlation_analysis_shark -y $T/sharks_correlation_final_kin_v2.yaml"
    cpus: 4
    env: $cpu_env
    after: [{name: plots, on: []}]
  - name: decision_context          # decision-context similarity: HIQL K = 5 vs SWIRL S-1 / S-2 (+ HIQL K = 2)
    cmd: "{py} -m naturenet.drivers.decision_context_similarity -y $D/sharks_decision_context.yaml"
    cpus: 4
    env: $cpu_env
    after: [{name: ee_series, on: []}, {name: ee_npz, on: []}]
  - name: animate                   # single-animal kinematic finals (missing panels are skipped by the tool)
    cmd: "{py} -m naturenet.models.irl.animate_tracks -y naturenet/config/v2/plots/sharks_animations.yaml --only kin_finals kin_finals_hiql_k2 kin_swirl_by_mode"
    cpus: 2
    env: $cpu_env
    after: [{name: plots, on: []}]
EOF
  echo "wrote $PF"
}
if want final || [ "$STAGE" = "check" ]; then
  write_final_plan
  if [ "$STAGE" = "check" ]; then DRY=1 launch "$PF"; else launch "$PF"; fi
fi

# ----------------------------------------------------------------------------- scales (plan_jobs)
link_scales_init() {   # kinematic baseline ARHMM (SWIRL_BASE_pc1T_kin run 0, seed 12345, K = 5) -> scales_kin/
  mkdir -p "$H/scales_kin"
  for k in 0 1 2 3 4; do
    src="$H/baselines_kin/SWIRL_BASE_pc1T_kin_loop_0_latent_${K}_states_${K}_hidden_12345_seed_arhmm_s_KF_$k.npz"
    [ -f "$src" ] || { echo "missing kinematic ARHMM init: $src" >&2; return 1; }
    ln -sf "$src" "$H/scales_kin/SHARK_SWIRL_SCALES_KIN_${K}_hidden_12345_seed_arhmm_s_KF_$k.npz"
  done
}
PS="$S/shark_swirl_kin_scales.yaml"
if want scales || [ "$STAGE" = "check" ]; then
  if [ "${DRY:-0}" != "1" ] && [ "$STAGE" != "check" ]; then link_scales_init; fi
  [ "$STAGE" = "check" ] && { link_scales_init && "$PY" -m naturenet.models.irl.run_swirl_scales_kfold -y "$SCL" --check || true; }
  # same state_dir as run_scales_kfold.sh, so its combined plan sees these jobs as done
  "$PY" -m naturenet.models.irl.plan_jobs --out "$PS" --state_dir "$S/scales_kin_tools" --scales "$SCL" --gpus "$GPUS"
  if [ "$STAGE" = "check" ]; then DRY=1 launch "$PS"; else launch "$PS"; fi
fi

# ----------------------------------------------------------------------------- multi-agent (plan_jobs)
PM="$S/shark_swirl_kin_ma_${MA_STAGE}.yaml"
if want ma || [ "$STAGE" = "check" ]; then
  case "$MA_STAGE" in screen) opt="@skip_null" ;; full) opt="" ;; *) echo "MA_STAGE must be screen or full" >&2; exit 1 ;; esac
  vs="${VARIANTS_SWIRL:+@variants=$VARIANTS_SWIRL}"
  # same state_dir as run_multiagent_kfold.sh; the scales merges into ma_scales_summary.csv are file-locked
  "$PY" -m naturenet.models.irl.plan_jobs --out "$PM" --state_dir "$S/ma_kin_tools" \
    --multiagent        "$MA@backend=swirl@K=$K${opt}" \
    --multiagent        "$MAC@backend=swirl@K=$K${opt}" \
    --scales_multiagent "$MAS@backends=swirl_scales@K=$K${vs}${opt}" \
    --gpus "$GPUS" --ma_cpus 4 --null_chunk 2
  if [ "$STAGE" = "check" ]; then DRY=1 launch "$PM"; else launch "$PM"; fi
fi

# ----------------------------------------------------------------------------- habitat (explicit; step 11h)
PH="$S/shark_habitat_tests.yaml"
if [ "$STAGE" = "habitat" ]; then
  HAB="$D/sharks_habitat_tests.yaml"; AVL="$H/drivers/availability"; HT="$H/drivers/habitat_tests"
  cpu_env="{JAX_PLATFORMS: cpu, NATURENET_JAX_PLATFORM: cpu}"
  cat > "$PH" <<EOF
# generated by scripts/run_shark_swirl_kin.sh (STAGE=habitat); edit the script, not this file
launcher: {state_dir: "$S/shark_habitat_tests", gpus: $GPUS_YAML, cpus: $CPUS, retries: 0}
vars: {py: "$PY", hab: "$HAB"}
jobs:
  - name: availability              # same files as the driver geometry stage (skipped when present)
    cmd: "{py} -m naturenet.drivers.availability -y {hab}"
    cpus: 2
    env: $cpu_env
    creates: ["$AVL/hammerhead_v2_dfs.pkl", "$AVL/sample_inputs_avl.yaml"]
  - name: sample_avl
    cmd: "{py} -m naturenet.environment.sample_env_inputs -y $AVL/sample_inputs_avl.yaml"
    cpus: 4
    env: $cpu_env
    after: [availability]
    creates: ["$AVL/hammerhead_v2_states_input_channels.json"]
  - name: scene                     # per-scene front / eddy thresholds
    cmd: "{py} -m naturenet.drivers.lagged_env -y {hab} --part scene"
    cpus: 4
    env: $cpu_env
    creates: ["$HT/scene_feature_stats.csv"]
  - name: tests                     # analysis 1 (features) and 3 (wet vs dry seasons)
    cmd: "{py} -m naturenet.drivers.habitat_tests -y {hab} --part {part}"
    cases: [{part: features}, {part: seasons}]
    cpus: 2
    env: $cpu_env
    after: [{name: sample_avl, on: []}, {name: scene, on: []}]
  - name: project_season            # reward projection + use-availability by season (separate out_dirs)
    cmd: "{py} -m naturenet.models.irl.reward_env_projection -y $C/{yml} -p $C/{p} --n_hidden {k}"
    cases:
      - {tool: hiql_k5, yml: sharks_hiql_final_kin_k5.yaml, p: sharks_reward_env_projection_hiql_season.yaml, k: 5}
      - {tool: hiql_k2, yml: sharks_hiql_final_kin_k2.yaml, p: sharks_reward_env_projection_hiql_season.yaml, k: 2}
      - {tool: swirl_s1, yml: sharks_swirl_final_kin_k5.yaml, p: sharks_reward_env_projection_s1_kin_season.yaml, k: 5}
    id_keys: [tool]
    cpus: 2
    env: $cpu_env
EOF
  echo "wrote $PH"
  launch "$PH"
  exit 0
fi

[ "${DRY:-0}" = "1" ] && [ "$STAGE" != "check" ] && { echo "DRY=1: plans written, nothing run"; exit 0; }

# ----------------------------------------------------------------------------- plots
if want plots; then
  "$PY" -m naturenet.models.irl.plot_swirl_scales -y "$SCL" || echo "SKIP scales plots (scales CV not finished?)"
  MD=$H/multiagent
  if [ -f "$MD/multiagent_swirl_kin/ma_summary.json" ]; then
    "$PY" -m naturenet.models.irl.plot_hiavi_multiagent -y "$MA" --K $K --subdir multiagent_swirl_kin
    if [ -f "$MD/multiagent_swirl_kin_coupling/ma_summary.json" ]; then
      "$PY" -m naturenet.models.irl.plot_hiavi_multiagent -y "$MAC" --K $K --subdir multiagent_swirl_kin_coupling
      "$PY" -m naturenet.models.irl.plot_multiagent_compare --dirs "$MD/multiagent_swirl_kin" "$MD/multiagent_swirl_kin_coupling" \
        --labels initial coupling --K $K --out "$MD/multiagent_swirl_kin_coupling/compare_initial" --B
    fi
    [ -f "$MD/multiagent_kin/ma_summary.json" ] && \
      "$PY" -m naturenet.models.irl.plot_multiagent_compare --dirs "$MD/multiagent_kin" "$MD/multiagent_swirl_kin" \
        --labels "HIQL+kin" "SWIRL+kin" --K $K --out "$MD/ma_compare_hiql_swirl_kin"
  else echo "SKIP multi-agent plots (no $MD/multiagent_swirl_kin/ma_summary.json)"; fi
  [ -f "$H/multiagent_scales_kin/ma_scales_summary.csv" ] && \
    "$PY" -m naturenet.models.irl.plot_scales_multiagent -y "$MAS" --K $K || true
  cpu "$PY" -m naturenet.models.irl.animate_tracks -y naturenet/config/v2/plots/sharks_animations.yaml \
    --only scales_swirl ma_swirl_k5 ma_scales_swirl_k5 || true
fi

# ----------------------------------------------------------------------------- tables
if want tables; then
  M=$H/manuscript_kin; mkdir -p "$M"
  args=(--out "$M/ms_sharks_kin")
  L=$H/ll_comparison_kin_nn/hammerhead_v2_kin_nn_ll_summary.csv
  [ -f "$L" ] && args+=(--ll_summary "$L") || echo "SKIP T1 (missing $L)"
  sc=()
  for s in "hiql_kin_k5=$H/scales_kin/HIQL_SCALES_KIN_K5_scales_summary.csv" \
           "swirl_kin_k5=$H/scales_kin/SHARK_SWIRL_SCALES_KIN_scales_summary.csv" \
           "hiql_kin_k2=$H/scales_kin/HIQL_SCALES_KIN_K2_scales_summary.csv"; do
    [ -f "${s#*=}" ] && sc+=("$s") || echo "SKIP T2 row (missing ${s#*=})"
  done
  [ ${#sc[@]} -gt 0 ] && args+=(--scales_summary "${sc[@]}")
  mc=()
  if [ -f "$H/multiagent/multiagent_kin/ma_summary.json" ]; then
    cpu $N.irl.plot_multiagent_compare --dirs "$H/multiagent/multiagent_kin" --labels "HIQL+kin" --K 5 2 \
      --out "$M/ma_compare_kin" && mc+=("$M/ma_compare_kin.csv")
  fi
  if [ -f "$H/multiagent/multiagent_swirl_kin/ma_summary.json" ]; then
    cpu $N.irl.plot_multiagent_compare --dirs "$H/multiagent/multiagent_swirl_kin" --labels "SWIRL+kin" --K $K \
      --out "$M/ma_compare_swirl_kin" && mc+=("$M/ma_compare_swirl_kin.csv")
  fi
  [ ${#mc[@]} -gt 0 ] && args+=(--ma_compare "${mc[@]}") || echo "SKIP T3 coupling (no multi-agent summaries)"
  [ -f "$H/multiagent_scales_kin/ma_scales_summary.csv" ] && args+=(--ma_scales "$H/multiagent_scales_kin/ma_scales_summary.csv") \
    || echo "SKIP T3 scales (no ma_scales_summary.csv)"
  cpu $N.irl.summarize_experiments "${args[@]}"
fi

# ----------------------------------------------------------------------------- drivers (explicit stage only)
if [ "$STAGE" = "drivers" ]; then
  PD="$S/plan_drivers_sharks.yaml"
  "$PY" -m naturenet.drivers.plan -y "$D/sharks_drivers.yaml" --out "$PD" --gpus "$GPUS" --cpus "$CPUS"
  launch "$PD" --only correlation ppc_collect coupling_env synthesis --force
fi

# ----------------------------------------------------------------------------- audit
if want audit || [ "$STAGE" = "check" ]; then
  rc=0
  "$PY" -m naturenet.drivers.manifest_audit -m naturenet/config/v2/audit/sharks_pipeline.yaml --steps 5b,6b,7-14 \
    ${N_ANIMALS:+--set N_ANIMALS=$N_ANIMALS} || rc=1
  for p in "$PF" "$PS" "$PM"; do [ -f "$p" ] && "$PY" -m sit_fuse.local_launcher status "$p" | tail -n 5; done
  [ "$STAGE" = "audit" ] && exit $rc
fi
echo "done (STAGE=$STAGE)"
