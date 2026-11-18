#!/usr/bin/env bash
# Whales (v2, kinematic tools at K = 5): every output and figure that does not depend on the decision-scales runs.
#   Tools: HIQL + kin K = 5 (final/hiql_kin/), SWIRL S-2 + kin K = 5 (final/swirl_kin/, S-1 of the same fit as an
#   extra view). Runbook steps 5/5b (LL tables), 7, 8, 8b, 10-12, 11h (habitat tests), 13 (T1 + T3 without the
#   scales rows) and 14b (coupling, explicit stage). Scales-dependent work (6b, T2, scales animations, the driver
#   study) stays with run_scales_kfold.sh / run_kin_downstream.sh / run_env_drivers.sh.
#
#   bash run_whale_prescales.sh                 # STAGE=all: sync, final, habitat, ll, plots, tables, audit
#   STAGE=<stage> bash run_whale_prescales.sh   # one stage:
#     sync      fill n_states / input_keys / variables into the v2 whale configs, incl. the new *_lag projections
#     check     sync dry run + launcher dry runs of the final and habitat plans + manifest audit (no fits)
#     final     launcher plan: HIQL fit (CPU) | SWIRL S-1/S-2 fit (GPU) -> plot_hiavi / plot_swirl_new ->
#               reward projection (HIQL, S-2, S-1) -> exploit / explore (series + fitted kinematics) ->
#               time-series correlation -> decision-context similarity -> animations (kin_finals, kin_hiql_by_mode)
#               Finished fits are skipped (creates = the final npz); REFIT=1 forces every final job.
#     habitat   launcher plan (literature-comparison analyses 1-2): available points (shared with the driver
#               geometry stage) -> env at available points -> scene front / eddy stats -> lagged wind / Ekman / wave
#               means (used + available) -> habitat_tests features | lags | lag_profiles -> lagged reward projections
#     ll        LL comparison plots (action-only K selection + kinematic / NN-SWIRL; inputs for T1)
#     plots     multi-agent (kinematic) figures if the summaries exist; no scales figures
#     tables    summarize_experiments: T1 + T3 coupling (scales rows skipped until the scales summaries exist)
#     coupling  explicit only: step 14b coupling power + coupling forms (HIQL on CPU, SWIRL on one GPU);
#               MA_STAGE=screen (default) | full (adds the nulls)
#     audit     manifest audit of steps 5b, 7-13 (incl. 11h) + launcher status
#   Env: GPUS (default 6; whale scales use 4,5 and multi-agent 0-3 by default, so avoid those while they run;
#        run_shark_swirl_kin.sh also defaults to 6,7), CPUS (default 16 of 48), S (launcher root), NN_SRC, PY,
#        DRY=1 (plans + dry runs only), REFIT=1, MA_STAGE.
#   final and habitat are independent of the scales CV; habitat needs the final series (it skips missing tools).
set -euo pipefail
STAGE="${STAGE:-all}"
NN_SRC="${NN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
S="${S:-/data/nlahaye/NatureNet/launcher}"
PY="${PY:-python}"
GPUS="${GPUS:-6}"
CPUS="${CPUS:-16}"
case "$GPUS" in *,*|auto|none) ;; *) GPUS="$GPUS," ;; esac   # one id needs a trailing comma (a bare N = ids 0..N-1)
GPUS_YAML="[${GPUS%,}]"; case "$GPUS" in auto|none) GPUS_YAML="$GPUS" ;; esac
MA_STAGE="${MA_STAGE:-screen}"
cd "$NN_SRC"
C=naturenet/config/v2/swirl
T=naturenet/config/v2/time_series
D=naturenet/config/v2/drivers
E=naturenet/config/v2/environment
P=naturenet/config/v2/plots
W=/data/nlahaye/NatureNet/Blue_Whale_v2
N="$PY -m naturenet.models"
K=5
FH="$C/whales_hiql_final_kin_k5.yaml"
FS="$C/whales_swirl_s2_final_kin_k5.yaml"
HD="$W/final/hiql_kin"
SD="$W/final/swirl_kin"
EE="$W/final/exploit_explore"
NPZ_H="$HD/WHALE_HIQL_KIN_FINAL_${K}_12345_hiavi.npz"
NPZ_S="$SD/WHALE_SWIRL_KIN_FINAL_${K}_12345_S2_naturenet_iter2.npz"
HAB="$D/whales_habitat_tests.yaml"
AVL="$W/drivers/availability"
HT="$W/drivers/habitat_tests"
RUID=SWIRL_WHALE_1517
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
GPU_ENV="{XLA_PYTHON_CLIENT_PREALLOCATE: \"false\"}"
CPU_ENV="{JAX_PLATFORMS: cpu, NATURENET_JAX_PLATFORM: cpu}"

# ----------------------------------------------------------------------------- sync
if want sync || [ "$STAGE" = "check" ]; then
  dry=""; { [ "$STAGE" = "check" ] || [ "${DRY:-0}" = "1" ]; } && dry="--dry_run"
  "$PY" -m naturenet.environment.sync_v2_configs patch -y "$E/whales_sync_v2.yaml" $dry
fi

# ----------------------------------------------------------------------------- final (steps 7, 8, 8b, 10-12, 14b.4)
PF="$S/whale_prescales_final.yaml"
write_final_plan() {
  cat > "$PF" <<EOF
# generated by scripts/run_whale_prescales.sh (STAGE=final); edit the script, not this file
launcher: {state_dir: "$S/whale_prescales_final", gpus: $GPUS_YAML, cpus: $CPUS, retries: 0}
vars:
  py: "$PY"
  fh: "$FH"
  fs: "$FS"
  ee: "$EE"
jobs:
  - name: fit_hiql                  # HIQL + kin K = 5, all animals (holdout = full-data final fit)
    cmd: "{py} -m naturenet.models.irl.run_hiavi_kfold -y {fh} --holdout"
    cpus: 8
    env: $CPU_ENV
    creates: ["$NPZ_H"]
  - name: fit_swirl                 # kinematic ARHMM -> S-1 -> S-2 at K = 5, all animals
    cmd: "{py} -m naturenet.models.irl.run_swirl -y {fs}"
    gpu: 1
    cpus: 4
    env: $GPU_ENV
    creates: ["$NPZ_S"]
  - name: plots_hiql                # maps, tracks, series_hiavi_s1/, state_reward_hiavi_s1.npy
    cmd: "{py} -m naturenet.models.irl.plot_hiavi -y {fh}"
    cpus: 4
    env: $CPU_ENV
    after: [fit_hiql]
  - name: plots_swirl               # series_linear_s1 / s2, state_reward_linear_s{1,2}.npy
    cmd: "{py} -m naturenet.models.irl.plot_swirl_new -y {fs}"
    gpu: 0.25
    cpus: 4
    env: $GPU_ENV
    after: [fit_swirl]
  - name: project_hiql              # reward-to-environment projection + use vs availability (step 11)
    cmd: "{py} -m naturenet.models.irl.reward_env_projection -y {fh} -p $C/whales_reward_env_projection_hiql.yaml --n_hidden $K"
    cpus: 2
    env: $CPU_ENV
    after: [plots_hiql]
  - name: project_swirl
    cmd: "{py} -m naturenet.models.irl.reward_env_projection -y {fs} -p $C/whales_reward_env_projection_{order}_kin.yaml --n_hidden $K"
    cases: [{order: s2}, {order: s1}]
    gpu: 0.25
    cpus: 2
    env: $GPU_ENV
    after: [{name: plots_swirl, on: []}]
  - name: ee_series                 # exploit / explore from the exported series (+ state reward; step 10)
    cmd: "{py} -m naturenet.models.irl.exploit_explore_init --data_dir {dir}/plots/series_{rep} --slices $T/whales_correlation_final_kin_v2.yaml --state_reward {dir}/plots/state_reward_{rep}.npy --out_dir {ee}/{label}"
    cases:
      - {label: whales_hiql_kin_k5,     dir: "$HD", rep: hiavi_s1}
      - {label: whales_swirl_kin_s2_k5, dir: "$SD", rep: linear_s2}
      - {label: whales_swirl_kin_s1_k5, dir: "$SD", rep: linear_s1}
    id_keys: [label]
    cpus: 2
    env: $CPU_ENV
    after: [{name: plots_hiql, on: []}, {name: plots_swirl, on: []}]
  - name: ee_npz                    # exploit / explore from the fitted kinematic parameters (same out_dir: after ee_series)
    cmd: "{py} -m naturenet.models.irl.exploit_explore_init --model_npz {npz} --out_dir {ee}/{label}"
    cases:
      - {label: whales_hiql_kin_k5,     npz: "$NPZ_H"}
      - {label: whales_swirl_kin_s2_k5, npz: "$NPZ_S"}
    id_keys: [label]
    cpus: 1
    env: $CPU_ENV
    after: [{name: ee_series, on: [label]}]
  - name: series                    # time-series correlation, all slices (step 12)
    cmd: "{py} -m naturenet.models.time_series_analysis.correlation_analysis_shark -y $T/whales_correlation_final_kin_v2.yaml"
    cpus: 4
    env: $CPU_ENV
    after: [{name: plots_hiql, on: []}, {name: plots_swirl, on: []}]
  - name: decision_context          # decision-context similarity HIQL vs SWIRL S-2 / S-1 (step 14b.4)
    cmd: "{py} -m naturenet.drivers.decision_context_similarity -y $D/whales_decision_context.yaml"
    cpus: 4
    env: $CPU_ENV
    after: [{name: ee_series, on: []}, {name: ee_npz, on: []}]
  - name: animate                   # single-animal kinematic finals (step 8b)
    cmd: "{py} -m naturenet.models.irl.animate_tracks -y $P/whales_animations.yaml --only kin_finals kin_hiql_by_mode"
    cpus: 2
    env: $CPU_ENV
    after: [{name: ee_series, on: []}]
EOF
  echo "wrote $PF"
}
if want final || [ "$STAGE" = "check" ]; then
  write_final_plan
  if [ "$STAGE" = "check" ]; then DRY=1 launch "$PF"
  elif [ "${REFIT:-0}" = "1" ]; then launch "$PF" --force
  else launch "$PF"; fi
fi

# ----------------------------------------------------------------------------- habitat (step 11h; analyses 1-2)
PH="$S/whale_prescales_habitat.yaml"
write_habitat_plan() {
  cat > "$PH" <<EOF
# generated by scripts/run_whale_prescales.sh (STAGE=habitat); edit the script, not this file
launcher: {state_dir: "$S/whale_prescales_habitat", gpus: $GPUS_YAML, cpus: $CPUS, retries: 0}
vars:
  py: "$PY"
  hab: "$HAB"
jobs:
  - name: availability              # step-selection available points; the driver geometry stage writes the same files
    cmd: "{py} -m naturenet.drivers.availability -y {hab}"
    cpus: 2
    env: $CPU_ENV
    creates: ["$AVL/${RUID}_dfs.pkl", "$AVL/sample_inputs_avl.yaml"]
  - name: sample_avl                # environment at the available points (same channels as the movement run)
    cmd: "{py} -m naturenet.environment.sample_env_inputs -y $AVL/sample_inputs_avl.yaml"
    cpus: 4
    env: $CPU_ENV
    after: [availability]
    creates: ["$AVL/${RUID}_states_input_channels.json"]
  - name: scene                     # per-scene front / eddy thresholds (scene_feature_stats.csv)
    cmd: "{py} -m naturenet.drivers.lagged_env -y {hab} --part scene"
    cpus: 4
    env: $CPU_ENV
    creates: ["$HT/scene_feature_stats.csv"]
  - name: lagged                    # 8 / 30-day means and 1-3 week lags at the used and available fixes
    cmd: "{py} -m naturenet.drivers.lagged_env -y {hab} --part lagged {flag}"
    cases: [{set: used, flag: ""}, {set: avl, flag: "--avl"}]
    id_keys: [set]
    cpus: 4
    env: $CPU_ENV
    after: [{name: availability, on: []}]
  - name: features                  # analysis 1: fronts / eddies by mode (MH odds ratios, mode contrasts)
    cmd: "{py} -m naturenet.drivers.habitat_tests -y {hab} --part features"
    cpus: 2
    env: $CPU_ENV
    after: [{name: sample_avl, on: []}, {name: scene, on: []}]
  - name: lags                      # analysis 2: lagged use-availability, mode association, reward association
    cmd: "{py} -m naturenet.drivers.habitat_tests -y {hab} --part lags"
    cpus: 2
    env: $CPU_ENV
    after: [{name: lagged, on: []}, {name: sample_avl, on: []}]
  - name: lag_profiles              # occupancy-conditioned lagged means per abstract state (projection extras)
    cmd: "{py} -m naturenet.drivers.habitat_tests -y {hab} --part lag_profiles"
    cpus: 2
    env: $CPU_ENV
    after: [{name: lagged, on: []}]
  - name: project_lag               # reward projection with the lagged state profiles (separate out_dirs)
    cmd: "{py} -m naturenet.models.irl.reward_env_projection -y {yml} -p $C/whales_reward_env_projection_{p}_lag.yaml --n_hidden $K"
    cases: [{tool: hiql, yml: "$FH", p: hiql}, {tool: swirl_s2, yml: "$FS", p: s2_kin}]
    id_keys: [tool]
    cpus: 2
    env: $CPU_ENV
    after: [{name: lag_profiles, on: []}]
EOF
  echo "wrote $PH"
}
if want habitat || [ "$STAGE" = "check" ]; then
  write_habitat_plan
  for d in "$HD/plots/series_hiavi_s1" "$SD/plots/series_linear_s2"; do
    [ -d "$d" ] || echo "NOTE habitat: missing $d (run STAGE=final first; habitat_tests skips missing tools)"
  done
  if [ "$STAGE" = "check" ]; then DRY=1 launch "$PH"; else launch "$PH"; fi
fi

# ----------------------------------------------------------------------------- coupling (explicit; step 14b.1-3)
PC="$S/whale_prescales_coupling_${MA_STAGE}.yaml"
if [ "$STAGE" = "coupling" ]; then
  case "$MA_STAGE" in screen) opt="@skip_null" ;; full) opt="" ;; *) echo "MA_STAGE must be screen or full" >&2; exit 1 ;; esac
  # same state_dir as run_multiagent_kfold.sh, so finished parts are reused
  "$PY" -m naturenet.models.irl.plan_jobs --out "$PC" --state_dir "$S/ma_kin_tools" \
    --multiagent "$C/whales_multiagent_kin_coupling.yaml@backend=hiavi@K=$K${opt}" \
    --multiagent "$C/whales_multiagent_swirl_kin_coupling.yaml@backend=swirl@K=$K${opt}" \
    --gpus "$GPUS" --ma_cpus 4 --null_chunk 2
  if [ "${DRY:-0}" != "1" ]; then
    cpu $N.irl.coupling_power -y "$C/whales_multiagent_kin_coupling.yaml" --K $K --variant temporal \
      --sim-form align --fit-forms mixture align --effects 0 0.5 1 2 3 --reps 2 --max-heldout 6 \
      --out "$W/multiagent/coupling_power_temporal"
  fi
  launch "$PC"
  [ "${DRY:-0}" = "1" ] || STAGE=plots bash "${BASH_SOURCE[0]}"
  exit 0
fi

[ "${DRY:-0}" = "1" ] && [ "$STAGE" != "check" ] && { echo "DRY=1: plans written, nothing run"; exit 0; }

# ----------------------------------------------------------------------------- ll (steps 5, 5b: T1 inputs)
if want ll; then
  cpu $N.irl.plot_ll_comparison -y "$C/whales_ll_comparison.yaml"        || echo "SKIP action-only LL comparison"
  cpu $N.irl.plot_ll_comparison -y "$C/whales_ll_comparison_kin_nn.yaml" || echo "SKIP kinematic LL comparison"
fi

# ----------------------------------------------------------------------------- plots (multi-agent, no scales)
if want plots; then
  MD=$W/multiagent
  if [ -f "$MD/multiagent_kin/ma_summary.json" ]; then
    cpu $N.irl.plot_hiavi_multiagent -y "$C/whales_multiagent_kin.yaml" --K $K
    [ -f "$MD/multiagent_kin_coupling/ma_summary.json" ] && \
      cpu $N.irl.plot_multiagent_compare --dirs "$MD/multiagent_kin" "$MD/multiagent_kin_coupling" \
        --labels initial coupling --K $K --out "$MD/multiagent_kin_coupling/compare_initial" --B
  else echo "SKIP HIQL multi-agent plots (no $MD/multiagent_kin/ma_summary.json)"; fi
  if [ -f "$MD/multiagent_swirl_kin/ma_summary.json" ]; then
    cpu $N.irl.plot_hiavi_multiagent -y "$C/whales_multiagent_swirl_kin.yaml" --K $K --subdir multiagent_swirl_kin
    [ -f "$MD/multiagent_swirl_kin_coupling/ma_summary.json" ] && \
      cpu $N.irl.plot_multiagent_compare --dirs "$MD/multiagent_swirl_kin" "$MD/multiagent_swirl_kin_coupling" \
        --labels initial coupling --K $K --out "$MD/multiagent_swirl_kin_coupling/compare_initial" --B
  else echo "SKIP SWIRL multi-agent plots (no $MD/multiagent_swirl_kin/ma_summary.json)"; fi
  [ -f "$MD/ma_final_K5.npz" ] || [ -f "$MD/multiagent_kin/ma_final_K5.npz" ] && \
    cpu $N.irl.animate_tracks -y "$P/whales_animations.yaml" --only ma_hiql || true
fi

# ----------------------------------------------------------------------------- tables (T1 + T3 coupling)
if want tables; then
  M=$W/manuscript_kin; mkdir -p "$M"
  args=(--out "$M/ms_whales_kin")
  L=$W/ll_comparison_kin_nn/whale_v2_kin_nn_ll_summary.csv
  [ -f "$L" ] && args+=(--ll_summary "$L") || echo "SKIP T1 (missing $L)"
  sc=()
  for s in "hiql_kin_k5=$W/scales_kin/WHALE_HIQL_SCALES_KIN_K5_scales_summary.csv" \
           "swirl_kin_k5=$W/scales_kin/WHALE_SWIRL_SCALES_KIN_scales_summary.csv"; do
    [ -f "${s#*=}" ] && sc+=("$s") || echo "SKIP T2 row (scales not finished: ${s#*=})"
  done
  [ ${#sc[@]} -gt 0 ] && args+=(--scales_summary "${sc[@]}")
  if [ -f "$W/multiagent/multiagent_kin/ma_summary.json" ]; then
    cpu $N.irl.plot_multiagent_compare --dirs "$W/multiagent/multiagent_kin" --labels "HIQL+kin" --K $K \
      --out "$M/ma_compare_kin" && args+=(--ma_compare "$M/ma_compare_kin.csv")
  else echo "SKIP T3 coupling (no multi-agent summary)"; fi
  [ -f "$W/multiagent_scales_kin/ma_scales_summary.csv" ] && args+=(--ma_scales "$W/multiagent_scales_kin/ma_scales_summary.csv") \
    || echo "SKIP T3 scales (multi-agent scales not finished)"
  cpu $N.irl.summarize_experiments "${args[@]}"
fi

# ----------------------------------------------------------------------------- audit
if want audit || [ "$STAGE" = "check" ]; then
  rc=0
  "$PY" -m naturenet.drivers.manifest_audit -m naturenet/config/v2/audit/whales_pipeline.yaml --steps 5b,7-13 || rc=1
  for p in "$PF" "$PH"; do [ -f "$p" ] && "$PY" -m sit_fuse.local_launcher status "$p" | tail -n 5; done
  [ "$STAGE" = "audit" ] && exit $rc
fi
echo "done (STAGE=$STAGE)"
