#!/usr/bin/env bash
# Final single-animal fits (all animals, holdout_fraction 0) of the chosen kinematic tools, then plots and the
# reward-environment projection:
#   whales  HIQL + kin K = 5 (primary)      whales_hiql_final_kin_k5.yaml
#           SWIRL S-2 + kin K = 5 (second)  whales_swirl_s2_final_kin_k5.yaml (kinematic ARHMM -> S-1 / S-2)
#   sharks  HIQL + kin K = 5 (primary)      sharks_hiql_final_kin_k5.yaml
#           HIQL + kin K = 2 (secondary)    sharks_hiql_final_kin_k2.yaml
#   bash run_final_fits.sh                  # fits (HIQL fits on CPU in parallel, SWIRL on one GPU), then plots
#   STAGE=fit | plots bash run_final_fits.sh
#   GPU      GPU for the SWIRL fit (default 6); HIQL_CPUS  CPU threads per HIQL fit (default 4; 3 fits run at once)
#   NN_SRC, PY as in run_multiagent_kfold.sh. Sync first (bash sync_v2_configs_all.sh).
# Outputs: <species root>/final/{hiql_kin,hiql_kin_k2,swirl_kin}/ (fits, *_kinematics.csv, plots/, reward_env_projection*/).
set -euo pipefail
STAGE="${STAGE:-all}"
NN_SRC="${NN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
GPU="${GPU:-6}"
HIQL_CPUS="${HIQL_CPUS:-4}"
cd "$NN_SRC"
C=naturenet/config/v2/swirl
HIQL=("whales_hiql_final_kin_k5:whales:5" "sharks_hiql_final_kin_k5:sharks:5" "sharks_hiql_final_kin_k2:sharks:2")
SW=whales_swirl_s2_final_kin_k5
LOG="${LOG:-/data/nlahaye/NatureNet/launcher/final_kin_logs}"
mkdir -p "$LOG"

cpu_env() {   # CPU-only JAX and bounded threads for one HIQL fit
  env JAX_PLATFORMS=cpu NATURENET_JAX_PLATFORM=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="$HIQL_CPUS" \
      MKL_NUM_THREADS="$HIQL_CPUS" OPENBLAS_NUM_THREADS="$HIQL_CPUS" NATURENET_HIAVI_RESTART_JOBS="$HIQL_CPUS" "$@"
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "fit" ]; then
  pids=()
  for e in "${HIQL[@]}"; do
    IFS=: read -r cfg sp K <<< "$e"
    echo "fit $cfg (log $LOG/$cfg.log)"
    cpu_env "$PY" -m naturenet.models.irl.run_hiavi_kfold -y "$C/$cfg.yaml" --holdout > "$LOG/$cfg.log" 2>&1 & pids+=($!)
  done
  echo "fit $SW on GPU $GPU (log $LOG/$SW.log)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m naturenet.models.irl.run_swirl -y "$C/$SW.yaml" > "$LOG/$SW.log" 2>&1 & pids+=($!)
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" = "0" ] || { echo "a final fit failed; see $LOG" >&2; exit 1; }
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "plots" ]; then
  for e in "${HIQL[@]}"; do
    IFS=: read -r cfg sp K <<< "$e"
    cpu_env "$PY" -m naturenet.models.irl.plot_hiavi -y "$C/$cfg.yaml"
    cpu_env "$PY" -m naturenet.models.irl.reward_env_projection -y "$C/$cfg.yaml" \
      -p "$C/${sp}_reward_env_projection_hiql.yaml" --n_hidden "$K"
  done
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m naturenet.models.irl.plot_swirl_new -y "$C/$SW.yaml"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m naturenet.models.irl.reward_env_projection -y "$C/$SW.yaml" \
    -p "$C/whales_reward_env_projection_s2_kin.yaml" --n_hidden 5
fi
