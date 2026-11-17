#!/usr/bin/env bash
# Single-animal decision-scales K-fold CV for the chosen kinematic tools (v2; same data / folds as the baselines):
#   whales  HIQL + kin K = 5 (primary)      whales_hiql_scales_kin_k5.yaml
#           SWIRL S-2 + kin K = 5 (second)  whales_swirl_scales_kin.yaml (fixed epochs; init = kinematic baseline ARHMM;
#                                           N1 = S-1 + kin, N2 = order 2 = the S-2 analogue)
#   sharks  HIQL + kin K = 5 (primary)      sharks_hiql_scales_kin_k5.yaml
#           SWIRL S-1 + kin K = 5 (second)  sharks_swirl_scales_kin.yaml (init = kinematic baseline ARHMM SWIRL_BASE_pc1T_kin)
#           HIQL + kin K = 2 (earlier secondary) sharks_hiql_scales_kin_k2.yaml
#   bash run_scales_kfold.sh               # link the ARHMM init, --check, plan, launch, plots
#   STAGE=check bash run_scales_kfold.sh   # link + segment / context report only
#   STAGE=plots bash run_scales_kfold.sh   # figures only
#   GPUS, CPUS, DRY=1, S, NN_SRC, PY as in run_multiagent_kfold.sh (HIQL jobs are CPU-only; SWIRL uses the GPUs)
# Sync first (bash sync_v2_configs_all.sh). Outputs: <species root>/scales_kin/<run_uid>_scales_cv.csv (+ fit files).
set -euo pipefail
STAGE="${STAGE:-run}"
NN_SRC="${NN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
S="${S:-/data/nlahaye/NatureNet/launcher}"
PY="${PY:-python}"
GPUS="${GPUS:-4,5}"
CPUS="${CPUS:-12}"
cd "$NN_SRC"
C=naturenet/config/v2/swirl
CFGS=("$C/whales_hiql_scales_kin_k5.yaml" "$C/whales_swirl_scales_kin.yaml"
      "$C/sharks_hiql_scales_kin_k5.yaml" "$C/sharks_swirl_scales_kin.yaml" "$C/sharks_hiql_scales_kin_k2.yaml")

if [ "$STAGE" = "plots" ]; then
  for c in "${CFGS[@]}"; do "$PY" -m naturenet.models.irl.plot_swirl_scales -y "$c"; done
  exit 0
fi

# Whale SWIRL scales start from the kinematic baseline ARHMM (WHALE_SWIRL_BASE_kin run 0, seed 12345, K = 5).
W=/data/nlahaye/NatureNet/Blue_Whale_v2
if [ "${DRY:-0}" != "1" ]; then
  mkdir -p "$W/scales_kin"
  for k in 0 1 2 3 4; do
    src="$W/baselines_kin/WHALE_SWIRL_BASE_kin_loop_0_latent_5_states_5_hidden_12345_seed_arhmm_s_KF_$k.npz"
    [ -f "$src" ] || { echo "missing kinematic ARHMM init: $src" >&2; exit 1; }
    ln -sf "$src" "$W/scales_kin/WHALE_SWIRL_SCALES_KIN_5_hidden_12345_seed_arhmm_s_KF_$k.npz"
  done
  # shark SWIRL scales: kinematic baseline ARHMM SWIRL_BASE_pc1T_kin run 0, seed 12345, K = 5
  H=/data/nlahaye/NatureNet/Hammerhead_Out_v2
  mkdir -p "$H/scales_kin"
  for k in 0 1 2 3 4; do
    src="$H/baselines_kin/SWIRL_BASE_pc1T_kin_loop_0_latent_5_states_5_hidden_12345_seed_arhmm_s_KF_$k.npz"
    [ -f "$src" ] || { echo "missing kinematic ARHMM init: $src" >&2; exit 1; }
    ln -sf "$src" "$H/scales_kin/SHARK_SWIRL_SCALES_KIN_5_hidden_12345_seed_arhmm_s_KF_$k.npz"
  done
fi

if [ "$STAGE" = "check" ]; then
  for c in "${CFGS[@]}"; do "$PY" -m naturenet.models.irl.run_swirl_scales_kfold -y "$c" --check; done
  exit 0
fi
[ "$STAGE" = "run" ] || { echo "STAGE must be run, check or plots" >&2; exit 1; }

plan="$S/scales_kin_tools.yaml"
mkdir -p "$S"
args=(); for c in "${CFGS[@]}"; do args+=(--scales "$c"); done
"$PY" -m naturenet.models.irl.plan_jobs --out "$plan" --state_dir "$S/scales_kin_tools" "${args[@]}" --gpus "$GPUS"
"$PY" -m sit_fuse.local_launcher run "$plan" --dry-run --gpus "$GPUS" --cpus "$CPUS" \
  | awk '/WARNING/ && w++ < 5; /^\[/ {n++} END {print n " jobs in the launcher dry run"}'
[ "${DRY:-0}" = "1" ] && exit 0
"$PY" -m sit_fuse.local_launcher run "$plan" --gpus "$GPUS" --cpus "$CPUS"
STAGE=plots bash "${BASH_SOURCE[0]}"
