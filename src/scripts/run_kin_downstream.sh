#!/usr/bin/env bash
# Downstream steps 10-13 for the chosen kinematics-enabled tools (v2 only; every path is a *_kin output):
#   whales  HIQL + kin K = 5 (primary)     final/hiql_kin/
#           SWIRL S-2 + kin K = 5 (second) final/swirl_kin/  (S-1 of the same fit as an extra view)
#   sharks  HIQL + kin K = 5 (primary)     final/hiql_kin/
#           SWIRL S-1 + kin K = 5 (second) final/swirl_kin/  (S-2 of the same fit as an extra view; replaces HIQL K = 2)
#           HIQL + kin K = 2 (earlier secondary, granularity comparison) final/hiql_kin_k2/
#   bash run_kin_downstream.sh                    # all stages, in order
#   STAGE=ee | project | series | tables bash run_kin_downstream.sh
#     ee       step 10  exploit / explore labels (series + fitted kinematics)
#     project  step 11  reward-to-environment projection (also run by run_final_fits.sh STAGE=plots)
#     series   step 12  time-series analysis (correlation_analysis_shark, *_kin_v2.yaml)
#     tables   step 13  multi-agent comparison table + manuscript tables (ms_<sp>_kin_*)
#   Needs: run_final_fits.sh (fits + plots), run_scales_kfold.sh, run_multiagent_kfold.sh (screen or full + plots),
#   plot_ll_comparison on the *_ll_comparison_kin_nn.yaml configs. Missing inputs are skipped with a message.
#   GPU (projection of the SWIRL fit, default 6), CPUS (threads for CPU steps, default 4), NN_SRC, PY.
set -uo pipefail
STAGE="${STAGE:-all}"
NN_SRC="${NN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
GPU="${GPU:-6}"
CPUS="${CPUS:-4}"
cd "$NN_SRC"
C=naturenet/config/v2/swirl
T=naturenet/config/v2/time_series
W=/data/nlahaye/NatureNet/Blue_Whale_v2
H=/data/nlahaye/NatureNet/Hammerhead_Out_v2
N="$PY -m naturenet.models"
export OMP_NUM_THREADS="$CPUS" MKL_NUM_THREADS="$CPUS" OPENBLAS_NUM_THREADS="$CPUS"
cpu() { env JAX_PLATFORMS=cpu NATURENET_JAX_PLATFORM=cpu CUDA_VISIBLE_DEVICES= "$@"; }
want() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }
have() { [ -e "$1" ] || { echo "SKIP (missing $1)"; return 1; }; }

# label:yaml:K:final dir:series dir:state_reward tag:projection yaml
TOOLS=(
  "whales_hiql_kin_k5:whales_hiql_final_kin_k5:5:$W/final/hiql_kin:series_hiavi_s1:hiavi_s1:whales_reward_env_projection_hiql"
  "whales_swirl_kin_s2_k5:whales_swirl_s2_final_kin_k5:5:$W/final/swirl_kin:series_linear_s2:linear_s2:whales_reward_env_projection_s2_kin"
  "whales_swirl_kin_s1_k5:whales_swirl_s2_final_kin_k5:5:$W/final/swirl_kin:series_linear_s1:linear_s1:whales_reward_env_projection_s1_kin"
  "sharks_hiql_kin_k5:sharks_hiql_final_kin_k5:5:$H/final/hiql_kin:series_hiavi_s1:hiavi_s1:sharks_reward_env_projection_hiql"
  "sharks_swirl_kin_s1_k5:sharks_swirl_final_kin_k5:5:$H/final/swirl_kin:series_linear_s1:linear_s1:sharks_reward_env_projection_s1_kin"
  "sharks_swirl_kin_s2_k5:sharks_swirl_final_kin_k5:5:$H/final/swirl_kin:series_linear_s2:linear_s2:sharks_reward_env_projection_s2_kin"
  "sharks_hiql_kin_k2:sharks_hiql_final_kin_k2:2:$H/final/hiql_kin_k2:series_hiavi_s1:hiavi_s1:sharks_reward_env_projection_hiql"
)
# fitted-model npz (<out_dir>/<run_uid>_<K>_<seed><suffix>.npz) for the parameter-based labels
NPZ=(
  "whales_hiql_kin_k5:$W/final/hiql_kin/WHALE_HIQL_KIN_FINAL_5_12345_hiavi.npz"
  "whales_swirl_kin_s2_k5:$W/final/swirl_kin/WHALE_SWIRL_KIN_FINAL_5_12345_S2_naturenet_iter2.npz"
  "whales_swirl_kin_s1_k5:$W/final/swirl_kin/WHALE_SWIRL_KIN_FINAL_5_12345_S1_naturenet_iter2_temp_2.npz"
  "sharks_hiql_kin_k5:$H/final/hiql_kin/SHARK_HIQL_KIN_FINAL_5_12345_hiavi.npz"
  "sharks_swirl_kin_s1_k5:$H/final/swirl_kin/SHARK_SWIRL_KIN_FINAL_5_12345_S1_naturenet_iter2_temp_2.npz"
  "sharks_swirl_kin_s2_k5:$H/final/swirl_kin/SHARK_SWIRL_KIN_FINAL_5_12345_S2_naturenet_iter2.npz"
  "sharks_hiql_kin_k2:$H/final/hiql_kin_k2/SHARK_HIQL_KIN_K2_FINAL_2_12345_hiavi.npz"
)

if want ee; then
  echo "== step 10: exploit / explore"
  for e in "${TOOLS[@]}"; do
    IFS=: read -r lab cfg K d sdir srt proj <<< "$e"
    sp=${lab%%_*}; root=$([ "$sp" = whales ] && echo "$W" || echo "$H")
    ls "$d/plots/$sdir"/*_time_series_pd.* >/dev/null 2>&1 || { echo "SKIP $lab (no series in $d/plots/$sdir)"; continue; }
    sr=(); [ -f "$d/plots/state_reward_$srt.npy" ] && sr=(--state_reward "$d/plots/state_reward_$srt.npy")
    cpu $N.irl.exploit_explore_init --data_dir "$d/plots/$sdir" --slices "$T/${sp}_correlation_final_kin_v2.yaml" \
        "${sr[@]}" --out_dir "$root/final/exploit_explore/$lab"
  done
  for e in "${NPZ[@]}"; do
    lab=${e%%:*}; f=${e#*:}; sp=${lab%%_*}; root=$([ "$sp" = whales ] && echo "$W" || echo "$H")
    have "$f" && cpu $N.irl.exploit_explore_init --model_npz "$f" --out_dir "$root/final/exploit_explore/$lab"
  done
fi

if want project; then
  echo "== step 11: reward-to-environment projection"
  for e in "${TOOLS[@]}"; do
    IFS=: read -r lab cfg K d sdir srt proj <<< "$e"
    have "$d" || continue
    if [[ $cfg == *swirl* ]]; then
      CUDA_VISIBLE_DEVICES="$GPU" $N.irl.reward_env_projection -y "$C/$cfg.yaml" -p "$C/$proj.yaml" --n_hidden "$K"
    else
      cpu $N.irl.reward_env_projection -y "$C/$cfg.yaml" -p "$C/$proj.yaml" --n_hidden "$K"
    fi
  done
fi

if want series; then
  echo "== step 12: time-series analysis"
  for sp in whales sharks; do
    cpu $N.time_series_analysis.correlation_analysis_shark -y "$T/${sp}_correlation_final_kin_v2.yaml"
  done
fi

if want tables; then
  echo "== step 13: manuscript tables"
  for sp in whales sharks; do
    if [ "$sp" = whales ]; then
      D=$W; L=whale_v2_kin_nn; KS=(5)
      SC=("hiql_kin_k5=$D/scales_kin/WHALE_HIQL_SCALES_KIN_K5_scales_summary.csv"
          "swirl_kin_k5=$D/scales_kin/WHALE_SWIRL_SCALES_KIN_scales_summary.csv")
    else
      D=$H; L=hammerhead_v2_kin_nn; KS=(5 2)
      SC=("hiql_kin_k5=$D/scales_kin/HIQL_SCALES_KIN_K5_scales_summary.csv"
          "swirl_kin_k5=$D/scales_kin/SHARK_SWIRL_SCALES_KIN_scales_summary.csv"
          "hiql_kin_k2=$D/scales_kin/HIQL_SCALES_KIN_K2_scales_summary.csv")
    fi
    M=$D/manuscript_kin; mkdir -p "$M"
    args=(--out "$M/ms_${sp}_kin")
    [ -f "$D/ll_comparison_kin_nn/${L}_ll_summary.csv" ] && args+=(--ll_summary "$D/ll_comparison_kin_nn/${L}_ll_summary.csv") \
      || echo "SKIP T1 $sp (run plot_ll_comparison -y $C/${sp}_ll_comparison_kin_nn.yaml)"
    sc=(); for s in "${SC[@]}"; do [ -f "${s#*=}" ] && sc+=("$s") || echo "SKIP T2 row (missing ${s#*=})"; done
    [ ${#sc[@]} -gt 0 ] && args+=(--scales_summary "${sc[@]}")
    if [ -f "$D/multiagent/multiagent_kin/ma_summary.json" ]; then
      cpu $N.irl.plot_multiagent_compare --dirs "$D/multiagent/multiagent_kin" --labels "HIQL+kin" \
          --K "${KS[@]}" --out "$M/ma_compare_kin" && args+=(--ma_compare "$M/ma_compare_kin.csv")
    else echo "SKIP T3 coupling $sp (no $D/multiagent/multiagent_kin/ma_summary.json)"; fi
    [ -f "$D/multiagent_scales_kin/ma_scales_summary.csv" ] && args+=(--ma_scales "$D/multiagent_scales_kin/ma_scales_summary.csv") \
      || echo "SKIP T3 scales $sp (no $D/multiagent_scales_kin/ma_scales_summary.csv)"
    cpu $N.irl.summarize_experiments "${args[@]}"
  done
fi
