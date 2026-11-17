#!/usr/bin/env bash
# Multi-agent K-fold (leave-one-animal-out) CV for the chosen kinematic behavior-analysis tools (v2, *_kin.yaml):
#   whales  HIQL + kin K = 5 (primary)    coupling (whales_multiagent_kin) + decision scales (whales_scales_multiagent_kin)
#           SWIRL S-2 + kin K = 5 (second) decision scales, swirl_scales (whales_scales_multiagent_swirl_kin; multi-agent
#                                         SWIRL is S-1 only, so S1_N2 = order 2 is the S-2 analogue; all 8 variants)
#   sharks  HIQL + kin K = 5 (primary) and K = 2 (earlier secondary)
#                                         coupling (sharks_multiagent_kin) + decision scales (sharks_scales_multiagent_kin)
#           SWIRL S-1 + kin K = 5 (secondary) coupling (sharks_multiagent_swirl_kin + _coupling align copy) and
#                                         decision scales, swirl_scales (sharks_scales_multiagent_swirl_kin)
# Replaces run_multiagent_all.sh / run_multiagent_action_only.sh / run_multiagent_kinematic.sh.
#   bash run_multiagent_kfold.sh                 # STAGE=screen: observed CV only, no nulls (default)
#   STAGE=full bash run_multiagent_kfold.sh      # adds the time-shift nulls; reuses the screen's observed parts
#   STAGE=plots bash run_multiagent_kfold.sh     # figures / summaries only (after screen or full)
#   VARIANTS_HIQL, VARIANTS_SWIRL   comma lists of multiagent.scales_variants names for the HIQL / SWIRL decision-scale
#                entries (default: every variant in the YAML), e.g. a screen of all 8, then nulls for a subset:
#                STAGE=full VARIANTS_HIQL=S1_N1,S2_N1_R2,S3_N1_dur bash run_multiagent_kfold.sh
#   GPUS, CPUS   launcher budget (defaults leave the rest of the 8-GPU / 48-CPU server free; HIQL jobs are CPU-only)
#   DRY=1        print the plan and the launcher dry run only
#   S            launcher plan / state root; NN_SRC  the src/ dir (default: the dir above scripts/); PY  python
# Sync first (bash sync_v2_configs_all.sh). Outputs: <out_dir>/multiagent_kin/, <...>/multiagent_scales_kin/.
set -euo pipefail
STAGE="${STAGE:-screen}"
NN_SRC="${NN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
S="${S:-/data/nlahaye/NatureNet/launcher}"
PY="${PY:-python}"
GPUS="${GPUS:-0,1,2,3}"
CPUS="${CPUS:-20}"
cd "$NN_SRC"
C=naturenet/config/v2/swirl

if [ "$STAGE" = "plots" ]; then
  "$PY" -m naturenet.models.irl.plot_hiavi_multiagent  -y "$C/whales_multiagent_kin.yaml" --K 5
  "$PY" -m naturenet.models.irl.plot_hiavi_multiagent  -y "$C/sharks_multiagent_kin.yaml" --K 5 2
  "$PY" -m naturenet.models.irl.plot_scales_multiagent -y "$C/whales_scales_multiagent_kin.yaml" --K 5
  "$PY" -m naturenet.models.irl.plot_scales_multiagent -y "$C/whales_scales_multiagent_swirl_kin.yaml" --K 5
  for k in 5 2; do
    "$PY" -m naturenet.models.irl.plot_scales_multiagent -y "$C/sharks_scales_multiagent_kin.yaml" --K "$k"
  done
  "$PY" -m naturenet.models.irl.plot_hiavi_multiagent  -y "$C/sharks_multiagent_swirl_kin.yaml" --K 5 --subdir multiagent_swirl_kin || true
  "$PY" -m naturenet.models.irl.plot_scales_multiagent -y "$C/sharks_scales_multiagent_swirl_kin.yaml" --K 5 || true
  exit 0
fi
case "$STAGE" in
  screen) opt="@skip_null" ;;
  full)   opt="" ;;
  *) echo "STAGE must be screen, full or plots" >&2; exit 1 ;;
esac

vh="${VARIANTS_HIQL:+@variants=$VARIANTS_HIQL}"
vs="${VARIANTS_SWIRL:+@variants=$VARIANTS_SWIRL}"
plan="$S/ma_kin_tools_${STAGE}.yaml"
mkdir -p "$S"
# One entry per YAML (shark K = 5 and K = 2 share one entry so their merges into one summary never race).
"$PY" -m naturenet.models.irl.plan_jobs --out "$plan" --state_dir "$S/ma_kin_tools" \
  --multiagent        "$C/whales_multiagent_kin.yaml@backend=hiavi@K=5${opt}" \
  --multiagent        "$C/sharks_multiagent_kin.yaml@backend=hiavi@K=5,2${opt}" \
  --scales_multiagent "$C/whales_scales_multiagent_kin.yaml@backends=hiql_scales@K=5${vh}${opt}" \
  --scales_multiagent "$C/whales_scales_multiagent_swirl_kin.yaml@backends=swirl_scales@K=5${vs}${opt}" \
  --scales_multiagent "$C/sharks_scales_multiagent_kin.yaml@backends=hiql_scales@K=5,2${vh}${opt}" \
  --multiagent        "$C/sharks_multiagent_swirl_kin.yaml@backend=swirl@K=5${opt}" \
  --multiagent        "$C/sharks_multiagent_swirl_kin_coupling.yaml@backend=swirl@K=5${opt}" \
  --scales_multiagent "$C/sharks_scales_multiagent_swirl_kin.yaml@backends=swirl_scales@K=5${vs}${opt}" \
  --gpus "$GPUS" --ma_cpus 4 --null_chunk 2

"$PY" -m sit_fuse.local_launcher run "$plan" --dry-run --gpus "$GPUS" --cpus "$CPUS" \
  | awk '/WARNING/ && w++ < 5; /^\[/ {n++} END {print n " jobs in the launcher dry run"}'
[ "${DRY:-0}" = "1" ] && exit 0
"$PY" -m sit_fuse.local_launcher run "$plan" --gpus "$GPUS" --cpus "$CPUS"
STAGE=plots bash "${BASH_SOURCE[0]}"
