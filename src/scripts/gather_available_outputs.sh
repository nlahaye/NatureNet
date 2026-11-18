#!/usr/bin/env bash
# Gather everything run_available_outputs.sh produced for one species (all stages, whatever exists so far) into
# per-group zips + MANIFEST.csv + SUMMARY.md under <out_dir>/<sp>_available_outputs_<stamp>/ (default
# /data/nlahaye/NatureNet/gathered/). Read-only on the run dirs; safe to run while the scales CV is still going.
#   Species wrappers: gather_available_outputs_whales.sh, gather_available_outputs_sharks.sh (SP=whales | sharks).
#
#   SP=whales bash gather_available_outputs.sh              # all groups
#   SP=whales DRY=1 bash gather_available_outputs.sh        # counts / sizes per group, nothing written
#   SP=whales ONLY=finals,scales,animations bash gather_available_outputs.sh
#   SP=whales SKIP=final_series,animations bash gather_available_outputs.sh   # small figures/tables-only bundle
#   Groups: finals final_series exploit_explore time_series decision_context scales animations habitat ll multiagent
#           tables levels seascape seascape_xai sitfuse_hierarchy audit launcher configs
#   Env: SP, NN_SRC, PY, S (launcher root), OUT (out_dir), ONLY, SKIP, SPLIT_MB (default 1900; 0 = one zip per
#        group), MAX_FILE_MB (default 200), STAMP, DRY=1, SPEC (default config/v2/gather/<sp>_available_outputs.yaml).
set -euo pipefail
SP="${SP:?set SP=whales or SP=sharks (or use gather_available_outputs_<sp>.sh)}"
case "$SP" in whales|sharks) ;; *) echo "SP must be whales or sharks" >&2; exit 1 ;; esac
NN_SRC="${NN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
S="${S:-/data/nlahaye/NatureNet/launcher}"
cd "$NN_SRC"
SPEC="${SPEC:-naturenet/config/v2/gather/${SP}_available_outputs.yaml}"

# launcher status of every plan run_available_outputs.sh (and its habitat stage) writes -> gathered with the logs
HAB=whale_prescales_habitat; PRE=whale_prescales
[ "$SP" = sharks ] && { HAB=shark_habitat_tests; PRE=shark_swirl_kin; }
if [ "${DRY:-0}" != "1" ] && [ -d "$S" ]; then
  {
    echo "# launcher status at $(date '+%Y-%m-%d %H:%M:%S') (gather_available_outputs.sh SP=$SP)"
    for p in "$S/available_outputs_${SP}.yaml" "$S/$HAB.yaml" "$S/available_${SP}_levels_env.yaml" \
             "$S/available_${SP}_levels_sweep.yaml" "$S/available_${SP}_seascape_env.yaml" \
             "$S/available_${SP}_seascape_sweep.yaml" "$S/available_${SP}_seascape_xai.yaml" "$S/${PRE}_final.yaml" \
             "$S/shark_swirl_kin_scales.yaml" "$S/shark_swirl_kin_ma_screen.yaml" "$S/shark_swirl_kin_ma_full.yaml"; do
      [ "$SP" = whales ] && [[ "$p" == */shark_* ]] && continue
      [ -f "$p" ] || continue
      echo; echo "== $p"
      "$PY" -m sit_fuse.local_launcher status "$p" 2>&1 || echo "(status failed)"
    done
  } > "$S/gather_status_${SP}.txt"
fi

args=(-y "$SPEC" --vars "S=$S" "NN=$NN_SRC")
[ -n "${OUT:-}" ] && args+=(--out "$OUT")
[ -n "${ONLY:-}" ] && args+=(--only "$ONLY")
[ -n "${SKIP:-}" ] && args+=(--skip "$SKIP")
[ -n "${SPLIT_MB:-}" ] && args+=(--split_mb "$SPLIT_MB")
[ -n "${MAX_FILE_MB:-}" ] && args+=(--max_file_mb "$MAX_FILE_MB")
[ -n "${STAMP:-}" ] && args+=(--stamp "$STAMP")
[ "${DRY:-0}" = "1" ] && args+=(--dry_run)
"$PY" -m naturenet.misc.gather_outputs "${args[@]}"
