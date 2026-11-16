#!/usr/bin/env bash
# Environmental-driver tests 1-7 + Rep_Taxonomy geometry / synthesis for both species (v2 only), on the local
# launcher. Guide: ENV_DRIVERS.md (repository root).
#   STAGE=setup   bash run_env_drivers.sh   # sync base configs, regenerate level configs, build the null levels,
#                                           # sync every level (needed once, and after any re-clustering)
#   STAGE=check   bash run_env_drivers.sh   # plan + launcher dry run + audit (no fits)
#   STAGE=run     bash run_env_drivers.sh   # plan + launch everything (default); resumable, skips finished work
#   STAGE=analyze bash run_env_drivers.sh   # analysis + geometry jobs only (fits already done)
#   STAGE=audit   bash run_env_drivers.sh   # progress + integrity report, resource accounting (exit 1 if open work)
#   SPECIES="whales" ONLY=fits,geometry GPUS=4,5 CPUS=24 DRY=1 ...
# Env: GPUS (default 4,5), CPUS (default 24 = half of the 48 cores), S (launcher root, default
# /data/nlahaye/NatureNet/launcher), NN_SRC (src/ dir), PY, REPTAXONOMY_SRC (optional), SPECIES (default both).
# The fine-level kinematic baselines / scales / multi-agent CV are run by run_scales_kfold.sh,
# run_multiagent_kfold.sh and the kinematic-baseline plans; the auditor checks they are complete.
set -euo pipefail
STAGE="${STAGE:-run}"
NN_SRC="${NN_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
S="${S:-/data/nlahaye/NatureNet/launcher}"
PY="${PY:-python}"
GPUS="${GPUS:-4,5}"
CPUS="${CPUS:-24}"
SPECIES="${SPECIES:-whales sharks}"
cd "$NN_SRC"
D=naturenet/config/v2/drivers
E=naturenet/config/v2/environment
L=naturenet/config/v2/levels
dry=""; [ "${DRY:-0}" = "1" ] && dry="--dry_run"

if [ "$STAGE" = "setup" ]; then
  for sp in $SPECIES; do
    "$PY" -m naturenet.environment.sync_v2_configs patch -y "$E/${sp}_sync_v2.yaml" $dry
    "$PY" -m naturenet.environment.make_level_configs -y "$L/${sp}_levels_v2.yaml"
    for n in "$L/$sp"/null_*/null.yaml; do
      [ -f "$n" ] || continue
      if [ -n "$dry" ]; then echo "would build $(dirname "$n")"; else "$PY" -m naturenet.drivers.env_null -y "$n"; fi
    done
    for s in "$L/$sp"/*/sync.yaml; do
      m=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['class_mapper'])" "$s")
      if [ -f "$m" ]; then
        "$PY" -m naturenet.environment.sync_v2_configs patch -y "$s" $dry
      else
        echo "skip $(dirname "$s") (level not built yet: $m)"
      fi
    done
  done
  echo "setup done; next: STAGE=check bash $(basename "${BASH_SOURCE[0]}")"
  exit 0
fi

audit() {
  local rc=0
  for sp in $SPECIES; do
    "$PY" -m naturenet.drivers.audit -y "$D/${sp}_drivers.yaml" --plan "$S/plan_drivers_${sp}.yaml" || rc=1
    [ -f "$S/plan_drivers_${sp}.yaml" ] && \
      "$PY" -m naturenet.drivers.resources --plan "$S/plan_drivers_${sp}.yaml" --out "$S/resources/drivers_${sp}" || true
  done
  return $rc
}

if [ "$STAGE" = "audit" ]; then audit; exit $?; fi

case "$STAGE" in
  run|check) ONLY="${ONLY:-fits,geometry,analysis}" ;;
  analyze)   ONLY="${ONLY:-geometry,analysis}" ;;
  *) echo "STAGE must be setup, check, run, analyze or audit" >&2; exit 1 ;;
esac

mkdir -p "$S"
for sp in $SPECIES; do
  plan="$S/plan_drivers_${sp}.yaml"
  "$PY" -m naturenet.drivers.plan -y "$D/${sp}_drivers.yaml" --out "$plan" --only "$ONLY" \
    --state_dir "$S/env_drivers_${sp}" --gpus "$GPUS" --cpus "$CPUS" --python "$PY"
  "$PY" -m sit_fuse.local_launcher run "$plan" --dry-run --gpus "$GPUS" --cpus "$CPUS" | tail -n 40
done
if [ "$STAGE" = "check" ] || [ "${DRY:-0}" = "1" ]; then audit || true; exit 0; fi

# One launcher per species, run one after the other so the two never exceed GPUS / CPUS together.
for sp in $SPECIES; do
  "$PY" -m sit_fuse.local_launcher run "$S/plan_drivers_${sp}.yaml" --gpus "$GPUS" --cpus "$CPUS" || true
done
audit
