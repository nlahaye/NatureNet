#!/usr/bin/env bash
# Re-render only the animations whose inputs changed (step 8b), for any rerun upstream of them:
#   series export (step 8: panel path, P(exploit) trace), exploit / explore tables (step 10: labels_csv, class colours),
#   multi-agent finals (step 14: ma_final_K*.npz) and the exported multi-agent series (14m).
# The outputs plan stamps non-scales animation jobs with nf "0", so a plain rerun keeps their done markers; here
# naturenet.models.irl.animation_staleness compares each job's inputs with its rendered files and only the stale /
# never-rendered jobs are forced.
#   bash scripts/rerender_animations_v2.sh snapshot          # BEFORE rerunning step 10 (saves the mode -> class maps)
#   bash scripts/rerender_animations_v2.sh check  [species]  # report only
#   bash scripts/rerender_animations_v2.sh render [species]  # report, then force the stale + missing jobs
# With a snapshot, a labels table that was rewritten with the same classes does not trigger a re-render.
set -euo pipefail

# ----------------------------------------------------------------------------- settings
NN_SRC=/home/nlahaye/NatureNet/src
PY=/home/nlahaye/SIT_FUSE/.pixi/envs/default/bin/python
S=/data/nlahaye/NatureNet/launcher                 # launcher root (plans + state dirs)
GPUS="6,"                                          # animations are CPU jobs; one GPU id needs the trailing comma
CPUS=12                                            # of 48 (2 per animation job)
SNAP=/data/nlahaye/NatureNet/launcher/animation_labels_snapshot   # labels snapshot dir (per species below)
ACTION="${1:-check}"
SPECIES="${2:-sharks whales}"
DRY="${DRY:-0}"                                    # 1: write the plan and do the launcher dry run only
# ----------------------------------------------------------------------------- end settings

export NN_SRC PY S GPUS CPUS
cd "$NN_SRC"
P=naturenet/config/v2/plots
for sp in $SPECIES; do
  echo "================ $sp ($ACTION)"
  base="$P/${sp}_animations.yaml"
  auto="$S/available_outputs_${sp}_auto_animations.yaml"     # written by the DRY=1 outputs plan below
  plan="$S/available_outputs_${sp}.yaml"
  if [ "$ACTION" = "snapshot" ]; then
    ys=("$base"); [ -f "$auto" ] && ys+=("$auto")
    "$PY" -m naturenet.models.irl.animation_staleness -y "${ys[@]}" --snapshot "$SNAP/$sp"
    continue
  fi
  # 1. write the outputs plan + the generated (auto) animation jobs; nothing runs with DRY=1
  DRY=1 STAGE=outputs bash scripts/run_available_outputs_${sp}.sh > /dev/null
  [ -f "$plan" ] || { echo "ERROR: $plan not written"; exit 1; }
  since=(); [ -f "$SNAP/$sp/labels_snapshot.json" ] && since=(--since "$SNAP/$sp")
  ys=("$base:animate"); [ -f "$auto" ] && ys+=("$auto:animate_auto")
  # 2. report
  "$PY" -m naturenet.models.irl.animation_staleness -y "${ys[@]}" "${since[@]}" --csv "$S/animation_staleness_${sp}.csv"
  [ "$ACTION" = "render" ] || continue
  read -r -a pats <<< "$("$PY" -m naturenet.models.irl.animation_staleness -y "${ys[@]}" "${since[@]}" --print only)"
  if [ ${#pats[@]} -eq 0 ]; then echo "$sp: every animation is current"; continue; fi
  echo "$sp: re-rendering ${#pats[@]} jobs"
  # 3. force only those jobs
  "$PY" -m sit_fuse.local_launcher run "$plan" --gpus "$GPUS" --cpus "$CPUS" --only "${pats[@]}" --force --dry-run | tail -3
  [ "$DRY" = "1" ] && continue
  "$PY" -m sit_fuse.local_launcher run "$plan" --gpus "$GPUS" --cpus "$CPUS" --only "${pats[@]}" --force \
    || echo "WARNING: some $sp animation jobs failed (python -m sit_fuse.local_launcher status $plan)"
done
