#!/usr/bin/env bash
# Multi-agent plan for the v2 kinematic tools (step 14 and its downstream 14m / 14l). Replaces run_multiagent_kfold.sh,
# run_whale_prescales.sh STAGE=coupling and run_shark_swirl_kin.sh STAGE=ma for the multi-agent fits: one plan per
# MA_STAGE with every multi-agent config of both species, on the shared state dir ma_kin_tools, so every part that
# already finished under the older scripts is reused (same commands = same done markers).
#
#   bash scripts/run_multiagent_v2.sh <stage> [species]      species: sharks | whales | "sharks whales" (default)
#   stages, in order:
#     screen      observed leave-one-animal-out CV, no nulls (every config below)
#     plots       figures + summaries of whatever finished (run_available_outputs STAGE=ma: plot_hiavi_multiagent,
#                 initial vs coupling compare, HIQL vs SWIRL compare, scales plots)
#     full        adds the time-shift nulls (reuses the screen's observed parts), then plots
#     power       coupling power analysis (CPU; simulated align coupling, recovered by mixture / align)
#     export      ma_final_K*.npz -> per-animal series CSVs (naturenet.models.irl.export_ma_series)
#     downstream  decision-context similarity (multi-agent vs single-animal) + habitat tests on the multi-agent fits
#     animate     re-render the multi-agent animation jobs (and any other stale ones)
#     levels      multi-agent coupling screen at the better-fitting environment levels (LEVELS_MA_<species>)
#     audit       step 13 tables + the manifest audit
#     gather      zip the multi-agent, drivers and levels groups for upload
#   DRY=1 bash scripts/run_multiagent_v2.sh <stage>   writes plans + launcher dry runs only
set -euo pipefail

# ----------------------------------------------------------------------------- settings
NN_SRC=/home/nlahaye/NatureNet/src
PY=/home/nlahaye/SIT_FUSE/.pixi/envs/default/bin/python
S=/data/nlahaye/NatureNet/launcher                 # launcher root (plans + state dirs)
STATE="$S/ma_kin_tools"                            # shared with run_multiagent_kfold.sh / prescales / swirl_kin
GPUS="0,1,2,3"                                     # SWIRL multi-agent jobs; HIQL multi-agent jobs are CPU-only
CPUS=20                                            # of 48 (level sweeps / finals keep GPU 6 and the other CPUs)
MA_CPUS=4                                          # CPUs per multi-agent job
NULL_CHUNK=2                                       # null replicates per launcher job
GATHER_OUT=/data/nlahaye/NatureNet/gathered
STAGE="${1:-screen}"
SPECIES="${2:-sharks whales}"
DRY="${DRY:-0}"

# decision-scale variants for the multi-agent scales entries (empty = every multiagent.scales_variants entry);
# trimming is decided after all results are in, so the screen runs all of them
VARIANTS_HIQL=""
VARIANTS_SWIRL=""

# species roots / ids
R_sharks=/data/nlahaye/NatureNet/Hammerhead_Out_v2;  UID_sharks=hammerhead_v2
R_whales=/data/nlahaye/NatureNet/Blue_Whale_v2;      UID_whales=SWIRL_WHALE_1517

# multi-agent entries per species: plan_jobs flag | config (config/v2/swirl) | backend | Ks
MA_sharks=(
  "multiagent|sharks_multiagent_kin|hiavi|5,2"                       # HIQL + kin K = 5 primary, K = 2 earlier secondary
  "multiagent|sharks_multiagent_kin_coupling|hiavi|5"                # mixture + align / matrix coupling forms
  "multiagent|sharks_multiagent_swirl_kin|swirl|5"                   # SWIRL S-1 + kin K = 5 secondary
  "multiagent|sharks_multiagent_swirl_kin_coupling|swirl|5"
  "scales_multiagent|sharks_scales_multiagent_kin|hiql_scales|5,2"
  "scales_multiagent|sharks_scales_multiagent_swirl_kin|swirl_scales|5"
)
MA_whales=(
  "multiagent|whales_multiagent_kin|hiavi|5"                         # HIQL + kin K = 5 primary
  "multiagent|whales_multiagent_kin_coupling|hiavi|5"
  "multiagent|whales_multiagent_swirl_kin|swirl|5"                   # SWIRL S-1 (multi-agent SWIRL is S-1; S-2 = scales S1_N2)
  "multiagent|whales_multiagent_swirl_kin_coupling|swirl|5"
  "scales_multiagent|whales_scales_multiagent_kin|hiql_scales|5"
  "scales_multiagent|whales_scales_multiagent_swirl_kin|swirl_scales|5"
)

# series export: <run dir under R/multiagent> | K | reference single-animal series (under R) | fits
EXPORT_sharks=(
  "multiagent_kin|5|final/hiql_kin/plots/series_hiavi_s1|indep temporal proximity"
  "multiagent_kin|2|final/hiql_kin_k2/plots/series_hiavi_s1|indep temporal"
  "multiagent_kin_coupling|5|final/hiql_kin/plots/series_hiavi_s1|indep best"
  "multiagent_swirl_kin|5|final/swirl_kin/plots/series_linear_s1|indep temporal"
  "multiagent_swirl_kin_coupling|5|final/swirl_kin/plots/series_linear_s1|indep best"
)
EXPORT_whales=(
  "multiagent_kin|5|final/hiql_kin/plots/series_hiavi_s1|indep temporal proximity"
  "multiagent_kin_coupling|5|final/hiql_kin/plots/series_hiavi_s1|indep best"
  "multiagent_swirl_kin|5|final/swirl_kin/plots/series_linear_s1|indep temporal"
  "multiagent_swirl_kin_coupling|5|final/swirl_kin/plots/series_linear_s1|indep best"
)

# habitat-test parts on the multi-agent fits (same parts as the single-animal step 11h runs)
HAB_PARTS_sharks="features seasons"
HAB_PARTS_whales="features lags"

# coupling power (stage power): kernel variant, effect sizes, replicates
POWER_VARIANT=temporal
POWER_EFFECTS="0 0.5 1 2 3"
POWER_REPS=2
POWER_HELDOUT=6

# environment levels for the multi-agent coupling screen (stage levels). Sharks: the sweep is complete and Seascape /
# k20 fit best (k12 ties k20). Whales: empty until the whale level and Seascape sweeps finish; regenerating the level
# configs while that sweep runs would reset n_states under the running launcher.
LEVELS_MA_sharks="seascape k20"
LEVELS_MA_whales=""
MA_LEVEL_STAGE=screen                              # screen | full (nulls only for the level(s) kept)

# gather groups (stage gather)
GATHER_GROUPS="multiagent,drivers,levels,audit,launcher"
# ----------------------------------------------------------------------------- end settings

export NN_SRC PY S
cd "$NN_SRC"
C=naturenet/config/v2/swirl
D=naturenet/config/v2/drivers
L=naturenet/config/v2/levels
CPU_ENV=(env JAX_PLATFORMS=cpu NATURENET_JAX_PLATFORM=cpu CUDA_VISIBLE_DEVICES=)
cpu() { "${CPU_ENV[@]}" "$PY" "$@"; }
launch() {   # plan [extra launcher args]
  local plan="$1"; shift
  "$PY" -m sit_fuse.local_launcher run "$plan" --dry-run --gpus "$GPUS" --cpus "$CPUS" "$@" \
    | awk '/WARNING/ && w++ < 5; /^\[/ {n++} END {print n " jobs in the launcher dry run"}'
  [ "$DRY" = "1" ] && return 0
  "$PY" -m sit_fuse.local_launcher run "$plan" --gpus "$GPUS" --cpus "$CPUS" "$@" \
    || echo "WARNING: some jobs failed (python -m sit_fuse.local_launcher status $plan)"
}
ykey() { "$PY" -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d = (d or {}).get(k)
print("" if d is None else d)' "$1" "$2"; }

ma_plan() {   # MA_STAGE: one plan with every entry of the selected species
  local st="$1" opt="" args=() sp e flag cfg be ks var
  case "$st" in screen) opt="@skip_null" ;; full) opt="" ;; *) echo "MA stage must be screen or full" >&2; exit 1 ;; esac
  for sp in $SPECIES; do
    local -n arr="MA_$sp"
    for e in "${arr[@]}"; do
      IFS='|' read -r flag cfg be ks <<< "$e"
      [ -f "$C/$cfg.yaml" ] || { echo "SKIP $cfg (no $C/$cfg.yaml)"; continue; }
      if [ "$flag" = "scales_multiagent" ]; then
        var=""; [ "$be" = "hiql_scales" ] && [ -n "$VARIANTS_HIQL" ] && var="@variants=$VARIANTS_HIQL"
        [ "$be" = "swirl_scales" ] && [ -n "$VARIANTS_SWIRL" ] && var="@variants=$VARIANTS_SWIRL"
        args+=(--scales_multiagent "$C/$cfg.yaml@backends=$be@K=$ks$var$opt")
      else
        args+=(--multiagent "$C/$cfg.yaml@backend=$be@K=$ks$opt")
      fi
    done
  done
  local plan="$S/ma_v2_${st}.yaml"
  [ "$SPECIES" = "sharks whales" ] || plan="$S/ma_v2_${st}_${SPECIES// /_}.yaml"
  "$PY" -m naturenet.models.irl.plan_jobs --out "$plan" --state_dir "$STATE" "${args[@]}" \
    --gpus "$GPUS" --ma_cpus "$MA_CPUS" --null_chunk "$NULL_CHUNK"
  echo "$plan"
}

plots() {
  for sp in $SPECIES; do
    DRY="$DRY" STAGE=ma bash scripts/run_available_outputs_${sp}.sh || echo "WARNING: $sp multi-agent plots incomplete"
  done
}

case "$STAGE" in
  screen|full)
    plan=$(ma_plan "$STAGE" | tail -1)
    launch "$plan"
    [ "$DRY" = "1" ] || plots
    ;;

  plots) plots ;;

  power)
    for sp in $SPECIES; do
      R_var="R_$sp"; R="${!R_var}"
      if [ "$DRY" = "1" ]; then echo "would run coupling_power -y $C/${sp}_multiagent_kin_coupling.yaml"; continue; fi
      # shellcheck disable=SC2086
      cpu -m naturenet.models.irl.coupling_power -y "$C/${sp}_multiagent_kin_coupling.yaml" --K 5 \
        --variant "$POWER_VARIANT" --sim-form align --fit-forms mixture align --effects $POWER_EFFECTS \
        --reps "$POWER_REPS" --max-heldout "$POWER_HELDOUT" --out "$R/multiagent/coupling_power_$POWER_VARIANT" \
        || echo "WARNING: $sp coupling power failed"
    done
    ;;

  export)
    for sp in $SPECIES; do
      R_var="R_$sp"; R="${!R_var}"; U_var="UID_$sp"; U="${!U_var}"
      local_arr="EXPORT_$sp[@]"
      for e in "${!local_arr}"; do
        IFS='|' read -r run K ref fits <<< "$e"
        npz="$R/multiagent/$run/ma_final_K$K.npz"
        [ -f "$npz" ] || { echo "SKIP export $sp $run K=$K (no $npz)"; continue; }
        [ -d "$R/$ref" ] || { echo "SKIP export $sp $run K=$K (no reference series $R/$ref)"; continue; }
        if [ "$DRY" = "1" ]; then echo "would export $npz ($fits)"; continue; fi
        # shellcheck disable=SC2086
        cpu -m naturenet.models.irl.export_ma_series --npz "$npz" --ref_series "$R/$ref" \
          --out "$R/multiagent/$run/series_K$K" --fits $fits --df_run_uid "$U" \
          || echo "WARNING: export failed for $sp $run K=$K"
      done
    done
    ;;

  downstream)
    for sp in $SPECIES; do
      parts_var="HAB_PARTS_$sp"; cases=""
      for p in ${!parts_var}; do cases+="{part: $p}, "; done
      plan="$S/ma_v2_downstream_${sp}.yaml"
      cat > "$plan" <<EOF
# generated by scripts/run_multiagent_v2.sh (STAGE=downstream, $sp); edit the script, not this file
launcher: {state_dir: "$S/ma_v2_downstream_${sp}", gpus: none, cpus: $CPUS, retries: 0}
vars: {py: "$PY", d: "$D"}
jobs:
  - name: decision_context_ma       # 14m: multi-agent vs single-animal decision contexts
    cmd: "{py} -m naturenet.drivers.decision_context_similarity -y {d}/${sp}_decision_context_ma.yaml"
    cpus: 4
    env: {JAX_PLATFORMS: cpu, NATURENET_JAX_PLATFORM: cpu}
  - name: habitat_ma                # 14m: habitat tests by mode on the multi-agent fits (thresholds shared with 11h)
    cmd: "{py} -m naturenet.drivers.habitat_tests -y {d}/${sp}_habitat_tests_ma.yaml --part {part}"
    cases: [${cases%, }]
    cpus: 2
    env: {JAX_PLATFORMS: cpu, NATURENET_JAX_PLATFORM: cpu}
EOF
      echo "wrote $plan"
      launch "$plan" --force          # inputs are re-exported each time; the jobs are cheap
    done
    ;;

  animate)
    DRY="$DRY" bash scripts/rerender_animations_v2.sh render "$SPECIES"
    ;;

  levels)
    for sp in $SPECIES; do
      lv_var="LEVELS_MA_$sp"; levels="${!lv_var}"
      [ -n "$levels" ] || { echo "SKIP levels $sp (LEVELS_MA_$sp is empty)"; continue; }
      R_var="R_$sp"; R="${!R_var}"
      # 1. regenerate the level configs (adds multiagent_kin_coupling.yaml), then restore n_states on every level
      if [ "$DRY" = "1" ]; then "$PY" -m naturenet.environment.make_level_configs -y "$L/${sp}_levels_v2.yaml" --dry_run | grep multiagent
      else
        "$PY" -m naturenet.environment.make_level_configs -y "$L/${sp}_levels_v2.yaml" > /dev/null
        for sy in "$L/$sp"/*/sync.yaml; do
          "$PY" -m naturenet.environment.sync_v2_configs patch -y "$sy" > /dev/null \
            && echo "synced $(basename "$(dirname "$sy")")" || echo "WARNING: sync failed for $sy (level env not built?)"
        done
      fi
      # 2. one plan for the chosen levels
      opt="@skip_null"; [ "$MA_LEVEL_STAGE" = "full" ] && opt=""
      args=(); dirs=("$R/multiagent/multiagent_kin_coupling"); labs=(fine)
      for lv in $levels; do
        f="$L/$sp/$lv/multiagent_kin_coupling.yaml"
        if [ "$DRY" != "1" ] && [ -z "$(ykey "$f" n_states)" ]; then echo "SKIP level $lv (n_states not synced in $f)"; continue; fi
        args+=(--multiagent "$f@backend=hiavi@K=5$opt")
        dirs+=("$R/levels/$lv/multiagent/multiagent_kin_coupling"); labs+=("$lv")
      done
      [ ${#args[@]} -gt 0 ] || { echo "SKIP levels $sp (no level ready)"; continue; }
      plan="$S/ma_v2_levels_${sp}_${MA_LEVEL_STAGE}.yaml"
      "$PY" -m naturenet.models.irl.plan_jobs --out "$plan" --state_dir "$S/ma_v2_levels_${sp}" "${args[@]}" \
        --gpus "$GPUS" --ma_cpus "$MA_CPUS" --null_chunk "$NULL_CHUNK"
      launch "$plan"
      [ "$DRY" = "1" ] && continue
      # 3. plots per level and the coupling gain by level (fine = v2 baseline)
      for lv in $levels; do
        f="$L/$sp/$lv/multiagent_kin_coupling.yaml"
        [ -f "$R/levels/$lv/multiagent/multiagent_kin_coupling/ma_summary.json" ] && \
          cpu -m naturenet.models.irl.plot_hiavi_multiagent -y "$f" --K 5 --subdir multiagent_kin_coupling || true
      done
      ok_dirs=(); ok_labs=()
      for i in "${!dirs[@]}"; do [ -f "${dirs[$i]}/ma_summary.json" ] && { ok_dirs+=("${dirs[$i]}"); ok_labs+=("${labs[$i]}"); }; done
      if [ ${#ok_dirs[@]} -ge 2 ]; then
        cpu -m naturenet.models.irl.plot_multiagent_compare --dirs "${ok_dirs[@]}" --labels "${ok_labs[@]}" --K 5 \
          --out "$R/levels/ma_compare_levels" --B || echo "WARNING: level compare failed"
      else echo "SKIP level compare (needs the fine coupling run and >= 1 level)"; fi
    done
    ;;

  audit)
    for sp in $SPECIES; do
      DRY="$DRY" STAGE=tables bash scripts/run_available_outputs_${sp}.sh || true
      DRY="$DRY" STAGE=audit  bash scripts/run_available_outputs_${sp}.sh || true
    done
    ;;

  gather)
    for sp in $SPECIES; do
      args=(-y "naturenet/config/v2/gather/${sp}_available_outputs.yaml" --vars "NN=$NN_SRC" --out "$GATHER_OUT"
            --only "$GATHER_GROUPS")
      [ "$DRY" = "1" ] && args+=(--dry_run)
      "$PY" -m naturenet.misc.gather_outputs "${args[@]}"
    done
    ;;

  *) echo "unknown stage $STAGE (screen | plots | full | power | export | downstream | animate | levels | audit | gather)" >&2
     exit 1 ;;
esac
echo "done (STAGE=$STAGE, species: $SPECIES)"
