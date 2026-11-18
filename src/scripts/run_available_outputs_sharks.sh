#!/usr/bin/env bash
# Sharks: every plot, analysis, animation / still the finished finals and the partial scales runs support (no xai).
#   bash run_available_outputs_sharks.sh                  # STAGE=all: outputs, habitat, ll, ma, tables, levels, seascape, audit
#   STAGE=outputs|habitat|ll|ma|tables|levels|seascape|audit bash run_available_outputs_sharks.sh
#   DRY=1 (plans + launcher dry runs only), GPUS (default 6), CPUS (default 16), SKIP_LEVELS=1, LEVELS, LEVELS_SWIRL,
#   LEVEL_FINAL + NH (16f), EXPECTED_FOLDS (default 5). See run_available_outputs.sh for the stage details.
SP=sharks exec bash "$(dirname "${BASH_SOURCE[0]}")/run_available_outputs.sh" "$@"
