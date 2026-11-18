#!/usr/bin/env bash
# Sharks: every plot, analysis, animation / still the finished finals and the partial scales runs support (no step-9 xai), plus steps 17b, 17c, 20 and 21.
#   bash run_available_outputs_sharks.sh                  # STAGE=all: pre, outputs, habitat, ll, ma, tables, levels, seascape, seascape_xai, step20_post, audit
#   STAGE=pre|outputs|habitat|ll|ma|tables|levels|seascape|seascape_xai|step20_post|audit bash run_available_outputs_sharks.sh
#   DRY=1 (plans + launcher dry runs only), GPUS (default 6), CPUS (default 16), SKIP_LEVELS=1, LEVELS, LEVELS_SWIRL,
#   LEVEL_FINAL + NH (16f), EXPECTED_FOLDS (default 5). Aliases: STAGE=step17 | step20 (sharks) | step21.
#   SKIP_STEP20=1, STEP20_FITS=0 (no shark scales / multi-agent CV), MA_STAGE, REMAKE_XAI=1. See run_available_outputs.sh for the stage details.
SP=sharks exec bash "$(dirname "${BASH_SOURCE[0]}")/run_available_outputs.sh" "$@"
