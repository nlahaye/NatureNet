#!/usr/bin/env bash
# Sharks: gather all run_available_outputs_sharks.sh outputs (every stage, whatever exists so far) into per-group zips +
# MANIFEST.csv + SUMMARY.md under /data/nlahaye/NatureNet/gathered/sharks_available_outputs_<stamp>/.
#   bash gather_available_outputs_sharks.sh                 # all groups
#   DRY=1 bash gather_available_outputs_sharks.sh           # per-group counts / sizes only
#   ONLY=finals,scales | SKIP=final_series,animations, SPLIT_MB, MAX_FILE_MB, OUT. See gather_available_outputs.sh.
SP=sharks exec bash "$(dirname "${BASH_SOURCE[0]}")/gather_available_outputs.sh" "$@"
