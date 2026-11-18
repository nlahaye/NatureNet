#!/usr/bin/env bash
# Whales: gather all run_available_outputs_whales.sh outputs (every stage, whatever exists so far) into per-group zips +
# MANIFEST.csv + SUMMARY.md under /data/nlahaye/NatureNet/gathered/whales_available_outputs_<stamp>/.
#   bash gather_available_outputs_whales.sh                 # all groups
#   DRY=1 bash gather_available_outputs_whales.sh           # per-group counts / sizes only
#   ONLY=finals,scales | SKIP=final_series,animations, SPLIT_MB, MAX_FILE_MB, OUT. See gather_available_outputs.sh.
SP=whales exec bash "$(dirname "${BASH_SOURCE[0]}")/gather_available_outputs.sh" "$@"
