#!/usr/bin/env bash
# Verify that offline history-log artifacts already exist.
# This check is read-only and never rebuilds HKG/P_train.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "[history-artifacts] ERROR: missing $path" >&2
    return 1
  fi
}

require_pattern() {
  local base="$1"
  if [ -f "$base" ]; then
    return 0
  fi
  require_file "${base}.procedural.jsonl"
}

echo "[history-artifacts] checking frozen offline artifacts (no rebuild)"

# Complete ScienceWorld and ALFWorld historical artifacts.
require_file artifacts/historical_kg/historical_world_model.json
require_pattern artifacts/procedural_memory/patterns.jsonl
require_file artifacts/historical_kg/alf_train/historical_world_model.json
require_pattern artifacts/procedural_memory/alf_patterns.jsonl

# Intentional derived artifacts used by E6/E7 controls.
for pct in 10 25 50 75 100; do
  require_file "artifacts/historical_kg/h_train_${pct}pct/historical_world_model.json"
  require_file "artifacts/manifests/science_hkg_${pct}pct.json"
done
require_file artifacts/historical_kg/crossvar_v123/historical_world_model.json
require_pattern artifacts/procedural_memory/crossvar_v123/patterns.jsonl
require_file artifacts/manifests/science.json
require_file artifacts/manifests/alf.json
require_file artifacts/manifests/crossvar.json
require_file artifacts/schedules/e02_continual_50ep.json

echo "[history-artifacts] all required artifacts are present; experiments will load them read-only"
