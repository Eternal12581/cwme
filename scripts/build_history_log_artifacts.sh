#!/usr/bin/env bash
# Build frozen H_train/P_train artifacts from existing text logs only.
# This script never invokes ExperimentRunner.py and therefore does not train.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
SCI_LOG="${SCI_HISTORY_LOG:-log/sci_7_a_on/result.log}"
ALF_LOG="${ALF_HISTORY_LOG:-log/alf_7_seen_27b/result.log}"

mkdir -p artifacts/legacy_logs artifacts/manifests

SCI_TRAJ=artifacts/legacy_logs/science_history.jsonl
ALF_TRAJ=artifacts/legacy_logs/alf_history.jsonl
CV_TRAJ=artifacts/legacy_logs/science_crossvar_history.jsonl

echo "=== Convert ScienceWorld historical log ==="
"$PYTHON" scripts/convert_legacy_logs_to_trajectories.py \
  --input "$SCI_LOG" --output "$SCI_TRAJ"

echo "=== Build ScienceWorld H_train/P_train ==="
"$PYTHON" scripts/build_historical_kg.py \
  config/experiments/e01_frozen_static_pm.yml "$SCI_TRAJ"
"$PYTHON" scripts/build_procedural_memory.py \
  config/experiments/e01_frozen_static_pm.yml "$SCI_TRAJ"

echo "=== Build HKG coverage sweep artifacts from the same log ==="
for cfg in \
  e06_hkg_train10pct.yml \
  e06_hkg_train25pct.yml \
  e06_hkg_train50pct.yml \
  e06_hkg_train75pct.yml \
  e06_hkg_train100pct.yml
do
  "$PYTHON" scripts/build_historical_kg.py "config/experiments/$cfg" "$SCI_TRAJ"
done

echo "=== Convert ALFWorld historical log ==="
"$PYTHON" scripts/convert_legacy_logs_to_trajectories.py \
  --input "$ALF_LOG" --output "$ALF_TRAJ"

echo "=== Build ALFWorld H_train/P_train ==="
"$PYTHON" scripts/build_historical_kg.py \
  config/experiments/e01_cwme_alf_unseen.yml "$ALF_TRAJ"
"$PYTHON" scripts/build_procedural_memory.py \
  config/experiments/e01_cwme_alf_unseen.yml "$ALF_TRAJ"

echo "=== Build E7 historical transfer artifacts ==="
"$PYTHON" scripts/convert_legacy_logs_to_trajectories.py \
  --input "$SCI_LOG" \
  --task change-the-state-of-matter-of \
  --output "$CV_TRAJ"
"$PYTHON" scripts/build_historical_kg.py \
  config/experiments/e07_crossvar_train_v123.yml "$CV_TRAJ"
"$PYTHON" scripts/build_procedural_memory.py \
  config/experiments/e07_crossvar_train_v123.yml "$CV_TRAJ"

echo "=== Freeze independent manifests ==="
"$PYTHON" scripts/freeze_artifacts.py --strict \
  --hkg artifacts/historical_kg/alf_train/historical_world_model.json \
  --patterns artifacts/procedural_memory/alf_patterns.jsonl \
  --manifest artifacts/manifests/alf.json
"$PYTHON" scripts/freeze_artifacts.py --strict \
  --hkg artifacts/historical_kg/crossvar_v123/historical_world_model.json \
  --patterns artifacts/procedural_memory/crossvar_v123/patterns.jsonl \
  --manifest artifacts/manifests/crossvar.json
for pct in 10 25 50 75 100
do
  "$PYTHON" scripts/freeze_artifacts.py --strict \
    --hkg "artifacts/historical_kg/h_train_${pct}pct/historical_world_model.json" \
    --patterns artifacts/procedural_memory/patterns.jsonl \
    --manifest "artifacts/manifests/science_hkg_${pct}pct.json"
done
"$PYTHON" scripts/freeze_artifacts.py --strict \
  --manifest artifacts/manifests/science.json

echo "History-log artifact build complete. No training was run."
