#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GT_ARTIFACT_DIR="$ROOT/artifacts/variants/gt_hkg"
SELF_OUTPUT="/mnt/workspace/outputs/cwme_self_evolve"
GT_OUTPUT="/mnt/workspace/outputs/cwme_gt_hkg"
mkdir -p "$GT_ARTIFACT_DIR" "$SELF_OUTPUT" "$GT_OUTPUT"

GT_SOURCE="$GT_ARTIFACT_DIR/gt_source_trajectories.jsonl"
GT_HKG="$GT_ARTIFACT_DIR/historical_kg/historical_world_model.json"

echo "[1/3] Collecting train ground-truth trajectories -> $GT_SOURCE"
python -u scripts/collect_ground_truth_trajectories.py \
  --environment scienceworld \
  --output "$GT_SOURCE" \
  --tasks boil melt freeze change-the-state-of-matter-of use-thermometer \
  --max-steps 100

echo "[2/3] Building filtered GT-HKG -> $GT_HKG"
python -u scripts/build_gt_hkg.py \
  --input "$GT_SOURCE" \
  --output "$GT_HKG"

echo "[3/3] Starting isolated variants in parallel"
python -u scripts/collect_training_trajectories.py \
  config/experiments/self_evolve.yml \
  > "$SELF_OUTPUT/collector.log" 2>&1 &
SELF_PID=$!

python -u scripts/collect_training_trajectories.py \
  config/experiments/gt_hkg_ablation.yml \
  > "$GT_OUTPUT/collector.log" 2>&1 &
GT_PID=$!

echo "self_evolve PID=$SELF_PID log=$SELF_OUTPUT/collector.log"
echo "gt_hkg PID=$GT_PID log=$GT_OUTPUT/collector.log"
wait "$SELF_PID"
wait "$GT_PID"
echo "Both CWME variants completed."
