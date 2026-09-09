#!/usr/bin/env bash
# CWME ablation matrix + component ablations
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run_one() {
  local cfg="$1"
  echo "========== $(basename "$cfg") =========="
  python ExperimentRunner.py "$cfg"
}

echo "=== Main causal chain: Base -> HKG -> Static PM -> CWME ==="
run_one config/platform/ablation_base.yml
run_one config/platform/ablation_hkg.yml
run_one config/platform/ablation_static_pm.yml
run_one config/platform/ablation_cwme.yml

echo "=== HKG static vs continual ==="
run_one config/platform/ablation_hkg.yml
run_one config/platform/ablation_hkg_continual.yml

echo "=== Static HKG + static PM ==="
run_one config/platform/ablation_static_hkg_pm.yml

echo "=== CWME component ablations ==="
run_one config/platform/ablation_no_revision.yml
run_one config/platform/ablation_no_consolidation.yml
run_one config/platform/ablation_no_episode_guard.yml
run_one config/platform/ablation_no_hkg.yml

echo "Ablation matrix complete."
echo "Summarize: python scripts/evaluate_cwme.py /mnt/workspace/outputs"
