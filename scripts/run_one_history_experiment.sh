#!/usr/bin/env bash
# Run exactly one experiment against already-built history-log artifacts.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/run_one_history_experiment.sh config/experiments/<config>.yml" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CFG="$1"
if [ ! -f "$CFG" ]; then
  echo "[one-history-run] ERROR: config not found: $CFG" >&2
  exit 2
fi

# This is intentionally a check, not a build. Build artifacts once with
# scripts/build_history_log_artifacts.sh before invoking this entry point.
bash scripts/check_history_log_artifacts.sh

JOB="$(basename "$CFG")"
JOB="${JOB%.yml}"
JOB="${JOB%.yaml}"

PGDATA="${PGDATA_ROOT:-/mnt/workspace/pg_store}/$JOB" \
DB_EXPORT_DIR="${DB_ROOT:-/mnt/workspace/db}/$JOB" \
DAVIS_CONFIG="$CFG" \
bash run.sh
