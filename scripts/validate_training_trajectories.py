#!/usr/bin/env python3
"""Audit exported trajectories before building offline artifacts.

Usage:
  python scripts/validate_training_trajectories.py artifacts/gt/science_train.jsonl
  python scripts/validate_training_trajectories.py artifacts/gt/science_train.jsonl --strict
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.trajectory_quality import trajectory_quality_errors


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        if path.lower().endswith(".json"):
            blob = json.load(fh)
            if isinstance(blob, list):
                return blob
            if isinstance(blob, dict) and isinstance(blob.get("episodes"), list):
                return list(blob["episodes"])
            return [blob]
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit training trajectory integrity")
    parser.add_argument("path")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when any row is not complete and successful")
    args = parser.parse_args()
    path = args.path if os.path.isabs(args.path) else os.path.join(_REPO_ROOT, args.path)
    rows = _load(path)

    failures: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    accepted = 0
    for index, row in enumerate(rows):
        task = str(row.get("task") or row.get("task_id") or "unknown")
        task_counts[task] += 1
        errors = trajectory_quality_errors(row)
        if errors:
            for error in errors:
                failures[error] += 1
            print(
                f"row={index} task={task} variation={row.get('variation_id', '')} "
                f"errors={','.join(errors)}"
            )
        else:
            accepted += 1

    print(f"episodes: {len(rows)}")
    print(f"complete_success: {accepted}")
    print(f"tasks: {dict(sorted(task_counts.items()))}")
    print(f"errors: {dict(sorted(failures.items()))}")
    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

