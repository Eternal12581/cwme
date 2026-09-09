#!/usr/bin/env python3
"""Merge sharded test outputs without rebuilding frozen memory artifacts.

Each input result directory must contain an ``episode_metrics.json`` either at
its root or below ``log/experiment_*``.  The optional trajectory inputs are
merged as JSONL only; HKG and Pattern files are never read or rewritten.

Example:
  python scripts/merge_test_shards.py \
      --input-result /mnt/workspace/outputs/test/run/host_01 \
      --input-result /mnt/workspace/outputs/test/run/host_02 \
      --input-trajectory artifacts/test/run/host_01/test_trajectories.jsonl \
      --input-trajectory artifacts/test/run/host_02/test_trajectories.jsonl \
      --output /mnt/workspace/outputs/test/run/merged \
      --output-trajectory artifacts/test/run/merged/test_trajectories.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_cwme import _collect_episode_rows, _scan_log_root


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_no}")
            rows.append(row)
    return rows


def _slot(row: dict) -> tuple[str, str]:
    task = str(row.get("task") or row.get("task_id") or "").strip().lower()
    variation = str(
        row.get("variation", row.get("variation_id", "<unknown>"))
        or "<unknown>"
    ).strip().lower()
    return task, variation


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _merge_metrics(inputs: list[Path], output: Path) -> tuple[list[dict], dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    input_counts: dict[str, int] = {}

    for result_dir in inputs:
        if not result_dir.is_dir():
            raise FileNotFoundError(f"Test result directory not found: {result_dir}")
        rows = _collect_episode_rows(str(result_dir))
        if not rows:
            raise ValueError(f"No episode metrics found under: {result_dir}")
        input_counts[str(result_dir)] = len(rows)
        occurrences: Counter[tuple[str, str]] = Counter()
        for row in rows:
            slot = _slot(row)
            occurrence = occurrences[slot]
            occurrences[slot] += 1
            key = (*slot, occurrence)
            if key in seen:
                raise ValueError(
                    "Overlapping test shard: "
                    f"task={slot[0]} variation={slot[1]} repeat={occurrence}"
                )
            seen.add(key)
            copied = dict(row)
            copied["global_episode_id"] = len(merged)
            copied["merge_source_result"] = str(result_dir)
            merged.append(copied)

    merged.sort(key=lambda row: int(row.get("global_episode_id", 0) or 0))
    _write_json(output / "episode_metrics.json", merged)
    return merged, {
        "input_result_dirs": [str(path) for path in inputs],
        "input_episode_counts": input_counts,
        "episode_count": len(merged),
        "tasks": sorted({_slot(row)[0] for row in merged if _slot(row)[0]}),
        "variation_ids": sorted(
            {_slot(row)[1] for row in merged if _slot(row)[1] != "<unknown>"}
        ),
    }


def _merge_trajectories(inputs: list[Path], output: Path) -> dict:
    if not inputs:
        return {"input_trajectory_files": [], "episode_count": 0}

    merged: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    counts: dict[str, int] = {}
    for source in inputs:
        if not source.is_file():
            raise FileNotFoundError(f"Test trajectory file not found: {source}")
        rows = _load_jsonl(source)
        counts[str(source)] = len(rows)
        occurrences: Counter[tuple[str, str]] = Counter()
        for row in rows:
            slot = _slot(row)
            occurrence = occurrences[slot]
            occurrences[slot] += 1
            key = (*slot, occurrence)
            if key in seen:
                raise ValueError(
                    "Overlapping test trajectory shard: "
                    f"task={slot[0]} variation={slot[1]} repeat={occurrence}"
                )
            seen.add(key)
            copied = dict(row)
            copied["episode_id"] = len(merged)
            copied["merge_source_file"] = str(source)
            merged.append(copied)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "input_trajectory_files": [str(path) for path in inputs],
        "input_trajectory_counts": counts,
        "episode_count": len(merged),
        "output": str(output),
    }


def merge_test_shards(
    input_results: list[str],
    output: str,
    *,
    input_trajectories: list[str] | None = None,
    output_trajectory: str = "",
) -> dict:
    result_dirs = [_resolve(value) for value in input_results]
    output_dir = _resolve(output)
    if output_dir in result_dirs:
        raise ValueError("Output directory cannot be one of the input result directories")

    rows, report = _merge_metrics(result_dirs, output_dir)
    trajectory_report = _merge_trajectories(
        [_resolve(value) for value in (input_trajectories or [])],
        _resolve(output_trajectory) if output_trajectory else output_dir / "test_trajectories.jsonl",
    )
    stats = _scan_log_root(str(output_dir))
    report.update({"trajectory": trajectory_report, "pooled_metrics": stats})
    _write_json(output_dir / "merge_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge sharded test metrics and trajectories without rebuilding memory"
    )
    parser.add_argument("--input-result", action="append", required=True)
    parser.add_argument("--input-trajectory", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-trajectory", default="")
    args = parser.parse_args()
    merge_test_shards(
        args.input_result,
        args.output,
        input_trajectories=args.input_trajectory,
        output_trajectory=args.output_trajectory,
    )


if __name__ == "__main__":
    main()
