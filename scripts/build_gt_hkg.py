#!/usr/bin/env python3
"""Build an isolated GT-HKG from executable ground-truth trajectories.

Only complete successful episodes are eligible. Within each episode the HKG
keeps reward/enabling transitions and drops no-op or failed actions. The
original GT JSONL is never modified.

Usage:
  python scripts/build_gt_hkg.py \
      --input artifacts/variants/gt_hkg/gt_source_trajectories.jsonl \
      --output artifacts/variants/gt_hkg/historical_kg/historical_world_model.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trajectory_quality import is_complete_success_trajectory
from core.world_model.historical_kg import HistoricalKG
from scripts.build_historical_kg import build_from_log_trajectories


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"GT trajectory line {line_no} is not an object")
            rows.append(row)
    return rows


def _positive_steps(row: dict) -> list[dict]:
    """Retain only observed reward/enabling transitions for declarative HKG."""
    kept: list[dict] = []
    for step in row.get("steps") or []:
        if not isinstance(step, dict) or step.get("valid") is False:
            continue
        try:
            score_gain = float(step.get("score_after", 0) or 0) > float(
                step.get("score_before", 0) or 0
            )
        except (TypeError, ValueError):
            score_gain = False
        state_change = bool(step.get("meaningful_change", False))
        if not state_change:
            # Do not treat a different prose observation as world progress;
            # only an explicit structured-state transition qualifies here.
            before = step.get("state_signature_before", "")
            after = step.get("state_signature_after", "")
            state_change = bool(before and after and before != after)
        if score_gain or state_change:
            copied = dict(step)
            copied["progress_kind"] = "reward" if score_gain else "enabling"
            copied["valid"] = True
            copied["meaningful_change"] = True
            kept.append(copied)
    return kept


def prepare(input_path: str, output_path: str) -> dict:
    source = Path(input_path).expanduser()
    if not source.is_absolute():
        source = ROOT / source
    target = Path(output_path).expanduser()
    if not target.is_absolute():
        target = ROOT / target
    if target.suffix.lower() != ".json":
        target = target / "historical_world_model.json"
    target = target.resolve()
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GT trajectory source not found: {source}")

    all_rows = _load_jsonl(source)
    selected: list[dict] = []
    dropped_incomplete = 0
    dropped_empty = 0
    for row in all_rows:
        if not is_complete_success_trajectory(row):
            dropped_incomplete += 1
            continue
        steps = _positive_steps(row)
        if not steps:
            dropped_empty += 1
            continue
        copied = dict(row)
        copied["steps"] = steps
        copied["source"] = {
            **dict(row.get("source") or {}),
            "hkg_filter": "complete_success_reward_or_enabling_only",
            "original_step_count": len(row.get("steps") or []),
            "retained_step_count": len(steps),
        }
        selected.append(copied)

    if not selected:
        raise RuntimeError(
            f"GT-HKG preparation retained 0 episodes from {source}; "
            "check that the source contains complete successful replays."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    build_from_log_trajectories(
        selected,
        repo_root=str(ROOT),
        output_dir=str(target.parent),
    )
    kg = HistoricalKG.load(str(target.parent), read_only=False)
    kg.metadata.update({
        "source": "ground_truth_filtered",
        "source_trajectory_path": str(source),
        "hkg_filter": "complete_success_reward_or_enabling_only",
        "source_episode_count": len(all_rows),
        "retained_episode_count": len(selected),
        "dropped_incomplete_episode_count": dropped_incomplete,
        "dropped_empty_episode_count": dropped_empty,
        "positive_transition_count": sum(len(row["steps"]) for row in selected),
    })
    kg.save(str(target.parent))
    report = {
        "source": str(source),
        "output": str(target),
        "source_episode_count": len(all_rows),
        "retained_episode_count": len(selected),
        "dropped_incomplete_episode_count": dropped_incomplete,
        "dropped_empty_episode_count": dropped_empty,
        "positive_transition_count": sum(len(row["steps"]) for row in selected),
        "hkg_edge_count": len(kg),
    }
    report_path = target.parent / "gt_hkg_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build isolated ground-truth HKG")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.input, args.output)


if __name__ == "__main__":
    main()
