#!/usr/bin/env python3
"""Merge independent worker trajectories and rebuild derived memory artifacts.

Workers must never write the same trajectory, Pattern, or HKG file.  This
command is the synchronization point: it renumbers episodes, checks for
overlapping task/variation slots, then rebuilds HKG and Pattern from the
merged trajectory source.

Examples:
  python scripts/merge_variant_artifacts.py \
      --input-trajectories artifacts/variants/self_evolve/distributed/run_01/workers/worker_01/training_trajectories.jsonl \
      --input-trajectories artifacts/variants/self_evolve/distributed/run_01/workers/worker_02/training_trajectories.jsonl \
      --output-namespace artifacts/variants/self_evolve/merged/run_01

Use ``--variant gt_hkg`` with ``--input-hkg-trajectories`` when the HKG source
is separate from the worker agent trajectories. The agent trajectories are
always used to rebuild the online Pattern artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_gt_hkg import prepare as build_gt_hkg
from scripts.build_historical_kg import build_from_log_trajectories


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


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
                raise ValueError(f"Trajectory row is not an object: {path}:{line_no}")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _slot(row: dict) -> tuple[str, str]:
    task = str(row.get("task_id") or row.get("task") or "").strip().lower()
    variation = str(
        row.get("variation_id", row.get("variation", "<unknown>"))
        or "<unknown>"
    ).strip().lower()
    return task, variation


def merge_trajectories(inputs: list[Path], output: Path) -> tuple[list[dict], dict]:
    resolved_inputs = [path.resolve() for path in inputs]
    if len(resolved_inputs) != len(set(resolved_inputs)):
        raise ValueError("Duplicate --input-trajectories paths are not allowed")
    if output.resolve() in set(resolved_inputs):
        raise ValueError(f"Output trajectory would overwrite an input: {output}")

    merged: list[dict] = []
    seen_slots: dict[tuple[str, str, int], Path] = {}
    source_counts: dict[tuple[Path, tuple[str, str]], int] = defaultdict(int)
    source_episode_ids: set[tuple[Path, str]] = set()

    for source in resolved_inputs:
        if not source.is_file():
            raise FileNotFoundError(f"Input trajectory file not found: {source}")
        for row in _load_jsonl(source):
            raw_id = str(row.get("episode_id", len(merged)))
            source_key = (source, raw_id)
            if source_key in source_episode_ids:
                raise ValueError(f"Duplicate episode_id in input file {source}: {raw_id}")
            source_episode_ids.add(source_key)

            slot = _slot(row)
            source_slot = (source, slot)
            repeat_index = source_counts[source_slot]
            source_counts[source_slot] += 1
            slot_key = (*slot, repeat_index)
            previous = seen_slots.get(slot_key)
            if previous is not None:
                raise ValueError(
                    "Overlapping task/variation episode across workers: "
                    f"{slot[0]}@{slot[1]} repeat={repeat_index} in {previous} and {source}. "
                    "Use disjoint task/variation assignments or merge intentionally in one file."
                )
            seen_slots[slot_key] = source

            copied = dict(row)
            copied["episode_id"] = len(merged)
            copied["source"] = {
                **dict(row.get("source") or {}),
                "merge_source_file": str(source),
                "merge_source_episode_id": raw_id,
                "merge_repeat_index": repeat_index,
            }
            if row.get("worker_id"):
                copied["source"]["merge_source_worker_id"] = row["worker_id"]
            merged.append(copied)

    if not merged:
        raise ValueError("No trajectory rows found in the input files")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as fh:
            for row in merged:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    return merged, {
        "input_files": [str(path) for path in resolved_inputs],
        "input_sha256": {str(path): _sha256(path) for path in resolved_inputs},
        "episode_count": len(merged),
        "tasks": sorted({row["task"] for row in merged if row.get("task")}),
        "variation_ids": sorted({str(row["variation_id"]) for row in merged if row.get("variation_id")}),
    }


def _threshold_from_config(config_path: str) -> float:
    if not config_path:
        return 0.65
    import yaml

    path = _resolve(config_path)
    with path.open(encoding="utf-8") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader) or {}
    return float(
        ((config.get("PROCEDURAL_MEMORY") or {}).get("validation_threshold", 0.65))
        or 0.65
    )


def merge_and_rebuild(
    inputs: list[str],
    output_namespace: str,
    *,
    variant: str = "self_evolve",
    hkg_inputs: list[str] | None = None,
    config_path: str = "",
    success_only: bool = False,
    full_trajectory: bool = False,
) -> dict:
    if variant not in {"self_evolve", "gt_hkg"}:
        raise ValueError(f"Unsupported variant: {variant}")
    input_paths = [_resolve(value) for value in inputs]
    namespace = _resolve(output_namespace)
    trajectory_path = namespace / "training_trajectories.jsonl"
    merged, source_report = merge_trajectories(input_paths, trajectory_path)

    hkg_dir = namespace / "historical_kg"
    hkg_path = hkg_dir / "historical_world_model.json"
    hkg_source_report = None
    if variant == "gt_hkg":
        # GT-HKG and online Pattern evolution intentionally consume different
        # sources. Fall back to agent trajectories only for the simple case
        # where callers explicitly use the same file for both.
        hkg_paths = hkg_inputs or inputs
        hkg_source_path = namespace / "gt_hkg_source_trajectories.jsonl"
        _, hkg_source_report = merge_trajectories(
            [_resolve(value) for value in hkg_paths],
            hkg_source_path,
        )
        hkg_report = build_gt_hkg(str(hkg_source_path), str(hkg_path))
    else:
        build_from_log_trajectories(
            merged,
            repo_root=str(ROOT),
            output_dir=str(hkg_dir),
        )
        hkg_report = {
            "hkg_edge_count": sum(
                1 for _ in json.loads(hkg_path.read_text(encoding="utf-8")).get("edges", {})
            )
            if hkg_path.is_file()
            else 0,
            "source": "merged_trajectories",
        }

    pattern_path = namespace / "procedural_memory" / "patterns.jsonl.procedural.jsonl"
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    from scripts.build_procedural_memory import build_from_trajectories

    pattern_threshold = _threshold_from_config(config_path)
    build_from_trajectories(
        merged,
        output_path=str(pattern_path),
        validation_threshold=pattern_threshold,
        success_only=success_only,
        full_trajectory=full_trajectory,
    )

    report = {
        "schema_version": 1,
        "variant": variant,
        "output_namespace": str(namespace),
        "outputs": {
            "trajectory": str(trajectory_path),
            "hkg": str(hkg_path),
            "pattern": str(pattern_path),
        },
        "source": source_report,
        "hkg": hkg_report,
        "pattern": {
            "validation_threshold": pattern_threshold,
            "success_only": success_only,
            "full_trajectory": full_trajectory,
        },
    }
    if hkg_source_report is not None:
        report["hkg_source"] = hkg_source_report
        report["outputs"]["hkg_source"] = str(hkg_source_path)
    report_path = namespace / "merge_report.json"
    manifest_path = namespace / "manifest.json"
    report["manifest"] = str(manifest_path)

    manifest = {
        "schema_version": 1,
        "kind": "merged_variant_artifacts",
        "variant": variant,
        "artifact_namespace": str(namespace),
        "paths": report["outputs"],
        "source_trajectory_files": source_report["input_files"],
        "source_episode_count": source_report["episode_count"],
        "merge_report": str(report_path),
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge worker trajectories and rebuild HKG/Pattern")
    parser.add_argument("--input-trajectories", action="append", required=True)
    parser.add_argument("--output-namespace", required=True)
    parser.add_argument("--variant", choices=("self_evolve", "gt_hkg"), default="self_evolve")
    parser.add_argument(
        "--input-hkg-trajectories",
        action="append",
        default=None,
        help="GT trajectory files used only to rebuild HKG (repeat the option per file)",
    )
    parser.add_argument("--config", default="", help="Optional YAML for Pattern validation_threshold")
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--full-trajectory", action="store_true")
    args = parser.parse_args()
    merge_and_rebuild(
        args.input_trajectories,
        args.output_namespace,
        variant=args.variant,
        hkg_inputs=args.input_hkg_trajectories,
        config_path=args.config,
        success_only=args.success_only,
        full_trajectory=args.full_trajectory,
    )


if __name__ == "__main__":
    main()
