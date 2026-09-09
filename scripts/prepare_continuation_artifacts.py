#!/usr/bin/env python3
"""Prepare immutable training artifacts from a merged continuation run.

The merged trajectory file is kept untouched.  By default only complete,
successful trajectories are copied into ``accepted_training_trajectories``
and used to rebuild H_train and P_train.  A provenance report and hashes are
written next to the derived artifacts so every continuation round is
reproducible.

Usage:
  python scripts/prepare_continuation_artifacts.py \
      --input artifacts/variants/self_evolve/merged/run_01/training_trajectories.jsonl \
      --output-namespace artifacts/variants/self_evolve/bootstrap/run_01 \
      --config config/experiments/self_evolve.yml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trajectory_quality import is_complete_success_trajectory
from scripts.build_historical_kg import build_from_log_trajectories
from scripts.build_procedural_memory import build_from_trajectories


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Trajectory row is not an object: {path}:{line_no}")
            rows.append(row)
    return rows


def _threshold(config_path: Path | None) -> float:
    if config_path is None:
        return 0.65
    import yaml

    with config_path.open(encoding="utf-8") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader) or {}
    return float(
        ((config.get("PROCEDURAL_MEMORY") or {}).get("validation_threshold", 0.65))
        or 0.65
    )


def prepare(
    input_path: Path,
    output_namespace: Path,
    *,
    config_path: Path | None = None,
    success_only: bool = True,
) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"Merged trajectory file not found: {input_path}")

    rows = _load_jsonl(input_path)
    selected = [row for row in rows if is_complete_success_trajectory(row)] if success_only else rows
    if not selected:
        raise RuntimeError(
            "No complete successful trajectories selected. "
            "Do not build continuation memory from this run."
        )

    output_namespace.mkdir(parents=True, exist_ok=True)
    accepted_path = output_namespace / "accepted_training_trajectories.jsonl"
    with accepted_path.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    hkg_dir = output_namespace / "historical_kg"
    pattern_path = output_namespace / "procedural_memory" / "patterns.jsonl.procedural.jsonl"
    build_from_log_trajectories(
        selected,
        repo_root=str(ROOT),
        output_dir=str(hkg_dir),
    )
    build_from_trajectories(
        selected,
        output_path=str(pattern_path),
        validation_threshold=_threshold(config_path),
        success_only=True,
        full_trajectory=False,
    )

    scores = [float(row.get("final_score", row.get("score", 0)) or 0) for row in rows]
    selected_tasks = Counter(str(row.get("task_id") or row.get("task") or "") for row in selected)
    report = {
        "schema_version": 1,
        "kind": "continuation_artifact_preparation",
        "source_trajectory": str(input_path),
        "source_sha256": _sha256(input_path),
        "source_episode_count": len(rows),
        "selected_episode_count": len(selected),
        "source_complete_success_count": sum(is_complete_success_trajectory(row) for row in rows),
        "source_score_ge_100_count": sum(score >= 100 for score in scores),
        "selection": "complete_success_only" if success_only else "all_rows",
        "selected_tasks": dict(sorted(selected_tasks.items())),
        "outputs": {
            "accepted_trajectories": str(accepted_path),
            "hkg": str(hkg_dir / "historical_world_model.json"),
            "pattern": str(pattern_path),
        },
    }
    for key, value in report["outputs"].items():
        path = Path(value)
        if path.is_file():
            report.setdefault("output_sha256", {})[key] = _sha256(path)

    report_path = output_namespace / "preparation_report.json"
    manifest_path = output_namespace / "manifest.json"
    report["manifest"] = str(manifest_path)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    manifest = {
        "schema_version": 1,
        "kind": "continuation_training_artifacts",
        "artifact_namespace": str(output_namespace),
        "source_trajectory": str(input_path),
        "source_sha256": report["source_sha256"],
        "selected_episode_count": len(selected),
        "paths": report["outputs"],
        "preparation_report": str(report_path),
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build provenance-tracked continuation artifacts")
    parser.add_argument("--input", required=True, help="Merged training_trajectories.jsonl")
    parser.add_argument("--output-namespace", required=True)
    parser.add_argument("--config", default="", help="Optional YAML for pattern validation threshold")
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Use all rows; discouraged for procedural memory because failures become training evidence",
    )
    args = parser.parse_args()
    prepare(
        _resolve(args.input),
        _resolve(args.output_namespace),
        config_path=_resolve(args.config) if args.config else None,
        success_only=not args.include_all,
    )


if __name__ == "__main__":
    main()
