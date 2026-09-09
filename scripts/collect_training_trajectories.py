#!/usr/bin/env python3
"""
Collect training trajectories via ExperimentRunner (CWME-compatible).

Usage:
  python scripts/collect_training_trajectories.py [config.yml]
      [--tasks TASK ...] [--variations ID ...]

Writes JSONL to artifacts/training_trajectories.jsonl (or EXPERIMENT.TRAJECTORY_OUTPUT).
Then build H_train:
  python scripts/build_historical_kg.py [config.yml] artifacts/training_trajectories.jsonl
"""
from __future__ import annotations

import json
import argparse
import os
import subprocess
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml


def _validate_runtime_imports() -> None:
    """Fail before model loading when a runtime module is not importable."""
    from core.reuse_policy_legacy import LegacyReusePolicy  # noqa: F401


def _count_trajectories(path: str) -> int:
    """Count valid JSONL records so an all-failed run cannot look successful."""
    if not os.path.isfile(path):
        return 0
    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                json.loads(line)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CWME trajectory collection with an optional task override"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=os.path.join(
            _REPO_ROOT, "config", "platform", "cwme_hkg_train_collect.yml",
        ),
        help="Experiment YAML (default: config/platform/cwme_hkg_train_collect.yml)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK",
        help="Override EXPERIMENT.TASKS for this run; provide one or more task names",
    )
    parser.add_argument(
        "--variations",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="Override EXPERIMENT.VARIATIONS; useful for a subset of one task's train set",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.load(fh, Loader=yaml.FullLoader)

    _validate_runtime_imports()

    exp = dict(cfg.get("EXPERIMENT") or {})
    if args.tasks:
        exp["TASKS"] = list(args.tasks)
        print(f"Task override: {', '.join(args.tasks)}")
    if args.variations is not None:
        if not args.variations:
            raise ValueError("--variations requires at least one variation id")
        exp["VARIATIONS"] = list(args.variations)
        print(f"Variation override: {', '.join(str(v) for v in args.variations)}")
    exp["COLLECT_TRAJECTORIES"] = True
    exp.setdefault("TRAJECTORY_OUTPUT", "artifacts/training_trajectories.jsonl")
    cfg["EXPERIMENT"] = exp

    print(f"Running trajectory collection with config: {config_path}")
    # The former fixed temp filename was a cross-process race when the two
    # variants were launched together.
    temp_dir = os.path.join(_REPO_ROOT, "artifacts")
    os.makedirs(temp_dir, exist_ok=True)
    fd, tmp_cfg = tempfile.mkstemp(
        prefix=f"_collect_{os.getpid()}_", suffix=".yml", dir=temp_dir,
    )
    os.close(fd)
    try:
        with open(tmp_cfg, "w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)
        subprocess.check_call(
            [sys.executable, os.path.join(_REPO_ROOT, "ExperimentRunner.py"), tmp_cfg],
            cwd=_REPO_ROOT,
        )
    finally:
        try:
            os.unlink(tmp_cfg)
        except FileNotFoundError:
            pass

    out = exp["TRAJECTORY_OUTPUT"]
    if not os.path.isabs(out):
        out = os.path.join(_REPO_ROOT, out)
    trajectory_count = _count_trajectories(out)
    if trajectory_count == 0:
        raise RuntimeError(
            "Trajectory collection produced 0 episodes. Check the ExperimentRunner "
            "errors above; no HKG/P_train artifacts should be built from this run."
        )
    print(f"Trajectories saved to: {out}")
    print(f"Trajectory episodes: {trajectory_count}")
    print("Build H_train with:")
    print(f"  python scripts/build_historical_kg.py {config_path} {out}")


if __name__ == "__main__":
    main()
