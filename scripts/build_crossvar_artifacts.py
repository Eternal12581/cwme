#!/usr/bin/env python3
"""
Offline E7 cross-variation artifact pipeline.

Train v1-3 trajectories -> H_train_v123 + P_train_v123, then test v4-5 with CWME.

Usage:
  python scripts/build_crossvar_artifacts.py [config/experiments/e07_crossvar_train_v123.yml]

Steps:
  1. Expect trajectories at WORLD_MODEL.trajectory_path (from prep train run)
  2. build_historical_kg.py
  3. build_procedural_memory.py
"""
from __future__ import annotations

import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _REPO_ROOT, "config", "experiments", "e07_crossvar_train_v123.yml",
    )
    if not os.path.isabs(config):
        config = os.path.join(_REPO_ROOT, config)

    py = sys.executable
    steps = [
        ([py, os.path.join(_REPO_ROOT, "scripts", "build_historical_kg.py"), config], "H_train"),
        ([py, os.path.join(_REPO_ROOT, "scripts", "build_procedural_memory.py"), config], "P_train"),
    ]
    for cmd, label in steps:
        print(f"\n=== Building {label} ===")
        rc = subprocess.call(cmd, cwd=_REPO_ROOT)
        if rc != 0:
            raise SystemExit(f"{label} build failed with exit code {rc}")

    print("\nCross-variation artifacts ready.")
    print("E7-A Pure transfer: python ExperimentRunner.py config/experiments/e07_crossvar_static_cwme_v45.yml")
    print("E7-B Continual transfer: python ExperimentRunner.py config/experiments/e07_crossvar_test_cwme_v45.yml")
    print("Backbone control: python ExperimentRunner.py config/experiments/e07_crossvar_test_cloff_v45.yml")


if __name__ == "__main__":
    main()
