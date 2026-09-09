#!/usr/bin/env python3
"""
Unified experiment validation entry point.

Wraps check_experiment_invariants.py for pre-run / post-run checks.

Usage:
  python scripts/validate_experiment.py config/experiments/e01_cwme_sci_main.yml --pre-run
  python scripts/validate_experiment.py /path/to/log/experiment_YYYYMMDD_HHMMSS
"""
from __future__ import annotations

import os
import runpy
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    target = os.path.join(_REPO_ROOT, "scripts", "check_experiment_invariants.py")
    sys.argv[0] = target
    runpy.run_path(target, run_name="__main__")
