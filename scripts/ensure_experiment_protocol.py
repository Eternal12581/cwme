#!/usr/bin/env python3
"""
Write explicit CWME protocol fields into experiment YAML files (idempotent).

Usage:
  python scripts/ensure_experiment_protocol.py
  python scripts/ensure_experiment_protocol.py --dry-run
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml

from core.cwme_protocol import apply_cwme_protocol_defaults

EXP_DIR = os.path.join(_REPO_ROOT, "config", "experiments")


def _needs_patch(path: str) -> bool:
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    cfg = yaml.load(raw, Loader=yaml.FullLoader) or {}
    before = yaml.dump(cfg, sort_keys=False)
    apply_cwme_protocol_defaults(cfg)
    after = yaml.dump(cfg, sort_keys=False)
    return before != after


def main() -> None:
    dry = "--dry-run" in sys.argv
    changed = 0
    unchanged = 0
    for name in sorted(os.listdir(EXP_DIR)):
        if not name.endswith((".yml", ".yaml")):
            continue
        path = os.path.join(EXP_DIR, name)
        if not _needs_patch(path):
            print(f"unchanged: {name}")
            unchanged += 1
            continue
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.load(fh, Loader=yaml.FullLoader) or {}
        apply_cwme_protocol_defaults(cfg)
        if dry:
            print(f"would patch: {name}")
            changed += 1
            continue
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, sort_keys=False, allow_unicode=True)
        changed += 1
        print(f"patched: {name}")
    print(f"Done ({changed} patched, {unchanged} unchanged).")


if __name__ == "__main__":
    main()
