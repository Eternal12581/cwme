#!/usr/bin/env python3
"""Audit cross-experiment protocol consistency before launching runs.

The audit is intentionally configuration-only.  It catches accidental changes
to the paired E02 schedule, task set, formal grounding boundary, and the
declared E03 single-component ablations before expensive model execution.

Usage:
  python scripts/audit_experiment_configs.py
  python scripts/audit_experiment_configs.py config/experiments/e02_*.yml
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    yaml = None
    _YAML_IMPORT_ERROR = exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grounding.grounding_mode import GroundingMode, resolve_grounding_mode_from_config


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.load(fh, Loader=yaml.FullLoader) or {}


def _path(cfg: dict) -> dict:
    return dict(cfg.get("EXPERIMENT") or {})


def _execution(cfg: dict) -> dict:
    return dict(cfg.get("EXECUTION") or {})


def _tasks(cfg: dict) -> tuple[str, ...]:
    return tuple(str(x).strip().lower() for x in (_path(cfg).get("TASKS") or []))


def _schedule(cfg: dict) -> str:
    curve = dict(_path(cfg).get("CONTINUAL_CURVE") or {})
    return str(curve.get("SCHEDULE_FILE") or "").replace("\\", "/")


def _find(paths: list[Path], stem: str) -> tuple[Path | None, dict | None]:
    for path in paths:
        if path.stem == stem:
            return path, _load(path)
    return None, None


def _check_formal(name: str, cfg: dict) -> list[str]:
    issues: list[str] = []
    exp = _path(cfg)
    exe = _execution(cfg)
    if not bool(exp.get("FORMAL_EXPERIMENT", False)):
        return issues
    mode = resolve_grounding_mode_from_config(cfg)
    if mode != GroundingMode.GROUNDING_ONLY:
        issues.append(f"{name}: formal experiment uses grounding_mode={mode.value}")
    for key in ("task_heuristics_enabled", "legacy_heuristics_enabled", "cwme_heuristic_fallback"):
        if bool(exe.get(key, False)):
            issues.append(f"{name}: formal experiment enables {key}")
    return issues


def _audit_e02(paths: list[Path], errors: list[str], warnings: list[str]) -> None:
    names = (
        "e02_backbone_continual_50ep",
        "e02_static_pm_50ep",
        "e02_cwme_continual_50ep",
    )
    loaded = []
    for name in names:
        path, cfg = _find(paths, name)
        if path is None or cfg is None:
            errors.append(f"E02 missing config: {name}.yml")
            continue
        loaded.append((name, cfg))
        errors.extend(_check_formal(name, cfg))
    if not loaded:
        return

    schedules = {_schedule(cfg) for _, cfg in loaded}
    if len(schedules) != 1 or not next(iter(schedules), ""):
        errors.append(f"E02 schedule mismatch: {sorted(schedules)}")

    task_sets = {_tasks(cfg) for _, cfg in loaded}
    if len(task_sets) != 1:
        errors.append(f"E02 task-set mismatch: {sorted(task_sets)}")

    counts = {
        int(dict(_path(cfg).get("CONTINUAL_CURVE") or {}).get("NUM_EPISODES", 0) or 0)
        for _, cfg in loaded
    }
    if len(counts) != 1 or next(iter(counts), 0) <= 0:
        errors.append(f"E02 episode-count mismatch: {sorted(counts)}")

    # The three rows are a controlled memory comparison.  Differences outside
    # these fields usually indicate that a baseline is no longer comparable.
    expected = {
        "e02_backbone_continual_50ep": (False, False, False, False),
        "e02_static_pm_50ep": (True, True, False, True),
        "e02_cwme_continual_50ep": (True, True, True, True),
    }
    for name, cfg in loaded:
        wm = dict(cfg.get("WORLD_MODEL") or {})
        pm = dict(cfg.get("PROCEDURAL_MEMORY") or {})
        exe = _execution(cfg)
        got = (
            bool(wm.get("historical_kg", False)),
            bool(pm.get("read", False)),
            bool(pm.get("write", False)),
            bool((cfg.get("CONTINUAL") or {}).get("enabled", False)),
        )
        if name in expected and got != expected[name]:
            errors.append(f"{name}: memory protocol={got}, expected={expected[name]}")
        if name == "e02_backbone_continual_50ep" and bool(exe.get("fast_path_enabled", False)):
            errors.append(f"{name}: fast_path_enabled must be false")


def _audit_e03(paths: list[Path], errors: list[str], warnings: list[str]) -> None:
    names = (
        "e03_full", "e03_wo_hkg", "e03_wo_pattern", "e03_wo_writeback",
        "e03_wo_fast_reuse", "e03_wo_verify",
    )
    loaded = {}
    for name in names:
        path, cfg = _find(paths, name)
        if path is None or cfg is None:
            errors.append(f"E03 missing config: {name}.yml")
            continue
        loaded[name] = cfg
        errors.extend(_check_formal(name, cfg))
    full = loaded.get("e03_full")
    if full is None:
        return

    def signature(cfg: dict) -> dict:
        wm = dict(cfg.get("WORLD_MODEL") or {})
        pm = dict(cfg.get("PROCEDURAL_MEMORY") or {})
        exe = _execution(cfg)
        return {
            "hkg": bool(wm.get("historical_kg", False)),
            "pattern_read": bool(pm.get("read", False)),
            "pattern_write": bool(pm.get("write", False)),
            "continual": bool((cfg.get("CONTINUAL") or {}).get("enabled", False)),
            "fast": bool(exe.get("fast_path_enabled", False)),
            "verify": bool(exe.get("pattern_verify_enabled", False)),
            "task_heuristics": bool(exe.get("task_heuristics_enabled", False)),
            "legacy_heuristics": bool(exe.get("legacy_heuristics_enabled", False)),
        }

    base = signature(full)
    expected_changes = {
        "e03_wo_hkg": {"hkg"},
        "e03_wo_fast_reuse": {"fast"},
        "e03_wo_verify": {"verify"},
        # This is deliberately an end-to-end procedural-memory ablation, not
        # a read-only toggle: pattern read/write and online evolution vanish.
        "e03_wo_pattern": {"pattern_read", "pattern_write", "continual", "fast", "verify"},
        "e03_wo_writeback": {"pattern_write"},
    }
    for name, changed in expected_changes.items():
        if name not in loaded:
            continue
        got = signature(loaded[name])
        actual = {key for key in base if got[key] != base[key]}
        if actual != changed:
            errors.append(f"{name}: changed fields={sorted(actual)}, expected={sorted(changed)}")


def main() -> None:
    if yaml is None:
        print(
            "FAIL: PyYAML is required for config auditing; install the runtime "
            "dependencies before running this script."
        )
        return 2
    raw = sys.argv[1:]
    if raw:
        paths = [Path(p) for item in raw for p in glob.glob(item)]
    else:
        paths = sorted((ROOT / "config" / "experiments").glob("e*.yml"))
    paths = [p if p.is_absolute() else ROOT / p for p in paths]
    paths = sorted({p.resolve() for p in paths if p.is_file()})
    if not paths:
        print("FAIL: no experiment configs found")
        return 2

    errors: list[str] = []
    warnings: list[str] = []
    for path in paths:
        try:
            errors.extend(_check_formal(path.stem, _load(path)))
        except Exception as exc:
            errors.append(f"{path.name}: YAML load failed: {exc}")
    _audit_e02(paths, errors, warnings)
    _audit_e03(paths, errors, warnings)

    print(f"Audited {len(paths)} experiment config(s)")
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        print(f"\nConfig audit FAILED ({len(errors)} issue(s))")
        return 1
    print("Config audit PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
