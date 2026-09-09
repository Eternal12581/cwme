#!/usr/bin/env python3
"""
Freeze a paired episode schedule to JSON (E2 / continual curves).

All conditions (Backbone / Static-PM / CWME) must read the same SCHEDULE_FILE.

Usage:
  python scripts/freeze_episode_schedule.py config/experiments/e02_cwme_continual_50ep.yml
  python scripts/freeze_episode_schedule.py config/experiments/e02_cwme_continual_50ep.yml \\
      --out artifacts/schedules/e02_continual_50ep.json

Requires ScienceWorld env for variation IDs (run once before formal E2 experiments).
"""
from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml

from envs import create_env
from utils import load_config, load_variation


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


def _variation_pairs(
    tasks: list[str],
    variations: list[int] | dict[str, list[int]],
    *,
    task_outer: bool,
) -> list[tuple[str, int]]:
    """Build pairs while supporting task-local variation indices."""
    if isinstance(variations, dict):
        if task_outer:
            return [
                (task, int(var))
                for task in tasks
                for var in variations.get(task, [])
            ]
        # Pair the same ordinal across tasks. This preserves a fair paired
        # stream even when task-local variation IDs are different.
        lengths = [len(variations.get(task, [])) for task in tasks]
        common = min(lengths, default=0)
        return [
            (task, int(variations[task][ordinal]))
            for ordinal in range(common)
            for task in tasks
        ]
    if task_outer:
        return [(task, int(var)) for task in tasks for var in variations]
    return [(task, int(var)) for var in variations for task in tasks]


def build_interleaved_schedule(
    tasks: list[str],
    variations: list[int] | dict[str, list[int]],
    num_episodes: int,
) -> list[dict]:
    """Variation-ordinal-outer, task-inner schedule."""
    pairs = _variation_pairs(tasks, variations, task_outer=False)
    if not pairs:
        raise ValueError("No (task, variation) pairs — check TASKS and SET/VARIATIONS")
    schedule = []
    for i in range(num_episodes):
        task, var = pairs[i % len(pairs)]
        schedule.append({"episode": i, "task": task, "variation": var})
    return schedule


def build_round_robin_schedule(
    tasks: list[str],
    variations: list[int] | dict[str, list[int]],
    num_episodes: int,
) -> list[dict]:
    """Legacy task-outer order (task × all variations). Prefer build_interleaved_schedule."""
    pairs = _variation_pairs(tasks, variations, task_outer=True)
    if not pairs:
        raise ValueError("No (task, variation) pairs — check TASKS and SET/VARIATIONS")
    schedule = []
    for i in range(num_episodes):
        task, var = pairs[i % len(pairs)]
        schedule.append({"episode": i, "task": task, "variation": var})
    return schedule


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _REPO_ROOT, "config", "experiments", "e02_cwme_continual_50ep.yml",
    )
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)

    out_arg = ""
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == "--out" and i + 1 < len(sys.argv):
            out_arg = sys.argv[i + 1]

    agent_config = _load_yaml(config_path)
    exp = dict(agent_config.get("EXPERIMENT") or {})
    curve = dict(exp.get("CONTINUAL_CURVE") or {})
    num_ep = int(curve.get("NUM_EPISODES", 50) or 50)
    tasks = list(exp.get("TASKS") or [])
    set_name = exp.get("SET", "test_mini")
    explicit_vars = exp.get("VARIATIONS")

    default_out = curve.get("SCHEDULE_FILE") or "artifacts/schedules/e02_continual_50ep.json"
    out_path = out_arg or default_out
    if not os.path.isabs(out_path):
        out_path = os.path.join(_REPO_ROOT, out_path)

    ini_path = os.path.join(_REPO_ROOT, "config", "config.ini")
    base_cfg = load_config(ini_path)
    if base_cfg is None:
        raise SystemExit(f"Missing config.ini: {ini_path}")

    env = create_env(agent_config)
    if explicit_vars is not None:
        # Explicit IDs remain supported, but ScienceWorld variation IDs are
        # task-local.  Validate a shared list against every task instead of
        # silently freezing an invalid schedule that later fails at env.load.
        if isinstance(explicit_vars, dict):
            variations = {
                str(task): [int(v) for v in (explicit_vars.get(task) or [])]
                for task in tasks
            }
            for task in tasks:
                env.load(
                    task,
                    variationIdx=0,
                    simplificationStr=str(exp.get("SIMPLIFICATION", "easy") or "easy"),
                    generateGoldPath=True,
                )
                allowed = {int(v) for v in load_variation(env, set_name)}
                invalid = [v for v in variations[task] if v not in allowed]
                if invalid:
                    raise RuntimeError(
                        f"Explicit VARIATIONS for task {task!r} contain IDs "
                        f"not in {set_name!r}: {invalid}; allowed={sorted(allowed)}"
                    )
            variation_meta = variations
        else:
            shared = [int(v) for v in explicit_vars]
            for task in tasks:
                env.load(
                    task,
                    variationIdx=0,
                    simplificationStr=str(exp.get("SIMPLIFICATION", "easy") or "easy"),
                    generateGoldPath=True,
                )
                allowed = {int(v) for v in load_variation(env, set_name)}
                invalid = [v for v in shared if v not in allowed]
                if invalid:
                    raise RuntimeError(
                        f"Shared VARIATIONS for task {task!r} contain IDs "
                        f"not in {set_name!r}: {invalid}; allowed={sorted(allowed)}. "
                        "Use task-local VARIATIONS mapping or remove VARIATIONS."
                    )
            variations = shared
            variation_meta = shared
    else:
        # ScienceWorld initializes its Java-side variation sets only after a
        # task is loaded. Query each task independently because variationIdx
        # is task-local and need not have the same numeric IDs across tasks.
        variations_by_task: dict[str, list[int]] = {}
        for task in tasks:
            env.load(
                task,
                variationIdx=0,
                simplificationStr=str(exp.get("SIMPLIFICATION", "easy") or "easy"),
                generateGoldPath=True,
            )
            selected = [int(v) for v in load_variation(env, set_name)]
            if not selected:
                raise RuntimeError(
                    f"No {set_name!r} variations found for ScienceWorld task {task!r}"
                )
            variations_by_task[task] = selected
        variations = variations_by_task
        variation_meta = variations_by_task

    schedule_mode = str(curve.get("SCHEDULE_MODE", "interleaved") or "interleaved").strip().lower()
    builder = build_round_robin_schedule if schedule_mode == "round_robin" else build_interleaved_schedule
    schedule = builder(tasks, variations, num_ep)
    payload = {
        "meta": {
            "source_config": os.path.relpath(config_path, _REPO_ROOT),
            "set": set_name,
            "schedule_mode": schedule_mode,
            "num_episodes": num_ep,
            "tasks": tasks,
            "variations": variation_meta,
            "pairs_per_cycle": len(tasks) * (
                len(variations) if isinstance(variations, list)
                else min(len(v) for v in variations.values())
            ),
        },
        "episodes": schedule,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Frozen {len(schedule)} episodes -> {out_path}")
    print(f"  tasks={len(tasks)} variations={variations}")
    print("  Update E2 configs: CONTINUAL_CURVE.SCHEDULE_FILE must point to this file.")


if __name__ == "__main__":
    main()
