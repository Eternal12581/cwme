#!/usr/bin/env python3
"""
Validate CWME experiment invariants (artifact freeze, P_train/H_train isolation, E7 checks).

Usage:
  python scripts/check_experiment_invariants.py /path/to/log/experiment_YYYYMMDD_HHMMSS
  python scripts/check_experiment_invariants.py config/experiments/e01_cwme_sci_main.yml --pre-run
"""
from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml

from core.experiment_protocol import (
    HEURISTIC_DECISION_KEYS,
    aggregate_decision_source_counts,
    build_protocol_snapshot,
    cwme_no_heuristic_mode,
    decision_source_share,
    validate_frozen_artifacts,
)


def _load_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _load_agent_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=yaml.FullLoader) or {}


def _collect_episode_rows(run_dir: str) -> list[dict]:
    metrics_path = os.path.join(run_dir, "episode_metrics.json")
    data = _load_json(metrics_path)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


def _check_heuristics(agent_config: dict) -> tuple[bool, str]:
    if not cwme_no_heuristic_mode(agent_config):
        return True, "heuristics allowed (legacy/backbone execution mode)"
    execution = dict(agent_config.get("EXECUTION") or {})
    task_heur = bool(execution.get("task_heuristics_enabled", False))
    legacy_heur = bool(execution.get("legacy_heuristics_enabled", False))
    if task_heur or legacy_heur:
        return False, "task/legacy heuristics enabled in CWME mode"
    return True, "CWME mode: no task heuristics"


def _check_memory_protocol(protocol: dict) -> list[tuple[str, bool, str]]:
    mem = dict(protocol.get("memory_protocol") or {})
    checks: list[tuple[str, bool, str]] = []

    p_frozen = mem.get("p_train_frozen", protocol.get("p_train_frozen"))
    checks.append(("P_train frozen", bool(p_frozen), f"p_train_frozen={p_frozen}"))

    h_frozen = mem.get("h_train_frozen", protocol.get("h_train_frozen"))
    checks.append(("H_train frozen", bool(h_frozen), f"h_train_frozen={h_frozen}"))

    delta_h = mem.get("delta_hkg_writable", protocol.get("delta_hkg_writable"))
    if delta_h is False:
        checks.append(("ΔH disabled", True, "delta_hkg_writable=false"))
    else:
        checks.append(("ΔH disabled", False, f"delta_hkg_writable={delta_h}"))

    return checks


def _check_evaluation_protocol(agent_config: dict, protocol: dict) -> list[tuple[str, bool, str]]:
    """Validate E1 frozen vs continual protocol separation."""
    role = str(protocol.get("scientific_role") or "")
    eval_proto = str(protocol.get("evaluation_protocol") or "")
    mem = dict(protocol.get("memory_protocol") or {})
    checks: list[tuple[str, bool, str]] = []

    if role == "main_comparison_frozen_evaluation":
        frozen_ok = (
            eval_proto == "frozen_standard_test"
            and not bool(mem.get("delta_pattern_writable", True))
            and not bool((agent_config.get("CONTINUAL") or {}).get("enabled", True))
        )
        checks.append((
            "E1 frozen protocol",
            frozen_ok,
            f"role={role} eval={eval_proto} delta_writable={mem.get('delta_pattern_writable')}",
        ))
    elif role == "main_comparison_continual_deployment":
        continual_ok = (
            eval_proto == "online_continual_deployment"
            and bool(mem.get("delta_pattern_writable", False))
        )
        checks.append((
            "E1 continual protocol",
            continual_ok,
            f"role={role} eval={eval_proto} delta_writable={mem.get('delta_pattern_writable')}",
        ))
    return checks


def check_pre_run(config_path: str) -> int:
    agent_config = _load_agent_config(config_path)
    exp = dict(agent_config.get("EXPERIMENT") or {})
    protocol = build_protocol_snapshot(agent_config, repo_root=_REPO_ROOT, config_yml_path=config_path)
    failures = 0

    print(f"Pre-run validation: {config_path}")
    print(f"  experiment_id={protocol.get('experiment_id')} name={protocol.get('experiment_name')}")
    role = protocol.get("scientific_role") or ""
    mode = protocol.get("execution_mode") or ""
    eval_proto = protocol.get("evaluation_protocol") or ""
    if role:
        print(f"  scientific_role={role} execution_mode={mode} evaluation_protocol={eval_proto}")
    if protocol.get("formal_experiment"):
        print(f"  formal_experiment=True artifact_freeze_enforced={protocol.get('artifact_freeze_enforced')}")

    ok, msg = _check_heuristics(agent_config)
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    failures += 0 if ok else 1

    for label, passed, detail in _check_memory_protocol(protocol):
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {detail}")
        failures += 0 if passed else 1

    for label, passed, detail in _check_evaluation_protocol(agent_config, protocol):
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {detail}")
        failures += 0 if passed else 1

    try:
        validate_frozen_artifacts(agent_config, repo_root=_REPO_ROOT, strict=True)
        print("  [PASS] artifact freeze")
    except RuntimeError as exc:
        enforce = bool((exp.get("ARTIFACT_FREEZE") or {}).get("enforce", exp.get("ENFORCE_ARTIFACT_FREEZE", False)))
        if enforce:
            print(f"  [FAIL] artifact freeze: {exc}")
            failures += 1
        else:
            print("  [SKIP] artifact freeze (ENFORCE_ARTIFACT_FREEZE not enabled)")

    schedule_hash = protocol.get("schedule_hash")
    if schedule_hash:
        print(f"  [PASS] schedule hash recorded ({schedule_hash[:20]}...)")
    elif exp.get("CONTINUAL_CURVE"):
        print("  [WARN] continual curve configured but no schedule hash yet")

    return failures


def check_post_run(run_dir: str) -> int:
    protocol_path = os.path.join(run_dir, "protocol.json")
    protocol = _load_json(protocol_path) or {}
    config_path = os.path.join(run_dir, "config.yml")
    agent_config: dict = {}
    if os.path.isfile(config_path):
        agent_config = _load_agent_config(config_path)
    rows = _collect_episode_rows(run_dir)
    failures = 0

    print(f"Post-run validation: {run_dir}")
    if not rows:
        print("  [FAIL] episode_metrics.json missing or empty")
        return 1

    final = rows[-1]
    exp_id = str(protocol.get("experiment_id") or (agent_config.get("EXPERIMENT") or {}).get("EXPERIMENT_ID", ""))

    for label, passed, detail in _check_memory_protocol(protocol):
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {detail}")
        failures += 0 if passed else 1

    train_start = int(rows[0].get("train_pattern_count", 0) or 0)
    train_end = int(final.get("train_pattern_count", 0) or 0)
    train_ok = train_start == train_end
    print(f"  [{'PASS' if train_ok else 'FAIL'}] P_train unchanged: {train_start} -> {train_end}")
    failures += 0 if train_ok else 1

    delta_end = int(final.get("delta_pattern_count", 0) or 0)
    exp_cfg = dict(agent_config.get("EXPERIMENT") or {})
    assert_no_writeback = bool(
        exp_cfg.get("ASSERT_NO_TEST_WRITEBACK")
        or exp_id == "E7-A"
    )
    if assert_no_writeback:
        ok = delta_end == 0
        print(f"  [{'PASS' if ok else 'FAIL'}] E7-A ΔP=0: final_delta_pattern_count={delta_end}")
        failures += 0 if ok else 1

    delta_writable = bool((protocol.get("memory_protocol") or {}).get("delta_pattern_writable", False))
    if delta_writable and (exp_id.startswith("E2") or exp_id == "E3"):
        has_growth = any(int(r.get("delta_pattern_count", 0) or 0) > 0 for r in rows)
        print(f"  [{'PASS' if has_growth else 'WARN'}] continual ΔP observed: max={max(int(r.get('delta_pattern_count', 0) or 0) for r in rows)}")

    ok, msg = _check_heuristics(agent_config)
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    failures += 0 if ok else 1

    if cwme_no_heuristic_mode(agent_config):
        counts = aggregate_decision_source_counts(rows)
        heur_share = decision_source_share(counts, set(HEURISTIC_DECISION_KEYS))
        cog_share = decision_source_share(counts, {"cognition"})
        # Formal grounding-only runs are causal ablations.  Any task-rule
        # decision, even a small fraction, changes the compared policy and
        # must fail validation rather than being hidden by a tolerance.
        heur_ok = heur_share == 0.0
        cog_ok = cog_share >= 0.05 or heur_share <= 0.02
        print(
            f"  [{'PASS' if heur_ok else 'FAIL'}] CWME heuristic share: {heur_share:.2%} "
            f"(cognition={cog_share:.2%})"
        )
        failures += 0 if heur_ok else 1
        if not cog_ok and heur_ok:
            print(f"  [WARN] cognition decision share low ({cog_share:.2%})")

    return failures


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    pre_run = "--pre-run" in sys.argv
    if not args:
        print("Usage: check_experiment_invariants.py <run_dir|config.yml> [--pre-run]")
        sys.exit(2)

    target = args[0]
    if not os.path.isabs(target):
        target = os.path.join(_REPO_ROOT, target)

    if pre_run or target.endswith((".yml", ".yaml")):
        failures = check_pre_run(target)
    else:
        failures = check_post_run(target)

    if failures:
        print(f"\nFAILED: {failures} invariant(s) violated")
        sys.exit(1)
    print("\nAll checked invariants passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
