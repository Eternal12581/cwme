#!/usr/bin/env python3
"""
Build procedural memory (P_train) from training trajectories.

Usage:
  python scripts/build_procedural_memory.py [config.yml] [trajectories.jsonl]
  python scripts/build_procedural_memory.py [config.yml] [trajectories.jsonl] --success-only
  python scripts/build_procedural_memory.py [config.yml] [trajectories.jsonl] --success-only --full-trajectory

Output:
  {persist_path}.procedural.jsonl  (same format as online CWME)

Trajectory JSONL format matches build_historical_kg.py.
"""
from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml

from core.evolution.transition_tracker import TransitionRecord
from core.pattern_library import PatternLibrary
from core.procedural.pattern_extractor import PatternExtractor
from core.procedural.pattern_validator import PatternValidator
from core.procedural.pattern_consolidator import PatternConsolidator
from core.procedural.pattern_normalizer import task_signature
from core.trajectory_quality import is_complete_success_trajectory


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


def _load_trajectories(path: str) -> list[dict]:
    trajectories: list[dict] = []
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        if isinstance(blob, list):
            return blob
        if isinstance(blob, dict) and "episodes" in blob:
            return list(blob["episodes"])
        return [blob]

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    return trajectories


def _resolve_output_path(agent_config: dict, repo_root: str) -> str:
    pm = dict(agent_config.get("PROCEDURAL_MEMORY") or {})
    if pm.get("persist_path"):
        base = str(pm["persist_path"])
    elif pm.get("load_path"):
        base = str(pm["load_path"])
    else:
        # Match the P_train path consumed by formal experiment configs and
        # freeze_artifacts.py.
        base = os.path.join(repo_root, "artifacts", "procedural_memory", "patterns.jsonl")
    if not base.endswith(".procedural.jsonl"):
        base = f"{base}.procedural.jsonl"
    if not os.path.isabs(base):
        base = os.path.join(repo_root, base)
    return base


def build_from_trajectories(
    trajectories: list[dict],
    *,
    output_path: str,
    validation_threshold: float = 0.65,
    success_only: bool = False,
    full_trajectory: bool = False,
) -> str:
    # Offline reconstruction must be a fresh artifact. Passing persist_path to
    # PatternLibrary here would auto-load an older artifact and silently mix
    # histories when this command is rerun with a different log.
    persist_path = output_path.replace(".procedural.jsonl", ".jsonl")
    plib = PatternLibrary(persist_path=None)
    extractor = PatternExtractor()
    validator = PatternValidator(threshold=validation_threshold)
    consolidator = PatternConsolidator()
    stored = 0

    for i, row in enumerate(trajectories):
        eid = int(row.get("episode_id", i))
        task = row.get("task", "") or ""
        task_id = (row.get("task_id", "") or "").strip().lower()
        final_score = float(row.get("final_score", 0) or 0)
        success = bool(row.get("success", False)) or final_score >= 100
        if success_only and not is_complete_success_trajectory(row):
            continue

        trajectory: list[TransitionRecord] = []
        for j, step in enumerate(row.get("steps", [])):
            trajectory.append(
                TransitionRecord(
                    episode_id=eid,
                    step=int(step.get("step", j)),
                    state_before=step.get(
                        "state_before",
                        step.get("observation_before", step.get("state_signature_before", "")),
                    ),
                    action=step.get("action", ""),
                    state_after=step.get(
                        "state_after",
                        step.get("observation_after", step.get("state_signature_after", "")),
                    ),
                    score_before=float(step.get("score_before", 0)),
                    score_after=float(step.get("score_after", 0)),
                    valid=bool(step.get("valid", True)),
                    meaningful_change=bool(step.get("meaningful_change", False)),
                    state_signature_before=step.get("state_signature_before", "") or "",
                    state_signature_after=step.get("state_signature_after", "") or "",
                    score_delta=float(step.get("score_delta", 0) or 0),
                    progress_kind=step.get("progress_kind", "") or "",
                    effect_signature=step.get("effect_signature", "") or "",
                )
            )

        candidate = extractor.extract(
            trajectory=trajectory,
            task=task,
            task_id=task_id,
            historical_kg=None,
            score=final_score,
            episode_id=eid,
            observation=(trajectory[-1].state_after if trajectory else ""),
            full_trajectory=full_trajectory,
        )
        result = validator.validate(
            candidate,
            trajectory,
            final_score=final_score,
            success=success,
            task=task,
            force=success or final_score > 0,
        )
        if not result.accepted or candidate is None:
            continue

        candidate.score = final_score
        candidate.confidence = max(candidate.confidence, result.quality)
        candidate.task_signature = task_signature(task, task_id=task_id)
        if success:
            candidate.outcome = "success"
            candidate.success_count = 1
        elif final_score > 0:
            candidate.outcome = "partial"
            candidate.partial_count = 1
        else:
            candidate.outcome = "failure"
            candidate.failure_count = 1
        candidate.update_status_from_confidence()

        before = plib.procedural_size()
        consolidator.consolidate(plib, candidate)
        after = plib.procedural_size()
        if after > before:
            stored += 1

    # Offline build produces immutable P_train (not runtime delta).
    promoted = plib.promote_delta_to_train()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plib.persist_path = persist_path
    plib.persist_train_artifact(output_path)
    print(
        f"Saved procedural memory: {plib.train_pattern_count()} train patterns "
        f"from {len(trajectories)} episodes ({stored} accepted, {promoted} promoted) -> {output_path}"
    )
    return output_path


def main() -> None:
    positional = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    config_path = positional[0] if positional else os.path.join(
        _REPO_ROOT, "config", "config.yml",
    )
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)

    agent_config = _load_config(config_path)
    agent_config.setdefault("EXPERIMENT", {})["SET"] = "train"

    output_path = _resolve_output_path(agent_config, _REPO_ROOT)
    pm = dict(agent_config.get("PROCEDURAL_MEMORY") or {})
    threshold = float(pm.get("validation_threshold", 0.65) or 0.65)

    traj_arg = positional[1] if len(positional) > 1 else ""
    success_only = "--success-only" in sys.argv[1:]
    full_trajectory = "--full-trajectory" in sys.argv[1:]
    wm_cfg = dict(agent_config.get("WORLD_MODEL") or {})
    default_traj = wm_cfg.get("trajectory_path") or os.path.join(
        _REPO_ROOT, "artifacts", "training_trajectories.jsonl",
    )
    traj_path = traj_arg or default_traj
    if not os.path.isabs(traj_path):
        traj_path = os.path.join(_REPO_ROOT, traj_path)

    print(f"Procedural memory build: output={output_path}")
    if full_trajectory:
        print("Trajectory mode: full ordered gold procedure (no last-gain/20-action truncation)")

    if not os.path.isfile(traj_path):
        print(f"Trajectory file not found: {traj_path}")
        print(
            "Export training trajectories first:\n"
            "  python scripts/collect_training_trajectories.py\n"
            "  python scripts/build_procedural_memory.py config/config.yml artifacts/training_trajectories.jsonl"
        )
        return

    trajectories = _load_trajectories(traj_path)
    print(f"Loading {len(trajectories)} episodes from {traj_path}")
    build_from_trajectories(
        trajectories,
        output_path=output_path,
        validation_threshold=threshold,
        success_only=success_only,
        full_trajectory=full_trajectory,
    )


if __name__ == "__main__":
    main()
