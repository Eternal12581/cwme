#!/usr/bin/env python3
"""
Build Historical World Knowledge Graph from training trajectories.

Usage:
  python scripts/build_historical_kg.py [config.yml] [trajectories.jsonl]
  python scripts/build_historical_kg.py [config.yml] [trajectories.jsonl] --success-only

Training episodes populate H_train and save to
artifacts/historical_kg/historical_world_model.json

Trajectory JSONL format (one episode per line):
  {"episode_id": 0, "task": "...", "task_id": "...", "final_score": 80,
   "success": false,
   "steps": [{"step": 0, "state_before": "...", "action": "...",
              "state_after": "...", "score_before": 0, "score_after": 0}]}
"""
from __future__ import annotations

import json
import os
import random
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.world_model.hkg_config import resolve_hkg_config
from core.world_model.kg_builder import EpisodeTrajectory, HistoricalKGBuilder
from core.world_model.kg_extractor import TrajectoryStep
from core.trajectory_quality import is_complete_success_trajectory


def _load_config(path: str) -> dict:
    import yaml

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


def build_from_log_trajectories(trajectories: list[dict], *, repo_root: str = "", output_dir: str = "") -> str:
    """Offline build when ReasoningAgent trajectories are exported as JSON."""
    repo_root = repo_root or _REPO_ROOT
    cfg = resolve_hkg_config({"WORLD_MODEL": {"historical_kg": True}}, repo_root=repo_root)
    builder = HistoricalKGBuilder()
    episodes: list[EpisodeTrajectory] = []
    for i, row in enumerate(trajectories):
        eid = int(row.get("episode_id", i))
        steps = [
            TrajectoryStep(
                state_before=s.get("state_signature_before") or s.get(
                    "state_before", s.get("observation_before", "")
                ),
                action=s.get("action", ""),
                state_after=s.get("state_signature_after") or s.get(
                    "state_after", s.get("observation_after", "")
                ),
                score_before=float(s.get("score_before", 0)),
                score_after=float(s.get("score_after", 0)),
                step=int(s.get("step", j)),
                valid=bool(s.get("valid", True)),
                meaningful_change=bool(s.get("meaningful_change", False)),
                progress_kind=s.get("progress_kind", "") or "",
                state_signature_before=s.get("state_signature_before", "") or "",
                state_signature_after=s.get("state_signature_after", "") or "",
                effect_signature=s.get("effect_signature", "") or "",
            )
            for j, s in enumerate(row.get("steps", []))
        ]
        episodes.append(
            EpisodeTrajectory(
                episode_id=eid,
                task=row.get("task", ""),
                task_id=row.get("task_id", ""),
                steps=steps,
                final_score=float(row.get("final_score", 0)),
                success=bool(row.get("success", False)),
            )
        )

    kg = builder.build(episodes)
    task_ids = sorted({str(row.get("task_id", "") or "") for row in trajectories if row.get("task_id")})
    task_names = sorted({
        str(row.get("task", "") or row.get("task_id", "") or "")
        for row in trajectories
        if row.get("task") or row.get("task_id")
    })
    variation_ids = sorted({str(row.get("variation_id", "") or "") for row in trajectories if row.get("variation_id")})

    success_count = partial_count = fail_count = 0
    for row in trajectories:
        score = int(row.get("final_score", row.get("score", 0)) or 0)
        if row.get("success") or score >= 100:
            success_count += 1
        elif score >= 50:
            partial_count += 1
        else:
            fail_count += 1

    split_metadata = {
        "split": "train",
        "num_episodes": len(trajectories),
        "task_ids": task_ids,
        "tasks": task_names,
        "task_names": task_names,
        "variation_ids": variation_ids,
        "success_episodes": success_count,
        "partial_ge50_episodes": partial_count,
        "failure_episodes": fail_count,
    }
    kg.metadata["split"] = split_metadata
    # Keep the same fields at the top level as well.  Runtime consumers use
    # episode/edge provenance, while artifact analysis and paper tables need
    # cheap, unambiguous aggregate counts.
    kg.metadata.update({
        "split_name": "train",
        "task_ids": task_ids,
        "tasks": task_names,
        "task_names": task_names,
        "variation_ids": variation_ids,
        "success_episodes": success_count,
        "partial_ge50_episodes": partial_count,
        "failure_episodes": fail_count,
        "source_episodes": len(trajectories),
    })
    out_dir = output_dir or os.path.dirname(cfg.resolve_path(repo_root))
    os.makedirs(out_dir, exist_ok=True)
    kg.save(out_dir)
    print(
        f"Saved Historical KG: {len(kg)} edges, {kg.metadata.get('episode_count', 0)} episodes -> {out_dir}\n"
        f"  Training episodes: {len(trajectories)}\n"
        f"  Successful (≥100): {success_count}\n"
        f"  Partial (50–99):   {partial_count}\n"
        f"  Failed (<50):      {fail_count}"
    )
    return out_dir


def _stratified_subsample(trajectories: list[dict], fraction: float, seed: int) -> list[dict]:
    """Sample trajectories proportionally per task_id (at least 1 per task)."""
    from collections import defaultdict

    by_task: dict[str, list[int]] = defaultdict(list)
    for i, traj in enumerate(trajectories):
        task_id = str(traj.get("task_id") or traj.get("task") or "unknown").strip().lower()
        by_task[task_id].append(i)

    rng = random.Random(seed)
    picked: set[int] = set()
    for indices in by_task.values():
        n = max(1, int(len(indices) * fraction))
        if n >= len(indices):
            picked.update(indices)
        else:
            picked.update(rng.sample(indices, n))
    return [trajectories[i] for i in sorted(picked)]


def main() -> None:
    positional = [arg for arg in sys.argv[1:] if arg != "--success-only"]
    success_only = "--success-only" in sys.argv[1:]
    config_path = positional[0] if positional else os.path.join(
        _REPO_ROOT, "config", "config.yml",
    )
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)

    agent_config = _load_config(config_path)
    agent_config.setdefault("EXPERIMENT", {})["SET"] = "train"
    agent_config.setdefault("WORLD_MODEL", {})["historical_kg"] = True

    cfg = resolve_hkg_config(agent_config, repo_root=_REPO_ROOT)
    out_path = cfg.resolve_path(_REPO_ROOT)

    traj_arg = positional[1] if len(positional) > 1 else ""
    wm_cfg = dict(agent_config.get("WORLD_MODEL") or {})
    default_traj = wm_cfg.get("trajectory_path") or os.path.join(
        _REPO_ROOT, "artifacts", "training_trajectories.jsonl",
    )
    traj_path = traj_arg or default_traj
    if not os.path.isabs(traj_path):
        traj_path = os.path.join(_REPO_ROOT, traj_path)

    print(f"HKG build config: enabled={cfg.enabled} output={out_path}")

    if os.path.isfile(traj_path):
        trajectories = _load_trajectories(traj_path)
        if success_only:
            trajectories = [
                row for row in trajectories
                if is_complete_success_trajectory(row)
            ]
            print(
                "Complete-success-only HKG build: "
                f"{len(trajectories)} trajectories"
            )
        fraction = float(wm_cfg.get("train_episode_fraction", 1.0) or 1.0)
        if 0 < fraction < 1.0:
            seed = int((agent_config.get("AGENT") or {}).get("SEED", 4901) or 4901)
            trajectories = _stratified_subsample(trajectories, fraction, seed)
            print(
                f"Stratified HKG subsample: fraction={fraction} -> "
                f"{len(trajectories)} episodes (seed={seed})"
            )
        print(f"Loading {len(trajectories)} episodes from {traj_path}")
        build_from_log_trajectories(
            trajectories,
            repo_root=_REPO_ROOT,
            output_dir=os.path.dirname(out_path),
        )
        return

    print(f"Trajectory file not found: {traj_path}")
    print(
        "Export training trajectories to JSONL, then re-run:\n"
        "  python scripts/build_historical_kg.py config/config.yml artifacts/training_trajectories.jsonl"
    )


if __name__ == "__main__":
    main()
