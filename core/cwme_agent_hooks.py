"""
ReasoningAgent HKG/CWME integration hooks.

Apply these in ReasoningAgent.__init__, reset(), and step() when
ReasoningAgent.py is present in the repo.

This module exists because the local workspace may ship without
ReasoningAgent.py (server-only copy).
"""
from __future__ import annotations

import re

from core.evolution.episode_guard import EpisodeGuard
from core.evolution.transition_tracker import TransitionTracker
from core.evolution.action_source import ActionSourceLog
from core.execution.execution_types import ExecutionMetrics
from core.world_model.agent_bridge import (
    configure_historical_kg,
)
from core.cwme_config import apply_cwme_experiment
from core.trajectory_collector import attach_trajectory_collector


def _pattern_action_matches(expected: str, actual: str, *, state_after: str = "") -> bool:
    """Match a stored stage against its admissible grounded command."""
    expected = (expected or "").strip().lower()
    actual = (actual or "").strip().lower()
    if not expected or not actual:
        return False
    if actual == expected or actual.startswith(expected + " "):
        return True
    # ScienceWorld may expose a generic door primitive. Use the returned state
    # to confirm that it reached the room represented by the Pattern stage.
    expected_nav = expected.startswith(("go to ", "open door to "))
    actual_nav = actual.startswith(("go to ", "open door to "))
    if expected_nav and actual_nav:
        def _nav_destination(command: str) -> str:
            if command.startswith("open door to "):
                return command[len("open door to "):].strip()
            if command.startswith("go to "):
                return command[len("go to "):].strip()
            return ""

        expected_dest = _nav_destination(expected)
        actual_dest = _nav_destination(actual)
        if expected_dest and actual_dest == expected_dest:
            return True
        if actual_dest in {"door", "the door"} and expected_dest:
            state_tokens = set(re.findall(r"[a-z0-9]+", (state_after or "").lower()))
            dest_tokens = set(re.findall(r"[a-z0-9]+", expected_dest.lower()))
            if dest_tokens and dest_tokens.issubset(state_tokens):
                return True
    aliases = {
        "pick up": ("take",),
        "take": ("pick up",),
        "activate": ("turn on",),
        "deactivate": ("turn off",),
    }
    return any(
        actual == alias or actual.startswith(alias + " ")
        for alias in aliases.get(expected, ())
    )


def attach_cwme_components(agent, agent_config: dict | None = None, *, repo_root: str = "") -> None:
    """Call once from ReasoningAgent.__init__ after pattern_library setup."""
    agent.transition_tracker = TransitionTracker()
    agent.episode_guard = EpisodeGuard()
    agent.action_source_log = ActionSourceLog()
    agent.execution_metrics = ExecutionMetrics()
    agent.agent_config = agent_config or getattr(agent, "agent_config", {}) or {}
    configure_historical_kg(agent, agent.agent_config, repo_root=repo_root)
    agent.cwme_cfg = apply_cwme_experiment(agent, agent.agent_config, repo_root=repo_root)
    from core.grounding.grounding_mode import set_grounding_mode_from_agent
    from core.world_model.execution_audit import attach_execution_audit
    set_grounding_mode_from_agent(agent)
    exp = dict((agent_config or {}).get("EXPERIMENT") or {})
    strict_audit = bool(exp.get("STRICT_HEURISTIC_AUDIT", False))
    attach_execution_audit(agent, strict=strict_audit)
    router = getattr(agent, "router", None)
    if router is not None and hasattr(router, "execution_metrics"):
        router.execution_metrics = agent.execution_metrics
    attach_trajectory_collector(agent, agent.agent_config, repo_root=repo_root)


def on_agent_reset(agent) -> None:
    """Call from ReasoningAgent.reset() — episode-local state only."""
    plib = getattr(agent, "pattern_library", None)
    if plib is not None and hasattr(plib, "reset_episode_state"):
        # Keep the learned artifacts, but never carry an execution cursor or
        # a failed stage from one environment variation into the next.
        plib.reset_episode_state()
    if hasattr(agent, "episode_guard") and agent.episode_guard is not None:
        agent.episode_guard.reset()
    if hasattr(agent, "transition_tracker") and agent.transition_tracker is not None:
        ep_idx = int((getattr(agent, "episode_progress", {}) or {}).get("episode_idx", 0) or 0)
        agent.transition_tracker.reset(episode_id=ep_idx)
    if hasattr(agent, "action_source_log") and agent.action_source_log is not None:
        agent.action_source_log.reset()
    if hasattr(agent, "execution_metrics") and agent.execution_metrics is not None:
        agent.execution_metrics.reset()
    # These fields are produced before env.step and consumed after it. They
    # must never leak into the next variation when one agent is reused.
    for name in (
        "_last_pattern_execution",
        "_last_route_decision",
        "_last_finalized_decision_key",
        "_cwme_fast_decision_cache",
        "_cwme_actor_decision_cache",
    ):
        if hasattr(agent, name):
            setattr(agent, name, None)
    from core.grounding.grounding_mode import set_grounding_mode_from_agent
    from core.world_model.execution_audit import reset_execution_audit
    set_grounding_mode_from_agent(agent)
    reset_execution_audit(agent)
    agent._score_hist = []
    agent._last_step_score = None
    agent._last_step_score_delta = 0.0


def on_agent_step_after_env(
    agent,
    *,
    state_before: str,
    action: str,
    state_after: str,
    score_before: float,
    score_after: float,
    env_response: str = "",
):
    """
    Call after env.step and score update inside ReasoningAgent.step().

    Records the transition only; HKG updates are owned by CWME.after_step().
    """
    tracker = getattr(agent, "transition_tracker", None)
    transition = None
    if tracker is not None:
        try:
            from core.substitute_engine import is_env_action_failure
            env_failed = is_env_action_failure(env_response)
        except Exception:
            env_failed = False
        transition = tracker.record(
            state_before=state_before,
            action=action,
            state_after=state_after,
            score_before=score_before,
            score_after=score_after,
            valid=not env_failed and float(score_after) != -100,
            env_response=env_response,
        )
    hist = getattr(agent, "_score_hist", None)
    if hist is not None:
        hist.append(float(score_after))
    prev = getattr(agent, "_last_step_score", None)
    if prev is not None:
        agent._last_step_score_delta = float(score_after) - float(prev)
    agent._last_step_score = float(score_after)
    marker = getattr(agent, "_last_pattern_execution", None)
    if marker:
        plib = getattr(agent, "pattern_library", None)
        expected = (
            marker.get("milestone") or marker.get("action") or ""
        ).strip().lower()
        actual = (action or "").strip().lower()
        action_matches = _pattern_action_matches(
            expected,
            actual,
            state_after=state_after,
        )
        observation_only = actual in {
            "look", "look around", "inventory", "task",
        } or actual.startswith(("look at ", "examine ", "look in "))
        try:
            from core.substitute_engine import is_env_action_failure
            env_failed = is_env_action_failure(env_response)
        except Exception:
            env_failed = False
        progressed = bool(
            action_matches
            and not observation_only
            and not env_failed
            and (
                float(score_after) > float(score_before)
                or getattr(transition, "meaningful_change", False)
            )
        )
        if plib is not None and hasattr(plib, "note_pattern_result"):
            plib.note_pattern_result(
                marker.get("pattern_id", ""),
                marker.get("stage_index", -1),
                success=progressed,
            )
            metrics = getattr(agent, "execution_metrics", None)
            if metrics is not None:
                if progressed:
                    metrics.pattern_stage_advance += 1
                else:
                    metrics.pattern_recovery += 1
        agent._last_pattern_execution = None
    return transition
