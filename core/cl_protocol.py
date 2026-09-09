"""
Continual-learning (CL-on vs CL-off) protocol helpers.

CL-off: Swift fast-path only — no pattern read/write, no blacklist filtering.
CL-on:  Pattern REUSE/force/blacklist layered on top of the same Swift backbone.
"""
from __future__ import annotations


def pattern_read_enabled(agent) -> bool:
    plib = getattr(agent, "pattern_library", None)
    return plib is not None and bool(getattr(plib, "enable_read", False))


def pattern_write_enabled(agent) -> bool:
    plib = getattr(agent, "pattern_library", None)
    return plib is not None and bool(getattr(plib, "enable_write", False))


def cwme_pattern_only_mode(agent) -> bool:
    """
    CWME decision mode: actions from pattern memory or cognition only.

    When True, legacy env heuristics (heat_setup, transfer, plateau force)
    are disabled; GroundingFacade applies valid-action filtering only.
    """
    return pattern_read_enabled(agent)


def grounding_only_mode(agent) -> bool:
    """True when execution must not use task/legacy heuristics."""
    from core.grounding.grounding_mode import GroundingMode, resolve_grounding_mode_from_agent
    return resolve_grounding_mode_from_agent(agent) == GroundingMode.GROUNDING_ONLY


def task_heuristics_enabled(agent) -> bool:
    """Task-specific deterministic heuristics (ScienceWorld policy shortcuts)."""
    cfg = getattr(agent, "cwme_cfg", None)
    if cfg is not None:
        return bool(getattr(cfg, "task_heuristics_enabled", False))
    exp = getattr(agent, "agent_config", {}) or {}
    exec_cfg = dict(exp.get("EXECUTION") or {})
    return bool(exec_cfg.get("task_heuristics_enabled", False))


def legacy_heuristics_enabled(agent) -> bool:
    """Legacy plateau / pattern-force reuse (CL-off baseline path)."""
    exp = getattr(agent, "agent_config", {}) or {}
    exec_cfg = dict(exp.get("EXECUTION") or {})
    if "legacy_heuristics_enabled" in exec_cfg:
        return bool(exec_cfg.get("legacy_heuristics_enabled"))
    cfg = getattr(agent, "cwme_cfg", None)
    if cfg is not None:
        return bool(getattr(cfg, "legacy_heuristics_enabled", False))
    return not pattern_read_enabled(agent)


def episode_guard_enabled(agent) -> bool:
    exp = getattr(agent, "agent_config", {}) or {}
    guard_cfg = dict(exp.get("EPISODE_GUARD") or {})
    if "enabled" in guard_cfg:
        return bool(guard_cfg.get("enabled"))
    return getattr(agent, "episode_guard", None) is not None


def is_action_blocked(agent, action: str) -> bool:
    """Episode-local guard only (respects CL-off)."""
    act = (action or "").strip().lower()
    if not act:
        return False
    if not episode_guard_enabled(agent):
        return False
    guard = getattr(agent, "episode_guard", None)
    if guard is None:
        return False
    try:
        from core.evolution.transition_tracker import state_key
        state_sig = state_key(getattr(agent, "observation", "") or "")
    except Exception:
        state_sig = ""
    try:
        return guard.is_blocked(act, state_signature=state_sig)
    except TypeError:
        return guard.is_blocked(act)


def pattern_milestone_for_agent(agent, *, score: int | None = None) -> str:
    """Grounded next pattern milestone from state packet (empty when CL-off)."""
    if not pattern_read_enabled(agent):
        return ""
    try:
        from core.state_packet import build_state_packet
        pkt = build_state_packet(agent, score=score)
        return (getattr(pkt, "pattern_next_milestone", "") or "").strip()
    except Exception:
        return ""


def pattern_reuse_action_ok(
    agent,
    action: str,
    task: str = "",
    env_valid=None,
    *,
    past_actions=None,
    observation: str = "",
    pattern_evidence=None,
) -> bool:
    """Unified CL reuse gate (cognition / scheduler / facade). CL-off → False."""
    if not pattern_read_enabled(agent):
        return False
    act = (action or "").strip()
    if not act:
        return False
    if is_action_blocked(agent, act):
        return False

    if grounding_only_mode(agent) and not task_heuristics_enabled(agent):
        plib = getattr(agent, "pattern_library", None)
        pat = plib.get_active_procedural() if plib is not None else None
        if pat is not None:
            from core.procedural.pattern_reuse_policy import is_reuse_eligible
            if not is_reuse_eligible(pat):
                return False
        act_l = act.lower()
        valid = {(str(v) or "").strip().lower() for v in (env_valid or []) if v}
        if valid:
            from core.grounding_core import ground_candidate_action
            grounded = (ground_candidate_action(act_l, valid) or act_l).strip().lower()
            if grounded not in valid:
                return False
        try:
            from core.scienceworld_grounding import is_safe_task_pattern_action
            return bool(is_safe_task_pattern_action(
                act_l,
                task=task,
                past_actions=past_actions,
                observation=observation,
                pattern_evidence=pattern_evidence,
            ))
        except Exception:
            return True

    try:
        from core.reuse_policy_legacy import pattern_force_action_ok
        return bool(pattern_force_action_ok(agent, act, task=task, env_valid=env_valid))
    except Exception:
        return False
