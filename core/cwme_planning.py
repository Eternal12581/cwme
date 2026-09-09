"""CWME trajectory planning helpers (LLM plan vs hybrid admissible seed)."""
from __future__ import annotations

from core.cl_protocol import cwme_pattern_only_mode


def _prediction_sentinel(state: dict | None = None) -> dict:
    """Return the terminal placeholder expected by ReasoningAgent.act_and_refine."""
    return {
        "state": dict(state or {}),
        "action": "<PREDICT>",
        "reward (env response)": "<PREDICT>",
        "next state": None,
        "done or termination": None,
    }


def _is_prediction_sentinel(item) -> bool:
    return isinstance(item, dict) and item.get("action") == "<PREDICT>"


def cwme_llm_planning_enabled(agent) -> bool:
    """
    Return whether CWME should pay for an open-loop 9B initial plan.

    ``llm`` preserves the original diagnostic behavior.  ``hybrid`` and
    ``heuristic`` start from the deterministic admissible-action seed and
    invoke cognition only when the live environment cannot advance it.
    """
    return cwme_initial_plan_mode(agent) == "llm"


def cwme_initial_plan_mode(agent) -> str:
    cfg = getattr(agent, "cwme_cfg", None)
    if cfg is not None:
        mode = getattr(cfg, "initial_plan_mode", "llm")
    else:
        exp = getattr(agent, "agent_config", {}) or {}
        exec_cfg = dict(exp.get("EXECUTION") or {})
        mode = exec_cfg.get(
            "cwme_initial_plan_mode",
            (exp.get("EXPERIMENT") or {}).get("CWME_INITIAL_PLAN_MODE", "llm"),
        )
    mode = str(mode or "llm").strip().lower()
    return mode if mode in ("llm", "hybrid", "heuristic") else "llm"


def cwme_heuristic_fallback_enabled(agent) -> bool:
    """Whether the formal CWME path may use the live task-family fallback."""
    # ``grounding_only`` is a hard protocol boundary. Several older configs
    # set the fallback flag for diagnostics while still carrying this mode;
    # honoring that combination contaminates backbone and ablation runs.
    from core.cl_protocol import grounding_only_mode
    if grounding_only_mode(agent):
        return False
    cfg = getattr(agent, "cwme_cfg", None)
    if cfg is not None:
        return bool(getattr(cfg, "heuristic_fallback_enabled", False))
    exp = getattr(agent, "agent_config", {}) or {}
    exec_cfg = dict(exp.get("EXECUTION") or {})
    return bool(
        exec_cfg.get(
            "cwme_heuristic_fallback",
            (exp.get("EXPERIMENT") or {}).get("CWME_HEURISTIC_FALLBACK", False),
        )
    )


def cwme_actor_stuck_threshold(agent) -> int:
    cfg = getattr(agent, "cwme_cfg", None)
    if cfg is not None:
        return max(1, int(getattr(cfg, "actor_stuck_threshold", 5) or 5))
    return 5


def cwme_skip_initial_plan_if_pattern(agent) -> bool:
    """Skip the per-episode 9B open-loop plan when P_train is executable."""
    cfg = getattr(agent, "cwme_cfg", None)
    if cfg is not None:
        return bool(getattr(cfg, "skip_initial_plan_if_pattern", True))
    exp = getattr(agent, "agent_config", {}) or {}
    exec_cfg = dict(exp.get("EXECUTION") or {})
    return bool(exec_cfg.get("cwme_skip_initial_plan_if_pattern", True))


def cwme_actor_max_attempts(agent) -> int:
    """Bound expensive Actor retries while retaining one repair attempt."""
    cfg = getattr(agent, "cwme_cfg", None)
    if cfg is not None:
        return max(1, int(getattr(cfg, "actor_max_attempts", 2) or 2))
    exp = getattr(agent, "agent_config", {}) or {}
    exec_cfg = dict(exp.get("EXECUTION") or {})
    return max(1, int(exec_cfg.get("cwme_actor_max_attempts", 2) or 2))


def cwme_initial_plan_steps(agent) -> int:
    """Bound the expensive open-loop plan used at episode start."""
    cfg = getattr(agent, "cwme_cfg", None)
    if cfg is not None:
        return max(1, int(getattr(cfg, "initial_plan_steps", 5) or 5))
    exp = getattr(agent, "agent_config", {}) or {}
    exec_cfg = dict(exp.get("EXECUTION") or {})
    return max(1, int(exec_cfg.get("cwme_initial_plan_steps", 5) or 5))


def cwme_refiner_max_attempts(agent) -> int:
    """Bound Refiner retries; malformed output has a safe default."""
    cfg = getattr(agent, "cwme_cfg", None)
    if cfg is not None:
        return max(1, int(getattr(cfg, "refiner_max_attempts", 2) or 2))
    exp = getattr(agent, "agent_config", {}) or {}
    exec_cfg = dict(exp.get("EXECUTION") or {})
    return max(1, int(exec_cfg.get("cwme_refiner_max_attempts", 2) or 2))


def minimal_cognition_trajectory(
    observation: str,
    inventory: str,
    *,
    max_steps: int,
    task: str = "",
) -> tuple[list, str]:
    """
    Empty-action open loop: each step defers to Pattern/Fast/Verify/9B Actor.

    Used when get_trajectory fails in CWME mode — never falls back to task rules.
    """
    state = {"observation": observation or "", "inventory": inventory or ""}
    hint = (task or "task goal").strip()
    trajectory: list[dict] = []
    for idx in range(max(1, int(max_steps or 1))):
        trajectory.append(
            {
                "state": state,
                "action": "",
                "reward (env response)": f"Decide step {idx + 1} toward: {hint}",
                "next state": state,
                "done or termination": False,
            }
        )
    # The execution loop treats the final record as a non-executable planner
    # sentinel.  Without it, a cognition-pad trajectory executes N-1 of the
    # requested N actions and silently loses the final recovery opportunity.
    trajectory.append(_prediction_sentinel(state))
    return trajectory, f"CWME per-step cognition ({len(trajectory)} steps)"


def pad_cognition_trajectory(
    trajectory: list,
    *,
    observation: str,
    inventory: str,
    pad_steps: int,
    task: str = "",
) -> tuple[list, str]:
    """Extend an existing trajectory with empty cognition placeholders."""
    state = {"observation": observation or "", "inventory": inventory or ""}
    # A planner trajectory ends in a <PREDICT> sentinel whose ``next state``
    # is intentionally None.  Recover the latest real state before removing
    # that sentinel, otherwise padding restarts from the episode's initial
    # observation and loses the live context.
    for item in reversed(trajectory or []):
        if not isinstance(item, dict):
            continue
        for key in ("next state", "state"):
            candidate = item.get(key)
            if isinstance(candidate, dict) and candidate:
                state = candidate
                break
        if state.get("observation") or state.get("inventory"):
            break

    base = list(trajectory or [])
    if base and _is_prediction_sentinel(base[-1]):
        base.pop()
    extra, msg = minimal_cognition_trajectory(
        state.get("observation", observation),
        state.get("inventory", inventory),
        max_steps=pad_steps,
        task=task,
    )
    # ``extra`` already contains its own sentinel. Keep exactly one sentinel
    # at the end so every appended placeholder is executable.
    if extra and _is_prediction_sentinel(extra[-1]):
        extra = extra[:-1]
    merged = base + extra + [_prediction_sentinel(state)]
    return merged, msg
