"""
Cognition decision routing (inner-ring Large LLM + architecture hints).

Provides phase/suggestion context for Actor hints and legacy CL-off baseline.
In CWME mode, action decisions flow through ExecutionRouter; scheduled env hints
(heat_setup, transfer, …) assist grounding only and never override 9B.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.action_primitives import canonicalize_env_action
from core.scienceworld_schema import STAGNANT_STEPS_BEFORE_REPLAN
from core.state_packet import build_state_packet, resolve_episode_phase
from core.suggestion_scheduler import ScheduledSuggestion, SuggestionScheduler
from core.task_semantics import task_manipulation_destination_rooms
from core.navigation_helpers import _nav_dest_room
from utils import (
    _agent_live_context,
    _normalize_room_name,
    action_verb_family,
    is_off_task_planned_action,
)

_LOW_VALUE_FAMILIES = frozenset({
    "meta", "look_around", "look_at", "examine",
})
_LOW_VALUE_ACTIONS = frozenset({"inventory", "task", "look around"})
# Sources allowed to override 9B planner/Actor decisions.
_DECISION_OVERRIDE_SOURCES = frozenset({
    "pattern", "historical_kg", "cognition",
})
# Environment hints used only for grounding normalization / Actor context.
_GROUNDING_HINT_SOURCES = frozenset({
    "pre_focus", "post_focus", "transfer", "heat_setup",
    "connect", "growth", "navigation",
})


@dataclass
class CognitionRoute:
    """Per-step cognition routing decision."""

    phase: str = "explore"
    som_sub_phase: str = ""
    transfer_pending: bool = False
    transfer_executable: bool = False
    current_room: str = ""
    foc_room: str = ""
    task: str = ""
    grounded_suggestion: str = ""
    raw_suggestion: str = ""
    suggestion_source: str = ""
    pattern_milestone: str = ""
    pattern_match_kind: str = ""
    prefer_grounded: bool = False
    steps_since_gain: int = 0
    score_max: int = 0
    current_score: int = 0
    learning_plateau: bool = False

    def has_suggestion(self) -> bool:
        return bool((self.grounded_suggestion or "").strip())


class CognitionRouter:
    """Selects grounded action hints and when to prefer them over planner/Actor."""

    def __init__(self, scheduler: SuggestionScheduler | None = None):
        self.scheduler = scheduler or SuggestionScheduler()

    def schedule(
        self,
        agent,
        env_valid=None,
        *,
        score: int | None = None,
    ) -> ScheduledSuggestion:
        packet = build_state_packet(agent, score=score)
        valid = env_valid if env_valid is not None else self._env_valid(agent)
        past = list(getattr(agent, "past_actions", []) or [])
        recent = {(a or "").strip().lower() for a in past[-5:] if a}
        failed = getattr(agent, "recent_failed_actions", None) or set()
        ctx_text = _agent_live_context(agent)
        return self.scheduler.schedule(
            packet,
            env_valid=valid,
            task=getattr(agent, "task", "") or "",
            ctx_text=ctx_text,
            action_history=past,
            recent=recent,
            failed=failed,
        )

    @staticmethod
    def _pattern_reuse_ok(
        agent,
        action: str,
        task: str = "",
        env_valid=None,
        *,
        past_actions=None,
        observation: str = "",
        pattern_evidence=None,
    ) -> bool:
        from core.cl_protocol import pattern_reuse_action_ok
        return pattern_reuse_action_ok(
            agent,
            action,
            task,
            env_valid=env_valid,
            past_actions=past_actions,
            observation=observation,
            pattern_evidence=pattern_evidence,
        )

    @staticmethod
    def _plateau_from_packet(packet, score: int | None, agent) -> tuple[int, int, int, bool]:
        prog = dict(
            getattr(packet, "episode_progress", None)
            or getattr(agent, "episode_progress", {})
            or {}
        )
        steps = int(prog.get("steps_since_gain", 0) or 0)
        score_max = int(
            getattr(packet, "score_max", 0) or prog.get("score_max", 0) or 0
        )
        if score is not None:
            cur = int(score)
        elif getattr(packet, "score", None) is not None:
            cur = int(packet.score)
        else:
            cur = int(getattr(agent, "last_score", 0) or 0)
        alf_zero_explore = False
        try:
            from envs.base import is_alfworld
            if (
                is_alfworld(getattr(agent, "env", None))
                and score_max < 25
                and cur < 100
            ):
                alf_zero_explore = True
        except Exception:
            alf_zero_explore = False
        plateau = (
            cur < 100
            and cur >= score_max
            and (
                steps >= STAGNANT_STEPS_BEFORE_REPLAN
                or (
                    bool((packet.pattern_next_milestone or "").strip())
                    and (
                        (score_max < 25 and steps >= 1)
                        or (score_max < 50 and steps >= 2)
                        or (score_max < 80 and steps >= 3)
                    )
                )
                or (
                    # AlfWorld explore + zero score: enable CL force/blacklist
                    # without waiting for a stored milestone (pick_two cold start).
                    alf_zero_explore and steps >= 2
                )
            )
            and bool(
                packet.transfer_pending
                or packet.heat_setup_pending
                or packet.phase in ("manipulation", "exploit", "focus_pending")
                or (packet.som_phase or "")
                or (packet.pattern_next_milestone or "")
                or alf_zero_explore
            )
        )
        return steps, score_max, cur, plateau

    def route(
        self,
        agent,
        env_valid=None,
        *,
        score: int | None = None,
    ) -> CognitionRoute:
        packet = build_state_packet(agent, score=score)
        phase = packet.phase or resolve_episode_phase(
            getattr(agent, "task", "") or "",
            f"{getattr(agent, 'observation', '')} {getattr(agent, 'inventory', '')}".lower(),
            dict(getattr(agent, "episode_progress", {}) or {}),
            list(getattr(agent, "past_actions", []) or []),
            int(getattr(agent, "last_score", 0) or 0),
            getattr(agent, "inventory", "") or "",
        )

        valid = env_valid if env_valid is not None else self._env_valid(agent)
        scheduled = self.scheduler.schedule(
            packet,
            env_valid=valid,
            task=getattr(agent, "task", "") or "",
            ctx_text=_agent_live_context(agent),
            action_history=list(getattr(agent, "past_actions", []) or []),
            recent={(a or "").strip().lower() for a in (getattr(agent, "past_actions", []) or [])[-5:] if a},
            failed=getattr(agent, "recent_failed_actions", None) or set(),
        )

        grounded = ""
        source = ""
        if scheduled.grounded and scheduled.action:
            grounded = scheduled.action
            source = scheduled.source
        elif scheduled.action and scheduled.source == "pattern":
            grounded = scheduled.action if valid and scheduled.action in {
                (v or "").strip().lower() for v in valid
            } else ""
            source = "pattern" if grounded else scheduled.source

        steps, score_max, cur, plateau = self._plateau_from_packet(packet, score, agent)
        valid_lower = {(v or "").strip().lower() for v in (valid or set())}
        plib = getattr(agent, "pattern_library", None)
        pattern_ms = (
            getattr(plib, "last_pattern_template", "")
            if plib is not None
            else ""
        ) or (packet.pattern_next_milestone or "")
        pattern_ms = pattern_ms.strip().lower()
        match_kind = (getattr(packet, "pattern_match_kind", "") or "").strip().lower()
        from core.cl_protocol import is_action_blocked
        if pattern_ms and is_action_blocked(agent, pattern_ms):
            pattern_ms = ""

        from core.cwme_config import pattern_hard_reuse_enabled
        from core.cl_protocol import cwme_pattern_only_mode
        hard_reuse = pattern_hard_reuse_enabled(agent)
        cwme_mode = cwme_pattern_only_mode(agent)

        # Exact / same-task hits may override only when hard reuse is enabled
        # and CWME fast path is off (legacy CL-off baseline experiments).
        from core.scienceworld_grounding import is_safe_task_pattern_action, should_defer_pattern_reuse
        task_s = getattr(agent, "task", "") or ""
        pattern_evidence = []
        try:
            from core.execution.artifact_evidence import execution_pattern_evidence

            pattern_evidence = execution_pattern_evidence(
                agent,
                task=task_s,
                observation=_agent_live_context(agent),
            )
        except Exception:
            pattern_evidence = []
        pattern_context = {
            "task": task_s,
            "past_actions": list(getattr(agent, "past_actions", []) or []),
            "observation": _agent_live_context(agent),
            "pattern_evidence": pattern_evidence,
        }
        defer_ms = should_defer_pattern_reuse(
            pattern_ms,
            score_max=score_max,
            steps_since_gain=steps,
            phase=phase,
            task=task_s,
            past_actions=list(getattr(agent, "past_actions", []) or []),
        )
        exact_pattern = (
            bool(pattern_ms)
            and match_kind == "exact"
            and is_safe_task_pattern_action(pattern_ms, **pattern_context)
            and not defer_ms
            and self._pattern_reuse_ok(
                agent,
                pattern_ms,
                task_s,
                env_valid=valid,
                past_actions=pattern_context["past_actions"],
                observation=pattern_context["observation"],
                pattern_evidence=pattern_context["pattern_evidence"],
            )
        )
        task_pattern = (
            bool(pattern_ms)
            and match_kind == "task"
            and is_safe_task_pattern_action(pattern_ms, **pattern_context)
            and not defer_ms
            and self._pattern_reuse_ok(
                agent,
                pattern_ms,
                task_s,
                env_valid=valid,
                past_actions=pattern_context["past_actions"],
                observation=pattern_context["observation"],
                pattern_evidence=pattern_context["pattern_evidence"],
            )
        )
        if (
            not cwme_mode
            and hard_reuse
            and plateau
            and (exact_pattern or task_pattern)
            and pattern_ms in valid_lower
            and not self.is_low_value_planned(pattern_ms)
        ):
            if (
                not grounded
                or source != "pattern"
                or self.is_low_value_planned(grounded)
                or grounded in {
                    (a or "").strip().lower()
                    for a in (getattr(agent, "past_actions", []) or [])[-3:]
                    if a
                }
            ):
                grounded = pattern_ms
                source = "pattern"

        # CWME: pattern reuse is routed by ExecutionRouter, not CognitionRouter.
        # Legacy scheduled hints (heat_setup, transfer, …) never override 9B.
        prefer = False
        if not cwme_mode and grounded and source in _DECISION_OVERRIDE_SOURCES:
            from core.scienceworld_grounding import is_process_cl_milestone
            if source == "pattern" and not defer_ms and hard_reuse:
                if match_kind == "exact" and (
                    plateau
                    or packet.phase in ("manipulation", "exploit")
                    or is_process_cl_milestone(grounded)
                ):
                    prefer = True
                elif match_kind == "task" and plateau and grounded:
                    try:
                        if is_safe_task_pattern_action(grounded, **pattern_context):
                            prefer = True
                    except Exception:
                        pass
        if source == "pattern" and defer_ms:
            prefer = False
            grounded = ""
            source = ""

        return CognitionRoute(
            phase=phase,
            som_sub_phase=packet.som_sub_phase or "",
            transfer_pending=bool(packet.transfer_pending),
            transfer_executable=bool(packet.transfer_executable),
            current_room=packet.current_room or "",
            foc_room=packet.foc_room or "",
            task=getattr(agent, "task", "") or "",
            grounded_suggestion=grounded,
            raw_suggestion=scheduled.raw or scheduled.action,
            suggestion_source=source if grounded else (scheduled.source if scheduled.action else ""),
            pattern_milestone=pattern_ms or (packet.pattern_next_milestone or "").strip().lower(),
            pattern_match_kind=match_kind,
            prefer_grounded=prefer,
            steps_since_gain=steps,
            score_max=score_max,
            current_score=cur,
            learning_plateau=plateau,
        )

    @staticmethod
    def _env_valid(agent) -> set:
        if not hasattr(agent, "env") or agent.env is None:
            return set()
        try:
            return set(agent.env.getValidActionObjectCombinations())
        except Exception:
            return set()

    @staticmethod
    def _nav_dest(action: str) -> str:
        return _nav_dest_room(action)

    def _is_transfer_ping_pong_nav(
        self,
        route: CognitionRoute,
        planned: str,
    ) -> bool:
        if not route.transfer_pending:
            return False
        dest = self._nav_dest(planned)
        if not dest:
            return False
        if route.som_sub_phase == "transfer_nav":
            dest_rooms = {
                _normalize_room_name(r)
                for r in task_manipulation_destination_rooms(route.task)
            }
            if dest in dest_rooms:
                return False
        if route.som_sub_phase == "transfer_nav_source":
            foc = _normalize_room_name(route.foc_room) if route.foc_room else ""
            if foc and dest == foc:
                return False
        cur = route.current_room
        foc = route.foc_room
        if cur and foc and cur == foc:
            return dest != foc
        if cur and foc and cur != foc and dest == cur:
            return True
        return False

    @staticmethod
    def is_low_value_planned(planned: str) -> bool:
        p = (planned or "").strip().lower()
        if p in _LOW_VALUE_ACTIONS:
            return True
        return action_verb_family(p) in _LOW_VALUE_FAMILIES

    def should_use_grounded_action(
        self,
        route: CognitionRoute,
        planned: str,
        env_valid: set,
        task: str,
        ctx_text: str,
        agent=None,
    ) -> bool:
        """Prefer architecture-grounded suggestion over planner/Actor when safer."""
        from core.cl_protocol import cwme_pattern_only_mode

        if agent is not None and cwme_pattern_only_mode(agent):
            # CWME: Pattern → ExecutionRouter → Fast/Verify/9B is the sole decision chain.
            return False

        raw = (route.grounded_suggestion or "").strip()
        grounded = canonicalize_env_action(raw, env_valid) if raw else ""
        if not grounded or grounded not in env_valid:
            if route.learning_plateau and route.pattern_milestone:
                grounded = canonicalize_env_action(route.pattern_milestone, env_valid)
                if not grounded or grounded not in env_valid:
                    return False
            else:
                return False
        if is_off_task_planned_action(grounded, task, ctx_text):
            return False

        src = (route.suggestion_source or "").strip().lower()
        if src in _GROUNDING_HINT_SOURCES:
            return False
        if src and src not in _DECISION_OVERRIDE_SOURCES:
            return False

        planned_norm = (planned or "").strip().lower().replace("green house", "greenhouse")
        if planned_norm == grounded:
            return False

        try:
            from core.navigation_helpers import _planned_work_area_nav_worth_keeping
            past = list(getattr(agent, "past_actions", []) or []) if agent else []
            cur = int(getattr(agent, "last_score", 0) or 0) if agent else 0
            if _planned_work_area_nav_worth_keeping(
                planned_norm, task, env_valid, past, ctx_text, cur,
            ):
                return False
        except Exception:
            pass

        grounded_family = action_verb_family(grounded)
        planned_family = action_verb_family(planned_norm)

        if route.learning_plateau and route.prefer_grounded:
            if self.is_low_value_planned(planned_norm):
                return True
            if planned_norm not in env_valid:
                return True
            if planned_family != grounded_family:
                return True
            if src in _DECISION_OVERRIDE_SOURCES and planned_norm != grounded:
                return True

        from core.cwme_config import pattern_hard_reuse_enabled

        mk = (getattr(route, "pattern_match_kind", "") or "").strip().lower()
        pms = (route.pattern_milestone or "").strip().lower()
        pattern_ok = False
        if mk in ("exact", "task") and pms and pattern_hard_reuse_enabled(agent):
            try:
                from core.scienceworld_grounding import is_safe_task_pattern_action
                from core.cl_protocol import pattern_reuse_action_ok
                pattern_ok = (
                    is_safe_task_pattern_action(
                        pms,
                        task=(task or ""),
                        past_actions=list(getattr(agent, "past_actions", []) or []),
                        observation=getattr(agent, "observation", "") or "",
                    )
                    and pattern_reuse_action_ok(
                        agent,
                        pms,
                        task or "",
                        env_valid=env_valid,
                        past_actions=list(getattr(agent, "past_actions", []) or []),
                        observation=getattr(agent, "observation", "") or "",
                    )
                )
            except Exception:
                pattern_ok = False
        if (
            pattern_hard_reuse_enabled(agent)
            and route.learning_plateau
            and pms
            and pms in env_valid
            and not self.is_low_value_planned(pms)
            and pattern_ok
        ):
            if self.is_low_value_planned(planned_norm):
                return True
            if planned_norm != pms and route.suggestion_source == "pattern":
                return True
            if planned_norm != grounded and grounded == pms:
                return True

        if route.suggestion_source == "pattern":
            if planned_norm not in env_valid:
                return True
            if self.is_low_value_planned(planned_norm):
                return True
            if route.phase in ("explore", "focus_pending"):
                if planned_norm.startswith(("go to ", "open door to ")):
                    return grounded_family not in ("meta", "look_around", "look_at", "examine")
                if action_verb_family(planned_norm) in ("meta", "look_around", "look_at", "examine"):
                    return True
            if route.learning_plateau and route.prefer_grounded:
                return True
            if route.phase in ("manipulation", "exploit") and route.prefer_grounded:
                if planned_family != grounded_family:
                    return True
            return route.prefer_grounded and planned_norm not in env_valid

        return False

    def seed_trajectory_actions(self, route: CognitionRoute, actions: list[str]) -> list[str]:
        """Prepend grounded suggestion to heuristic trajectory extension / fallback."""
        seed = (route.grounded_suggestion or "").strip().lower()
        if not seed and route.learning_plateau:
            seed = (route.pattern_milestone or "").strip().lower()
        if seed.startswith("focus on "):
            try:
                from core.scienceworld_policy import sci_focus_action_allowed
                task_s = getattr(route, "task", "") or ""
                if not task_s or not sci_focus_action_allowed(seed, task_s):
                    seed = ""
            except Exception:
                seed = ""
        if not seed:
            return list(actions)
        tail = [(a or "").strip().lower() for a in actions if a]
        if seed in tail:
            return tail
        return [seed] + tail

    def actor_priority_hint(self, route: CognitionRoute) -> str:
        """Extra Actor instruction when a grounded route is active."""
        if not route.grounded_suggestion:
            if route.transfer_pending and route.raw_suggestion:
                return (
                    f"TRANSFER_INTENT (prefer if valid): {route.raw_suggestion}"
                )
            if route.learning_plateau and route.pattern_milestone:
                return (
                    f"ROUTED_ACTION (pattern, prefer if valid): {route.pattern_milestone}"
                )
            return ""
        src = route.suggestion_source or "architecture"
        hint = (
            f"ROUTED_ACTION ({src}, prefer if valid): {route.grounded_suggestion}"
        )
        if (
            route.pattern_milestone
            and route.pattern_milestone != route.grounded_suggestion
        ):
            hint += f"\nPATTERN_MILESTONE (fallback): {route.pattern_milestone}"
        return hint
