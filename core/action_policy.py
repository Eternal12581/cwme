"""
Action policy: architecture-layer validation for routed / grounded actions.

Task-agnostic guards delegated to utils.*; extends with routing-specific rules.
"""
from __future__ import annotations

import re

from core.environment_discovery import is_heat_fixture_phrase
from core.task_semantics import task_manipulation_destination_rooms
from utils import (
    _PRE_FOCUS_HEAT_FIXTURE_WORDS,
    _action_content_tokens,
    _answer_box_action_disallowed,
    _focus_blocked_by_measurement_gate,
    _focused_substance_container_room,
    _heat_setup_pending,
    _is_absurd_post_focus_manipulation,
    _is_absurd_pour_action,
    _is_distractor_pour_action,
    _is_measurement_setup_action,
    _is_post_focus_unproductive,
    _normalize_room_name,
    _substance_transfer_pending,
    _task_needs_measurement_before_focus,
    _task_target_tokens,
    action_verb_family,
    is_off_task_planned_action,
)

from core.transfer_resolver import (
    is_transfer_container_shuffle,
    transfer_sub_phase,
)

_MANIP_FIXTURE_ALT = "|".join(sorted(_PRE_FOCUS_HEAT_FIXTURE_WORDS))


def _action_activates_heat_fixture(action: str) -> bool:
    al = (action or "").strip().lower()
    if not al.startswith("activate "):
        return False
    target = al[len("activate ") :].strip()
    return is_heat_fixture_phrase(target) or any(
        w in target for w in _PRE_FOCUS_HEAT_FIXTURE_WORDS
    )
_SINK_DISTRACTOR_RE = re.compile(r"move (.+?) to sink\b")
_STOVE_DISTRACTOR_RE = re.compile(rf"move (.+?) to (?:{_MANIP_FIXTURE_ALT})\b")
_HEAT_TRANSFER_RE = re.compile(rf"\b(?:pour .+ into|move .+ to) (?:{_MANIP_FIXTURE_ALT})\b")


_ARCH_MANIP_SOURCES = frozenset({"post_focus", "transfer", "heat_setup"})


class ActionPolicyEngine:
    """Validates whether an architecture-routed action should be executed."""

    def allows_routed(
        self,
        agent,
        action: str,
        source: str,
        ctx_text: str = "",
        *,
        score: int | None = None,
    ) -> bool:
        allowed, _reason = self.check_routed(
            agent, action, source, ctx_text, score=score,
        )
        return allowed

    def check_routed(
        self,
        agent,
        action: str,
        source: str,
        ctx_text: str = "",
        *,
        score: int | None = None,
    ) -> tuple[bool, str]:
        act = (action or "").strip().lower()
        if not act:
            return False, "empty action"
        task = getattr(agent, "task", "") or ""
        ctx = ctx_text or f"{getattr(agent, 'observation', '')} {getattr(agent, 'inventory', '')}".lower()
        past_actions = list(getattr(agent, "past_actions", []) or [])
        current_score = score if score is not None else int(getattr(agent, "last_score", 0) or 0)

        if is_off_task_planned_action(act, task, ctx):
            return False, "off-task"

        if (
            source in _ARCH_MANIP_SOURCES
            and action_verb_family(act) == "activate"
            and _heat_setup_pending(
                task, past_actions, ctx, current_score,
            )
            and not _action_activates_heat_fixture(act)
        ):
            return False, "non-heat activate during heat setup"

        if (
            source in _ARCH_MANIP_SOURCES
            and action_verb_family(act) == "activate"
            and _substance_transfer_pending(
                task, past_actions, ctx, current_score,
            )
            and not _action_activates_heat_fixture(act)
        ):
            return False, "non-heat activate during transfer"

        if _is_absurd_pour_action(act, task, ctx):
            return False, "absurd pour"

        if _answer_box_action_disallowed(act, task, past_actions):
            return False, "answer selection before measurement"

        if _is_distractor_pour_action(act, task, past_actions, ctx):
            if source in _ARCH_MANIP_SOURCES and _HEAT_TRANSFER_RE.search(act):
                pass
            else:
                return False, "distractor pour"

        if source in _ARCH_MANIP_SOURCES:
            if is_transfer_container_shuffle(
                act, task, ctx, past_actions, current_score=current_score,
            ):
                return False, "transfer container shuffle"
            if _heat_setup_pending(
                task, past_actions, ctx, current_score,
            ) and act.startswith(("go to ", "open door to ")):
                foc_room = _focused_substance_container_room(
                    task, past_actions, ctx,
                )
                if foc_room:
                    dest = (
                        act[len("go to ") :].strip()
                        if act.startswith("go to ")
                        else act.replace("open door to ", "", 1).strip()
                    )
                    if _normalize_room_name(dest) == foc_room:
                        return False, "heat setup blocks substance-room nav"
            if (
                _HEAT_TRANSFER_RE.search(act)
                and action_verb_family(act) in ("pour", "move")
            ):
                return True, "transfer whitelist"
            if _is_post_focus_unproductive(
                act, task, past_actions, ctx, current_score,
            ):
                return False, "post-focus unproductive"
            if _is_absurd_post_focus_manipulation(act, task):
                return False, "absurd manipulation"
            if self._is_distractor_relocation(act, task):
                return False, "distractor relocation"

        if source == "pre_focus" and act.startswith("focus on"):
            if _focus_blocked_by_measurement_gate(
                act, task, past_actions, ctx, getattr(agent, "inventory", "") or "",
            ):
                return False, "measurement pending before focus"

        if source == "pre_focus" and not act.startswith("focus on"):
            fam = action_verb_family(act)
            inv = getattr(agent, "inventory", "") or ""
            if fam in ("pour", "move", "activate", "deactivate"):
                if not _is_measurement_setup_action(
                    act, task, ctx, past_actions, inv,
                ):
                    return False, "pre-focus manipulation"
            if fam in ("pick_up", "take", "use") and _task_needs_measurement_before_focus(
                task, past_actions, ctx, inv,
            ):
                if not _is_measurement_setup_action(
                    act, task, ctx, past_actions, inv,
                ):
                    return False, "pre-focus measurement"

        if source == "pattern" and act.startswith("focus on"):
            if _is_post_focus_unproductive(
                act, task, past_actions, ctx, current_score,
            ):
                return False, "pattern focus unproductive"

        return True, "ok"

    def allows_exploit_substitute(
        self,
        agent,
        action: str,
        env_valid: set,
        ctx_text: str = "",
        *,
        score: int | None = None,
        snap=None,
    ) -> bool:
        """Filter exploit-phase substitutes that stall transfer sub-phases."""
        allowed, _reason = self.check_exploit_substitute(
            agent, action, env_valid, ctx_text, score=score, snap=snap,
        )
        return allowed

    def check_exploit_substitute(
        self,
        agent,
        action: str,
        env_valid: set,
        ctx_text: str = "",
        *,
        score: int | None = None,
        snap=None,
    ) -> tuple[bool, str]:
        act = (action or "").strip().lower()
        if not act:
            return False, "empty"
        task = getattr(agent, "task", "") or ""
        ctx = ctx_text or f"{getattr(agent, 'observation', '')} {getattr(agent, 'inventory', '')}".lower()
        past_actions = list(getattr(agent, "past_actions", []) or [])
        current_score = score if score is not None else int(getattr(agent, "last_score", 0) or 0)

        transfer_p = False
        transfer_exec = False
        heat_setup = False
        if snap is not None:
            transfer_p = bool(snap.transfer_pending)
            transfer_exec = bool(snap.transfer_executable)
            heat_setup = bool(getattr(snap, "heat_setup_pending", False))
        else:
            transfer_p = _substance_transfer_pending(
                task, past_actions, ctx, current_score,
                getattr(agent, "recent_failed_actions", None),
            )
            heat_setup = _heat_setup_pending(
                task, past_actions, ctx, current_score,
                getattr(agent, "recent_failed_actions", None),
            )

        if heat_setup:
            family = action_verb_family(act)
            if family == "activate" and not _action_activates_heat_fixture(act):
                return False, "non-heat activate during heat setup"
            if act.startswith("go to ") or act.startswith("open door to "):
                foc_room = (
                    snap.foc_room if snap else
                    _focused_substance_container_room(task, past_actions, ctx)
                )
                if foc_room:
                    dest = (
                        act[len("go to ") :].strip()
                        if act.startswith("go to ")
                        else act.replace("open door to ", "", 1).strip()
                    )
                    if _normalize_room_name(dest) == foc_room:
                        return False, "heat setup blocks substance-room nav"

        if not transfer_p:
            return True, "ok"

        family = action_verb_family(act)
        if family == "activate" and not _action_activates_heat_fixture(act):
            return False, "non-heat activate during transfer"

        if family in ("meta", "look_around", "look_at", "examine") or act in (
            "inventory", "task", "look around", "wait",
        ):
            if transfer_exec or (snap and snap.current_room == snap.foc_room):
                from core.transfer_resolver import resolve_transfer_fast
                foc_room = snap.foc_room if snap else ""
                cur_room = snap.current_room if snap else ""
                fast = resolve_transfer_fast(
                    env_valid,
                    task,
                    transfer_pending=True,
                    transfer_executable=transfer_exec,
                    current_room=cur_room,
                    foc_room=foc_room,
                    action_history=past_actions,
                    ctx_text=ctx,
                    recent=set(),
                    failed=getattr(agent, "recent_failed_actions", None) or set(),
                    current_score=current_score,
                )
                if fast:
                    return False, "transfer action available"

        if act.startswith("go to ") or act.startswith("open door to "):
            if snap and snap.foc_room and snap.current_room == snap.foc_room:
                sub = snap.som_sub_phase or transfer_sub_phase(
                    task, past_actions, ctx,
                    current_room=snap.current_room,
                    foc_room=snap.foc_room,
                    current_score=current_score,
                )
                dest = (
                    act[len("go to ") :].strip()
                    if act.startswith("go to ")
                    else act.replace("open door to ", "", 1).strip()
                )
                dest_norm = _normalize_room_name(dest)
                delivery_rooms = {
                    _normalize_room_name(r)
                    for r in task_manipulation_destination_rooms(
                        task, ctx, past_actions, env_valid,
                    )
                }
                foc_norm = _normalize_room_name(snap.foc_room) if snap.foc_room else ""
                if sub == "transfer_nav_source" and foc_norm and dest_norm == foc_norm:
                    return True, "ok"
                if sub == "transfer_nav" and dest_norm in delivery_rooms:
                    return True, "ok"
                if sub == "transfer_prep":
                    return False, "nav away from substance room during prep"
                return False, "nav away from substance room"

        if is_transfer_container_shuffle(
            act, task, ctx, past_actions, current_score=current_score,
        ):
            return False, "transfer container shuffle"

        return True, "ok"

    @staticmethod
    def _is_distractor_relocation(action: str, task: str) -> bool:
        """Block moving unrelated visible objects (food) during substance tasks."""
        al = (action or "").strip().lower()
        if action_verb_family(al) != "move":
            return False
        task_t = _task_target_tokens(task)
        for pattern in (_SINK_DISTRACTOR_RE, _STOVE_DISTRACTOR_RE):
            m = pattern.search(al)
            if not m:
                continue
            obj_t = _action_content_tokens(m.group(1))
            if obj_t and not (obj_t & task_t):
                foodish = obj_t & {
                    "apple", "banana", "orange", "potato", "egg",
                    "bread", "tomato", "carrot",
                }
                if foodish:
                    return True
        return False
