"""
Episode phase detection, focus/heat/transfer/substance semantics.
"""
from __future__ import annotations

import re

import numpy as np


from core.scienceworld_schema import *  # noqa: F403
from core.action_primitives import *  # noqa: F403

import core.action_primitives as _action_primitives
from core._grounding_merge import merge_module_exports
from core.environment_discovery import (
    active_heat_fixtures,
    any_heat_fixture_in_context,
    closed_containers_in_observation,
    discover_fixtures_from_observation,
    fixture_is_active,
    fixture_is_unavailable,
    fixture_last_seen_room,
    fixture_visible_in_context,
    heat_destination_fixtures,
    heat_fixtures_in_environment,
    heating_progress_stalled,
    is_heat_fixture_phrase,
    is_cool_fixture_phrase,
    pick_next_heat_escalation_action,
    primary_heat_fixture_unavailable,
    room_visit_counts,
)

merge_module_exports(globals(), _action_primitives)

def make_episode_progress():
    return {
        "score_max": 0,
        "steps_since_gain": 0,
        "steps_since_relevant_state_change": 0,
        "last_gain_family": "",
    }


def update_episode_progress(
    prog,
    prev_score,
    current_score,
    action,
    *,
    progress_kind: str = "",
    meaningful_change: bool | None = None,
):
    kind = (progress_kind or "").strip().lower()
    if not kind:
        kind = "reward" if current_score > prev_score else (
            "enabling" if meaningful_change else "no_op"
        )
    prog.setdefault("steps_since_relevant_state_change", 0)
    if current_score > prev_score:
        prog["score_max"] = max(prog["score_max"], current_score)
        prog["steps_since_gain"] = 0
        prog["steps_since_relevant_state_change"] = 0
        prog["last_gain_family"] = action_verb_family(action or "")
    else:
        prog["steps_since_gain"] += 1
        if kind == "enabling":
            prog["steps_since_relevant_state_change"] = 0
        else:
            prog["steps_since_relevant_state_change"] += 1
    return prog


def is_exploit_phase(prog, current_score):
    return (
        prog.get("score_max", 0) > 0
        and current_score < 100
        and prog.get("steps_since_gain", 0) >= EXPLOIT_STAGNANT_THRESHOLD
    )


def is_post_focus_manipulation_exploit(
    task: str,
    action_history,
    observation: str,
    inventory: str,
    current_score: int | None,
) -> bool:
    """Treat post-focus manipulation as exploit so grounding prefers connect/pick/etc."""
    ctx_text = f"{observation or ''} {inventory or ''}"
    return _task_post_focus_manipulation_pending(
        task, action_history, ctx_text, current_score,
    )


def should_extend_trajectory_for_subgoals(
    task: str,
    action_history,
    observation: str,
    inventory: str,
    current_score: int,
    score_max: int,
    *,
    at_trajectory_end: bool,
    termination: bool,
    steps_taken: int,
    step_limit: int,
    base_look_ahead: int = 3,
    force_incomplete: bool = False,
) -> tuple[bool, str, int]:
    """
    Extend short look-ahead trajectories when a multi-phase subgoal is still open.
    Task-agnostic: focus pending, or post-focus manipulation (connect, growth, etc.).
    ``force_incomplete`` covers env families (e.g. AlfWorld) that stay at score 0
    until success and still need more steps.
    """
    if not at_trajectory_end or termination or current_score >= 100:
        return False, "", 0
    if steps_taken >= step_limit:
        return False, "", 0
    ctx_text = f"{observation or ''} {inventory or ''}"
    if _task_post_focus_manipulation_pending(
        task, action_history, ctx_text, current_score,
    ):
        extra = max(5, base_look_ahead + 2)
        return True, "post-focus manipulation pending", extra
    if _focus_subgoal_pending(
        task, action_history, ctx_text, current_score,
    ) and score_max < 100:
        return True, "focus subgoal pending", max(3, base_look_ahead)
    if force_incomplete and current_score < 100:
        remain = max(1, int(step_limit) - int(steps_taken))
        extra = min(remain, max(15, int(base_look_ahead or 3) * 3))
        return True, "incomplete episode budget", extra
    return False, "", 0


def replan_look_ahead(base_look_ahead: int) -> int:
    """Shorter horizon on replan to cut 9B planning cost without changing initial plan depth."""
    return max(2, int(base_look_ahead) - 1)


def _is_low_cost_navigation_action(action: str) -> bool:
    al = (action or "").strip().lower().replace("green house", "greenhouse")
    if al.startswith(("go to ", "open door to ")):
        return True
    fam = action_verb_family(al)
    return fam in ("look_around", "look_at", "look_in", "examine", "meta")


def wait_action_allowed(
    action: str,
    task: str,
    ctx_text: str,
    action_history=None,
    current_score: int | None = None,
) -> bool:
    """Allow wait only during substance fill or post-focus heat/cool phases."""
    al = (action or "").strip().lower()
    if al not in ("wait",) and not al.startswith("wait "):
        return True
    if (
        _substance_prep_phase(task, ctx_text, action_history) == "wait_fill"
        and _container_at_liquid_source(ctx_text, action_history)
    ):
        return True
    post_phase = _state_of_matter_post_focus_phase(
        task, current_score, action_history, ctx_text,
    )
    if post_phase in ("heat", "cool", "precool"):
        return True
    # Growth / life-stage tasks often require waiting after planting/watering.
    if (
        _task_needs_growth_after_focus(task)
        and _task_focus_already_satisfied(task, action_history, ctx_text, current_score)
    ):
        return True
    if _focus_subgoal_pending(task, action_history, ctx_text, current_score):
        return False
    if (current_score or 0) == 0 and _task_needs_substance_preparation(task):
        return False
    return post_phase is not None


def can_bypass_actor(
    planned: str,
    env_valid: set,
    task: str,
    ctx_text: str,
    action_history=None,
) -> bool:
    """Skip 9B Actor when planner action is already env-valid and low-risk."""
    p = (planned or "").strip().lower().replace("green house", "greenhouse")
    if not p or p not in env_valid:
        return False
    if is_off_task_planned_action(p, task, ctx_text):
        return False
    if p.startswith(("focus on", "teleport", "move ", "connect ", "pick up ", "take ")):
        return False
    fam = action_verb_family(p)
    if fam in ("pick_up", "take", "pour", "activate", "deactivate", "connect", "use", "mix", "put"):
        return False
    # Never bypass meta / look-around: Swift seed plans thrash on these and
    # starve process verbs (pour/mix/nav-to-preferred). Observe verbs stay Actor/fast.
    if fam in ("meta", "look_around") or p in {"task", "inventory", "look around"}:
        return False
    past = list(action_history or [])
    if fam in ("look_at", "look_in", "examine"):
        # Allow at most one consecutive observe; repeats must go through fast/Actor.
        recent_obs = sum(
            1 for a in past[-3:]
            if action_verb_family(a) in ("look_at", "look_in", "examine", "look_around", "meta")
        )
        if recent_obs >= 1:
            return False
    if _focus_subgoal_pending(task, action_history, ctx_text, None):
        if fam in ("look_at", "examine"):
            prep_phase = _substance_prep_phase(task, ctx_text, action_history)
            if prep_phase == "solid_search":
                return False
            candidates = _collect_visible_focus_candidates(
                env_valid, ctx_text, task, set(), action_history=action_history,
            )
            if candidates:
                return False
        if _substance_examined_pending_focus(task, action_history, ctx_text):
            return False
    if fam == "wait":
        return wait_action_allowed(p, task, ctx_text, action_history)
    # True navigation only (go/open-door) — not look/meta (see above).
    if p.startswith(("go to ", "open door to ")):
        if _is_in_exploration_navigation_phase(task, ctx_text, action_history):
            if _is_deprioritized_nav_dest(p, task):
                return False
            if _is_blocked_explore_nav_substitute(
                p, task, ctx_text, action_history, env_valid,
            ):
                return False
        return True
    if fam == "open" and not p.startswith("open door"):
        return True
    return False


def compute_perception_skip_flags(
    episode_progress,
    last_score: int,
    stagnant_steps: int,
    step_index: int,
    last_action: str = "",
):
    """
    Decide which perception LLM calls to skip on the upcoming env step.
    Conservative while score_max==0; more aggressive once the agent has made progress.
    """
    score_max = episode_progress.get("score_max", 0)
    steps_since_gain = episode_progress.get("steps_since_gain", 0)
    plateau_stagnant = (
        score_max > 0
        and last_score >= score_max
        and steps_since_gain >= EXPLOIT_STAGNANT_THRESHOLD
    )
    score_at_peak = score_max > 0 and last_score >= score_max
    nav_action = _is_low_cost_navigation_action(last_action)

    skip_kg = False
    skip_action_reflect = False
    skip_wm_refine = nav_action

    if score_max > 0:
        if plateau_stagnant:
            skip_kg = step_index % 2 == 1
            skip_action_reflect = stagnant_steps >= 1
            skip_wm_refine = True
        elif score_at_peak and steps_since_gain >= 1:
            skip_kg = step_index % 2 == 1
            skip_action_reflect = nav_action or stagnant_steps >= 2
            if nav_action:
                skip_wm_refine = True
    elif nav_action and step_index >= 2:
        skip_action_reflect = True
        skip_wm_refine = True
        skip_kg = step_index % 3 != 0

    return skip_kg, skip_action_reflect, skip_wm_refine


def should_force_heavy_perception(
    agent,
    *,
    stagnant_steps: int,
    last_score: int,
) -> bool:
    """
    Task-agnostic perception triggers (failure, stagnation, uncertainty).

    Replaces environment-specific flags like heat_setup_pending / transfer_pending
    for deciding when to re-enable heavy perception while stuck.
    """
    if agent is None:
        return stagnant_steps >= 5

    episode_progress = dict(getattr(agent, "episode_progress", {}) or {})
    failed = getattr(agent, "recent_failed_actions", None) or set()
    if failed:
        return True

    score_delta = getattr(agent, "_last_step_score_delta", None)
    if score_delta is not None and score_delta < 0:
        return True

    score_max = int(episode_progress.get("score_max", 0) or 0)
    steps_since_gain = int(episode_progress.get("steps_since_gain", 0) or 0)
    if score_max > 0 and steps_since_gain >= 5 and stagnant_steps >= 5:
        return True

    ctx = _agent_live_context(agent)
    cur_room = _extract_current_room_name(ctx)
    prev_room = getattr(agent, "_prev_perception_room", "") or ""
    if prev_room and cur_room and prev_room != cur_room:
        return True

    prev_obs = getattr(agent, "_prev_obs_snapshot", "") or ""
    cur_obs = (getattr(agent, "observation", "") or "").strip()
    if prev_obs and cur_obs and prev_obs != cur_obs and stagnant_steps >= 3:
        prev_tokens = set(re.findall(r"\b[a-z]{4,}\b", prev_obs.lower()))
        cur_tokens = set(re.findall(r"\b[a-z]{4,}\b", cur_obs.lower()))
        if prev_tokens and cur_tokens:
            union = prev_tokens | cur_tokens
            overlap = len(prev_tokens & cur_tokens) / max(len(union), 1)
            if overlap < 0.5:
                return True

    return False


def is_exploration_compatible(planned: str, executed: str) -> bool:
    pf = action_verb_family(planned)
    ef = action_verb_family(executed)
    if pf == ef:
        return True
    if pf in EXPLORATION_VERB_FAMILIES and ef in EXPLORATION_VERB_FAMILIES:
        return True
    el = (executed or "").strip().lower()
    if pf in EXPLORATION_VERB_FAMILIES and (
        el.startswith("open door to ")
        or el.startswith("go to ")
        or el.startswith("look in ")
    ):
        return True
    pt = _action_content_tokens(planned)
    et = _action_content_tokens(executed)
    if not pt or not et:
        return True
    return bool(pt & et)


def should_skip_refiner_llm(
    env_failed: bool,
    current_score: int,
    prev_score: int,
    planned: str,
    executed: str,
    steps_since_gain: int,
) -> bool:
    """Skip 9B Refiner when execution matches plan and score is stable."""
    if env_failed or current_score < prev_score:
        return False
    if (
        current_score >= prev_score
        and steps_since_gain >= STAGNANT_STEPS_BEFORE_REPLAN
    ):
        return True
    if is_exploration_compatible(planned, executed) and current_score >= prev_score:
        return True
    if (
        current_score >= prev_score
        and (
            _is_low_cost_navigation_action(planned)
            or _is_low_cost_navigation_action(executed)
        )
    ):
        return True
    if (
        current_score > 0
        and current_score >= prev_score
        and action_verb_family(executed) == action_verb_family(planned)
    ):
        return True
    return False


def detect_navigation_loop(action_history, window=None):
    window = window or (NAV_LOOP_MIN_LEN * 2 + 2)
    hist = [(a or "").strip().lower() for a in (action_history or [])[-window:]]
    nav_families = [
        action_verb_family(a)
        for a in hist
        if a.startswith(("go to ", "open door to "))
    ]
    if len(nav_families) >= 4:
        for size in (2, 3):
            tail = nav_families[-size * 2:]
            if len(tail) == size * 2 and tail[:size] == tail[size:]:
                return True
    go_dests = [a[len("go to ") :].strip() for a in hist if a.startswith("go to ")]
    if len(go_dests) >= 3 and len(set(go_dests[-3:])) == 1:
        return True
    if len(hist) >= 4:
        toggles = [action_verb_family(a) for a in hist[-4:] if action_verb_family(a) in _TOGGLE_VERB_FAMILIES]
        if len(toggles) >= 4:
            return True
    return False


def effective_max_replans(replan_count, steps_since_gain=0, score_max=0, base_max=DEFAULT_MAX_REPLANS):
    """Allow extra replans when score has plateaued after prior progress."""
    bonus = 0
    if score_max > 0 and steps_since_gain >= EXPLOIT_STAGNANT_THRESHOLD:
        bonus = 1
    if score_max >= 26 and steps_since_gain < STAGNANT_EARLY_STOP_THRESHOLD:
        bonus += 1
    return base_max + bonus
def _is_absurd_pour_action(action: str, task: str = "", ctx_text: str = "") -> bool:
    """Block self-pour and pouring into rooms instead of heat/prep vessels."""
    al = (action or "").strip().lower()
    if action_verb_family(al) != "pour":
        return False
    m = re.match(r"^pour (.+?) into (.+)$", al)
    if not m:
        return False
    src, dest = m.group(1).strip(), m.group(2).strip()
    src_core = _entity_core_phrase(src)
    dest_core = _entity_core_phrase(dest)
    if src_core and dest_core and src_core == dest_core:
        return True
    if src == dest:
        return True
    dest_tokens = _action_content_tokens(dest)
    if dest_core in SCIENCEWORLD_ROOM_NAMES:
        return True
    if dest_tokens & set(SCIENCEWORLD_ROOM_NAMES):
        return True
    if "bathroom" in dest and not (dest_tokens & {"oven", "stove", "toilet", "sink"}):
        return True
    if dest in ("inventory", "agent") or dest_core in ("inventory", "agent"):
        if _task_needs_substance_preparation(task) or _task_has_post_focus_phase(task):
            return True
    if dest_core in {
        "table", "counter", "desk", "chair", "bed", "floor", "ground",
        "wall", "door",
    }:
        return True
    if dest_tokens & {
        "table", "counter", "desk", "chair", "bed", "floor", "ground",
    } and not (dest_tokens & {"pot", "cup", "bowl", "pan", "beaker", "flask", "jar"}):
        return True
    # Answer-box pours are gated by ``_answer_selection_pending`` (needs history);
    # do not treat them as unconditionally absurd here.
    return False


def _agent_live_context(agent) -> str:
    """Prefer fresh env.look() over cached observation for grounding decisions."""
    observation = getattr(agent, "observation", "") or ""
    inventory = getattr(agent, "inventory", "") or ""
    if hasattr(agent, "env") and agent.env is not None:
        try:
            fresh = agent.env.look() or ""
            if fresh:
                observation = fresh
        except Exception:
            pass
    return f"{observation} {inventory}".lower()


def _look_confirms_task_substance(look_obj: str, task: str, ctx_text: str = "") -> bool:
    """True when a look/examine target confirms the task substance."""
    obj = (look_obj or "").strip().lower()
    if not obj:
        return False
    if _false_task_substance_match(obj, task, ctx_text):
        return False
    targets = _specific_task_tokens(task) or _task_target_tokens(task)
    if not targets:
        return False
    obj_core = _entity_core_phrase(obj)
    for t in targets:
        if t in obj or obj_core == t or obj == t or t in _action_content_tokens(obj):
            if _false_task_substance_match(obj, task, ctx_text):
                continue
            return True
    if "substance in" in obj and _focus_matches_task_substance(obj, task, ctx_text):
        return True
    container = obj.replace("substance in ", "", 1).strip() if "substance in" in obj else obj
    if container and _container_holds_task_substance(container, task, ctx_text):
        return True
    return False


def _is_distractor_pour_action(
    action: str,
    task: str,
    action_history=None,
    ctx_text: str = "",
) -> bool:
    """Pour from unrelated visible objects during a substance task."""
    al = (action or "").strip().lower()
    if action_verb_family(al) != "pour":
        return False
    m = re.match(r"^pour (.+?) into ", al)
    if not m:
        return False
    src = m.group(1).strip()
    src_t = _action_content_tokens(src)
    task_t = _task_target_tokens(task)
    if src_t & task_t:
        return False
    if _object_matches_focused_substance(src, task, action_history, ctx_text):
        return False
    if _container_holds_task_substance(src, task, ctx_text, action_history):
        return False
    if src_t & _DISTRACTOR_FOOD_OBJECTS:
        return True
    if src_t & _DISTRACTOR_VESSEL_CONTENT_WORDS:
        return True
    junk = {
        "chair", "counter", "table", "bowl", "cup", "apple", "banana",
        "orange", "potato", "drawer", "cupboard", "fridge", "freezer",
        "battery", "switch", "bulb", "wire", "motor", "buzzer", "anode", "cathode",
        "paint", "ink",
    }
    return bool(src_t & junk)


def _is_absurd_post_focus_manipulation(action: str, task: str) -> bool:
    """Pour/pick/move on fixed or circuit components after focus milestone."""
    al = (action or "").strip().lower()
    family = action_verb_family(action)
    obj = _action_manipulation_object(action)
    if not obj:
        return False
    if _is_fixed_installation_target(obj):
        if _task_requires_connect(task) and family in ("pick_up", "take"):
            obj_tokens = _action_content_tokens(obj)
            if obj_tokens & (_CIRCUIT_HINT_TOKENS | _task_target_tokens(task)):
                return False
        return family in ("pour", "pick_up", "take", "move")
    if _task_requires_connect(task) and family in ("pour", "pick_up", "take"):
        if _action_content_tokens(obj) & _CIRCUIT_HINT_TOKENS:
            return True
    if family == "pour" and re.search(r"\bpour .+ into (?:inventory|agent)\b", al):
        if _is_fixed_installation_target(obj):
            return True
        if not _is_prep_container_object(obj) and "substance" not in obj:
            if _action_content_tokens(obj) & (
                _CIRCUIT_HINT_TOKENS | _NON_PORTABLE_PICK_WORDS
            ):
                return True
    if _task_has_post_focus_phase(task) and family == "move":
        m = re.match(r"move (.+?) to (.+)", al)
        if m and is_heat_fixture_phrase(m.group(2)):
            obj_t = _action_content_tokens(m.group(1))
            task_t = _task_target_tokens(task)
            if obj_t & _DISTRACTOR_FOOD_OBJECTS and not (obj_t & task_t):
                return True
    return False


def _task_needs_growth_after_focus(task: str) -> bool:
    """Tasks that require environmental changes after focusing a seed/plant."""
    task_l = (task or "").lower()
    if not re.search(r"\bfocus\b", task_l):
        return False
    has_organism = bool(
        re.search(r"\b(seed|plant|plants|fruit|flower|reproduction|life\s*stage)\b", task_l)
    )
    has_growth = bool(
        re.search(r"\bgrow(?:s|ing|th)?\b|\breproduc", task_l)
        or re.search(r"make changes.{0,40}environment|change.{0,30}environment", task_l)
    )
    return has_organism and has_growth


def _solid_substance_visible(task: str, ctx_text: str) -> bool:
    if _task_needs_liquid_fixture_prep(task, ctx_text):
        return False
    return not _task_target_missing(task, ctx_text)


def _solid_prep_needs_room_search(task: str, ctx_text: str) -> bool:
    """
    True when the task substance/entity is not visible and every *useful*
    closable container in the current room has been inspected — navigate
    elsewhere (task-agnostic). Polarity-wrong containers (freezer on boil,
    oven as "storage") do not block room search.
    Also true for connect tasks searching for a missing component.
    """
    if not _task_target_missing(task, ctx_text):
        return False
    if _task_requires_connect(task):
        return True
    if not _task_needs_substance_preparation(task):
        return False
    if _task_needs_liquid_fixture_prep(task, ctx_text):
        return False
    useful = [
        c
        for c in _closed_storage_containers_in_context(ctx_text)
        if _storage_useful_for_substance_search(c, task)
    ]
    if useful:
        return False
    return True


def _is_relocation_to_room(action: str) -> bool:
    al = (action or "").strip().lower()
    m = re.match(r"move .+? to (.+)", al)
    if not m:
        return False
    dest = _normalize_room_name(m.group(1))
    return dest in SCIENCEWORLD_ROOM_NAMES


def _is_object_manipulation_move(action: str) -> bool:
    """True when move targets a fixture/object, not a room (pseudo-teleport)."""
    al = (action or "").strip().lower()
    return al.startswith("move ") and not _is_relocation_to_room(action)


def _is_disallowed_substitute_action(action: str) -> bool:
    """Reject menu disambiguation digits and other non-commands."""
    al = (action or "").strip()
    return bool(re.fullmatch(r"\d+", al))


def _is_destructive_pour_to_room(action: str) -> bool:
    al = (action or "").strip().lower()
    m = re.match(r"pour .+? into (.+)", al)
    if not m:
        return False
    dest = _normalize_room_name(m.group(1).strip())
    return dest in SCIENCEWORLD_ROOM_NAMES


def _task_requires_connect(task: str) -> bool:
    """True when task wording implies building/connecting components (not task-ID specific)."""
    task_l = (task or "").lower()
    if extract_task_content_tokens(task) & _CIRCUIT_HINT_TOKENS:
        return True
    return bool(re.search(r"\bcircuit\b|electrical|connect|powers?\s+it\s+on|power\s+on", task_l))


def _task_has_post_focus_phase(task: str) -> bool:
    """True for multi-step tasks: initial focus/setup, then further environment changes."""
    task_l = (task or "").lower()
    if _task_is_threshold_measurement_task(task):
        # Measure-then-box tasks need instrument use / heat, not melt/boil post-focus.
        return False
    # Avoid treating "melting point" as a melt post-focus task.
    stripped = re.sub(r"\b(melting|boiling|freezing)\s+point\b", " ", task_l)
    if re.search(
        r"\b(melt|melts|melted|boil|boils|boiled|freeze|freezes|froze|frozen|"
        r"state\s+of\s+matter|change.{0,40}state.{0,40}matter)\b",
        stripped,
    ) and re.search(r"\bfocus\b", task_l):
        return True
    if _task_requires_connect(task) and re.search(r"\bfocus\b", task_l):
        return True
    return bool(
        re.search(r"first,.+then,", task_l, re.S)
        or re.search(r"then,.+(?:until|stage|complete|finish)", task_l, re.S)
        or re.search(r"make changes|change.{0,30}environment|life stage|reproduction", task_l)
    )


def _task_post_focus_manipulation_pending(
    task: str,
    action_history=None,
    ctx_text: str = "",
    current_score: int | None = None,
) -> bool:
    """True after focus milestone when further manipulation is still required."""
    if current_score is not None and current_score >= 100:
        return False
    if not _task_focus_already_satisfied(
        task, action_history, ctx_text, current_score,
    ):
        return False
    if _task_requires_connect(task):
        return True
    if _task_has_post_focus_phase(task):
        return True
    if _task_needs_growth_after_focus(task):
        return True
    return False


def _task_target_in_inventory(task: str, ctx_text: str) -> bool:
    """True when a task-relevant object token appears in inventory text."""
    inv = _inventory_text(ctx_text)
    if not inv:
        return False
    targets = _specific_task_tokens(task) or _task_target_tokens(task)
    if not targets:
        return False
    inv_tokens = _action_content_tokens(inv)
    if targets & inv_tokens:
        return True
    return any(re.search(rf"\b{re.escape(t)}\b", inv) for t in targets if len(t) > 2)


def _is_counterproductive_plateau_action(
    action: str,
    task: str,
    action_history=None,
    ctx_text: str = "",
    current_score: int | None = None,
) -> bool:
    """Block wait/disconnect/detours while post-focus manipulation remains."""
    if not _task_post_focus_manipulation_pending(
        task, action_history, ctx_text, current_score,
    ):
        return False
    al = (action or "").strip().lower()
    family = action_verb_family(action)
    if al.startswith("disconnect") or family == "disconnect":
        return True
    if family == "deactivate":
        return True
    if family == "wait":
        return not wait_action_allowed(
            action, task, ctx_text, action_history, current_score,
        )
    if family == "activate" and _task_requires_connect(task):
        fixture = re.sub(r"^(activate|deactivate)\s+", "", al).strip()
        fix_tokens = _action_content_tokens(fixture)
        task_relevant = (
            _task_target_tokens(task) | _CIRCUIT_HINT_TOKENS | _specific_task_tokens(task)
        )
        if not (fix_tokens & task_relevant):
            return True
    if family in ("look_at", "look_in", "examine") and _task_requires_connect(task):
        obj = _action_manipulation_object(action)
        if obj and _action_content_tokens(obj) & (
            _CIRCUIT_HINT_TOKENS | _task_target_tokens(task)
        ):
            return True
    return False


def _requires_focus_before_manipulation(
    task: str, action_history=None, ctx_text: str = "",
    current_score: int | None = None,
) -> bool:
    """Block pick/connect/etc. until a task-relevant focus has succeeded."""
    if _task_focus_already_satisfied(task, action_history, ctx_text, current_score):
        return False
    task_l = (task or "").lower()
    if re.search(r"\bfocus\b", task_l):
        return True
    if _task_has_post_focus_phase(task):
        return True
    if _task_requires_connect(task):
        return True
    return False


def _task_expects_context_entity_focus(task: str) -> bool:
    """Task asks to focus/compare entities without naming the exact answer object."""
    task_l = (task or "").lower()
    if re.search(r"\bunknown\s+substance\b", task_l):
        return True
    if not re.search(r"\bfocus\b", task_l):
        return False
    # Find-and-deliver / identify: category nouns (plant, living thing) plus
    # destination colour/room tokens must not force Actor off-task on the
    # live instance (peach tree / bee / pillow).
    try:
        from core.pattern_library import _task_family_flags
        fam = _task_family_flags(task)
        if fam.get("identify") and not (
            fam.get("circuit")
            or fam.get("measure")
            or fam.get("mechanics")
            or fam.get("som")
            or fam.get("chemistry")
        ):
            return True
    except Exception:
        pass
    if _COMPARISON_TASK_RE.search(task_l):
        # Genetics names the answer boxes; do not pick ambient organisms.
        sem = _extract_comparison_semantics(task)
        if sem and sem[0] == "genetics":
            return False
        return True
    specific = _specific_task_tokens(task)
    if not specific:
        return True
    if specific <= _GENERIC_FOCUS_ENTITY_WORDS:
        return True
    vague_only = specific - _GENERIC_FOCUS_ENTITY_WORDS
    if not vague_only or vague_only <= _FOCUS_VAGUE_OBJECT_WORDS:
        return True
    return False


def _object_lacks_required_task_tokens(obj_tokens: set[str], task: str) -> bool:
    """True when object partially overlaps task nouns but misses mandatory ones."""
    instructed = _extract_focus_instruction_tokens(task)
    if instructed:
        if obj_tokens & instructed:
            return False
        spill = obj_tokens & (
            _specific_task_tokens(task) - instructed - _GENERIC_FOCUS_ENTITY_WORDS
        )
        if spill and _task_needs_substance_preparation(task):
            # ``tin cup`` shares ``tin`` with boil-tin but instructs ``substance``.
            vesselish = obj_tokens & (
                set(_TASK_SUBSTANCE_CONTAINER_WORDS) | set(_PRE_FOCUS_PREP_CONTAINER_HINTS)
            )
            if vesselish:
                return False
        return bool(spill)
    required = _specific_task_tokens(task)
    if not required or not obj_tokens:
        return False
    if required <= obj_tokens:
        return False
    overlap = obj_tokens & required
    if not overlap:
        return False
    return bool(required - obj_tokens)


def _is_forbidden_manipulation_target(obj_phrase: str) -> bool:
    obj_l = (obj_phrase or "").strip().lower()
    if not obj_l:
        return True
    if "door" in obj_l and "indoor" not in obj_l:
        return True
    tokens = _action_content_tokens(obj_l)
    return bool(tokens & _FORBIDDEN_MANIPULATION_TARGETS)


def _is_fixed_installation_target(obj_phrase: str) -> bool:
    """Non-portable objects that should not be poured/picked (bulbs, fixtures, etc.)."""
    tokens = _action_content_tokens(obj_phrase)
    if tokens & frozenset({"bulb", "switch", "battery", "thermometer", "stopwatch"}):
        return True
    if "light" in tokens and "bulb" in tokens:
        return True
    return bool(tokens & _NON_PORTABLE_PICK_WORDS)


def _current_room_in_task_area(task: str, ctx_text: str) -> bool:
    """True when agent should keep searching locally (not cross-room solid search)."""
    if _solid_prep_needs_room_search(task, ctx_text):
        return False
    current = _extract_current_room_name(ctx_text)
    if not current:
        return False
    hints = _extract_task_location_hint_rooms(task)
    if hints and current in hints:
        return True
    if not _task_target_missing(task, ctx_text):
        return True
    if _closed_storage_containers_in_context(ctx_text):
        return True
    return bool(_context_task_anchor_tokens(task, ctx_text))


def _requires_location_before_focus(task: str, ctx_text: str) -> bool:
    """True when the agent must navigate to the task work area before focusing."""
    hints = _extract_task_location_hint_rooms(task)
    if hints:
        current = _extract_current_room_name(ctx_text)
        if not current:
            return True
        if current in hints:
            return False
        if _task_target_visible(task, ctx_text) or _task_substance_visible(task, ctx_text):
            return False
        return True
    # Structural primary work area (greenhouse, workshop, outside, …).
    # Secondary rooms in the preferred list are hops, not arrival.
    try:
        from core.navigation_helpers import _already_in_work_home
        if _already_in_work_home(task, ctx_text):
            return False
        if _task_target_visible(task, ctx_text) or _task_substance_visible(task, ctx_text):
            return False
        preferred = _preferred_target_rooms(task, ctx_text=ctx_text)
        if not preferred:
            return False
        return True
    except Exception:
        preferred = _preferred_target_rooms(task, ctx_text=ctx_text)
        primary = [r for r in preferred if r not in _NAV_CONNECTOR_ROOMS]
        if not primary:
            return False
        current = _extract_current_room_name(ctx_text)
        if not current:
            return True
        if current == primary[0]:
            return False
        if _task_target_visible(task, ctx_text) or _task_substance_visible(task, ctx_text):
            return False
        return True


def _task_is_threshold_measurement_task(task: str) -> bool:
    """
    Measure a quantity (often with a named instrument), then focus on an answer box.

    Distinct from phase-change *prep* tasks (``melt chocolate`` / ``boil water``):
    ``melting point`` / ``boiling point`` are measurement nouns, not melt/boil goals.
    """
    task_l = (task or "").lower()
    if re.search(r"\b(?:dominant|recessive|allele|genetic|mendel)\b", task_l):
        return False
    if not re.search(r"\b(measure|determine|find|read)\b", task_l):
        return False
    has_quantity = bool(
        re.search(
            r"\b(melting\s+point|boiling\s+point|freezing\s+point|"
            r"temperature|celsius|fahrenheit|degrees)\b",
            task_l,
        )
        or _measurement_tools_mentioned(task_l)
    )
    if not has_quantity:
        return False
    # Explicit measure-then-box wording (may be truncated mid-sentence in some
    # agent.task copies — still treat named *point* quantities as threshold).
    if re.search(
        r"\bif\b.{0,240}\bfocus on\b.{0,100}\b(?:box|answer)\b"
        r"|\bfocus on (?:the )?(?:blue|orange|yellow|red|green|black|white|pink|purple)\s*box\b",
        task_l,
        re.S,
    ):
        return True
    # Melting/boiling/freezing-point measurement is always the threshold family
    # even when the answer-box clause is missing from a truncated task string.
    return bool(
        re.search(r"\b(?:melting|boiling|freezing)\s+point\b", task_l)
    )


def _answer_box_destination(text: str) -> bool:
    """True when a phrase names a colored/answer box receptacle."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if re.search(r"\banswer\s+box\b", t):
        return True
    return bool(
        re.search(
            r"\b(?:blue|orange|yellow|red|green|black|white|pink|purple)\s+box\b",
            t,
        )
        or re.search(r"\bthe box\b", t)
    )


def _involves_answer_box_receptacle(action: str) -> bool:
    """True when pour/move/focus/use treats an answer box as source or destination."""
    al = (action or "").strip().lower()
    fam = action_verb_family(al)
    if fam == "focus" and al.startswith("focus on "):
        return _answer_box_destination(al[len("focus on "):])
    if fam == "use":
        # ``use thermometer on blue box`` is never informative setup.
        return _answer_box_destination(_use_action_patient(al))
    if fam == "pour":
        m = re.match(r"^pour (.+?) into (.+)$", al)
        if not m:
            return False
        return _answer_box_destination(m.group(1)) or _answer_box_destination(m.group(2))
    if fam == "move":
        m = re.match(r"^move (.+?) to (.+)$", al)
        if not m:
            return False
        return _answer_box_destination(m.group(1)) or _answer_box_destination(m.group(2))
    return False


def _is_answer_selection_action(action: str) -> bool:
    """Focus/pour/move that commits to (or shuffles) an answer-box receptacle."""
    return _involves_answer_box_receptacle(action)


def _answer_selection_pending(task: str, action_history=None) -> bool:
    """True until the task-relevant measuring instrument has been used."""
    if not _task_is_threshold_measurement_task(task):
        return False
    return not _measurement_tool_already_used(action_history, task)


def _answer_box_action_disallowed(
    action: str, task: str, action_history=None, ctx_text: str = "",
) -> bool:
    """
    Gate answer-box commits for measure-then-box and mutually exclusive if-boxes.

    - Focus on a box is only valid after the instrument has been used (threshold)
      or when the matching if-clause is justified (conditional genetics).
    - Pour/move involving answer boxes is never the instructed answer.
    - ``use <instrument> on <box>`` is never a valid measurement patient.
    """
    if not _involves_answer_box_receptacle(action):
        return False
    fam = action_verb_family(action)
    if _task_is_threshold_measurement_task(task):
        if fam in ("pour", "move"):
            return True
        if fam == "use":
            return True
        if fam == "focus":
            return _answer_selection_pending(task, action_history)
        return False
    if _mutually_exclusive_answer_boxes(task):
        if fam in ("pour", "move", "use"):
            return True
        if fam == "focus":
            return not _conditional_answer_commit_allowed(
                action, task, action_history, ctx_text,
            )
    return False



def _task_substance_focus_done(
    task: str,
    action_history=None,
    ctx_text: str = "",
) -> bool:
    """True after focusing the measured substance (not merely the instrument)."""
    tools = _measurement_task_tools(task) | _PRE_FOCUS_PREP_TOOL_WORDS
    targets = _specific_task_tokens(task) or _task_target_tokens(task)
    substance = {
        t for t in targets
        if t not in tools and t not in _FOCUS_VAGUE_OBJECT_WORDS
        and t not in {"box", "answer", "point", "kitchen", "unknown"}
    }
    hist = [(a or "").strip().lower() for a in (action_history or [])[-24:]]
    for act in reversed(hist):
        if not act.startswith("focus on "):
            continue
        obj = act[len("focus on "):].strip()
        if _answer_box_destination(obj):
            continue
        ot = _action_content_tokens(obj)
        if ot & tools and not (ot & substance):
            continue
        if substance and (ot & substance):
            return True
        if "substance" in obj and _focus_matches_task_substance(obj, task, ctx_text):
            return True
    return False


def _task_needs_substance_preparation(task: str) -> bool:
    """True when the task substance may need to be created/obtained before focus."""
    task_l = (task or "").lower()
    try:
        from core.alfworld_policy import task_looks_alfworld
        if task_looks_alfworld(task):
            return False
    except Exception:
        pass
    # Measure melting/boiling *point* then pick a box — not a melt/boil prep task.
    if _task_is_threshold_measurement_task(task):
        return False
    if re.search(r"state\s+of\s+matter|change.{0,40}state.{0,40}matter", task_l):
        return True
    # Strip measurement nouns so "melting point" does not look like verb "melt".
    stripped = re.sub(
        r"\b(melting|boiling|freezing)\s+point\b", " ", task_l,
    )
    # Phase-change / thermal verbs. Named substance tokens and an explicit
    # "focus" clause are optional: "freeze the substance" still needs kitchen
    # freezer/stove, not the art studio.
    if re.search(
        r"\b(melts?|melted|boils?|boiled|freezes?|froze|frozen|"
        r"combust\w*|burns?|burned|burnt|heats?|heated|heating|"
        r"cools?|cooled|cooling|thaws?|thawed|evaporat\w*)\b",
        stripped,
    ):
        return True
    if not _task_target_tokens(task):
        return False
    # Create-then-focus chemistry: mix/pour the product before any focus commit.
    if re.search(r"create the substance", task_l) and re.search(
        r"\b(?:chemistry|mix|recipe|ingredient|pour|dissolve)\b", task_l,
    ):
        return True
    return False


_SOLID_SUBSTANCE_TARGETS = frozenset()  # deprecated: use prep structure + observation

_DISTRACTOR_FOOD_OBJECTS = frozenset({
    "apple", "banana", "orange", "potato", "bread", "tomato", "onion",
})


def _task_needs_liquid_fixture_prep(task: str, ctx_text: str = "") -> bool:
    """True when prep should use sink/bathtub fill (not pickup from containers)."""
    if not _task_needs_substance_preparation(task):
        return False
    targets = _task_target_tokens(task)
    if not targets:
        return False
    if targets & {"water", "liquid", "steam"}:
        return True
    task_l = (task or "").lower()
    if "water" in task_l:
        return True
    if _task_favors_heat_change(task) and not (targets & {"water", "liquid", "steam", "ice"}):
        return False
    if _task_substance_visible(task, ctx_text):
        return False
    return False


_LIQUID_FIXTURE_DESTINATIONS = frozenset({"sink", "bathtub", "faucet"})


def _action_manipulation_object(action: str) -> str:
    """Primary object phrase for pick/take/move/activate style actions."""
    al = (action or "").strip().lower()
    if al.startswith("pick up "):
        return al[8:].strip()
    if al.startswith("take "):
        return re.sub(r"\s+from\s+.*", "", al[5:].strip())
    if al.startswith("move "):
        m = re.match(r"move (.+?) to ", al)
        return m.group(1).strip() if m else ""
    if al.startswith(("activate ", "deactivate ", "open ", "close ")):
        return al.split(" ", 1)[-1].strip()
    return ""


def _primary_pick_object(obj_phrase: str) -> str:
    """Item actually picked, stripping qualifiers like 'orange in bowl'."""
    obj = (obj_phrase or "").strip().lower()
    for sep in (" in ", " on ", " from "):
        if sep in obj:
            obj = obj.split(sep, 1)[0].strip()
    return obj


def _room_observation_incomplete(ctx_text: str) -> bool:
    """True when context lacks a full room listing (e.g. right after navigation)."""
    ctx = (ctx_text or "").lower().strip()
    if not ctx:
        return True
    if re.search(r"(?:room|location) is called the ", ctx):
        return False
    if "in it, you see" in ctx:
        return False
    if re.search(r"you move to the \w+", ctx) and len(ctx) < 200:
        return True
    return False


def _obj_contains_word(obj_l: str, word: str) -> bool:
    """Whole-word match so 'pot' does not match 'potato' or 'cup' in 'cupboard'."""
    if not obj_l or not word:
        return False
    return bool(re.search(rf"\b{re.escape(word.strip().lower())}\b", obj_l.strip().lower()))


_PREP_CONTAINER_RANK: dict[str, int] = {
    "metal pot": 0,
    "pot": 1,
    "glass cup": 2,
    "ceramic cup": 3,
    "tin cup": 4,
    "cup": 5,
    "bowl": 6,
    "jar": 7,
    "pan": 8,
}


def _prep_container_rank(obj_phrase: str) -> int:
    obj_l = (obj_phrase or "").strip().lower()
    best = 99
    for name, rank in _PREP_CONTAINER_RANK.items():
        if _obj_contains_word(obj_l, name):
            best = min(best, rank)
    return best


# Substance-search storage only. Oven/stove are heat fixtures, not hideouts.
_PREP_STORAGE_CONTAINERS = ("cupboard", "drawer", "fridge", "freezer")


def _task_favors_cold_storage(task: str) -> bool:
    """Tasks where fridge/freezer is a plausible substance source (not heat/melt prep)."""
    task_l = (task or "").lower()
    if re.search(r"\bfreeze\b|\bfrozen\b|\bcool\b", task_l):
        return True
    return bool(_task_target_tokens(task) & {"ice", "frozen"})


def _task_favors_heat_change(task: str) -> bool:
    """Tasks that heat/combust substance rather than cool it."""
    task_l = (task or "").lower()
    return bool(re.search(r"\b(melt|boil|heat|combust|burn)\b", task_l))


def _storage_is_heat_appliance(container: str) -> bool:
    """True for oven/stove/furnace — process fixtures, not substance storage."""
    c = (container or "").strip().lower()
    return bool(
        re.search(r"\b(?:oven|stove|microwave|furnace|hot\s*plate)\b", c)
    )


def _storage_useful_for_substance_search(container: str, task: str) -> bool:
    """Polarity filter: heat SoM ignores cold storage; never hunt inside ovens."""
    if _storage_is_heat_appliance(container):
        return False
    if _storage_is_cold(container) and _task_favors_heat_change(task):
        return False
    if (
        _storage_is_cabinet(container)
        and _task_favors_cold_storage(task)
        and not _task_favors_heat_change(task)
    ):
        # Freeze tasks still open cupboard/drawer after cold storage.
        return True
    return True


def _task_allows_foundry_heat(task: str) -> bool:
    """True only when the instruction names foundry / blast furnace.

    Kitchen boil/melt and melting-point *measurement* use stove/oven. The
    substring ``melting`` in a task id must not license the foundry furnace.
    """
    task_l = (task or "").lower()
    if _task_is_threshold_measurement_task(task):
        return bool(re.search(r"\b(?:foundry|blast\s*furnace)\b", task_l))
    return bool(re.search(r"\b(?:foundry|blast\s*furnace)\b", task_l))


def _closed_storage_containers_in_context(ctx_text: str) -> list[str]:
    """Closed containers in the observation (fixed names + generic parsing)."""
    parsed = closed_containers_in_observation(ctx_text)
    ctx = (ctx_text or "").lower()
    found: list[str] = list(parsed)
    seen = set(found)
    for name in _PREP_STORAGE_CONTAINERS:
        if name in seen:
            continue
        if not re.search(rf"\b{re.escape(name)}\b", ctx):
            continue
        if re.search(rf"\b{re.escape(name)}\b(?:[^.\n]{{0,48}})\bclosed\b", ctx):
            found.append(name)
            seen.add(name)
    return found


def _storage_is_cold(container: str) -> bool:
    """True for fridge / freezer including 'ultra low temperature freezer'."""
    c = (container or "").strip().lower()
    return bool(re.search(r"\b(?:fridge|freezer|refrigerator|cooler)\b", c))


def _cold_open_blocked_on_heat(act: str, task: str) -> bool:
    """True when opening cold/heat storage must not run during heat SoM search."""
    al = (act or "").strip().lower()
    if not al.startswith("open ") or al.startswith("open door"):
        return False
    if not _task_favors_heat_change(task):
        return False
    return _storage_is_cold(al) or _storage_is_heat_appliance(al)


def _storage_is_cabinet(container: str) -> bool:
    c = (container or "").strip().lower()
    return bool(re.search(r"\b(?:cupboard|drawer|cabinet)\b", c))


def _storage_container_open_score(container: str, task: str) -> int:
    """Rank which closed container to open first (task-agnostic substance prep)."""
    if _storage_is_heat_appliance(container):
        return -100
    if not _storage_useful_for_substance_search(container, task):
        return -100
    score = max(0, 80 - _prep_container_rank(container) * 8)
    cold = _task_favors_cold_storage(task)
    heat = _task_favors_heat_change(task)
    if _storage_is_cabinet(container):
        score += 45 if not cold else 15
    if _storage_is_cold(container):
        if cold:
            score += 65
        elif heat:
            # Boil/melt never keep hunting in cold storage — leave room instead.
            # Match substring so "ultra low temperature freezer" is covered.
            score -= 120
        else:
            score += 10
    return score


def _pick_closed_container_for_prep(
    env_valid: set,
    recent: set,
    task: str,
    ctx_text: str,
    failed_actions: set | None = None,
    action_history=None,
) -> str | None:
    """Open the best closed in-room container when the task substance is not visible."""
    if not _task_target_missing(task, ctx_text):
        return None
    if not (
        _task_needs_substance_preparation(task)
        or _solid_prep_needs_room_search(task, ctx_text)
    ):
        return None
    phase = _substance_prep_phase(
        task, ctx_text, action_history, failed_actions,
    )
    if phase not in ("open", "solid_search", "pick"):
        useful = [
            c
            for c in _closed_storage_containers_in_context(ctx_text)
            if _storage_useful_for_substance_search(c, task)
        ]
        if not useful:
            return None
    failed = failed_actions or set()
    ranked: list[tuple[int, str]] = []
    for container in _closed_storage_containers_in_context(ctx_text):
        if not _storage_useful_for_substance_search(container, task):
            continue
        act = f"open {container}"
        if act in env_valid and act not in recent and act not in failed:
            sc = _storage_container_open_score(container, task)
            if sc < 0:
                continue
            ranked.append((sc, act))
    if not ranked:
        for act in env_valid:
            al = (act or "").strip().lower()
            if not al.startswith("open ") or al.startswith("open door"):
                continue
            if " in " not in al:
                continue
            if act in recent or act in failed:
                continue
            # Heat SoM: never fall through to open freezer/fridge/oven.
            if _task_favors_heat_change(task) and (
                _storage_is_cold(al) or _storage_is_heat_appliance(al)
            ):
                continue
            if _storage_is_heat_appliance(al):
                continue
            if not _storage_useful_for_substance_search(al, task):
                continue
            return act
        return None
    ranked.sort(key=lambda x: (-x[0], x[1]))
    best = ranked[0][1]
    # Heat tasks must never open cold storage even if it ranked top somehow.
    if _task_favors_heat_change(task) and _storage_is_cold(best):
        for _sc, act in ranked[1:]:
            if not _storage_is_cold(act):
                return act
        return None
    return best


def _unopened_storage_in_current_room(ctx_text: str, task: str = "") -> bool:
    """True when useful closed cupboard/drawer/fridge/freezer remain in-room."""
    closed = _closed_storage_containers_in_context(ctx_text)
    if not task:
        return bool(closed)
    return any(_storage_useful_for_substance_search(c, task) for c in closed)


def _defer_navigation_for_unopened_storage(
    task: str,
    ctx_text: str,
    env_valid: set,
    recent: set | None = None,
    failed_actions: set | None = None,
    action_history=None,
) -> bool:
    """Do not leave the room while a polarity-useful closed container can open."""
    if not _task_target_missing(task, ctx_text):
        return False
    if not _unopened_storage_in_current_room(ctx_text, task):
        return False
    recent = recent or set()
    return _pick_closed_container_for_prep(
        env_valid, recent, task, ctx_text, failed_actions, action_history,
    ) is not None


def exploration_substitute_when_target_missing(
    task: str,
    env_valid: set,
    recent: set,
    ctx_text: str,
    action_history=None,
    failed_actions: set | None = None,
    logger=None,
    current_score: int | None = None,
    *,
    task_tokens: set | None = None,
    planned_tokens: set | None = None,
) -> str | None:
    """
    Unified fallback when the planner picks exploration but the task target is invisible.
    Priority: location hints → open closed containers → scored prep → task-aware nav → door nav.
    """
    if not _task_target_missing(task, ctx_text):
        return None
    task_tokens = task_tokens or extract_task_content_tokens(task)
    planned_tokens = planned_tokens or set()

    if _extract_task_location_hint_rooms(task):
        hint_nav = _nav_to_task_location_hints(task, env_valid, recent, ctx_text)
        if hint_nav and hint_nav not in recent:
            if logger:
                logger.info(f"Substitute: location hint nav: {hint_nav!r}")
            return hint_nav

    # Prefer structure-based rooms (workshop for circuits/substances) before opening junk.
    pref_nav = _nav_toward_preferred_rooms(
        task, env_valid, recent, ctx_text, action_history,
    )
    if pref_nav and pref_nav not in recent:
        if logger:
            logger.info(f"Substitute: preferred-room nav: {pref_nav!r}")
        return pref_nav

    if _task_needs_substance_preparation(task):
        container = _pick_closed_container_for_prep(
            env_valid, recent, task, ctx_text, failed_actions, action_history,
        )
        if (
            container
            and container not in recent
            and not _cold_open_blocked_on_heat(container, task)
        ):
            if logger:
                logger.info(f"Substitute: missing target, open container: {container!r}")
            return container

    prep_phase = _substance_prep_phase(
        task, ctx_text, action_history, failed_actions,
    )
    if prep_phase == "solid_search":
        if not _defer_navigation_for_unopened_storage(
            task, ctx_text, env_valid, recent, failed_actions, action_history,
        ):
            task_nav = _nav_toward_unexplored_rooms(
                task, env_valid, recent, ctx_text, action_history,
            )
            if not task_nav:
                task_nav = _nav_toward_missing_target(
                    task, env_valid, recent, ctx_text, action_history,
                )
            if task_nav and task_nav not in recent:
                if logger:
                    logger.info(f"Substitute: solid search nav: {task_nav!r}")
                return task_nav

    if (
        _focus_subgoal_pending(task, action_history, ctx_text, current_score)
        and _task_needs_substance_preparation(task)
    ):
        for act in _prep_actions_for_missing_substance(
            task, env_valid, recent, ctx_text, action_history, failed_actions,
        ):
            if act not in recent:
                if logger:
                    logger.info(f"Substitute: missing target, prep: {act!r}")
                return act

    if not _defer_navigation_for_unopened_storage(
        task, ctx_text, env_valid, recent, failed_actions, action_history,
    ):
        task_nav = _nav_toward_missing_target(
            task, env_valid, recent, ctx_text, action_history,
        )
        if task_nav and task_nav not in recent:
            if logger:
                logger.info(f"Substitute: missing target, navigate: {task_nav!r}")
            return task_nav

    for nav in _navigation_actions_from_observation(
        env_valid, ctx_text, recent, task_tokens, planned_tokens, action_history, task=task,
    ):
        if nav in recent:
            continue
        if _is_blocked_explore_nav_substitute(
            nav, task, ctx_text, action_history, env_valid,
        ):
            continue
        if logger:
            logger.info(f"Substitute: missing target, explore nav: {nav!r}")
        return nav
    return None


def _best_visible_prep_container(ctx_text: str) -> str | None:
    ctx = (ctx_text or "").lower()
    found: list[tuple[int, str]] = []
    for name, rank in _PREP_CONTAINER_RANK.items():
        if re.search(rf"\b{re.escape(name)}\b", ctx):
            found.append((rank, name))
    if not found:
        return None
    found.sort(key=lambda x: (x[0], x[1]))
    return found[0][1]


def _agent_holds_inferior_prep_container(
    ctx_text: str, action_history=None,
) -> bool:
    """True when agent holds a prep vessel but a better one is visible in-room."""
    if _prep_container_in_liquid_fixture(ctx_text, action_history):
        return False
    held = _held_prep_container_phrase(ctx_text, action_history)
    if not held:
        return False
    best_visible = _best_visible_prep_container(ctx_text)
    if not best_visible:
        return False
    return _prep_container_rank(held) > _prep_container_rank(best_visible)


def _remap_fixture_focus_to_substance_in_container(
    planned: str, ctx_text: str, task: str,
) -> str:
    """Map bare container focus (focus on metal pot) to substance-in-container phrasing."""
    p = (planned or "").strip().lower()
    if not p.startswith("focus on "):
        return planned
    obj = p[len("focus on ") :].strip()
    if "substance in" in obj or "substance called" in obj:
        return planned
    if not _task_needs_substance_preparation(task):
        return planned
    if _is_prep_container_object(obj) and _entity_visible_in_context(obj, ctx_text):
        return f"focus on substance in {obj}"
    for phrase in _observation_substance_focus_plans(ctx_text, task):
        if obj in phrase or _obj_contains_word(phrase, obj.split()[-1]):
            return phrase
    return planned


def _is_prep_container_object(obj_phrase: str) -> bool:
    """True for portable vessels used to obtain/prepare a task substance (not food/furniture)."""
    obj_l = _primary_pick_object(obj_phrase)
    if not obj_l:
        return False
    obj_tokens = _action_content_tokens(obj_l)
    if obj_tokens & _NON_PORTABLE_PICK_WORDS:
        return False
    if obj_tokens & (_LOW_VALUE_EXPLORE_OBJECTS | _FORBIDDEN_MANIPULATION_TARGETS):
        return False
    return any(_obj_contains_word(obj_l, h) for h in _PRE_FOCUS_PREP_CONTAINER_HINTS)


def _allowed_prep_vessel_focus(obj_phrase: str, task: str, ctx_text: str = "") -> bool:
    """
    Lab-vessel focus during substance-prep tasks is on-task.

    SOM/chemistry agents often focus ``metal pot`` / ``tin cup`` before the
    substance itself is addressable. Rejecting those as off-task zeroed boil
    scores on both CL-on and CL-off. Still reject vessels whose residual tokens
    name a conflicting substance (e.g. orange juice when boiling lead).
    """
    if not _task_needs_substance_preparation(task):
        return False
    if not _is_prep_container_object(obj_phrase):
        return False
    obj_l = (obj_phrase or "").strip().lower()
    targets = _task_target_tokens(task)
    obj_tokens = _action_content_tokens(obj_l) - _FOCUS_VAGUE_OBJECT_WORDS
    material = {
        "metal", "glass", "ceramic", "wood", "plastic",
        "large", "small", "tiny", "giant", "ultra",
        # Vessel material adjectives (tin cup / steel pot) are not substance
        # residuals — boiling lead still needs the tin cup as a prep vessel.
        "tin", "steel", "copper", "iron", "aluminum", "aluminium", "bronze",
        "brass", "silver", "gold",
    }
    vessel = set(_TASK_SUBSTANCE_CONTAINER_WORDS) | set(_PRE_FOCUS_PREP_CONTAINER_HINTS)
    residual = obj_tokens - vessel - material
    if residual and targets and not (residual & targets):
        return False
    # Homonym vessels: boil tin / melt lead must not focus ``tin cup`` / ``lead pot``.
    if targets & material and any(
        w in obj_l for w in ("cup", "pot", "beaker", "jug", "bowl", "jar")
    ):
        if (obj_tokens & targets & material) and not re.search(
            r"\b(?:substance|block|ingot|pellet|sheet|chunk|bar|metal)\b", obj_l,
        ):
            return False
    # Pure prep vessels (metal pot / glass cup with no conflicting substance
    # tokens) stay on-task even when visibility parsing fails. Requiring
    # visibility previously marked ``focus on metal pot`` as Actor off-task
    # on boil-lead and replaced it with hallway navigation (score stayed 0).
    if not residual:
        return True
    if _entity_visible_in_context(obj_l, ctx_text):
        return True
    # Visibility parsers are noisy; token-overlap vessels (tin cup / boil tin) OK.
    if targets and (obj_tokens & targets):
        return True
    return False


def _task_is_life_grow_family(task: str) -> bool:
    """True for grow-plant/fruit and related life-stage tasks."""
    task_l = (task or "").lower()
    try:
        from core.pattern_library import _task_family_flags
        fam = _task_family_flags(task)
        if fam.get("life") or fam.get("life_stage") or fam.get("grow"):
            return True
    except Exception:
        pass
    return bool(re.search(
        r"\b(?:grow|crosspollinat|life\s*stage|seeds?\s+can\s+be\s+found)\b",
        task_l,
    ))


def _allowed_life_grow_manipulation(
    obj_phrase: str,
    task: str,
    ctx_text: str = "",
    action_history=None,
) -> bool:
    """
    Seed jars, living payloads, and prep cups during grow/life tasks.

    ScienceWorld aliases ``pick up ceramic cup`` to the seed jar and treats
    long seed-stage phrases as the focus target — not the trailing cup vessel.
    """
    if not _task_is_life_grow_family(task):
        return False
    obj_l = (obj_phrase or "").strip().lower()
    if not obj_l:
        return False
    obj_tokens = _action_content_tokens(obj_l)
    targets = _task_target_tokens(task)
    if re.search(
        r"\b(?:seed|seeds|living|organism|stage|plant|grown|fruit|sapling|seedling)\b",
        obj_l,
    ):
        if targets and (obj_tokens & targets):
            return True
        if re.search(r"\b(?:seed|seeds|living|organism|stage)\b", obj_l):
            return True
    if _is_prep_container_object(obj_l):
        if re.search(r"\bseeds?\b", (task or "").lower()):
            return True
        hist = [(a or "").strip().lower() for a in (action_history or [])[-20:]]
        for act in reversed(hist):
            if not act.startswith("focus on"):
                continue
            foc_obj = act[len("focus on ") :].strip()
            if re.search(r"\b(?:seed|plant|living|organism|stage)\b", foc_obj):
                return True
            if targets and (_action_content_tokens(foc_obj) & targets):
                return True
    return False


def _substance_confirmed_in_history(
    obj_phrase: str, task: str, action_history=None,
) -> bool:
    """True when a recent look/examine targeted the task substance entity."""
    targets = _task_target_tokens(task)
    if not targets:
        return False
    obj_core = _entity_core_phrase(obj_phrase)
    obj_tokens = _action_content_tokens(obj_phrase)
    if not (obj_core in targets or (obj_tokens & targets)):
        return False
    for act in reversed([a.strip().lower() for a in (action_history or [])[-6:]]):
        if act in ("look around", "inventory", "task", "wait"):
            continue
        if act.startswith(("look at ", "examine ")):
            examined = act.split(" ", 2)[-1].strip()
            if _entity_core_phrase(examined) == obj_core:
                return True
            if _action_content_tokens(examined) & obj_tokens & targets:
                return True
        if act.startswith("focus on "):
            continue
    return False


def _inventory_text(ctx_text: str) -> str:
    ctx = (ctx_text or "").lower()
    if "inventory" not in ctx:
        return ""
    return ctx.split("inventory", 1)[-1][:800]


def _action_releases_object(action: str) -> str:
    """Return the object phrase released by put/drop/move-to-surface, else ''."""
    al = (action or "").strip().lower()
    if al.startswith(("put down ", "drop ")):
        return _action_manipulation_object(action)
    if al.startswith("move ") and re.search(
        r"\bto (?:counter|table|stove|oven|cupboard|drawer|fridge|freezer|floor|chair|bed)\b",
        al,
    ):
        return _action_manipulation_object(action)
    return ""


def _prep_container_in_liquid_fixture(
    ctx_text: str, action_history=None,
) -> str:
    """Return prep container phrase sitting in sink/bathtub, else ''."""
    ctx = (ctx_text or "").lower()
    m = re.search(
        r"\bin the (?:sink|bathtub) is:[^\n]*"
        r"\b(metal pot|glass cup|ceramic cup|tin cup|"
        r"(?:metal |glass |ceramic )?(?:pot|cup|jar|bowl|pan))\b",
        ctx,
        re.I,
    )
    if m:
        return m.group(1).strip().lower()
    hist = [a.strip().lower() for a in (action_history or [])[-8:]]
    for act in reversed(hist):
        if re.match(r"move .+ to (?:sink|bathtub|faucet)\b", act):
            mobj = _action_manipulation_object(act)
            if mobj and _is_prep_container_object(mobj):
                return mobj
            break
        if act.startswith(("pick up ", "take ")):
            break
    return ""


def _active_prep_container_phrase(
    ctx_text: str, action_history=None,
) -> str:
    """Prep vessel currently held or placed at the active liquid fixture."""
    held = _held_prep_container_phrase(ctx_text, action_history)
    if held:
        return held
    return _prep_container_in_liquid_fixture(ctx_text, action_history)


def _agent_holds_prep_container(
    ctx_text: str, action_history=None,
) -> bool:
    """True when the agent already carries a portable prep vessel."""
    return bool(_held_prep_container_phrase(ctx_text, action_history))


def _held_prep_container_phrase(
    ctx_text: str, action_history=None,
) -> str:
    inv = _inventory_text(ctx_text)
    for hint in ("metal pot", "glass cup", "ceramic cup", "tin cup", "pot", "cup", "jar", "bowl", "pan"):
        if inv and re.search(rf"\b{re.escape(hint)}\b", inv):
            return hint
    if _prep_container_in_liquid_fixture(ctx_text, action_history):
        return ""
    released_cores: set[str] = set()
    released_tokens: set[str] = set()
    for act in reversed([a.strip().lower() for a in (action_history or [])[-15:]]):
        rel = _action_releases_object(act)
        if rel:
            released_cores.add(_entity_core_phrase(rel))
            released_tokens |= _action_content_tokens(rel)
            continue
        if act.startswith(("pick up ", "take ")):
            obj = _action_manipulation_object(act)
            if not _is_prep_container_object(obj):
                continue
            core = _entity_core_phrase(obj)
            if core in released_cores or (released_tokens & _action_content_tokens(obj)):
                continue
            return obj
    return ""


def _visible_prep_container_in_room(
    ctx_text: str, action_history=None,
) -> bool:
    ctx = (ctx_text or "").lower()
    has_visible = bool(re.search(
        r"\b(metal pot|glass cup|ceramic cup|tin cup|(?:metal |glass |ceramic )?(?:pot|cup|jar|bowl))\b",
        ctx,
    ))
    if not has_visible:
        return False
    if _agent_holds_inferior_prep_container(ctx_text, action_history):
        return True
    return not _agent_holds_prep_container(ctx_text, action_history)


def _liquid_source_activated(
    action_history=None, ctx_text: str = "", failed_actions: set | None = None,
) -> bool:
    ctx = (ctx_text or "").lower()
    if re.search(r"\b(?:sink|bathtub|faucet), which is turned on\b", ctx):
        return True
    failed = failed_actions or set()
    hist = " ".join((a or "").strip().lower() for a in (action_history or [])[-12:])
    for fixture in ("sink", "bathtub", "faucet"):
        if re.search(rf"\bactivate {re.escape(fixture)}\b", hist):
            act = f"activate {fixture}"
            if act in failed:
                continue
            if _liquid_fixture_failed(fixture, failed, action_history, ctx_text):
                continue
            return True
    return False


def _liquid_fixture_failed(
    fixture: str,
    failed_actions: set | None = None,
    action_history=None,
    ctx_text: str = "",
) -> bool:
    """True when a liquid fixture recently failed activation or is reported broken."""
    fixture = (fixture or "").strip().lower()
    failed = failed_actions or set()
    for act in failed:
        al = (act or "").strip().lower()
        if al.startswith("activate ") and fixture in al:
            return True
    ctx = (ctx_text or "").lower()
    if fixture in ctx and re.search(rf"\b{re.escape(fixture)}\b.*appears broken", ctx):
        return True
    return False


def _available_liquid_fixtures(
    ctx_text: str,
    action_history=None,
    failed_actions: set | None = None,
) -> list[str]:
    """Ordered liquid fixtures still usable for substance prep."""
    ctx = (ctx_text or "").lower()
    out: list[str] = []
    for fixture in ("sink", "bathtub", "faucet"):
        if fixture not in ctx and fixture not in " ".join(
            (a or "").lower() for a in (action_history or [])[-8:]
        ):
            continue
        if _liquid_fixture_failed(fixture, failed_actions, action_history, ctx_text):
            continue
        out.append(fixture)
    return out


def _container_holds_task_substance(
    container_phrase: str,
    task: str,
    ctx_text: str = "",
    action_history=None,
) -> bool:
    """True when observation/inventory shows ``container`` holds a task substance.

    Accepts optional ``action_history`` for held-container fallbacks. A prior
    duplicate 3-arg definition overwrote this and crashed 4-arg callers.
    """
    targets = _task_target_tokens(task)
    if not targets:
        return False
    container = (container_phrase or "").strip().lower()
    if not container:
        return False
    ctx = (ctx_text or "").lower()
    inv = _inventory_text(ctx_text)
    for blob in (inv, ctx):
        if not blob:
            continue
        for t in targets:
            if re.search(
                rf"{re.escape(container)} \(containing (?:a |an )?(?:substance called )?{re.escape(t)}\b",
                blob,
            ):
                return True
            if re.search(rf"{re.escape(container)} \(containing (?:a |an )?{re.escape(t)}\b", blob):
                return True
            # Loose co-occurrence in the same sentence fragment.
            if re.search(
                rf"\b{re.escape(container)}\b(?:(?!\.).){{0,96}}containing (?:a |an )?(?:substance called )?{re.escape(t)}\b",
                blob,
            ):
                return True
    held = _active_prep_container_phrase(ctx_text, action_history)
    if held and (held in container or container in held):
        if _task_substance_visible(task, ctx_text):
            inv_only = _inventory_text(ctx_text)
            if inv_only and any(re.search(rf"\b{re.escape(t)}\b", inv_only) for t in targets):
                return True
    return False


def _prep_container_ready_for_focus(
    task: str, ctx_text: str, action_history=None,
) -> bool:
    """True when prep produced a focusable task substance (not merely an empty vessel)."""
    if _task_substance_visible(task, ctx_text):
        return True
    held = _active_prep_container_phrase(ctx_text, action_history)
    if held and _container_holds_task_substance(held, task, ctx_text, action_history):
        return True
    ctx = (ctx_text or "").lower()
    for m in re.finditer(
        r"\b(metal pot|glass cup|ceramic cup|tin cup|pot|cup|jar|bowl|pan)\b"
        r"(?: \(containing (?:a |an )?(?:substance called )?([a-z][a-z0-9 ]+)\))?",
        ctx,
    ):
        subst = (m.group(2) or "").strip()
        if subst and (_action_content_tokens(subst) & _task_target_tokens(task)):
            return True
    return False


def _best_move_to_liquid_source(
    env_valid: set,
    ctx_text: str,
    action_history=None,
    failed_actions: set | None = None,
) -> str | None:
    """Best grounded move of held prep container to an active/working liquid fixture."""
    held = _held_prep_container_phrase(ctx_text, action_history)
    if not held:
        return None
    held_core = held.split()[-1]
    fixtures = _available_liquid_fixtures(ctx_text, action_history, failed_actions)
    if not fixtures:
        fixtures = ["sink", "bathtub"]
    ctx = (ctx_text or "").lower()
    ranked: list[tuple[int, str]] = []
    for va in env_valid:
        al = (va or "").strip().lower()
        if not al.startswith("move ") or not re.search(r"\bto (sink|bathtub|faucet)\b", al):
            continue
        mobj = _action_manipulation_object(va)
        if not mobj or not (_obj_contains_word(mobj, held) or _obj_contains_word(mobj, held_core)):
            continue
        dest = re.search(r"\bto (sink|bathtub|faucet)\b", al)
        if not dest:
            continue
        fixture = dest.group(1)
        score = 80
        if fixture in fixtures:
            score += 20
        if _liquid_source_activated(action_history, ctx_text) and fixture in ctx:
            if re.search(rf"\b{re.escape(fixture)}, which is turned on\b", ctx):
                score += 25
        if _liquid_fixture_failed(fixture, failed_actions, action_history, ctx_text):
            score = 0
        if score > 0:
            ranked.append((score, va))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked[0][1] if ranked else None


def _focus_target_off_task(
    obj_phrase: str, task: str, ctx_text: str = "", action_history=None,
) -> bool:
    """True when a focus target has no overlap with the task substance (e.g. orange juice for boil water)."""
    obj_l = (obj_phrase or "").strip().lower()
    if not obj_l:
        return False
    comparison_sem = _extract_comparison_semantics(task)
    if comparison_sem:
        if _comparison_focus_aligns(obj_phrase, task, ctx_text):
            return False
        if _entity_visible_in_context(obj_phrase, ctx_text):
            attr, direction = comparison_sem
            if _comparison_entity_priority(obj_phrase, attr, direction, task) >= 45:
                return False
    targets = _task_target_tokens(task)
    if not targets:
        return False
    # ``substance called X`` is on-task only when X intersects targets.
    if "substance called" in obj_l:
        for t in targets:
            if re.search(rf"substance called {re.escape(t)}\b", obj_l):
                return False
        return True
    # ``substance in C`` is on-task only when C is shown to hold the task substance.
    if "substance in" in obj_l:
        m = re.search(r"substance in ([a-z][a-z0-9 ]+)", obj_l)
        if not m:
            return True
        container = m.group(1).strip()
        if _container_holds_task_substance(container, task, ctx_text):
            return False
        if _substance_confirmed_in_history(obj_phrase, task, action_history):
            return False
        return True
    if _substance_confirmed_in_history(obj_phrase, task, action_history):
        return False
    obj_tokens = _action_content_tokens(obj_l) - _FOCUS_VAGUE_OBJECT_WORDS
    if obj_tokens & targets:
        return False
    if _task_substance_visible(task, ctx_text) and _focus_matches_task_substance(
        obj_phrase, task, ctx_text,
    ):
        return False
    return bool(obj_tokens)

def _is_post_focus_unproductive(
    action: str,
    task: str,
    action_history=None,
    ctx_text: str = "",
    current_score: int | None = None,
) -> bool:
    """Block repeat focus / detours / absurd manipulation after focus milestone."""
    if not _task_focus_already_satisfied(task, action_history, ctx_text, current_score):
        return False
    if not (
        _task_has_post_focus_phase(task)
        or _task_requires_connect(task)
        or _task_needs_growth_after_focus(task)
    ):
        return False
    if _is_counterproductive_plateau_action(
        action, task, action_history, ctx_text, current_score,
    ):
        return True
    al = (action or "").strip().lower()
    if _is_absurd_post_focus_manipulation(action, task):
        return True
    if _is_absurd_pour_action(action, task, ctx_text):
        return True
    if _is_distractor_pour_action(action, task, action_history, ctx_text):
        return True
    # Contaminating task vessels with paint/ink (or pouring them into task cups).
    # Skip for growth tasks: soil/water destinations are productive, not distractors.
    if (
        (_task_needs_substance_preparation(task) or _task_has_post_focus_phase(task))
        and not _task_needs_growth_after_focus(task)
    ):
        if any(w in al for w in _DISTRACTOR_VESSEL_CONTENT_WORDS):
            task_t = _task_target_tokens(task)
            if not (task_t & _DISTRACTOR_VESSEL_CONTENT_WORDS):
                return True
        if action_verb_family(action) == "move":
            m = re.match(r"move (.+?) to (.+)", al)
            if m:
                dest = m.group(2).strip()
                dest_room = _normalize_room_name(dest)
                if dest_room in _LOW_VALUE_SEARCH_ROOMS and _heat_setup_pending(
                    task, action_history, ctx_text, current_score, None,
                ):
                    return True
                if any(w in dest for w in _DISTRACTOR_VESSEL_CONTENT_WORDS):
                    return True
    if _heat_setup_pending(
        task, action_history, ctx_text, current_score, None,
    ):
        al = (action or "").strip().lower()
        if action_verb_family(action) == "activate" and not any(
            w in al for w in _PRE_FOCUS_HEAT_FIXTURE_WORDS
        ):
            return True
        if al.startswith(("go to ", "open door to ")):
            foc_room = _focused_substance_container_room(
                task, action_history, ctx_text,
            )
            if foc_room:
                dest = _nav_dest_room(al)
                if dest == foc_room:
                    return True
    scr = int(current_score or 0)
    min_xfer = _transfer_min_score(task, action_history, None)
    if scr >= min_xfer and _focused_substance_is_solid(task):
        if _substance_visible_on_heat_fixture(ctx_text, task, action_history):
            if action_verb_family(action) in ("pick_up", "take"):
                obj = _action_manipulation_object(action)
                if obj and _object_matches_focused_substance(
                    obj, task, action_history, ctx_text,
                ):
                    return True
            m = re.match(r"move (.+?) to (.+)", al)
            if m and _object_matches_focused_substance(
                m.group(1), task, action_history, ctx_text,
            ) and re.search(
                r"\bto (?:bowl|cup|pot|pan|beaker|flask|jar)\b", al,
            ):
                return True
    if al.startswith("focus on"):
        return True
    if _task_requires_connect(task) and _task_focus_already_satisfied(
        task, action_history, ctx_text, current_score,
    ):
        family = action_verb_family(action)
        # After focus, circuit progress is connect/pick — not activate/mix/eat.
        if family in ("activate", "deactivate", "mix", "eat", "drink"):
            return True
        if family == "pour":
            obj = _action_manipulation_object(action)
            if obj and _action_content_tokens(obj) & _CIRCUIT_HINT_TOKENS:
                return True
        if family in ("pick_up", "take"):
            obj = _action_manipulation_object(action)
            if obj and _task_target_in_inventory(task, ctx_text):
                if _action_content_tokens(obj) & (
                    _CIRCUIT_HINT_TOKENS | _task_target_tokens(task)
                ):
                    return True
        if family == "move":
            obj = _action_manipulation_object(action)
            if obj and (
                _is_fixed_installation_target(obj)
                or _action_content_tokens(obj) & _CIRCUIT_HINT_TOKENS
            ):
                return True
        return False
    if _task_requires_connect(task) and action_verb_family(action) == "pour":
        obj = _action_manipulation_object(action)
        if obj and _action_content_tokens(obj) & _CIRCUIT_HINT_TOKENS:
            return True
    if _task_requires_connect(task) and action_verb_family(action) in ("pick_up", "take"):
        obj = _action_manipulation_object(action)
        if obj and _task_target_in_inventory(task, ctx_text):
            if _action_content_tokens(obj) & (
                _CIRCUIT_HINT_TOKENS | _task_target_tokens(task)
            ):
                return True
    if _task_requires_connect(task) and action_verb_family(action) == "move":
        obj = _action_manipulation_object(action)
        if obj and (
            _is_fixed_installation_target(obj)
            or _action_content_tokens(obj) & _CIRCUIT_HINT_TOKENS
        ):
            return True
    if al.startswith(("go to ", "open door to ")):
        dest = _nav_dest_room(al)
        current = _extract_current_room_name(ctx_text)
        dest_norm = dest
        if current and dest_norm == current:
            return True
        foc_room = _focused_substance_container_room(
            task, action_history, ctx_text,
        )
        transfer_p = _substance_transfer_pending(
            task, action_history, ctx_text, current_score,
        )
        som_heat = _state_of_matter_post_focus_phase(
            task, current_score, action_history, ctx_text,
        ) == "heat"
        if foc_room and dest_norm == foc_room and current != foc_room:
            if som_heat or transfer_p:
                return False
        if transfer_p and foc_room and current == foc_room and dest_norm != foc_room:
            return True
        ctx = (ctx_text or "").lower()
        if any_heat_fixture_in_context(ctx_text) and _current_room_in_task_area(task, ctx_text):
            if transfer_p and foc_room and dest_norm == foc_room:
                return False
            if not (foc_room and dest_norm == foc_room):
                return True
    return False


def _container_at_liquid_source(ctx_text: str, action_history=None) -> bool:
    if _prep_container_in_liquid_fixture(ctx_text, action_history):
        return True
    ctx = (ctx_text or "").lower()
    if re.search(r"\bin the (?:sink|bathtub) is:.*\b(?:pot|cup|jar|bowl)\b", ctx, re.S):
        return True
    return False


def _liquid_prep_stalled(
    task: str,
    ctx_text: str,
    action_history=None,
    failed_actions: set | None = None,
) -> bool:
    """True when kitchen liquid prep failed or wait did not produce the task substance."""
    if not _task_needs_liquid_fixture_prep(task, ctx_text):
        return False
    fixtures = _available_liquid_fixtures(ctx_text, action_history, failed_actions)
    kitchen_fixtures = [f for f in fixtures if f in ("sink", "faucet")]
    if not kitchen_fixtures and _liquid_fixture_failed(
        "sink", failed_actions, action_history, ctx_text,
    ):
        return True
    if not _container_at_liquid_source(ctx_text, action_history):
        return False
    if not _liquid_source_activated(action_history, ctx_text, failed_actions):
        return False
    if _prep_container_ready_for_focus(task, ctx_text, action_history):
        return False
    hist = [a.strip().lower() for a in (action_history or [])[-12:]]
    waits = sum(1 for a in hist if a in ("wait",) or a.startswith("wait "))
    return waits >= 2


def _agent_holds_primary_prep_container(
    ctx_text: str,
    action_history=None,
    task: str = "",
    preferred: set | None = None,
) -> bool:
    """True when agent holds the best available prep vessel (not a stopgap bowl/jar)."""
    if _agent_holds_inferior_prep_container(ctx_text, action_history):
        return False
    held = _held_prep_container_phrase(ctx_text, action_history)
    if not held:
        return False
    preferred = preferred or (_prep_preferred_containers(task, ctx_text, action_history) if task else set())
    if preferred and any(p in held or held in p for p in preferred):
        return True
    ctx = (ctx_text or "").lower()
    if re.search(r"\bmetal pot\b", ctx) and "pot" not in held:
        return False
    if re.search(r"\b(glass cup|ceramic cup|tin cup)\b", ctx) and not any(
        w in held for w in ("cup", "pot")
    ):
        return False
    return _is_prep_container_object(held)


def _substance_prep_phase(
    task: str, ctx_text: str, action_history=None,
    failed_actions: set | None = None,
) -> str:
    """Next prep step: open → pick → fill → move_liquid → wait_fill → focus_ready."""
    preferred = _prep_preferred_containers(task, ctx_text, action_history)
    if _prep_container_ready_for_focus(task, ctx_text, action_history):
        return "focus_ready"
    targets = _task_target_tokens(task)
    needs_liquid = _task_needs_liquid_fixture_prep(task, ctx_text)
    if not needs_liquid and _task_needs_substance_preparation(task) and _task_has_post_focus_phase(task):
        if not _task_target_missing(task, ctx_text):
            return "focus_ready"
        if _solid_prep_needs_room_search(task, ctx_text):
            return "solid_search"
        useful_closed = [
            c
            for c in _closed_storage_containers_in_context(ctx_text)
            if _storage_useful_for_substance_search(c, task)
        ]
        if useful_closed:
            return "open"
        if (
            _task_favors_cold_storage(task)
            and re.search(r"\b(?:fridge|freezer)\b.*\bclosed\b", (ctx_text or "").lower())
        ):
            return "open"
        if re.search(r"\b(?:fridge|freezer)\b.*\bopen\b", (ctx_text or "").lower()):
            return "pick"
    if _container_at_liquid_source(ctx_text, action_history) and needs_liquid:
        if _liquid_source_activated(action_history, ctx_text, failed_actions):
            return "wait_fill"
        fixtures = _available_liquid_fixtures(ctx_text, action_history, failed_actions)
        if fixtures:
            return "fill"
        if _liquid_prep_stalled(task, ctx_text, action_history, failed_actions):
            return "alt_liquid_nav"
        return "fill"
    if _agent_holds_inferior_prep_container(ctx_text, action_history):
        return "pick"
    ctx = (ctx_text or "").lower()
    best_vis = _best_visible_prep_container(ctx_text)
    if re.search(r"\b(?:cupboard|drawer)\b.*\bclosed\b", ctx):
        if not best_vis or _prep_container_rank(best_vis) > _prep_container_rank("pot"):
            return "open"
    if needs_liquid:
        fixtures = _available_liquid_fixtures(ctx_text, action_history, failed_actions)
        if _liquid_source_activated(action_history, ctx_text, failed_actions):
            if _agent_holds_primary_prep_container(ctx_text, action_history, task, preferred):
                if _container_at_liquid_source(ctx_text, action_history):
                    return "wait_fill"
                return "move_liquid"
        if _agent_holds_primary_prep_container(ctx_text, action_history, task, preferred):
            if fixtures:
                return "fill"
            if _liquid_prep_stalled(task, ctx_text, action_history, failed_actions):
                return "alt_liquid_nav"
            return "alt_liquid_nav"
    if _visible_prep_container_in_room(ctx_text, action_history):
        return "pick"
    if re.search(r"\b(?:cupboard|drawer|fridge|freezer)\b.*\bclosed\b", ctx):
        return "open"
    if re.search(r"\bdoor is closed\b", ctx):
        return "open"
    return "pick"


def _prep_preferred_containers(
    task: str,
    ctx_text: str,
    action_history=None,
    planned: str = "",
) -> set[str]:
    preferred: set[str] = set()
    p = (planned or "").strip().lower()
    if p.startswith("focus on "):
        obj = p[len("focus on ") :].strip()
        m = re.search(r"substance in ([a-z][a-z0-9 ]+)", obj)
        if m:
            preferred.add(m.group(1).strip())
        elif _is_prep_container_object(obj):
            preferred.add(obj)
    for phrase in _observation_substance_focus_plans(ctx_text, task):
        m = re.search(r"substance in ([a-z][a-z0-9 ]+)", phrase)
        if m:
            preferred.add(m.group(1).strip())
    for act in reversed([a.strip().lower() for a in (action_history or [])[-12:]]):
        if act.startswith("focus on "):
            obj = act[len("focus on ") :].strip()
            m = re.search(r"substance in ([a-z][a-z0-9 ]+)", obj)
            if m:
                preferred.add(m.group(1).strip())
            elif _is_prep_container_object(obj):
                preferred.add(obj)
            break
    return preferred


def _container_observed_with_foreign_substance(
    container_phrase: str, task: str, ctx_text: str,
) -> bool:
    """True when observation shows the container holds a non-task substance."""
    targets = _task_target_tokens(task)
    if not targets:
        return False
    container = (container_phrase or "").strip().lower()
    ctx = (ctx_text or "").lower()
    # Explicit distractor contents (paint cups, etc.) regardless of "substance called".
    if re.search(
        rf"\b{re.escape(container)}\b.*?\bcontaining\b.*?\b({'|'.join(_DISTRACTOR_VESSEL_CONTENT_WORDS)})\b",
        ctx,
    ) or re.search(
        rf"\bcontaining\b.*?\b({'|'.join(_DISTRACTOR_VESSEL_CONTENT_WORDS)})\b.*?\b{re.escape(container)}\b",
        container,
    ):
        if not (targets & _DISTRACTOR_VESSEL_CONTENT_WORDS):
            return True
    if any(w in container for w in _DISTRACTOR_VESSEL_CONTENT_WORDS):
        if not (targets & _DISTRACTOR_VESSEL_CONTENT_WORDS):
            return True
    for m in re.finditer(
        rf"{re.escape(container)} \(containing (?:a |an )?substance called ([a-z][a-z0-9 ]+)\)",
        ctx,
    ):
        if not (_action_content_tokens(m.group(1)) & targets):
            return True
    core = container.split()[-1]
    for m in re.finditer(
        rf"\b{re.escape(core)} \(containing (?:a |an )?substance called ([a-z][a-z0-9 ]+)\)",
        ctx,
    ):
        if not (_action_content_tokens(m.group(1)) & targets):
            return True
    return False


def _is_grounded_pick_object(obj_phrase: str, ctx_text: str) -> bool:
    """Pick targets must appear in observation/inventory; reject hallucinated qualifiers."""
    obj = (obj_phrase or "").strip().lower()
    if not obj:
        return False
    if " containing " in obj and not _ctx_contains_phrase(ctx_text, obj):
        base = obj.split(" containing ", 1)[0].strip()
        if not _entity_visible_in_context(base, ctx_text):
            return False
        return _ctx_contains_phrase(ctx_text, obj)
    return _entity_visible_in_context(obj, ctx_text)


def _score_substance_prep_action(
    va: str,
    phase: str,
    preferred: set[str],
    task: str,
    ctx_text: str,
    action_history=None,
) -> int:
    fam = action_verb_family(va)
    al = (va or "").strip().lower()
    mobj = _action_manipulation_object(va)
    score = 0
    phase_scores = {
        "open": {"open": 95, "look_around": 70},
        "pick": {"pick_up": 94, "take": 94, "open": 55, "look_around": 65},
        "move_liquid": {"move": 96, "look_around": 50},
        "fill": {"activate": 97, "look_around": 45},
        "wait_fill": {"wait": 98, "look_around": 35},
        "alt_liquid_nav": {"look_around": 40},
        "solid_search": {"look_around": 55},
        "focus_ready": {"look_around": 40},
    }
    score = phase_scores.get(phase, {}).get(fam, 0)
    if phase == "solid_search":
        # Never score polarity-wrong opens during solid room search.
        if fam == "open":
            if not _storage_useful_for_substance_search(al, task):
                return 0
            if any(w in al for w in ("fridge", "freezer", "cupboard", "drawer")):
                score += 40 if _task_favors_cold_storage(task) else 25
        if al.startswith("go to ") or al.startswith("open door to "):
            dest = al.split(" ", 2)[-1].strip()
            visits = room_visit_counts(action_history).get(dest, 0)
            score += max(10, 55 - visits * 15)
        if fam == "look_around":
            # Full room listing already known — prefer nav over look thrash.
            if not _room_observation_incomplete(ctx_text):
                return 0
            score += 20
    if _room_observation_incomplete(ctx_text):
        if fam == "look_around":
            score += 50
        elif fam in ("pick_up", "take") and phase in ("pick", "open"):
            score -= 35
    elif fam == "look_around" and phase in ("open", "pick", "solid_search"):
        # Complete observation: look around is never the best prep step.
        return 0
    if fam in ("pick_up", "take") and not _is_prep_container_object(mobj):
        return 0
    if fam == "open" and any(w in al for w in ("cupboard", "drawer")):
        score += 4
    targets = _task_target_tokens(task)
    ctx_l = (ctx_text or "").lower()
    if phase == "open" and fam == "open" and not _task_needs_liquid_fixture_prep(task, ctx_text):
        if _storage_is_heat_appliance(al) or not _storage_useful_for_substance_search(al, task):
            return 0
        if "freezer" in al:
            if _task_favors_cold_storage(task):
                score += 90
            elif _task_favors_heat_change(task):
                return 0
            else:
                score += 10
        elif "fridge" in al:
            if _task_favors_cold_storage(task):
                score += 75
            elif _task_favors_heat_change(task):
                return 0
            else:
                score += 8
        elif any(w in al for w in ("cupboard", "drawer")):
            score += 55 if _task_favors_heat_change(task) else 25
        elif _task_favors_cold_storage(task) and any(w in al for w in ("fridge", "freezer")):
            score += 35
    if fam in ("pick_up", "take") and _is_prep_container_object(mobj):
        if preferred and any(p in mobj or mobj in p for p in preferred):
            score += 18
        best_vis = _best_visible_prep_container(ctx_text)
        if best_vis and _obj_contains_word(mobj, best_vis):
            score += 14
        elif best_vis and not _obj_contains_word(mobj, best_vis):
            score -= 28
        if "pot" in mobj:
            score += 8
        if _container_observed_with_foreign_substance(mobj, task, ctx_text):
            score -= 40
    if fam == "move" and re.search(r"\bto (?:sink|bathtub|faucet)\b", al) and _is_prep_container_object(mobj):
        score += 12
        held = _held_prep_container_phrase(ctx_text, action_history)
        if held and (held in mobj or mobj in held):
            score += 18
        if phase == "move_liquid":
            score += 20
    if fam == "activate" and any(w in al for w in _PRE_FOCUS_LIQUID_SOURCE_WORDS):
        if not _task_needs_liquid_fixture_prep(task, ctx_text):
            return 0
        fixture = next((w for w in _PRE_FOCUS_LIQUID_SOURCE_WORDS if w in al), "")
        if fixture and _liquid_fixture_failed(fixture, None, action_history, ctx_text):
            return 0
        score += 8
    if phase == "wait_fill" and fam in ("pick_up", "take", "open", "activate"):
        score -= 45
    if phase == "wait_fill" and fam == "move":
        score -= 30
    if (fam == "wait" or al.startswith("wait ")) and not _container_at_liquid_source(
        ctx_text, action_history,
    ):
        return 0
    if _container_at_liquid_source(ctx_text, action_history) and fam in ("pick_up", "take"):
        score -= 60
    if phase == "pick" and fam == "open":
        score -= 25
        ctx_l = (ctx_text or "").lower()
        if _visible_prep_container_in_room(ctx_text, action_history):
            score -= 35
        if "drawer" in al and re.search(r"\bcupboard.*open\b", ctx_l):
            if re.search(r"\bmetal pot\b", ctx_l):
                score -= 40
    if phase == "move_liquid" and fam in ("pick_up", "take", "open"):
        score -= 50
    if phase == "fill" and fam in ("pick_up", "take", "open", "move"):
        score -= 50
    if phase == "open" and fam in ("pick_up", "take", "move", "activate"):
        score -= 35
    return score


def _is_pre_focus_preparation_action(
    action: str, task: str, ctx_text: str = "", action_history=None,
) -> bool:
    """Allow container/fixture prep before focus when the substance is not visible yet."""
    if _is_measurement_setup_action(action, task, ctx_text, action_history):
        return True
    if not _task_needs_substance_preparation(task):
        return False
    if _prep_container_ready_for_focus(task, ctx_text, action_history):
        return False
    al = (action or "").strip().lower()
    family = action_verb_family(action)
    if family == "open":
        phase = _substance_prep_phase(task, ctx_text, action_history)
        if phase != "open":
            return False
        target = _action_manipulation_object(action)
        task_l = (task or "").lower()
        if any(w in target for w in ("freezer", "fridge", "oven")):
            if re.search(r"\bfreeze\b", task_l):
                return "freezer" in target
            return False
        if "drawer" in target and _visible_prep_container_in_room(ctx_text, action_history):
            ctx_l = (ctx_text or "").lower()
            if re.search(r"\bcupboard.*open\b", ctx_l) and re.search(r"\bmetal pot\b", ctx_l):
                return False
        return True
    if family in ("pick_up", "take"):
        phase = _substance_prep_phase(task, ctx_text, action_history)
        if phase in ("wait_fill", "fill", "move_liquid") and _container_at_liquid_source(
            ctx_text, action_history,
        ):
            return False
        preferred = _prep_preferred_containers(task, ctx_text, action_history)
        if _agent_holds_primary_prep_container(ctx_text, action_history, task, preferred):
            return False
        obj = _action_manipulation_object(action)
        if not _is_grounded_pick_object(obj, ctx_text):
            return False
        if _container_observed_with_foreign_substance(obj, task, ctx_text):
            return False
        primary = _primary_pick_object(obj)
        primary_tokens = _action_content_tokens(primary)
        if primary_tokens & _NON_PORTABLE_PICK_WORDS:
            return False
        if primary_tokens & (_LOW_VALUE_EXPLORE_OBJECTS | _FORBIDDEN_MANIPULATION_TARGETS):
            return False
        return _is_prep_container_object(primary)
    if family == "move":
        mobj = _action_manipulation_object(action)
        if re.search(r"\bto\s+(sink|bathtub|faucet)\b", al):
            if not _task_needs_liquid_fixture_prep(task, ctx_text):
                return False
            phase = _substance_prep_phase(task, ctx_text, action_history)
            if phase != "move_liquid":
                return False
            return _is_prep_container_object(mobj)
        m_dest = re.match(r"move .+? to (.+)", al)
        if m_dest and (
            is_heat_fixture_phrase(m_dest.group(1))
            or is_cool_fixture_phrase(m_dest.group(1))
        ):
            return _is_prep_container_object(mobj)
        return _is_prep_container_object(mobj)
    if family in ("activate", "deactivate"):
        target = _action_manipulation_object(action)
        if any(w in target for w in _PRE_FOCUS_LIQUID_SOURCE_WORDS):
            if not _task_needs_liquid_fixture_prep(task, ctx_text):
                return False
            if _liquid_fixture_failed(target, None, action_history, ctx_text):
                return False
            return True
        return False
    if al in ("wait",) or al.startswith("wait "):
        return (
            _substance_prep_phase(task, ctx_text, action_history) == "wait_fill"
            and _container_at_liquid_source(ctx_text, action_history)
        )
    if al == "look around" and _current_room_in_task_area(task, ctx_text):
        if _substance_prep_phase(task, ctx_text, action_history) == "solid_search":
            return False
        # Already have a full room listing — further look around is thrash.
        if not _room_observation_incomplete(ctx_text):
            return False
        return True
    return False


def _is_premature_heat_before_focus(
    action: str, task: str, action_history=None, ctx_text: str = "",
) -> bool:
    """Block heat or liquid fixtures before focus when prep is wrong for the task."""
    if not _requires_focus_before_manipulation(task, action_history, ctx_text):
        return False
    if not _task_needs_substance_preparation(task):
        return False
    if action_verb_family(action) != "activate":
        return False
    al = (action or "").strip().lower()
    if any(w in al for w in ("sink", "bathtub", "faucet")):
        return not _task_needs_liquid_fixture_prep(task, ctx_text)
    if is_heat_fixture_phrase(_action_manipulation_object(action) or al.replace("activate ", "", 1)):
        return True
    return False


def _is_post_focus_liquid_fixture_misactivation(
    action: str, task: str, ctx_text: str, action_history=None,
) -> bool:
    """After focus on melt/boil tasks, sink/bathtub activation rarely completes heating."""
    if not _task_needs_substance_preparation(task):
        return False
    if not _task_focus_already_satisfied(task, action_history, ctx_text):
        return False
    if action_verb_family(action) != "activate":
        return False
    target = _action_manipulation_object(action)
    if not any(w in target for w in _PRE_FOCUS_LIQUID_SOURCE_WORDS):
        return False
    task_l = (task or "").lower()
    return bool(re.search(r"\b(melt|boil)\b", task_l))


def _focused_substance_focus_phrase(
    task: str, action_history=None, ctx_text: str = "",
) -> str:
    """Object phrase from the most recent task-aligned focus action."""
    hist = [(a or "").strip().lower() for a in (action_history or [])[-24:]]
    for act in reversed(hist):
        if not act.startswith("focus on "):
            continue
        obj = act[len("focus on ") :].strip()
        if _focus_object_aligns_with_task(obj, task, ctx_text):
            return obj
    return ""


def _focused_substance_container_room(
    task: str, action_history=None, ctx_text: str = "",
) -> str:
    """Room likely holding the focused substance (observation + history)."""
    phrase = _focused_substance_focus_phrase(task, action_history, ctx_text)
    current = _extract_current_room_name(ctx_text)
    if not phrase:
        return current or ""
    pl = phrase.lower()
    fixture_candidates = list(heat_fixtures_in_environment(None, ctx_text))
    fixture_candidates.extend(
        w for w in (
            "toilet", "bathtub", "sink", "freezer", "fridge", "stove", "oven",
            "blast furnace", "metal pot", "pot", "cup",
        )
        if w in pl
    )
    for fixture in fixture_candidates:
        if fixture.lower() not in pl and f"substance in {fixture.lower()}" not in pl:
            continue
        if fixture_visible_in_context(ctx_text, fixture) and current:
            return current
        room = fixture_last_seen_room(fixture, action_history, ctx_text)
        if room:
            return room
    return current or ""


def _object_matches_focused_substance(
    obj_phrase: str,
    task: str,
    action_history=None,
    ctx_text: str = "",
) -> bool:
    """True when obj is the focused container or holds the focused substance."""
    focused = _focused_substance_focus_phrase(task, action_history, ctx_text)
    if not focused:
        return False
    obj_l = (obj_phrase or "").strip().lower()
    if not obj_l:
        return False
    if "substance in" in focused:
        m = re.search(r"substance in (.+)", focused)
        if m and m.group(1).strip() in obj_l:
            return True
    if _action_content_tokens(focused) & _action_content_tokens(obj_l):
        return True
    return _container_holds_task_substance(obj_l, task, ctx_text, action_history)


def _substance_visible_on_heat_fixture(
    ctx_text: str,
    task: str,
    action_history=None,
) -> bool:
    """True when task/focused substance appears on any visible heat fixture."""
    ctx_l = (ctx_text or "").lower()
    tokens = set(_task_target_tokens(task))
    focused = _focused_substance_focus_phrase(task, action_history, ctx_text)
    if focused:
        tokens |= _action_content_tokens(focused.replace("focus on ", "", 1))
    tokens -= {"substance", "focus", "on", "the", "a", "an", "called"}
    if not tokens:
        tokens = {"substance"}
    fixtures = heat_fixtures_in_environment(None, ctx_text) or ["stove", "oven"]
    for fixture in fixtures:
        fl = re.escape(fixture.lower())
        for tok in tokens:
            if len(tok) < 2:
                continue
            if re.search(rf"\b{fl}\b[^.\n]{{0,80}}\b{re.escape(tok)}\b", ctx_l):
                return True
            if re.search(rf"\bon the {fl} is:[^.\n]*\b{re.escape(tok)}\b", ctx_l):
                return True
    return False


def _substance_visible_in_cool_fixture(
    ctx_text: str,
    task: str,
    action_history=None,
) -> bool:
    """True when task/focused substance is already inside a cooling fixture."""
    ctx_l = (ctx_text or "").lower()
    tokens = set(_task_target_tokens(task))
    focused = _focused_substance_focus_phrase(task, action_history, ctx_text)
    if focused:
        tokens |= _action_content_tokens(focused.replace("focus on ", "", 1))
    tokens -= {"substance", "focus", "on", "the", "a", "an", "called"}
    if not tokens:
        return False
    cool_names = ("freezer", "fridge", "refrigerator", "cooler")
    for fixture in cool_names:
        fl = re.escape(fixture)
        for tok in tokens:
            if len(tok) < 2:
                continue
            if re.search(
                rf"\bin the (?:ultra low temperature )?{fl} is:[^.\n]*\b{re.escape(tok)}\b",
                ctx_l,
            ):
                return True
            if re.search(
                rf"\b{fl}\b[^.\n]{{0,100}}\b{re.escape(tok)}\b",
                ctx_l,
            ):
                return True
    return False


def _dest_has_heat_fixture(dest_phrase: str) -> bool:
    dest_l = (dest_phrase or "").lower()
    if is_heat_fixture_phrase(dest_l):
        return True
    return any(w in dest_l for w in _PRE_FOCUS_HEAT_FIXTURE_WORDS)


def _transfer_action_targets_substance(
    action: str,
    task: str,
    action_history=None,
    ctx_text: str = "",
) -> bool:
    """True when pour/move action manipulates the focused or task substance."""
    al = (action or "").strip().lower()
    family = action_verb_family(al)
    task_t = _task_target_tokens(task)
    if family == "pour":
        m = re.match(r"^pour (.+?) into (.+)", al)
        if not m or not _dest_has_heat_fixture(m.group(2)):
            return False
        src = m.group(1).strip()
        if _object_matches_focused_substance(src, task, action_history, ctx_text):
            return True
        src_t = _action_content_tokens(src)
        return bool(src_t & task_t) or "substance" in src
    if family == "move":
        m = re.match(r"^move (.+?) to (.+)", al)
        if not m or not _dest_has_heat_fixture(m.group(2)):
            return False
        obj = m.group(1).strip()
        if _object_matches_focused_substance(obj, task, action_history, ctx_text):
            return True
        return bool(_action_content_tokens(obj) & task_t)
    return False


def _context_has_visible_fixtures(ctx_text: str, fixture_words: frozenset[str]) -> bool:
    if not fixture_words:
        return False
    ctx_l = (ctx_text or "").lower()
    return any(re.search(rf"\b{re.escape(w)}\b", ctx_l) for w in fixture_words)


def _task_manipulation_destination_fixture_words(
    task: str,
    ctx_text: str = "",
    env_valid: set | None = None,
) -> frozenset[str]:
    if not _task_has_post_focus_phase(task):
        return frozenset()
    fixtures = set(heat_fixtures_in_environment(env_valid, ctx_text))
    if _task_favors_cold_storage(task):
        fixtures.update(
            f for f in discover_fixtures_from_observation(ctx_text, heat=False)
        )
    if not fixtures and _task_favors_heat_change(task):
        fixtures.update(_PRE_FOCUS_HEAT_FIXTURE_WORDS)
    return frozenset(fixtures)


def _fixture_associated_rooms(
    fixture: str,
    action_history=None,
    ctx_text: str = "",
) -> tuple[str, ...]:
    """Rooms where fixture was last seen; no static token→room map."""
    room = fixture_last_seen_room(fixture, action_history, ctx_text)
    if room:
        return (room,)
    current = _extract_current_room_name(ctx_text)
    if current and fixture_visible_in_context(ctx_text, fixture):
        return (current,)
    return ()


def _task_manipulation_destination_rooms(
    task: str,
    ctx_text: str = "",
    action_history=None,
    env_valid: set | None = None,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for room in _extract_task_location_hint_rooms(task):
        rn = _normalize_room_name(room)
        if rn not in seen:
            seen.add(rn)
            ordered.append(rn)
    current = _extract_current_room_name(ctx_text)
    for fixture in _task_manipulation_destination_fixture_words(task, ctx_text, env_valid):
        if fixture_visible_in_context(ctx_text, fixture) and current:
            rn = _normalize_room_name(current)
            if rn not in seen:
                seen.add(rn)
                ordered.insert(0, rn)
            continue
        for room in _fixture_associated_rooms(fixture, action_history, ctx_text):
            if room not in seen:
                seen.add(room)
                ordered.append(room)
    return ordered


def _focused_substance_source_fixtures(
    task: str,
    action_history=None,
    ctx_text: str = "",
) -> frozenset[str]:
    focused = _focused_substance_focus_phrase(task, action_history, ctx_text).lower()
    if not focused:
        return frozenset()
    pool = _PRE_FOCUS_LIQUID_SOURCE_WORDS | frozenset({"toilet"})
    found: set[str] = set()
    for w in pool:
        if w in focused:
            found.add(w)
    m = re.search(r"substance in (.+)", focused)
    if m:
        container = m.group(1).strip()
        for w in pool:
            if w in container:
                found.add(w)
    return frozenset(found)


def _pour_source_aligns_with_focus(
    src_phrase: str,
    task: str,
    action_history=None,
    ctx_text: str = "",
) -> bool:
    if _object_matches_focused_substance(src_phrase, task, action_history, ctx_text):
        return True
    src_l = (src_phrase or "").strip().lower()
    if not src_l:
        return False
    focused = _focused_substance_focus_phrase(task, action_history, ctx_text)
    if focused:
        m = re.search(r"substance in (.+)", focused)
        if m and m.group(1).strip() in src_l:
            return True
    fixtures = _focused_substance_source_fixtures(task, action_history, ctx_text)
    return any(w in src_l for w in fixtures)


def _prep_collect_pour_min_score() -> int:
    return 94


def _manipulation_delivery_action_re(task: str = "") -> re.Pattern[str]:
    fixtures = _task_manipulation_destination_fixture_words(task) or _PRE_FOCUS_HEAT_FIXTURE_WORDS
    alt = "|".join(re.escape(w) for w in sorted(fixtures))
    return re.compile(rf"\b(?:pour .+ into|move .+ to) (?:{alt})\b")


def _transfer_shuffle_destination_fixtures(
    task: str,
    action_history=None,
    ctx_text: str = "",
) -> frozenset[str]:
    focused_fixtures = _focused_substance_source_fixtures(task, action_history, ctx_text)
    return focused_fixtures | _PRE_FOCUS_LIQUID_SOURCE_WORDS | frozenset({"toilet"})


# Public aliases for core.task_semantics re-export
context_has_visible_fixtures = _context_has_visible_fixtures
task_manipulation_destination_fixture_words = _task_manipulation_destination_fixture_words
fixture_associated_rooms = _fixture_associated_rooms
task_manipulation_destination_rooms = _task_manipulation_destination_rooms
focused_substance_source_fixtures = _focused_substance_source_fixtures
pour_source_aligns_with_focus = _pour_source_aligns_with_focus
prep_collect_pour_min_score = _prep_collect_pour_min_score
manipulation_delivery_action_re = _manipulation_delivery_action_re
transfer_shuffle_destination_fixtures = _transfer_shuffle_destination_fixtures

def _solid_substance_prep_in_preferred_room(task: str, ctx_text: str) -> bool:
    """Legacy hook: fixture visibility must not imply the task target is already here."""
    if _task_target_missing(task, ctx_text):
        return False
    return False


def _state_of_matter_post_focus_phase(
    task: str,
    current_score: int | None,
    action_history=None,
    ctx_text: str = "",
) -> str | None:
    """
    After focus, infer cool/heat/precool from task polarity + observation.
    Score is only a soft completion hint (score>=95 → done), not the phase driver.
    """
    focus_ok = _task_focus_already_satisfied(
        task, action_history, ctx_text, current_score,
    )
    if not focus_ok and (current_score is None or current_score < 30):
        return None
    if not _task_needs_substance_preparation(task) or not _task_has_post_focus_phase(task):
        return None
    score = current_score if current_score is not None else 0
    if score >= 95:
        return None

    favors_cold = _task_favors_cold_storage(task)
    favors_heat = _task_favors_heat_change(task)
    # Generic "change state of matter" without cool/heat verb → heat by default.
    task_l = (task or "").lower()
    if not favors_cold and not favors_heat:
        if re.search(r"state\s+of\s+matter|change.{0,40}state", task_l):
            favors_heat = True
        else:
            return None

    in_cool = _substance_visible_in_cool_fixture(ctx_text, task, action_history)
    on_heat = _substance_visible_on_heat_fixture(ctx_text, task, action_history)

    if favors_cold and not favors_heat:
        return "cool"
    if favors_heat and not favors_cold:
        # Observation-first: if already on heat, stay in heat; if still cold-stored
        # while melting, allow a short precool-exit via heat (not score gates).
        if on_heat:
            return "heat"
        if in_cool and re.search(r"\bmelt\b", task_l):
            # Melting a pre-cooled solid: leave freezer then heat (handled by transfer/heat).
            return "heat"
        return "heat"
    # Both cool and heat cues (rare): prefer observation polarity.
    if in_cool and not on_heat:
        return "cool"
    return "heat"


def _heat_fixture_active(ctx_text: str, fixture: str) -> bool:
    return fixture_is_active(ctx_text, fixture)


def _focused_substance_is_solid(task: str) -> bool:
    """
    True when the substance is obtained by search/pickup rather than sink fill.
    Uses task structure (prep + focus + state change), not a substance whitelist.
    """
    if _task_needs_liquid_fixture_prep(task, ""):
        return False
    if _task_needs_substance_preparation(task) and _task_has_post_focus_phase(task):
        targets = _task_target_tokens(task)
        if targets & {"water", "liquid", "steam"}:
            return False
        return True
    return _task_favors_cold_storage(task) and "ice" in (task or "").lower()


def _heat_fixtures_fully_active(
    ctx_text: str,
    action_history=None,
    failed_actions: set | None = None,
    env_valid: set | None = None,
) -> bool:
    """True when at least one usable heat fixture in context is active."""
    failed = failed_actions or set()
    fixtures = heat_fixtures_in_environment(env_valid, ctx_text)
    if not fixtures:
        return _heat_fixture_active(ctx_text, "stove") or _heat_fixture_active(ctx_text, "oven")
    for fixture in fixtures:
        if fixture_is_unavailable(fixture, action_history, failed, ctx_text):
            continue
        if fixture_is_active(ctx_text, fixture, action_history):
            return True
    return False


def _heat_setup_pending(
    task: str,
    action_history=None,
    ctx_text: str = "",
    current_score: int | None = None,
    failed_actions: set | None = None,
) -> bool:
    """True when heat phase still needs fixture activation before substance transfer."""
    if not _task_focus_already_satisfied(
        task, action_history, ctx_text, current_score,
    ):
        return False
    if current_score is not None and current_score >= 95:
        return False
    som = _state_of_matter_post_focus_phase(
        task, current_score, action_history, ctx_text,
    )
    if som != "heat":
        return False
    if _solid_heat_prep_pending(
        task, action_history, ctx_text, current_score, failed_actions,
    ):
        return True
    if _focused_substance_is_solid(task):
        return False
    # Observation-first: liquids need an active heat fixture before pour/transfer.
    if _stove_unavailable(action_history, failed_actions, ctx_text):
        return False
    return not _heat_fixtures_fully_active(
        ctx_text, action_history, failed_actions, env_valid=None,
    )


def _solid_heat_prep_pending(
    task: str,
    action_history=None,
    ctx_text: str = "",
    current_score: int | None = None,
    failed_actions: set | None = None,
) -> bool:
    """
    Solid substances need container pickup + heat fixture activation before transfer.
    Driven by observation (on-heat / heat-active), not score bands.
    """
    if not _focused_substance_is_solid(task):
        return False
    if _state_of_matter_post_focus_phase(
        task, current_score, action_history, ctx_text,
    ) != "heat":
        return False
    if not _task_focus_already_satisfied(
        task, action_history, ctx_text, current_score,
    ):
        return False
    if current_score is not None and current_score >= 95:
        return False
    on_heat = _substance_visible_on_heat_fixture(ctx_text, task, action_history)
    heat_on = _heat_fixtures_fully_active(ctx_text, action_history, failed_actions)
    if on_heat and heat_on:
        return False
    if on_heat and not heat_on:
        # Delivered but fixture still cold — activate only.
        return True
    if _substance_transfer_completed(task, action_history, ctx_text, current_score):
        return False
    # Not yet on heat: own vessel/open/activate until a heat source is ready;
    # once heat is ready, transfer (incl. pick) owns the next steps.
    return not heat_on


def _pick_solid_heat_container_prep_substitute(
    env_valid: set,
    recent: set,
    ctx_text: str,
    task: str,
    action_history=None,
    current_score: int | None = None,
    failed_actions: set | None = None,
    logger=None,
) -> str | None:
    """After solid-substance focus: pick vessel, load substance, move to heat fixture."""
    if not _solid_heat_prep_pending(
        task, action_history, ctx_text, current_score, failed_actions,
    ):
        return None
    failed = failed_actions or set()
    held = _held_prep_container_phrase(ctx_text, action_history)
    task_t = _task_target_tokens(task)
    focused_obj = _focused_substance_focus_phrase(task, action_history, ctx_text)
    if focused_obj.startswith("focus on "):
        focused_obj = focused_obj[len("focus on ") :].strip()
    ranked: list[tuple[int, str]] = []

    for va in env_valid:
        al = (va or "").strip().lower()
        if al in recent or al in failed:
            continue
        if _is_disallowed_substitute_action(va):
            continue
        if not action_grounded_in_context(va, ctx_text):
            continue
        if _is_distractor_pour_action(va, task, action_history, ctx_text):
            continue
        family = action_verb_family(va)
        score = 0

        if family in ("pick_up", "take"):
            obj = _action_manipulation_object(va)
            if obj and _is_prep_container_object(obj) and not held:
                score = 82

        elif family == "move":
            m = re.match(r"move (.+?) to (.+)", al)
            if m:
                obj_phrase, dest = m.group(1).strip(), m.group(2).strip()
                if is_heat_fixture_phrase(dest):
                    if held and (
                        _object_matches_focused_substance(
                            obj_phrase, task, action_history, ctx_text,
                        )
                        or focused_obj in obj_phrase
                        or held in obj_phrase
                    ):
                        score = 96
                    elif held and held in obj_phrase:
                        score = 94
                elif _is_prep_container_object(dest) and (
                    _object_matches_focused_substance(
                        obj_phrase, task, action_history, ctx_text,
                    )
                    or focused_obj in obj_phrase
                    or bool(task_t & _action_content_tokens(obj_phrase))
                ):
                    score = 90

        elif family == "activate" and is_heat_fixture_phrase(
            al[len("activate ") :].strip() if al.startswith("activate ") else "",
        ):
            score = 78

        elif family == "open":
            open_target = al[len("open ") :].strip() if al.startswith("open ") else ""
            if is_heat_fixture_phrase(open_target):
                score = 72

        if score > 0:
            ranked.append((score, al))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    if ranked:
        if logger:
            logger.info(f"Solid heat prep substitute: {ranked[0][1]!r}")
        return ranked[0][1]
    return None


def _transfer_min_score(
    task: str,
    action_history=None,
    failed_actions: set | None = None,
) -> int:
    """
    Soft legacy floor used only by ranking heuristics.
    Phase gates use observation predicates instead of this threshold.
    """
    if _focused_substance_is_solid(task):
        return 0
    if not _stove_unavailable(action_history, failed_actions, ""):
        return 0
    return 0


def _heat_sources_ready(
    ctx_text: str,
    action_history=None,
    env_valid: set | None = None,
) -> bool:
    """True when any discovered heat fixture is active."""
    if active_heat_fixtures(ctx_text, action_history, env_valid):
        return True
    for act in reversed([(a or "").strip().lower() for a in (action_history or [])[-20:]]):
        if act.startswith("activate ") and is_heat_fixture_phrase(act[len("activate ") :]):
            return True
    return False


def _stove_unavailable(
    action_history=None,
    failed_actions: set | None = None,
    ctx_text: str = "",
    env_valid: set | None = None,
) -> bool:
    return primary_heat_fixture_unavailable(
        action_history, failed_actions, ctx_text, env_valid,
    )


def _substance_transfer_completed(
    task: str,
    action_history=None,
    ctx_text: str = "",
    current_score: int | None = None,
) -> bool:
    """True when focused substance has been delivered to a heat fixture."""
    if _substance_visible_on_heat_fixture(ctx_text, task, action_history):
        return True
    score = int(current_score or 0)
    if score >= 95:
        return True
    focused = _focused_substance_focus_phrase(task, action_history, ctx_text)
    if not focused:
        return False
    saw_focus = False
    for act in (action_history or []):
        al = (act or "").strip().lower()
        if al.startswith("focus on"):
            saw_focus = True
            continue
        if not saw_focus:
            continue
        family = action_verb_family(al)
        if family == "pour":
            m = re.match(r"^pour (.+?) into (.+)", al)
            if m and is_heat_fixture_phrase(m.group(2)):
                if _object_matches_focused_substance(
                    m.group(1), task, action_history, ctx_text,
                ):
                    return True
        elif family == "move":
            m = re.match(r"^move (.+?) to (.+)", al)
            if m and is_heat_fixture_phrase(m.group(2)):
                if _object_matches_focused_substance(
                    m.group(1), task, action_history, ctx_text,
                ):
                    return True
    return False


def _substance_transfer_pending(
    task: str,
    action_history=None,
    ctx_text: str = "",
    current_score: int | None = None,
    failed_actions: set | None = None,
) -> bool:
    """True when focused substance still needs moving/pouring onto a heat source."""
    if not _task_focus_already_satisfied(
        task, action_history, ctx_text, current_score,
    ):
        return False
    if current_score is not None and current_score >= 100:
        return False
    if _substance_transfer_completed(task, action_history, ctx_text, current_score):
        return False
    if _substance_visible_on_heat_fixture(ctx_text, task, action_history):
        return False
    if not _heat_sources_ready(ctx_text, action_history):
        return False
    if _heat_setup_pending(
        task, action_history, ctx_text, current_score, failed_actions,
    ):
        return False
    if not _focused_substance_focus_phrase(task, action_history, ctx_text):
        return False
    return True


def _state_of_matter_manipulation_sub_phase(
    task: str,
    current_score: int | None,
    action_history=None,
    ctx_text: str = "",
    failed_actions: set | None = None,
) -> str:
    """Finer post-focus sub-phase: heat_setup | transfer | wait (observation-first)."""
    som = _state_of_matter_post_focus_phase(
        task, current_score, action_history, ctx_text,
    )
    if not som:
        return ""
    if som == "cool":
        if _substance_visible_in_cool_fixture(ctx_text, task, action_history):
            return "wait"
        return "cool"
    if _heat_setup_pending(
        task, action_history, ctx_text, current_score, failed_actions,
    ):
        return "heat_setup"
    if _substance_transfer_pending(
        task, action_history, ctx_text, current_score, failed_actions,
    ):
        return "transfer"
    on_heat = _substance_visible_on_heat_fixture(ctx_text, task, action_history)
    if on_heat and (
        active_heat_fixtures(ctx_text, action_history)
        or _heat_sources_ready(ctx_text, action_history)
    ):
        return "wait"
    if som == "heat" and not _heat_sources_ready(ctx_text, action_history):
        return "heat_setup"
    return som


def _pick_substance_transfer_substitute(
    env_valid: set,
    recent: set,
    ctx_text: str,
    task: str,
    action_history=None,
    current_score: int | None = None,
    failed_actions: set | None = None,
    logger=None,
) -> str | None:
    """Pick move/pour/pick actions to place focused substance on oven/stove."""
    if not _substance_transfer_pending(
        task, action_history, ctx_text, current_score, failed_actions,
    ):
        return None
    focused = _focused_substance_focus_phrase(task, action_history, ctx_text)
    if not focused:
        return None
    failed = failed_actions or set()
    foc_room = _focused_substance_container_room(task, action_history, ctx_text)
    cur_room = _extract_current_room_name(ctx_text)
    dest_fixtures = heat_destination_fixtures(
        env_valid, ctx_text, action_history, failed,
    )
    focused_obj = focused if not focused.startswith("focus on") else focused.replace("focus on ", "", 1).strip()
    ranked: list[tuple[int, str]] = []
    nav_ranked: list[tuple[int, str]] = []
    recent_nav_to_foc = foc_room and any(
        (a or "").strip().lower() in {f"go to {foc_room}", f"open door to {foc_room}"}
        for a in list(recent)[-5:]
    )

    for va in env_valid:
        al = (va or "").strip().lower()
        if al in recent or al in failed:
            continue
        if _is_disallowed_substitute_action(va):
            continue
        family = action_verb_family(va)
        heat_xfer = family in ("pour", "move") and _heat_transfer_action_grounded(va, ctx_text)
        if not heat_xfer and not action_grounded_in_context(va, ctx_text):
            continue
        if not heat_xfer and _is_post_focus_unproductive(
            va, task, action_history, ctx_text, current_score,
        ):
            continue
        score = 0

        if family == "move":
            m = re.match(r"move (.+?) to (.+)", al)
            if m:
                obj_phrase, dest = m.group(1).strip(), m.group(2).strip()
                if is_heat_fixture_phrase(dest) and (
                    _object_matches_focused_substance(
                        obj_phrase, task, action_history, ctx_text,
                    )
                    or focused_obj in obj_phrase
                    or _action_content_tokens(focused_obj) <= _action_content_tokens(obj_phrase)
                ):
                    score = 88

        elif family == "pour":
            m = re.match(r"^pour (.+?) into (.+)", al)
            if m and not _is_distractor_pour_action(va, task, action_history, ctx_text):
                src, dest = m.group(1).strip(), m.group(2).strip()
                src_t = _action_content_tokens(src)
                task_t = _task_target_tokens(task)
                if is_heat_fixture_phrase(dest) and (
                    _object_matches_focused_substance(
                        src, task, action_history, ctx_text,
                    )
                    or focused_obj in src
                    or "substance in" in focused_obj
                    or (src_t & task_t)
                ):
                    score = 92 if cur_room == foc_room else 82

        elif family in ("pick_up", "take"):
            obj = _action_manipulation_object(va)
            if obj and (
                _object_matches_focused_substance(obj, task, action_history, ctx_text)
                or focused_obj in obj
                or _action_content_tokens(focused_obj) & _action_content_tokens(obj)
            ):
                if _focused_substance_is_solid(task) and int(current_score or 0) >= _transfer_min_score(
                    task, action_history, failed_actions,
                ):
                    if _substance_visible_on_heat_fixture(
                        ctx_text, task, action_history,
                    ):
                        score = 0
                    else:
                        score = 76
                else:
                    score = 76

        elif al.startswith(("go to ", "open door to ")) and foc_room:
            dest = _nav_dest_room(al)
            if dest == foc_room and cur_room != foc_room and not recent_nav_to_foc:
                nav_ranked.append((58, al))

        elif family == "open" and cur_room == foc_room:
            if any(w in al for w in dest_fixtures):
                score = 70

        if score > 0:
            ranked.append((score, al))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    if ranked:
        choice = ranked[0][1]
        if logger:
            logger.info(f"Post-focus transfer substitute: {choice!r}")
        return choice
    if cur_room == foc_room:
        return None
    nav_ranked.sort(key=lambda x: (-x[0], x[1]))
    if nav_ranked:
        choice = nav_ranked[0][1]
        if logger:
            logger.info(f"Post-focus transfer nav: {choice!r}")
        return choice
    return None
def stagnant_early_stop_threshold(
    task: str,
    episode_progress: dict,
    past_actions,
    ctx_text: str,
    current_score: int,
) -> int:
    """Dynamic early-stop limit: allow more steps during post-focus state-of-matter work."""
    threshold = STAGNANT_EARLY_STOP_THRESHOLD
    score_max = int(episode_progress.get("score_max", 0) or 0)
    if score_max >= 70 and current_score < 100:
        if _state_of_matter_post_focus_phase(
            task, current_score, past_actions, ctx_text,
        ):
            threshold += STAGNANT_EARLY_STOP_SOM_BONUS + 8
        elif _task_post_focus_manipulation_pending(
            task, past_actions, ctx_text, current_score,
        ):
            threshold += 8
    elif (
        score_max >= 26
        and current_score < 70
        and _task_post_focus_manipulation_pending(
            task, past_actions, ctx_text, current_score,
        )
    ):
        threshold += 6
    elif (
        score_max >= 35
        and current_score < 100
        and _state_of_matter_post_focus_phase(task, current_score, past_actions, ctx_text)
    ):
        threshold += STAGNANT_EARLY_STOP_SOM_BONUS + 4
    if (
        score_max <= 6
        and current_score < 30
        and _focus_subgoal_pending(task, past_actions, ctx_text)
    ):
        for act in reversed([(a or "").strip().lower() for a in (past_actions or [])[-8:]]):
            if act.startswith(("look at ", "examine ")):
                threshold += 8
                break
    return threshold

def _score_state_of_matter_action(
    va: str, phase: str, ctx_text: str, task: str, action_history=None,
    current_score: int | None = None,
    failed_actions: set | None = None,
) -> int:
    """Rank env actions for a post-focus temperature phase (all state-of-matter tasks)."""
    al = (va or "").strip().lower()
    family = action_verb_family(va)
    score = 0
    transfer_pending = _substance_transfer_pending(
        task, action_history, ctx_text, current_score, failed_actions,
    )
    if phase == "heat" and transfer_pending:
        foc_room = _focused_substance_container_room(task, action_history, ctx_text)
        cur_room = _extract_current_room_name(ctx_text)
        heat_fixtures = heat_fixtures_in_environment(set(), ctx_text)
        if not heat_fixtures:
            heat_fixtures = ["stove", "oven"]
        dest_fixtures = tuple(
            f for f in heat_fixtures
            if not fixture_is_unavailable(f, action_history, failed_actions, ctx_text)
        ) or ("oven",)
        focused = _focused_substance_focus_phrase(task, action_history, ctx_text)
        focused_obj = (
            focused.replace("focus on ", "", 1).strip() if focused else ""
        )
        if family == "move":
            m = re.match(r"move (.+?) to (.+)", al)
            if m and any(d in m.group(2) for d in dest_fixtures):
                if _object_matches_focused_substance(
                    m.group(1), task, action_history, ctx_text,
                ) or (focused_obj and focused_obj in m.group(1)):
                    score += 90
        elif family == "pour":
            m = re.match(r"^pour (.+?) into (.+)", al)
            if m and not _is_distractor_pour_action(va, task, action_history, ctx_text):
                if any(d in m.group(2) for d in dest_fixtures) and (
                    _object_matches_focused_substance(
                        m.group(1), task, action_history, ctx_text,
                    )
                    or (focused_obj and focused_obj in m.group(1))
                ):
                    score += 85
        elif family in ("pick_up", "take"):
            obj = _action_manipulation_object(va)
            if obj and _object_matches_focused_substance(
                obj, task, action_history, ctx_text,
            ):
                score += 78
        elif al.startswith(("go to ", "open door to ")) and foc_room:
            dest = _nav_dest_room(al)
            if dest == foc_room and cur_room != foc_room:
                score += 55
            elif foc_room and cur_room == foc_room and dest != foc_room:
                score -= 40
            elif cur_room == foc_room:
                score = 0
        if score > 0:
            return score
    elif phase == "precool":
        if family == "open" and "freezer" in al:
            score += 42
        if family == "move" and re.search(r"\bto freezer\b", al):
            score += 48
        if al == "wait" or al.startswith("wait "):
            score += 38
        if family in ("look_at", "examine") and (
            "substance" in al
            or bool(_action_content_tokens(al) & _task_target_tokens(task))
        ):
            score += 18
        if family == "use" and "thermometer" in al:
            score += 22
    elif phase == "heat":
        heat_setup = _heat_setup_pending(
            task, action_history, ctx_text, current_score, failed_actions,
        )
        held = _held_prep_container_phrase(ctx_text, action_history)
        task_t = _task_target_tokens(task)
        scr = int(current_score or 0)
        min_xfer = _transfer_min_score(task, action_history, failed_actions)
        substance_on_heat = _substance_visible_on_heat_fixture(
            ctx_text, task, action_history,
        )
        heat_fixtures = heat_fixtures_in_environment(None, ctx_text) or ["stove", "oven"]
        heat_stalled = heating_progress_stalled(scr, action_history)
        if family == "move":
            m_move = re.match(r"move (.+?) to (.+)", al)
            if m_move:
                obj_phrase, dest = m_move.group(1).strip(), m_move.group(2).strip()
                obj_t = _action_content_tokens(obj_phrase)
                if is_heat_fixture_phrase(dest):
                    if obj_t & _DISTRACTOR_FOOD_OBJECTS and not (obj_t & task_t):
                        score -= 70
                    elif (
                        obj_t & task_t
                        or _object_matches_focused_substance(
                            obj_phrase, task, action_history, ctx_text,
                        )
                        or _container_holds_task_substance(
                            obj_phrase, task, ctx_text, action_history,
                        )
                    ):
                        score += 58
                    else:
                        score += 6
                    if heat_stalled and is_heat_fixture_phrase(dest):
                        score += 72
                elif re.search(r"\bto (?:sink|bathtub)\b", al):
                    score += 18
                elif re.search(r"\bto (metal pot|pot|cup|bowl|pan)\b", al):
                    if obj_t & task_t or _object_matches_focused_substance(
                        obj_phrase, task, action_history, ctx_text,
                    ):
                        score += 38
                    else:
                        score += 10
                else:
                    score += 12
                if held and held.split()[-1] in al:
                    score += 20
                if (
                    scr >= min_xfer
                    and substance_on_heat
                    and is_heat_fixture_phrase(dest)
                    and (
                        obj_t & task_t
                        or _object_matches_focused_substance(
                            obj_phrase, task, action_history, ctx_text,
                        )
                    )
                ):
                    score -= 85
                if (
                    scr >= min_xfer
                    and substance_on_heat
                    and re.search(r"\bto (?:bowl|cup|pot|pan|beaker|flask|jar)\b", al)
                ):
                    score -= 55
        if family in ("pick_up", "take"):
            obj = _action_manipulation_object(va)
            obj_t = _action_content_tokens(obj or "")
            if obj and (
                _object_matches_focused_substance(obj, task, action_history, ctx_text)
                or _container_holds_task_substance(obj, task, ctx_text, action_history)
            ):
                score += 46
            elif obj_t & task_t:
                score += 40
            elif obj_t & _DISTRACTOR_FOOD_OBJECTS:
                score -= 50
            if scr >= min_xfer and substance_on_heat and _object_matches_focused_substance(
                obj or "", task, action_history, ctx_text,
            ):
                score -= 70
        if family == "pour":
            src = _action_manipulation_object(va)
            src_t = _action_content_tokens(src or "")
            pour_dest = ""
            m_p = re.match(r"^pour .+? into (.+)", al)
            if m_p:
                pour_dest = m_p.group(1).strip()
            into_heat = is_heat_fixture_phrase(pour_dest)
            into_container = bool(re.search(r"\binto (?:metal pot|pot|cup|bowl)\b", al))
            if _is_absurd_pour_action(va, task, ctx_text):
                score -= 120
            elif _is_distractor_pour_action(va, task, action_history, ctx_text):
                score -= 90
            elif into_container or into_heat:
                if (
                    src_t & task_t
                    or _object_matches_focused_substance(
                        src or "", task, action_history, ctx_text,
                    )
                    or _container_holds_task_substance(
                        src or "", task, ctx_text, action_history,
                    )
                ):
                    score += 65 if into_heat else 52
                elif held and held.split()[-1] in al:
                    score += 40
                else:
                    score -= 35
                foc_room = _focused_substance_container_room(task, action_history, ctx_text)
                cur_room = _extract_current_room_name(ctx_text)
                if foc_room and cur_room == foc_room:
                    score += 45
                if active_heat_fixtures(ctx_text, action_history):
                    score += 25
        if family == "open":
            open_target = al[len("open ") :].strip() if al.startswith("open ") else ""
            if is_heat_fixture_phrase(open_target):
                score += 28
                if heat_stalled:
                    score += 55
        if family == "activate":
            act_target = al[len("activate ") :].strip() if al.startswith("activate ") else ""
            if is_heat_fixture_phrase(act_target):
                score += 44
                if heat_stalled:
                    score += 70
                if _current_room_in_task_area(task, ctx_text):
                    score += 20
                if act_target and fixture_is_active(ctx_text, act_target, action_history):
                    score -= 28
                elif (
                    scr < _transfer_min_score(task, action_history, failed_actions)
                    and act_target == "stove"
                    and not _stove_unavailable(action_history, failed_actions, ctx_text)
                ):
                    score += 35
                if heat_setup and not fixture_is_active(
                    ctx_text, act_target, action_history,
                ):
                    score += 55
            else:
                # Switches/batteries are not heat sources; do not outrank pour/move.
                score -= 80
        if heat_setup and al.startswith(("go to ", "open door to ")):
            foc_room = _focused_substance_container_room(task, action_history, ctx_text)
            dest = _nav_dest_room(al)
            if foc_room and dest == foc_room:
                score -= 60
        if family == "activate" and any(w in al for w in _PRE_FOCUS_LIQUID_SOURCE_WORDS):
            score -= 35
        if al.startswith(("go to ", "open door to ")):
            dest = _nav_dest_room(al)
            preferred = _task_preferred_rooms(task)
            ctx = (ctx_text or "").lower()
            foc_room = _focused_substance_container_room(task, action_history, ctx_text)
            if foc_room and dest == foc_room:
                score += 55
            elif dest in preferred[:2]:
                score += 40
            elif not _current_room_in_task_area(task, ctx_text):
                score += 24
            if (
                _current_room_in_task_area(task, ctx_text)
                and any_heat_fixture_in_context(ctx_text)
                and not foc_room
            ):
                score -= 35
            elif not any_heat_fixture_in_context(ctx_text):
                score += 12
        if al == "wait" or al.startswith("wait "):
            score += 32
            if active_heat_fixtures(ctx_text, action_history):
                score += 35
            else:
                score -= 18
            if not any_heat_fixture_in_context(ctx_text):
                score -= 12
        if family in ("look_at", "examine") and (
            "substance" in al
            or bool(_action_content_tokens(al) & _task_target_tokens(task))
        ):
            score += 16
        if family == "use" and "thermometer" in al:
            score += 20
    elif phase == "cool":
        already_cooling = _substance_visible_in_cool_fixture(
            ctx_text, task, action_history,
        )
        if family == "open" and any(
            w in al for w in ("freezer", "fridge", "refrigerator", "cooler")
        ):
            score += 10 if already_cooling else 40
        if family == "move" and re.search(
            r"\bto (?:ultra low temperature )?(?:freezer|fridge|refrigerator|cooler)\b",
            al,
        ):
            obj_phrase = ""
            m_cool_move = re.match(r"move (.+?) to ", al)
            if m_cool_move:
                obj_phrase = m_cool_move.group(1).strip()
            if already_cooling:
                # Redundant re-move once substance is cooling — prefer wait.
                score -= 40
            elif _object_matches_focused_substance(
                obj_phrase, task, action_history, ctx_text,
            ) or bool(_action_content_tokens(obj_phrase) & _task_target_tokens(task)):
                score += 45
            else:
                score += 20
        if al == "wait" or al.startswith("wait "):
            score += 70 if already_cooling else 40
        if family in ("look_at", "examine") and (
            "substance" in al
            or bool(_action_content_tokens(al) & _task_target_tokens(task))
        ):
            score += 28 if already_cooling else 20
        if family == "use" and "thermometer" in al:
            score += 55 if already_cooling else 28
    if score > 0:
        score += len(_action_content_tokens(va) & _task_target_tokens(task)) * 4
    return score
def _pick_state_of_matter_post_focus_substitute(
    env_valid: set,
    recent: set,
    ctx_text: str,
    task: str,
    action_history,
    current_score: int | None,
    failed_actions: set | None,
    logger=None,
) -> str | None:
    """Post-focus substitute for melt/boil/freeze style tasks."""
    failed = failed_actions or set()
    held = _held_prep_container_phrase(ctx_text, action_history)
    solid_prep = _solid_heat_prep_pending(
        task, action_history, ctx_text, current_score, failed,
    )
    needs_escalation = (
        solid_prep
        or heating_progress_stalled(current_score, action_history)
        or _stove_unavailable(action_history, failed, ctx_text)
    )
    if solid_prep:
        prep_sub = _pick_solid_heat_container_prep_substitute(
            env_valid, recent, ctx_text, task, action_history,
            current_score, failed, logger,
        )
        if prep_sub and prep_sub not in recent:
            return prep_sub
    if needs_escalation:
        esc = pick_next_heat_escalation_action(
            env_valid, recent, ctx_text, action_history, failed,
            held_container=held or "metal pot",
        )
        if esc and esc not in recent:
            if logger:
                logger.info(f"Heat escalation substitute: {esc!r}")
            return esc
    transfer = _pick_substance_transfer_substitute(
        env_valid, recent, ctx_text, task, action_history,
        current_score, failed_actions, logger,
    )
    if transfer and transfer not in recent:
        return transfer
    phase = _state_of_matter_post_focus_phase(
        task, current_score, action_history, ctx_text,
    )
    if not phase:
        return None
    failed = failed_actions or set()
    ranked: list[tuple[int, str]] = []
    for va in env_valid:
        if va in recent or va in failed:
            continue
        if _is_disallowed_substitute_action(va):
            continue
        if not action_grounded_in_context(va, ctx_text):
            continue
        if not is_safe_explore_action(va, task, ctx_text):
            continue
        if _is_premature_heat_before_focus(va, task, action_history, ctx_text):
            continue
        if _is_post_focus_liquid_fixture_misactivation(va, task, ctx_text, action_history):
            continue
        if _is_post_focus_unproductive(
            va, task, action_history, ctx_text, current_score,
        ):
            continue
        sc = _score_state_of_matter_action(
            va, phase, ctx_text, task, action_history,
            current_score=current_score, failed_actions=failed,
        )
        if sc > 0:
            ranked.append((sc, va))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    if ranked and logger:
        logger.info(f"State-of-matter post-focus ({phase}): {ranked[0][1]!r}")
    return ranked[0][1] if ranked else None
def _score_growth_action(va: str, ctx_text: str, task: str) -> int:
    """Rank actions that advance grow/reproduction tasks after seed/plant focus."""
    al = (va or "").strip().lower()
    family = action_verb_family(va)
    score = 0
    ctx = (ctx_text or "").lower()
    # Reject pour/move onto furniture — only soil/planters progress growth.
    _growth_furniture = {
        "chair", "table", "counter", "bed", "desk", "shelf", "drawer", "cupboard",
    }
    if re.search(r"containing nothing", al):
        return 0
    # Bare empty vessels never advance grow — force soil/seed/water instead.
    if family in ("pick_up", "take") and re.search(
        r"\b(?:cup|bowl|jug|beaker|pot)\b", al,
    ) and not re.search(
        r"\b(?:seed|soil|fertilizer|plant|water|substance)\b", al,
    ):
        return 0
    if re.search(r"\b(?:juice|soda|milk)\b", al) and not re.search(
        r"\b(?:juice|recipe|ingredient)\b", (task or "").lower(),
    ):
        return 0
    if family == "pour":
        dest_m = re.search(r"\binto (.+)$", al)
        dest = (dest_m.group(1) if dest_m else "").strip()
        dest_t = _action_content_tokens(dest)
        if dest_t & _growth_furniture:
            return 0
        if _normalize_room_name(dest) in SCIENCEWORLD_ROOM_NAMES:
            return 0
        if dest_t & {"sink", "drain", "toilet", "bathtub", "shower"}:
            return 0
        if dest_t & {"soil", "pot", "planter", "flower", "jug"}:
            score += 38
        else:
            score -= 20
    if family == "go" or al.startswith(("go to ", "open door to ", "teleport to ")):
        dest = _nav_dest_room(va)
        if dest in {"greenhouse", "outside"}:
            score += 55
        elif dest in _growth_furniture or dest in {"bathroom", "kitchen", "bedroom"}:
            score -= 25
    if family == "wait" or al.startswith("wait "):
        score += 42
        try:
            current = _extract_current_room_name(ctx_text)
            if current and current not in {"greenhouse", "outside"}:
                score -= 35
        except Exception:
            pass
    if family == "move" and re.search(
        r"\bto (soil|flower pot|pot|planter|greenhouse|outside)\b", al,
    ):
        score += 48
    if family == "move":
        dest_m = re.search(r"\bto (.+)$", al)
        dest_t = _action_content_tokens((dest_m.group(1) if dest_m else "").strip())
        if dest_t & _growth_furniture:
            return 0
    if family == "activate" and any(w in al for w in ("sink", "faucet", "shovel")):
        if re.search(r"\b(?:sink|faucet)\b", al):
            try:
                current = _extract_current_room_name(ctx_text)
                if current and current not in {"greenhouse", "outside"}:
                    return 0
            except Exception:
                return 0
        score += 28
    if family == "open" and "door" in al:
        score += 18
    if family == "use" and "shovel" in al:
        score += 32
    if family == "use" and re.search(r"\b(?:soil|fertilizer|seed)\b", al):
        score += 36
    if "soil" in ctx and family == "move":
        score += 16
    if "seed" in ctx or "plant" in ctx:
        if family in ("wait", "pour", "move", "use"):
            score += 12
    score += len(_action_content_tokens(va) & _task_target_tokens(task)) * 5
    return score


def _pick_growth_post_focus_substitute(
    env_valid: set,
    recent: set,
    ctx_text: str,
    task: str,
    action_history,
    current_score: int | None,
    failed_actions: set | None,
    logger=None,
) -> str | None:
    """Post-focus substitute for grow-plant / grow-fruit style tasks."""
    if not _task_needs_growth_after_focus(task):
        return None
    if not _task_focus_already_satisfied(
        task, action_history, ctx_text, current_score,
    ):
        return None
    failed = failed_actions or set()
    try:
        from core.navigation_helpers import (
            _extract_current_room_name,
            _nav_actions_toward_room,
            _preferred_target_rooms,
        )
        current = _extract_current_room_name(ctx_text)
        valid_l = {(str(v) or "").strip().lower() for v in (env_valid or [])}
        recent_l = {str(x).strip().lower() for x in (recent or set()) if x}
        blocked = recent_l | {str(x).strip().lower() for x in failed if x}
        for room in _preferred_target_rooms(task, action_history, ctx_text) or []:
            if room in {"greenhouse", "outside"} and current and room == current:
                continue
            for cand in _nav_actions_toward_room(room, valid_l, blocked, ctx_text):
                if cand not in recent and cand not in failed:
                    if logger:
                        logger.info(f"Growth post-focus nav substitute: {cand!r}")
                    return cand
    except Exception:
        pass
    ranked: list[tuple[int, str]] = []
    for va in env_valid:
        if va in recent or va in failed:
            continue
        if _is_disallowed_substitute_action(va):
            continue
        al = (va or "").strip().lower()
        if re.search(r"\binto (?:sink|drain|toilet|bathtub|shower|inventory)\b", al):
            continue
        if al.startswith("activate ") and re.search(r"\b(?:sink|faucet)\b", al):
            continue
        if not action_grounded_in_context(va, ctx_text):
            continue
        if not is_safe_explore_action(va, task, ctx_text):
            continue
        if _is_post_focus_unproductive(
            va, task, action_history, ctx_text, current_score,
        ):
            continue
        if va.strip().lower().startswith("focus on"):
            continue
        sc = _score_growth_action(va, ctx_text, task)
        if sc > 0:
            ranked.append((sc, va))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    if ranked and logger:
        logger.info(f"Growth post-focus substitute: {ranked[0][1]!r}")
    return ranked[0][1] if ranked else None


def _prep_actions_for_missing_substance(
    task: str,
    env_valid: set,
    recent: set,
    ctx_text: str,
    action_history=None,
    failed_actions: set | None = None,
    planned: str = "",
) -> list[str]:
    """Ordered prep actions before focus when the task substance is not visible."""
    if not _task_needs_substance_preparation(task):
        return []
    if _prep_container_ready_for_focus(task, ctx_text, action_history):
        return []
    failed = failed_actions or set()
    phase = _substance_prep_phase(task, ctx_text, action_history, failed_actions)
    if phase == "focus_ready":
        return []
    preferred = _prep_preferred_containers(task, ctx_text, action_history, planned)
    ranked: list[tuple[int, str]] = []
    for va in env_valid:
        if va in recent or va in failed:
            continue
        if not action_grounded_in_context(va, ctx_text):
            continue
        if not _is_pre_focus_preparation_action(va, task, ctx_text, action_history):
            continue
        if _is_premature_heat_before_focus(va, task, action_history, ctx_text):
            continue
        score = _score_substance_prep_action(
            va, phase, preferred, task, ctx_text, action_history,
        )
        if score <= 0:
            continue
        if _cold_open_blocked_on_heat(va, task):
            continue
        ranked.append((score, va))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [a for _, a in ranked]


def _room_has_substance_container_hints(ctx_text: str, task: str) -> bool:
    """True when observation/history suggests a focusable container may be present."""
    if _observation_substance_focus_plans(ctx_text, task):
        return True
    ctx = (ctx_text or "").lower()
    if re.search(r"\b(toilet|metal pot|bathtub)\b", ctx):
        return True
    if re.search(r"\binventory\b.*\b(pot|cup|jar)\b", ctx, re.S):
        return True
    return False


def _focus_substitute_nav_allowed(
    nav_action: str,
    task: str,
    ctx_text: str,
    action_history=None,
    failed_actions: set | None = None,
) -> bool:
    """During focus subgoal, avoid leaving a room that may already hold the target."""
    if not nav_action:
        return False
    al = (nav_action or "").strip().lower()
    if not al.startswith(("go to ", "open door to ")):
        return True
    if not _focus_subgoal_pending(task, action_history, ctx_text):
        return True
    dest = _nav_dest_room(nav_action)
    current = _extract_current_room_name(ctx_text)
    if current and dest == current:
        return False
    prep_phase = _substance_prep_phase(
        task, ctx_text, action_history, failed_actions,
    )
    if prep_phase == "alt_liquid_nav" and dest == "bathroom":
        return True
    if prep_phase == "solid_search":
        return True
    if _liquid_prep_stalled(task, ctx_text, action_history, failed_actions) and dest == "bathroom":
        return True
    if prep_phase in ("move_liquid", "fill", "wait_fill") and _current_room_in_task_area(
        task, ctx_text,
    ):
        fixtures = _available_liquid_fixtures(ctx_text, action_history, failed_actions)
        if fixtures and dest in ("bathroom", "hallway", "outside"):
            if not _liquid_prep_stalled(task, ctx_text, action_history, failed_actions):
                return False
    if _current_room_in_task_area(task, ctx_text) and prep_phase not in (
        None, "focus_ready",
    ):
        if dest not in (current,) and _available_liquid_fixtures(
            ctx_text, action_history, failed_actions,
        ):
            return False
    if _task_target_missing(task, ctx_text) or _current_room_weak_for_task(task, ctx_text):
        if dest in _task_deprioritized_rooms(task):
            return False
        if (
            _is_connector_only_nav_dest(nav_action, task)
            and _preferred_room_reachable_from_context(task, ctx_text)
        ):
            return False
        if (
            dest in _preferred_target_rooms(task)
            and dest not in _task_deprioritized_rooms(task)
        ):
            return True
    if _current_room_in_task_area(task, ctx_text):
        return False
    if _room_has_substance_container_hints(ctx_text, task):
        return False
    return True

def _history_substance_container_focus_plans(
    action_history, ctx_text: str, task: str,
) -> list[str]:
    """Infer container-focus phrases after liquid-source activation in recent history."""
    if not _task_needs_substance_preparation(task):
        return []
    hist = " ".join((a or "").strip().lower() for a in (action_history or [])[-20:])
    if not re.search(r"\bactivate (?:sink|bathtub|faucet)\b", hist):
        return []
    ctx = (ctx_text or "").lower()
    seen: set[str] = set()
    out: list[str] = []
    for m in re.finditer(r"\b(metal pot|glass cup|ceramic cup|tin cup|pot)\b", ctx):
        phrase = f"focus on substance in {m.group(1)}"
        if phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
    return out


_SEQUENCED_COMPARISON_WORD_RE = re.compile(
    r"\b(longest|shortest|largest|smallest|highest|lowest|"
    r"most|least|dominant|recessive|heaviest|lightest|"
    r"steepest|shallowest|fastest|slowest)\b",
    re.I,
)
_CONDITIONAL_FOCUS_CLAUSE_RE = re.compile(
    # Single-clause If-P-then-focus only. Do not span sentences: "determine if
    # unknown substance is conductive. First, focus on …" is sequencing, not a
    # Mendelian if/then answer.
    r"\bif\b([^.]{0,100}?),\s*focus on (?:the |a |an )?([^.,;]+)",
    re.I,
)
_GHOST_NUMBERED_SLOT_RE = re.compile(
    r"^(?:the |a |an )?([a-z][a-z\-]*)\s+\d+$",
    re.I,
)
_CREATE_SUBSTANCE_RE = re.compile(
    r"create the substance\s+'([^']+)'"
    r"|create the substance\s+called\s+([^.,;]+)"
    r"|create (?:a |an |the )?((?:[a-z][a-z\-]*\s+){0,3}paint)\b",
    re.I,
)
_DONE_FOCUS_PRODUCT_RE = re.compile(
    r"when you are (?:completely )?done,?\s*focus on (?:the |a |an )?([^.,;]+)",
    re.I,
)
_INSTRUCTED_FOCUS_RE = re.compile(
    r"\bfocus on (?:the |a |an )?([^.,;]+)",
    re.I,
)
_PRIMARY_PAINT_RE = re.compile(
    r"^(?:red|yellow|blue|orange|green|violet|purple|white|black) paint$",
    re.I,
)
_GENETICS_SPECIMEN_WORDS = frozenset({
    "seed", "pea", "plant", "flower", "sprout", "seedling",
})
_NON_ORGANISM_FOCUS_WORDS = frozenset({
    "inventory", "air", "agent", "axe", "shovel", "dibble", "rake", "trowel",
    "scissors", "knife", "tongs", "spade", "door", "table", "bed", "fountain",
    "pit", "wood", "ground", "wall", "floor",
})
_ORGANISM_HINT_RE = re.compile(
    r"\b(?:egg|baby|adult|juvenile|larva|animal|insect|bird|fish|"
    r"mouse|ant|bee|wolf|bear|fox|rabbit|chipmunk|tortoise|elephant|"
    r"whale|parrot|eagle|owl|beaver|fly|frog|toad)\b",
    re.I,
)


def _comparison_word_direction(word: str) -> str:
    w = (word or "").strip().lower()
    if w in _COMPARISON_MIN_WORDS:
        return "min"
    return "max"


def _sequenced_comparison_directions(task: str) -> list[str]:
    """
    Ordered max/min stages from 'First ... longest ... Then ... shortest' wording.

    A task that names both poles must not collapse to the last (or any-min) pole.
    """
    task_l = (task or "").lower()
    m = re.search(r"\bfirst\b(.{0,220}?)\bthen\b(.{0,220})", task_l, re.S)
    dirs: list[str] = []
    if m:
        for chunk in m.groups():
            wm = _SEQUENCED_COMPARISON_WORD_RE.search(chunk or "")
            if not wm:
                continue
            d = _comparison_word_direction(wm.group(1))
            if not dirs or dirs[-1] != d:
                dirs.append(d)
        if len(dirs) >= 2:
            return dirs
    # ``longest lived then the shortest`` without an explicit First/Then wrapper.
    if re.search(
        r"\b(longest|largest|most|highest)\b.{0,100}\bthen\b.{0,100}"
        r"\b(shortest|smallest|least|lowest)\b",
        task_l,
        re.S,
    ):
        return ["max", "min"]
    if re.search(
        r"\b(shortest|smallest|least|lowest)\b.{0,100}\bthen\b.{0,100}"
        r"\b(longest|largest|most|highest)\b",
        task_l,
        re.S,
    ):
        return ["min", "max"]
    return []


def _sequenced_comparison_stages_done(
    task: str,
    action_history=None,
    current_score: int | None = None,
) -> int:
    """How many sequenced comparison focuses have already scored."""
    stages = _sequenced_comparison_directions(task)
    if not stages:
        return 0
    sc = int(current_score or 0)
    if sc >= 80:
        return len(stages)
    if sc >= 20:
        return min(len(stages) - 1, max(1, sc // 30))
    return 0


def _active_comparison_semantics(
    task: str,
    action_history=None,
    current_score: int | None = None,
) -> tuple[str, str] | None:
    """Comparison (attribute, direction) for the current sequenced stage."""
    base = _extract_comparison_semantics(task)
    if not base:
        return None
    attr, direction = base
    stages = _sequenced_comparison_directions(task)
    if stages:
        done = _sequenced_comparison_stages_done(task, action_history, current_score)
        direction = stages[min(done, len(stages) - 1)]
    return (attr, direction)


def _extract_comparison_semantics(task: str) -> tuple[str, str] | None:
    """Return (attribute, direction) for superlative/comparison tasks; direction is max|min."""
    task_l = (task or "").lower()
    if not _COMPARISON_TASK_RE.search(task_l):
        return None
    stages = _sequenced_comparison_directions(task)
    if stages:
        direction = stages[0]
    else:
        direction = "min" if any(
            re.search(rf"\b{re.escape(w)}\b", task_l) for w in _COMPARISON_MIN_WORDS
        ) else "max"
    if re.search(r"\b(life\s*span|lifespan|longevity|longest.?lived|live[sd]?)\b", task_l):
        return ("lifespan", direction)
    if re.search(r"\b(dominant|recessive|allele|genetic)\b", task_l):
        return ("genetics", direction)
    if re.search(r"\b(largest|smallest|size|tall|heavy|weight|big|small)\b", task_l):
        return ("size", direction)
    if re.search(r"\b(hottest|coldest|temperature|warm|cold|heat)\b", task_l):
        return ("temperature", direction)
    if re.search(r"\b(steepest|shallowest|angle|inclin)", task_l):
        return ("angle", direction)
    if re.search(r"\bfriction\b", task_l):
        return ("friction", direction)
    return ("general", direction)


def _conditional_focus_clauses(task: str) -> list[tuple[str, str]]:
    """Parse 'If P, focus on X. If Q, focus on Y.' into (condition, object) pairs."""
    task_l = (task or "").lower()
    out: list[tuple[str, str]] = []
    for m in _CONDITIONAL_FOCUS_CLAUSE_RE.finditer(task_l):
        cond = (m.group(1) or "").strip()
        obj = (m.group(2) or "").strip()
        obj = re.split(r"\s+\bif\b", obj, maxsplit=1)[0].strip()
        obj = re.sub(r"^(?:the|a|an)\s+", "", obj)
        if cond and obj:
            out.append((cond, obj))
    return out


def _mutually_exclusive_answer_boxes(task: str) -> list[str]:
    """Colored/answer boxes from ≥2 mutually exclusive if-clauses."""
    boxes: list[str] = []
    for _cond, obj in _conditional_focus_clauses(task):
        if _answer_box_destination(obj) and obj not in boxes:
            boxes.append(obj)
    return boxes if len(boxes) >= 2 else []


def _ordered_focus_instruction_phrases(task: str) -> list[str]:
    """Non-box, non-conditional 'focus on X' phrases in task order."""
    t = (task or "").strip()
    if not t:
        return []
    stripped = re.sub(
        r"\bif\b[^.]{0,160}?,\s*focus on (?:the |a |an )?[^.,;]+",
        " ",
        t,
        flags=re.I,
    )
    stripped = re.sub(
        r"when you are (?:completely )?done,?\s*focus on (?:the |a |an )?[^.,;]+",
        " ",
        stripped,
        flags=re.I,
    )
    out: list[str] = []
    for m in _INSTRUCTED_FOCUS_RE.finditer(stripped):
        obj = re.sub(r"\s+", " ", (m.group(1) or "").strip().lower())
        obj = re.sub(r"^(?:the|a|an)\s+", "", obj)
        obj = re.split(r"\s+(?:next|then|after|and then)\b", obj, maxsplit=1)[0].strip()
        if not obj or _answer_box_destination(obj):
            continue
        if obj not in out:
            out.append(obj)
    return out


def _focus_matches_instructed_phrase(obj: str, phrase: str) -> bool:
    obj_l = re.sub(r"^(?:the|a|an)\s+", "", (obj or "").strip().lower())
    phrase_l = re.sub(r"^(?:the|a|an)\s+", "", (phrase or "").strip().lower())
    if not obj_l or not phrase_l:
        return False
    if obj_l == phrase_l or phrase_l in obj_l or obj_l in phrase_l:
        return True
    return bool(_action_content_tokens(obj_l) & _action_content_tokens(phrase_l))


def _instructed_focus_is_generic(phrase: str) -> bool:
    """True for category-only instructions (substance / animal / thing), not named loads."""
    toks = _action_content_tokens(phrase) - {"the", "a", "an"}
    if not toks:
        return True
    generic = _GENERIC_FOCUS_ENTITY_WORDS | {"thing", "things"}
    return toks <= generic


def _current_instructed_focus_phrase(task: str, action_history=None) -> str:
    phrases = _ordered_focus_instruction_phrases(task)
    if not phrases:
        return ""
    hist = [(a or "").strip().lower() for a in (action_history or [])]
    done = 0
    for phrase in phrases:
        satisfied = False
        for act in hist:
            if not act.startswith("focus on "):
                continue
            prev = act[len("focus on "):].strip()
            if _focus_matches_instructed_phrase(prev, phrase):
                satisfied = True
                break
        if not satisfied:
            break
        done += 1
    if done >= len(phrases):
        return ""
    return phrases[done]


def _focus_is_organism_pot_contents(obj: str, task: str) -> bool:
    """True for 'substance in flower pot' — water, not the plant/seed."""
    obj_l = (obj or "").strip().lower()
    m = re.search(r"\bsubstance in (.+)$", obj_l)
    if not m:
        return False
    container = m.group(1)
    if not re.search(r"\b(?:flower\s*pot|self[-\s]?watering)\b", container):
        return False
    task_l = (task or "").lower()
    if re.search(r"\b(?:boil|melt|freeze|chemistry|mix|pour|dissolve)\b", task_l):
        return False
    return bool(re.search(
        r"\b(?:plant|seed|grow|life\s*stages?|living|organism)\b", task_l,
    ))


def _life_stage_too_late(obj: str, task: str, action_history=None) -> bool:
    """Block focusing a dead/adult plant when the task asks earliest→latest."""
    task_l = (task or "").lower()
    if not re.search(r"\bearliest to latest\b", task_l):
        return False
    obj_l = (obj or "").strip().lower()
    if not re.search(r"\b(?:dead|wilted)\b", obj_l):
        return False
    hist = [(a or "").strip().lower() for a in (action_history or [])]
    early = re.compile(r"\b(?:seed|egg|sprout|seedling|baby|juvenile)\b", re.I)
    for act in hist:
        if act.startswith("focus on ") and early.search(act):
            return False
    return True


def _sequenced_focus_blocked(obj: str, task: str, action_history=None) -> bool:
    """True when focusing `obj` skips an earlier instructed focus stage."""
    if _focus_is_organism_pot_contents(obj, task):
        return True
    if _life_stage_too_late(obj, task, action_history):
        return True
    phrases = _ordered_focus_instruction_phrases(task)
    current = _current_instructed_focus_phrase(task, action_history)
    # Instruments / circuit fixtures are never a substitute for the instructed
    # load or substance (focus on switch while the task says focus on buzzer).
    try:
        from core.pattern_library import _INSTRUMENT_FOCUS_RE
        inst = bool(_INSTRUMENT_FOCUS_RE.match(f"focus on {obj}"))
    except Exception:
        inst = bool(re.search(
            r"\b(?:thermometer|stopwatch|balance|switch|generator|"
            r"anode|cathode|battery|wire|light\s*bulb)\b",
            (obj or "").lower(),
        ))
    if inst:
        if current and _focus_matches_instructed_phrase(obj, current):
            return False
        if current or phrases:
            return True
    if not current:
        return False
    if len(phrases) >= 2:
        return not _focus_matches_instructed_phrase(obj, current)
    # Single specific instruction (electric buzzer, unknown substance): match it.
    if not _instructed_focus_is_generic(current):
        toks = _action_content_tokens(current)
        if 1 <= len(toks) <= 4 and not _COMPARISON_TASK_RE.search(current):
            return not _focus_matches_instructed_phrase(obj, current)
    return False


def _created_substance_product(task: str) -> str:
    """Product phrase from create-X / 'when you are done, focus on X'."""
    t = (task or "").strip()
    if not t:
        return ""
    tl = t.lower()
    create_mix = bool(re.search(r"\b(?:chemistry|mix|recipe|ingredient|paint|create)\b", tl))
    if create_mix:
        m = _DONE_FOCUS_PRODUCT_RE.search(tl)
        if m:
            return (m.group(1) or "").strip(" .").lower()
    m = _CREATE_SUBSTANCE_RE.search(t)
    if m:
        return next((g.strip().lower() for g in m.groups() if g), "")
    return ""


def _created_product_visible(task: str, ctx_text: str) -> bool:
    """True when the created product itself (full phrase) is in the observation."""
    product = _created_substance_product(task)
    if not product:
        return True
    ctx = (ctx_text or "").lower()
    if not ctx.strip():
        return False
    if re.search(rf"substance called {re.escape(product)}\b", ctx):
        return True
    if re.search(
        rf"containing (?:a |an )?(?:substance called )?{re.escape(product)}\b",
        ctx,
    ):
        return True
    return False


def _focus_is_create_ingredient_not_product(obj_phrase: str, task: str) -> bool:
    """True when focus names an ingredient token of a not-yet-complete create product."""
    product = _created_substance_product(task)
    if not product:
        return False
    obj_l = re.sub(r"^(?:the |a |an )", "", (obj_phrase or "").strip().lower())
    obj_l = re.sub(r"^substance called\s+", "", obj_l).strip()
    if not obj_l:
        return False
    if obj_l == product or obj_l.endswith(product) or product in obj_l:
        return False
    # "sugar" is inside "sugar water"; "peanut" inside "peanut butter … sandwich".
    if obj_l != product and obj_l in product and len(obj_l) >= 3:
        return True
    prod_tok = _action_content_tokens(product)
    obj_tok = _action_content_tokens(obj_l)
    if not obj_tok or not prod_tok:
        return False
    if obj_tok < prod_tok:
        return True
    if len(prod_tok) >= 2 and obj_tok <= prod_tok:
        return True
    return False


def _created_substance_focus_allowed(
    obj_phrase: str, task: str, ctx_text: str = "",
) -> bool:
    """Gate focus commits on create-then-focus chemistry tasks."""
    product = _created_substance_product(task)
    obj_l = re.sub(r"^(?:the |a |an )", "", (obj_phrase or "").strip().lower())
    obj_l = re.sub(r"^substance called\s+", "", obj_l).strip()
    task_l = (task or "").lower()
    if re.search(r"\bpaint\b", task_l) and re.search(r"\b(?:mix|create)\b", task_l):
        if obj_l == "paint" or _PRIMARY_PAINT_RE.match(obj_l):
            if not product or not (
                obj_l == product or obj_l in product or product in obj_l
            ):
                return False
    if not product:
        return True
    if _focus_is_create_ingredient_not_product(obj_phrase, task):
        return False
    if not _created_product_visible(task, ctx_text):
        return False
    # Require the product phrase itself — never an ingredient substring
    # ("sugar" ⊂ "sugar water", "paper" ⊂ "red paper").
    return bool(product == obj_l or product in obj_l)


def _is_answer_box_only_focus_task(task: str) -> bool:
    """True when instructed focus destinations are if/then answer boxes (not specimens)."""
    # Measure-then-box families still need instrument/substance focus first.
    if _task_is_threshold_measurement_task(task):
        return False
    if _mutually_exclusive_answer_boxes(task):
        task_l = (task or "").lower()
        if re.search(r"\b(?:dominant|recessive|allele|mendel|genetic)\b", task_l):
            return True
        clauses = _conditional_focus_clauses(task)
        if clauses and all(_answer_box_destination(obj) for _c, obj in clauses):
            if not re.search(r"\b(?:melting|boiling|freezing|thermometer|temperature)\b", task_l):
                return True
        return False
    clauses = _conditional_focus_clauses(task)
    if clauses and all(_answer_box_destination(obj) for _c, obj in clauses):
        task_l = (task or "").lower()
        if re.search(r"\b(?:dominant|recessive|allele|mendel|genetic)\b", task_l):
            return True
        if re.search(r"\b(?:melting|boiling|freezing|thermometer|temperature)\b", task_l):
            return False
    task_l = (task or "").lower()
    if re.search(r"\b(?:dominant|recessive|allele|mendel|genetic)\b", task_l):
        if clauses or re.search(r"\bfocus on (?:the )?(?:\w+\s+)?box\b", task_l):
            return True
    sem = _extract_comparison_semantics(task)
    return bool(sem and sem[0] == "genetics")


def _is_genetics_specimen_focus(obj_phrase: str, task: str) -> bool:
    """True when focusing a seed/plant would be a terminal wrong-answer on genetics."""
    task_l = (task or "").lower()
    genetics = bool(re.search(
        r"\b(?:dominant|recessive|allele|mendel|genetic)\b", task_l,
    ))
    if not genetics:
        return False
    if _answer_box_destination(obj_phrase):
        return False
    obj = (obj_phrase or "").strip().lower()
    if not obj:
        return False
    # Genetics if/then-box: any non-box focus is a terminal wrong-answer.
    return True


def _pick_justified_answer_box_focus(
    env_valid,
    task: str,
    action_history=None,
    ctx_text: str = "",
    recent=None,
    failed=None,
) -> str | None:
    """
    Visible if/then answer box justified by named phenotype or observation polarity.

    Family-level (mutually exclusive boxes), not a task-id table.
    """
    if not _is_answer_box_only_focus_task(task):
        return None
    recent_l = {str(x).strip().lower() for x in (recent or set()) if x}
    failed_l = {str(x).strip().lower() for x in (failed or set()) if x}
    cands: list[str] = []
    for va in env_valid or []:
        al = (va or "").strip().lower()
        if not al.startswith("focus on ") or al in recent_l or al in failed_l:
            continue
        obj = al[len("focus on "):].strip()
        if not _answer_box_destination(obj):
            continue
        if _conditional_answer_commit_allowed(al, task, action_history, ctx_text):
            cands.append(al)
    if not cands:
        return None
    cands.sort(key=lambda x: (len(x), x))
    return cands[0]


def _identify_delivery_box(obj_phrase: str, task: str) -> bool:
    """True when obj is the box a find/identify task says to move the thing into."""
    task_l = (task or "").lower()
    if not re.search(r"\b(?:find|identify|locate)\b", task_l):
        return False
    if not re.search(r"\bmove (?:it|them|the \w+) to\b", task_l):
        return False
    return bool(_answer_box_destination(obj_phrase))


def _inferred_if_clause_polarity(task: str) -> str | None:
    """
    Mendelian phenotype named in the instruction (not a task-id table).

    Classic pea mappings only. If-clauses that mention an unknown plant still
    count when the instruction also names a known phenotype (round/wrinkled
    seed, flower color, height). Tasks with no named phenotype stay unknown
    and must be resolved from observation, not pea priors.
    """
    t = (task or "").lower()
    # Named phenotypes beat the boilerplate "unknown plant" in if-clauses.
    # Classic Mendel wording always has both; skipping on unknown+plant made
    # every if/then box illegal and collapsed CL-on onto CL-off guessing.
    if re.search(r"\bround\s+seeds?(?:\s+shape)?\b", t):
        return "dominant"
    if re.search(r"\bwrinkled\s+seeds?\b", t):
        return "recessive"
    if re.search(r"\byellow\s+seeds?\b", t):
        return "dominant"
    if re.search(r"\bgreen\s+seeds?\b", t) and re.search(r"\b(?:color|colour)\b", t):
        return "recessive"
    if re.search(r"\btall\s+(?:plant\s+)?height\b", t):
        return "dominant"
    if re.search(r"\b(?:short|dwarf)\s+(?:plant\s+)?height\b", t):
        return "recessive"
    if re.search(r"\bpurple\s+flower", t):
        return "dominant"
    if re.search(r"\bwhite\s+flower", t):
        return "recessive"
    return None


def _conditional_answer_commit_allowed(
    action: str,
    task: str,
    action_history=None,
    ctx_text: str = "",
) -> bool:
    """
    True when focusing an if/then answer box is justified.

    Mutually exclusive boxes must not be guessed alphabetically. A named
    Mendelian phenotype selects the matching if-clause; otherwise deny.

    Do not require a prior specimen *focus*: in ScienceWorld that action is a
    terminal wrong-answer (-100). Evidence is the named phenotype, or
    look/examine observations — never the instruction text (it names both
    dominant and recessive).
    """
    boxes = _mutually_exclusive_answer_boxes(task)
    if not boxes:
        return True
    al = (action or "").strip().lower()
    if not al.startswith("focus on "):
        return True
    obj = re.sub(r"^(?:the|a|an)\s+", "", al[len("focus on "):].strip())
    if not _answer_box_destination(obj):
        return True
    polarity = _inferred_if_clause_polarity(task)
    if not polarity:
        ctx = (ctx_text or "").lower()
        hist = " ".join((a or "") for a in (action_history or [])[-8:])
        blob = f"{ctx} {hist}".lower()
        if re.search(r"\bdominant\b", blob) and not re.search(r"\brecessive\b", blob):
            polarity = "dominant"
        elif re.search(r"\brecessive\b", blob) and not re.search(r"\bdominant\b", blob):
            polarity = "recessive"
    if not polarity:
        return False
    for cond, clause_obj in _conditional_focus_clauses(task):
        if polarity not in cond:
            continue
        if not _answer_box_destination(clause_obj):
            continue
        return clause_obj in obj or obj in clause_obj
    return False


def _is_ghost_numbered_slot(obj_phrase: str) -> bool:
    """True for env-listed but unexecutable slots such as 'bee 0'."""
    obj = (obj_phrase or "").strip().lower()
    return bool(_GHOST_NUMBERED_SLOT_RE.match(obj))


def _measurement_tools_mentioned(text: str) -> set[str]:
    """Instrument names from task/observation/inventory (not a task-id list)."""
    blob = (text or "").lower()
    return {
        w for w in _PRE_FOCUS_PREP_TOOL_WORDS
        if re.search(rf"\b{re.escape(w)}\b", blob)
    }


def _measurement_task_tools(task: str) -> set[str]:
    """
    Instruments appropriate for this task.

    Prefer tools named in the instruction; otherwise infer from the measured
    quantity (timing/angle vs temperature), not from a task-id list.
    """
    mentioned = _measurement_tools_mentioned(task)
    if mentioned:
        return mentioned
    task_l = (task or "").lower()
    timing = frozenset({"stopwatch", "timer"})
    thermal = frozenset({"thermometer"})
    if re.search(
        r"\b(timing|stopwatch|timer|angle|friction|inclin|"
        r"elapsed|duration|how long|how steep|shallowest|steepest|"
        r"fastest|slowest)\b",
        task_l,
    ):
        return {w for w in _PRE_FOCUS_PREP_TOOL_WORDS if w in timing} or set(timing)
    if re.search(
        r"\b(temperature|melting|thermometer|celsius|fahrenheit|"
        r"degrees|boil|freeze|heat(?:ed|ing)?)\b",
        task_l,
    ):
        return {w for w in _PRE_FOCUS_PREP_TOOL_WORDS if w in thermal} or set(thermal)
    return set(_PRE_FOCUS_PREP_TOOL_WORDS)


def _use_action_instrument_head(action: str) -> str:
    """Object being used: 'use stopwatch on block' → 'stopwatch', not the patient."""
    al = (action or "").strip().lower()
    if not al.startswith("use "):
        return ""
    rest = al[4:].strip()
    head, _, _ = rest.partition(" on ")
    return head.strip()


def _use_action_patient(action: str) -> str:
    """Patient of a use action: 'use thermometer on apple' → 'apple'."""
    al = (action or "").strip().lower()
    if not al.startswith("use "):
        return ""
    rest = al[4:].strip()
    _, sep, patient = rest.partition(" on ")
    if not sep:
        return ""
    return patient.strip()


def _measurement_use_patient_is_distractor(patient: str, task: str) -> bool:
    """
    True when ``use <instrument> on X`` probes a non-informative target.

    Ambient/self targets, kitchen distractor foods, and answer-colored boxes are
    not melting-point samples unless the task itself names that food as the
    measured substance.
    """
    pat = (patient or "").strip().lower()
    if not pat:
        return False
    if re.search(r"\b(?:agent|air|self|ground|floor|sky|ceiling|wall)\b", pat):
        return True
    # Answer boxes are the *response*, never the thermometer patient.
    if _answer_box_destination(pat):
        return True
    tokens = _action_content_tokens(pat)
    if not tokens:
        return False
    task_tokens = extract_task_content_tokens(task) | _specific_task_tokens(task)
    if tokens & _DISTRACTOR_FOOD_OBJECTS:
        # Allow when the task explicitly names this food as the substance.
        if tokens & task_tokens:
            return False
        return True
    # Furniture / fixtures are not thermal samples for threshold tasks.
    if tokens & (
        _NON_PORTABLE_PICK_WORDS
        | _LOW_VALUE_EXPLORE_OBJECTS
        | _MEASUREMENT_PROBE_DISTRACTOR_WORDS
    ):
        if tokens & task_tokens:
            return False
        return True
    return False


def _measurement_use_patient_is_preferred(patient: str, task: str) -> bool:
    """Substance / substance-in-container patients for threshold measurement."""
    pat = (patient or "").strip().lower()
    if not pat or _measurement_use_patient_is_distractor(pat, task):
        return False
    if "substance" in pat:
        return True
    if _focus_matches_task_substance(pat, task):
        return True
    # Common heated vessels that hold the unknown/known sample.
    if re.search(
        r"\b(?:metal pot|ceramic cup|glass cup|tin cup|cup|pot|beaker|bowl)\b",
        pat,
    ):
        return True
    return False


def _measurement_use_is_informative(action: str, task: str) -> bool:
    """Instrument use that can unlock answer-box selection / count as measured."""
    tools = _measurement_task_tools(task)
    if not _use_action_is_instrument(action, tools):
        return False
    patient = _use_action_patient(action)
    if not patient:
        # Bare ``use thermometer`` does not unlock answer boxes — it often
        # reads ambient air and stalls mid-episode with a false "measured".
        return False
    if _measurement_use_patient_is_distractor(patient, task):
        return False
    # Prefer explicit substance / vessel patients; other non-distractors
    # (e.g. named metals) still count as informative once heated.
    return True


def _use_action_is_instrument(action: str, tools=None) -> bool:
    """True for `use <instrument>` / `use <instrument> on X`, not `use axe on stopwatch`."""
    tools = tools or _PRE_FOCUS_PREP_TOOL_WORDS
    head = _use_action_instrument_head(action)
    if not head:
        return False
    return any(re.search(rf"\b{re.escape(w)}\b", head) for w in tools)


def _object_matches_probe(obj: str, probe_name: str) -> bool:
    """True when a pick/move object refers to the same portable probe."""
    o = re.sub(r"\s+", " ", (obj or "").strip().lower())
    n = re.sub(r"\s+", " ", (probe_name or "").strip().lower())
    if not o or not n:
        return False
    if o == n or n in o or (len(o) > 3 and o in n):
        return True
    ot, nt = _action_content_tokens(o), _action_content_tokens(n)
    return bool(ot and nt and (ot <= nt or nt <= ot))


def _measurement_probe_is_distractor(obj_phrase: str, task: str) -> bool:
    """Furniture, circuit parts, vessels, and décor are not sliding/timing probes."""
    obj_l = (obj_phrase or "").strip().lower()
    if not obj_l:
        return True
    # Threshold measure-then-box: sample vessels / substance containers are the
    # payload to heat — not distractors (unlike inclined-plane / timing probes).
    if _task_is_threshold_measurement_task(task):
        if _answer_box_destination(obj_l):
            return True
        if re.search(
            r"\b(?:paint|closet|fountain|wood|shovel|drawing|sewer|fire\s*pit)\b", obj_l,
        ):
            return True
        if re.search(
            r"\b(?:metal pot|ceramic cup|glass cup|tin cup|cup|pot|beaker|"
            r"substance)\b",
            obj_l,
        ) and not re.search(r"\bfire\s*pit\b", obj_l):
            return False
        if re.search(r"\b(?:furnace|stove|oven|blast\s*furnace)\b", obj_l):
            return True
    tokens = _action_content_tokens(obj_l)
    if tokens & _PRE_FOCUS_PREP_TOOL_WORDS:
        return True
    if _is_forbidden_manipulation_target(obj_l) or _is_fixed_installation_target(obj_l):
        return True
    if tokens & (
        _PRE_FOCUS_PREP_CONTAINER_HINTS
        | _DISTRACTOR_FOOD_OBJECTS
        | _NON_PORTABLE_PICK_WORDS
        | _LOW_VALUE_EXPLORE_OBJECTS
        | _DISTRACTOR_VESSEL_CONTENT_WORDS
        | _MEASUREMENT_PROBE_DISTRACTOR_WORDS
    ):
        return True
    task_tokens = extract_task_content_tokens(task)
    if (tokens & _CIRCUIT_HINT_TOKENS) and not (task_tokens & _CIRCUIT_HINT_TOKENS):
        return True
    anchors = _focus_relevant_tokens(task) | _specific_task_tokens(task)
    if anchors and (tokens & anchors):
        portable = tokens & frozenset({
            "block", "brick", "ball", "marble", "weight", "cube", "cart",
            "box", "probe", "slider", "mass", "steel",
        })
        if not portable:
            return True
    return False


def _measurement_movable_probes(env_valid, anchors, task: str) -> list[str]:
    """Objects the env can place onto a comparison entity (observation-driven)."""
    names: list[str] = []
    anchors = anchors or set()
    for raw in env_valid or []:
        al = str(raw).strip().lower()
        m = re.match(r"move (.+?) to (.+)$", al)
        if not m:
            continue
        src, dest = m.group(1).strip(), m.group(2).strip()
        if _answer_box_destination(dest):
            continue
        if re.search(r"\bfire\s*pit\b", src) or is_heat_fixture_phrase(src):
            continue
        if not (_action_content_tokens(dest) & anchors):
            continue
        if _measurement_probe_is_distractor(src, task):
            continue
        if src not in names:
            names.append(src)
    return names


def _measurement_probe_on_comparison(
    ctx_text: str, probe_names: list[str], anchors: set[str],
) -> bool:
    """True when a valid probe already sits on a comparison entity in the observation."""
    ctx = (ctx_text or "").lower()
    if not ctx or not probe_names or not anchors:
        return False
    for name in probe_names:
        if len(name) < 3:
            continue
        esc_n = re.escape(name)
        for t in anchors:
            if len(t) < 4:
                continue
            esc_t = re.escape(t)
            if re.search(
                rf"{esc_n}.{{0,80}}(?:on (?:the )?{esc_t})"
                rf"|{esc_t}[^\n]{{0,80}}(?:contains|is:).{{0,80}}{esc_n}",
                ctx,
            ):
                return True
    return False


def _measurement_heat_started(action_history=None) -> bool:
    """True once a heat fixture was activated or a post-heat wait ran."""
    for act in action_history or []:
        al = (act or "").strip().lower()
        fam = action_verb_family(al)
        if fam == "wait":
            return True
        if fam == "activate" and is_heat_fixture_phrase(
            _action_manipulation_object(al) or al.replace("activate ", "", 1),
        ):
            return True
        if fam == "move":
            m = re.match(r"move .+? to (.+)$", al)
            if m and is_heat_fixture_phrase(m.group(1)):
                return True
    return False


def _score_threshold_measurement_setup(
    action: str,
    task: str,
    action_history=None,
    *,
    base: int | None = None,
) -> int:
    """
    Rank measure-then-box setup actions.

    Prefer placing/activating heat before ``use <thermometer>`` so CL and fast
    paths do not read ambient air and stall on a premature instrument use.
    """
    fam = action_verb_family(action)
    heated = _measurement_heat_started(action_history)
    tools = _measurement_task_tools(task)
    al = (action or "").strip().lower()
    sc = int(base) if base is not None else 40
    # ScienceWorld often emits ``wait1`` / ``wait2`` rather than bare ``wait``.
    if fam == "other" and re.match(r"^wait\d*$", al):
        fam = "wait"
    if fam == "use" and _use_action_is_instrument(al, tools):
        # Ambient / self / distractor / answer-box readings are not informative.
        patient = _use_action_patient(al)
        if re.search(r"\bon (?:agent|air|self)\b", al) or (
            patient and _measurement_use_patient_is_distractor(patient, task)
        ) or _answer_box_destination(patient or ""):
            sc = 1
        else:
            sc = 95 if heated else 35
            if patient and _measurement_use_patient_is_preferred(patient, task):
                sc += 20
            # Before heat: prefer placing/activating over premature instrument use.
            if not heated:
                sc = min(sc, 40)
    elif fam == "activate" and is_heat_fixture_phrase(
        _action_manipulation_object(al) or al.replace("activate ", "", 1),
    ):
        sc = 40 if heated else 92
    elif fam == "move":
        m = re.match(r"move (.+?) to (.+)$", al)
        if m and is_heat_fixture_phrase(m.group(2)):
            if _measurement_probe_is_distractor(m.group(1), task):
                sc = 1
            else:
                sc = 50 if heated else 88
        else:
            sc = 12
    elif fam in ("pick_up", "take") and any(
        re.search(rf"\b{re.escape(w)}\b", al) for w in tools
    ):
        sc = 70
    elif fam in ("pick_up", "take") and re.search(
        r"\b(?:metal pot|ceramic cup|glass cup|tin cup|cup|pot|beaker|"
        r"substance|tin|lead|chocolate|ice|water)\b",
        al,
    ) and not re.search(r"\bfire\s*pit\b", al):
        if _measurement_probe_is_distractor(
            _action_manipulation_object(al) or al, task,
        ):
            sc = 1
        else:
            # Acquire the sample / vessel so it can be moved onto a heat fixture.
            sc = 78 if not heated else 55
    elif fam in ("pick_up", "take") and re.search(r"\bfire\s*pit\b", al):
        sc = 1
    elif fam == "wait":
        # Prefer waiting only after an informative instrument use, not after
        # heat alone (otherwise the agent stalls at ~8 without measuring).
        if any(
            _measurement_use_is_informative(a, task)
            for a in (action_history or [])
        ):
            sc = 85
        else:
            sc = 25 if heated else 15
    return sc


def _measurement_tool_already_used(action_history, task: str = "") -> bool:
    """True after an informative measuring-instrument use (not distractor probes)."""
    tools = _measurement_task_tools(task) if task else _PRE_FOCUS_PREP_TOOL_WORDS
    for act in action_history or []:
        if not _use_action_is_instrument(act, tools):
            continue
        if task and not _measurement_use_is_informative(act, task):
            continue
        return True
    return False


def _task_is_experimental_identification(task: str) -> bool:
    """
    Identify-among-alternatives that requires an in-world measurement.

    Encyclopedic comparisons (longest lifespan, dominant allele) are excluded:
    they never mention a measuring instrument or experimental quantity.
    Named-substance / phase-change tasks (focus on chocolate, then melt) are
    excluded: they require focus first, then measurement.
    Threshold measure-then-box tasks (melting point + colored boxes) are included.
    """
    if _task_is_threshold_measurement_task(task):
        return True
    # Genetics / encyclopedic if-boxes are not instrument measurement.
    if _mutually_exclusive_answer_boxes(task) and re.search(
        r"\b(?:dominant|recessive|allele|genetic|mendel)\b", (task or "").lower(),
    ):
        return False
    if _task_needs_substance_preparation(task):
        return False
    if not (
        _task_expects_context_entity_focus(task)
        or _extract_comparison_semantics(task)
    ):
        return False
    task_l = (task or "").lower()
    if _measurement_tools_mentioned(task_l):
        return True
    return bool(
        re.search(
            r"\b(timing|stopwatch|angle|friction|inclin|"
            r"elapsed|duration|how long|how steep)\b",
            task_l,
        )
    )


def _task_needs_measurement_before_focus(
    task: str,
    action_history=None,
    ctx_text: str = "",
    inventory: str = "",
) -> bool:
    """
    Prefer instrument setup before committing to an answer focus.

    For threshold measure-then-box tasks this is True until the instrument is
    used (drives thermometer pick/use / heat setup). Callers that gate *all*
    focus must still allow substance/instrument focus and only ban answer
    boxes via ``_answer_selection_pending`` / ``_is_answer_selection_action``.
    """
    if _task_is_threshold_measurement_task(task):
        return _answer_selection_pending(task, action_history)
    if not _task_is_experimental_identification(task):
        return False
    if _task_focus_already_satisfied(task, action_history, ctx_text):
        return False
    if _measurement_tool_already_used(action_history, task):
        return False
    return True


def _focus_blocked_by_measurement_gate(
    action: str,
    task: str,
    action_history=None,
    ctx_text: str = "",
    inventory: str = "",
) -> bool:
    """
    True when a focus action must wait for measurement / evidence.

    Threshold tasks only block answer-box focuses; mutually exclusive if-boxes
    are blocked until the matching clause is justified; other experimental ID
    tasks block any focus until the instrument has been used.
    """
    if not (action or "").strip().lower().startswith("focus on"):
        return False
    if _answer_box_action_disallowed(action, task, action_history, ctx_text):
        return True
    obj = (action or "").strip().lower()
    if obj.startswith("focus on "):
        obj = obj[len("focus on "):].strip()
        if _sequenced_focus_blocked(obj, task, action_history):
            return True
    if _task_is_threshold_measurement_task(task):
        return False
    if not _task_needs_measurement_before_focus(
        task, action_history, ctx_text, inventory,
    ):
        return False
    # Experimental-ID still needs the instrument in hand, but focusing the
    # thermometer/balance itself is setup — not a premature answer commit.
    if re.search(r"\b(?:thermometer|balance|stopwatch)\b", obj):
        return False
    return True


def _is_measurement_setup_action(
    action: str,
    task: str,
    ctx_text: str = "",
    action_history=None,
    inventory: str = "",
    env_valid=None,
) -> bool:
    """Pick/use instrument, place a probe on the comparison entity, or wait after timing."""
    if not _task_needs_measurement_before_focus(
        task, action_history, ctx_text, inventory,
    ):
        return False
    al = (action or "").strip().lower()
    if not al:
        return False
    family = action_verb_family(action)
    tools = _measurement_task_tools(task)
    anchors = _focus_relevant_tokens(task) | _specific_task_tokens(task)
    obj = _action_manipulation_object(action)

    if family == "use":
        if not _use_action_is_instrument(al, tools):
            return False
        # Ambient / self / distractor / answer-box readings are never setup.
        patient = _use_action_patient(al)
        if re.search(r"\bon (?:agent|air|self)\b", al):
            return False
        if patient and (
            _measurement_use_patient_is_distractor(patient, task)
            or _answer_box_destination(patient)
        ):
            return False
        return True
    if family in ("pick_up", "take") and any(
        re.search(rf"\b{re.escape(w)}\b", al) for w in tools
    ):
        return True
    # Thermal threshold tasks need heat before a meaningful thermometer reading.
    if (
        _task_is_threshold_measurement_task(task)
        and family == "activate"
        and is_heat_fixture_phrase(
            _action_manipulation_object(action) or al.replace("activate ", "", 1),
        )
    ):
        return True
    if family == "wait" or re.match(r"^wait\d*$", al):
        recent = [(a or "").strip().lower() for a in (action_history or [])[-6:]]
        return any(_use_action_is_instrument(a, tools) for a in recent) or (
            _task_is_threshold_measurement_task(task)
            and any(
                action_verb_family(a) == "activate"
                and is_heat_fixture_phrase(
                    _action_manipulation_object(a) or a.replace("activate ", "", 1),
                )
                for a in recent
            )
        )
    if family == "move":
        m = re.match(r"move (.+?) to (.+)$", al)
        if not m:
            return False
        src, dest = m.group(1).strip(), m.group(2).strip()
        # Substance/container onto a heat fixture is valid setup for melting-point tasks.
        if (
            _task_is_threshold_measurement_task(task)
            and is_heat_fixture_phrase(dest)
            and not _answer_box_destination(dest)
        ):
            return not _measurement_probe_is_distractor(src, task)
        if not (_action_content_tokens(dest) & anchors):
            return False
        return not _measurement_probe_is_distractor(src, task)
    if family in ("pick_up", "take"):
        obj_l = (obj or al or "").strip().lower()
        if re.search(r"\bfire\s*pit\b", obj_l) or is_heat_fixture_phrase(obj_l):
            return False
        obj_tokens = _action_content_tokens(obj)
        if obj_tokens & tools:
            return True
        if _measurement_probe_is_distractor(obj, task):
            return False
        # Do not pick the answer fixture itself (the thing we will later focus on).
        if anchors and (obj_tokens & anchors) and not (obj_tokens - anchors - _FOCUS_VAGUE_OBJECT_WORDS):
            return False
        # Threshold tasks: pick sample vessels so they can be heated.
        if _task_is_threshold_measurement_task(task) and re.search(
            r"\b(?:metal pot|ceramic cup|glass cup|tin cup|cup|pot|beaker|"
            r"substance)\b",
            al,
        ):
            if _measurement_probe_is_distractor(obj, task):
                return False
            return True
        probes = _measurement_movable_probes(env_valid, anchors, task)
        if probes:
            return any(_object_matches_probe(obj, n) for n in probes)
        if env_valid is not None:
            return False
        return bool(obj)
    return False


def _comparison_entity_priority(
    entity_phrase: str, attribute: str, direction: str, task: str,
) -> int:
    """Score how well an entity fits a comparison axis (higher = better candidate)."""
    if attribute == "genetics":
        # Mendelian commits are answer boxes, not organism ranks.
        return 0
    entity = _entity_core_phrase(entity_phrase).lower()
    tokens = set(re.findall(r"[a-z]+", entity))
    rank_map = _COMPARISON_ENTITY_RANKS.get(attribute, _COMPARISON_ENTITY_RANKS["lifespan"])
    score = 0
    for hint, val in rank_map.items():
        if hint in entity or hint in tokens:
            score = max(score, val)
    if score == 0:
        # Inventory fruit / furniture must not win comparison focus.
        try:
            if _focus_target_is_incidental_fixture(entity_phrase, task):
                return 0
        except Exception:
            pass
        # Unlisted organisms only mid-prior on lifespan/size. Tools / inventory
        # / fixtures must not inherit that prior (focus on axe → -100).
        if attribute in ("lifespan", "size"):
            if tokens & _NON_ORGANISM_FOCUS_WORDS:
                return 0
            if _ORGANISM_HINT_RE.search(entity):
                return 50
            return 0
        return 0
    if direction == "min":
        score = 100 - score
    task_l = (task or "").lower()
    if re.search(r"\banimal", task_l):
        if tokens & {"seed", "plant", "bulb", "flower", "tree", "soil"}:
            score -= 45
        # Eggs are a life-stage, not a lifespan. Do not let
        # "egg giant tortoise" invert a min-task into a long-lived win.
        if "egg" in entity and attribute in ("lifespan", "general") and direction != "min":
            egg_bonus = max(
                (val - 12 for hint, val in rank_map.items() if hint in entity),
                default=0,
            )
            score = max(score, egg_bonus) if egg_bonus else score - 12
    if re.search(r"\bplant", task_l) and tokens & {
        "animal", "beaver", "ant", "bear", "wolf", "mouse", "chipmunk",
    }:
        score -= 45
    return score


def _focus_action_comparison_priority(action: str, task: str) -> int:
    """Comparison-axis priority for a focus action (0 when not a comparison task)."""
    comp = _extract_comparison_semantics(task)
    if not comp:
        return 0
    al = (action or "").strip().lower()
    if not al.startswith("focus on "):
        return 0
    obj = al[len("focus on ") :].strip()
    attr, direction = comp
    return _comparison_entity_priority(obj, attr, direction, task)


def _extract_focus_instruction_tokens(task: str) -> set[str]:
    """Tokens from explicit 'focus on a/an/the X' phrasing in the task."""
    task_l = (task or "").lower()
    tokens: set[str] = set()
    for m in re.finditer(
        r"\bfocus\s+on\s+(?:a|an|the)?\s*([a-z][a-z0-9 \-]+?)(?:\s+with|\s+that|\s+until|[.,]|$)",
        task_l,
    ):
        tokens |= _action_content_tokens(m.group(1).strip())
    if re.search(r"\bunknown\s+substance\b", task_l):
        tokens |= _labeled_entity_tokens("unknown substance")
        for m in re.finditer(r"unknown\s+substance\s+([A-Za-z0-9])\b", task or ""):
            tokens.add(m.group(1).lower())
    return tokens - _FOCUS_VAGUE_OBJECT_WORDS


def _focus_relevant_tokens(task: str) -> set[str]:
    """Tokens a pre-manipulation focus should align with (instruction + specific nouns)."""
    instructed = _extract_focus_instruction_tokens(task)
    specific = _specific_task_tokens(task)
    nouns = specific - _GENERIC_FOCUS_ENTITY_WORDS - SCIENCEWORLD_ROOM_NAMES
    if instructed:
        return instructed | nouns
    if nouns:
        return nouns
    return specific - SCIENCEWORLD_ROOM_NAMES


def _focus_object_aligns_with_task(
    obj_phrase: str,
    task: str,
    ctx_text: str = "",
    comparison_sem: tuple[str, str] | None = None,
) -> bool:
    """True when a focus target matches task instructions (not a color/distractor variant)."""
    comp = comparison_sem if comparison_sem is not None else _extract_comparison_semantics(task)
    if comp:
        return _comparison_focus_aligns(
            obj_phrase, task, ctx_text, comparison_sem=comp,
        )
    obj_l = (obj_phrase or "").strip().lower()
    if _is_focus_on_room_or_location(obj_l):
        return False
    # Instructed if/then boxes are the genetics commit, not a distractor color.
    if _answer_box_destination(obj_l):
        boxes = _mutually_exclusive_answer_boxes(task)
        if boxes and any(b in obj_l or obj_l in b for b in boxes):
            return _conditional_answer_commit_allowed(
                f"focus on {obj_l}", task, None, ctx_text,
            )
    if "substance in" in obj_l:
        return _task_needs_substance_preparation(task)
    if "substance called" in obj_l:
        return _focus_matches_task_substance(obj_phrase, task, ctx_text)
    if re.search(r"\bunknown\s+substance\b", (task or "").lower()):
        if "unknown substance" in obj_l:
            return True
    obj_tokens = _labeled_entity_tokens(obj_l)
    targets = _task_target_tokens(task)
    if not targets:
        return True
    if _is_distractor_object_for_task(obj_tokens, targets, task):
        return False
    if _focus_is_create_ingredient_not_product(obj_l, task):
        return False
    if not _created_substance_focus_allowed(obj_l, task, ctx_text):
        return False
    if _identify_delivery_box(obj_l, task):
        return False
    if _allowed_prep_vessel_focus(obj_l, task, ctx_text):
        return True
    if _allowed_life_grow_manipulation(obj_l, task, ctx_text):
        return True
    try:
        from core.scienceworld_policy import _focus_category_matches
        cat = _focus_category_matches(obj_l, task)
    except Exception:
        cat = None
    if cat is False:
        return False
    if cat is True:
        return True
    # Sub-component focus (anode/cathode) is not the task focus milestone.
    if obj_tokens & {"anode", "cathode"} and not (targets & {"anode", "cathode"}):
        return False
    focus_rel = _focus_relevant_tokens(task)
    instructed = _extract_focus_instruction_tokens(task)
    if instructed:
        core = instructed - _FOCUS_VAGUE_OBJECT_WORDS
        if core and (obj_tokens & core):
            return True
    if focus_rel:
        if not (obj_tokens & focus_rel):
            return False
        instructed = _extract_focus_instruction_tokens(task)
        if instructed:
            # Require distinctive nouns; allow dropping optional modifiers
            # (env often shortens "electric buzzer" → "buzzer").
            required = (
                instructed
                - _OPTIONAL_FOCUS_MODIFIERS
                - _FOCUS_VAGUE_OBJECT_WORDS
                - _AMBIGUOUS_OVERLAP_TOKENS
            )
            if required and not (obj_tokens & required):
                return False
    elif not (obj_tokens & targets):
        return False
    return True


def _comparison_focus_aligns(
    obj_phrase: str,
    task: str,
    ctx_text: str = "",
    comparison_sem: tuple[str, str] | None = None,
    action_history=None,
    current_score: int | None = None,
) -> bool:
    """True when a focus target fits a superlative/comparison entity task."""
    comp = comparison_sem if comparison_sem is not None else _active_comparison_semantics(
        task, action_history, current_score,
    )
    if not comp:
        comp = _extract_comparison_semantics(task)
    if not comp:
        return False
    obj_l = (obj_phrase or "").strip().lower()
    foc_tokens = _action_content_tokens(obj_l) - _FOCUS_VAGUE_OBJECT_WORDS
    if "substance in" in obj_l or "substance called" in obj_l:
        return (
            _task_needs_substance_preparation(task)
            and _focus_matches_task_substance(obj_phrase, task, ctx_text)
        )
    if _focus_target_is_incidental_fixture(obj_l, task):
        return False
    if _is_ghost_numbered_slot(obj_l):
        return False
    attr, direction = comp
    if attr == "genetics" or _is_answer_box_only_focus_task(task):
        # If/then boxes are the only legal focus. Check this before the
        # non-organism fixture filter so a colored box is not treated as décor.
        if not _answer_box_destination(obj_l):
            return False
        boxes = _mutually_exclusive_answer_boxes(task)
        if boxes:
            return _conditional_answer_commit_allowed(
                f"focus on {obj_l}", task, action_history, ctx_text,
            )
        return True
    if obj_l in _NON_ORGANISM_FOCUS_WORDS or (foc_tokens & _NON_ORGANISM_FOCUS_WORDS):
        return False
    if not foc_tokens:
        return False
    if _is_fixture_focus_without_substance(obj_l, _focus_relevant_tokens(task)):
        return False
    focus_rel = _focus_relevant_tokens(task)
    pri = _comparison_entity_priority(obj_phrase, attr, direction, task)
    if direction == "min" and attr in ("lifespan", "size"):
        # Shortest/smallest must not commit the long-lived / large entity
        # (tortoise on lifespan-shortest is a terminal -100).
        max_pri = _comparison_entity_priority(obj_phrase, attr, "max", task)
        if max_pri >= 70:
            return False
    # Color-only overlap with if-clause boxes must not validate fruit/furniture.
    distinctive = focus_rel - _AMBIGUOUS_OVERLAP_TOKENS - _FOCUS_VAGUE_OBJECT_WORDS
    if distinctive and (foc_tokens & distinctive) and pri >= 45:
        return True
    if focus_rel and (foc_tokens & focus_rel) and not (foc_tokens <= _AMBIGUOUS_OVERLAP_TOKENS):
        if pri >= 45 or attr not in ("lifespan", "size"):
            return True
    return pri >= 45


_INCIDENTAL_FOCUS_FIXTURES = frozenset({
    "bed", "mattress", "pillow", "sheet", "blanket", "sofa", "couch", "painting",
    "picture", "poster", "chair", "desk", "dresser", "lamp", "clock", "finger",
    "orange", "apple", "banana", "potato", "bread", "tomato", "onion",
    "drawing", "toilet", "bathtub", "sink", "sewer",
})
# Bare inventory fruit / décor. Organisms (bee, mouse, …) are NOT listed —
# they are legitimate comparison/identify targets.
_DISTRACTOR_BARE_FOCUS = frozenset({
    "orange", "apple", "banana", "potato", "bread", "tomato", "onion",
    "painting", "drawing", "picture", "poster",
})
_DISTRACTOR_FOOD_WORDS = frozenset({
    "orange", "apple", "banana", "potato", "bread", "tomato", "onion",
})


def _focus_target_is_incidental_fixture(obj_phrase: str, task: str) -> bool:
    """Room furniture / distractors that must not satisfy context-entity focus."""
    obj_l = (obj_phrase or "").strip().lower()
    if _answer_box_destination(obj_l):
        return False
    # Genetics if/then-box tasks: specimen focus is a -100 commit, not evidence.
    if _is_genetics_specimen_focus(obj_phrase, task):
        return True
    core = re.sub(r"^(?:the|a|an)\s+", "", obj_l).strip()
    task_l = (task or "").strip().lower()
    head = _primary_pick_object(core)
    # ``orange in bowl`` is still fruit even when the task names an orange *box*.
    if head in _DISTRACTOR_FOOD_WORDS and not re.search(
        r"\b(?:recipe|sandwich|cook|eat|food|fruit|ingredient)\b", task_l,
    ):
        if not re.search(rf"\b{re.escape(head)}\s+box\b", core):
            return True
    if core in _DISTRACTOR_BARE_FOCUS:
        named = bool(re.search(rf"\b{re.escape(core)}\b", task_l))
        if not named:
            return True
        if core in _DISTRACTOR_FOOD_WORDS and not re.search(
            r"\b(?:recipe|sandwich|cook|eat|food|fruit|ingredient)\b", task_l,
        ):
            return True
    foc = _action_content_tokens(obj_phrase) - _FOCUS_VAGUE_OBJECT_WORDS
    incidental = foc & (
        _INCIDENTAL_FOCUS_FIXTURES | _NON_PORTABLE_PICK_WORDS | _LOW_VALUE_EXPLORE_OBJECTS
    )
    if not incidental:
        return False
    task_l = (task or "").lower()
    relevant = (
        _focus_relevant_tokens(task)
        | _specific_task_tokens(task)
        | _extract_focus_instruction_tokens(task)
    ) - SCIENCEWORLD_ROOM_NAMES - _GENERIC_FOCUS_ENTITY_WORDS - _AMBIGUOUS_OVERLAP_TOKENS
    if foc & relevant:
        return False
    if re.search(r"\bunknown\s+substance\b", task_l):
        return True
    if re.search(
        r"\b(?:animal|plant|pea|seed|genetic|mendel|lifespan|trait|"
        r"conductivity|substance|find|identify|locate|non[- ]?living)\b",
        task_l,
    ):
        if _task_is_life_grow_family(task) or _allowed_life_grow_manipulation(
            obj_phrase, task,
        ):
            return False
        return True
    if _COMPARISON_TASK_RE.search(task_l):
        return True
    return False


def _is_fixture_focus_without_substance(foc_obj: str, focus_tokens: set[str]) -> bool:
    """Container/fixture focus when the task names a substance that is not in the phrase."""
    foc_l = (foc_obj or "").strip().lower()
    if "substance in" in foc_l:
        if re.search(r"substance in .*\b(?:flower\s*pot|self[-\s]?watering)\b", foc_l):
            return True
        return False
    if not focus_tokens:
        return False
    foc_tokens = _action_content_tokens(foc_obj)
    if foc_tokens & focus_tokens:
        return False
    fixtures = foc_tokens & (_NON_PORTABLE_PICK_WORDS | _TASK_SUBSTANCE_CONTAINER_WORDS)
    return bool(fixtures)


def _score_focus_candidate(
    foc_obj: str,
    task: str,
    ctx_text: str,
    planned_tokens: set[str],
    planned_visible: bool,
    comparison_sem: tuple[str, str] | None,
) -> int:
    """Unified ranking for visible focus substitutes (comparison and substance tasks)."""
    foc_tokens = _action_content_tokens(foc_obj)
    focus_relevant = _focus_relevant_tokens(task)
    specific = _specific_task_tokens(task)
    anchor = _context_task_anchor_tokens(task, ctx_text)
    score = 0

    if comparison_sem:
        attr, direction = comparison_sem
        score += _comparison_entity_priority(foc_obj, attr, direction, task) * 3
        if foc_tokens & _ENTITY_STAGE_WORDS:
            pri = _comparison_entity_priority(foc_obj, attr, direction, task)
            if pri < 55:
                score -= 60
        if planned_visible:
            score += len(foc_tokens & planned_tokens) * 8
        elif planned_tokens:
            overlap = len(foc_tokens & planned_tokens) * 2
            pri = _comparison_entity_priority(foc_obj, attr, direction, task)
            if not planned_visible and pri < 55:
                overlap = 0
            score += overlap
        if attr == "genetics":
            act = f"focus on {foc_obj}"
            if _answer_box_destination(foc_obj) and _conditional_answer_commit_allowed(
                act, task, None, ctx_text,
            ):
                score += 90
            elif _answer_box_destination(foc_obj):
                score -= 120
    else:
        rel = foc_tokens & focus_relevant
        spec = foc_tokens & specific
        score += len(rel) * 25 + len(spec) * 18 + len(foc_tokens & anchor) * 12
        if focus_relevant and rel >= focus_relevant:
            score += 35
        if "substance called" in foc_obj:
            score += 14
        if "unknown substance" in foc_obj:
            score += 40
        if planned_visible:
            score += len(foc_tokens & planned_tokens) * 10
        elif planned_tokens:
            score += len(foc_tokens & planned_tokens) * 3
        instructed = _extract_focus_instruction_tokens(task)
        try:
            from core.scienceworld_policy import _focus_category_matches
            cat = _focus_category_matches(foc_obj, task)
            if cat is True:
                score += 45
            elif cat is False:
                score -= 80
        except Exception:
            pass
        if _identify_delivery_box(foc_obj, task):
            score -= 120
        if instructed:
            # Reward hitting any instructed focus target; do not require all of them.
            core = instructed - _FOCUS_VAGUE_OBJECT_WORDS
            hit = foc_tokens & core
            if core and not hit:
                score -= 40
            else:
                score += len(hit) * 12
        if _is_fixture_focus_without_substance(foc_obj, focus_relevant or specific):
            score -= 90
        if foc_tokens & _NON_PORTABLE_PICK_WORDS and not (foc_tokens & (focus_relevant | specific)):
            score -= 70
        if foc_tokens & _LOW_VALUE_EXPLORE_OBJECTS:
            score -= 45

    score -= foc_obj.count(" in ") * 4
    if _action_phrase_likely_ambiguous(foc_obj):
        score -= 20
    return score


def _focus_has_task_overlap(
    foc_obj: str, task: str, comparison_sem: tuple[str, str] | None = None,
    ctx_text: str = "",
) -> bool:
    """True when a focus target aligns with task instructions."""
    return _focus_object_aligns_with_task(foc_obj, task, ctx_text, comparison_sem)


def _is_pre_focus_destructive_action(
    action: str,
    task: str,
    action_history=None,
    ctx_text: str = "",
) -> bool:
    """Block eat/mix/drink on task-relevant entities before focus is satisfied."""
    if not _requires_focus_before_manipulation(task, action_history, ctx_text):
        return False
    al = (action or "").strip().lower()
    family = action_verb_family(action)
    if family not in _PRE_FOCUS_DESTRUCTIVE_FAMILIES:
        return False
    if family == "mix":
        target = al.split(" ", 1)[-1].strip() if " " in al else ""
        if target in SCIENCEWORLD_ROOM_NAMES:
            return True
    va_tokens = _action_content_tokens(action)
    anchor = _focus_relevant_tokens(task) or _specific_task_tokens(task)
    return bool(va_tokens & anchor)


def _pre_focus_substitute_allowed(
    action: str, task: str, action_history=None, ctx_text: str = "",
) -> bool:
    """During pre-focus phase, only substitutes that advance the focus subgoal."""
    if not _requires_focus_before_manipulation(task, action_history, ctx_text):
        return True
    family = action_verb_family(action)
    if family == "focus":
        return True
    if family == "open":
        if _task_needs_substance_preparation(task) and _task_target_missing(task, ctx_text):
            return _is_pre_focus_preparation_action(
                action, task, ctx_text, action_history,
            )
        return True
    if family == "look_in" and _task_target_missing(task, ctx_text):
        return True
    al = (action or "").strip().lower()
    if _is_pre_focus_preparation_action(action, task, ctx_text, action_history):
        return True
    if action_verb_family(action) == "activate" and _is_premature_heat_before_focus(
        action, task, action_history, ctx_text,
    ):
        return False
    if al == "look around" and _focus_subgoal_pending(task, action_history, ctx_text):
        if _current_room_in_task_area(task, ctx_text) or _room_has_substance_container_hints(
            ctx_text, task,
        ):
            return True
    if al.startswith(("go to ", "open door to ")) and _task_target_missing(task, ctx_text):
        return _focus_substitute_nav_allowed(
            action, task, ctx_text, action_history,
        )
    return False


def _focus_subgoal_pending(
    task: str, action_history=None, ctx_text: str = "",
    current_score: int | None = None,
) -> bool:
    """True while task requires an initial focus that has not yet succeeded."""
    return _requires_focus_before_manipulation(
        task, action_history, ctx_text, current_score,
    ) and not _task_focus_already_satisfied(
        task, action_history, ctx_text, current_score,
    )


def _observation_substance_focus_plans(ctx_text: str, task: str) -> list[str]:
    """Infer focus phrases from container/substance wording in the observation."""
    if not _task_needs_substance_preparation(task):
        return []
    ctx = (ctx_text or "").lower()
    targets = _task_target_tokens(task)
    if not targets or not ctx.strip():
        return []
    seen: set[str] = set()
    out: list[str] = []

    def _add(phrase: str) -> None:
        phrase = (phrase or "").strip().lower()
        if phrase and phrase not in seen:
            seen.add(phrase)
            out.append(phrase)

    for m in re.finditer(r"in the ([a-z][a-z0-9 ]+?) is:([^.\n]+)", ctx):
        container = m.group(1).strip()
        inner = m.group(2)
        for sm in re.finditer(r"(?:a |an )?substance called ([a-z][a-z0-9 ]+)", inner):
            subst = sm.group(1).strip()
            if _action_content_tokens(subst) & targets:
                _add(f"focus on substance in {container}")
                _add(f"focus on substance called {subst}")

    for m in re.finditer(
        r"([a-z][a-z0-9 ]+?) \(containing (?:a |an )?substance called ([a-z][a-z0-9 ]+)",
        ctx,
    ):
        container, subst = m.group(1).strip(), m.group(2).strip()
        if _action_content_tokens(subst) & targets:
            _add(f"focus on substance in {container}")
            _add(f"focus on substance called {subst}")

    for m in re.finditer(
        r"\b(metal pot|glass cup|ceramic cup|tin cup)\b(?: \(containing[^)]*\))?",
        ctx,
    ):
        container = m.group(1).strip()
        _add(f"focus on substance in {container}")

    return out


def _task_entity_bigram_focus_plans(task: str) -> list[str]:
    """Adjacent task-content pairs that may name env entity types (e.g. seed plant)."""
    tokens: list[str] = []
    seen: set[str] = set()
    for w in re.findall(r"[a-z0-9]+", (task or "").lower()):
        if w in _TASK_STOPWORDS or len(w) <= 2 or w in _GENERIC_FOCUS_ENTITY_WORDS:
            continue
        if w in seen:
            continue
        seen.add(w)
        tokens.append(w)
    out: list[str] = []
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if a != b:
            out.append(f"focus on {a} {b}")
            out.append(f"focus on {b} {a}")
    return out


def _synthesized_focus_plans(task: str, planned: str = "", ctx_text: str = "") -> list[str]:
    """Generate focus plan phrases from task wording and observation for fuzzy env matching."""
    seen: set[str] = set()
    out: list[str] = []
    p = (planned or "").strip().lower()

    def _add(phrase: str) -> None:
        phrase = (phrase or "").strip().lower()
        if phrase and phrase not in seen:
            seen.add(phrase)
            out.append(phrase)

    if p.startswith("focus on"):
        _add(p)
    focus_rel = _focus_relevant_tokens(task)
    instructed = _extract_focus_instruction_tokens(task)
    nouns = sorted(_specific_task_tokens(task) - _GENERIC_FOCUS_ENTITY_WORDS)
    if instructed:
        _add(f"focus on {' '.join(sorted(instructed))}")
    if len(nouns) >= 2:
        _add(f"focus on {' '.join(nouns[:2])}")
    for noun in nouns[:4]:
        _add(f"focus on {noun}")
    for tok in sorted(focus_rel)[:5]:
        _add(f"focus on {tok}")
        _add(f"focus on substance called {tok}")
    for phrase in _task_entity_bigram_focus_plans(task):
        _add(phrase)
    for phrase in _observation_substance_focus_plans(ctx_text, task):
        _add(phrase)
    return out


def _synthesized_focus_plans_with_history(
    task: str, planned: str = "", ctx_text: str = "", action_history=None,
) -> list[str]:
    """Like _synthesized_focus_plans but merges history-inferred container focus."""
    base = _synthesized_focus_plans(task, planned, ctx_text)
    seen = {p.strip().lower() for p in base}
    for phrase in _history_substance_container_focus_plans(action_history, ctx_text, task):
        pl = phrase.strip().lower()
        if pl and pl not in seen:
            seen.add(pl)
            base.append(phrase)
    return base


def _pick_focus_from_fuzzy_plans(
    task: str,
    env_valid: set,
    ctx_text: str,
    recent: set,
    failed_actions: set | None,
    planned: str,
    env_valid_for_malformed: set | None = None,
    action_history=None,
) -> str | None:
    """Try fuzzy focus matching across synthesized and planned phrases."""
    failed = failed_actions or set()
    env_pool = env_valid_for_malformed or env_valid
    comparison_sem = _extract_comparison_semantics(task)
    planned_obj = (planned or "").lower().replace("focus on", "", 1).strip()
    env_lower = {(va or "").strip().lower(): va for va in env_valid}
    plans = (
        _synthesized_focus_plans_with_history(task, planned, ctx_text, action_history)
        if action_history
        else _synthesized_focus_plans(task, planned, ctx_text)
    )
    best_cand: str | None = None
    best_pri = -1
    for plan in plans:
        plan_l = plan.strip().lower()
        exact = env_lower.get(plan_l)
        if exact and plan_l not in recent and plan_l not in failed:
            foc_obj = plan_l[len("focus on ") :].strip()
            if not _is_malformed_focus_object(foc_obj, planned_obj, env_pool, task):
                if _focus_has_task_overlap(foc_obj, task, comparison_sem, ctx_text):
                    if allow_focus_action(exact, task, ctx_text, ""):
                        if comparison_sem:
                            pri = _focus_action_comparison_priority(exact, task)
                            if pri > best_pri:
                                best_pri = pri
                                best_cand = exact
                        else:
                            return exact
        for cand in _fuzzy_focus_actions(plan, env_valid, ctx_text, task):
            if cand in recent or cand in failed:
                continue
            foc_obj = cand[len("focus on ") :].strip()
            if _is_malformed_focus_object(foc_obj, planned_obj, env_pool, task):
                continue
            if not _focus_has_task_overlap(foc_obj, task, comparison_sem, ctx_text):
                continue
            if comparison_sem:
                pri = _focus_action_comparison_priority(cand, task)
                if pri > best_pri:
                    best_pri = pri
                    best_cand = cand
            else:
                return cand
    return best_cand


def _best_visible_focus_candidate(
    env_valid: set,
    ctx_text: str,
    task: str,
    recent: set,
    failed_actions: set | None = None,
    planned: str = "",
    action_history=None,
) -> str | None:
    """Top-ranked grounded focus action, or None."""
    candidates = _collect_visible_focus_candidates(
        env_valid, ctx_text, task, recent, failed_actions, planned=planned,
        action_history=action_history,
    )
    return candidates[0] if candidates else None


def _has_duplicate_adjacent_tokens(phrase: str) -> bool:
    words = (phrase or "").lower().split()
    return any(len(words) > 1 and words[i] == words[i + 1] for i in range(len(words) - 1))


def _entity_core_phrase(obj_phrase: str) -> str:
    """Strip ScienceWorld 'substance called' wrapper to get the entity name."""
    obj = (obj_phrase or "").strip().lower()
    m = re.match(r"^(?:a |an )?substance called\s+(.+)$", obj)
    if m:
        return m.group(1).strip()
    return obj


def _entity_visible_in_context(obj_phrase: str, ctx_text: str) -> bool:
    """Match entity phrases against observation text (incl. substance-called vs bare noun)."""
    obj = (obj_phrase or "").strip().lower()
    if not obj:
        return False
    if _ctx_contains_phrase(ctx_text, obj):
        return True
    core = _entity_core_phrase(obj)
    if core != obj and _ctx_contains_phrase(ctx_text, core):
        return True
    ctx = (ctx_text or "").lower()
    if core:
        for article in ("a", "an"):
            if re.search(rf"\b{article} {re.escape(core)}\b", ctx):
                return True
    return False


def _is_malformed_focus_object(
    foc_obj: str,
    planned_obj: str = "",
    env_valid: set | None = None,
    task: str = "",
) -> bool:
    """Reject duplicate tokens (baby baby) or truncated multi-word targets (baby ant -> ant)."""
    foc_obj = (foc_obj or "").strip().lower()
    if not foc_obj:
        return True
    # Env may list these, but executing them is ``No known action``.
    if _is_ghost_numbered_slot(foc_obj) or _has_duplicate_adjacent_tokens(foc_obj):
        return True
    # ScienceWorld env strings may repeat tokens (e.g. "baby baby brown bear"); trust env_valid.
    if env_valid:
        for va in env_valid:
            al = (va or "").strip().lower()
            if al.startswith("focus on ") and al[len("focus on ") :].strip() == foc_obj:
                return False
    planned_obj = (planned_obj or "").strip().lower()
    if planned_obj.startswith("focus on "):
        planned_obj = planned_obj[len("focus on ") :].strip()
    planned_tokens = _action_content_tokens(planned_obj)
    foc_tokens = _action_content_tokens(foc_obj)
    if len(planned_tokens) >= 2 and foc_tokens and foc_tokens < planned_tokens:
        if _task_expects_context_entity_focus(task) or _extract_comparison_semantics(task):
            if foc_tokens & (_ENTITY_STAGE_WORDS | planned_tokens):
                return False
        focus_rel = _focus_relevant_tokens(task) if task else set()
        instructed = _extract_focus_instruction_tokens(task) if task else set()
        anchor = instructed or focus_rel
        if anchor and (foc_tokens & anchor):
            return False
        return True
    return False


def _pick_solid_cold_storage_open(
    env_valid: set,
    recent: set,
    task: str,
    ctx_text: str,
    failed_actions: set | None = None,
) -> str | None:
    """Backward-compatible alias: cold-biased container open when appropriate."""
    if not _task_favors_cold_storage(task):
        return None
    return _pick_closed_container_for_prep(
        env_valid, recent, task, ctx_text, failed_actions,
    )


def _pick_hidden_substance_container_open(
    env_valid: set,
    recent: set,
    task: str,
    ctx_text: str,
    failed_actions: set | None = None,
) -> str | None:
    """Backward-compatible alias for in-room hidden container opening."""
    return _pick_closed_container_for_prep(
        env_valid, recent, task, ctx_text, failed_actions,
    )


def _substance_examined_pending_focus(
    task: str, action_history=None, ctx_text: str = "",
    current_score: int | None = None,
) -> bool:
    """True after look/examine confirmed task substance but focus not yet done."""
    if not _focus_subgoal_pending(task, action_history, ctx_text, current_score):
        return False
    targets = _specific_task_tokens(task) or _task_target_tokens(task)
    if not targets:
        return False
    for act in reversed([(a or "").strip().lower() for a in (action_history or [])[-12:]]):
        if not act.startswith(("look at ", "examine ")):
            continue
        obj = act.split(" ", 2)[-1].strip()
        if _look_confirms_task_substance(obj, task, ctx_text):
            return True
    return False


def _examined_focus_allowed(
    foc_action: str,
    foc_obj: str,
    task: str,
    ctx_text: str,
    action_history,
    examined: set[str],
) -> bool:
    """Allow container/substance focus after look/examine confirmed the target."""
    if allow_focus_action(foc_action, task, ctx_text, "", action_history):
        return True
    targets = _specific_task_tokens(task) or _task_target_tokens(task)
    if not (examined & targets):
        return False
    if "substance in" in foc_obj and _focus_matches_task_substance(foc_obj, task, ctx_text):
        return True
    if _action_content_tokens(foc_obj) & examined:
        if _false_task_substance_match(foc_obj, task, ctx_text):
            return False
        if "substance called" in foc_obj:
            return True
        if _task_substance_visible(task, ctx_text):
            return True
        return False
    return False


def _scan_env_valid_focus_for_examined(
    env_valid: set,
    examined: set[str],
    ctx_text: str,
    task: str,
    recent: set,
    failed: set,
    action_history=None,
    logger=None,
) -> str | None:
    """Pick any env-valid focus action matching examined task substance tokens."""
    if not examined or not env_valid:
        return None
    ranked: list[tuple[int, str]] = []
    for va in env_valid:
        al = (va or "").strip().lower()
        if not al.startswith("focus on ") or al in recent or al in failed:
            continue
        foc_obj = al[len("focus on ") :].strip()
        foc_tokens = _action_content_tokens(foc_obj)
        if not (foc_tokens & examined):
            if "substance in" not in foc_obj:
                continue
            if not _focus_matches_task_substance(foc_obj, task, ctx_text):
                continue
        if not _examined_focus_allowed(
            va, foc_obj, task, ctx_text, action_history, examined,
        ):
            continue
        score = 0
        if "substance in" in foc_obj:
            score += 40
        if foc_tokens & examined:
            score += 30
        ranked.append((score, al))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    if ranked:
        choice = ranked[0][1]
        if logger:
            logger.info(f"Focus substitute (examined env scan): {choice!r}")
        return choice
    return None


def _pick_focus_from_examined_substance(
    env_valid: set,
    recent: set,
    ctx_text: str,
    task: str,
    action_history=None,
    logger=None,
    failed_actions: set | None = None,
) -> str | None:
    """After look at/examine confirmed a task substance, promote to focus."""
    if _task_focus_already_satisfied(task, action_history, ctx_text):
        return None
    if not _focus_subgoal_pending(task, action_history, ctx_text):
        return None
    failed = failed_actions or set()
    targets = _specific_task_tokens(task) or _task_target_tokens(task)
    if not targets:
        return None
    examined: set[str] = set()
    for act in reversed([(a or "").strip().lower() for a in (action_history or [])[-10:]]):
        if not act.startswith(("look at ", "examine ")):
            continue
        obj = act.split(" ", 2)[-1].strip()
        if _look_confirms_task_substance(obj, task, ctx_text):
            for t in targets:
                examined.add(t)
            break
    if not examined:
        return None
    for t in sorted(examined, key=len, reverse=True):
        phrases: list[str] = []
        for phrase in _observation_substance_focus_plans(ctx_text, task):
            if t in phrase:
                phrases.append(phrase)
        phrases.extend([
            f"focus on substance called {t}",
            f"focus on {t}",
        ])
        seen_p: set[str] = set()
        ordered: list[str] = []
        for phrase in phrases:
            phrase = (phrase or "").strip().lower()
            if phrase and phrase not in seen_p:
                seen_p.add(phrase)
                ordered.append(phrase)
        for phrase in ordered:
            if phrase in env_valid and phrase not in recent and phrase not in failed:
                foc_obj = phrase[len("focus on ") :].strip() if phrase.startswith("focus on ") else phrase
                if _examined_focus_allowed(
                    phrase, foc_obj, task, ctx_text, action_history, examined,
                ):
                    if logger:
                        logger.info(f"Focus substitute (examined substance): {phrase!r}")
                    return phrase
    scanned = _scan_env_valid_focus_for_examined(
        env_valid, examined, ctx_text, task, recent, failed, action_history, logger,
    )
    if scanned and scanned not in recent:
        return scanned
    ranked = _collect_visible_focus_candidates(
        env_valid, ctx_text, task, recent, failed_actions,
        action_history=action_history,
    )
    for cand in ranked:
        cand_tokens = _action_content_tokens(cand.replace("focus on", "", 1))
        if cand_tokens & examined and cand not in recent:
            if logger:
                logger.info(f"Focus substitute (examined ranked): {cand!r}")
            return cand
    return None


def _collect_visible_focus_candidates(
    env_valid: set,
    ctx_text: str,
    task: str,
    recent: set,
    failed_actions: set | None = None,
    planned: str = "",
    action_history=None,
    current_score: int | None = None,
) -> list[str]:
    """Rank grounded focus actions; for comparison tasks try all visible entities."""
    if _task_focus_already_satisfied(task, action_history, ctx_text, current_score):
        return []
    failed = failed_actions or set()
    planned_obj = (planned or "").lower().replace("focus on", "", 1).strip()
    planned_tokens = _action_content_tokens(planned_obj)
    comparison = bool(_COMPARISON_TASK_RE.search((task or "").lower()))
    comparison_sem = (
        _active_comparison_semantics(task, action_history, current_score)
        if comparison else None
    )
    planned_visible = bool(
        planned_obj and _entity_visible_in_context(planned_obj, ctx_text)
    )
    ranked: list[tuple[int, str]] = []
    focus_relevant = _focus_relevant_tokens(task)
    for va in env_valid:
        al = (va or "").strip().lower()
        if not al.startswith("focus on ") or va in recent or al in failed:
            continue
        foc_obj = al[len("focus on ") :].strip()
        if _is_focus_on_room_or_location(foc_obj):
            continue
        if _focus_target_is_incidental_fixture(foc_obj, task):
            continue
        if _is_genetics_specimen_focus(foc_obj, task):
            continue
        if not _created_substance_focus_allowed(foc_obj, task, ctx_text):
            continue
        if _identify_delivery_box(foc_obj, task):
            continue
        if _is_malformed_focus_object(foc_obj, planned_obj, env_valid, task):
            continue
        if not allow_focus_action(va, task, ctx_text, "", action_history):
            continue
        # Never rank answer-box focuses while the instrument is still unused.
        if _focus_blocked_by_measurement_gate(
            va, task, action_history, ctx_text,
        ):
            continue
        foc_tokens = _action_content_tokens(foc_obj)
        if foc_tokens & (_FORBIDDEN_MANIPULATION_TARGETS | _LOW_VALUE_EXPLORE_OBJECTS):
            continue
        if not comparison_sem and _is_fixture_focus_without_substance(
            foc_obj, focus_relevant,
        ):
            continue
        if not _focus_has_task_overlap(foc_obj, task, comparison_sem, ctx_text):
            continue
        score = _score_focus_candidate(
            foc_obj, task, ctx_text, planned_tokens, planned_visible, comparison_sem,
        )
        if _substance_confirmed_in_history(va, task, action_history):
            score += 85
        # Prefer the measured substance over the instrument once the instrument is held/focused.
        if _task_is_threshold_measurement_task(task):
            tools = _measurement_task_tools(task) | _PRE_FOCUS_PREP_TOOL_WORDS
            measured = _measurement_tool_already_used(action_history, task)
            thermo_focused = any(
                (a or "").strip().lower().startswith("focus on ")
                and (_action_content_tokens(a) & tools)
                for a in (action_history or [])
            )
            if measured:
                # After an informative reading, answer boxes win; instrument focus stalls.
                if _answer_box_destination(foc_obj):
                    score += 100
                elif foc_tokens & tools:
                    score -= 100
            elif thermo_focused and foc_tokens & tools:
                # Already focused the instrument — stop re-focusing; go measure.
                score -= 100
            elif foc_tokens & tools and not _task_substance_focus_done(
                task, action_history, ctx_text,
            ):
                score -= 35
            elif not _task_substance_focus_done(task, action_history, ctx_text):
                if _focus_matches_task_substance(foc_obj, task, ctx_text) or (
                    "substance" in foc_obj
                ):
                    score += 40
        if not comparison_sem and score <= 0:
            continue
        if comparison_sem and not _comparison_focus_aligns(
            foc_obj, task, ctx_text,
            comparison_sem=comparison_sem,
            action_history=action_history,
            current_score=current_score,
        ):
            continue
        ranked.append((score, va))
    ranked.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
    return [a for _, a in ranked]


def _pick_focus_substitute(
    planned: str,
    env_valid: set,
    recent: set,
    ctx_text: str,
    task: str,
    action_history=None,
    logger=None,
    failed_actions: set | None = None,
    last_gain_family: str = "",
    steps_since_gain: int = 0,
    current_score: int | None = None,
) -> str | None:
    """Focus substitute: try alternate focus phrasings before prep/nav fallbacks."""
    p = (planned or "").strip().lower()
    manip_obj = p[len("focus on ") :].strip() if p.startswith("focus on ") else p
    comparison_sem = _extract_comparison_semantics(task)

    if _task_focus_already_satisfied(task, action_history, ctx_text, current_score):
        manip_kw = dict(
            action_history=action_history,
            last_gain_family=last_gain_family,
            failed_actions=failed_actions,
            steps_since_gain=steps_since_gain,
            current_score=current_score,
        )
        # Still need instrument use before answer-box guesses.
        if _answer_selection_pending(task, action_history):
            for va in env_valid:
                if (va or "").strip().lower() in recent:
                    continue
                if _is_measurement_setup_action(
                    va, task, ctx_text, action_history, "", env_valid,
                ):
                    if logger:
                        logger.info(f"Focus substitute (measurement setup): {va!r}")
                    return va
        for picker, label in (
            (_pick_connect_exploit_substitute, "connect"),
            (_pick_post_focus_component_pick_substitute, "pick"),
            (_pick_exploit_manipulation_substitute, "manipulation"),
        ):
            sub = picker(
                env_valid, recent, ctx_text, task, logger, **manip_kw,
            )
            if sub and sub not in recent:
                if logger:
                    logger.info(f"Focus substitute (post-focus {label}): {sub!r}")
                return sub
        return None

    if _requires_location_before_focus(task, ctx_text):
        nav = _nav_to_task_location_hints(task, env_valid, recent, ctx_text)
        if nav and nav not in recent:
            if logger:
                logger.info(f"Focus substitute (need hint room): {nav!r}")
            return nav
        nav = _nav_toward_missing_target(
            task, env_valid, recent, ctx_text, action_history,
        )
        if nav and nav not in recent:
            if logger:
                logger.info(f"Focus substitute (nav to task area): {nav!r}")
            return nav
        return None

    p = _remap_fixture_focus_to_substance_in_container(p, ctx_text, task)

    box_now = _pick_justified_answer_box_focus(
        env_valid, task, action_history, ctx_text,
        recent=recent, failed=failed_actions,
    )
    if box_now:
        if logger:
            logger.info(f"Focus substitute (answer-box polarity): {box_now!r}")
        return box_now

    # Threshold measure-then-box: after an informative instrument use, prefer
    # answer-box focuses over instrument/substance thrash.
    if (
        _task_is_threshold_measurement_task(task)
        and _measurement_tool_already_used(action_history, task)
    ):
        for cand in _collect_visible_focus_candidates(
            env_valid, ctx_text, task, recent, failed_actions, planned=p,
            action_history=action_history, current_score=current_score,
        ):
            foc = (cand or "").strip().lower()
            obj = foc[len("focus on "):].strip() if foc.startswith("focus on ") else foc
            if _answer_box_destination(obj):
                if logger:
                    logger.info(f"Focus substitute (answer box after measure): {cand!r}")
                return cand

    examined_focus = _pick_focus_from_examined_substance(
        env_valid, recent, ctx_text, task, action_history, logger, failed_actions,
    )
    if examined_focus and examined_focus not in recent:
        return examined_focus

    if _focus_subgoal_pending(task, action_history, ctx_text, current_score):
        if _task_needs_liquid_fixture_prep(task, ctx_text) and _task_target_missing(
            task, ctx_text,
        ):
            current = _extract_current_room_name(ctx_text)
            if current != "bathroom":
                nav_opts = [a for a in ("go to bathroom", "open door to bathroom") if a in env_valid]
                nav_opts.sort(key=lambda x: (0 if x.startswith("go to ") else 1, x))
                for nav in nav_opts:
                    if nav not in recent:
                        if logger:
                            logger.info(f"Focus substitute (liquid nav): {nav!r}")
                        return nav

    create_product_pending = bool(
        _created_substance_product(task)
        and not _created_product_visible(task, ctx_text)
    )
    if (
        not create_product_pending
        and _focus_subgoal_pending(task, action_history, ctx_text, current_score)
        and (
            _task_target_visible(task, ctx_text)
            or any(
                _substance_confirmed_in_history(f"focus on {t}", task, action_history)
                for t in _task_target_tokens(task)
            )
        )
    ):
        tools = _measurement_task_tools(task) | _PRE_FOCUS_PREP_TOOL_WORDS
        measured = _measurement_tool_already_used(action_history, task)
        for t in sorted(_task_target_tokens(task), key=len, reverse=True):
            if t in _GENERIC_FOCUS_ENTITY_WORDS and _task_expects_context_entity_focus(task):
                continue
            # After measurement, do not re-focus the instrument as "substance".
            if measured and t in tools:
                continue
            if _focus_is_create_ingredient_not_product(t, task):
                continue
            if _created_substance_product(task) and not _created_product_visible(
                task, ctx_text,
            ):
                continue
            for phrase in (
                f"focus on substance called {t}",
                f"focus on {t}",
                f"focus on substance in {_active_prep_container_phrase(ctx_text, action_history) or 'metal pot'}",
            ):
                if phrase in env_valid and phrase not in recent:
                    if allow_focus_action(phrase, task, ctx_text, "", action_history):
                        if logger:
                            logger.info(f"Focus substitute (visible substance): {phrase!r}")
                        return phrase

    for cand in _collect_visible_focus_candidates(
        env_valid, ctx_text, task, recent, failed_actions, planned=p,
        action_history=action_history, current_score=current_score,
    ):
        if logger:
            logger.info(f"Focus substitute (ranked candidate): {cand!r}")
        return cand

    fuzzy = _pick_focus_from_fuzzy_plans(
        task, env_valid, ctx_text, recent, failed_actions, p, env_valid,
        action_history=action_history,
    )
    if fuzzy and fuzzy not in recent:
        if logger:
            logger.info(f"Focus substitute (fuzzy): {fuzzy!r}")
        return fuzzy

    prep_phase = _substance_prep_phase(
        task, ctx_text, action_history, failed_actions,
    )
    skip_prep = (
        _substance_examined_pending_focus(
            task, action_history, ctx_text, current_score,
        )
        or (
            _task_needs_substance_preparation(task)
            and _task_substance_visible(task, ctx_text)
            and not _task_needs_liquid_fixture_prep(task, ctx_text)
        )
    )
    if (
        _focus_subgoal_pending(task, action_history, ctx_text, current_score)
        and _task_needs_substance_preparation(task)
        and not skip_prep
        and prep_phase != "focus_ready"
        and not _prep_container_ready_for_focus(task, ctx_text, action_history)
    ):
        for act in _prep_actions_for_missing_substance(
            task, env_valid, recent, ctx_text, action_history, failed_actions, planned=p,
        ):
            if act not in recent:
                if logger:
                    logger.info(f"Focus substitute (prep): {act!r}")
                return act
        if prep_phase == "alt_liquid_nav":
            for nav in ("go to bathroom", "open door to bathroom"):
                if nav in env_valid and nav not in recent:
                    if logger:
                        logger.info(f"Focus substitute (alt liquid nav): {nav!r}")
                    return nav

    if _focus_subgoal_pending(task, action_history, ctx_text, current_score) and _task_target_missing(
        task, ctx_text,
    ):
        if _current_room_in_task_area(task, ctx_text):
            for act in _in_room_target_search_actions(
                task, env_valid, recent, ctx_text, action_history,
                planned=f"focus on {manip_obj}" if manip_obj else planned,
                last_gain_family=last_gain_family,
                failed_actions=failed_actions,
                steps_since_gain=steps_since_gain,
            ):
                if act not in recent and _pre_focus_substitute_allowed(
                    act, task, action_history, ctx_text,
                ):
                    if logger:
                        logger.info(f"Focus substitute (in-room search): {act!r}")
                    return act

    if _current_room_in_task_area(task, ctx_text):
        if not _substance_examined_pending_focus(
            task, action_history, ctx_text, current_score,
        ):
            for act in _in_room_target_search_actions(
                task, env_valid, recent, ctx_text, action_history,
                planned=f"focus on {manip_obj}" if manip_obj else planned,
                last_gain_family=last_gain_family,
                failed_actions=failed_actions,
                steps_since_gain=steps_since_gain,
            ):
                if act not in recent and _pre_focus_substitute_allowed(
                    act, task, action_history, ctx_text,
                ):
                    if logger:
                        logger.info(f"Focus substitute (in-room search): {act!r}")
                    return act

    safe = map_focus_to_safe_action(
        f"focus on {manip_obj}" if manip_obj else planned, env_valid, task, ctx_text,
    )
    if safe and safe not in recent and safe not in (failed_actions or set()):
        foc_obj = safe[len("focus on ") :].strip() if safe.startswith("focus on ") else ""
        if safe.startswith("focus on") and not _focus_has_task_overlap(
            foc_obj, task, comparison_sem, ctx_text,
        ):
            safe = None
        if safe and not _is_malformed_focus_object(foc_obj, manip_obj, env_valid, task):
            if logger:
                logger.info(f"Focus substitute (safe): {safe!r}")
            return safe

    if _task_target_missing(task, ctx_text) and (
        _current_room_in_task_area(task, ctx_text)
        or _room_has_substance_container_hints(ctx_text, task)
    ):
        for act in _in_room_target_search_actions(
            task, env_valid, recent, ctx_text, action_history,
            planned=p if p.startswith("focus on") else f"focus on {manip_obj}",
            last_gain_family=last_gain_family,
            failed_actions=failed_actions,
            steps_since_gain=steps_since_gain,
        ):
            if act not in recent and _pre_focus_substitute_allowed(
                act, task, action_history, ctx_text,
            ):
                if logger:
                    logger.info(f"Focus substitute (in-room prep/search): {act!r}")
                return act
        if (
            _task_needs_substance_preparation(task)
            and "look around" in env_valid
            and "look around" not in recent
            and _room_observation_incomplete(ctx_text)
        ):
            if logger:
                logger.info("Focus substitute (look around for substance)")
            return "look around"
    if _focus_subgoal_pending(task, action_history, ctx_text, current_score) and (
        _task_target_missing(task, ctx_text)
        or _current_room_weak_for_task(task, ctx_text)
    ):
        if _task_needs_substance_preparation(task):
            container = _pick_closed_container_for_prep(
                env_valid, recent, task, ctx_text, failed_actions, action_history,
            )
            if container and container not in recent:
                # Final heat/cold polarity guard (ultra-low freezer names included).
                if not _cold_open_blocked_on_heat(container, task):
                    if logger:
                        logger.info(f"Focus substitute (open storage): {container!r}")
                    return container
            if _defer_navigation_for_unopened_storage(
                task, ctx_text, env_valid, recent, failed_actions, action_history,
            ):
                return None
        nav = _nav_toward_preferred_rooms(
            task, env_valid, recent, ctx_text, action_history,
        )
        if (
            nav
            and nav not in recent
            and _focus_substitute_nav_allowed(
                nav, task, ctx_text, action_history, failed_actions,
            )
        ):
            if logger:
                logger.info(f"Focus substitute (task nav): {nav!r}")
            return nav
    return None

def _action_phrase_likely_ambiguous(phrase: str) -> bool:
    """Heuristic: nested object phrases often trigger disambiguation menus."""
    p = (phrase or "").strip().lower()
    if not p:
        return False
    nested = p.count(" in ") + p.count(" on ") + p.count(" containing ")
    if nested >= 2:
        return True
    return len(p.split()) > 8


def _task_post_focus_verb_tokens(task: str) -> set[str]:
    """Manipulation verbs explicitly mentioned in the task for steps after initial focus."""
    task_l = (task or "").lower()
    found: set[str] = set()
    for family in MANIPULATION_VERB_FAMILIES:
        if family in ("focus", "other"):
            continue
        phrase = family.replace("_", " ")
        if phrase in task_l or re.search(rf"\b{re.escape(phrase)}\b", task_l):
            found.add(family)
    return found


def _post_focus_progress_tokens(task: str, ctx_text: str) -> set[str]:
    """Tokens likely to advance a task after its initial focus/substance step."""
    tokens = (
        _task_target_tokens(task)
        | _context_task_anchor_tokens(task, ctx_text)
        | extract_task_content_tokens(task)
        | _task_post_focus_verb_tokens(task)
    )
    return tokens - _AMBIGUOUS_OVERLAP_TOKENS - _FOCUS_VAGUE_OBJECT_WORDS


def _manipulation_lacks_task_overlap(
    action: str, task: str, ctx_text: str, after_focus: bool = False,
) -> bool:
    """True when manipulation targets have no overlap with task-relevant tokens."""
    va_tokens = _action_content_tokens(action)
    if not va_tokens:
        return False
    anchor_tokens = _context_task_anchor_tokens(task, ctx_text)
    if anchor_tokens and (va_tokens & anchor_tokens):
        return False
    progress = _post_focus_progress_tokens(task, ctx_text) if after_focus else (
        _task_target_tokens(task) | anchor_tokens
    )
    return not (va_tokens & progress)


def _connect_is_task_relevant(action: str, task: str, ctx_text: str = "") -> bool:
    """Connect endpoints must overlap task/circuit/context anchors when task requires wiring."""
    al = (action or "").strip().lower()
    m = re.match(r"connect (.+?) to (.+)", al)
    if not m:
        return True
    if not _task_requires_connect(task):
        return True
    src_t = _action_content_tokens(m.group(1))
    dst_t = _action_content_tokens(m.group(2))
    endpoints = src_t | dst_t
    task_tokens = _task_target_tokens(task)
    anchor_tokens = _context_task_anchor_tokens(task, ctx_text)
    relevant = _CIRCUIT_HINT_TOKENS | task_tokens | anchor_tokens
    if endpoints & relevant:
        return True
    if _normalize_room_name(m.group(1)) in SCIENCEWORLD_ROOM_NAMES:
        return False
    if _normalize_room_name(m.group(2)) in SCIENCEWORLD_ROOM_NAMES:
        return False
    return False

def _planned_needs_navigation(planned: str, ctx_text: str) -> bool:
    """True when substitute should consider door/go (not in-room look/examine)."""
    p = (planned or "").strip().lower().replace("green house", "greenhouse")
    if p in ("look around", "inventory", "task"):
        return False
    vf = action_verb_family(p)
    if vf == "move" or p.startswith(("go to ", "open door to ", "teleport")):
        return True
    if p.startswith("look in "):
        dest = p[len("look in ") :].strip()
        return dest in SCIENCEWORLD_ROOM_NAMES and _is_remote_room_peek(p, ctx_text)
    return False


def _rooms_mentioned_in_planned(planned: str) -> set[str]:
    p = (planned or "").lower().replace("green house", "greenhouse")
    tokens = _action_content_tokens(p)
    return {r for r in SCIENCEWORLD_ROOM_NAMES if r in p or r.replace(" ", "") in _collapse_spaces(p) or _action_content_tokens(r) <= tokens}


def _task_target_tokens(task: str) -> set[str]:
    return {
        t for t in extract_task_content_tokens(task)
        if t not in _FOCUS_VAGUE_OBJECT_WORDS
    }


def _specific_task_tokens(task: str) -> set[str]:
    """Task tokens excluding colors/vague words — used to reject partial matches."""
    return _task_target_tokens(task) - _AMBIGUOUS_OVERLAP_TOKENS


def _is_distractor_object_for_task(
    obj_tokens: set[str], task_tokens: set[str], task: str = "",
) -> bool:
    """
    True when object shares only ambiguous tokens (e.g. color) with the task
    but omits task-specific nouns (e.g. focus red paint when task mentions red light bulb).
    """
    specific = task_tokens - _AMBIGUOUS_OVERLAP_TOKENS - _FOCUS_VAGUE_OBJECT_WORDS
    if not specific or not obj_tokens:
        return False
    overlap = obj_tokens & task_tokens
    if not overlap:
        return False
    if obj_tokens & specific:
        return False
    if overlap <= _AMBIGUOUS_OVERLAP_TOKENS:
        if task:
            instructed = _extract_focus_instruction_tokens(task)
            if obj_tokens & instructed:
                return False
        return True
    return False


def _spurious_task_overlap(action_tokens: set[str], task_tokens: set[str]) -> bool:
    """True when overlap is only via ambiguous tokens (e.g. color) but action adds unrelated nouns."""
    if _is_distractor_object_for_task(action_tokens, task_tokens):
        return True
    overlap = action_tokens & task_tokens
    if not overlap:
        return False
    specific = overlap - _AMBIGUOUS_OVERLAP_TOKENS
    if specific:
        return False
    extra = action_tokens - task_tokens - _FOCUS_VAGUE_OBJECT_WORDS - _AMBIGUOUS_OVERLAP_TOKENS
    return bool(extra)


def _task_focus_already_satisfied(
    task: str,
    action_history,
    ctx_text: str = "",
    current_score: int | None = None,
) -> bool:
    """True after a successful focus on the task-relevant object/substance."""
    targets = _task_target_tokens(task)
    hist = [(a or "").strip().lower() for a in (action_history or [])[-20:]]
    for act in reversed(hist):
        if not act.startswith("focus on"):
            continue
        obj = act[len("focus on ") :].strip()
        if not _focus_object_aligns_with_task(obj, task, ctx_text):
            continue
        if _mutually_exclusive_answer_boxes(task):
            if not _answer_box_destination(obj):
                continue
            if not _conditional_answer_commit_allowed(
                act, task, action_history, ctx_text,
            ):
                continue
            return True
        obj_tokens = _action_content_tokens(obj)
        stages = _sequenced_comparison_directions(task)
        if stages:
            done = _sequenced_comparison_stages_done(
                task, action_history, current_score,
            )
            if done < len(stages):
                continue
        if targets and (obj_tokens & targets):
            # Instrument-only focus does not complete measure-then-box tasks.
            if _task_is_threshold_measurement_task(task):
                tools = _measurement_task_tools(task) | _PRE_FOCUS_PREP_TOOL_WORDS
                if (obj_tokens & tools) and not _task_substance_focus_done(
                    task, action_history, ctx_text,
                ):
                    continue
                if _answer_box_destination(obj):
                    continue
            return True
        if _task_needs_substance_preparation(task) and (
            "substance in" in obj or "substance called" in obj
        ):
            return True
        if _focus_matches_task_substance(obj, task, ctx_text):
            return True
        if _comparison_focus_aligns(obj, task, ctx_text):
            # Sequenced longest-then-shortest: first aligned focus is not the end.
            stages = _sequenced_comparison_directions(task)
            if stages:
                done = _sequenced_comparison_stages_done(
                    task, action_history, current_score,
                )
                if done < len(stages):
                    continue
            return True
        if _planned_object_in_context(act, ctx_text) and obj_tokens & targets:
            return True
        # Align helper already passed — treat as focus milestone for post-focus tasks.
        if _task_has_post_focus_phase(task) or _task_requires_connect(task):
            return True
    # Score milestone after a focus attempt (ScienceWorld often awards ~50–63 on focus).
    if (
        current_score is not None
        and current_score >= 50
        and any(a.startswith("focus on") for a in hist)
    ):
        if _task_has_post_focus_phase(task) or _task_requires_connect(task):
            return True
        if re.search(r"\b(?:find|identify|locate)\b", (task or "").lower()) and re.search(
            r"\bmove (?:it|them|the \w+) to\b", (task or "").lower(),
        ):
            return True
    return False


def _complement_family_bonus(family: str, last_gain_family: str) -> int:
    if not last_gain_family:
        return 0
    preferred = _MANIPULATION_COMPLEMENT.get(
        last_gain_family,
        _MANIPULATION_COMPLEMENT.get("other", ()),
    )
    if not preferred:
        return 0
    try:
        rank = preferred.index(family)
        return max(0, 14 - rank * 2)
    except ValueError:
        return 0


def _score_manipulation_candidate(
    va: str,
    ctx_text: str,
    task: str,
    recent: set,
    action_history=None,
    last_gain_family: str = "",
    failed_actions: set | None = None,
    suppress_focus: bool = False,
    plateau_steps: int = 0,
    current_score: int | None = None,
) -> int | None:
    """Rank in-room / exploit manipulation actions (task-agnostic). Returns None to skip."""
    if _is_disallowed_substitute_action(va):
        return None
    if va in recent or va in (failed_actions or set()):
        return None
    # Block answer-box commits / box-shuffles for measure-then-box tasks.
    if _answer_box_action_disallowed(va, task, action_history):
        return None
    al = (va or "").strip().lower()
    if al.startswith("disconnect"):
        if _task_requires_connect(task) and _task_focus_already_satisfied(
            task, action_history, ctx_text, current_score,
        ):
            return None
    if al in ("look around", "inventory", "task"):
        return None
    if al.startswith(("go to ", "open door to ", "teleport")):
        return None
    if not action_grounded_in_context(va, ctx_text):
        return None
    if not is_safe_explore_action(va, task, ctx_text):
        return None
    if _is_post_focus_unproductive(
        va, task, action_history, ctx_text, current_score,
    ):
        return None
    if _is_counterproductive_plateau_action(
        va, task, action_history, ctx_text, current_score,
    ):
        return None
    if _task_focus_already_satisfied(task, action_history, ctx_text) and _is_absurd_post_focus_manipulation(
        va, task,
    ):
        return None
    if al.startswith("focus on"):
        if suppress_focus or not allow_focus_action(va, task, ctx_text, ""):
            return None
        foc_obj = al[len("focus on ") :].strip()
        if _is_fixture_focus_without_substance(foc_obj, _focus_relevant_tokens(task)):
            return None
        if _object_lacks_required_task_tokens(_action_content_tokens(foc_obj), task):
            return None
        if _task_focus_already_satisfied(task, action_history, ctx_text):
            return None
    family = action_verb_family(va)
    if family == "connect":
        m_conn = re.match(r"connect (.+?) to (.+)", al)
        if m_conn and (
            _is_forbidden_manipulation_target(m_conn.group(1))
            or _is_forbidden_manipulation_target(m_conn.group(2))
        ):
            return None
        if not _connect_is_task_relevant(va, task, ctx_text):
            return None
    elif family == "move":
        m_move = re.match(r"move (.+?) to (.+)", al)
        if m_move and _is_forbidden_manipulation_target(m_move.group(1)):
            return None
    elif family == "pour" and _is_destructive_pour_to_room(va):
        return None
    task_tokens = _task_target_tokens(task)
    specific_task = _specific_task_tokens(task)
    anchor_tokens = _context_task_anchor_tokens(task, ctx_text)
    va_tokens = _action_content_tokens(va)
    if _spurious_task_overlap(va_tokens, task_tokens):
        return None
    if _is_distractor_object_for_task(va_tokens, task_tokens, task):
        return None
    if family in ("pick_up", "take", "pour"):
        if specific_task and not (va_tokens & anchor_tokens) and not (va_tokens & specific_task):
            return None
        if va_tokens & _LOW_VALUE_EXPLORE_OBJECTS:
            return None
        fixtures = va_tokens & _NON_PORTABLE_PICK_WORDS
        if fixtures and not (fixtures & specific_task):
            return None
    if _is_pre_focus_destructive_action(va, task, action_history, ctx_text):
        return None
    if _is_premature_heat_before_focus(va, task, action_history, ctx_text):
        return None
    if _requires_focus_before_manipulation(
        task, action_history, ctx_text, current_score,
    ):
        if _is_pre_focus_preparation_action(va, task, ctx_text, action_history):
            pass
        else:
            blocked = {"pick_up", "take", "connect", "pour", "use"}
            if family in blocked:
                return None
            if family == "move":
                return None
            if family in ("activate", "deactivate"):
                return None
    focus_satisfied = _task_focus_already_satisfied(
        task, action_history, ctx_text, current_score,
    )
    post_focus = _task_has_post_focus_phase(task)
    if focus_satisfied and _is_post_focus_liquid_fixture_misactivation(
        va, task, ctx_text, action_history,
    ):
        return None
    if focus_satisfied:
        if family == "focus":
            return None
        if family == "move" and _is_relocation_to_room(va):
            return None
        if family in ("pick_up", "take", "pour", "move", "use") and _manipulation_lacks_task_overlap(
            va, task, ctx_text, after_focus=True,
        ):
            return None
    score = len(va_tokens & anchor_tokens) * 5 + len(va_tokens & specific_task) * 4
    score += len(va_tokens & task_tokens) * 2
    score += _complement_family_bonus(family, last_gain_family)
    if family == "open":
        score += 8
    elif family == "connect":
        score += 12
        if focus_satisfied:
            score += 25
            if _task_requires_connect(task):
                score += 35
        conn_t = va_tokens & (_CIRCUIT_HINT_TOKENS | task_tokens)
        score += len(conn_t) * 10
    elif family in ("pick_up", "take", "pour", "use", "move"):
        score += 9
        if focus_satisfied and _task_requires_connect(task):
            if family == "pour":
                score -= 45
            elif family in ("pick_up", "take"):
                obj_tokens = _action_content_tokens(va)
                if obj_tokens & (_CIRCUIT_HINT_TOKENS | _task_target_tokens(task)):
                    if _task_target_in_inventory(task, ctx_text):
                        score -= 35
                    else:
                        score += 28
                else:
                    score -= 45
            elif family == "move":
                score -= 18
        elif focus_satisfied and family in ("pour", "pick_up", "take"):
            pass
        if focus_satisfied and family == "move" and not _task_requires_connect(task):
            score -= 18
        if focus_satisfied and post_focus and family in MANIPULATION_VERB_FAMILIES - {"focus"}:
            score += len(va_tokens & _post_focus_progress_tokens(task, ctx_text)) * 6
    elif family == "wait":
        score += 7
        if focus_satisfied and post_focus:
            if wait_action_allowed(
                va, task, ctx_text, action_history, current_score,
            ):
                score += 22
            else:
                score -= 50
    elif family == "disconnect":
        score -= 80
    elif family == "activate":
        score += 5
        if focus_satisfied and post_focus:
            if _is_counterproductive_plateau_action(
                va, task, action_history, ctx_text, current_score,
            ):
                score -= 40
            else:
                score += 10
    elif family == "deactivate":
        score += 2
    elif family == "focus":
        foc_obj = al[len("focus on ") :].strip()
        focus_rel = _focus_relevant_tokens(task)
        if _is_fixture_focus_without_substance(foc_obj, focus_rel):
            return None
        if focus_rel and not (_action_content_tokens(foc_obj) & focus_rel):
            if not _extract_comparison_semantics(task):
                return None
        if _object_lacks_required_task_tokens(_action_content_tokens(foc_obj), task):
            return None
        rel = _action_content_tokens(foc_obj) & focus_rel
        score += len(rel) * 22 + len(va_tokens & specific_task) * 14
        if focus_rel and rel >= focus_rel:
            score += 40
        if comparison_sem := _extract_comparison_semantics(task):
            attr, direction = comparison_sem
            score += _comparison_entity_priority(foc_obj, attr, direction, task) * 2
        score += 3
    hist = [(a or "").strip().lower() for a in (action_history or [])[-6:]]
    if family in _TOGGLE_VERB_FAMILIES and plateau_steps >= 2:
        toggle_count = sum(1 for a in hist if action_verb_family(a) in _TOGGLE_VERB_FAMILIES)
        score -= min(12, toggle_count * 4)
        fixture = re.sub(r"^(open|close|activate|deactivate)\s+", "", al).strip()
        same_fixture = sum(
            1 for a in hist
            if action_verb_family(a) in _TOGGLE_VERB_FAMILIES
            and re.sub(r"^(open|close|activate|deactivate)\s+", "", a).strip() == fixture
        )
        if same_fixture >= 2:
            score -= 18
    if va in hist[-3:]:
        score -= 10
    # Threshold measure-then-box: prefer instrument use / heat over guessing.
    if _answer_selection_pending(task, action_history):
        if family == "use" and _use_action_is_instrument(al, _measurement_task_tools(task)):
            score += 55
        elif family == "activate" and is_heat_fixture_phrase(
            _action_manipulation_object(va) or al.replace("activate ", "", 1),
        ):
            score += 42
        elif family == "focus" and not _is_answer_selection_action(va):
            foc_obj = al[len("focus on "):].strip() if al.startswith("focus on ") else ""
            if _focus_matches_task_substance(foc_obj, task, ctx_text) or "substance" in foc_obj:
                score += 30
    return score
def _genuine_task_substance_focus(
    obj_phrase: str, task: str, ctx_text: str = "",
) -> bool:
    """Positive substance/entity focus match; does not call _false_task_substance_match."""
    targets = _task_target_tokens(task)
    if not targets:
        return True
    obj_l = (obj_phrase or "").strip().lower().replace("green house", "greenhouse")
    if not obj_l:
        return False
    obj_tokens = _action_content_tokens(obj_l)
    if obj_tokens & targets:
        if _is_distractor_object_for_task(obj_tokens, targets, task):
            return False
        instructed = _extract_focus_instruction_tokens(task)
        if instructed:
            core = instructed - _FOCUS_VAGUE_OBJECT_WORDS
            if core and (obj_tokens & core):
                return True
        specific = _specific_task_tokens(task)
        if specific and not (obj_tokens & specific):
            return False
        instructed = _extract_focus_instruction_tokens(task)
        if instructed:
            # Tasks often list several sequential focuses ("thermometer, then tin,
            # then <box>"). Matching any one instructed target is enough.
            core = instructed - _FOCUS_VAGUE_OBJECT_WORDS
            if core and not (obj_tokens & core):
                return False
        return True
    for t in targets:
        if re.search(rf"substance called {re.escape(t)}\b", obj_l):
            return True
    if "substance in" in obj_l:
        m = re.search(r"substance in ([a-z][a-z0-9 ]+)", obj_l)
        if m:
            container = m.group(1).strip()
            ctx = (ctx_text or "").lower()
            for t in targets:
                if re.search(
                    rf"{re.escape(container)} \(containing (?:a |an )?(?:substance called )?{re.escape(t)}\b",
                    ctx,
                ):
                    return True
                if re.search(
                    rf"\bin the {re.escape(container)} is:.*?{re.escape(t)}\b",
                    ctx,
                    re.S | re.I,
                ):
                    return True
                if re.search(
                    rf"in the {re.escape(container)} is:.*?substance called {re.escape(t)}",
                    ctx,
                    re.S | re.I,
                ):
                    return True
    return obj_tokens <= _FOCUS_VAGUE_OBJECT_WORDS


def _false_task_substance_match(obj_phrase: str, task: str, ctx_text: str = "") -> bool:
    """True when obj embeds a task token but is a container/distractor, not the task entity."""
    if _focus_is_create_ingredient_not_product(obj_phrase, task):
        return True
    obj_tokens = _action_content_tokens(obj_phrase)
    targets = _task_target_tokens(task)
    if not targets or not (obj_tokens & targets):
        return False
    obj_l = (obj_phrase or "").strip().lower()
    if "substance called" in obj_l:
        return False
    if "substance in" in obj_l:
        ctx = (ctx_text or "").lower()
        m = re.search(r"substance in ([a-z][a-z0-9 ]+)", obj_l)
        if m:
            container = m.group(1).strip()
            for t in targets:
                if re.search(
                    rf"{re.escape(container)} \(containing (?:a |an )?(?:substance called )?{re.escape(t)}\b",
                    ctx,
                ):
                    return False
            # Prep tasks still must not focus arbitrary vessels (fountain/sink water).
            if _container_holds_task_substance(container, task, ctx):
                return False
            return True
        return True
    if _is_distractor_object_for_task(obj_tokens, targets, task):
        return True
    extra = obj_tokens - targets - _FOCUS_VAGUE_OBJECT_WORDS
    if extra & _TASK_SUBSTANCE_CONTAINER_WORDS:
        # ``tin cup`` while boiling tin is a prep vessel, not a distractor.
        if _allowed_prep_vessel_focus(obj_l, task, ctx_text):
            return False
        return True
    material_colors = {"metal", "glass", "ceramic", "wood", "plastic"}
    if extra & material_colors:
        if _allowed_prep_vessel_focus(obj_l, task, ctx_text):
            return False
        return True
    if extra & _AMBIGUOUS_OVERLAP_TOKENS:
        if obj_tokens & targets and not (extra - targets):
            return False
        if _allowed_prep_vessel_focus(obj_l, task, ctx_text):
            return False
        if len(extra) > len(obj_tokens & targets):
            return True
        return True
    if len(extra) >= 2:
        return True
    ctx = (ctx_text or "").lower()
    if ctx and _ctx_contains_phrase(ctx, obj_l):
        overlap = obj_tokens & targets
        if len(overlap) >= 2 or overlap >= targets:
            return False
    return False


def _focus_matches_task_substance(obj_phrase: str, task: str, ctx_text: str = "") -> bool:
    """True when a focus target names the task substance (not a distractor like orange or air)."""
    if _false_task_substance_match(obj_phrase, task, ctx_text):
        return False
    return _genuine_task_substance_focus(obj_phrase, task, ctx_text)


def _task_substance_visible(task: str, ctx_text: str) -> bool:
    """True only when the task substance itself is named, not a substring (painting) or tin cup."""
    targets = _task_target_tokens(task)
    if not targets:
        return True
    ctx = (ctx_text or "").lower()
    for t in targets:
        if re.search(rf"substance called {re.escape(t)}\b", ctx):
            return True
        if re.search(
            rf"containing (?:a )?substance called {re.escape(t)}\b", ctx,
        ):
            return True
        if re.search(rf"containing {re.escape(t)}\b", ctx):
            return True
        if re.search(rf"\binventory\b.*?\b{re.escape(t)}\b", ctx, re.S):
            if not re.search(rf"\b{re.escape(t)}\s+(?:cup|pot|jar|can|bowl|pan)\b", ctx):
                return True
    return False


def _entity_tokens_in_context(task: str, ctx_text: str) -> set[str]:
    """Task tokens that appear as entities in observation/inventory (not only substance called)."""
    targets = _task_target_tokens(task)
    if not targets:
        return set()
    ctx = (ctx_text or "").lower()
    found: set[str] = set()
    for t in targets:
        if re.search(rf"\b{re.escape(t)}\b", ctx):
            found.add(t)
    life_hints = targets & {"animal", "plant", "life", "lifespan", "span", "longest", "lived", "bulb", "light"}
    if life_hints:
        task_l = (task or "").lower()
        living = bool(re.search(r"\b(?:living\s+things?|organism)\b", task_l))
        plant_only = bool(re.search(r"\bplant\b", task_l)) and not (
            re.search(r"\banimal\b", task_l) or living
        )
        animal_only = bool(re.search(r"\banimal\b", task_l)) and not (
            re.search(r"\bplant\b", task_l) or living
        )
        # Animal stages (egg/baby) must not count as a plant being present, and
        # plant stages (seed) must not count as an animal. Bare "adult" is both.
        _animal_stages = frozenset({"egg", "baby", "larva", "juvenile"})
        _plant_stages = frozenset({"seed", "sprout", "seedling"})
        for stage in _ENTITY_STAGE_WORDS:
            if not re.search(rf"\b{re.escape(stage)}\b", ctx):
                continue
            if plant_only and stage in _animal_stages:
                continue
            if animal_only and stage in _plant_stages:
                continue
            if stage == "adult" and (plant_only or animal_only):
                continue
            found.add(stage)
    return found


def _context_task_anchor_tokens(task: str, ctx_text: str) -> set[str]:
    """Task tokens that also appear in the current observation/inventory."""
    task_t = _task_target_tokens(task)
    if not task_t:
        return set()
    ctx = (ctx_text or "").lower()
    anchors = {t for t in task_t if re.search(rf"\b{re.escape(t)}\b", ctx)}
    anchors |= _entity_tokens_in_context(task, ctx_text)
    return anchors


def _task_target_visible(task: str, ctx_text: str) -> bool:
    if _task_substance_visible(task, ctx_text):
        return True
    # Category-typed find/identify: a wrong-family organism (frog egg on a
    # plant search) is not the target — do not treat the room as arrived.
    try:
        from core.scienceworld_policy import _category_matching_object_visible
        cat = _category_matching_object_visible(task, ctx_text)
        if cat is False:
            if _task_is_life_grow_family(task) and _entity_tokens_in_context(
                task, ctx_text,
            ):
                return True
            return False
        if cat is True:
            return True
    except Exception:
        pass
    # Mutually exclusive answer boxes are UI, not the experimental target.
    # Treat them as visible only once a seed/plant (or other non-box target) is.
    if _mutually_exclusive_answer_boxes(task):
        ctx = (ctx_text or "").lower()
        return bool(re.search(
            r"\b(?:pea|seed|plant|flower\s*pot|sprout)\b", ctx,
        ))
    return bool(_entity_tokens_in_context(task, ctx_text))


def _task_target_missing(task: str, ctx_text: str) -> bool:
    return not _task_target_visible(task, ctx_text)


def should_block_focus_before_visible(
    action: str, task: str, ctx_text: str, action_history=None,
) -> bool:
    """Block focus only when the object is not in context and not task-relevant."""
    al = (action or "").strip().lower().replace("green house", "greenhouse")
    if not al.startswith("focus on"):
        return False
    obj = al.replace("focus on", "", 1).strip()
    if _extract_comparison_semantics(task):
        if (
            _comparison_focus_aligns(obj, task, ctx_text)
            and _entity_visible_in_context(obj, ctx_text)
        ):
            return False
    if _requires_location_before_focus(task, ctx_text):
        return True
    if _planned_object_in_context(al, ctx_text):
        return False
    if _current_room_in_task_area(task, ctx_text) and _entity_visible_in_context(obj, ctx_text):
        if _task_expects_context_entity_focus(task) or _focus_matches_task_substance(obj, task, ctx_text):
            return False
    if _substance_confirmed_in_history(obj, task, action_history):
        return False
    if is_off_task_planned_action(al, task, ctx_text):
        return True
    if _false_task_substance_match(obj, task, ctx_text):
        return True
    if _task_focus_already_satisfied(task, action_history, ctx_text):
        return True
    if _focus_matches_task_substance(obj, task, ctx_text):
        return False
    if _task_target_visible(task, ctx_text):
        return not _entity_visible_in_context(obj, ctx_text)
    return True


def _manipulation_phase_active(
    task: str,
    ctx_text: str,
    action_history,
    current_score: int | None = None,
    score_max: int | None = None,
    steps_since_gain: int = 0,
) -> bool:
    """After task-relevant focus or score plateau — prefer manipulation over navigation."""
    if _task_focus_already_satisfied(
        task, action_history, ctx_text, current_score,
    ):
        return True
    if _requires_focus_before_manipulation(task, action_history, ctx_text):
        return False
    for act in reversed([(a or "").strip().lower() for a in (action_history or [])[-15:]]):
        if act.startswith("focus on"):
            obj = act[len("focus on ") :].strip()
            if _focus_object_aligns_with_task(obj, task, ctx_text):
                return True
            if _planned_object_in_context(act, ctx_text):
                obj_tokens = _action_content_tokens(obj)
                if obj_tokens & _task_target_tokens(task):
                    if not _is_distractor_object_for_task(
                        obj_tokens, _task_target_tokens(task), task,
                    ):
                        return True
    if current_score is None or score_max is None:
        return False
    return (
        score_max > 0
        and current_score >= score_max
        and current_score < 100
        and steps_since_gain >= 1
    )


def _substance_focus_subgoal_satisfied(
    task: str,
    ctx_text: str,
    action_history,
    current_score: int | None = None,
    score_max: int | None = None,
    steps_since_gain: int = 0,
) -> bool:
    return _manipulation_phase_active(
        task, ctx_text, action_history, current_score, score_max, steps_since_gain,
    )


def _in_room_progress_actions(
    env_valid: set,
    recent: set,
    ctx_text: str,
    task: str,
    force_any_room: bool = False,
    action_history=None,
    last_gain_family: str = "",
    failed_actions: set | None = None,
    steps_since_gain: int = 0,
    current_score: int | None = None,
) -> list[str]:
    """Open/activate/pick actions in the current room when movement is blocked."""
    current = _extract_current_room_name(ctx_text)
    preferred = _task_preferred_rooms(task)
    if not force_any_room:
        if not current or not preferred or current != preferred[0]:
            return []
    elif not current:
        return []
    suppress_focus = _task_focus_already_satisfied(
        task, action_history, ctx_text, current_score,
    )
    ranked: list[tuple[int, str]] = []
    for va in env_valid:
        sc = _score_manipulation_candidate(
            va, ctx_text, task, recent, action_history,
            last_gain_family, failed_actions, suppress_focus, steps_since_gain,
            current_score=current_score,
        )
        if sc is not None:
            ranked.append((sc, va))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [a for _, a in ranked]


def _nested_container_accessible(phrase: str, ctx_text: str) -> bool:
    """Nested objects (drawer in cupboard) require the parent container to be open."""
    phrase = (phrase or "").strip().lower()
    if not phrase:
        return True
    if " in " not in phrase:
        return not _container_door_closed(phrase, ctx_text)
    _, parent = phrase.split(" in ", 1)
    parent = parent.strip()
    if _container_door_closed(parent, ctx_text):
        return False
    if parent in ("cupboard", "freezer", "fridge", "oven", "drawer"):
        ctx = ctx_text or ""
        if not re.search(
            rf"{re.escape(parent)}.*\bdoor is open\b|in the {re.escape(parent)} is:",
            ctx,
            re.I,
        ):
            return parent not in ("counter", "desk")
    return True


def _drawer_recently_opened(action_history) -> str | None:
    for act in reversed([(a or "").strip().lower() for a in (action_history or [])[-6:]]):
        if act.startswith("open drawer"):
            return act[len("open ") :].strip()
    return None


def _substitute_open_drawer_plan(
    planned: str,
    env_valid: set,
    recent: set,
    ctx_text: str,
    action_history,
    logger,
) -> str | None:
    p = (planned or "").strip().lower()
    if not p.startswith("open ") or "drawer" not in p:
        return None
    opened = _drawer_recently_opened(action_history)
    if opened:
        for prefix in ("look in", "look at"):
            for cand in _fuzzy_object_actions(prefix, opened, env_valid, ctx_text):
                if cand not in recent:
                    if logger:
                        logger.info(
                            f"Substitute for redundant open drawer {planned!r}: {cand}"
                        )
                    return cand
    if _container_door_closed("cupboard", ctx_text) and "cupboard" in (ctx_text or ""):
        if "open cupboard" in env_valid and "open cupboard" not in recent:
            if logger:
                logger.info(
                    f"Substitute for blocked open drawer {planned!r}: open cupboard"
                )
            return "open cupboard"
    for cand in _fuzzy_open_actions("drawer", env_valid, ctx_text):
        obj = cand[5:].strip()
        if not _nested_container_accessible(obj, ctx_text):
            continue
        if cand not in recent:
            if logger:
                logger.info(f"Substitute for blocked plan {planned!r}: {cand}")
            return cand
    return None


def _parent_context_score(obj: str, ctx_text: str, base_word: str) -> int:
    """Prefer 'drawer in counter' when observation says 'On the counter is: ... drawer'."""
    obj = (obj or "").lower().strip()
    base_word = (base_word or "").lower().strip()
    if " in " not in obj or not base_word:
        return 0
    parent = obj.split(" in ", 1)[1].strip()
    patterns = (
        rf"on the {re.escape(parent)} is:.*?\b{re.escape(base_word)}\b",
        rf"in the {re.escape(parent)} is:.*?\b{re.escape(base_word)}\b",
        rf"{re.escape(parent)}.*?\b{re.escape(base_word)}\b",
    )
    ctx = ctx_text or ""
    for pat in patterns:
        if re.search(pat, ctx, re.I | re.S):
            return 20
    return 0


def _fuzzy_match_score(
    prefix: str, object_phrase: str, action: str, ctx_text: str,
) -> int:
    phrase = (object_phrase or "").strip().lower()
    al = (action or "").strip().lower()
    score = 0
    if al == f"{prefix} {phrase}":
        score += 30
    if al.startswith(prefix + " "):
        score += 10
    obj = al[len(prefix) + 1 :].strip() if al.startswith(prefix + " ") else al
    if _collapse_spaces(obj) == _collapse_spaces(phrase):
        score += 25
    phrase_tokens = _action_content_tokens(phrase)
    obj_tokens = _action_content_tokens(obj)
    if phrase_tokens and phrase_tokens <= obj_tokens:
        score += 8 + len(phrase_tokens & obj_tokens)
    base_word = phrase.split()[0] if phrase.split() else phrase
    score += _parent_context_score(obj, ctx_text, base_word)
    if not _nested_container_accessible(obj, ctx_text):
        score -= 80
    if prefix == "look in" and al.startswith("look in "):
        score += 6
    elif prefix == "look at" and al.startswith("look at "):
        score += 4
    elif prefix == "open" and al.startswith("open "):
        score += 5
    return score


def _fuzzy_object_actions(prefix: str, object_phrase: str, env_valid: set, ctx_text: str) -> list[str]:
    """Match env action strings when planner/Actor wording differs slightly from obs."""
    phrase = (object_phrase or "").strip().lower()
    if not phrase:
        return []
    direct = f"{prefix} {phrase}"
    matches: list[str] = []
    if direct in env_valid and _ctx_contains_phrase(ctx_text, phrase):
        matches.append(direct)
    phrase_tokens = _action_content_tokens(phrase)
    if " in " in phrase:
        short = phrase.split(" in ", 1)[0].strip()
        if short and short != phrase:
            direct_short = f"{prefix} {short}"
            if direct_short in env_valid and _ctx_contains_phrase(ctx_text, short):
                matches.append(direct_short)
    for va in env_valid:
        al = va.strip().lower()
        if not al.startswith(prefix + " "):
            continue
        obj = al[len(prefix) + 1 :]
        if _ctx_contains_phrase(ctx_text, obj) and (
            _collapse_spaces(obj) == _collapse_spaces(phrase)
            or (phrase_tokens and phrase_tokens <= _action_content_tokens(obj))
            or (phrase_tokens and phrase_tokens <= _action_content_tokens(va))
        ):
            matches.append(va)
    seen: set[str] = set()
    deduped: list[str] = []
    for va in matches:
        if va not in seen:
            seen.add(va)
            deduped.append(va)
    deduped.sort(
        key=lambda va: (
            -_fuzzy_match_score(prefix, phrase, va, ctx_text),
            len(va),
            va,
        )
    )
    return [va for va in deduped if _nested_container_accessible(
        va[len(prefix) + 1 :].strip() if va.startswith(prefix + " ") else va,
        ctx_text,
    )]


def _fuzzy_connect_actions(
    planned: str, env_valid: set, ctx_text: str, task: str = "",
) -> list[str]:
    """Match connect X to Y when planner wording differs from env action strings."""
    p = (planned or "").strip().lower()
    m = re.match(r"connect (.+?) to (.+)", p)
    if not m:
        return []
    src, dst = m.group(1).strip(), m.group(2).strip()
    src_t, dst_t = _action_content_tokens(src), _action_content_tokens(dst)
    task_t = _task_target_tokens(task)
    ranked: list[tuple[int, str]] = []
    for va in env_valid:
        al = va.strip().lower()
        mm = re.match(r"connect (.+?) to (.+)", al)
        if not mm:
            continue
        if _is_forbidden_manipulation_target(mm.group(1)) or _is_forbidden_manipulation_target(mm.group(2)):
            continue
        if not _connect_is_task_relevant(va, task, ctx_text):
            continue
        if not action_grounded_in_context(va, ctx_text):
            continue
        s_t, d_t = _action_content_tokens(mm.group(1)), _action_content_tokens(mm.group(2))
        score = len(s_t & src_t) * 5 + len(d_t & dst_t) * 5 + len((s_t | d_t) & task_t) * 4
        score += len((s_t | d_t) & _CIRCUIT_HINT_TOKENS) * 8
        if al == p:
            score += 30
        if score > 0:
            ranked.append((score, va))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [a for _, a in ranked]


def _fuzzy_focus_actions(
    planned: str, env_valid: set, ctx_text: str, task: str = "",
) -> list[str]:
    """Match focus on X when planner wording differs from env action strings."""
    obj = (planned or "").lower().replace("focus on", "", 1).strip()
    if not obj:
        return []
    obj_tokens = _action_content_tokens(obj)
    task_tokens = _task_target_tokens(task)
    specific = _specific_task_tokens(task)
    focus_rel = _focus_relevant_tokens(task)
    comparison_sem = _extract_comparison_semantics(task)
    ranked: list[tuple[int, str]] = []
    for va in env_valid:
        al = va.strip().lower()
        if not al.startswith("focus on "):
            continue
        if not allow_focus_action(va, task, ctx_text, ""):
            continue
        foc_obj = al[len("focus on ") :].strip()
        foc_tokens = _action_content_tokens(foc_obj)
        if _is_malformed_focus_object(foc_obj, obj, env_valid, task):
            continue
        task_anchor = focus_rel | specific
        if task_anchor and not comparison_sem:
            if not (foc_tokens & task_anchor):
                if not ("substance in" in foc_obj and _task_needs_substance_preparation(task)):
                    continue
        elif comparison_sem and "substance in" in foc_obj:
            if not _comparison_focus_aligns(foc_obj, task, ctx_text):
                continue
        match_tokens = obj_tokens | task_anchor
        overlap = foc_tokens & match_tokens
        if obj_tokens and not overlap:
            if not ("substance in" in foc_obj and _task_needs_substance_preparation(task)):
                continue
        score = len(foc_tokens & obj_tokens) * 5 + len(foc_tokens & task_tokens) * 6
        score += len(foc_tokens & specific) * 7 + len(overlap) * 10
        if _collapse_spaces(foc_obj) == _collapse_spaces(obj):
            score += 35
        if obj_tokens and obj_tokens <= foc_tokens:
            score += 12 + len(obj_tokens & foc_tokens)
        if "substance called" in al and (obj_tokens & task_tokens or overlap):
            score += 10
        if "substance called" in obj and "substance in" in foc_obj:
            if _focus_matches_task_substance(foc_obj, task, ctx_text):
                score += 28
        if "substance in" in foc_obj and "substance called" in obj:
            if _focus_matches_task_substance(foc_obj, task, ctx_text):
                score += 28
        if _is_distractor_object_for_task(foc_tokens, task_tokens, task):
            continue
        if _false_task_substance_match(foc_obj, task, ctx_text):
            continue
        if _object_lacks_required_task_tokens(foc_tokens, task):
            continue
        nested = foc_obj.count(" in ") + foc_obj.count(" on ") + foc_obj.count(" containing ")
        score -= nested * 8
        score -= max(0, len(foc_obj.split()) - 6) * 3
        if _action_phrase_likely_ambiguous(foc_obj):
            score -= 25
        if comparison_sem:
            attr, direction = comparison_sem
            pri = _comparison_entity_priority(foc_obj, attr, direction, task)
            if pri < 45:
                continue
            score += pri * 4
            if foc_tokens & _ENTITY_STAGE_WORDS and pri < 55:
                score -= 50
        if score > 0:
            ranked.append((score, va))
    if comparison_sem:
        ranked.sort(
            key=lambda x: (
                -_focus_action_comparison_priority(x[1], task),
                -x[0],
                len(x[1]),
                x[1],
            ),
        )
    else:
        ranked.sort(
            key=lambda x: (
                -x[0],
                _action_phrase_likely_ambiguous(x[1][len("focus on ") :].strip()),
                len(x[1]),
                x[1],
            ),
        )
    return [a for _, a in ranked]


def _recently_visited_rooms(action_history, n: int = 4) -> set[str]:
    rooms: set[str] = set()
    for act in (action_history or [])[-n:]:
        al = (act or "").strip().lower().replace("green house", "greenhouse")
        if al.startswith("go to "):
            rooms.add(al[len("go to ") :].strip())
        elif al.startswith("open door to "):
            rooms.add(al[len("open door to ") :].strip())
    return rooms


def detect_manipulation_stagnation(action_history, steps_since_gain: int = 0) -> bool:
    """True when recent steps are in-room manipulation without score progress."""
    if steps_since_gain < EXPLOIT_STAGNANT_THRESHOLD:
        return False
    hist = [(a or "").strip().lower() for a in (action_history or [])[-8:]]
    stagnant_families = {
        "pick_up", "take", "pour", "open", "activate", "deactivate", "use", "move", "connect",
    }
    manip_count = sum(1 for a in hist if action_verb_family(a) in stagnant_families)
    return manip_count >= 3


def _current_room_weak_for_task(task: str, ctx_text: str) -> bool:
    """True when observation lacks task-specific anchor tokens (should explore elsewhere)."""
    # Create-then-focus chemistry: recipe work is in the hinted kitchen, not foundry.
    product = _created_substance_product(task)
    if product:
        try:
            current = _extract_current_room_name(ctx_text)
        except Exception:
            current = ""
        work_rooms = {"kitchen", "hallway"}
        if re.search(r"\bpaint\b", (task or "").lower()):
            work_rooms = {"art studio", "hallway", "kitchen"}
        if current and current not in work_rooms:
            return True
    # SOM phase-change: kitchen/workshop fixtures, not decorative rooms.
    # Generic "freeze the substance" has no specific tokens, so the later
    # `if not specific: return False` must not treat the art studio as fine.
    try:
        from core.pattern_library import _task_family_flags
        fam_weak = _task_family_flags(task)
    except Exception:
        fam_weak = {}
    if fam_weak.get("som") and not fam_weak.get("chemistry"):
        try:
            current = _extract_current_room_name(ctx_text)
        except Exception:
            current = ""
        ctx = (ctx_text or "").lower()
        freeze_only = bool(
            re.search(r"\bfreeze\b", (task or "").lower())
            and not re.search(r"\b(?:boil|melt)\b", (task or "").lower())
        )
        if freeze_only:
            has_fix = bool(re.search(r"\b(?:freezer|fridge|refrigerator)\b", ctx))
        else:
            has_fix = bool(re.search(r"\b(?:stove|oven|hot\s*plate|burner)\b", ctx))
            if _task_allows_foundry_heat(task):
                has_fix = has_fix or bool(
                    re.search(r"\b(?:blast furnace|furnace)\b", ctx)
                )
        if has_fix:
            return False
        if current in {"art studio", "bedroom", "bathroom", "foundry", "living room"}:
            return True
    # Measure-then-box: colored answer boxes are distractors; need instrument/heat room.
    if _task_is_threshold_measurement_task(task):
        ctx = (ctx_text or "").lower()
        tools = _measurement_task_tools(task) or {"thermometer"}
        has_tool = any(re.search(rf"\b{re.escape(t)}\b", ctx) for t in tools if t)
        # Kitchen stove/oven only. Foundry blast furnace is a different family
        # of heat and must not count as "already in a measure work room".
        has_heat = bool(re.search(r"\b(?:stove|oven|hot\s*plate|burner)\b", ctx))
        if _task_allows_foundry_heat(task):
            has_heat = has_heat or bool(
                re.search(r"\b(?:blast furnace|furnace)\b", ctx)
            )
        if has_tool or has_heat:
            return False
        return True
    specific = _specific_task_tokens(task)
    if not specific:
        return False
    if _solid_substance_prep_in_preferred_room(task, ctx_text):
        return False
    anchors = _context_task_anchor_tokens(task, ctx_text)
    return not (anchors & specific)


def _wire_navigation_helpers() -> None:
    """Bind episode-phase helpers into navigation_helpers (cross-module calls)."""
    import importlib

    nav = importlib.import_module("core.navigation_helpers")
    for name, val in globals().items():
        if name.startswith("__") or name == "_wire_navigation_helpers":
            continue
        if not hasattr(nav, name):
            setattr(nav, name, val)


_wire_navigation_helpers()

import core.navigation_helpers as _navigation_helpers
from core._grounding_merge import merge_module_exports

merge_module_exports(globals(), _navigation_helpers)


def planner_phase_navigation_hint(task: str, ctx_text: str) -> str:
    """Prompt guidance for 9B planner during explore/focus-pending phases."""
    lines: list[str] = []
    hints = _extract_task_location_hint_rooms(task)
    preferred = _task_preferred_rooms(task)
    if hints:
        lines.append(f"Task location hints: {', '.join(hints)}.")
    elif preferred:
        lines.append(f"Preferred rooms from task wording: {', '.join(preferred[:3])}.")
    if _task_target_missing(task, ctx_text) or _current_room_weak_for_task(task, ctx_text):
        lines.append(
            "The task target is NOT visible in the current observation (or the current "
            "room lacks task-relevant objects). Predict `go to <connected room>` toward "
            "a preferred or hint room, or `look around`. Do NOT predict `focus on` until "
            "the target appears in the observation. Do NOT predict `teleport`."
        )
        closed = _closed_storage_containers_in_context(ctx_text)
        if closed:
            lines.append(
                "Closed containers are visible ("
                + ", ".join(closed[:4])
                + "): open them before leaving the room."
            )
        phase = _substance_prep_phase(task, ctx_text, None, None)
        if phase == "solid_search":
            doors = _extract_door_destinations(ctx_text)
            if doors:
                lines.append(
                    "Local containers exhausted; search other connected rooms next: "
                    + ", ".join(doors[:5])
                    + "."
                )
    deprior = sorted(_task_deprioritized_rooms(task, None))
    if deprior:
        lines.append(f"Avoid navigating to: {', '.join(deprior[:6])}.")
    return "\n".join(lines)


def planner_phase_focus_hint(
    task: str,
    ctx_text: str,
    *,
    past_actions=None,
    env_valid=None,
    current_score: int | None = None,
    inventory: str = "",
) -> str:
    """Prompt guidance for focus-pending and post-focus manipulation phases."""
    lines: list[str] = []
    ctx = f"{ctx_text or ''} {inventory or ''}".lower()
    history = past_actions or []

    if _focus_subgoal_pending(task, history, ctx, current_score):
        comp = _extract_comparison_semantics(task)
        if comp:
            attr, direction = comp
            rank_map = _COMPARISON_ENTITY_RANKS.get(
                attr, _COMPARISON_ENTITY_RANKS["lifespan"],
            )
            if direction == "max":
                preferred = sorted(rank_map, key=rank_map.get, reverse=True)[:5]
                avoid = sorted(rank_map, key=rank_map.get)[:4]
            else:
                preferred = sorted(rank_map, key=rank_map.get)[:5]
                avoid = sorted(rank_map, key=rank_map.get, reverse=True)[:4]
            lines.append(
                f"Comparison task ({attr}, {direction}): prefer entities like "
                f"{', '.join(preferred)}; avoid {', '.join(avoid)} and immature "
                f"stages (baby, egg) unless the task explicitly asks for them."
            )
            lines.append(
                "Only predict `focus on <entity>` when that entity appears in the "
                "current observation. Prefer the best-matching visible candidate."
            )
        elif _task_needs_substance_preparation(task):
            lines.append(
                "Substance task: focus the named substance or its container when visible. "
                "Do not focus fixtures (stove, sink) before the substance is located."
            )
        else:
            focus_rel = _focus_relevant_tokens(task)
            if focus_rel:
                lines.append(
                    f"Focus target should match task nouns: {', '.join(sorted(focus_rel)[:8])}."
                )
            lines.append(
                "Do not predict `focus on` until the target is visible in the observation."
            )

        if env_valid:
            recent = set((a or "").strip().lower() for a in history[-3:])
            candidates = _collect_visible_focus_candidates(
                env_valid, ctx, task, recent,
                action_history=history, current_score=current_score,
            )
            if candidates:
                lines.append(
                    "Top grounded focus candidates (prefer in order): "
                    + ", ".join(candidates[:3])
                )
            elif _task_target_missing(task, ctx):
                lines.append(
                    "No valid focus target visible yet — navigate or `look around` first."
                )

    elif _task_focus_already_satisfied(task, history, ctx, current_score):
        som_phase = _state_of_matter_post_focus_phase(
            task, current_score, history, ctx,
        )
        if som_phase == "heat":
            lines.append(
                "Post-focus heat phase: activate stove/oven or move the focused "
                "substance/container to heat source. Do NOT move unrelated objects "
                "(apple, banana) to the stove."
            )
        elif som_phase == "precool":
            lines.append(
                "Post-focus precool phase: open freezer or move focused substance to freezer, then wait."
            )
        elif som_phase == "cool":
            lines.append(
                "Post-focus cool phase: move focused substance to freezer or activate cooling."
            )
        elif _task_has_post_focus_phase(task):
            verbs = _task_post_focus_verb_tokens(task)
            if verbs:
                lines.append(
                    f"Post-focus manipulation: prioritize actions involving "
                    f"{', '.join(sorted(verbs)[:6])} on the focused target."
                )
        if env_valid:
            recent = set((a or "").strip().lower() for a in history[-3:])
            som_sub = _pick_state_of_matter_post_focus_substitute(
                env_valid, recent, ctx, task, history, current_score, set(),
            )
            if som_sub:
                lines.append(f"Suggested next action: {som_sub}")

    return "\n".join(lines)
