"""
Task-agnostic environment discovery from observations and action history.

Design: infer containers, heat fixtures, and exploration state from what the
agent can see — not from per-task or per-substance lookup tables.
"""
from __future__ import annotations

import re

# Semantic hints for fixture classification (not an exhaustive object list).
_HEAT_FIXTURE_HINTS = frozenset({
    "furnace", "stove", "oven", "burner", "heater", "forge", "kiln", "fire",
})
_COOL_FIXTURE_HINTS = frozenset({
    "freezer", "fridge", "refrigerator", "cooler",
})

# Known storage containers (merged with observation parsing).
_STORAGE_CONTAINER_NAMES = (
    "cupboard", "drawer", "fridge", "freezer", "oven", "jar", "closet", "cabinet",
)


def room_visit_counts(action_history) -> dict[str, int]:
    """How many times each room was entered this episode."""
    counts: dict[str, int] = {}
    for act in action_history or []:
        al = (act or "").strip().lower().replace("green house", "greenhouse")
        room = None
        if al.startswith("go to "):
            room = al[len("go to ") :].strip()
        elif al.startswith("open door to "):
            room = al[len("open door to ") :].strip()
        if room:
            counts[room] = counts.get(room, 0) + 1
    return counts


def is_heat_fixture_phrase(phrase: str) -> bool:
    """True when a fixture name plausibly provides heating (not cooling)."""
    pl = (phrase or "").strip().lower()
    if not pl:
        return False
    if any(h in pl for h in _COOL_FIXTURE_HINTS):
        return False
    return any(h in pl for h in _HEAT_FIXTURE_HINTS)


def is_cool_fixture_phrase(phrase: str) -> bool:
    pl = (phrase or "").strip().lower()
    return any(h in pl for h in _COOL_FIXTURE_HINTS)


def closed_containers_in_observation(ctx_text: str) -> list[str]:
    """
    Parse closed containers from a room observation.
    Complements fixed-name lists with generic 'door is closed' patterns.
    """
    ctx = (ctx_text or "").lower()
    found: list[str] = []
    seen: set[str] = set()

    for name in _STORAGE_CONTAINER_NAMES:
        if not re.search(rf"\b{re.escape(name)}\b", ctx):
            continue
        if re.search(rf"\b{re.escape(name)}\b(?:[^.\n]{{0,48}})\bclosed\b", ctx):
            if name not in seen:
                seen.add(name)
                found.append(name)

    for m in re.finditer(
        r"\ba ([a-z][a-z0-9 ]{0,40}?)\.\s*(?:the )?\1(?: door)?(?:[^.\n]{0,40})?\b(?:door )?is closed\b",
        ctx,
    ):
        name = m.group(1).strip()
        if name and name not in seen:
            seen.add(name)
            found.append(name)

    return found


def discover_fixtures_from_observation(
    ctx_text: str,
    *,
    heat: bool = True,
) -> list[str]:
    """Extract activatable fixture names visible in the current observation."""
    ctx = (ctx_text or "").lower()
    fixtures: list[str] = []
    seen: set[str] = set()

    for m in re.finditer(
        r"\ba ([a-z][a-z0-9 ]{1,40}?),\s*which is (?:turned (?:off|on)|activated|deactivated)",
        ctx,
    ):
        name = m.group(1).strip()
        ok = is_heat_fixture_phrase(name) if heat else is_cool_fixture_phrase(name)
        if ok and name not in seen:
            seen.add(name)
            fixtures.append(name)

    for m in re.finditer(r"\ba (blast furnace|ultra low temperature freezer)\b", ctx):
        name = m.group(1).strip()
        ok = is_heat_fixture_phrase(name) if heat else is_cool_fixture_phrase(name)
        if ok and name not in seen:
            seen.add(name)
            fixtures.append(name)

    return fixtures


def _heat_fixture_rank(name: str) -> tuple[int, str]:
    """Stronger / higher-temperature fixtures first (generic ordering)."""
    nl = (name or "").lower()
    if "furnace" in nl:
        return (0, name)
    if "oven" in nl:
        return (1, name)
    if "stove" in nl:
        return (2, name)
    return (3, name)


def heat_fixtures_in_environment(
    env_valid: set | None,
    ctx_text: str = "",
) -> list[str]:
    """
    All heat fixture targets derivable from valid actions and the observation.
    Ordered strongest-first so escalation tries higher-capacity sources next.
    """
    seen: set[str] = set()
    result: list[str] = []

    for act in env_valid or ():
        al = (act or "").strip().lower()
        if al.startswith("activate "):
            target = al[len("activate ") :].strip()
            if is_heat_fixture_phrase(target) and target not in seen:
                seen.add(target)
                result.append(target)
        m = re.match(r"^move .+? to (.+)$", al)
        if m:
            dest = m.group(1).strip()
            if is_heat_fixture_phrase(dest) and dest not in seen:
                seen.add(dest)
                result.append(dest)
        if al.startswith("open ") and is_heat_fixture_phrase(al[len("open ") :].strip()):
            target = al[len("open ") :].strip()
            if target not in seen:
                seen.add(target)
                result.append(target)

    for f in discover_fixtures_from_observation(ctx_text, heat=True):
        if f not in seen:
            seen.add(f)
            result.append(f)

    result.sort(key=_heat_fixture_rank)
    return result


def fixture_is_active(
    ctx_text: str,
    fixture: str,
    action_history=None,
) -> bool:
    """True when observation or recent history shows the fixture is on."""
    fl = re.escape((fixture or "").strip().lower())
    if not fl:
        return False
    ctx = (ctx_text or "").lower()
    if re.search(rf"\b{fl}\b[^.\n]*turned on", ctx):
        return True
    if re.search(rf"\b{fl}\b[^.\n]*(?:now )?activated", ctx):
        return True
    target_act = f"activate {(fixture or '').strip().lower()}"
    for act in reversed([(a or "").strip().lower() for a in (action_history or [])[-20:]]):
        if act == target_act:
            return True
    return False


def fixture_is_unavailable(
    fixture: str,
    action_history=None,
    failed_actions: set | None = None,
    ctx_text: str = "",
) -> bool:
    """True when activate has failed or the environment reports the fixture broken."""
    fl = (fixture or "").strip().lower()
    if not fl:
        return False
    ctx = (ctx_text or "").lower()
    if re.search(
        rf"\b{re.escape(fl)}\b[^.\n]*(?:broken|can't be activated|cannot be activated)",
        ctx,
    ):
        return True
    failed = failed_actions or set()
    return f"activate {fl}" in failed


def active_heat_fixtures(
    ctx_text: str,
    action_history=None,
    env_valid: set | None = None,
) -> list[str]:
    """Heat fixtures currently active according to observation/history."""
    return [
        f for f in heat_fixtures_in_environment(env_valid, ctx_text)
        if fixture_is_active(ctx_text, f, action_history)
    ]


def pick_next_heat_escalation_action(
    env_valid: set,
    recent: set,
    ctx_text: str,
    action_history=None,
    failed_actions: set | None = None,
    *,
    held_container: str = "",
) -> str | None:
    """
    When heating plateaus, try the next stronger/unused heat fixture:
    open → activate → move container onto fixture.
    """
    failed = failed_actions or set()
    fixtures = heat_fixtures_in_environment(env_valid, ctx_text)

    for fixture in fixtures:
        if fixture_is_unavailable(fixture, action_history, failed, ctx_text):
            continue
        open_act = f"open {fixture}"
        if open_act in env_valid and open_act not in recent and open_act not in failed:
            ctx_l = (ctx_text or "").lower()
            if re.search(rf"\b{re.escape(fixture.lower())}\b[^.\n]*closed", ctx_l):
                return open_act

    for fixture in fixtures:
        if fixture_is_unavailable(fixture, action_history, failed, ctx_text):
            continue
        if fixture_is_active(ctx_text, fixture, action_history):
            continue
        act = f"activate {fixture}"
        if act in env_valid and act not in recent and act not in failed:
            return act

    if held_container:
        for fixture in fixtures:
            if fixture_is_unavailable(fixture, action_history, failed, ctx_text):
                continue
            act = f"move {held_container} to {fixture}"
            if act in env_valid and act not in recent and act not in failed:
                return act

    return None


def recent_monitoring_action_count(action_history, window: int = 12) -> int:
    """Count examine / thermometer steps — signals post-focus heating plateau."""
    n = 0
    for act in reversed([(a or "").strip().lower() for a in (action_history or [])[-window:]]):
        if act.startswith(("examine ", "look at ")):
            n += 1
        elif act.startswith("use ") and "thermometer" in act:
            n += 1
    return n


def heating_progress_stalled(
    current_score: int | None,
    action_history=None,
    *,
    min_score: int = 38,
    max_score: int = 76,
    monitoring_threshold: int = 6,
) -> bool:
    """Generic: focused substance on heat but score not advancing."""
    score = int(current_score or 0)
    if score < min_score or score >= max_score:
        return False
    return recent_monitoring_action_count(action_history) >= monitoring_threshold


def fixture_visible_in_context(ctx_text: str, fixture: str) -> bool:
    fl = re.escape((fixture or "").strip().lower())
    return bool(fl and re.search(rf"\b{fl}\b", (ctx_text or "").lower()))


def container_visible_in_context(ctx_text: str, container: str) -> bool:
    return fixture_visible_in_context(ctx_text, container)


def current_room_from_context(ctx_text: str) -> str | None:
    ctx = (ctx_text or "").lower()
    m = re.search(r"(?:room|location) is called the ([a-z][a-z ]+)", ctx)
    if m:
        return m.group(1).strip().replace("green house", "greenhouse")
    return None


def fixture_last_seen_room(
    fixture: str,
    action_history=None,
    ctx_text: str = "",
) -> str | None:
    """Last room where fixture was visible or activated (from history + observation)."""
    fixture_l = (fixture or "").strip().lower()
    if not fixture_l:
        return None
    current = current_room_from_context(ctx_text)
    if current and fixture_visible_in_context(ctx_text, fixture_l):
        return current
    room: str | None = None
    for act in action_history or []:
        al = (act or "").strip().lower().replace("green house", "greenhouse")
        if al.startswith("go to "):
            room = al[len("go to ") :].strip()
        elif al.startswith("open door to "):
            room = al[len("open door to ") :].strip()
        elif al.startswith("activate ") and fixture_l in al and room:
            return room
    return room


def any_heat_fixture_in_context(ctx_text: str, env_valid: set | None = None) -> bool:
    return bool(heat_fixtures_in_environment(env_valid, ctx_text))


def primary_heat_fixture_unavailable(
    action_history=None,
    failed_actions: set | None = None,
    ctx_text: str = "",
    env_valid: set | None = None,
) -> bool:
    """True when every visible heat fixture is broken or failed to activate."""
    fixtures = heat_fixtures_in_environment(env_valid, ctx_text)
    if not fixtures:
        return fixture_is_unavailable("stove", action_history, failed_actions, ctx_text)
    usable = [
        f for f in fixtures
        if not fixture_is_unavailable(f, action_history, failed_actions, ctx_text)
    ]
    return not usable


def heat_destination_fixtures(
    env_valid: set | None,
    ctx_text: str,
    action_history=None,
    failed_actions: set | None = None,
) -> tuple[str, ...]:
    """Ordered heat destinations for move/pour (strongest available first)."""
    fixtures = heat_fixtures_in_environment(env_valid, ctx_text)
    if not fixtures:
        fixtures = ["stove", "oven"]
    out = [
        f for f in fixtures
        if not fixture_is_unavailable(f, action_history, failed_actions, ctx_text)
    ]
    return tuple(out) if out else ("oven",)


def rooms_to_search_for_missing_container(
    action_history=None,
    ctx_text: str = "",
) -> list[str]:
    """Rooms to try when a container is not in the current observation."""
    counts = room_visit_counts(action_history)
    doors = []
    try:
        from core.navigation_helpers import _extract_door_destinations
        doors = _extract_door_destinations(ctx_text)
    except Exception:
        pass
    ranked = sorted(
        doors,
        key=lambda r: (counts.get(r, 0), r),
    )
    return ranked
