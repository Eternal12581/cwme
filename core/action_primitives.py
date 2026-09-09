"""
Action tokenization, verb families, and context grounding primitives.
"""
from __future__ import annotations

import re

from core.scienceworld_schema import *  # noqa: F403

def _action_content_tokens(action: str) -> set[str]:
    text = (action or "").lower().replace("green house", "greenhouse")
    words = re.findall(r"[a-z0-9]+", text)
    return {w for w in words if w not in _ACTION_STOPWORDS and len(w) > 1}


action_content_tokens = _action_content_tokens


def _labeled_entity_tokens(phrase: str) -> set[str]:
    """Content tokens plus single-letter ScienceWorld labels (substance U, material D)."""
    toks = set(_action_content_tokens(phrase))
    for m in re.finditer(r"\b([a-z0-9])\b", (phrase or "").lower()):
        toks.add(m.group(1))
    return toks


def canonicalize_env_action(act: str, env_valid) -> str:
    """Case-insensitive lookup; returns the env's canonical admissible string."""
    a = (act or "").strip()
    if not a:
        return ""
    if not env_valid:
        return a
    lower_map = {
        (str(v) or "").strip().lower(): (str(v) or "").strip()
        for v in env_valid
        if v
    }
    return lower_map.get(a.lower(), a)


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


_PHRASE_ALIASES = {
    "fridge": "refrigerator",
    "refrigerator": "fridge",
    "bookshelf": "book shelf",
    "book shelf": "bookshelf",
    "cupboard": "cup board",
}


def _ctx_contains_phrase(ctx_text: str, phrase: str) -> bool:
    """Fuzzy check: 'bookshelf' matches 'book shelf' in observation."""
    ctx = (ctx_text or "").lower()
    phrase = (phrase or "").lower().strip().replace("green house", "greenhouse")
    if not phrase:
        return True
    if phrase in ctx:
        return True
    if _collapse_spaces(phrase) in _collapse_spaces(ctx):
        return True
    alt = _PHRASE_ALIASES.get(phrase)
    if alt and (alt in ctx or _collapse_spaces(alt) in _collapse_spaces(ctx)):
        return True
    tokens = _action_content_tokens(phrase)
    return bool(tokens) and all(tok in ctx for tok in tokens)


def extract_task_content_tokens(task: str) -> set[str]:
    text = (task or "").lower()
    words = re.findall(r"[a-z0-9]+", text)
    return {w for w in words if w not in _TASK_STOPWORDS and len(w) > 2}


def action_verb_family(action: str) -> str:
    a = (action or "").lower().strip()
    if a.startswith("focus on"):
        return "focus"
    if a.startswith("examine"):
        return "examine"
    if a.startswith("look at"):
        return "look_at"
    if a.startswith("look in"):
        return "look_in"
    if a.startswith("look around"):
        return "look_around"
    if a.startswith("pick up"):
        return "pick_up"
    if a.startswith("take "):
        return "take"
    if a.startswith("put "):
        return "put"
    if a.startswith("open door") or a.startswith("teleport to") or a.startswith("move"):
        return "move"
    if a.startswith("open "):
        return "open"
    if a.startswith("pour"):
        return "pour"
    if a.startswith("mix"):
        return "mix"
    if a.startswith("eat "):
        return "eat"
    if a.startswith("drink "):
        return "drink"
    if a.startswith("connect"):
        return "connect"
    if a.startswith("disconnect"):
        return "disconnect"
    if a.startswith("use "):
        return "use"
    if a.startswith("activate") or a.startswith("turn on"):
        return "activate"
    if a.startswith("deactivate") or a.startswith("turn off"):
        return "deactivate"
    if a == "wait" or (a.startswith("wait") and a[4:].isdigit()):
        # ScienceWorld emits wait1 / wait2; treat as wait family.
        return "wait"
    if a in ("inventory", "task"):
        return "meta"
    return "other"


