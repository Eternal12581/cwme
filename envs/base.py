"""Shared helpers for multi-environment support."""

from __future__ import annotations


def env_name(env) -> str:
    """Return lowercase env tag: 'scienceworld' | 'alfworld' | ..."""
    name = getattr(env, "env_name", None)
    if name:
        return str(name).strip().lower()
    # Legacy ScienceWorldEnv instances have no tag.
    cls = type(env).__name__.lower()
    if "alfworld" in cls or "alfred" in cls:
        return "alfworld"
    return "scienceworld"


def is_scienceworld(env) -> bool:
    return env_name(env) == "scienceworld"


def is_alfworld(env) -> bool:
    return env_name(env) == "alfworld"
