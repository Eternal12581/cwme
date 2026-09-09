"""Extract legacy REUSE blocks from grounding_facade backup into reuse_policy_legacy.py."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "code" / "core" / "grounding_facade.py"
OUT = ROOT / "core" / "reuse_policy_legacy.py"


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"backup source missing: {SRC}")
    lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)

    helpers = "".join(lines[167:500])       # _pattern_force_repeat_ok .. _sci_cl_force_grounded
    methods = "".join(lines[595:1027])    # resolve_actor_override + _pattern_force_candidate
    resolve_body = "".join(lines[1088:1605])  # legacy body after mem_fast

    header = '''\
"""Legacy REUSE heuristics — CL-off baseline only.

Plateau, pattern-force, scheduled sources (heat_setup, transfer, connect, growth).
CWME mode (pattern read on) must not call this module.
"""
from __future__ import annotations

import re

from core.action_primitives import canonicalize_env_action
from core.cognition_router import CognitionRouter
from core.cl_protocol import is_action_blocked as _agent_action_blocked
from core.grounding_core import (
    env_valid_lookup as _env_valid_lookup,
    ground_pattern_milestone as _ground_pattern_ms,
)
from core.scienceworld_grounding import is_safe_task_pattern_action
from utils import action_verb_family, is_off_task_planned_action


def _raw_env_look(agent, env=None, fallback: str = "") -> str:
    env = env if env is not None else getattr(agent, "env", None)
    if env is not None:
        try:
            look = env.look()
            if look:
                return str(look)
        except Exception:
            pass
    return fallback or (getattr(agent, "observation", "") or "")


def _safe_task_pattern(act: str, task: str = "") -> bool:
    try:
        return bool(is_safe_task_pattern_action(act, task=task))
    except TypeError:
        try:
            return bool(is_safe_task_pattern_action(act))
        except Exception:
            return False
    except Exception:
        return False


def _pattern_read_enabled(agent) -> bool:
    plib = getattr(agent, "pattern_library", None)
    return plib is not None and bool(getattr(plib, "enable_read", False))


'''

    methods = methods.replace(
        "    def resolve_actor_override(\n        self,",
        "    @staticmethod\n    def resolve_actor_override(\n        facade,",
    )
    methods = methods.replace(
        "    def _pattern_force_candidate(\n        self,",
        "    @staticmethod\n    def _pattern_force_candidate(\n        facade,",
    )
    methods = methods.replace("        self.sanitize_route", "        facade.sanitize_route")

    resolve_sig = '''\
    @staticmethod
    def resolve_reuse(
        facade,
        orchestrator,
        agent,
        planned: str,
        env_valid: set,
        ctx_text: str,
        *,
        route,
        planned_norm: str,
        current_score: int,
        past: list,
        recent: set,
        failed: set,
        match_kind: str,
        raw_ms: str,
        steps_sg: int,
        task_for_cl: str,
        read_on: bool,
        score: int | None = None,
        logger=None,
        log_prefix: str = "REUSE",
    ):
        from core.grounding_facade import ExecutionDecision

'''

    body = helpers + "\nclass LegacyReusePolicy:\n" + methods + "\n" + resolve_sig + resolve_body

    body = body.replace("self._pattern_force_candidate", "LegacyReusePolicy._pattern_force_candidate")
    body = body.replace("self.resolve_actor_override", "LegacyReusePolicy.resolve_actor_override")
    body = body.replace("self._note_reuse", "facade._note_reuse")
    body = body.replace("self._finalize_decision", "facade._finalize_decision")

    footer = "\n\npattern_force_action_ok = _pattern_force_action_ok\n"
    OUT.write_text(header + body + footer, encoding="utf-8")
    print(f"Wrote {OUT} ({len((header + body + footer).splitlines())} lines)")


if __name__ == "__main__":
    main()
