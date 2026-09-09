"""Sanity checks for CWME grounding/execution split (no heavy env imports)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._bootstrap_utils_lite import bootstrap_utils_lite

bootstrap_utils_lite()

from core.cl_protocol import cwme_pattern_only_mode, grounding_only_mode
from core.execution.memory_fast import legacy_reuse_heuristics_enabled
from core.grounding_core import ensure_env_valid, env_valid_lookup
from core.execution_pipeline import ExecutionPipeline


class _Agent:
    def __init__(self, *, read: bool, grounding_only: bool = False):
        self.pattern_library = _Plib(read)
        self.agent_config = {
            "EXECUTION": {
                "grounding_mode": "grounding_only" if grounding_only else "legacy",
                "legacy_heuristics_enabled": not grounding_only,
                "task_heuristics_enabled": False,
            },
        }
        if grounding_only:
            from core.grounding.grounding_mode import GroundingMode, set_grounding_mode
            set_grounding_mode(GroundingMode.GROUNDING_ONLY)


class _Plib:
    def __init__(self, read: bool):
        self.enable_read = read


class _Orch:
    def phase_snapshot(self, agent, score=None):
        return _Snap()

    def route_cognition(self, agent, env_valid=None, score=None):
        return _Route()


class _Snap:
    heat_setup_pending = True
    transfer_pending = True
    transfer_executable = True
    current_room = "kitchen"
    foc_room = "kitchen"
    phase = "manipulation"


class _Route:
    grounded_suggestion = "activate oven"
    suggestion_source = "heat_setup"


def main() -> None:
    cwme = _Agent(read=True, grounding_only=True)
    base = _Agent(read=False, grounding_only=True)
    legacy = _Agent(read=False, grounding_only=False)
    assert cwme_pattern_only_mode(cwme)
    assert not cwme_pattern_only_mode(base)
    assert grounding_only_mode(cwme)
    assert grounding_only_mode(base)
    assert not legacy_reuse_heuristics_enabled(cwme)
    assert not legacy_reuse_heuristics_enabled(base)
    assert legacy_reuse_heuristics_enabled(legacy)

    valid = {"activate oven", "go to kitchen"}
    assert env_valid_lookup("ACTIVATE OVEN", valid) == "activate oven"
    assert ensure_env_valid("ACTIVATE OVEN", valid) == "activate oven"
    assert ensure_env_valid("invalid cmd", valid) == ""

    pipe = ExecutionPipeline()
    final, source, _ = pipe.reroute(
        cwme,
        "look around",
        orchestrator=_Orch(),
        env_valid=valid,
        ctx_text="",
        task="melt",
        past_actions=[],
        current_score=0,
        scheduled=None,
    )
    assert final == "look around"
    assert source == "find_valid", "CWME mode must not apply heat_setup/transfer reroutes"

    print("Grounding split sanity checks passed.")


if __name__ == "__main__":
    main()
