"""Unit tests for CWME grounding-only execution path."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grounding.grounding_mode import GroundingMode, set_grounding_mode
from core.grounding_core import ground_candidate_action, ground_milestone_env_only
from core.scienceworld_grounding import (
    _task_family_flags,
    ground_pattern_action,
    is_safe_task_pattern_action,
)
from core.scienceworld_policy import sci_action_allowed, sci_focus_action_allowed
from core.world_model.execution_audit import CWMEExecutionAudit, get_execution_audit


@pytest.fixture(autouse=True)
def grounding_only_mode():
    set_grounding_mode(GroundingMode.GROUNDING_ONLY)
    audit = get_execution_audit()
    audit.reset()
    audit.strict = False
    yield
    audit.reset()


def test_cwme_does_not_use_task_heuristics_in_pattern_safety():
    audit = get_execution_audit()
    assert is_safe_task_pattern_action("pick up", task="freeze")
    assert is_safe_task_pattern_action("activate", task="test-conductivity")
    assert not is_safe_task_pattern_action("go to bedroom", task="boil")
    assert audit.task_heuristic_calls == 0


def test_grounding_only_allows_context_supported_navigation_only():
    assert is_safe_task_pattern_action(
        "go to kitchen",
        task="boil water",
        observation="You are in the hallway. A door to the kitchen is visible.",
    )
    assert not is_safe_task_pattern_action(
        "go to bedroom",
        task="boil water",
        observation="You are in the hallway. A door to the kitchen is visible.",
    )


def test_actor_safety_gate_rejects_legal_but_off_task_focus():
    task = "Boil water in the kitchen."
    context = "This room is called the kitchen. A stove and water are here."
    assert not sci_action_allowed(
        "focus on orange juice",
        task,
        env_valid={"focus on orange juice"},
        task_aware=True,
        ctx_text=context,
    )
    assert not sci_action_allowed(
        "focus on sodium chloride",
        task,
        env_valid={"focus on sodium chloride"},
        task_aware=True,
        ctx_text=context,
    )


def test_final_grounding_gate_does_not_reuse_an_unrelated_named_focus():
    """A stale pattern must not authorize its own terminal focus target."""
    from core.scienceworld_grounding import is_safe_grounding_only_execution_action

    boil_task = "Boil lead. First, focus on the substance."
    assert not is_safe_grounding_only_execution_action(
        "focus on thermometer",
        task=boil_task,
        observation="This room is called the kitchen. A thermometer is here.",
        pattern_evidence=["focus on thermometer: unobserved -> focused"],
    )
    assert is_safe_grounding_only_execution_action(
        "focus on substance in metal pot",
        task=boil_task,
        observation="A metal pot containing the substance is here.",
    )
    assert is_safe_grounding_only_execution_action(
        "focus on thermometer",
        task="Use a thermometer to measure the temperature.",
        observation="A thermometer is here.",
    )


def test_substance_first_rejects_bare_vessel_focus():
    task = "Boil water. First, focus on the substance."
    assert not sci_focus_action_allowed(
        "focus on metal pot",
        task,
        ctx_text="A metal pot and a tin substance are here.",
    )
    assert sci_focus_action_allowed(
        "focus on substance in metal pot",
        task,
        ctx_text="A metal pot containing the substance is here.",
    )


def test_repeated_planner_observation_is_not_returned_in_formal_mode():
    from core.grounding_facade import GroundingFacade

    class FakeEnv:
        env_name = "scienceworld"

        def getValidActionObjectCombinations(self):
            return {"look around", "activate stove", "focus on orange juice"}

        def look(self):
            return "This room is called the kitchen. A stove is here."

    agent = SimpleNamespace(
        agent_config={
            "EXECUTION": {
                "grounding_mode": "grounding_only",
                "legacy_heuristics_enabled": False,
            },
        },
        task="Boil water in the kitchen.",
        observation="This room is called the kitchen.",
        inventory="",
        past_actions=["<START>", "look around"],
        env=FakeEnv(),
        logger=None,
        pattern_library=None,
        recent_failed_actions=set(),
        _last_finalized_decision_key=None,
    )
    route = SimpleNamespace(
        pattern_match_kind="",
        pattern_milestone="",
        grounded_suggestion="",
        suggestion_source="",
        prefer_grounded=False,
        steps_since_gain=0,
        learning_plateau=False,
        som_sub_phase="",
        phase="",
    )
    orchestrator = SimpleNamespace(
        route_cognition=lambda *args, **kwargs: route,
        try_memory_guided_fast=lambda *args, **kwargs: None,
    )
    decision = GroundingFacade().resolve_trajectory_step(
        orchestrator,
        agent,
        "look around",
        agent.env,
        ctx_text=agent.observation,
        score=0,
    )
    assert decision is None
    unsafe_plan = GroundingFacade().resolve_trajectory_step(
        orchestrator,
        agent,
        "focus on orange juice",
        agent.env,
        ctx_text=agent.observation,
        score=0,
    )
    assert unsafe_plan is None


def test_task_family_flags_blocked_in_grounding_only():
    audit = get_execution_audit()
    flags = _task_family_flags("freeze")
    assert all(v is False for v in flags.values())
    assert audit.suppressed_task_heuristic_checks >= 1
    assert audit.blocked_task_heuristic_calls == 0
    assert audit.task_heuristic_calls == 0


def test_ground_pattern_action_env_only():
    valid = {"pick up red substance", "activate stove", "go to kitchen"}
    g = ground_pattern_action("pick up", valid, task="boil")
    assert g in valid
    g2 = ground_pattern_action("activate", valid, task="freeze")
    assert g2 == "activate stove"


def test_env_only_grounding_filters_unsafe_concrete_candidates():
    valid = {"activate sink", "activate stove", "focus on cup", "focus on thermometer"}
    assert ground_milestone_env_only(
        "activate", valid, task="boil", environment="scienceworld",
    ) == "activate stove"
    assert ground_milestone_env_only(
        "focus on", valid, task="use thermometer", environment="scienceworld",
    ) == "focus on thermometer"


def test_ground_milestone_env_only_no_task_filter():
    valid = {"pour water into green cup", "mix ingredients"}
    assert ground_milestone_env_only(
        "pour", valid, task="grow-plant", environment="scienceworld",
    ) in valid


def test_grounding_only_does_not_replace_concrete_target_by_verb_only():
    assert ground_milestone_env_only(
        "activate stove",
        {"activate microwave"},
        task="boil water", environment="scienceworld",
    ) == ""
    assert ground_milestone_env_only(
        "go to kitchen",
        {"go to hallway"},
        task="boil water", environment="scienceworld",
    ) == ""


def test_empty_action_snapshot_never_leaks_stored_pattern():
    assert ground_milestone_env_only(
        "pick up sewer", set(), task="grow-plant", environment="scienceworld",
    ) == ""
    assert ground_candidate_action("pick up sewer", set()) == ""
    assert ground_pattern_action("pick up sewer", set(), task="grow-plant") == ""


def test_grounding_only_rejects_executable_but_off_task_science_probes():
    # ScienceWorld lists these commands as admissible, but they are room
    # fixture / vague-vessel probes rather than transferable milestones.
    assert not is_safe_task_pattern_action("activate sink", task="boil")
    assert not is_safe_task_pattern_action("activate toilet", task="melt")
    assert not is_safe_task_pattern_action("activate bathtub", task="freeze")
    assert not is_safe_task_pattern_action("focus on cup", task="boil")
    assert not is_safe_task_pattern_action("focus on air", task="use thermometer")
    assert is_safe_task_pattern_action("activate stove", task="boil")
    assert is_safe_task_pattern_action("focus on thermometer", task="use thermometer")


def test_grounding_only_rejects_legal_fixture_process_probes():
    # These commands are legal in ScienceWorld, but are not a task-directed
    # recovery for a boil episode and can produce the terminal -100 answer.
    assert not is_safe_task_pattern_action(
        "mix art studio",
        task="Boil tin. First, focus on the substance.",
        observation="This room is called the art studio.",
    )
    assert not is_safe_task_pattern_action(
        "pour cup into toilet",
        task="Boil tin. First, focus on the substance.",
        observation="This room is called the bathroom. A toilet is here.",
    )
    assert not is_safe_task_pattern_action(
        "focus on substance in toilet",
        task="Boil tin. First, focus on the substance.",
        observation="The toilet contains water.",
    )


def test_pattern_navigation_can_match_door_primitive_after_transition():
    from core.cwme_agent_hooks import _pattern_action_matches

    assert _pattern_action_matches(
        "go to hallway",
        "go to door",
        state_after="You move through the door to the hallway.",
    )
    assert not _pattern_action_matches(
        "go to hallway",
        "go to door",
        state_after="You move through the door to the bathroom.",
    )


def test_grounding_only_allows_context_supported_fixtures_and_vessels():
    assert is_safe_task_pattern_action(
        "activate sink",
        task="Clean the substance using the sink.",
        observation="The sink is here.",
    )
    assert is_safe_task_pattern_action(
        "focus on cup",
        task="Boil the water in the cup.",
        observation="A cup containing water is here.",
    )
    assert is_safe_task_pattern_action(
        "activate bathtub",
        task="Activate the bathtub.",
        observation="The bathtub is here.",
    )
    assert is_safe_task_pattern_action(
        "focus on jug",
        task="Boil water.",
        past_actions=["pour water into jug"],
    )


def test_grounding_only_allows_pattern_transition_evidence():
    assert is_safe_task_pattern_action(
        "activate sink",
        task="Continue the procedure.",
        pattern_evidence=["activate sink: dry -> wet"],
    )
    assert not is_safe_task_pattern_action(
        "activate sink",
        task="Continue the procedure.",
        pattern_evidence=["activate stove: cold -> hot"],
    )


def test_unknown_action_snapshot_remains_explicit_opt_out():
    assert ground_milestone_env_only(
        "pick up", None, task="grow-plant", environment="scienceworld",
    ) == "pick up"


def test_grounding_core_does_not_apply_science_rules_to_alfworld():
    valid = {"activate sink", "activate stove"}
    assert ground_milestone_env_only(
        "activate sink", valid, task="clean an object", environment="alfworld",
    ) == "activate sink"


def test_execution_router_grounds_abstract_milestone_with_live_context():
    """A legal distractor must not suppress a usable fast-path candidate."""
    from core.execution.execution_router import ExecutionRouter
    from core.procedural.pattern import ProceduralPattern

    pattern = ProceduralPattern.create(
        task_signature="boil",
        actions=["activate"],
        score=100,
        task_id="boil",
    )
    pattern.confidence = 0.95
    pattern.status = "active"
    pattern.outcome = "success"

    class Library:
        enable_read = True
        last_pattern_confidence = 0.95

        def get_active_procedural(self):
            return pattern

    class Env:
        env_name = "scienceworld"

    agent = SimpleNamespace(
        env=Env(),
        pattern_library=Library(),
        cwme_cfg=SimpleNamespace(
            fast_path_enabled=True,
            pattern_verify_enabled=True,
            pattern_reuse_mode="hard",
        ),
        episode_guard=None,
        past_actions=[],
        agent_config={"AGENT": {"ENVIRONMENT": "ScienceWorld"}},
    )
    route = ExecutionRouter().route(
        agent,
        milestone="activate",
        match_kind="exact",
        env_valid={"activate sink", "activate stove"},
        task="Boil water in the kitchen.",
        observation="This room is called the kitchen. A stove is here.",
    )
    assert route.grounded_action == "activate stove"
    assert route.fast_allowed


def test_grounding_only_keeps_abstract_process_milestones_available():
    assert is_safe_task_pattern_action("activate", task="freeze")
    assert is_safe_task_pattern_action("pick up", task="boil")
    assert is_safe_task_pattern_action("use", task="use thermometer")


def test_state_packet_does_not_inject_task_rules_in_formal_mode():
    from core.state_packet import build_state_packet

    class ForbiddenEngine:
        def plan(self, *args, **kwargs):
            raise AssertionError("task-specific phase engine was called")

    class ForbiddenPhase:
        def snapshot(self, *args, **kwargs):
            raise AssertionError("task-specific phase controller was called")

    agent = SimpleNamespace(
        agent_config={"EXECUTION": {"grounding_mode": "grounding_only"}},
        task="Boil water in the kitchen.",
        observation="This room is called the kitchen.",
        inventory="",
        past_actions=[],
        last_score=0,
        episode_progress={"score_max": 0},
        pre_focus=ForbiddenEngine(),
        post_focus=ForbiddenEngine(),
        orchestrator=SimpleNamespace(phase=ForbiddenPhase()),
        env=None,
        _wm_subgraph=[],
        _wm_episodic=[],
        past_reflections="",
        pattern_library=None,
        hkg_retriever=None,
    )
    packet = build_state_packet(agent)
    assert packet.phase == "observe"
    assert packet.focus_hint == ""
    assert packet.pre_focus_suggestion == ""
    assert packet.post_focus_suggestion == ""
    assert not packet.transfer_pending
    assert not packet.heat_setup_pending


def test_formal_route_sanitization_does_not_call_task_policy():
    from core.grounding_facade import GroundingFacade

    class FakeEnv:
        env_name = "scienceworld"

        def getValidActionObjectCombinations(self):
            return {"activate stove"}

        def look(self):
            return "This room is called the kitchen."

    class ForbiddenPolicy:
        def check_routed(self, *args, **kwargs):
            raise AssertionError("legacy ActionPolicyEngine was called")

    agent = SimpleNamespace(
        agent_config={"EXECUTION": {"grounding_mode": "grounding_only"}},
        task="Boil water in the kitchen.",
        observation="This room is called the kitchen.",
        inventory="",
        past_actions=[],
        env=FakeEnv(),
        logger=None,
    )
    facade = GroundingFacade(policy=ForbiddenPolicy())
    valid_route = SimpleNamespace(
        grounded_suggestion="activate stove",
        suggestion_source="pattern",
        prefer_grounded=True,
    )
    facade.sanitize_route(agent, valid_route)
    assert valid_route.grounded_suggestion == "activate stove"
    assert valid_route.prefer_grounded is True

    invalid_route = SimpleNamespace(
        grounded_suggestion="activate sink",
        suggestion_source="pattern",
        prefer_grounded=True,
    )
    facade.sanitize_route(agent, invalid_route)
    assert invalid_route.grounded_suggestion == ""
    assert invalid_route.suggestion_source == ""
    assert invalid_route.prefer_grounded is False


def test_heuristic_mode_uses_task_family():
    set_grounding_mode(GroundingMode.TASK_HEURISTIC)
    audit = get_execution_audit()
    audit.reset()
    from core.scienceworld_grounding import is_safe_task_pattern_action_heuristic
    result = is_safe_task_pattern_action_heuristic("connect", task="boil")
    assert result is False


def test_execution_audit_valid_cwme_run():
    audit = CWMEExecutionAudit()
    assert audit.is_valid_cwme_run()
    audit.record_task_heuristic("test")
    assert not audit.is_valid_cwme_run()
    assert audit.to_dict()["task_heuristic_calls"] == 1
