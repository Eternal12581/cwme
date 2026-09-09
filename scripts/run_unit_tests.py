#!/usr/bin/env python3
"""Lightweight unit test runner (no pytest required)."""
from __future__ import annotations

import sys
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.experiment_protocol import cwme_no_heuristic_mode
from core.grounding.grounding_mode import GroundingMode, set_grounding_mode
from core.grounding_core import ground_milestone_env_only
from core.scienceworld_grounding import (
    _task_family_flags,
    ground_pattern_action,
    is_safe_task_pattern_action,
    is_safe_task_pattern_action_heuristic,
)
from core.world_model.execution_audit import CWMEExecutionAudit, get_execution_audit
from core.execution.fast_executor import FastExecutor
from core.pattern_library import PatternLibrary
from core.evolution.transition_tracker import TransitionTracker
from core.procedural.pattern import ProceduralPattern
from core.procedural.pattern_reviser import PatternReviser


def _check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(name)
    print(f"  PASS: {name}")


def main() -> None:
    print("=== Unit tests ===")
    set_grounding_mode(GroundingMode.GROUNDING_ONLY)
    audit = get_execution_audit()
    audit.reset()

    _check("pattern safety no heuristics", is_safe_task_pattern_action("pick up", task="freeze"))
    _check("pattern safety blocks nav", not is_safe_task_pattern_action("go to bedroom", task="boil"))
    _check(
        "grounding blocks off-task room process",
        not is_safe_task_pattern_action(
            "mix art studio",
            task="Boil tin. First, focus on the substance.",
            observation="This room is called the art studio.",
        ),
    )
    _check(
        "grounding blocks toilet transfer",
        not is_safe_task_pattern_action(
            "pour cup into toilet",
            task="Boil tin. First, focus on the substance.",
            observation="This room is called the bathroom. A toilet is here.",
        ),
    )
    _check(
        "grounding blocks toilet focus",
        not is_safe_task_pattern_action(
            "focus on substance in toilet",
            task="Boil tin. First, focus on the substance.",
            observation="The toilet contains water.",
        ),
    )
    from core.cwme_agent_hooks import _pattern_action_matches
    _check(
        "pattern navigation matches door transition",
        _pattern_action_matches(
            "go to hallway",
            "go to door",
            state_after="You move through the door to the hallway.",
        ),
    )
    _check(
        "pattern navigation rejects wrong door transition",
        not _pattern_action_matches(
            "go to hallway",
            "go to door",
            state_after="You move through the door to the bathroom.",
        ),
    )
    _check("audit clean after safety", audit.task_heuristic_calls == 0)

    flags = _task_family_flags("freeze")
    _check("task family blocked", all(v is False for v in flags.values()))
    _check("suppressed task check recorded", audit.suppressed_task_heuristic_checks >= 1)
    _check("no blocked heuristic violation", audit.blocked_task_heuristic_calls == 0)

    valid = {"pick up red substance", "activate stove", "go to kitchen"}
    _check("ground pattern env only", ground_pattern_action("pick up", valid, task="boil") in valid)
    _check(
        "ground milestone env only",
        ground_milestone_env_only(
            "pour", {"pour water into green cup"}, task="x", environment="scienceworld",
        ) != "",
    )
    _check(
        "formal transfer evidence blocks arbitrary sibling",
        ground_milestone_env_only(
            "move",
            {"move cup to sink", "move bowl to oven"},
            task="change state",
            environment="scienceworld",
            pattern_evidence=["current_pattern_action: move metal pot to freezer"],
        ) == "",
    )
    _check(
        "grounding keeps concrete activate target",
        ground_milestone_env_only(
            "activate stove", {"activate microwave"}, task="boil water",
            environment="scienceworld",
        ) == "",
    )
    _check(
        "grounding keeps concrete navigation target",
        ground_milestone_env_only(
            "go to kitchen", {"go to hallway"}, task="boil water",
            environment="scienceworld",
        ) == "",
    )

    from core.execution.execution_router import ExecutionRouter

    class _ScienceEnv:
        env_name = "scienceworld"

    class _PatternStore:
        enable_read = True
        last_pattern_confidence = 0.95

        def get_active_procedural(self):
            return source_pattern

    source_pattern = ProceduralPattern.create(
        task_signature="boil",
        task_id="boil",
        actions=["activate"],
        score=100,
    )
    source_pattern.confidence = 0.95
    source_pattern.status = "active"
    source_pattern.outcome = "success"
    router_agent = SimpleNamespace(
        env=_ScienceEnv(),
        pattern_library=_PatternStore(),
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
        router_agent,
        milestone="activate",
        match_kind="exact",
        env_valid={"activate sink", "activate stove"},
        task="Boil water in the kitchen.",
        observation="This room is called the kitchen. A stove is here.",
    )
    _check(
        "router selects contextual action over legal distractor",
        route.grounded_action == "activate stove" and route.fast_allowed,
    )

    # An abstract source milestone is a cross-variation transfer and must use
    # the verifier before execution. This guards against the old state where
    # Pattern retrieval was visible in logs but never reached the 4B model.
    transfer_route = ExecutionRouter().route(
        router_agent,
        milestone="activate",
        match_kind="exact",
        env_valid={"activate stove"},
        task="Boil water in the kitchen.",
        observation="This room is called the kitchen. A stove is here.",
        pattern_confidence=0.95,
        pattern_id="p-transfer",
    )
    _check(
        "abstract cross-variation action requires verification",
        transfer_route.level.name == "VERIFIED_REUSE"
        and transfer_route.grounded_action == "activate stove",
    )

    class _VerifyRouter:
        def __init__(self):
            self.calls = 0

        def call(self, task, prompt, json=False, schema=None):
            assert task == "pattern_verify"
            self.calls += 1
            return {"approve": True, "reason": "test"}, {"sent": 1, "received": 1}

    router_agent.router = _VerifyRouter()
    transfer_decision = FastExecutor().execute(
        router_agent,
        transfer_route,
        env_valid={"activate stove"},
        task="Boil water in the kitchen.",
        observation="This room is called the kitchen. A stove is here.",
    )
    _check(
        "verified reuse calls the 4B verifier",
        transfer_decision is not None
        and transfer_decision.source == "pattern_verify"
        and router_agent.router.calls == 1,
    )

    router_agent.cwme_cfg.pattern_verify_enabled = False
    no_verify_route = ExecutionRouter().route(
        router_agent,
        milestone="activate",
        match_kind="exact",
        env_valid={"activate stove"},
        task="Boil water in the kitchen.",
        observation="This room is called the kitchen. A stove is here.",
        pattern_confidence=0.95,
        pattern_id="p-transfer",
    )
    _check(
        "disabled verifier falls back to cognition",
        no_verify_route.level.name == "FULL_COGNITION"
        and no_verify_route.reason == "verify_disabled",
    )

    # Full trajectories are ordered procedures: a successful action advances
    # one cursor position, while a failed action cools only that position.
    from core.procedural.milestone_selector import MilestoneSelector
    from core.world_model.historical_kg import HistoricalKG
    from core.world_model.kg_retriever import HistoricalKGRetriever
    cursor_pattern = ProceduralPattern.create(
        task_signature="boil", task_id="boil",
        actions=["activate stove", "focus on water"], score=100,
    )
    cursor_pattern.confidence = 0.95
    cursor_pattern.status = "active"
    cursor_pattern.outcome = "success"
    cursor_lib = PatternLibrary()
    cursor_lib.enable_read = True
    cursor_lib._train_procedural["boil::boil"] = [cursor_pattern]
    cursor_kg = HistoricalKG()
    cursor_kg.metadata["episode_count"] = 1
    cursor_kg.add_edge(
        "activate(stove)", "precedes", "focus(water)", episode_id=0,
    )
    selector = MilestoneSelector()
    first = selector.select(
        cursor_lib, "boil", ["<START>"], task_id="boil",
        env_valid={"activate stove"}, observation="A stove is here.",
        environment="scienceworld", historical_kg=cursor_kg,
        hkg_retriever=HistoricalKGRetriever(),
    )
    _check("ordered pattern selects first stage", first == "activate stove")
    first_idx = cursor_lib.last_pattern_stage_index
    cursor_lib.note_pattern_result(cursor_pattern.pattern_id, first_idx, success=True)
    second = selector.select(
        cursor_lib, "boil", ["<START>", "activate stove"], task_id="boil",
        env_valid={"focus on water"}, observation="A body of water is here.",
        environment="scienceworld", historical_kg=cursor_kg,
        hkg_retriever=HistoricalKGRetriever(),
    )
    _check(
        "ordered pattern advances after environment progress",
        second == "focus on water" and cursor_lib.active_pattern_cursor == 1,
    )
    # A procedure selected earlier in the episode must not be replaced by a
    # sibling same-task trajectory when its next stage is unavailable.
    sibling_pattern = ProceduralPattern.create(
        task_signature="boil", task_id="boil",
        actions=["go to hallway"], score=100,
    )
    sibling_pattern.confidence = 0.99
    sibling_pattern.status = "active"
    sibling_pattern.outcome = "success"
    locked_lib = PatternLibrary()
    locked_lib.enable_read = True
    locked_lib._train_procedural["boil::boil"] = [cursor_pattern, sibling_pattern]
    locked_lib.begin_pattern_cursor(cursor_pattern.pattern_id)
    locked_lib.active_pattern_cursor = 1
    locked_next = selector.select(
        locked_lib, "boil", ["<START>", "activate stove"], task_id="boil",
        env_valid={"go to hallway"}, observation="A hallway door is visible.",
        environment="scienceworld",
    )
    _check(
        "ordered pattern does not splice sibling trajectory",
        locked_next == "",
    )
    cursor_lib.note_pattern_result(cursor_pattern.pattern_id, 1, success=False)
    _check(
        "failed pattern stage is cooled",
        cursor_lib.is_pattern_stage_failed(cursor_pattern.pattern_id, 1),
    )

    # Durable memory keeps verified rewards and enabling effects only.  A
    # changed response string, an already-satisfied action, or an invalid
    # action must not become a reusable procedure step.
    tracker = TransitionTracker()
    enabling = tracker.record(
        state_before="You are in hallway.",
        action="go to kitchen",
        state_after="You move to the kitchen.",
        score_before=0,
        score_after=0,
        env_response="You move to the kitchen.",
    )
    noop = tracker.record(
        state_before="The drawer is open.",
        action="open drawer",
        state_after="The drawer is already open.",
        score_before=0,
        score_after=0,
        env_response="The drawer is already open.",
    )
    failure = tracker.record(
        state_before="kitchen",
        action="mix art studio",
        state_after="Nothing happens.",
        score_before=0,
        score_after=-100,
        env_response="Invalid action: cannot mix art studio.",
    )
    _check(
        "transition classifier keeps enabling action",
        enabling.progress_kind == "enabling"
        and enabling.state_signature_after == "agent.location=kitchen",
    )
    _check("transition classifier drops no-op", noop.progress_kind == "no_op")
    _check(
        "transition classifier records failure",
        failure.progress_kind == "failure" and not failure.valid,
    )
    from core.procedural.pattern_extractor import PatternExtractor
    compact_pattern = PatternExtractor().extract(
        trajectory=[enabling, noop, failure],
        task="go to kitchen",
        task_id="navigate",
        score=0,
        full_trajectory=True,
    )
    _check(
        "pattern contains only verified progress",
        compact_pattern is not None
        and compact_pattern.actions == ["go to kitchen"]
        and compact_pattern.action_metadata[0]["progress_kind"] == "enabling",
    )

    from core.cwme_agent_hooks import on_agent_step_after_env
    hook_agent = SimpleNamespace(
        pattern_library=cursor_lib,
        transition_tracker=TransitionTracker(),
        execution_metrics=SimpleNamespace(pattern_stage_advance=0, pattern_recovery=0),
        _score_hist=[],
        _last_pattern_execution={
            "pattern_id": cursor_pattern.pattern_id,
            "stage_index": 0,
            "milestone": "activate stove",
            "action": "activate stove",
        },
    )
    on_agent_step_after_env(
        hook_agent,
        state_before="kitchen",
        action="activate stove",
        state_after="kitchen stove active",
        score_before=0,
        score_after=0,
        env_response="The stove is now active.",
    )
    _check(
        "environment progress advances pattern cursor",
        cursor_lib.active_pattern_cursor >= 1
        and hook_agent.execution_metrics.pattern_stage_advance == 1,
    )
    cursor_lib.active_pattern_cursor = 0
    hook_agent._last_pattern_execution = {
        "pattern_id": cursor_pattern.pattern_id,
        "stage_index": 0,
        "milestone": "activate stove",
        "action": "activate stove",
    }
    on_agent_step_after_env(
        hook_agent,
        state_before="kitchen",
        action="activate stove",
        state_after="kitchen",
        score_before=0,
        score_after=0,
        env_response="Invalid action: cannot activate the stove.",
    )
    _check(
        "failed environment action does not advance pattern",
        cursor_lib.active_pattern_cursor == 0
        and hook_agent.execution_metrics.pattern_recovery == 1,
    )

    # Formal replan control must not consult the legacy task/phase heuristic.
    import core.orchestrator as orchestrator_module
    from core.orchestrator import Orchestrator

    original_replan_heuristic = orchestrator_module.should_replan_heuristic
    orchestrator_module.should_replan_heuristic = lambda *args, **kwargs: (
        (_ for _ in ()).throw(AssertionError("formal replan heuristic leaked"))
    )
    try:
        replan = Orchestrator().evaluate_replan(
            {}, {}, 0, 0, 1, False, 0, "look", "look",
            {}, {"score_max": 0, "steps_since_gain": 1},
            agent=SimpleNamespace(
                agent_config={"EXECUTION": {"grounding_mode": "grounding_only"}},
                last_exec_source="find_valid",
            ),
        )
    finally:
        orchestrator_module.should_replan_heuristic = original_replan_heuristic
    _check("formal replan skips task heuristic", replan.reason == "formal_grounding_only")
    replan_after_stagnation = Orchestrator().evaluate_replan(
        {}, {}, 0, 0, 5, False, 0, "look", "look",
        {}, {"score_max": 0, "steps_since_gain": 5,
             "steps_since_relevant_state_change": 5},
        agent=SimpleNamespace(
            agent_config={"EXECUTION": {"grounding_mode": "grounding_only"}},
            last_exec_source="find_valid",
        ),
    )
    _check(
        "formal stagnation opens cognition recovery",
        replan_after_stagnation.should_replan
        and replan_after_stagnation.reason == "formal_effect_stagnant",
    )

    from core.substitute_engine import _ground_cwme_planned_action
    alf_valid = {"put magazine in diningtable 1", "take magazine from bed 1"}
    _check(
        "alfworld move surface grounding",
        _ground_cwme_planned_action(
            ["move magazine to diningtable 1"],
            alf_valid,
            allow_alfworld_surface_matching=True,
        ) == "put magazine in diningtable 1",
    )
    _check(
        "alfworld take surface grounding",
        _ground_cwme_planned_action(
            ["pick up magazine from bed 1"],
            alf_valid,
            allow_alfworld_surface_matching=True,
        ) == "take magazine from bed 1",
    )
    _check(
        "grounding rejects unknown actor action",
        _ground_cwme_planned_action(
            ["pick up toiletpaperhanger 1"],
            alf_valid,
            allow_alfworld_surface_matching=True,
        ) == "",
    )
    _check(
        "grounding rejects previously failed action",
        _ground_cwme_planned_action(
            ["move magazine to diningtable 1"],
            {"put magazine in diningtable 1", "look"},
            allow_alfworld_surface_matching=True,
            blocked_actions={"put magazine in diningtable 1"},
        ) == "",
    )

    set_grounding_mode(GroundingMode.TASK_HEURISTIC)
    _check("heuristic connect on boil", is_safe_task_pattern_action_heuristic("connect", task="boil") is False)

    cfg = {
        "EXPERIMENT": {"ENABLE_PATTERN_READ": False},
        "EXECUTION": {"grounding_mode": "grounding_only", "legacy_heuristics_enabled": False},
    }
    _check("backbone grounding only", cwme_no_heuristic_mode(cfg))

    # Formal StatePacket construction must not call task-specific phase
    # engines merely to build a cognition prompt.
    from core.state_packet import build_state_packet

    class _ForbiddenEngine:
        def plan(self, *args, **kwargs):
            raise AssertionError("task phase engine leaked into formal packet")

    class _ForbiddenPhase:
        def snapshot(self, *args, **kwargs):
            raise AssertionError("phase controller leaked into formal packet")

    packet_agent = SimpleNamespace(
        agent_config=cfg,
        task="Boil water in the kitchen.",
        observation="This room is called the kitchen.",
        inventory="",
        past_actions=[],
        last_score=0,
        episode_progress={"score_max": 0},
        pre_focus=_ForbiddenEngine(),
        post_focus=_ForbiddenEngine(),
        orchestrator=SimpleNamespace(phase=_ForbiddenPhase()),
        env=None,
        _wm_subgraph=[],
        _wm_episodic=[],
        past_reflections="",
        pattern_library=None,
        hkg_retriever=None,
    )
    packet = build_state_packet(packet_agent)
    _check("formal packet has no task rule hints", packet.phase == "observe")

    # Formal ALFWorld no-pattern routing must not invoke the legacy fast
    # substitute.  Exact admissible grounding remains allowed.
    from core.grounding_facade import GroundingFacade
    import core.grounding_facade as grounding_facade_module

    class _FakeAlfWorld:
        env_name = "alfworld"

        def getValidActionObjectCombinations(self):
            return {"look"}

        def look(self):
            return "You are in a room."

    alf_agent = SimpleNamespace(
        agent_config=cfg,
        task="Move an object.",
        observation="You are in a room.",
        inventory="",
        past_actions=[],
        last_score=0,
        env=_FakeAlfWorld(),
        pattern_library=SimpleNamespace(enable_read=False),
        logger=None,
        recent_failed_actions=set(),
        episode_progress={"score_max": 0},
    )
    facade = GroundingFacade()
    facade.resolve_reuse_action = lambda *args, **kwargs: None
    original_fast = grounding_facade_module.alfworld_fast_substitute
    grounding_facade_module.alfworld_fast_substitute = lambda *args, **kwargs: (
        (_ for _ in ()).throw(AssertionError("formal ALF fast policy leaked"))
    )
    try:
        decision = facade.resolve_trajectory_step(
            object(), alf_agent, "", alf_agent.env, ctx_text=alf_agent.observation, score=0,
        )
    finally:
        grounding_facade_module.alfworld_fast_substitute = original_fast
    _check("formal ALF skips legacy fast policy", decision is None)

    from core.cwme_protocol import _is_formal_experiment, _apply_artifact_freeze_defaults
    formal_exp = {"EXPERIMENT_ID": "E3", "EXPERIMENT_NAME": "ablation"}
    _apply_artifact_freeze_defaults(formal_exp)
    _check("formal auto-freeze", bool(formal_exp.get("ARTIFACT_FREEZE", {}).get("enforce")))
    _check("sanity not formal", not _is_formal_experiment({"EXPERIMENT_ID": "SANITY-S1"}))

    from core.cwme_protocol import apply_cwme_protocol_defaults
    standalone = apply_cwme_protocol_defaults({"EXPERIMENT": {"EXPERIMENT_ID": "E3"}})
    _check("standalone config defaults NUM_AGENTS", standalone["EXPERIMENT"]["NUM_AGENTS"] == 1)
    explicit = apply_cwme_protocol_defaults({"EXPERIMENT": {"NUM_AGENTS": 3}})
    _check("explicit NUM_AGENTS preserved", explicit["EXPERIMENT"]["NUM_AGENTS"] == 3)

    invalid = CWMEExecutionAudit()
    invalid.record_task_heuristic("x")
    _check("audit invalid", not invalid.is_valid_cwme_run())

    # P_train/ΔP isolation: a previous run's output is a writer snapshot,
    # never an implicit training source, and failed revisions shadow source P.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train = tmp / "train.jsonl.procedural.jsonl"
        delta_base = tmp / "run" / "patterns.jsonl"
        source = ProceduralPattern.create(
            task_signature="boil", task_id="boil",
            actions=["focus on pot", "activate stove"], score=100,
        )
        train.write_text(json.dumps(source.to_dict()) + "\n", encoding="utf-8")
        cold = PatternLibrary(persist_path=str(delta_base))
        _check("delta output is not implicit P_train", cold.train_pattern_count() == 0)
        cold.load_train_procedural(str(train))
        revised = cold.ensure_mutable_pattern(source)
        PatternReviser().revise(revised, outcome="failure", reused=True)
        view = cold.combined_procedural_store()
    _check(
        "failed delta shadows P_train",
        sum(len(bucket) for bucket in view.values()) == 1
        and next(iter(view.values()))[0].pattern_id == revised.pattern_id,
    )

    # Read-only Pattern configurations must not create a revision delta even
    # when the outer continual engine is enabled for a paired baseline.
    from core.evolution.continual_engine import ContinualWorldModelEvolution, EvolutionConfig
    readonly_agent = SimpleNamespace(
        pattern_library=PatternLibrary(enable_read=True, enable_write=False),
        execution_metrics=None,
        episode_progress={"score_max": 0},
        past_actions=["<START>", "activate stove"],
        task="boil",
        last_score=0,
        logger=None,
    )
    readonly_pattern = ProceduralPattern.create(
        task_signature="boil", task_id="boil",
        actions=["activate stove"], score=100,
    )
    readonly_pattern.confidence = 0.95
    readonly_pattern.status = "active"
    readonly_pattern.outcome = "success"
    readonly_agent.pattern_library._train_procedural["boil::boil"] = [readonly_pattern]
    ContinualWorldModelEvolution(config=EvolutionConfig(enabled=True, revision=True))._maybe_revise_on_reuse(
        readonly_agent, score=0, prev_score=0, transition=None,
    )
    _check(
        "read-only pattern does not create revision delta",
        readonly_agent.pattern_library.delta_pattern_count() == 0,
    )

    print("\nAll unit tests passed.")


if __name__ == "__main__":
    main()
