"""Regression tests for evidence validation and route-level telemetry."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evolution.transition_tracker import TransitionRecord
from core.execution.execution_types import ComputationLevel, ExecutionMetrics
from core.procedural.pattern import ProceduralPattern
from core.procedural.pattern_validator import PatternValidator
from core.procedural.pattern_reuse_policy import is_reuse_eligible
from core.procedural.milestone_selector import MilestoneSelector
from core.procedural.pattern_retriever import PatternRetriever
from core.procedural.pattern_extractor import PatternExtractor
from core.procedural.pattern_normalizer import normalize_full_trajectory_actions
from core.pattern_library import PatternLibrary
from core.evolution.continual_engine import ContinualWorldModelEvolution
from core.alfworld_policy import (
    alf_goal_has_constraints,
    alf_pattern_reuse_ok,
    alfworld_fast_substitute,
)
from core.cwme_config import resolve_cwme_experiment
from core.cwme_planning import (
    cwme_heuristic_fallback_enabled,
    cwme_initial_plan_mode,
    cwme_llm_planning_enabled,
    minimal_cognition_trajectory,
    pad_cognition_trajectory,
)
from core.grounding.grounding_mode import GroundingMode, set_grounding_mode
from core.trajectory_quality import is_complete_success_trajectory
from core.trajectory_quality import trajectory_quality_errors
from core.scienceworld_policy import (
    sci_action_allowed,
    sci_nonempty_fallback,
    sciworld_fast_substitute,
)
from core.scienceworld_grounding import is_safe_grounding_only_execution_action
from scripts.collect_ground_truth_trajectories import (
    _resolve_alf_action,
    _resolve_alf_prerequisite,
)


class _Route:
    def __init__(self, level, reason, pattern_id=""):
        self.level = level
        self.reason = reason
        self.pattern_id = pattern_id


class TestCWMECore(unittest.TestCase):
    def test_enabling_only_partial_pattern_is_not_reused(self):
        pattern = ProceduralPattern.create(
            task_signature="boil",
            actions=["go to kitchen", "open cupboard"],
            score=30,
            task_id="boil",
        )
        pattern.outcome = "partial"
        pattern.confidence = 0.80
        pattern.action_metadata = [
            {"action": "go to kitchen", "progress_kind": "enabling", "score_delta": 0},
            {"action": "open cupboard", "progress_kind": "enabling", "score_delta": 0},
        ]
        self.assertFalse(is_reuse_eligible(pattern))

        pattern.action_metadata.append(
            {"action": "activate stove", "progress_kind": "reward", "score_delta": 5},
        )
        self.assertTrue(is_reuse_eligible(pattern))

    def test_reward_backed_delta_outranks_static_success_for_same_task(self):
        library = PatternLibrary(max_patterns_per_sig=5)
        train = ProceduralPattern.create(
            task_signature="boil", task_id="boil",
            actions=["go to kitchen"], score=100,
        )
        train.outcome = "success"
        train.confidence = 0.99
        key = "boil::boil"
        library._train_procedural[key] = [train]

        delta = ProceduralPattern.create(
            task_signature="boil", task_id="boil",
            actions=["activate stove"], score=45,
            action_metadata=[{
                "action": "activate stove",
                "progress_kind": "reward",
                "score_delta": 45,
            }],
        )
        delta.outcome = "partial"
        delta.confidence = 0.65
        library.add_procedural(delta)

        bucket, kind = PatternRetriever().retrieve(
            library, "Boil water in the kitchen.", task_id="boil",
        )
        self.assertEqual(kind, "exact")
        self.assertEqual(bucket[0].pattern_id, delta.pattern_id)

        selected = MilestoneSelector().select(
            library,
            "Boil water in the kitchen.",
            [],
            task_id="boil",
            env_valid={"go to kitchen", "activate stove"},
            observation="This room is called the kitchen. A stove is here.",
            environment="scienceworld",
        )
        self.assertEqual(selected, "activate stove")

    def test_selector_prefers_pattern_matching_current_state(self):
        stale = ProceduralPattern.create(
            task_signature="pick_and_place_simple",
            task_id="pick_and_place_simple",
            actions=["go to drawer 1"], score=100,
            preconditions=["before go to drawer: agent.location=kitchen"],
            expected_transitions=[
                "go to drawer 1: agent.location=kitchen -> agent.location=drawer",
            ],
        )
        stale.outcome = "success"
        stale.confidence = 0.99
        matched = ProceduralPattern.create(
            task_signature="pick_and_place_simple",
            task_id="pick_and_place_simple",
            actions=["go to cabinet 1"], score=100,
            preconditions=["before go to cabinet: agent.location=hallway"],
            expected_transitions=[
                "go to cabinet 1: agent.location=hallway -> agent.location=cabinet",
            ],
        )
        matched.outcome = "success"
        matched.confidence = 0.65

        class Retriever:
            def retrieve(self, library, task, *, task_id=""):
                return [stale, matched], "exact"

        library = SimpleNamespace(enable_read=True, last_match_kind="", last_served_milestone="")
        selected = MilestoneSelector(retriever=Retriever()).select(
            library,
            "pick_and_place_simple",
            [],
            task_id="pick_and_place_simple",
            env_valid={"go to drawer 1", "go to cabinet 1"},
            observation="agent.location=hallway",
            environment="alfworld",
        )
        self.assertEqual(selected, "go to cabinet 1")
        self.assertGreaterEqual(library.last_pattern_context_score, 0.65)

    def test_selector_falls_back_when_no_pattern_matches_current_state(self):
        pattern = ProceduralPattern.create(
            task_signature="pick_and_place_simple",
            task_id="pick_and_place_simple",
            actions=["go to drawer 1"], score=100,
            preconditions=["before go to drawer: agent.location=kitchen"],
            expected_transitions=[
                "go to drawer 1: agent.location=kitchen -> agent.location=drawer",
            ],
        )
        pattern.outcome = "success"
        pattern.confidence = 0.99

        class Retriever:
            def retrieve(self, library, task, *, task_id=""):
                return [pattern], "exact"

        library = SimpleNamespace(enable_read=True, last_match_kind="", last_served_milestone="")
        selected = MilestoneSelector(retriever=Retriever()).select(
            library,
            "pick_and_place_simple",
            [],
            task_id="pick_and_place_simple",
            env_valid={"go to drawer 1"},
            observation="agent.location=hallway",
            environment="alfworld",
        )
        self.assertEqual(selected, "")

    def test_scienceworld_zero_score_changes_do_not_trigger_pattern_writeback(self):
        engine = ContinualWorldModelEvolution()
        agent = SimpleNamespace(
            env=SimpleNamespace(env_name="scienceworld"),
            pattern_library=SimpleNamespace(enable_write=True),
            transition_tracker=SimpleNamespace(),
            past_actions=["teleport to foundry", "open blast furnace"],
            logger=None,
        )
        self.assertFalse(engine._should_structural_writeback(agent, score=0))
        self.assertFalse(engine._should_structural_writeback(agent, score=77))
        self.assertFalse(
            engine._evolve_procedural(
                agent, SimpleNamespace(phase="", som_sub_phase=""),
                score=0, force=True,
            )
        )

    def test_static_success_remains_fallback_for_unverified_delta(self):
        library = PatternLibrary(max_patterns_per_sig=1)
        train = ProceduralPattern.create(
            task_signature="boil", task_id="boil",
            actions=["activate stove"], score=100,
        )
        train.outcome = "success"
        train.confidence = 0.99
        library._train_procedural["boil::boil"] = [train]

        delta = ProceduralPattern.create(
            task_signature="boil", task_id="boil",
            actions=["open cupboard"], score=30,
            action_metadata=[{
                "action": "open cupboard",
                "progress_kind": "enabling",
                "score_delta": 0,
            }],
        )
        delta.outcome = "partial"
        delta.confidence = 0.99
        library.add_procedural(delta)

        bucket, _ = PatternRetriever().retrieve(
            library, "Boil water in the kitchen.", task_id="boil",
        )
        self.assertEqual([p.pattern_id for p in bucket], [train.pattern_id])

    def test_validator_accepts_clean_reward_pattern_with_narrow_tolerance(self):
        pattern = ProceduralPattern.create(
            task_signature="thermometer",
            task_id="use-thermometer",
            actions=["focus on mercury"],
            score=45,
            expected_transitions=[
                "focus on mercury: cold mercury -> hot mercury",
            ],
        )
        trajectory = [
            TransitionRecord(
                action="focus on mercury",
                state_before="mercury is cold",
                state_after="mercury is now warm",
                score_before=0,
                score_after=45,
                valid=True,
                meaningful_change=True,
                progress_kind="reward",
                score_delta=45,
            )
        ]
        result = PatternValidator().validate(
            pattern, trajectory, final_score=45, success=False, force=True,
        )
        self.assertLess(result.quality, 0.65)
        self.assertGreaterEqual(result.quality, 0.60)
        self.assertTrue(result.accepted)

    def test_extractor_excludes_unvalidated_exploration_after_reward(self):
        trajectory = [
            TransitionRecord(
                action="go to workshop",
                state_before="agent.location=kitchen",
                state_after="agent.location=workshop",
                valid=True,
                meaningful_change=True,
                progress_kind="enabling",
            ),
            TransitionRecord(
                action="focus on mercury",
                state_before="agent.location=workshop",
                state_after="agent.focus=mercury",
                score_before=0,
                score_after=77,
                valid=True,
                meaningful_change=True,
                progress_kind="reward",
                score_delta=77,
            ),
            TransitionRecord(
                action="open freezer",
                state_before="agent.focus=mercury",
                state_after="freezer.state=open",
                score_before=77,
                score_after=77,
                valid=True,
                meaningful_change=True,
                progress_kind="enabling",
            ),
        ]
        pattern = PatternExtractor().extract(
            trajectory=trajectory,
            task="Use a thermometer to find the melting point.",
            task_id="use-thermometer",
            score=77,
        )
        self.assertIsNotNone(pattern)
        self.assertTrue(any(action.startswith("focus on") for action in pattern.actions))
        self.assertNotIn("open freezer", pattern.actions)

    def test_legacy_reuse_policy_imports(self):
        from core.reuse_policy_legacy import LegacyReusePolicy

        self.assertTrue(LegacyReusePolicy)

    def test_legacy_reuse_policy_override_call_keeps_facade_argument(self):
        from core.reuse_policy_legacy import LegacyReusePolicy
        import inspect

        params = list(inspect.signature(LegacyReusePolicy.resolve_actor_override).parameters)
        self.assertEqual(
            params[:6],
            ["facade", "orchestrator", "agent", "planned", "env_valid", "ctx_text"],
        )

    def test_trajectory_collector_flushes_each_completed_episode(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from core.trajectory_collector import maybe_export_episode

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "training_trajectories.jsonl"
            record = TransitionRecord(
                episode_id=0,
                step=0,
                action="go to kitchen",
                state_before="agent.location=hall",
                state_after="agent.location=kitchen",
                valid=True,
                meaningful_change=True,
            )
            agent = SimpleNamespace(
                episode_progress={"episode_idx": 0},
                transition_tracker=SimpleNamespace(episode_records=lambda: [record]),
                task="boil",
                task_id="boil",
                _wm_subgraph=[],
            )
            # Use the real collector while keeping the test fixture small.
            from core.trajectory_collector import TrajectoryCollector
            agent.trajectory_collector = TrajectoryCollector(str(out))
            maybe_export_episode(agent, final_score=0)
            self.assertTrue(out.is_file())
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 1)

    def test_complete_success_trajectory_accepts_shorter_raw_gold_plan(self):
        row = {
            "success": True,
            "final_score": 100,
            "source": {
                "gold_action_count": 39,
                "executed_action_count": 2,
                "truncated_by_max_steps": False,
            },
            "steps": [
                {"state_before": "s0", "state_after": "s1", "valid": True},
                {"state_before": "s1", "state_after": "s2", "valid": True},
            ],
        }
        self.assertTrue(is_complete_success_trajectory(row))

    def test_complete_success_trajectory_rejects_corrupt_exports(self):
        base = {
            "success": True,
            "final_score": 100,
            "source": {"executed_action_count": 2},
            "steps": [
                {"state_before": "s0", "state_after": "s1", "valid": True},
                {"state_before": "s1", "state_after": "s2", "valid": True},
            ],
        }
        for mutate, reason in (
            (lambda row: row["source"].update(truncated_by_max_steps=True), "truncated"),
            (lambda row: row["steps"].__getitem__(1).update(valid=False), "invalid_steps"),
            (lambda row: row["steps"].__getitem__(1).update(state_before="other"), "state_chain_mismatch"),
        ):
            row = {
                "success": base["success"],
                "final_score": base["final_score"],
                "source": dict(base["source"]),
                "steps": [dict(step) for step in base["steps"]],
            }
            mutate(row)
            self.assertFalse(is_complete_success_trajectory(row))
            self.assertIn(reason, trajectory_quality_errors(row))
    def test_cognition_padding_has_one_terminal_sentinel(self):
        trajectory, _ = minimal_cognition_trajectory(
            "initial", "", max_steps=3, task="boil",
        )
        self.assertEqual(len(trajectory), 4)
        self.assertEqual(trajectory[-1]["action"], "<PREDICT>")
        self.assertTrue(all(item["action"] == "" for item in trajectory[:-1]))

        padded, _ = pad_cognition_trajectory(
            trajectory,
            observation="live",
            inventory="pot",
            pad_steps=2,
            task="boil",
        )
        self.assertEqual(len(padded), 6)
        self.assertEqual(
            sum(item["action"] == "<PREDICT>" for item in padded), 1,
        )
        self.assertEqual(padded[-1]["action"], "<PREDICT>")
        self.assertEqual(padded[-3]["state"]["observation"], "initial")

    def test_alfworld_high_pddl_goto_uses_discrete_location_argument(self):
        action, exact = _resolve_alf_action(
            {
                "discrete_action": {"action": "GotoLocation", "args": ["sidetable"]},
                "planner_action": {"action": "GotoLocation", "location": "loc|1|-7|2|45"},
            },
            ["go to sidetable", "go to desk"],
        )
        self.assertTrue(exact)
        self.assertEqual(action, "go to sidetable")

        action, exact = _resolve_alf_action(
            {
                "discrete_action": {"action": "GotoLocation", "args": ["sidetable"]},
                "planner_action": {"action": "GotoLocation", "location": "loc|1|-7|2|45"},
            },
            ["go to side table", "go to desk"],
        )
        self.assertTrue(exact)
        self.assertEqual(action, "go to side table")

        action, exact = _resolve_alf_action(
            {
                "discrete_action": {"action": "PickupObject", "args": ["alarmclock"]},
                "planner_action": {
                    "action": "PickupObject",
                    "coordinateObjectId": ["AlarmClock", [1.0, 1.0]],
                    "coordinateReceptacleObjectId": ["SideTable", [0.9, 0.9]],
                    "objectId": "AlarmClock|+00.25|+00.81|-02.51",
                },
            },
            [
                "take alarmclock 1 from sidetable 1",
                "take alarmclock 2 from sidetable 1",
                "take alarmclock 3 from sidetable 1",
                "take creditcard 1 from sidetable 1",
            ],
        )
        self.assertTrue(exact)
        self.assertEqual(action, "take alarmclock 1 from sidetable 1")

        prerequisite = _resolve_alf_prerequisite(
            {
                "discrete_action": {"action": "PickupObject", "args": ["apple"]},
                "planner_action": {
                    "action": "PickupObject",
                    "coordinateReceptacleObjectId": ["Microwave", [0.9, 0.9]],
                },
            },
            ["open microwave 1", "go to cabinet 1"],
        )
        self.assertEqual(prerequisite, "open microwave 1")

        action, exact = _resolve_alf_action(
            {
                "discrete_action": {"action": "ToggleObject", "args": ["desklamp"]},
                "planner_action": {
                    "action": "ToggleObject",
                    "objectId": "DeskLamp|+00.10|+00.20|-00.30",
                },
            },
            ["use desklamp 1", "use coffeemachine 1"],
        )
        self.assertTrue(exact)
        self.assertEqual(action, "use desklamp 1")

        action, exact = _resolve_alf_action(
            {
                "discrete_action": {"action": "CleanObject", "args": {}},
                "planner_action": {
                    "action": "CleanObject",
                    "cleanObjectId": "Apple|-00.10|+01.06|-00.75",
                    "coordinateObjectId": ["Apple", [-0.38, -0.38]],
                    "coordinateReceptacleObjectId": ["SinkBasin", [-7.14, -7.14]],
                    # In CleanObject, objectId identifies the sink, not Apple.
                    "objectId": "Sink|-01.79|+00.90|-03.75|SinkBasin",
                },
            },
            ["clean apple 1 with sinkbasin 1", "move apple 1 to sinkbasin 1"],
        )
        self.assertTrue(exact)
        self.assertEqual(action, "clean apple 1 with sinkbasin 1")

    def test_full_gold_normalization_preserves_order_and_repeated_actions(self):
        actions = [
            "look around",
            "go to kitchen",
            "take apple 1 from table 1",
            "take banana 1 from table 1",
            "go to workshop",
        ]
        normalized = normalize_full_trajectory_actions(
            actions,
            task_id="pick_and_place_simple",
            score=100,
        )
        self.assertEqual(
            normalized,
            ["go to kitchen", "take", "take", "go to workshop"],
        )
        long_normalized = normalize_full_trajectory_actions(
            ["activate stove"] * 25,
            task_id="boil",
            score=100,
        )
        self.assertEqual(len(long_normalized), 25)

    def test_pattern_reuse_does_not_skip_an_ungroundable_prefix(self):
        set_grounding_mode(GroundingMode.GROUNDING_ONLY)
        pattern = ProceduralPattern.create(
            task_signature="boil",
            task_id="boil",
            actions=["focus on water", "activate stove"],
            score=100,
        )

        class Retriever:
            def retrieve(self, library, task, *, task_id=""):
                return [pattern], "exact"

        library = SimpleNamespace(
            enable_read=True,
            last_match_kind="",
            last_served_milestone="",
        )
        selected = MilestoneSelector(retriever=Retriever()).select(
            library,
            "Boil water in the kitchen.",
            [],
            task_id="boil",
            env_valid={"activate stove"},
            observation="This room is called the kitchen.",
        )
        self.assertEqual(selected, "")

    def test_artifact_evidence_resolves_abstract_pattern_to_current_target(self):
        from core.execution.artifact_evidence import execution_pattern_evidence
        from core.execution.execution_router import ExecutionRouter

        set_grounding_mode(GroundingMode.GROUNDING_ONLY)
        pattern = ProceduralPattern.create(
            task_signature="boil",
            task_id="boil",
            actions=["activate"],
            score=100,
            expected_transitions=["activate stove: cold stove -> hot stove"],
        )
        pattern.outcome = "success"
        pattern.confidence = 0.95
        library = SimpleNamespace(
            enable_read=True,
            get_active_procedural=lambda: pattern,
            last_pattern_template="activate",
            last_served_milestone="activate stove",
            last_expected_transition="activate stove: cold stove -> hot stove",
            last_pattern_preconditions=[],
        )
        agent = SimpleNamespace(
            pattern_library=library,
            task="Boil water in the kitchen.",
            observation="This room is called the kitchen. A stove and sink are here.",
            env=SimpleNamespace(env_name="scienceworld"),
            cwme_cfg=SimpleNamespace(
                learned_reuse_enabled=True,
                pattern_reuse_mode="hard",
                pattern_verify_enabled=True,
            ),
            past_actions=["<START>"],
            episode_guard=None,
            recent_failed_actions=set(),
            historical_kg=None,
            hkg_retriever=None,
        )
        evidence = execution_pattern_evidence(
            agent, task=agent.task, observation=agent.observation,
        )
        route = ExecutionRouter().route(
            agent,
            milestone="activate",
            match_kind="exact",
            env_valid={"activate stove", "activate sink"},
            task=agent.task,
            past_actions=agent.past_actions,
            observation=agent.observation,
            pattern_evidence=evidence,
        )
        self.assertEqual(route.grounded_action, "activate stove")

    def test_artifact_evidence_survives_final_grounding_gate(self):
        from core.grounding_facade import ExecutionDecision, GroundingFacade

        set_grounding_mode(GroundingMode.GROUNDING_ONLY)
        pattern = ProceduralPattern.create(
            task_signature="boil",
            task_id="boil",
            actions=["activate stove"],
            score=100,
            expected_transitions=["activate stove: cold stove -> hot stove"],
        )
        pattern.outcome = "success"
        pattern.confidence = 0.95
        library = SimpleNamespace(
            get_active_procedural=lambda: pattern,
            last_pattern_template="activate stove",
            last_served_milestone="activate stove",
            last_expected_transition="activate stove: cold stove -> hot stove",
            last_pattern_preconditions=[],
        )
        env = SimpleNamespace(
            env_name="scienceworld",
            look=lambda: "This room is called the kitchen. A stove is here.",
        )
        agent = SimpleNamespace(
            env=env,
            task="Boil water in the kitchen.",
            observation=env.look(),
            inventory="",
            past_actions=[],
            pattern_library=library,
            historical_kg=None,
            hkg_retriever=None,
            agent_config={"EXECUTION": {"grounding_mode": "grounding_only"}},
            logger=None,
        )
        decision = GroundingFacade._finalize_decision(
            agent,
            ExecutionDecision(action="activate stove", source="pattern_fast"),
        )
        self.assertIsNotNone(decision)

    def test_hkg_first_action_and_precondition_constraint(self):
        from core.world_model.historical_kg import HistoricalKG
        from core.world_model.kg_retriever import HistoricalKGRetriever

        kg = HistoricalKG(read_only=False)
        kg.add_edge("focus(seed)", "precedes", "open(door_to_kitchen)")
        kg.add_edge("oven.door_open", "requires", "activate(stove)")
        retriever = HistoricalKGRetriever()
        self.assertIsNot(
            retriever.action_constraint_signal(
                kg,
                "open door to kitchen",
                past_actions=["<START>"],
                observation="This room is called the hallway.",
            ),
            False,
        )
        self.assertFalse(
            retriever.action_constraint_signal(
                kg,
                "activate stove",
                past_actions=["open door to kitchen"],
                observation="This room is called the kitchen.",
            )
        )

    def test_validator_scores_expected_effect(self):
        pattern = ProceduralPattern.create(
            task_signature="task",
            actions=["activate stove"],
            score=100,
            expected_transitions=[
                "activate stove: cold stove -> hot stove",
            ],
        )
        trajectory = [
            TransitionRecord(
                action="activate stove",
                state_before="cold stove",
                state_after="hot stove",
                score_before=0,
                score_after=100,
                meaningful_change=True,
            )
        ]
        result = PatternValidator().validate(
            pattern,
            trajectory,
            final_score=100,
            success=True,
        )
        self.assertEqual(result.expected_transition_match, 1.0)
        self.assertGreaterEqual(result.transition_consistency, 0.99)
        self.assertTrue(result.accepted)

    def test_validator_does_not_accept_unmatched_effect_as_exact(self):
        pattern = ProceduralPattern.create(
            task_signature="task",
            actions=["activate stove"],
            score=100,
            expected_transitions=[
                "activate stove: cold stove -> frozen stove",
            ],
        )
        trajectory = [
            TransitionRecord(
                action="activate stove",
                state_before="cold stove",
                state_after="hot stove",
                score_before=0,
                score_after=100,
                meaningful_change=True,
            )
        ]
        result = PatternValidator().validate(
            pattern,
            trajectory,
            final_score=100,
            success=True,
        )
        self.assertLess(result.expected_transition_match, 1.0)

    def test_route_telemetry_is_exported(self):
        metrics = ExecutionMetrics()
        metrics.record_route(_Route(ComputationLevel.DIRECT_REUSE, "high_confidence_reuse", "P1"))
        metrics.record_route(_Route(ComputationLevel.FULL_COGNITION, "blocked_or_invalid", "P1"))
        summary = metrics.summary()
        self.assertEqual(summary["route_level_counts"]["direct_reuse"], 1)
        self.assertEqual(summary["route_level_counts"]["full_cognition"], 1)
        self.assertEqual(summary["route_reason_counts"]["blocked_or_invalid"], 1)
        self.assertEqual(summary["route_pattern_counts"]["P1"], 2)

    def test_unparsed_alfworld_task_id_does_not_block_navigation_reuse(self):
        self.assertFalse(alf_goal_has_constraints("pick_and_place_simple"))
        self.assertTrue(
            alf_pattern_reuse_ok("go to drawer 1", "pick_and_place_simple", [])
        )

    def test_parsed_alfworld_goal_still_blocks_wrong_navigation(self):
        task = "Move a magazine from the bed to the table."
        self.assertTrue(alf_goal_has_constraints(task))
        self.assertFalse(alf_pattern_reuse_ok("go to drawer 1", task, []))
        self.assertTrue(alf_pattern_reuse_ok("go to bed 1", task, []))

    def test_hybrid_mode_skips_open_loop_llm_plan(self):
        config = {
            "EXPERIMENT": {},
            "EXECUTION": {
                "cwme_initial_plan_mode": "hybrid",
                "cwme_heuristic_fallback": True,
                "grounding_mode": "task_heuristic",
            },
        }
        agent = SimpleNamespace(
            agent_config=config,
            cwme_cfg=resolve_cwme_experiment(config),
        )
        self.assertEqual(cwme_initial_plan_mode(agent), "hybrid")
        self.assertFalse(cwme_llm_planning_enabled(agent))
        self.assertTrue(cwme_heuristic_fallback_enabled(agent))

    def test_grounding_only_hard_disables_heuristic_fallback(self):
        config = {
            "EXPERIMENT": {},
            "EXECUTION": {
                "cwme_initial_plan_mode": "hybrid",
                "cwme_heuristic_fallback": True,
                "grounding_mode": "grounding_only",
            },
        }
        agent = SimpleNamespace(
            agent_config=config,
            cwme_cfg=resolve_cwme_experiment(config),
        )
        self.assertFalse(cwme_heuristic_fallback_enabled(agent))

    def test_alfworld_empty_high_level_plan_recovers_source_navigation(self):
        valid = {
            "go to bed 1",
            "go to drawer 1",
            "look",
            "take magazine 1 from bed 1",
        }
        action = alfworld_fast_substitute(
            valid,
            {"<START>"},
            "You are in the bedroom.",
            "Move a magazine from the bed to the table.",
            ["<START>"],
            look="You are in the bedroom.",
        )
        self.assertEqual(action, "go to bed 1")

    def test_template_pattern_cannot_leak_wrong_alfworld_navigation(self):
        pattern = ProceduralPattern.create(
            task_signature="pick_and_place_simple",
            task_id="pick_and_place_simple",
            actions=["go to drawer 1"],
            score=100,
        )

        class Library:
            enable_read = True
            max_patterns_per_sig = 5
            last_match_kind = ""
            last_served_milestone = ""

            def combined_procedural_store(self):
                return {"pick_and_place_simple::pick_and_place_simple": [pattern]}

        selected = MilestoneSelector().select(
            Library(),
            "Move a magazine from the bed to the table.",
            [],
            task_id="pick_and_place_simple",
            env_valid={"go to drawer 1", "go to bed 1"},
        )
        self.assertEqual(selected, "")

    def test_protocol_defaults_keep_llm_mode_for_unspecified_configs(self):
        config = {"EXPERIMENT": {}, "EXECUTION": {}}
        agent = SimpleNamespace(
            agent_config=config,
            cwme_cfg=resolve_cwme_experiment(config),
        )
        self.assertEqual(cwme_initial_plan_mode(agent), "llm")
        self.assertTrue(cwme_llm_planning_enabled(agent))
        self.assertFalse(cwme_heuristic_fallback_enabled(agent))

    def test_scienceworld_fast_path_checks_environment_and_avoids_repeat(self):
        set_grounding_mode(GroundingMode.GROUNDING_ONLY)
        self.assertFalse(
            sci_action_allowed(
                "pick up sewer",
                "grow-plant",
                env_valid={"go to bathroom"},
            )
        )
        self.assertTrue(
            sci_action_allowed(
                "pour water into flower pot",
                "grow-plant",
                env_valid={"pour water into flower pot"},
                task_aware=False,
            )
        )
        action = sciworld_fast_substitute(
            {"look around", "go to bathroom"},
            {"look around"},
            "This room is called the living room.",
            "Seeds can be found in the bathroom. First, focus on a seed.",
            ["look around"],
            current_score=0,
            steps_since_gain=1,
        )
        self.assertEqual(action, "go to bathroom")

    def test_grounding_only_final_execution_gate_blocks_actor_bypass(self):
        self.assertFalse(
            is_safe_grounding_only_execution_action(
                "activate sink",
                task="Boil water in the kitchen.",
                observation="This room is called the kitchen. A sink is here.",
            )
        )
        self.assertFalse(
            is_safe_grounding_only_execution_action(
                "focus on air",
                task="Use the thermometer.",
                observation="The thermometer is here.",
            )
        )
        self.assertTrue(
            is_safe_grounding_only_execution_action(
                "activate stove",
                task="Boil water in the kitchen.",
                observation="This room is called the kitchen. A stove is here.",
            )
        )
        self.assertTrue(
            is_safe_grounding_only_execution_action(
                "go to kitchen",
                task="Boil water in the kitchen.",
            )
        )

    def test_facade_final_gate_covers_non_pattern_decisions(self):
        from core.grounding_facade import GroundingFacade, ExecutionDecision

        class FakeScienceWorld:
            env_name = "scienceworld"

            def look(self):
                return "This room is called the bathroom. A sink is here."

        agent = SimpleNamespace(
            env=FakeScienceWorld(),
            task="Boil water in the kitchen.",
            observation="This room is called the bathroom. A sink is here.",
            inventory="",
            past_actions=[],
            agent_config={
                "EXPERIMENT": {},
                "EXECUTION": {"grounding_mode": "grounding_only"},
            },
            logger=None,
        )
        blocked = GroundingFacade._finalize_decision(
            agent,
            ExecutionDecision(action="activate sink", source="planner_grounded"),
        )
        self.assertIsNone(blocked)

        allowed = GroundingFacade._finalize_decision(
            agent,
            ExecutionDecision(action="go to kitchen", source="planner_grounded"),
        )
        self.assertIsNotNone(allowed)

    def test_fast_success_requires_progress_but_tracks_env_acceptance(self):
        metrics = ExecutionMetrics()
        metrics.fast_path_attempts = 1
        metrics.fast_path_env_success = 1
        metrics.fast_path_success = 0
        self.assertEqual(metrics.summary()["fast_path_env_success"], 1)
        self.assertEqual(metrics.summary()["fast_success_rate"], 0.0)

    def test_scienceworld_fallback_skips_current_room_and_repeated_look(self):
        action = sci_nonempty_fallback(
            "Boil water in the kitchen.",
            {"go to kitchen", "look around", "activate stove"},
            {"look around"},
            "This room is called the kitchen.",
            ["go to kitchen", "look around", "look around"],
            task_aware=True,
        )
        self.assertEqual(action, "activate stove")

    def test_scienceworld_fallback_does_not_repeat_navigation_destination(self):
        action = sci_nonempty_fallback(
            "Find a seed in the bathroom.",
            {"go to bathroom", "look around"},
            {"go to bathroom"},
            "This room is called the living room.",
            ["go to bathroom", "look around"],
            task_aware=True,
        )
        self.assertEqual(action, "look around")

    def test_scienceworld_fallback_applies_focus_safety_in_grounding_only(self):
        action = sci_nonempty_fallback(
            "Boil lead in the kitchen.",
            {"focus on bed", "look around"},
            set(),
            "This room is called the kitchen. A bed is here.",
            [],
            task_aware=True,
        )
        self.assertEqual(action, "look around")

    def test_scienceworld_hybrid_fallback_runs_before_actor(self):
        from core.grounding_facade import GroundingFacade

        class FakeScienceWorld:
            env_name = "scienceworld"

            def getValidActionObjectCombinations(self):
                return {"look around", "activate stove"}

            def look(self):
                return "This room is called the kitchen."

        config = {
            "EXPERIMENT": {},
            "EXECUTION": {
                "grounding_mode": "grounding_only",
                "cwme_heuristic_fallback": True,
            },
        }
        agent = SimpleNamespace(
            agent_config=config,
            cwme_cfg=resolve_cwme_experiment(config),
            task="Boil water in the kitchen.",
            past_actions=["<START>", "look around"],
            last_score=0,
            inventory="",
            observation="This room is called the kitchen.",
            episode_progress={"steps_since_gain": 1},
            recent_failed_actions=set(),
            pattern_library=SimpleNamespace(enable_read=True),
        )
        facade = GroundingFacade()
        facade.resolve_reuse_action = lambda *args, **kwargs: None
        decision = facade.resolve_trajectory_step(
            object(),
            agent,
            "look around",
            FakeScienceWorld(),
            ctx_text=agent.observation,
            score=0,
        )
        # The flag cannot re-enable task-family shortcuts inside the formal
        # grounding-only protocol; the caller must invoke the Actor instead.
        self.assertIsNone(decision)

    def test_scienceworld_task_heuristic_mode_keeps_explicit_fallback(self):
        from core.grounding_facade import GroundingFacade

        class FakeScienceWorld:
            env_name = "scienceworld"

            def getValidActionObjectCombinations(self):
                return {"look around", "activate stove"}

            def look(self):
                return "This room is called the kitchen."

        config = {
            "EXPERIMENT": {},
            "EXECUTION": {
                "grounding_mode": "task_heuristic",
                "cwme_heuristic_fallback": True,
            },
        }
        agent = SimpleNamespace(
            agent_config=config,
            cwme_cfg=resolve_cwme_experiment(config),
            task="Boil water in the kitchen.",
            past_actions=["<START>", "look around"],
            last_score=0,
            inventory="",
            observation="This room is called the kitchen.",
            episode_progress={"steps_since_gain": 1},
            recent_failed_actions=set(),
            pattern_library=SimpleNamespace(enable_read=True),
        )
        facade = GroundingFacade()
        facade.resolve_reuse_action = lambda *args, **kwargs: None
        decision = facade.resolve_trajectory_step(
            object(),
            agent,
            "look around",
            FakeScienceWorld(),
            ctx_text=agent.observation,
            score=0,
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "activate stove")
        self.assertEqual(decision.source, "sciworld_fast")


if __name__ == "__main__":
    unittest.main()
