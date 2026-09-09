"""Minimal sanity checks for CWME HKG overlay (no heavy agent imports)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.world_model.historical_kg import HistoricalKG, _norm_id
from core.world_model.kg_updater import HistoricalKGUpdater
from core.world_model.kg_extractor import HistoricalKGExtractor, TrajectoryStep
from core.world_model.kg_retriever import HistoricalKGRetriever
from core.world_model.combined_view import CombinedHistoricalKG
from core.evolution.progress_evaluator import ProgressEvaluator
from core.evolution.transition_tracker import TransitionTracker


def main() -> None:
    train = HistoricalKG(read_only=False)
    train.metadata["train_episode_count"] = 10
    train.metadata["episode_count"] = 10
    train.add_edge("water", "located_at", "kitchen", episode_id=0)
    train.add_edge("water", "located_at", "kitchen", episode_id=1)
    assert train.edges[list(train.edges.keys())[0]].confidence == 0.2

    merge_src = HistoricalKG(read_only=False)
    merge_src.metadata["train_episode_count"] = 2
    merge_src.add_edge("oven", "has_state", "inactive", episode_id=0)
    merge_src.add_edge("oven", "has_state", "inactive", episode_id=1)
    merged_before = train.merge_from(merge_src)
    oven_key = next(k for k, e in train.edges.items() if e.source == "oven")
    assert train.edges[oven_key].episode_support == 2, train.edges[oven_key].episode_support

    changes_to = train.add_edge("oven.inactive", "changes_to", "oven.active", episode_id=0)
    src_type = train.nodes[_norm_id(changes_to.source)].node_type
    tgt_type = train.nodes[_norm_id(changes_to.target)].node_type
    assert src_type == "STATE", src_type
    assert tgt_type == "STATE", tgt_type

    updater = HistoricalKGUpdater(h_train=train, update_test_kg=True)
    extractor = HistoricalKGExtractor()
    edges = extractor.extract_action_transition(
        "The oven is inactive.",
        "activate oven",
        "The oven is active.",
    )
    assert edges, "expected action transition edges"

    causes = [e for e in edges if e.relation == "causes"]
    assert causes, "expected causes edge from action transition"
    assert "oven" in causes[0].target, f"expected structured state node, got {causes[0].target!r}"
    assert "you" not in causes[0].target.lower(), "must not use whole-sentence state nodes"

    requires = [e for e in edges if e.relation == "requires" and e.source == "oven.inactive"]
    assert requires, "expected state→requires→action edge"

    seq = extractor.extract_action_sequence([
        TrajectoryStep(
            state_before="oven inactive", action="activate oven",
            state_after="oven active", score_before=0, score_after=10,
        ),
        TrajectoryStep(
            state_before="oven active", action="use oven",
            state_after="water hot", score_before=10, score_after=20,
        ),
    ])
    assert any(e.relation == "precedes" for e in seq)
    assert not any(e.relation == "followed_by" for e in seq)
    assert not any(e.relation == "enables" for e in seq)

    # REQUIRES must be action-relevant only (not water/door when activating oven).
    noisy = extractor.extract_action_transition(
        "The oven is inactive. The water is cold. The door is open.",
        "activate oven",
        "The oven is active.",
    )
    req_sources = {e.source for e in noisy if e.relation == "requires"}
    assert "oven.inactive" in req_sources
    assert "water.cold" not in req_sources
    assert "door.open" not in req_sources

    has_state = [e for e in edges if e.relation == "has_state" and e.source == "oven"]
    assert has_state and has_state[0].target == "active"

    updater.add_test_edge(edges[0], episode_id=0)
    delta_edge = list(updater.h_delta.edges.values())[0]
    assert delta_edge.episode_support == 1, f"episode_support={delta_edge.episode_support}"
    assert delta_edge.confidence >= 0.15, f"confidence={delta_edge.confidence}"

    combined = updater.h_combined
    assert isinstance(combined, CombinedHistoricalKG)
    assert len(combined) >= 1

    retriever = HistoricalKGRetriever(min_confidence=0.15)
    facts, trans = retriever.retrieve(
        combined, task="heat water", observation="oven active water in kitchen",
    )
    assert facts or trans, "retriever should return hits"

    obs_edges = extractor.extract_observation_facts(
        "The oven is inactive. You are in the kitchen."
    )
    assert any(e.source == "oven" and e.target == "inactive" for e in obs_edges)
    assert any(e.source == "agent" and e.relation == "located_at" for e in obs_edges)

    for edge in extractor.extract_action_transition(
        "a", "activate oven", "You are in kitchen oven active",
    ):
        assert "you_are" not in edge.target.replace(".", "_")
        assert "you are" not in edge.target.lower()

    pe = ProgressEvaluator()
    assert pe.evaluate(["take x", "put y"]) >= 2

    tracker = TransitionTracker()
    tracker.record(
        state_before="s0", action="take x", state_after="s1",
        score_before=0, score_after=0, valid=True,
    )
    tracker.record(
        state_before="s1", action="invalid", state_after="s1",
        score_before=0, score_after=0, valid=False,
    )
    valid_rate = sum(1 for t in tracker.episode_records() if t.valid) / len(tracker)
    assert valid_rate == 0.5

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        train.save(tmp)
        loaded = HistoricalKG.load(tmp, read_only=True)
        oven_edges = [e for e in loaded.edges.values() if e.source == "oven"]
        assert oven_edges, "expected oven edge after reload"
        key = oven_edges[0].key
        assert loaded._edge_episodes.get(key), "episode_ids must restore _edge_episodes"
        assert loaded.edges[key].episode_ids, "episode_ids must persist in KGEdge"

    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
