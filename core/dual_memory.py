"""
Dual-memory READ facade (Memory Hub surface for cognition).

M_t read surface:
  Historical KG (H_train ∪ ΔH) + Working Memory (WM) + Procedural Memory (patterns)

Persistent Postgres KG WRITE/STORE remain in DGAR; this module aggregates
read-only historical world knowledge with WM and patterns for cognition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.state_packet import StatePacket, build_state_packet


@dataclass
class MemorySnapshot:
    """Unified read snapshot for one decision step."""

    phase: str
    som_phase: str = ""
    som_sub_phase: str = ""
    historical_facts: list[str] = field(default_factory=list)
    historical_transitions: list[str] = field(default_factory=list)
    semantic_facts: list[str] = field(default_factory=list)
    episodic_observations: list[str] = field(default_factory=list)
    pattern_hint: str = ""
    pattern_next_milestone: str = ""
    pattern_confidence: float = 0.0
    pattern_status: str = ""
    pattern_preconditions: list[str] = field(default_factory=list)
    pattern_expected_transition: str = ""
    pre_focus_suggestion: str = ""
    post_focus_suggestion: str = ""
    transfer_pending: bool = False
    transfer_executable: bool = False
    heat_setup_pending: bool = False
    current_room: str = ""
    foc_room: str = ""
    score: int = 0
    score_max: int = 0

    @classmethod
    def from_packet(cls, packet: StatePacket) -> "MemorySnapshot":
        return cls(
            phase=packet.phase,
            som_phase=packet.som_phase,
            som_sub_phase=packet.som_sub_phase,
            historical_facts=list(packet.historical_kg_facts),
            historical_transitions=list(packet.historical_kg_transitions),
            semantic_facts=list(packet.wm_subgraph),
            episodic_observations=list(packet.episodic_hints),
            pattern_hint=packet.pattern_hint,
            pattern_next_milestone=packet.pattern_next_milestone,
            pattern_confidence=float(packet.pattern_confidence or 0),
            pattern_status=packet.pattern_status or "",
            pattern_preconditions=list(packet.pattern_preconditions),
            pattern_expected_transition=packet.pattern_expected_transition or "",
            pre_focus_suggestion=packet.pre_focus_suggestion,
            post_focus_suggestion=packet.post_focus_suggestion,
            transfer_pending=packet.transfer_pending,
            transfer_executable=packet.transfer_executable,
            heat_setup_pending=packet.heat_setup_pending,
            current_room=packet.current_room,
            foc_room=packet.foc_room,
            score=packet.score,
            score_max=packet.score_max,
        )


class DualMemory:
    """Aggregates working memory and pattern library for cognition layers."""

    def read(self, agent, *, score: int | None = None) -> MemorySnapshot:
        return MemorySnapshot.from_packet(build_state_packet(agent, score=score))

    def planner_block(self, agent, *, score: int | None = None) -> str:
        return build_state_packet(agent, score=score).to_planner_block()

    def actor_block(self, agent, *, score: int | None = None) -> str:
        return build_state_packet(agent, score=score).to_actor_block()

    def refiner_block(self, agent, *, score: int | None = None) -> str:
        return build_state_packet(agent, score=score).to_refiner_block()

    def fusion_supplement(self, agent, *, score: int | None = None) -> str:
        mf = getattr(agent, "memory_fusion", None)
        if mf is None:
            return ""
        return mf.planner_supplement(agent, score=score)
