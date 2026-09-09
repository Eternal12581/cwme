"""Per-step decision source instrumentation for experiment attribution."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActionSource(str, Enum):
    COGNITION = "cognition"
    PATTERN_SOFT = "pattern_soft"
    PATTERN_HARD = "pattern_hard"
    GROUNDING = "grounding"
    HEURISTIC = "heuristic"
    FALLBACK = "fallback"


@dataclass
class StepDecisionRecord:
    step: int = 0
    action: str = ""
    source: str = ActionSource.COGNITION.value
    decision_source: str = ""
    route_level: int = 2
    llm_called: bool = False
    fast_success: bool | None = None
    pattern_reuse: bool = False
    heuristic: bool = False
    grounding: bool = False
    pattern_id: str = ""
    milestone: str = ""
    hkg_hits: int = 0
    valid_action: bool = True
    score_gain: float = 0.0


@dataclass
class ActionSourceLog:
    records: list[StepDecisionRecord] = field(default_factory=list)

    def record(
        self,
        *,
        step: int,
        action: str,
        source: ActionSource | str,
        decision_source: str = "",
        route_level: int = 2,
        llm_called: bool = False,
        fast_success: bool | None = None,
        pattern_reuse: bool = False,
        heuristic: bool = False,
        grounding: bool = False,
        pattern_id: str = "",
        milestone: str = "",
        hkg_hits: int = 0,
        valid_action: bool = True,
        score_gain: float = 0.0,
    ) -> None:
        src = source.value if isinstance(source, ActionSource) else str(source)
        self.records.append(
            StepDecisionRecord(
                step=int(step),
                action=(action or "").strip(),
                source=src,
                decision_source=(decision_source or src).strip(),
                route_level=int(route_level),
                llm_called=bool(llm_called),
                fast_success=fast_success,
                pattern_reuse=bool(pattern_reuse),
                heuristic=bool(heuristic),
                grounding=bool(grounding),
                pattern_id=pattern_id or "",
                milestone=milestone or "",
                hkg_hits=int(hkg_hits or 0),
                valid_action=bool(valid_action),
                score_gain=float(score_gain or 0.0),
            )
        )

    def distribution(self) -> dict[str, float]:
        if not self.records:
            return {}
        counts: dict[str, int] = {}
        for rec in self.records:
            key = rec.decision_source or rec.source
            counts[key] = counts.get(key, 0) + 1
        total = len(self.records)
        return {k: round(v / total, 4) for k, v in sorted(counts.items())}

    def summary(self) -> dict:
        if not self.records:
            return {}
        dist = self.distribution()
        llm_steps = sum(1 for r in self.records if r.llm_called)
        fast_attempts = sum(
            1 for r in self.records if r.decision_source in {"pattern_fast", "pattern_verify"}
        )
        fast_success = sum(
            1 for r in self.records
            if r.decision_source in {"pattern_fast", "pattern_verify"} and r.fast_success is True
        )
        return {
            "decision_source_distribution": dist,
            "llm_decision_rate": round(llm_steps / len(self.records), 4),
            "fast_attempt_rate": round(fast_attempts / len(self.records), 4),
            "fast_success_rate": round(fast_success / fast_attempts, 4) if fast_attempts else 0.0,
            "total_steps": len(self.records),
        }

    def mark_last_fast_success(self, *, success: bool, score_gain: float = 0.0) -> None:
        if not self.records:
            return
        rec = self.records[-1]
        if rec.decision_source in {"pattern_fast", "pattern_verify"}:
            rec.fast_success = bool(success)
            rec.score_gain = float(score_gain)

    def reset(self) -> None:
        self.records.clear()
