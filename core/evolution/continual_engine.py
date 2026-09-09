"""
Continual World-Model Evolution engine (CWME core Φ).

M_{t+1} = Φ(M_t, τ_t)

Orchestrates:
  - transition evaluation
  - procedural pattern extract / validate / consolidate / revise
  - historical KG updates (via agent_bridge)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

from core.evolution.evolution_types import ContinualStepResult
from core.evolution.progress_evaluator import ProgressEvaluator
from core.dual_memory import DualMemory, MemorySnapshot
from core.orchestrator import Orchestrator
from core.procedural.pattern_consolidator import PatternConsolidator
from core.procedural.pattern_extractor import PatternExtractor
from core.procedural.pattern_reviser import PatternReviser
from core.procedural.pattern_validator import PatternValidator
from core.world_model.agent_bridge import after_env_step_hkg, finalize_episode_hkg, get_combined_historical_kg


@dataclass
class EvolutionConfig:
    enabled: bool = True
    validation_threshold: float = 0.65
    revision: bool = True
    consolidation: bool = True


def resolve_evolution_config(agent_config: dict | None) -> EvolutionConfig:
    agent_config = agent_config or {}
    cont = dict(agent_config.get("CONTINUAL") or {})
    pm = dict(agent_config.get("PROCEDURAL_MEMORY") or {})
    baseline = dict(agent_config.get("BASELINE") or {})
    enabled = bool(cont.get("enabled", True))
    if baseline.get("procedural_memory") is False and "enabled" not in cont:
        enabled = False
    if pm.get("enabled") is False:
        enabled = False
    if pm.get("write") is False and not cont.get("enabled"):
        enabled = False
    return EvolutionConfig(
        enabled=enabled,
        validation_threshold=float(pm.get("validation_threshold", 0.65) or 0.65),
        revision=bool(pm.get("revision", True)),
        consolidation=bool(pm.get("consolidation", True)),
    )


class ContinualWorldModelEvolution:
    """CWME outer-ring evolution engine (standalone, no legacy CL loop)."""

    def __init__(
        self,
        dual_memory: DualMemory | None = None,
        *,
        config: EvolutionConfig | None = None,
    ):
        self.dual_memory = dual_memory or DualMemory()
        self.config = config or EvolutionConfig()
        self.extractor = PatternExtractor()
        self.validator = PatternValidator(threshold=self.config.validation_threshold)
        self.consolidator = PatternConsolidator()
        self.reviser = PatternReviser()
        self.progress_evaluator = ProgressEvaluator()
        self._last_milestone: str = ""
        self._last_pattern_id: str = ""

    def after_step(
        self,
        agent,
        *,
        score: int,
        prev_score: int,
        transition=None,
    ) -> ContinualStepResult:
        """Per-step: episode guard, HKG delta, revision, optional incremental evolution."""
        Orchestrator.invalidate_step_cache(agent)
        dual = getattr(agent, "dual_memory", None) or self.dual_memory
        snapshot = dual.read(agent, score=score)

        self._update_episode_guard(
            agent,
            score=score,
            prev_score=prev_score,
            transition=transition,
        )

        if transition is None:
            tracker = getattr(agent, "transition_tracker", None)
            if tracker is not None and len(tracker) > 0:
                transition = tracker.episode_records()[-1]

        if transition is not None:
            after_env_step_hkg(agent, transition=transition)
            guard = getattr(agent, "episode_guard", None)
            if guard is not None and hasattr(guard, "note_transition"):
                from core.evolution.transition_tracker import state_key
                state = state_key(
                    getattr(transition, "state_signature_after", "")
                    or getattr(transition, "state_after", "")
                    or getattr(agent, "observation", "")
                )
                guard.note_transition(
                    state,
                    getattr(transition, "action", "") or "",
                    progress=bool(getattr(transition, "meaningful_change", False)),
                )

        if self.config.enabled and self.config.revision:
            self._maybe_revise_on_reuse(
                agent,
                score=score,
                prev_score=prev_score,
                transition=transition,
            )

        wrote = False
        if self.config.enabled:
            if score > prev_score and score > 0:
                wrote = self._evolve_procedural(
                    agent, snapshot, score=int(score), force=True,
                )
            elif self._should_structural_writeback(agent, score=score):
                wrote = self._evolve_procedural(
                    agent, snapshot, score=int(score), force=True,
                )

        phase = snapshot.phase or ""
        if snapshot.som_sub_phase:
            phase = f"{phase}/{snapshot.som_sub_phase}"
        return ContinualStepResult(
            snapshot=snapshot,
            wrote_pattern=wrote,
            phase_label=phase,
        )

    def finalize_episode(self, agent, *, final_score: int) -> ContinualStepResult:
        """Episode-end: extract → validate → consolidate → persist."""
        Orchestrator.invalidate_step_cache(agent)
        dual = getattr(agent, "dual_memory", None) or self.dual_memory
        snapshot = dual.read(agent, score=final_score)

        wrote = False
        if self.config.enabled:
            wrote = self._evolve_procedural(
                agent, snapshot, score=int(final_score), force=True,
            )

        try:
            ep_id = int((getattr(agent, "episode_progress", {}) or {}).get("episode_idx", 0) or 0)
            finalize_episode_hkg(agent, episode_id=ep_id)
        except Exception as exc:
            logger = getattr(agent, "logger", None)
            if logger:
                logger.exception("[HKG] finalize failed")
            raise

        phase = snapshot.phase or ""
        if snapshot.som_sub_phase:
            phase = f"{phase}/{snapshot.som_sub_phase}"
        return ContinualStepResult(
            snapshot=snapshot,
            wrote_pattern=wrote,
            phase_label=phase,
        )

    def try_incremental_writeback(
        self,
        agent,
        snapshot: MemorySnapshot,
        *,
        score: int,
        force: bool = False,
    ) -> bool:
        if not self.config.enabled:
            return False
        return self._evolve_procedural(agent, snapshot, score=int(score), force=force)

    def _update_episode_guard(
        self,
        agent,
        *,
        score: int,
        prev_score: int,
        transition=None,
    ) -> None:
        guard = getattr(agent, "episode_guard", None)
        if guard is None:
            return

        past = list(getattr(agent, "past_actions", []) or [])
        last_action = (past[-1] if past else "").strip().lower()

        if score > prev_score:
            guard.note_score_gain()
        elif last_action and last_action != "<start>":
            guard.note_action(last_action, progress=False)
            route = getattr(agent, "_last_route_decision", None)
            fast_used = route is not None and getattr(route, "fast_allowed", False)
            transition_failed = (
                transition is not None
                and (
                    not getattr(transition, "valid", True)
                    or not getattr(transition, "meaningful_change", False)
                )
            )
            if fast_used and (
                int(score) < int(prev_score) or transition_failed
            ):
                pid = getattr(route, "pattern_id", "") or ""
                ms = getattr(route, "milestone", "") or last_action
                guard.note_pattern_failure(pid, ms)

    def _should_structural_writeback(self, agent, *, score: int) -> bool:
        plib = getattr(agent, "pattern_library", None)
        if plib is None or not getattr(plib, "enable_write", False):
            return False

        # ScienceWorld exposes many state-changing actions at every score
        # level (navigation, opening arbitrary fixtures, teleporting). They
        # remain trajectory/HKG evidence, but only a score increase can prove
        # that a procedure should be written. Otherwise an already rewarded
        # pattern is repeatedly expanded with arbitrary later exploration.
        try:
            from envs.base import is_scienceworld
            if is_scienceworld(getattr(agent, "env", None)):
                return False
        except Exception:
            pass

        past = list(getattr(agent, "past_actions", []) or [])
        credit = self.progress_evaluator.evaluate(past)
        prog = dict(getattr(agent, "episode_progress", {}) or {})
        prev_credit = int(prog.get("structural_credit", 0) or 0)

        if int(score) <= 0 and credit >= 2 and credit > prev_credit:
            prog["structural_credit"] = credit
            try:
                agent.episode_progress = prog
            except Exception as exc:
                _logger.warning("[CWME] failed to persist structural_credit: %s", exc)
            return True

        if 0 < int(score) < 100 and credit >= 1 and credit > prev_credit:
            prog["structural_credit"] = credit
            try:
                agent.episode_progress = prog
            except Exception as exc:
                _logger.warning("[CWME] failed to persist structural_credit: %s", exc)
            return True

        return False

    def _classify_outcome(
        self,
        agent,
        *,
        score: int,
        prev_score: int,
        transition=None,
    ) -> str:
        if transition is not None and not getattr(transition, "valid", True):
            return "failure"

        prog = dict(getattr(agent, "episode_progress", {}) or {})
        ep_max = int(prog.get("score_max", score) or score)
        if int(score) >= 100 or ep_max >= 100:
            return "success"
        if transition is not None and getattr(transition, "meaningful_change", False):
            return "partial"
        if int(score) > int(prev_score):
            return "partial"
        return "neutral"

    def _evolve_procedural(
        self,
        agent,
        snapshot: MemorySnapshot,
        *,
        score: int,
        force: bool,
    ) -> bool:
        plib = getattr(agent, "pattern_library", None)
        if plib is None or not getattr(plib, "enable_write", False):
            return False

        past = list(getattr(agent, "past_actions", []) or [])
        store_score = int(score)
        if store_score <= 0:
            try:
                from envs.base import is_scienceworld
                if is_scienceworld(getattr(agent, "env", None)):
                    if getattr(agent, "logger", None):
                        agent.logger.info(
                            "[CWME] skip zero-score ScienceWorld writeback; "
                            "retain transitions until a reward validates the prefix",
                        )
                    return False
            except Exception:
                pass
            if store_score <= 0:
                credit = self.progress_evaluator.evaluate(past)
                if not (force and credit >= 2):
                    return False
                partial_cap = max(1, int(getattr(plib, "min_partial_score", 30) or 30) - 1)
                store_score = min(max(credit * 12, 1), partial_cap)

        tracker = getattr(agent, "transition_tracker", None)
        trajectory = tracker.episode_records() if tracker else []
        hkg = get_combined_historical_kg(agent)

        phase = snapshot.phase or ""
        if snapshot.som_sub_phase:
            phase = f"{phase}/{snapshot.som_sub_phase}"

        observation = ""
        if trajectory:
            observation = (getattr(trajectory[-1], "state_after", "") or "").strip()
        elif getattr(agent, "observation", ""):
            observation = (agent.observation or "").strip()

        # Formal writeback must avoid the online sanitizer's task-family
        # rules, which can delete valid setup actions from a grounding-only
        # trajectory. The full-trajectory path is policy-free.
        from core.cl_protocol import grounding_only_mode
        candidate = self.extractor.extract(
            trajectory=trajectory,
            task=getattr(agent, "task", "") or "",
            task_id=getattr(agent, "task_id", "") or "",
            historical_kg=hkg,
            score=float(store_score),
            phase=phase,
            episode_id=int((getattr(agent, "episode_progress", {}) or {}).get("episode_idx", 0) or 0),
            observation=observation,
            working_memory=list(getattr(agent, "_wm_subgraph", []) or []),
            full_trajectory=grounding_only_mode(agent),
        )

        ep_max = int(
            (getattr(agent, "episode_progress", {}) or {}).get("score_max", store_score)
            or store_score
        )
        success = store_score >= 100 or ep_max >= 100
        result = self.validator.validate(
            candidate,
            trajectory,
            final_score=float(store_score),
            success=success,
            task=getattr(agent, "task", "") or "",
            force=force,
        )
        logger = getattr(agent, "logger", None)
        if not result.accepted:
            if logger:
                logger.info(
                    "[CWME] pattern rejected quality=%.2f reason=%s",
                    result.quality,
                    result.reason,
                )
            return False

        if candidate is None:
            return False

        candidate.score = float(store_score)
        candidate.confidence = max(candidate.confidence, result.quality)
        if success:
            candidate.outcome = "success"
            candidate.success_count = 1
        elif store_score > 0 or self.progress_evaluator.evaluate(past) >= 2:
            candidate.outcome = "partial"
            candidate.partial_count = 1
        else:
            candidate.outcome = "failure"
            candidate.failure_count = 1
        candidate.update_status_from_confidence()

        before = plib.procedural_size()
        if self.config.consolidation:
            merged = self.consolidator.consolidate(plib, candidate)
        else:
            plib.add_procedural(candidate)
            merged = candidate

        plib.persist_all_procedural()
        after = plib.procedural_size()
        if logger:
            logger.info(
                "[CWME] pattern stored id=%s quality=%.2f outcome=%s lib=%s->%s",
                merged.pattern_id,
                result.quality,
                getattr(merged, "outcome", ""),
                before,
                after,
            )
        return after > before or (before > 0 and merged is not None)

    def _maybe_revise_on_reuse(
        self,
        agent,
        *,
        score: int,
        prev_score: int,
        transition=None,
    ) -> None:
        plib = getattr(agent, "pattern_library", None)
        # Static-PM and w/o-writeback are read-only ablations.  Previously
        # revision ran here even when PatternLibrary.enable_write was false,
        # silently copying P_train into an in-memory delta and making those
        # rows behave like CWME.
        if (
            plib is None
            or not getattr(plib, "enable_write", False)
            or not self.config.revision
        ):
            return
        milestone = (getattr(plib, "last_served_milestone", "") or "").strip().lower()
        if not milestone:
            return
        past = list(getattr(agent, "past_actions", []) or [])
        executed = (past[-1] if past else "").strip().lower()
        if not executed:
            return
        reused = milestone in executed or executed.startswith(milestone + " ")
        if not reused:
            return
        pat = plib.get_active_procedural()
        if pat is None:
            return
        if hasattr(plib, "ensure_mutable_pattern"):
            pat = plib.ensure_mutable_pattern(pat)
        outcome = self._classify_outcome(
            agent, score=score, prev_score=prev_score, transition=transition,
        )
        if outcome in {"failure", "invalid"}:
            guard = getattr(agent, "episode_guard", None)
            if guard is not None and hasattr(guard, "note_pattern_failure"):
                guard.note_pattern_failure(pat.pattern_id, milestone)
        self.reviser.revise(pat, outcome=outcome, reused=True)
        if hasattr(plib, "persist_all_procedural"):
            plib.persist_all_procedural()
        logger = getattr(agent, "logger", None)
        if logger:
            logger.info(
                "[CWME] pattern revised id=%s outcome=%s conf=%.2f status=%s",
                pat.pattern_id,
                outcome,
                pat.confidence,
                pat.status,
            )
