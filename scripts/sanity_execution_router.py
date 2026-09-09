"""Sanity checks for Memory-Guided Fast Execution router (3-tier)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._bootstrap_utils_lite import bootstrap_utils_lite

bootstrap_utils_lite()

from core.execution.execution_router import ExecutionRouter
from core.execution.execution_types import ComputationLevel
from core.execution.fast_executor import FastExecutor
from core.execution.pattern_verifier import PatternVerifyResult, verify_pattern_action
from core.evolution.episode_guard import EpisodeGuard


class _Agent:
    def __init__(self):
        self.observation = "water in kitchen oven inactive"
        self.past_actions = []
        self.pattern_library = _Plib()
        self.episode_guard = EpisodeGuard()
        self.cwme_cfg = _Cfg()
        self.execution_metrics = _Metrics()


class _Plib:
    last_pattern_confidence = 0.85
    last_active_pattern_id = "p1"
    last_pattern_preconditions = ["before activate: kitchen"]
    enable_read = True

    def get_active_procedural(self):
        return None


class _Cfg:
    fast_path_enabled = True
    pattern_reuse_mode = "hard"
    pattern_verify_enabled = True


class _Metrics:
    pattern_retrievals = 0
    pattern_accepted = 0
    pattern_rejected = 0
    verify_accept_count = 0
    verify_reject_count = 0

    def record_model_call(self, **kwargs):
        pass


class _Router:
    def call(self, task, prompt, json=False, schema=None):
        assert task == "pattern_verify"
        return {"approve": True, "reason": "ok"}, {"sent": 10, "received": 5}


def main() -> None:
    agent = _Agent()
    router = ExecutionRouter(tau_high=0.75, tau_mid=0.50)
    valid = {"activate oven", "go to kitchen", "look around"}

    route = router.route(
        agent,
        milestone="activate oven",
        match_kind="exact",
        env_valid=valid,
        task="melt lead",
        observation=agent.observation,
        pattern_confidence=0.85,
        pattern_id="p1",
    )
    assert route.reuse_quality > 0.5, route.reuse_quality
    assert route.level == ComputationLevel.DIRECT_REUSE, route.level

    fast = FastExecutor().execute(
        agent, route, env_valid=valid, task="melt lead", observation=agent.observation,
    )
    assert fast is not None
    assert fast.action == "activate oven"
    assert fast.computation_level == int(route.level)
    assert fast.source == "pattern_fast"

    mid_route = router.route(
        agent,
        milestone="activate oven",
        match_kind="verb",
        env_valid=valid,
        observation=agent.observation,
        pattern_confidence=0.55,
        pattern_id="p1",
    )
    assert mid_route.level == ComputationLevel.VERIFIED_REUSE, mid_route.level

    agent.router = _Router()
    verified = FastExecutor().execute(
        agent,
        mid_route,
        env_valid=valid,
        task="melt lead",
        observation=agent.observation,
    )
    assert verified is not None
    assert verified.source == "pattern_verify"
    assert verified.computation_level == int(ComputationLevel.VERIFIED_REUSE)

    agent.router = None
    blocked_verify = FastExecutor().execute(
        agent,
        mid_route,
        env_valid=valid,
        task="melt lead",
        observation=agent.observation,
    )
    assert blocked_verify is None, "VERIFIED_REUSE without router must fail closed"

    agent.episode_guard.note_pattern_failure("p1", "activate oven")
    blocked = router.route(
        agent,
        milestone="activate oven",
        match_kind="exact",
        env_valid=valid,
        pattern_confidence=0.85,
        pattern_id="p1",
    )
    assert blocked.level == ComputationLevel.FULL_COGNITION

    agent.cwme_cfg.pattern_verify_enabled = False
    agent.episode_guard = EpisodeGuard()
    disabled_mid = router.route(
        agent,
        milestone="activate oven",
        match_kind="verb",
        env_valid=valid,
        observation=agent.observation,
        pattern_confidence=0.55,
        pattern_id="p1",
    )
    assert disabled_mid.level == ComputationLevel.FULL_COGNITION, disabled_mid.level
    assert disabled_mid.reason == "verify_disabled", disabled_mid.reason

    print("Execution router sanity checks passed.")


if __name__ == "__main__":
    main()
