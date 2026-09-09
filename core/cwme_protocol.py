"""
CWME Default Protocol — unified hyperparameters for paper experiments (E1–E7).

Applied at ExperimentRunner load time when keys are absent from YAML.
Explicit YAML values always win (E6 sweeps, ablations).
"""
from __future__ import annotations

from copy import deepcopy

# Paper-frozen defaults (see config/experiments/CWME_PROTOCOL.md)
CWME_AGENT_DEFAULTS = {
    "MAX_LOOK_AHEAD": 5,
    "MAX_QUERY": 3,
}

CWME_EXPERIMENT_DEFAULTS = {
    # ExperimentRunner indexes this field directly. Keep standalone paper
    # configs runnable without depending on config/config.yml.
    "NUM_AGENTS": 1,
    "PATTERN_MAX_PER_SIG": 5,
    "PATTERN_MIN_RECORD_SCORE": 30,
    "PATTERN_MIN_PARTIAL_SCORE": 20,
    "SIMPLIFICATION": "easy",
}

CWME_PROCEDURAL_DEFAULTS = {
    "source": "train_artifact",
    "allow_missing_pattern": False,
    "validation_threshold": 0.65,
    "revision": True,
    "consolidation": True,
}

CWME_AGENT_DEFAULTS_EXTRA = {
    "USE_TEMPORAL_KG": True,
    "USE_ACTOR_CRITIC": True,
}

CWME_WORLD_MODEL_DEFAULTS = {
    "update_test_kg": False,
    "allow_missing_hkg": False,
}

CWME_EXECUTION_DEFAULTS = {
    "fast_path_enabled": True,
    "pattern_verify_enabled": True,
    "reuse_threshold_high": 0.75,
    "reuse_threshold_mid": 0.50,
    "grounding_mode": "grounding_only",
    "task_heuristics_enabled": False,
    "legacy_heuristics_enabled": False,
    # Keep legacy formal configs unchanged; online CWME configs opt into
    # hybrid explicitly so ablations remain interpretable.
    "cwme_initial_plan_mode": "llm",
    "cwme_heuristic_fallback": False,
    "cwme_actor_stuck_threshold": 5,
    "cwme_stall_replan_threshold": 5,
    "cwme_skip_initial_plan_if_pattern": True,
    "cwme_actor_max_attempts": 2,
}

CWME_ARTIFACT_FREEZE_DEFAULTS = {
    "enforce": True,
    "require_hkg": True,
    "require_pattern": True,
}


def _is_formal_experiment(exp: dict) -> bool:
    """Formal paper experiments require artifact freeze unless opted out."""
    if exp.get("FORMAL_EXPERIMENT") is False:
        return False
    if exp.get("FORMAL_EXPERIMENT") is True:
        return True
    exp_id = str(exp.get("EXPERIMENT_ID", "") or "").upper()
    if exp_id.startswith("SANITY") or exp_id in {"PREP", "PREP-TRAIN", "PREP_TRAIN"}:
        return False
    return exp_id.startswith(("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"))


def _apply_artifact_freeze_defaults(exp: dict) -> None:
    if not _is_formal_experiment(exp):
        freeze = exp.setdefault("ARTIFACT_FREEZE", {})
        if "enforce" not in freeze:
            freeze["enforce"] = False
        return
    freeze = exp.setdefault("ARTIFACT_FREEZE", {})
    for k, v in CWME_ARTIFACT_FREEZE_DEFAULTS.items():
        _setdefault_nested(freeze, k, v)
    _setdefault_nested(freeze, "enforce", True)
    role = str(exp.get("SCIENTIFIC_ROLE") or "")
    if role == "main_comparison_frozen_evaluation" or str(exp.get("EXPERIMENT_ID", "")).upper() in {
        "E1-FROZEN", "E1_STATIC", "E1-STATIC",
    }:
        _setdefault_nested(exp, "ASSERT_NO_TEST_WRITEBACK", True)


def _setdefault_nested(target: dict, key: str, value) -> None:
    if key not in target or target[key] is None:
        target[key] = value


def apply_cwme_protocol_defaults(agent_config: dict | None) -> dict:
    """Fill missing CWME protocol fields; return the same dict (mutated)."""
    if not agent_config:
        return {}
    cfg = agent_config

    agent = cfg.setdefault("AGENT", {})
    for k, v in CWME_AGENT_DEFAULTS.items():
        _setdefault_nested(agent, k, v)
    for k, v in CWME_AGENT_DEFAULTS_EXTRA.items():
        _setdefault_nested(agent, k, v)

    wm = cfg.setdefault("WORLD_MODEL", {})
    if wm.get("historical_kg", True):
        for k, v in CWME_WORLD_MODEL_DEFAULTS.items():
            _setdefault_nested(wm, k, v)

    exp = cfg.setdefault("EXPERIMENT", {})
    for k, v in CWME_EXPERIMENT_DEFAULTS.items():
        _setdefault_nested(exp, k, v)
    _apply_artifact_freeze_defaults(exp)

    pm = cfg.setdefault("PROCEDURAL_MEMORY", {})
    if pm.get("enabled", True):
        for k, v in CWME_PROCEDURAL_DEFAULTS.items():
            _setdefault_nested(pm, k, v)

    exec_cfg = cfg.setdefault("EXECUTION", {})
    for k, v in CWME_EXECUTION_DEFAULTS.items():
        _setdefault_nested(exec_cfg, k, v)

    return cfg


def merged_protocol_view(agent_config: dict | None) -> dict:
    """Return effective protocol after defaults (for logging / manifest)."""
    cfg = apply_cwme_protocol_defaults(deepcopy(agent_config or {}))
    agent = cfg.get("AGENT") or {}
    exp = cfg.get("EXPERIMENT") or {}
    pm = cfg.get("PROCEDURAL_MEMORY") or {}
    exec_cfg = cfg.get("EXECUTION") or {}
    return {
        "max_look_ahead": agent.get("MAX_LOOK_AHEAD"),
        "max_query": agent.get("MAX_QUERY"),
        "pattern_k": exp.get("PATTERN_MAX_PER_SIG"),
        "pattern_min_record_score": exp.get("PATTERN_MIN_RECORD_SCORE"),
        "pattern_min_partial_score": exp.get("PATTERN_MIN_PARTIAL_SCORE"),
        "validation_threshold": pm.get("validation_threshold"),
        "reuse_threshold_high": exec_cfg.get("reuse_threshold_high"),
        "reuse_threshold_mid": exec_cfg.get("reuse_threshold_mid"),
        "fast_path_enabled": exec_cfg.get("fast_path_enabled"),
        "learned_reuse_enabled": exec_cfg.get("learned_reuse_enabled"),
        "pattern_verify_enabled": exec_cfg.get("pattern_verify_enabled"),
    }
