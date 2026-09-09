"""
Unified CWME experiment configuration.

Maps YAML sections to agent runtime:
  BASELINE / WORLD_MODEL / PROCEDURAL_MEMORY / CONTINUAL / EPISODE_GUARD
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from core.evolution.continual_engine import EvolutionConfig, resolve_evolution_config
from core.execution.execution_router import ExecutionRouter
from core.world_model.hkg_config import HKGConfig, resolve_hkg_config


def _artifact_path(value: str, repo_root: str) -> str:
    path = str(value or "").strip()
    if not path:
        return ""
    if repo_root and not os.path.isabs(path):
        path = os.path.join(repo_root, path)
    return os.path.abspath(path)


def _pattern_candidates(agent_config: dict | None, *, repo_root: str = "") -> list[str]:
    """Return P_train candidates without treating an online output as input.

    ``persist_path`` is deliberately absent here. It is the destination for
    the current run's delta memory. P_train must be explicit or come from the
    immutable artifact namespace selected by the experiment.
    """
    cfg = agent_config or {}
    pm = dict(cfg.get("PROCEDURAL_MEMORY") or {})
    exp = dict(cfg.get("EXPERIMENT") or {})
    explicit = pm.get("load_path") or pm.get("procedural_load_path")
    if explicit:
        base = _artifact_path(explicit, repo_root)
        return [base] if base.endswith(".procedural.jsonl") else [base, f"{base}.procedural.jsonl"]

    set_name = str(exp.get("SET", "") or "").strip().lower()
    source = str(pm.get("source", "") or "").strip().lower()
    # A self-evolution train run must start empty. Offline train_artifact and
    # every non-train evaluation may consume an existing frozen P_train.
    may_autoload = source == "train_artifact" or set_name not in {"", "train"}
    if not may_autoload:
        return []

    raw: list[str] = []
    namespace = str(exp.get("ARTIFACT_NAMESPACE", "") or "").strip()
    if namespace:
        raw.extend([
            os.path.join(namespace, "procedural_memory", "patterns.jsonl"),
            os.path.join(namespace, "patterns.jsonl"),
        ])
    raw.append(os.path.join("artifacts", "procedural_memory", "patterns.jsonl"))

    candidates: list[str] = []
    for value in raw:
        base = _artifact_path(value, repo_root)
        if base.endswith(".procedural.jsonl"):
            candidates.append(base)
        else:
            candidates.extend([base, f"{base}.procedural.jsonl"])
    return candidates


def resolve_pattern_load_path(
    agent_config: dict | None,
    *,
    repo_root: str = "",
    require_exists: bool = False,
) -> str:
    """Resolve the immutable P_train path consumed by runtime READ."""
    candidates = _pattern_candidates(agent_config, repo_root=repo_root)
    for path in candidates:
        if os.path.isfile(path):
            return path
    return "" if require_exists else (candidates[0] if candidates else "")


@dataclass
class CWMEExperimentConfig:
    hkg: HKGConfig
    evolution: EvolutionConfig
    mode: str = "cwme"  # base | hkg | static_pm | cwme
    pattern_read: bool = True
    pattern_write: bool = True
    pattern_persist: bool = False
    pattern_persist_path: str = ""
    procedural_persist_path: str = ""
    pattern_reuse_mode: str = "hard"  # soft | hard
    fast_path_enabled: bool = True
    # Learned Pattern reuse is independent from the legacy fast-path switch.
    learned_reuse_enabled: bool = False
    pattern_verify_enabled: bool = True
    reuse_threshold_high: float = 0.75
    reuse_threshold_mid: float = 0.50
    episode_guard_enabled: bool = True
    update_test_kg: bool = False
    task_heuristics_enabled: bool = False
    legacy_heuristics_enabled: bool = False
    grounding_mode: str = "grounding_only"
    # The initial plan and the task-family fallback are separate from the
    # formal grounding mode.  This lets the main online protocol use cheap,
    # admissible environment actions while retaining Actor cognition for
    # states that the policy cannot advance.
    initial_plan_mode: str = "llm"  # llm | hybrid | heuristic
    heuristic_fallback_enabled: bool = False
    actor_stuck_threshold: int = 5
    stall_replan_threshold: int = 5
    empty_action_replan_attempts: int = 2
    # Avoid paying for an open-loop 9B plan when P_train already provides a
    # grounded first milestone.  This is a memory optimization, not a task
    # heuristic.
    skip_initial_plan_if_pattern: bool = True
    actor_max_attempts: int = 2
    initial_plan_steps: int = 5
    refiner_max_attempts: int = 2


def resolve_cwme_experiment(
    agent_config: dict | None,
    *,
    repo_root: str = "",
) -> CWMEExperimentConfig:
    agent_config = agent_config or {}
    baseline = dict(agent_config.get("BASELINE") or {})
    pm = dict(agent_config.get("PROCEDURAL_MEMORY") or {})
    exp = dict(agent_config.get("EXPERIMENT") or {})

    if baseline.get("historical_kg") is False and not agent_config.get("WORLD_MODEL"):
        mode = "base"
    elif pm.get("enabled") and pm.get("read") and not pm.get("write", True):
        mode = "static_pm"
    elif resolve_hkg_config(agent_config, repo_root=repo_root).enabled and not resolve_evolution_config(agent_config).enabled:
        hkg_cfg = resolve_hkg_config(agent_config, repo_root=repo_root)
        mode = "hkg_continual" if hkg_cfg.update_test_kg else "hkg"
    else:
        mode = "cwme"

    hkg = resolve_hkg_config(agent_config, repo_root=repo_root)
    evo = resolve_evolution_config(agent_config)

    if baseline.get("historical_kg") is False:
        hkg.enabled = False
    if baseline.get("procedural_memory") is False:
        pm_enabled = False
    else:
        pm_enabled = bool(pm.get("enabled", True))

    pattern_read = bool(
        pm.get("read", exp.get("ENABLE_PATTERN_READ", True))
        if pm_enabled or "read" in pm
        else exp.get("ENABLE_PATTERN_READ", True)
    )
    pattern_write = bool(
        pm.get("write", exp.get("ENABLE_PATTERN_WRITE", True))
        if pm_enabled
        else exp.get("ENABLE_PATTERN_WRITE", True)
    )
    if not evo.enabled:
        pattern_write = False if "write" not in pm else bool(pm.get("write", False))

    log_root = (exp.get("LOG_ROOT") or "").strip()
    pattern_persist = bool(exp.get("PATTERN_PERSIST", False) or pm.get("persist", False))
    pattern_path = ""
    proc_path = ""
    # Explicit artifact paths are authoritative. Otherwise two variants can
    # silently share LOG_ROOT/patterns.jsonl.
    if pm.get("persist_path"):
        pattern_path = _artifact_path(pm["persist_path"], repo_root)
        proc_path = f"{pattern_path}.procedural.jsonl"
    elif pattern_persist and log_root:
        pattern_path = _artifact_path(os.path.join(log_root, "patterns.jsonl"), repo_root)
        proc_path = f"{pattern_path}.procedural.jsonl"
    elif pm.get("load_path"):
        pattern_path = _artifact_path(pm["load_path"], repo_root)
        proc_path = f"{pattern_path}.procedural.jsonl"

    guard_cfg = dict(agent_config.get("EPISODE_GUARD") or {})
    guard_on = bool(guard_cfg.get("enabled", True))
    reuse_mode = str(
        pm.get("reuse_mode")
        or exp.get("PATTERN_REUSE_MODE")
        or "hard"
    ).strip().lower()
    if reuse_mode not in ("soft", "hard"):
        reuse_mode = "hard"

    exec_cfg = dict(agent_config.get("EXECUTION") or {})
    fast_path_enabled = bool(exec_cfg.get("fast_path_enabled", True))
    learned_reuse_enabled = bool(
        exec_cfg.get("learned_reuse_enabled", fast_path_enabled)
    )
    pattern_verify_enabled = bool(exec_cfg.get("pattern_verify_enabled", True))
    tau_high = float(exec_cfg.get("reuse_threshold_high", 0.75) or 0.75)
    tau_mid = float(exec_cfg.get("reuse_threshold_mid", 0.50) or 0.50)
    task_heuristics = bool(exec_cfg.get("task_heuristics_enabled", False))
    legacy_heuristics = bool(exec_cfg.get("legacy_heuristics_enabled", False))
    initial_plan_mode = str(
        exec_cfg.get("cwme_initial_plan_mode", exp.get("CWME_INITIAL_PLAN_MODE", "llm"))
        or "llm"
    ).strip().lower()
    if initial_plan_mode not in ("llm", "hybrid", "heuristic"):
        initial_plan_mode = "llm"
    heuristic_fallback = bool(
        exec_cfg.get("cwme_heuristic_fallback", exp.get("CWME_HEURISTIC_FALLBACK", False))
    )
    actor_stuck_threshold = max(
        1,
        int(exec_cfg.get("cwme_actor_stuck_threshold", 5) or 5),
    )
    stall_replan_threshold = max(
        1,
        int(exec_cfg.get("cwme_stall_replan_threshold", 5) or 5),
    )
    empty_action_replan_attempts = max(
        1,
        int(exec_cfg.get("cwme_empty_action_replan_attempts", 2) or 2),
    )
    skip_initial_plan_if_pattern = bool(
        exec_cfg.get("cwme_skip_initial_plan_if_pattern", True)
    )
    actor_max_attempts = max(
        1,
        int(exec_cfg.get("cwme_actor_max_attempts", 2) or 2),
    )
    initial_plan_steps = max(
        1,
        int(exec_cfg.get("cwme_initial_plan_steps", 5) or 5),
    )
    refiner_max_attempts = max(
        1,
        int(exec_cfg.get("cwme_refiner_max_attempts", 2) or 2),
    )
    from core.grounding.grounding_mode import resolve_grounding_mode_from_config
    grounding_mode = resolve_grounding_mode_from_config(agent_config).value

    return CWMEExperimentConfig(
        mode=mode,
        hkg=hkg,
        evolution=evo,
        pattern_read=pattern_read,
        pattern_write=pattern_write,
        pattern_persist=pattern_persist,
        pattern_persist_path=pattern_path,
        procedural_persist_path=proc_path,
        pattern_reuse_mode=reuse_mode,
        fast_path_enabled=fast_path_enabled,
        learned_reuse_enabled=learned_reuse_enabled,
        pattern_verify_enabled=pattern_verify_enabled,
        reuse_threshold_high=tau_high,
        reuse_threshold_mid=tau_mid,
        episode_guard_enabled=guard_on,
        update_test_kg=bool(hkg.update_test_kg),
        task_heuristics_enabled=task_heuristics,
        legacy_heuristics_enabled=legacy_heuristics,
        grounding_mode=grounding_mode,
        initial_plan_mode=initial_plan_mode,
        heuristic_fallback_enabled=heuristic_fallback,
        actor_stuck_threshold=actor_stuck_threshold,
        stall_replan_threshold=stall_replan_threshold,
        empty_action_replan_attempts=empty_action_replan_attempts,
        skip_initial_plan_if_pattern=skip_initial_plan_if_pattern,
        actor_max_attempts=actor_max_attempts,
        initial_plan_steps=initial_plan_steps,
        refiner_max_attempts=refiner_max_attempts,
    )


def apply_cwme_experiment(agent, agent_config: dict | None, *, repo_root: str = "") -> CWMEExperimentConfig:
    """Apply CWME YAML settings onto a constructed ReasoningAgent."""
    agent_config = agent_config or {}
    pm = dict(agent_config.get("PROCEDURAL_MEMORY") or {})
    cfg = resolve_cwme_experiment(agent_config, repo_root=repo_root)
    plib = getattr(agent, "pattern_library", None)
    if plib is not None:
        plib.enable_read = cfg.pattern_read
        plib.enable_write = cfg.pattern_write
        if cfg.pattern_persist_path and cfg.pattern_persist:
            plib.persist_path = cfg.pattern_persist_path
            plib.delta_persist_path = cfg.pattern_persist_path
        load_proc = resolve_pattern_load_path(
            agent_config,
            repo_root=repo_root,
            require_exists=False,
        )
        allow_missing = bool(pm.get("allow_missing_pattern", False))
        is_train = str((agent_config.get("EXPERIMENT") or {}).get("SET", "")).strip().lower() == "train"
        if load_proc and not os.path.isfile(load_proc) and not allow_missing and not is_train:
            candidates = _pattern_candidates(agent_config, repo_root=repo_root)
            if not any(os.path.isfile(p) for p in candidates):
                exp_set = (agent_config.get("EXPERIMENT") or {}).get("SET", "")
                if str(exp_set).strip().lower() != "train":
                    raise RuntimeError(
                        f"[Pattern] procedural memory enabled but artifact missing: {load_proc}. "
                        "Build P_train offline or set PROCEDURAL_MEMORY.allow_missing_pattern: true."
                    )
        if load_proc and os.path.isfile(load_proc):
            if load_proc.endswith(".procedural.jsonl"):
                plib.load_train_procedural(load_proc)
            else:
                plib.load_train_procedural(load_proc, legacy=True)
            if logger := getattr(agent, "logger", None):
                logger.info(
                    "[Pattern] loaded P_train: patterns=%s path=%s",
                    plib.train_pattern_count(),
                    plib.train_load_path,
                )
        elif getattr(agent, "logger", None) and cfg.pattern_read and not is_train:
            agent.logger.warning(
                "[Pattern] P_train not loaded: resolved_path=%s; read=%s",
                load_proc or "<none>",
                cfg.pattern_read,
            )
        # ``procedural_persist_path`` is a test-time ΔP destination.  It is
        # intentionally never used as a fallback P_train source.

        exp = dict(agent_config.get("EXPERIMENT") or {})
        if "PATTERN_MAX_PER_SIG" in exp:
            plib.max_patterns_per_sig = int(exp["PATTERN_MAX_PER_SIG"] or 5)
        elif "PATTERN_MAX_PER_SIG" in pm:
            plib.max_patterns_per_sig = int(pm["PATTERN_MAX_PER_SIG"] or 5)
        if hasattr(plib, "apply_retrieval_budget"):
            plib.apply_retrieval_budget()

    evo = getattr(agent, "evolution_engine", None)
    if evo is not None:
        evo.config = cfg.evolution

    if getattr(agent, "hkg_config", None) is not None:
        agent.hkg_config.enabled = cfg.hkg.enabled
        agent.hkg_config.update_test_kg = cfg.hkg.update_test_kg

    if not cfg.episode_guard_enabled:
        agent.episode_guard = None

    router = getattr(agent, "orchestrator", None)
    if router is not None and hasattr(router, "execution_router"):
        router.execution_router = ExecutionRouter(
            tau_high=cfg.reuse_threshold_high,
            tau_mid=cfg.reuse_threshold_mid,
        )

    logger = getattr(agent, "logger", None)
    if logger:
        logger.info(
            "[CWME] mode=%s hkg=%s pattern(read=%s,write=%s,reuse=%s,fast=%s,learned_reuse=%s) "
            "continual=%s pattern_K=%s train=%s delta=%s",
            cfg.mode,
            cfg.hkg.enabled,
            cfg.pattern_read,
            cfg.pattern_write,
            cfg.pattern_reuse_mode,
            cfg.fast_path_enabled,
            cfg.learned_reuse_enabled,
            cfg.evolution.enabled,
            getattr(plib, "max_patterns_per_sig", "?") if plib is not None else "?",
            getattr(plib, "train_pattern_count", lambda: 0)() if plib is not None else 0,
            getattr(plib, "delta_pattern_count", lambda: 0)() if plib is not None else 0,
        )
    return cfg


def pattern_hard_reuse_enabled(agent) -> bool:
    """True when pattern may override cognition (hard reuse); False for soft/suggest-only."""
    cfg = getattr(agent, "cwme_cfg", None)
    if cfg is None:
        return True
    mode = str(getattr(cfg, "pattern_reuse_mode", "hard") or "hard").strip().lower()
    return mode != "soft"
