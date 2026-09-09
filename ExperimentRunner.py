import json
import logging
import asyncio
import os
import random
import pickle
import sys
import uuid
import time
from multiprocessing import Process
import numpy as np

# Headless backend — avoids GUI/display hangs after long GPU runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ReasoningAgent import ReasoningAgent
from core.model_router import resolve_agent_models
from envs import create_env, normalize_environment_name
from datetime import datetime
import yaml
from concurrent.futures import ProcessPoolExecutor
from utils import *

# Paths relative to this file so runs do not depend on shell cwd.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_INI = os.path.join(_REPO_ROOT, "config", "config.ini")
_CONFIG_YML = os.path.join(_REPO_ROOT, "config", "config.yml")
# _DEFAULT_LOG_ROOT = "/mnt/nfsData19/Zhaoshuyuan/Houxinrui/WorldModel/DAVIS-main_Qwen3.5-9B"
_DEFAULT_LOG_ROOT = "/data/ZhaoShuyuan/Zhaoshuyuan/HouXinrui/Baseline/Qwen3.5-9B"


def _resolve_config_yml_path() -> str:
    """Allow ``python ExperimentRunner.py [config.yml]`` for CL-on/off ablations."""
    if len(sys.argv) > 1 and str(sys.argv[1]).endswith((".yml", ".yaml")):
        path = sys.argv[1]
        if not os.path.isabs(path):
            path = os.path.join(_REPO_ROOT, path)
        return os.path.abspath(path)
    return _CONFIG_YML


def resolve_log_root(agent_config: dict) -> str:
    """Directory for experiment logs and result.log (configurable via EXPERIMENT.LOG_ROOT)."""
    exp = agent_config.get("EXPERIMENT", {}) if agent_config else {}
    root = (exp.get("LOG_ROOT") or _DEFAULT_LOG_ROOT).strip()
    return os.path.abspath(root)


def redirect_console_to_result_log(log_root: str) -> str:
    """Send stdout/stderr to LOG_ROOT/result.log so nohup shell redirect is optional."""
    os.makedirs(log_root, exist_ok=True)
    result_path = os.path.join(log_root, "result.log")
    result_file = open(result_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = result_file
    sys.stderr = result_file
    return result_path


def setup_experiment_logger(log_file, agent_id=None):
    logger_name = f'experiment_logger_{agent_id}' if agent_id else 'experiment_logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Load configuration (absolute paths → reproducible regardless of cwd)
_CONFIG_YML = _resolve_config_yml_path()
config = load_config(_CONFIG_INI)
if config is None:
    raise SystemExit(f"Missing or invalid config.ini: {_CONFIG_INI}")
with open(_CONFIG_YML, "r", encoding="utf-8") as agentconfig:
    agent_config = yaml.load(agentconfig, Loader=yaml.FullLoader)

from core.cwme_protocol import apply_cwme_protocol_defaults

apply_cwme_protocol_defaults(agent_config)

_LOG_ROOT = resolve_log_root(agent_config)
_RESULT_LOG = redirect_console_to_result_log(_LOG_ROOT)
print("Successfully read!")
print(f"Log root: {_LOG_ROOT}")
print(f"Console output -> {_RESULT_LOG}")

# Get seed, hyperparameters, and task from the configuration
_agent = agent_config.get("AGENT") or {}
seed = int(_agent.get("SEED", 4901))
max_look_ahead = int(_agent.get("MAX_LOOK_AHEAD", 5))
max_query = int(_agent.get("MAX_QUERY", 3))

random.seed(seed)
np.random.seed(seed)

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Ensure the global experiment directory exists (under NFS LOG_ROOT)
global_experiment_dir = os.path.join(_LOG_ROOT, f"log/experiment_{timestamp}")
os.makedirs(global_experiment_dir, exist_ok=True)

# Run-scoped KG namespace: same AGENT.ID across days will not read prior runs.
agent_config.setdefault("EXPERIMENT", {})
agent_config["EXPERIMENT"]["KG_RUN_ID"] = f"experiment_{timestamp}"

# Save configuration file in the global experiment directory
config_save_path = os.path.join(global_experiment_dir, 'config.yml')
with open(config_save_path, 'w') as outfile:
    yaml.dump(agent_config, outfile)

try:
    from core.experiment_protocol import write_protocol_snapshot, write_artifact_manifest
    _proto_path = write_protocol_snapshot(
        agent_config,
        global_experiment_dir,
        repo_root=_REPO_ROOT,
        config_yml_path=_CONFIG_YML,
    )
except Exception as _proto_exc:
    _proto_path = ""
    print(f"Warning: protocol.json not written: {_proto_exc}")

try:
    from core.experiment_protocol import write_artifact_manifest
    _manifest_path = write_artifact_manifest(
        agent_config,
        repo_root=_REPO_ROOT,
    )
    if _manifest_path:
        print(f"Artifact manifest -> {_manifest_path}")
except Exception as _manifest_exc:
    raise RuntimeError(f"Artifact namespace validation failed: {_manifest_exc}") from _manifest_exc

# Setup global logger
global_log_file = os.path.join(global_experiment_dir, 'global_experiment.log')
global_logger = setup_experiment_logger(global_log_file)

global_logger.info(f"Seed: {seed}")
global_logger.info(f"Log root: {_LOG_ROOT}")
global_logger.info(f"Result log: {_RESULT_LOG}")
global_logger.info(f"Experiment dir: {global_experiment_dir}")
global_logger.info(f'Max look ahead: {max_look_ahead} steps')
global_logger.info(f'Max query: {max_query} queries')

# Initialize the agent parameters
agent_id = agent_config['AGENT']['ID']
cognition_model, perception_model = resolve_agent_models(agent_config)
agent_model = cognition_model
agent_kgmodel = perception_model
global_logger.info(f"Cognition model: {cognition_model}")
global_logger.info(f"Perception model: {perception_model}")


def _resolve_base_kg_uuid(agent_config: dict) -> str:
    """UUID of the pretrained / training world model (never wiped by CLEAN_KG_ON_EXIT)."""
    exp = agent_config.get("EXPERIMENT") or {}
    base = (exp.get("BASE_KG_UUID") or "").strip()
    if base:
        return base
    return str((agent_config.get("AGENT") or {}).get("ID") or "agent")


def _resolve_kg_uuid(agent_config: dict) -> str:
    """
    KG rows are keyed by agent UUID in Postgres (column type uuid).

    RUN_SCOPED_KG (default True):
      Deterministic UUIDv5(base_kg_uuid, KG_RUN_ID) so one ExperimentRunner keeps
      writes across episodes, while a later process cannot read that run's deltas.
      Default start is an empty run namespace (本次专用空库). Optional SEED_BASE_KG
      can copy BASE_KG_UUID → this run uuid.
    RUN_SCOPED_KG False:
      uuid = BASE_KG_UUID / AGENT.ID (legacy; writes mutate the base graph).
    """
    base = _resolve_base_kg_uuid(agent_config)
    exp = agent_config.get("EXPERIMENT") or {}
    if not bool(exp.get("RUN_SCOPED_KG", True)):
        return base
    run_id = (exp.get("KG_RUN_ID") or "").strip()
    if not run_id:
        run_id = datetime.now().strftime("adhoc_%Y%m%d_%H%M%S")
    try:
        base_u = uuid.UUID(str(base))
    except ValueError:
        # Non-UUID agent ids: derive a stable namespace then uuid5.
        base_u = uuid.uuid5(uuid.NAMESPACE_URL, f"pcm-davis-base:{base}")
    return str(uuid.uuid5(base_u, run_id))


def split_tasks_across_agents(tasks, num_agents):
    """Partition `tasks` into `num_agents` contiguous slices (some may be empty). Avoids zero-sized chunks."""
    if num_agents < 1:
        num_agents = 1
    if not tasks:
        return [[] for _ in range(num_agents)]
    n = len(tasks)
    base, extra = divmod(n, num_agents)
    out = []
    idx = 0
    for i in range(num_agents):
        take = base + (1 if i < extra else 0)
        out.append(tasks[idx : idx + take])
        idx += take
    return out


def _resolve_variations(env, set_name, variation_list=None, logger=None, task=None):
    """Resolve shared or task-local variation indices."""
    if isinstance(variation_list, dict):
        key = str(task or "").strip()
        if key not in variation_list:
            raise ValueError(
                f"No task-local VARIATIONS configured for task {task!r}; "
                f"available tasks={sorted(variation_list)}"
            )
        variation_list = variation_list[key]
    if variation_list is not None:
        if logger and set_name != "train":
            try:
                test_ids = set(env.getVariationsTest())
                train_ids = set(env.getVariationsTrain())
                explicit = [int(v) for v in variation_list]
                if explicit and all(v in train_ids and v not in test_ids for v in explicit):
                    logger.warning(
                        "VARIATIONS %s look like TRAIN indices but SET=%s — "
                        "remove VARIATIONS to use test splits, or set SET: train.",
                        explicit,
                        set_name,
                    )
            except Exception:
                pass
        return list(variation_list)
    return load_variation(env, set_name)


def _should_keep_train_patterns(agent, agent_config) -> bool:
    plib = getattr(agent, "pattern_library", None)
    if plib is None:
        return False
    pm = dict(agent_config.get("PROCEDURAL_MEMORY") or {})
    pattern_source = str(pm.get("source", "") or "").strip().lower()
    if pattern_source == "train_artifact":
        return True
    if hasattr(plib, "has_train_patterns") and plib.has_train_patterns():
        return True
    return plib.procedural_size() > 0


def _clear_pattern_library(agent, agent_config, *, full: bool = False, reason: str = "") -> None:
    plib = getattr(agent, "pattern_library", None)
    if plib is None:
        return
    keep_train = _should_keep_train_patterns(agent, agent_config) and not full
    if keep_train and hasattr(plib, "clear_delta"):
        plib.clear_delta()
        global_logger.info(
            "Cleared PatternLibrary delta only (%s); kept P_train=%s",
            reason or "scope",
            plib.train_pattern_count(),
        )
    else:
        plib.clear()
        global_logger.info("Cleared PatternLibrary (%s)", reason or "full")


def _resolve_schedule_path(path: str, repo_root: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def _load_schedule_document(path: str, repo_root: str):
    """Load and minimally validate a frozen schedule document."""
    if not path:
        return None
    full = _resolve_schedule_path(path, repo_root)
    if not os.path.isfile(full):
        return None
    with open(full, encoding="utf-8") as fh:
        blob = json.load(fh)
    rows = blob.get("episodes") if isinstance(blob, dict) else blob
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid schedule file (empty episodes): {full}")
    schedule: list[tuple[str, int]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "task" not in row or "variation" not in row:
            raise ValueError(f"Invalid schedule row {i} in {full}")
        schedule.append((str(row["task"]), int(row["variation"])))
    return full, blob, schedule


def _load_schedule_file(path: str, repo_root: str) -> list[tuple[str, int]] | None:
    """Load frozen episode schedule from JSON (artifacts/schedules/*.json)."""
    document = _load_schedule_document(path, repo_root)
    return document[2] if document is not None else None


def _validate_formal_schedule(
    schedule: list[tuple[str, int]],
    *,
    blob,
    tasks,
    expected_episodes: int,
    path: str,
    strict: bool,
    legal_variations: dict[str, set[int]] | None = None,
) -> None:
    """Reject a paired formal run when its schedule is semantically stale."""
    if not strict:
        return
    normalize = lambda value: str(value or "").strip().lower()
    expected_tasks = {normalize(task) for task in (tasks or []) if normalize(task)}
    schedule_tasks = {normalize(task) for task, _ in schedule if normalize(task)}
    errors = []
    if schedule_tasks != expected_tasks:
        errors.append(
            "task set mismatch: config=%s schedule=%s"
            % (sorted(expected_tasks), sorted(schedule_tasks))
        )
    if expected_episodes > 0 and len(schedule) != expected_episodes:
        errors.append(
            "episode count mismatch: config=%s schedule=%s"
            % (expected_episodes, len(schedule))
        )
    if legal_variations is not None:
        invalid_pairs = []
        for task, variation in schedule:
            task_key = normalize(task)
            allowed = legal_variations.get(task_key)
            if allowed is not None and int(variation) not in allowed:
                invalid_pairs.append(
                    f"{task}@{int(variation)} (allowed={sorted(allowed)})"
                )
        if invalid_pairs:
            errors.append(
                "invalid task-local variations: " + ", ".join(invalid_pairs[:12])
                + (" ..." if len(invalid_pairs) > 12 else "")
            )
    if isinstance(blob, dict):
        meta = blob.get("meta")
        if isinstance(meta, dict) and "tasks" in meta:
            meta_tasks = {normalize(task) for task in (meta.get("tasks") or []) if normalize(task)}
            if meta_tasks != expected_tasks:
                errors.append(
                    "meta.tasks mismatch: config=%s schedule=%s"
                    % (sorted(expected_tasks), sorted(meta_tasks))
                )
        if isinstance(meta, dict) and meta.get("num_episodes") is not None:
            try:
                meta_count = int(meta["num_episodes"])
            except (TypeError, ValueError):
                errors.append("meta.num_episodes is not an integer")
            else:
                if meta_count != len(schedule):
                    errors.append(
                        "meta.num_episodes mismatch: meta=%s rows=%s"
                        % (meta_count, len(schedule))
                    )
    if errors:
        raise RuntimeError(
            "Formal continual schedule validation failed for %s:\n  %s\n"
            "Re-freeze the schedule from the same experiment config."
            % (path, "\n  ".join(errors))
        )


def _load_schedule_legal_variations(env, tasks, set_name: str, simplification: str):
    """Resolve task-local variation IDs for a formal ScienceWorld schedule."""
    # ScienceWorld variation IDs are task-local.  A shared list such as
    # ``[24, 25, 26, 27, 28]`` is therefore not a valid semantic contract
    # unless every task happens to expose those IDs.
    if not (
        callable(getattr(env, "load", None))
        and callable(getattr(env, "getVariationsTrain", None))
        and callable(getattr(env, "getVariationsTest", None))
    ):
        return None

    legal: dict[str, set[int]] = {}
    for task in tasks or []:
        key = str(task or "").strip().lower()
        if not key:
            continue
        try:
            env.load(
                task,
                variationIdx=0,
                simplificationStr=str(simplification or "easy"),
                generateGoldPath=True,
            )
            legal[key] = {int(v) for v in load_variation(env, set_name)}
        except Exception as exc:
            raise RuntimeError(
                f"Unable to resolve {set_name!r} variations for task {task!r} "
                "while validating the formal schedule"
            ) from exc
        if not legal[key]:
            raise RuntimeError(
                f"No {set_name!r} variations available for task {task!r} "
                "while validating the formal schedule"
            )
    return legal


def _build_episode_schedule(exp_cfg, tasks, env, set_name, variations, logger, *, repo_root: str = ""):
    """
    Optional flat episode schedule for continual curves (E2).

    Priority:
      1. CONTINUAL_CURVE.SCHEDULE_FILE — frozen JSON (paired E2 protocol)
      2. CONTINUAL_CURVE.EPISODE_SCHEDULE — inline list of {task, variation}
      3. NUM_EPISODES + SCHEDULE_MODE — dynamic (dev only; not for formal E2)
    """
    curve = dict(exp_cfg.get("CONTINUAL_CURVE") or {})
    repo_root = repo_root or _REPO_ROOT

    sched_file = curve.get("SCHEDULE_FILE") or curve.get("schedule_file") or ""
    require_file = bool(curve.get("REQUIRE_SCHEDULE_FILE", False))
    formal_schedule = bool(exp_cfg.get("FORMAL_EXPERIMENT", False)) or require_file
    if sched_file:
        document = _load_schedule_document(sched_file, repo_root)
        loaded = document[2] if document is not None else None
        if loaded is not None:
            legal_variations = None
            if formal_schedule:
                legal_variations = _load_schedule_legal_variations(
                    env,
                    tasks,
                    set_name,
                    str(exp_cfg.get("SIMPLIFICATION", "easy") or "easy"),
                )
            _validate_formal_schedule(
                loaded,
                blob=document[1],
                tasks=tasks,
                expected_episodes=int(curve.get("NUM_EPISODES", 0) or 0),
                path=document[0],
                strict=formal_schedule,
                legal_variations=legal_variations,
            )
            if logger:
                logger.info(
                    "CONTINUAL_CURVE: loaded frozen schedule (%s episodes) from %s",
                    len(loaded),
                    sched_file,
                )
            return loaded
        if formal_schedule:
            full = _resolve_schedule_path(sched_file, repo_root)
            raise RuntimeError(
                f"Formal experiment requires a frozen schedule, but it is missing: {full}. "
                "Run: python scripts/freeze_episode_schedule.py config/experiments/e02_cwme_continual_50ep.yml"
            )
        if logger:
            logger.warning(
                "CONTINUAL_CURVE.SCHEDULE_FILE not found (%s); falling back to dynamic schedule",
                sched_file,
            )

    explicit = curve.get("EPISODE_SCHEDULE")
    if explicit:
        inline = [(str(row["task"]), int(row["variation"])) for row in explicit]
        legal_variations = None
        if formal_schedule:
            legal_variations = _load_schedule_legal_variations(
                env,
                tasks,
                set_name,
                str(exp_cfg.get("SIMPLIFICATION", "easy") or "easy"),
            )
        _validate_formal_schedule(
            inline,
            blob=None,
            tasks=tasks,
            expected_episodes=int(curve.get("NUM_EPISODES", 0) or 0),
            path="<inline EPISODE_SCHEDULE>",
            strict=formal_schedule,
            legal_variations=legal_variations,
        )
        return inline

    num_ep = int(curve.get("NUM_EPISODES", 0) or 0)
    mode = str(curve.get("SCHEDULE_MODE", "interleaved") or "interleaved").strip().lower()
    if num_ep <= 0 or mode not in ("round_robin", "sequential", "interleaved"):
        return None

    if formal_schedule:
        raise RuntimeError(
            "Formal experiment requires SCHEDULE_FILE or EPISODE_SCHEDULE. "
            "Freeze schedule before formal experiments."
        )

    if mode == "round_robin":
        pairs = [(task, variation) for task in tasks for variation in _resolve_variations(env, set_name, variations, logger=logger)]
    else:
        task_vars = _resolve_variations(env, set_name, variations, logger=logger)
        pairs = [(task, variation) for variation in task_vars for task in tasks]
    if not pairs:
        return None

    if logger:
        logger.warning(
            "CONTINUAL_CURVE: using dynamic %s schedule (%s episodes) — "
            "not recommended for formal paired E2; freeze SCHEDULE_FILE instead",
            mode,
            num_ep,
        )
    return [pairs[i % len(pairs)] for i in range(num_ep)]


def run_agent(agent_config, tasks, set, simplification, global_experiment_dir, max_steps, variations=None):
    # Initialize environment and connection within the process
    try:
        connection_pool = get_connection_pool(config)
        agent_id = agent_config['AGENT']['ID']
        cognition_model, perception_model = resolve_agent_models(agent_config)
        agent_model = cognition_model
        agent_kgmodel = perception_model
        exp_cfg = agent_config.get("EXPERIMENT", {}) or {}
        base_kg_uuid = _resolve_base_kg_uuid(agent_config)
        kg_uuid = _resolve_kg_uuid(agent_config)
        store_kg = bool(exp_cfg.get("STORE_KG", False))
        clean_kg_on_exit = bool(exp_cfg.get("CLEAN_KG_ON_EXIT", True))
        # Default: empty dedicated KG for this run (do not inherit pretrained base).
        seed_base_kg = bool(exp_cfg.get("SEED_BASE_KG", False))

        env = create_env(agent_config)
        global_logger.info(
            "Environment: %s (env_name=%s)",
            normalize_environment_name(
                (agent_config.get("AGENT") or {}).get("ENVIRONMENT", "ScienceWorld")
            ),
            getattr(env, "env_name", type(env).__name__),
        )
        agent = ReasoningAgent(
            config, global_logger, connection_pool, kg_uuid, max_steps, max_look_ahead, max_query,
            env, agent_model, agent_kgmodel, agent_config=agent_config,
        )
        agent.config_agent_id = agent_id  # human-facing ID; KG uses kg_uuid
        agent.base_kg_uuid = base_kg_uuid
        agent.agent_config = agent_config

        # Always start from an empty run namespace (本次专用空库).
        if bool(exp_cfg.get("RUN_SCOPED_KG", True)) and kg_uuid != base_kg_uuid:
            try:
                n_cleared = agent.clear_run_kg()
                global_logger.info(
                    "EMPTY_RUN_KG: cleared %s prior fact_tuples for run=%s",
                    n_cleared, kg_uuid,
                )
            except Exception as exc:
                global_logger.warning("EMPTY_RUN_KG clear failed: %s", exc)
        # Pattern starts empty unless offline P_train was loaded during agent init.
        pm_cfg = dict(agent_config.get("PROCEDURAL_MEMORY") or {})
        plib = getattr(agent, "pattern_library", None)
        if plib is not None:
            pattern_source = str(pm_cfg.get("source", "") or "").strip().lower()
            has_preloaded = (
                pattern_source == "train_artifact"
                or (hasattr(plib, "has_train_patterns") and plib.has_train_patterns())
                or plib.procedural_size() > 0
            )
            if not has_preloaded:
                plib.clear()
                global_logger.info("EMPTY_RUN_PATTERN: PatternLibrary cleared at run start")
            else:
                if hasattr(plib, "clear_delta"):
                    plib.clear_delta()
                global_logger.info(
                    "PATTERN_TRAIN_KEPT: train=%s delta=%s source=%s K=%s",
                    getattr(plib, "train_pattern_count", plib.procedural_size)(),
                    getattr(plib, "delta_pattern_count", lambda: 0)(),
                    pattern_source or "preloaded",
                    getattr(plib, "max_patterns_per_sig", "?"),
                )

        if seed_base_kg and kg_uuid != base_kg_uuid:
            n_copied = agent.seed_kg_from_base(base_kg_uuid)
            global_logger.info(
                "SEED_BASE_KG: copied %s fact_tuples from base=%s -> run=%s",
                n_copied, base_kg_uuid, kg_uuid,
            )
        elif seed_base_kg and kg_uuid == base_kg_uuid:
            global_logger.info(
                "SEED_BASE_KG skipped (run uuid == base uuid=%s); writes will mutate base",
                base_kg_uuid,
            )
        else:
            global_logger.info(
                "SEED_BASE_KG=false: run kg_uuid=%s starts as empty dedicated store",
                kg_uuid,
            )

        repeat_episodes = max(1, int(exp_cfg.get("REPEAT_EPISODES", 1) or 1))
        clear_pattern_each = bool(exp_cfg.get("CLEAR_PATTERN_EACH_EPISODE", False))
        # A continual run is expected to accumulate ΔP across task boundaries.
        # Task-conditioned retrieval prevents an unrelated task's procedure
        # from becoming executable, so clearing it here only discards online
        # learning before a later variation can use it.
        clear_pattern_on_task_switch = bool(
            exp_cfg.get("CLEAR_PATTERN_ON_TASK_SWITCH", False)
        )
        global_logger.info(
            "KG isolation: config_agent_id=%s base_kg_uuid=%s kg_uuid=%s STORE_KG=%s "
            "RUN_SCOPED_KG=%s SEED_BASE_KG=%s CLEAN_KG_ON_EXIT=%s",
            agent_id,
            base_kg_uuid,
            kg_uuid,
            store_kg,
            exp_cfg.get("RUN_SCOPED_KG", True),
            seed_base_kg,
            clean_kg_on_exit,
        )
        global_logger.info(
            "CL controls: REPEAT_EPISODES=%s CLEAR_PATTERN_EACH_EPISODE=%s "
            "CLEAR_PATTERN_ON_TASK_SWITCH=%s "
            "ENABLE_PATTERN_READ=%s ENABLE_PATTERN_WRITE=%s PATTERN_PERSIST=%s "
            "pattern_lib(read=%s,write=%s)",
            repeat_episodes,
            clear_pattern_each,
            clear_pattern_on_task_switch,
            exp_cfg.get("ENABLE_PATTERN_READ", True),
            exp_cfg.get("ENABLE_PATTERN_WRITE", True),
            exp_cfg.get("PATTERN_PERSIST", False),
            getattr(agent.pattern_library, "enable_read", True),
            getattr(agent.pattern_library, "enable_write", True),
        )
        per_task_results = []
        global_episode_id = 0
        try:
            if hasattr(agent.env, "configure_split"):
                split = agent.env.configure_split(set)
                global_logger.info("Env split configured from SET=%s -> %s", set, split)

            episode_schedule = _build_episode_schedule(
                exp_cfg, tasks, agent.env, set, variations, global_logger,
                repo_root=_REPO_ROOT,
            )
            if episode_schedule is not None:
                global_logger.info(
                    "CONTINUAL_CURVE: running %s sequential episodes (flat schedule)",
                    len(episode_schedule),
                )
                try:
                    from core.experiment_protocol import write_protocol_snapshot
                    proto_path = write_protocol_snapshot(
                        agent_config,
                        global_experiment_dir,
                        repo_root=_REPO_ROOT,
                        config_yml_path=_CONFIG_YML,
                        episode_schedule=episode_schedule,
                    )
                    global_logger.info("Experiment protocol saved -> %s", proto_path)
                except Exception as exc:
                    global_logger.warning("Failed to write protocol.json: %s", exc)

            def _run_one_episode(task, variation, ep_idx, repeat_total):
                nonlocal global_episode_id
                episode_wall_start = time.time()
                gid = global_episode_id
                if clear_pattern_each:
                    _clear_pattern_library(
                        agent, agent_config, full=True,
                        reason=f"task={task} var={variation} ep={ep_idx}",
                    )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = os.path.join(
                    global_experiment_dir,
                    f"log/{task}/var_{variation}_ep{ep_idx}_gid{gid}_t_{timestamp}/",
                )
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, 'experiment.log')
                logger = setup_experiment_logger(log_file, agent_id)
                agent.logger = logger
                agent.env.load(
                    task, variationIdx=variation,
                    simplificationStr=simplification, generateGoldPath=True,
                )
                agent.env.reset()
                logger.info(agent.env.getTaskDescription())
                plib = getattr(agent, "pattern_library", None)
                logger.info(
                    "Episode start: global_id=%s task=%s variation=%s local_ep=%s/%s "
                    "pattern_total=%s train=%s delta=%s kg_uuid=%s store_kg=%s",
                    gid,
                    task,
                    variation,
                    ep_idx + 1,
                    repeat_total,
                    plib.size() if plib is not None else 0,
                    plib.train_pattern_count() if plib is not None else 0,
                    plib.delta_pattern_count() if plib is not None else 0,
                    getattr(agent, "this_uuid", ""),
                    store_kg,
                )
                agent.reset(episode_id=gid)
                agent.task_id = str(task or "")
                agent.variation_id = str(variation)
                global_episode_id += 1

                try:
                    scores, t = asyncio.run(run_single_experiment(agent, store=store_kg))
                except Exception as ep_exc:
                    global_logger.exception(
                        "Episode failed global_id=%s task=%s variation=%s ep=%s/%s: %s",
                        gid, task, variation, ep_idx + 1, repeat_total, ep_exc,
                    )
                    if logger is not None:
                        logger.exception(
                            "Episode failed task=%s variation=%s ep=%s: %s",
                            task, variation, ep_idx + 1, ep_exc,
                        )
                    return {
                        "scores": [0],
                        "times": [0.0],
                        "meta": {
                            "variation": variation,
                            "episode": ep_idx,
                            "global_episode_id": gid,
                            "max_score": 0,
                            "pattern_lib_size_after": plib.size() if plib is not None else 0,
                            "train_pattern_count": plib.train_pattern_count() if plib is not None else 0,
                            "delta_pattern_count": plib.delta_pattern_count() if plib is not None else 0,
                            "reuse_proposed": 0,
                            "reuse_adopted": 0,
                            "pattern_force": 0,
                            "episode_status": "error",
                            "error": f"{type(ep_exc).__name__}: {ep_exc}",
                        },
                    }

                ep_max = max(scores) if scores else 0
                episode_wall_end = time.time()
                reuse = getattr(agent, "reuse_stats", None) or {}
                total_actions = int(reuse.get("total_executed_actions", 0) or 0)
                train_actions = int(reuse.get("train_pattern_actions", 0) or 0)
                ptr = round(train_actions / total_actions, 4) if total_actions > 0 else 0.0
                nr = round(
                    int(reuse.get("full_cognition_actions", 0) or 0) / total_actions,
                    4,
                ) if total_actions > 0 else 0.0
                exec_metrics = getattr(agent, "execution_metrics", None)
                hkg = getattr(agent, "hkg_train", None) or getattr(agent, "historical_kg", None)
                exec_record = {}
                if exec_metrics is not None and hasattr(exec_metrics, "episode_record"):
                    exec_record = exec_metrics.episode_record(
                        episode_id=gid,
                        task_id=str(task or ""),
                        variation_id=str(variation),
                        score=ep_max,
                    )
                exec_audit = getattr(agent, "execution_audit", None)
                audit_record = exec_audit.to_dict() if exec_audit is not None else {}
                wall_time_s = _episode_duration_sec(
                    t,
                    wall_start=episode_wall_start,
                    wall_end=episode_wall_end,
                )
                direct_steps = int(exec_record.get("direct_reuse", 0) or 0)
                verified_steps = int(exec_record.get("verified_reuse", 0) or 0)
                action_count = int(exec_record.get("action_count", total_actions) or total_actions)
                reuse_steps = direct_steps + verified_steps
                reuse_rate = round(reuse_steps / action_count, 4) if action_count > 0 else 0.0
                meta = {
                    "task": task,
                    "variation": variation,
                    "episode": ep_idx,
                    "global_episode_id": gid,
                    "max_score": ep_max,
                    "wall_time_s": round(wall_time_s, 2),
                    "pattern_lib_size_after": plib.size() if plib is not None else 0,
                    "train_pattern_count": plib.train_pattern_count() if plib is not None else 0,
                    "delta_pattern_count": plib.delta_pattern_count() if plib is not None else 0,
                    "reuse_proposed": int(reuse.get("proposed", 0) or 0),
                    "reuse_adopted": int(reuse.get("adopted", 0) or 0),
                    "pattern_force": int(reuse.get("pattern_force", 0) or 0),
                    "train_pattern_actions": train_actions,
                    "delta_pattern_actions": int(reuse.get("delta_pattern_actions", 0) or 0),
                    "pattern_transfer_rate": ptr,
                    "pattern_reuse_rate": reuse_rate,
                    "novelty_rate": nr,
                    "hkg_edges": len(hkg) if hkg is not None else 0,
                    "episode_status": getattr(agent, "episode_status", "unknown"),
                    "grounding_blocked_actions": int(
                        getattr(agent, "_grounding_blocked_actions", 0) or 0
                    ),
                    "grounding_recovery_actions": int(
                        getattr(agent, "_grounding_recovery_actions", 0) or 0
                    ),
                    "ambiguous_resolution_steps": int(
                        getattr(agent, "_ambiguous_resolution_steps", 0) or 0
                    ),
                    "execution_audit": audit_record,
                    **exec_record,
                }
                src_log = getattr(agent, "action_source_log", None)
                if src_log is not None and hasattr(src_log, "summary"):
                    meta["action_source_summary"] = src_log.summary()
                global_logger.info(
                    "Finished global_id=%s task=%s variation=%s ep=%s/%s score=%s "
                    "patterns total=%s train=%s delta=%s",
                    gid, task, variation, ep_idx + 1, repeat_total, ep_max,
                    plib.size() if plib is not None else 0,
                    plib.train_pattern_count() if plib is not None else 0,
                    plib.delta_pattern_count() if plib is not None else 0,
                )
                plot_scores_vs_time(
                    scores, t, f"{task}_v{variation}_ep{ep_idx}", variation,
                    os.path.join(log_dir, 'figure.png'),
                )
                return {"scores": scores, "times": t, "meta": meta}

            if episode_schedule is not None:
                task_buckets: dict[str, dict] = {}
                for sched_idx, (task, variation) in enumerate(episode_schedule):
                    result = _run_one_episode(task, variation, sched_idx, len(episode_schedule))
                    bucket = task_buckets.setdefault(task, {"scores": [], "times": [], "meta": []})
                    bucket["scores"].append(result["scores"])
                    bucket["times"].append(result["times"])
                    bucket["meta"].append(result["meta"])
                for task, payload in task_buckets.items():
                    per_task_results.append({"task": task, **payload})
            else:
                for task in tasks:
                    task_scores = []
                    task_times = []
                    task_meta = []
                    if (
                        hasattr(agent, "pattern_library")
                        and not clear_pattern_each
                        and clear_pattern_on_task_switch
                        and getattr(agent, "_cl_pattern_task", None) not in (None, task)
                    ):
                        _clear_pattern_library(
                            agent, agent_config, full=False,
                            reason=f"task switch {getattr(agent, '_cl_pattern_task', None)} -> {task}",
                        )
                    if hasattr(agent, "pattern_library"):
                        agent._cl_pattern_task = task
                    agent.env.load(
                        task, variationIdx=0,
                        simplificationStr=simplification, generateGoldPath=True,
                    )
                    task_variations = _resolve_variations(
                        agent.env, set, variations, logger=global_logger, task=task,
                    )
                    global_logger.info(f"Task {task} variations: {task_variations}")
                    for variation in task_variations:
                        for ep_idx in range(repeat_episodes):
                            result = _run_one_episode(task, variation, ep_idx, repeat_episodes)
                            task_scores.append(result["scores"])
                            task_times.append(result["times"])
                            task_meta.append(result["meta"])
                    per_task_results.append({
                        'task': task,
                        'scores': task_scores,
                        'times': task_times,
                        'meta': task_meta,
                    })
        finally:
            collector = getattr(agent, "trajectory_collector", None)
            if collector is not None:
                try:
                    out = collector.flush()
                    global_logger.info(
                        "Saved %s training trajectories -> %s",
                        len(collector.episodes),
                        out,
                    )
                except Exception as exc:
                    global_logger.warning("Trajectory flush failed: %s", exc)
            # Drop this run's KG rows only — never delete BASE_KG_UUID / pretrained graph.
            if clean_kg_on_exit and kg_uuid != base_kg_uuid:
                try:
                    agent.reset_memory()
                    global_logger.info(
                        "CLEAN_KG_ON_EXIT: reset_memory for run kg_uuid=%s (base=%s untouched)",
                        kg_uuid, base_kg_uuid,
                    )
                except Exception as exc:
                    global_logger.warning("CLEAN_KG_ON_EXIT failed: %s", exc)
            elif clean_kg_on_exit and kg_uuid == base_kg_uuid:
                global_logger.warning(
                    "CLEAN_KG_ON_EXIT skipped: refusing to wipe base_kg_uuid=%s",
                    base_kg_uuid,
                )

        return per_task_results
    
    except Exception as e:
        global_logger.error(f"An error occurred: {e}")
        raise e

async def run_single_experiment(agent, store=False):
    await agent.update(store=store)
    scores, t = await agent.act_and_refine(store=store)
    agent.logger.info(f"Scores: {scores}, Time: {t}, store_kg={store}")
    return scores, t

def run_parallel_experiments(
    num_agents=2,
    tasks=None,
    variations=None,
    simplification='easy',
    max_steps=40,
    set='test_mini',
):
    tasks = tasks or []
    task_chunks = split_tasks_across_agents(tasks, num_agents)

    # NUM_AGENTS=1: run in-process. ProcessPoolExecutor + CUDA/transformers often
    # hangs on worker teardown after CLEAN_KG, so the parent never receives
    # results and never writes summary.txt.
    if int(num_agents) <= 1:
        return [
            run_agent(
                agent_config,
                task_chunks[0],
                set,
                simplification,
                global_experiment_dir,
                max_steps,
                variations,
            )
        ]

    with ProcessPoolExecutor(max_workers=num_agents) as executor:
        futures = []
        for i in range(num_agents):
            futures.append(
                executor.submit(
                    run_agent,
                    agent_config,
                    task_chunks[i],
                    set,
                    simplification,
                    global_experiment_dir,
                    max_steps,
                    variations,
                )
            )

        results = [f.result() for f in futures]
    return results

# Function to plot scores vs time
def plot_scores_vs_time(scores, t, task, variation, plot_file):
    if not t:
        return
    # Honor EXPERIMENT.PLOT_RESULTS (default true for backward compatibility).
    try:
        if not bool(agent_config.get("EXPERIMENT", {}).get("PLOT_RESULTS", True)):
            return
    except Exception:
        pass
    try:
        t_minutes = [(time - min(t)) / 60 for time in t]
        plt.figure()
        plt.plot(t_minutes, scores)
        plt.ylim((0, 100))
        plt.xlabel('Time (minutes)')
        plt.ylabel('Scores')
        plt.title(f'{task}_{variation}')
        plt.savefig(plot_file)
        plt.close()
    except Exception as exc:
        # Never block episode completion / summary on plotting failures.
        try:
            plt.close("all")
        except Exception:
            pass
        logging.getLogger("experiment_logger").warning(
            "plot_scores_vs_time failed (%s): %s", plot_file, exc
        )

def _episode_max_score(score_seq):
    """One variation run: list of per-step scores -> episode score (same as run_agent logging)."""
    if score_seq is None:
        return 0.0
    if isinstance(score_seq, (list, tuple)):
        flat = []
        for x in score_seq:
            if isinstance(x, (list, tuple)):
                flat.extend(x)
            else:
                flat.append(float(x))
        return float(max(flat)) if flat else 0.0
    return float(score_seq)


def _episode_duration_sec(time_seq, *, wall_start=None, wall_end=None):
    """Wall-clock seconds for one variation run."""
    if time_seq and len(time_seq) >= 2:
        return max(0.0, float(time_seq[-1]) - float(time_seq[0]))
    if wall_start is not None and wall_end is not None:
        return max(0.0, float(wall_end) - float(wall_start))
    return 0.0


# Function to summarize results
def summarize_results(results, summary_file):
    # Per task: one float per variation/episode run (not nested step lists).
    total_scores = {}
    total_times = {}
    total_meta = {}

    print(results)
    for agent_results in results:
        if not agent_results:
            continue
        for task_result in agent_results:
            task = task_result['task']
            if task not in total_scores:
                total_scores[task] = []
                total_times[task] = []
                total_meta[task] = []
            for score_seq in task_result.get('scores', []):
                total_scores[task].append(_episode_max_score(score_seq))
            task_times = list(task_result.get('times', []))
            task_meta = list(task_result.get('meta') or [])
            total_meta[task].extend(task_meta)
            for idx, time_seq in enumerate(task_times):
                # ``time_seq`` starts at the first environment action and
                # therefore omits initial planning/model load time.  Prefer
                # the wall-clock duration captured around the whole episode
                # so efficiency comparisons include the same setup cost.
                duration = 0.0
                if idx < len(task_meta):
                    try:
                        duration = max(
                            0.0,
                            float(task_meta[idx].get("wall_time_s", 0) or 0),
                        )
                    except (AttributeError, TypeError, ValueError):
                        duration = 0.0
                if duration <= 0.0:
                    duration = _episode_duration_sec(time_seq)
                # A terminal first action legitimately produces one timestamp.
                # The runner records the complete wall-clock duration in meta;
                # use it here so summary.txt agrees with per-episode records.
                if duration <= 0.0 and idx < len(task_meta):
                    try:
                        duration = max(
                            0.0,
                            float(task_meta[idx].get('wall_time_s', 0) or 0),
                        )
                    except (AttributeError, TypeError, ValueError):
                        duration = 0.0
                total_times[task].append(duration)

    lines = []
    for task in sorted(total_scores.keys()):
        scores = total_scores[task]
        times = total_times.get(task, [])
        meta = total_meta.get(task, [])
        if not scores:
            continue
        avg_score = float(np.mean(scores))
        avg_time = float(np.mean(times)) if times else 0.0
        lines.append(
            f"Task: {task} - Runs: {len(scores)} - "
            f"Per-run max scores: {scores} - "
            f"Average max score: {avg_score:.2f} - "
            f"Average duration (s): {avg_time:.1f}"
        )
        if meta:
            # Continual-learning view: score by (variation, episode index).
            by_var = {}
            for row in meta:
                by_var.setdefault(row.get("variation"), []).append(row)
            for var in sorted(by_var.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                rows = sorted(by_var[var], key=lambda r: int(r.get("episode", 0)))
                curve = [int(r.get("max_score", 0)) for r in rows]
                sizes = [int(r.get("pattern_lib_size_after", -1)) for r in rows]
                proposed = [int(r.get("reuse_proposed", 0) or 0) for r in rows]
                adopted = [int(r.get("reuse_adopted", 0) or 0) for r in rows]
                forced = [int(r.get("pattern_force", 0) or 0) for r in rows]
                extra = ""
                if any(proposed) or any(adopted):
                    extra = (
                        f" reuse_proposed={proposed} reuse_adopted={adopted} "
                        f"pattern_force={forced}"
                    )
                errors = [
                    (r.get("error") or "").strip()
                    for r in rows
                    if (r.get("error") or "").strip()
                ]
                if errors:
                    # Keep one line readable; duplicate messages collapsed.
                    uniq = list(dict.fromkeys(errors))
                    err_txt = " | ".join(uniq[:3])
                    if len(uniq) > 3:
                        err_txt += f" | ...(+{len(uniq) - 3} more)"
                    extra += f" error={err_txt!r}"
                lines.append(
                    f"  var={var} episode_scores={curve} pattern_lib_size_after={sizes}{extra}"
                )

    summary = "\n".join(lines) if lines else "No results to summarize."
    with open(summary_file, 'w') as f:
        f.write(summary)

    print(summary)


def _flatten_episode_metrics(results) -> list[dict]:
    """Collect per-episode rows sorted by global_episode_id."""
    rows: list[dict] = []
    for agent_results in results or []:
        if not agent_results:
            continue
        for task_result in agent_results:
            task = task_result.get("task", "")
            for meta in task_result.get("meta") or []:
                if not isinstance(meta, dict):
                    continue
                row = dict(meta)
                row.setdefault("task", task)
                rows.append(row)
    rows.sort(key=lambda r: int(r.get("global_episode_id", r.get("episode_id", 0)) or 0))
    return rows


def _export_continual_curve(results, exp_cfg: dict, out_path: str) -> None:
    """E2: per-episode Score / |P_t| / reuse / 9B calls for paper figures."""
    curve_cfg = dict(exp_cfg.get("CONTINUAL_CURVE") or {})
    if not curve_cfg:
        return
    episodes = []
    for row in _flatten_episode_metrics(results):
        episodes.append({
            "episode": int(row.get("episode", 0) or 0),
            "global_episode_id": int(row.get("global_episode_id", 0) or 0),
            "task": row.get("task", ""),
            "variation": row.get("variation"),
            "score": int(row.get("max_score", row.get("score", 0)) or 0),
            "pattern_size": int(row.get("pattern_lib_size_after", 0) or 0),
            "train_pattern_count": int(row.get("train_pattern_count", 0) or 0),
            "delta_pattern_count": int(row.get("delta_pattern_count", 0) or 0),
            "reuse_rate": float(row.get("pattern_reuse_rate", 0) or 0),
            "pattern_transfer_rate": float(row.get("pattern_transfer_rate", 0) or 0),
            "novelty_rate": float(row.get("novelty_rate", 0) or 0),
            "cognition_9b_calls": int(row.get("model_calls_9b", 0) or 0),
            "cognition_27b_calls": int(row.get("model_calls_27b", 0) or 0),
            "cognition_calls": int(row.get("cognition_calls", 0) or 0),
            "perception_calls": int(row.get("perception_calls", 0) or 0),
            "model_calls_by_name": dict(row.get("model_calls_by_name") or {}),
            "tokens_by_model": dict(row.get("tokens_by_model") or {}),
            "fast_path_ratio": float(row.get("fast_path_ratio", 0) or 0),
            "wall_time_s": float(row.get("wall_time_s", 0) or 0),
        })
    payload = {
        "experiment_id": exp_cfg.get("EXPERIMENT_ID", ""),
        "experiment_name": exp_cfg.get("EXPERIMENT_NAME", ""),
        "num_episodes": len(episodes),
        "episodes": episodes,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    global_logger.info("Continual curve saved -> %s (%s episodes)", out_path, len(episodes))


def _verify_run_invariants(results, agent_config: dict) -> None:
    """Post-run checks for test contamination, frozen P_train, and CWME routing."""
    from core.experiment_protocol import (
        HEURISTIC_DECISION_KEYS,
        aggregate_decision_source_counts,
        cwme_no_heuristic_mode,
        decision_source_share,
    )

    exp_cfg = dict(agent_config.get("EXPERIMENT") or {})
    exp_id = str(exp_cfg.get("EXPERIMENT_ID", "") or "")
    rows = _flatten_episode_metrics(results)
    if not rows:
        return

    final = rows[-1]
    train_start = int(rows[0].get("train_pattern_count", 0) or 0)
    train_end = int(final.get("train_pattern_count", 0) or 0)
    if train_start != train_end:
        raise RuntimeError(
            f"P_train invariant failed: train_pattern_count {train_start} -> {train_end}"
        )

    assert_no_writeback = bool(exp_cfg.get("ASSERT_NO_TEST_WRITEBACK")) or exp_id == "E7-A"
    delta_end = int(final.get("delta_pattern_count", 0) or 0)
    if assert_no_writeback and delta_end > 0:
        raise RuntimeError(
            f"E7-A / no-writeback invariant failed: final_delta_pattern_count={delta_end} (expected 0)"
        )

    if cwme_no_heuristic_mode(agent_config):
        counts = aggregate_decision_source_counts(rows)
        heur_share = decision_source_share(counts, set(HEURISTIC_DECISION_KEYS))
        # Formal grounding-only runs must have zero heuristic decisions.  A
        # nonzero tolerance would let a hidden task shortcut contaminate an
        # ablation while still passing post-run validation.
        if heur_share > 0.0:
            raise RuntimeError(
                f"CWME heuristic bypass detected: heuristic decision share={heur_share:.2%} (>0%)"
            )
        for row in rows:
            audit = row.get("execution_audit") or {}
            if audit and not audit.get("valid_cwme_run", True):
                raise RuntimeError(
                    "CWME execution audit invalid: "
                    f"task_heuristic_calls={audit.get('task_heuristic_calls')} "
                    f"legacy_heuristic_calls={audit.get('legacy_heuristic_calls')}"
                )
            if audit and int(audit.get("blocked_task_heuristic_calls", 0) or 0) > 0:
                raise RuntimeError(
                    "CWME blocked heuristic attempt detected: "
                    f"blocked_task_heuristic_calls={audit.get('blocked_task_heuristic_calls')}"
                )


def persist_experiment_metrics(results, exp_dir: str, exp_cfg: dict) -> None:
    """Write structured JSON for evaluate_cwme / multiseed aggregation."""
    rows = _flatten_episode_metrics(results)
    if not rows:
        return
    os.makedirs(exp_dir, exist_ok=True)
    metrics_path = os.path.join(exp_dir, "episode_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    global_logger.info("Episode metrics saved -> %s (%s episodes)", metrics_path, len(rows))

    structured = []
    for agent_results in results or []:
        if not agent_results:
            continue
        structured.extend(agent_results)
    struct_path = os.path.join(exp_dir, "results_structured.json")
    with open(struct_path, "w", encoding="utf-8") as fh:
        json.dump(structured, fh, indent=2, ensure_ascii=False, default=str)

    curve_path = os.path.join(exp_dir, "continual_curve.json")
    if exp_cfg.get("CONTINUAL_CURVE"):
        _export_continual_curve(results, exp_cfg, curve_path)


# Execute the experiment
if __name__ == '__main__':
    # Experiment parameters
    num_agent = agent_config['EXPERIMENT']['NUM_AGENTS']
    exp_cfg = agent_config['EXPERIMENT']
    # Do not shadow the built-in set(), which is used by post-run invariant checks.
    eval_set = exp_cfg['SET']
    if 'VARIATIONS' in exp_cfg and exp_cfg['VARIATIONS'] is not None:
        variations = (
            {
                str(task): [int(value) for value in values]
                for task, values in exp_cfg['VARIATIONS'].items()
            }
            if isinstance(exp_cfg['VARIATIONS'], dict)
            else list(exp_cfg['VARIATIONS'])
        )
        variation_mode = f"explicit VARIATIONS={variations}"
    else:
        variations = None
        variation_mode = f"SET={eval_set} (load_variation per task)"
    max_steps = exp_cfg['MAX_STEPS']
    simplifications = exp_cfg['SIMPLIFICATION']
    tasks = exp_cfg['TASKS']
    env_name = normalize_environment_name(
        (agent_config.get("AGENT") or {}).get("ENVIRONMENT", "ScienceWorld")
    )
    global_logger.info(f"Config YAML: {_CONFIG_YML}")
    global_logger.info(f"Environment: {env_name}")
    global_logger.info(f"Tasks: {tasks}, {variation_mode}")
    global_logger.info(
        "CL protocol: SET=%s REPEAT_EPISODES=%s CLEAR_PATTERN_EACH_EPISODE=%s "
        "ENABLE_PATTERN_READ=%s ENABLE_PATTERN_WRITE=%s STORE_KG=%s "
        "RUN_SCOPED_KG=%s SEED_BASE_KG=%s",
        eval_set,
        exp_cfg.get("REPEAT_EPISODES", 1),
        exp_cfg.get("CLEAR_PATTERN_EACH_EPISODE", False),
        exp_cfg.get("ENABLE_PATTERN_READ", True),
        exp_cfg.get("ENABLE_PATTERN_WRITE", True),
        exp_cfg.get("STORE_KG", False),
        exp_cfg.get("RUN_SCOPED_KG", True),
        exp_cfg.get("SEED_BASE_KG", False),
    )
    try:
        from core.experiment_protocol import validate_frozen_artifacts
        enforce = bool(
            (exp_cfg.get("ARTIFACT_FREEZE") or {}).get("enforce")
            or exp_cfg.get("ENFORCE_ARTIFACT_FREEZE", False)
        )
        if enforce:
            validate_frozen_artifacts(agent_config, repo_root=_REPO_ROOT, strict=True)
            global_logger.info("Artifact freeze validation passed")
    except RuntimeError as exc:
        global_logger.error("Artifact freeze validation failed: %s", exc)
        raise
    except Exception as exc:
        global_logger.warning("Artifact freeze validation skipped: %s", exc)
    results = run_parallel_experiments(
        num_agent, tasks, variations, simplifications, max_steps, eval_set,
    )
    
    summary_file = os.path.join(global_experiment_dir, 'summary.txt')
    summarize_results(results, summary_file)
    global_logger.info(f"Summary saved to {summary_file}")
    persist_experiment_metrics(results, global_experiment_dir, exp_cfg)
    try:
        _verify_run_invariants(results, agent_config)
        global_logger.info("Post-run invariants passed")
    except RuntimeError as exc:
        global_logger.error("Post-run invariant check failed: %s", exc)
        raise

    global_logger.info(f"Experiment completed with results: {results}")
