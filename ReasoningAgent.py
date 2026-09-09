"""
Description: 
    This module defines the ReasoningAgent class, which utilizes a knowledge graph 
    and a large language model (LLM) to interact with text environments
    (ScienceWorld or AlfWorld via envs.create_env).
    The agent performs reasoning tasks, interacts with the environment, updates 
    its internal state, and refines its actions based on feedback.
    
    The module also includes functions for loading configuration, setting up 
    logging, and running experiments.
"""

from utils import *
from utils import (  # not exported by `import *` (leading underscore)
    _extract_current_room_name,
    fallback_replan_trajectory,
    stagnant_early_stop_threshold,
)
from kg_graph.KnowledgeGraph import KnowledgeGraph
from kg_graph.arigraph_working_memory import AriGraphWorkingMemory
import time
import gc
import torch
import kg_graph.schema as sm
from pprint import pprint
import os
import re
import asyncio
from threading import Thread
from sentence_transformers import SentenceTransformer
from core.model_router import ModelRouter, resolve_agent_models
from core.orchestrator import Orchestrator
from core.dual_memory import DualMemory
from core.post_focus import PostFocusEngine
from core.pre_focus import PreFocusEngine
from core.memory_fusion import MemoryFusionEngine
from core.dgar import DGAREngine
from core.pattern_library import PatternLibrary
from core.evolution.continual_engine import ContinualWorldModelEvolution, resolve_evolution_config
from core.state_packet import build_state_packet
from core.cwme_agent_hooks import (
    attach_cwme_components,
    on_agent_reset,
    on_agent_step_after_env,
)
from core.cl_protocol import pattern_read_enabled as _pattern_read_enabled
from core.cl_protocol import pattern_milestone_for_agent
from core.cl_protocol import grounding_only_mode
from core.action_primitives import action_verb_family
from core.alfworld_policy import cmd_verb_family
from core.cwme_planning import (
    cwme_actor_stuck_threshold,
    cwme_actor_max_attempts,
    cwme_initial_plan_steps,
    cwme_refiner_max_attempts,
    cwme_heuristic_fallback_enabled,
    cwme_initial_plan_mode,
    cwme_llm_planning_enabled,
    cwme_skip_initial_plan_if_pattern,
    minimal_cognition_trajectory,
    pad_cognition_trajectory,
)
from envs.base import is_alfworld, is_scienceworld

class ReasoningAgent():
    def __init__(self, config, logger, connection_pool, this_uuid, step_limit, look_ahead, max_query, env, agent_model='/mnt/nfsData19/Zhaoshuyuan/Houxinrui/model/Qwen3.5-9B', kg_model='', agent_config=None):
        """
        Initializes the ReasoningAgent with the given configuration, logger, connection pool, UUID, and environment.

        Args:
            config (ConfigParser): Configuration object containing various settings.
            logger (logging.Logger): Logger object for logging messages.
            connection_pool (psycopg2.pool.SimpleConnectionPool): Connection pool for database connections.
            this_uuid (str): Unique identifier for the agent instance.
            env: Text env adapter (ScienceWorldEnv or AlfWorldEnvAdapter).
        """
        
        # set up logger
        self.logger = logger
        self.config = config

        # Setup Connection Pool for Agent. Exit if fail.
        self.connection_pool = connection_pool
        # if not self.connection_pool:
        #     print("Failed DB Connection. Exiting")
        #     self.logger.error(f"FAILED STARTUP: Error connecting to database. SHUTDOWN.")
        #     exit()

        self.rel_path = os.path.dirname(__file__)
        self.agent_config = agent_config or {}
        cognition_default, perception_default = resolve_agent_models(self.agent_config)
        self.cognition_model = agent_model or cognition_default
        self.perception_model = kg_model or perception_default or self.cognition_model
        self.model = self.cognition_model
        self.router = ModelRouter(self.agent_config, config, logger=logger)
        self.orchestrator = Orchestrator(base_look_ahead=look_ahead)
        self.dgar = DGAREngine(logger=logger)
        exp_cfg = (self.agent_config or {}).get("EXPERIMENT", {}) or {}
        log_root = (exp_cfg.get("LOG_ROOT") or "").strip()
        if not log_root and config.has_section("EXPERIMENT"):
            log_root = (config.get("EXPERIMENT", "LOG_ROOT", fallback="") or "").strip()
        # Default: no shared disk patterns (keeps CL ablations clean). Opt-in via PATTERN_PERSIST.
        pattern_path = None
        pm_cfg = dict((self.agent_config or {}).get("PROCEDURAL_MEMORY") or {})
        if pm_cfg.get("persist_path"):
            pattern_path = str(pm_cfg["persist_path"])
        elif bool(exp_cfg.get("PATTERN_PERSIST", False)) and log_root:
            pattern_path = os.path.join(log_root, "patterns.jsonl")
        self.pattern_library = PatternLibrary(
            persist_path=pattern_path,
            max_patterns_per_sig=int(exp_cfg.get("PATTERN_MAX_PER_SIG", 5) or 5),
            min_record_score=int(exp_cfg.get("PATTERN_MIN_RECORD_SCORE", 30) or 30),
            min_partial_score=int(exp_cfg.get("PATTERN_MIN_PARTIAL_SCORE", 20) or 20),
            enable_read=bool(exp_cfg.get("ENABLE_PATTERN_READ", True)),
            enable_write=bool(exp_cfg.get("ENABLE_PATTERN_WRITE", True)),
            allow_verb_transfer=bool(exp_cfg.get("PATTERN_VERB_TRANSFER", False)),
            context_match_threshold=float(
                pm_cfg.get("context_match_threshold", 0.65) or 0.65
            ),
        )
        self.task_id = ""
        self.reuse_stats = {
            "proposed": 0,
            "adopted": 0,
            "pattern_force": 0,
            "by_source": {},
            "train_pattern_actions": 0,
            "delta_pattern_actions": 0,
            "full_cognition_actions": 0,
            "total_executed_actions": 0,
        }
        self.dual_memory = DualMemory()
        evo_cfg = resolve_evolution_config(self.agent_config)
        self.evolution_engine = ContinualWorldModelEvolution(
            dual_memory=self.dual_memory,
            config=evo_cfg,
        )
        self.post_focus = PostFocusEngine()
        self.pre_focus = PreFocusEngine()
        self.memory_fusion = MemoryFusionEngine()
        if logger:
            logger.info(
                "Model routing: cognition=%s perception=%s",
                self.cognition_model,
                self.perception_model,
            )
        preload_local_chat_models(
            config,
            self.cognition_model,
            self.perception_model,
            logger=logger,
        )

        self.kg = KnowledgeGraph(
            self,
            prompts_path=self.rel_path + '/kg_graph/sciworld_prompt',
            config=config,
            ner_llm=True,
            model=self.perception_model,
            cognition_model=self.cognition_model,
        )
        self.token_sent = 0
        self.token_received = 0
        self.inventory = ''
        self.env = env
        self.task = ''
        self.total_reflection = 0
        self.observation = ''
        self.past_actions = ['<START>']
        self.action_res = ['NONE']
        self.step_limit = step_limit
        self.look_ahead = look_ahead
        self.max_query = max_query
        embed_model = config.get(
            "DEFAULT",
            "LOCAL_EMBED_MODEL",
            fallback="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.sbert = SentenceTransformer(embed_model)
        self.arigraph_wm = AriGraphWorkingMemory(
            model=self.perception_model,
            config=self.config,
            embedder=self.sbert,
            logger=self.logger,
            router=self.router,
        )
        self._wm_prev_subgraph = []
        self._wm_subgraph: list[str] = []
        self._wm_episodic: list[str] = []
        self._obs_hist = []
        self._alf_look_memory = []
        self._known_locations = set()
        self._wm_prev_location = None
        self._wm_context_text = ""
        self.possible_actions = self.env.getPossibleActions()
        self.possible_objects = self.env.getPossibleObjects()
        self.kg_log = []
        self.this_uuid = this_uuid
        self.trajectory = []
        self.actor_prompt = open(self.rel_path+'/prompts/actor.txt').read()
        self.refiner_prompt = open(self.rel_path+'/prompts/refiner_v2.txt').read()
        self.reflect_prompt = open(self.rel_path+'/prompts/reflection.txt').read()
        self.past_reflections = ""
        self.replan_count = 0
        self.stagnant_steps = 0
        self.last_exec_source = ""
        self._last_finalized_decision_key = None
        self._cwme_fast_decision_cache = None
        self._cwme_actor_decision_cache = None
        self._cwme_invalid_actor_streak = 0
        self.last_score = 0
        self.episode_progress = make_episode_progress()
        self._score_hist: list[float] = []
        attach_cwme_components(self, self.agent_config, repo_root=self.rel_path)

    def prompts_update(self):
        """
        Updates the prompts for the knowledge graph and the agent.
        """
        self.kg.reload_prompt()
        self.actor_prompt = open(self.rel_path+'/prompts/actor.txt').read()
        self.refiner_prompt = open(self.rel_path+'/prompts/refiner_v2.txt').read()
        self.reflect_prompt = open(self.rel_path+'/prompts/reflection.txt').read()

    def reset(self, episode_id=None):
        """
        Resets the internal state of the agent.
        """
        self.inventory = ''
        # self.env = None
        self.task = ''
        self.token_received = 0
        self.token_sent = 0
        self.observation = ''
        self.past_actions = ['<START>']
        self.action_res = ['NONE']
        self.kg_log = []
        self.trajectory = []

        self.kg.reset()
        self.arigraph_wm.clear()
        self._wm_prev_subgraph = []
        self._obs_hist = []
        self._alf_look_memory = []
        self._known_locations = set()
        self._wm_prev_location = None
        self._wm_context_text = ""
        self._wm_subgraph = []
        self._wm_episodic = []
        self.memory_fusion.reset_episode()
        self.replan_count = 0
        self.stagnant_steps = 0
        self.last_exec_source = ""
        self.last_score = 0
        self.episode_progress = make_episode_progress()
        if episode_id is not None:
            self.episode_progress["episode_idx"] = int(episode_id)
        self.reuse_stats = {
            "proposed": 0,
            "adopted": 0,
            "pattern_force": 0,
            "by_source": {},
            "train_pattern_actions": 0,
            "delta_pattern_actions": 0,
            "full_cognition_actions": 0,
            "total_executed_actions": 0,
        }
        on_agent_reset(self)

    def get_working_memory_context(self):
        """Prepended to KG queries during planning (KnowledgeGraph.query)."""
        return self._wm_context_text or ""

    def get_memory_fusion_context(self, query: str = "") -> str:
        """Fused WM + entity replay + pattern READ for KG query (Memory Hub)."""
        block = self.memory_fusion.build_kg_context(self, query=query)
        base = block or self.get_working_memory_context()
        try:
            snap = self.dual_memory.read(self)
            extras = []
            if snap.pattern_hint:
                extras.append(snap.pattern_hint)
            if snap.pattern_next_milestone:
                extras.append(f"PATTERN_NEXT_MILESTONE: {snap.pattern_next_milestone}")
            if snap.historical_facts:
                extras.append(
                    "HISTORICAL_WORLD_KNOWLEDGE:\n- "
                    + "\n- ".join(snap.historical_facts[:8])
                )
            if snap.historical_transitions:
                extras.append(
                    "HISTORICAL_TRANSITIONS:\n- "
                    + "\n- ".join(snap.historical_transitions[:6])
                )
            if extras:
                joined = "\n".join(extras)
                return f"{base}\n\n{joined}".strip() if base else joined
        except Exception:
            pass
        return base

    def get_planner_context(self) -> str:
        """Structured context for cognition (9B) planning prompts."""
        base = self.dual_memory.planner_block(self)
        supplement = self.dual_memory.fusion_supplement(self)
        if supplement:
            return f"{base}\n\n{supplement}".strip() if base else supplement
        return base

    def _parse_location_hint(self, text):
        if not text:
            return ""
        loc = _extract_current_room_name((text or "").lower())
        if loc:
            return loc
        m = re.search(
            r"(?i)(?:you are in|you are inside|in the room|current room|located in)[^\n,.\:]{0,40}?([A-Za-z][A-Za-z0-9 _\-]{1,60})",
            text,
        )
        if m:
            return m.group(1).strip()
        return ""

    def _build_items1(self):
        items = {}
        for obj in (self.possible_objects or [])[:64]:
            k = str(obj).lower().strip()
            if k:
                items[k] = 2
        for w in re.findall(r"[a-zA-Z]{4,}", self.task or "")[:10]:
            wl = w.lower()
            if wl not in items:
                items[wl] = 1
        if not items:
            items["environment"] = 1
        return items

    def reflect(self, text):
        """
        Generates a reflection based on the given text using an LLM. This is for easier parsing into relational database.

        Args:
            text (str): Text to reflect on.

        Returns:
            str: Reflection result from the LLM.
        """
        prompt = self.reflect_prompt.format(text)
        res, token_count = self.router.call("reflect", prompt)
        return res, token_count

    async def update(
        self,
        action=False,
        store=True,
        skip_action_reflect=False,
        skip_wm_refine=False,
    ):
        """
        Updates the agent's state via DGAR (Detect/Generate/Apply/Remove).

        Args:
            action (bool): Indicates whether an action has been performed.
            store (bool): Indicates whether to store the extracted relations in the knowledge graph.
            skip_action_reflect (bool): Skip the extra LLM reflect on action outcome.
            skip_wm_refine (bool): Skip the second AriGraph WM refine LLM call.
        """
        # A read-only episode must not pay for the write-side perception
        # pipeline. This also covers the initial update before the scheduler
        # has produced a per-step perception plan. DGAR's cheap branch still
        # refreshes observation, inventory, valid actions, and location data.
        if not store:
            skip_action_reflect = True
            skip_wm_refine = True
        await self.dgar.perceive_step(
            self,
            action=action,
            store_kg=store,
            skip_action_reflect=skip_action_reflect,
            skip_wm_refine=skip_wm_refine,
        )
        
    async def step(
        self,
        action,
        store=True,
        skip_kg=False,
        skip_action_reflect=False,
        skip_wm_refine=False,
    ):
        """
        Executes a single step in the environment using the given action.

        Args:
            action (str): Action to perform.
            store (bool): Indicates whether to store the extracted relations in the knowledge graph.
            skip_kg (bool): Skip knowledge-graph relation extraction for this step.
            skip_action_reflect (bool): Skip the post-action LLM reflect call.
            skip_wm_refine (bool): Skip AriGraph WM refine LLM on this step.

        Returns:
            dict: Updated state information.
            str: Next state observation.
            dict: Additional info from the environment.
            bool: Termination status.
        """
        state_before = (self.observation or "")
        if not state_before.strip():
            try:
                state_before = self.env.look() or ""
            except Exception:
                state_before = ""
        score_before = float(self.last_score or 0)

        self.past_actions.append(action)
        next_state, _, termination, info = self.env.step(action)
        self.action_res.append(next_state)
        # Keep live env text in sync (critical for AlfWorld holding / process).
        try:
            inv = self.env.inventory()
            if inv is not None:
                self.inventory = str(inv)
        except Exception:
            pass
        try:
            look = self.env.look()
            if look:
                self.observation = str(look)
        except Exception:
            if next_state:
                self.observation = str(next_state)

        #treating relocation actions differently from regular actions
        obs_l = (next_state or "").lower()
        relocated = (
            "move to" in obs_l
            or "teleport" in obs_l
            or "you arrive at" in obs_l  # AlfWorld / TextWorld navigation
        )
        if relocated:
            await self.update(
                store=store and not skip_kg,
                skip_wm_refine=skip_wm_refine,
            )
        else:
            await self.update(
                action=True,
                store=store and not skip_kg,
                skip_action_reflect=skip_action_reflect,
                skip_wm_refine=skip_wm_refine,
            )

        score_after = float(info.get("score", score_before) if isinstance(info, dict) else score_before)
        state_after = (self.observation or next_state or "")
        self._last_transition = on_agent_step_after_env(
            self,
            state_before=state_before,
            action=action,
            state_after=state_after,
            score_before=score_before,
            score_after=score_after,
            env_response=str(next_state or ""),
        )
        metrics = getattr(self, "execution_metrics", None)
        if metrics is not None and hasattr(metrics, "record_env_step"):
            metrics.record_env_step()

        return {'observation': self.observation, 'inventory': self.inventory, 'valid_receptacles': self.possible_objects}, next_state, info, termination

    def _record_execution_outcome(
        self,
        *,
        prev_score: float,
        current_score: float,
        response: str,
        pattern_milestone: str = "",
    ) -> bool:
        """Record the outcome before terminal/step-limit branches are taken."""
        env_failed = is_env_action_failure(response) or current_score == -100
        transition = getattr(self, "_last_transition", None)
        meaningful = bool(
            transition is not None
            and getattr(transition, "meaningful_change", False)
        )
        metrics = getattr(self, "execution_metrics", None)
        if metrics is not None:
            raw_level = getattr(self, "_last_exec_computation_level", None)
            level = int(2 if raw_level is None else raw_level)
            if level in (0, 1):
                metrics.fast_path_attempts += 1
                if not env_failed:
                    metrics.fast_path_env_success += 1
                if not env_failed and (current_score > prev_score or meaningful):
                    metrics.fast_path_success += 1
            src_key = (
                getattr(self, "_last_decision_source", "")
                or self.last_exec_source
                or "unknown"
            ).strip()
            metrics.decision_source_counts[src_key] = int(
                metrics.decision_source_counts.get(src_key, 0) or 0
            ) + 1
            if src_key == "cognition" or self.last_exec_source in {
                "actor_llm", "cognition_route",
            }:
                metrics.cognition_decision_steps += 1

        src_log = getattr(self, "action_source_log", None)
        if src_log is not None and hasattr(src_log, "mark_last_fast_success"):
            src_log.mark_last_fast_success(
                success=(
                    not env_failed
                    and (current_score > prev_score or meaningful)
                ),
                score_gain=float(current_score - prev_score),
            )

        if pattern_milestone and self.logger:
            source = (getattr(self, "last_exec_source", "") or "").strip()
            pattern_source = source.startswith("pattern")
            progress = not env_failed and (current_score > prev_score or meaningful)
            self.logger.info(
                "[PatternPROGRESS] milestone=%r source=%s score_gain=%s "
                "meaningful_change=%s progress=%s",
                pattern_milestone,
                source,
                current_score - prev_score,
                meaningful,
                bool(pattern_source and progress),
            )
        return env_failed

    async def act_and_refine(self, store=True):
        """
        Performs actions and refines the agent's strategy based on feedback.
        Args:
            max_steps (int): Maximum number of steps to perform in imagination.
            max_query (int): Maximum number of queries to make to the LLM per planning step.
            store (bool): Indicates whether to store the extracted relations in the knowledge graph.

        Returns:
            list: Scores obtained during the execution.
            list: Timestamps of each step.
        """
        reason_model = self.cognition_model
        # Keep the task entity binding immutable for this episode. Refiner
        # suggestions are local recovery hints, not permission to rewrite
        # targets such as ``lead`` into ``tin``.
        episode_task = str(getattr(self, "task", "") or "").strip()
        self._episode_task = episode_task
        last_token_sent = 0
        last_token_received = 0
        use_cwme_plan = cwme_llm_planning_enabled(self)
        swift_env = is_alfworld(self.env) or is_scienceworld(self.env)

        if swift_env and not use_cwme_plan and not grounding_only_mode(self):
            # Legacy swift path: heuristic seed (Backbone / CL-off baselines).
            seed_min = 20 if is_alfworld(self.env) else 18
            seed_steps = max(int(self.look_ahead or 5), seed_min)
            seed_steps = min(seed_steps, max(seed_min, int(self.step_limit or 50)))
            self.trajectory, printout = fallback_replan_trajectory(
                self.task,
                self.observation,
                self.inventory,
                [],
                max_steps=seed_steps,
                action_history=[],
                current_score=0,
                action_seeder=lambda acts: self.orchestrator.seed_heuristic_plan(
                    self, acts, score=0,
                ),
            )
            total_token_sent = 0
            total_token_received = 0
            if self.logger:
                env_tag = "AlfWorld" if is_alfworld(self.env) else "ScienceWorld"
                self.logger.info(
                    "%s initial plan: %s seed (%s steps), skipped get_trajectory",
                    env_tag, cwme_initial_plan_mode(self), len(self.trajectory),
                )
        elif swift_env and (use_cwme_plan or grounding_only_mode(self)):
            # CWME: use an existing executable P_train milestone as the cold
            # start when possible.  This avoids one redundant open-loop 9B
            # call per episode while keeping the decision path heuristic-free.
            # Re-plan from the live state in short windows. The previous
            # 18/20-step open loop paid for stale action/state predictions.
            seed_min = cwme_initial_plan_steps(self)
            seed_steps = min(
                max(int(self.look_ahead or 5), seed_min),
                max(1, int(self.step_limit or 50)),
            )
            pattern_seed = ""
            if cwme_skip_initial_plan_if_pattern(self):
                try:
                    pattern_seed = pattern_milestone_for_agent(
                        self, score=int(getattr(self, "last_score", 0) or 0),
                    )
                except Exception:
                    pattern_seed = ""
            if pattern_seed:
                self.trajectory, printout = minimal_cognition_trajectory(
                    self.observation,
                    self.inventory,
                    max_steps=seed_steps,
                    task=self.task,
                )
                total_token_sent = 0
                total_token_received = 0
                if self.logger:
                    self.logger.info(
                        "CWME initial plan: skipped 9B; reusable P_train milestone=%r",
                        pattern_seed,
                    )
            else:
                try:
                    self.trajectory, printout, total_token_sent, total_token_received = self.kg.get_trajectory(
                        valid_objects=self.possible_objects,
                        task=self.task,
                        observation=self.observation,
                        inventory=self.inventory,
                        MAX_STEPS=seed_steps,
                        MAX_QUERIES=self.max_query,
                        model=reason_model,
                    )
                    if self.logger:
                        env_tag = "AlfWorld" if is_alfworld(self.env) else "ScienceWorld"
                        self.logger.info(
                            "%s CWME initial plan: 9B get_trajectory (%s steps, sent=%s recv=%s)",
                            env_tag,
                            len(self.trajectory),
                            total_token_sent,
                            total_token_received,
                        )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            "CWME get_trajectory failed (%s); using per-step cognition plan", e,
                        )
                    self.trajectory, printout = minimal_cognition_trajectory(
                        self.observation,
                        self.inventory,
                        max_steps=seed_steps,
                        task=self.task,
                    )
                    total_token_sent = 0
                    total_token_received = 0
        else:
            try:
                self.trajectory, printout, total_token_sent, total_token_received = self.kg.get_trajectory(
                    valid_objects=self.possible_objects,
                    task=self.task,
                    observation=self.observation,
                    inventory=self.inventory,
                    MAX_STEPS=self.look_ahead,
                    MAX_QUERIES=self.max_query,
                    model=reason_model
                )
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Initial get_trajectory failed: {e}; using heuristic seed"
                    )
                seed_min = 18 if is_scienceworld(self.env) else 5
                seed_steps = max(int(self.look_ahead or 5), seed_min)
                self.trajectory, printout = fallback_replan_trajectory(
                    self.task,
                    self.observation,
                    self.inventory,
                    [],
                    max_steps=seed_steps,
                    action_history=[],
                    current_score=0,
                    action_seeder=lambda acts: self.orchestrator.seed_heuristic_plan(
                        self, acts, score=0,
                    ),
                )
                total_token_sent = 0
                total_token_received = 0
        self.token_sent += total_token_sent
        self.token_received += total_token_received
        s = self.observation
        print(f'New trajectory: \n{printout}')
        print('<FINISHED INITIAL PLANNING>')
        scores = []
        t = []
        i = 0
        termination = False
        self.replan_count = 0
        self.stagnant_steps = 0
        self.last_exec_source = ""
        self.last_score = 0
        # These keys are state-conditioned and must not cross episode
        # boundaries, otherwise an old variation can replay a decision or
        # suppress the first metric record of the new episode.
        self._cwme_fast_decision_cache = None
        self._cwme_actor_decision_cache = None
        self._cwme_invalid_actor_streak = 0
        self._grounding_blocked_actions = 0
        self._grounding_recovery_actions = 0
        self._ambiguous_resolution_steps = 0
        self.episode_status = "running"
        self._last_finalized_decision_key = None
        self._last_route_decision = None
        saved_episode_idx = (getattr(self, "episode_progress", {}) or {}).get("episode_idx")
        self.episode_progress = make_episode_progress()
        if saved_episode_idx is not None:
            self.episode_progress["episode_idx"] = int(saved_episode_idx)
        # act_and_refine can be entered repeatedly by the experiment runner;
        # clear episode-local memory here as well as in reset().
        on_agent_reset(self)
        self.recent_failed_actions: set[str] = set()
        empty_action_replans = 0
        replan_signatures: set[tuple[str, ...]] = set()

        def _trajectory_signature(trajectory) -> tuple[str, ...]:
            """Compact plan identity used to reject duplicate replans."""
            actions: list[str] = []
            for item in list(trajectory or [])[:-1]:
                if not isinstance(item, dict):
                    continue
                action = str(item.get("action", "") or "").strip().lower()
                if action == "<predict>":
                    continue
                actions.append(action or "<empty>")
            return tuple(actions[-12:])

        initial_signature = _trajectory_signature(self.trajectory)
        if initial_signature:
            replan_signatures.add(initial_signature)
        empty_action_replan_limit = max(
            1,
            int(getattr(getattr(self, "cwme_cfg", None), "empty_action_replan_attempts", 2) or 2),
        )

        def _mark_pending_pattern_failed() -> None:
            marker = getattr(self, "_last_pattern_execution", None)
            if not marker:
                return
            plib = getattr(self, "pattern_library", None)
            try:
                if plib is not None and hasattr(plib, "note_pattern_result"):
                    plib.note_pattern_result(
                        marker.get("pattern_id", ""),
                        marker.get("stage_index", -1),
                        success=False,
                    )
                    metrics = getattr(self, "execution_metrics", None)
                    if metrics is not None:
                        metrics.pattern_recovery += 1
            finally:
                self._last_pattern_execution = None

        def _replan_after_empty_action(reason: str) -> bool:
            """Recover from an empty grounded action using the live state."""
            nonlocal empty_action_replans, last_token_sent, last_token_received
            if empty_action_replans >= empty_action_replan_limit:
                return False
            if not swift_env:
                return False
            empty_action_replans += 1
            self.replan_count += 1
            try:
                live_observation = self.env.look() or self.observation or ""
            except Exception:
                live_observation = self.observation or ""
            if use_cwme_plan or grounding_only_mode(self):
                try:
                    new_trajectory, printout, sent, received = self.kg.get_trajectory(
                        valid_objects=self.possible_objects,
                        task=self.task,
                        observation=live_observation,
                        inventory=self.inventory,
                        MAX_STEPS=max(1, min(int(self.look_ahead or 5), int(self.step_limit or 50))),
                        MAX_QUERIES=self.max_query,
                        sequence=None,
                        reflection=(
                            f"The previous action was not executable ({reason}). "
                            "Use the current live state and return an exact admissible action."
                        ),
                        model=reason_model,
                    )
                except Exception as exc:
                    if self.logger:
                        self.logger.warning(
                            "CWME empty-action replan failed (%s)", exc,
                        )
                    return False
            else:
                new_trajectory, printout = fallback_replan_trajectory(
                    self.task,
                    live_observation,
                    self.inventory,
                    [],
                    max_steps=max(1, min(int(self.look_ahead or 5), int(self.step_limit or 50))),
                    action_history=self.past_actions,
                    current_score=self.last_score,
                    action_seeder=lambda acts: self.orchestrator.seed_heuristic_plan(
                        self, acts, score=self.last_score,
                    ),
                )
                sent, received = last_token_sent, last_token_received
            if not new_trajectory or len(new_trajectory) <= 1:
                return False
            signature = _trajectory_signature(new_trajectory)
            if signature and signature in replan_signatures:
                if self.logger:
                    self.logger.warning(
                        "CWME rejected duplicate empty-action replan: "
                        "reason=%s steps=%s",
                        reason,
                        len(new_trajectory) - 1,
                    )
                return False
            if signature:
                replan_signatures.add(signature)
            self.token_sent += max(0, int(sent or 0) - int(last_token_sent or 0))
            self.token_received += max(0, int(received or 0) - int(last_token_received or 0))
            last_token_sent = int(sent or 0)
            last_token_received = int(received or 0)
            self.trajectory = new_trajectory
            self._cwme_actor_decision_cache = None
            self._cwme_fast_decision_cache = None
            if self.logger:
                self.logger.warning(
                    "CWME empty-action recovery replan %s/%s reason=%s steps=%s",
                    empty_action_replans,
                    empty_action_replan_limit,
                    reason,
                    len(new_trajectory) - 1,
                )
            print(f"<REPLANNING (empty_action, count={empty_action_replans})>")
            print(f"New trajectory: \n{printout}")
            return True

        while i < len(self.trajectory) - 1 and (not termination):
            self._step_used_actor_llm = False
            self.act(i, self.model)
            sar_hat = self.trajectory[i]
            if not (sar_hat.get("action", "") or "").strip():
                _mark_pending_pattern_failed()
                if _replan_after_empty_action("planner_or_grounding_empty"):
                    i = 0
                    s = self.observation
                    continue
                self.episode_status = "no_action"
                self.logger.error(
                    "CWME stopped before env.step: no admissible action was produced"
                )
                break
            raw_action = sar_hat['action']
            preds = [raw_action] if isinstance(raw_action, str) else list(raw_action)
            raw_obs = self.env.look()
            exploit = (
                is_exploit_phase(self.episode_progress, self.last_score)
                or is_post_focus_manipulation_exploit(
                    self.task,
                    self.past_actions,
                    self.observation,
                    self.inventory,
                    self.last_score,
                )
            )
            last_action = self.past_actions[-1] if self.past_actions else ""
            perception = self.orchestrator.plan_perception(
                self.episode_progress,
                self.last_score,
                self.stagnant_steps,
                len(self.past_actions),
                last_action,
                agent=self,
            )
            skip_kg = perception.skip_kg
            skip_action_reflect = perception.skip_action_reflect
            skip_wm_refine = perception.skip_wm_refine
            exec_decision = self.orchestrator.resolve_execution(
                self,
                preds,
                self.env,
                self.past_actions,
                self.sbert,
                self.logger,
                task=self.task,
                raw_obs=raw_obs,
                current_score=self.last_score,
                score_max=self.episode_progress["score_max"],
                steps_since_gain=self.episode_progress["steps_since_gain"],
                exploit_phase=exploit,
                last_gain_family=self.episode_progress.get("last_gain_family", ""),
                failed_actions=self.recent_failed_actions,
            )
            a_hat = exec_decision.action
            self.last_exec_source = getattr(exec_decision, "source", "") or ""
            decision_level = getattr(exec_decision, "computation_level", None)
            self._last_exec_computation_level = int(
                2 if decision_level is None else decision_level
            )
            self._last_decision_source = getattr(exec_decision, "decision_source", "") or self.last_exec_source
            if getattr(self, "_step_used_actor_llm", False):
                self._last_decision_source = "cognition"
                self._last_exec_computation_level = 2
            if not (a_hat or "").strip() and _pattern_read_enabled(self):
                a_hat = self._invoke_cognition_actor(
                    preds,
                    plan_hint=(sar_hat.get("reward (env response)") or "") if isinstance(sar_hat, dict) else "",
                )
                if a_hat:
                    self._last_decision_source = "cognition"
                    self._last_exec_source = "actor_llm"
                    self._last_exec_computation_level = 2
                    # This is the direct Actor recovery branch: there is no
                    # ExecutionDecision to pass through GroundingFacade's
                    # normal metric finalizer, but the action still consumes
                    # one full-cognition decision and must be counted.
                    metrics = getattr(self, "execution_metrics", None)
                    if metrics is not None:
                        from core.execution.execution_types import ComputationLevel
                        metrics.record_step(
                            level=ComputationLevel.FULL_COGNITION,
                            source="actor_llm",
                        )
                    src_log = getattr(self, "action_source_log", None)
                    if src_log is not None:
                        from core.evolution.action_source import ActionSource
                        src_log.record(
                            step=len(self.past_actions or []),
                            action=a_hat,
                            source=ActionSource.COGNITION,
                            decision_source="cognition",
                            route_level=2,
                            llm_called=True,
                        )
            if (
                not (a_hat or "").strip()
                and not _pattern_read_enabled(self)
                and not grounding_only_mode(self)
            ):
                try:
                    env_valid = set(self.env.getValidActionObjectCombinations())
                except Exception:
                    env_valid = set()
                recent = {(a or "").strip().lower() for a in (self.past_actions or [])[-5:] if a}
                if is_scienceworld(self.env):
                    from core.scienceworld_policy import sci_nonempty_fallback
                    fill = sci_nonempty_fallback(
                        self.task, env_valid, recent,
                        (self.observation or "") + " " + (self.inventory or ""),
                        self.past_actions,
                        failed=getattr(self, "recent_failed_actions", None) or set(),
                    )
                    a_hat = fill or "look around"
                    self.last_exec_source = "sciworld_fast"
                elif is_alfworld(self.env):
                    a_hat = "look" if any(
                        (str(v) or "").strip().lower() == "look" for v in env_valid
                    ) else (a_hat or "look")
                    self.last_exec_source = "alfworld_fast"

            if not (a_hat or "").strip():
                _mark_pending_pattern_failed()
                if _replan_after_empty_action("execution_resolution_empty"):
                    i = 0
                    s = self.observation
                    continue
                self.logger.error(
                    "CWME stopped before env.step: execution returned no admissible action"
                )
                break

            # Final execution boundary: all automatic sources, including
            # Actor recovery and artifact reuse, must be revalidated against
            # the live environment state.  Grounding earlier in the step is
            # only a proposal check; it can become stale or return a
            # normalized spelling.  Never call env.step with an inferred
            # command.
            canonical_action = self._canonical_live_env_action(
                a_hat,
                source=self.last_exec_source or self._last_decision_source,
            )
            if not canonical_action:
                _mark_pending_pattern_failed()
                if _replan_after_empty_action("not_in_live_admissible_actions"):
                    i = 0
                    s = self.observation
                    continue
                self.episode_status = "no_action"
                self.logger.error(
                    "CWME stopped before env.step: action was not in the "
                    "current admissible action set"
                )
                break
            a_hat = canonical_action
            self._mark_pattern_execution_for_action(a_hat, exec_decision)

            # A4: whether Pattern milestone landed as the executed action.
            try:
                pkt = build_state_packet(self, score=self.last_score)
                milestone = (getattr(pkt, "pattern_next_milestone", "") or "").strip().lower()
            except Exception:
                milestone = ""
            if self.logger:
                executed = (a_hat or "").strip().lower()
                hit = bool(milestone) and executed == milestone
                if not hit and milestone and executed:
                    # Abstract process milestones (heat/take/…) count as hit when
                    # the executed command shares the same verb family.
                    try:
                        mf = cmd_verb_family(milestone)
                        ef = cmd_verb_family(executed)
                        if mf and mf == ef and mf not in ("go", "open", "look", "examine", "other"):
                            hit = True
                        elif milestone in executed or executed.startswith(milestone + " "):
                            hit = True
                    except Exception:
                        pass
                self.logger.info(
                    "[PatternHIT] milestone=%r executed=%r hit=%s "
                    "source=%s routed_from=%s plateau_steps=%s",
                    milestone,
                    executed,
                    hit,
                    getattr(exec_decision, "source", ""),
                    getattr(exec_decision, "routed_from", ""),
                    int((self.episode_progress or {}).get("steps_since_gain", 0) or 0),
                )

            try:
                s_next, r, info, termination = await self.step(
                    a_hat,
                    store,
                    skip_kg=skip_kg,
                    skip_action_reflect=skip_action_reflect,
                    skip_wm_refine=skip_wm_refine,
                )
                ambig_steps = 0
                while is_env_ambiguous_response(r) and not termination and ambig_steps < 2:
                    ambig_steps += 1
                    self._ambiguous_resolution_steps = int(
                        getattr(self, "_ambiguous_resolution_steps", 0) or 0
                    ) + 1
                    disambig = resolve_ambiguous_menu_action(
                        r, self.env, self.task, self.env.look(), a_hat, self.logger,
                    )
                    if not disambig:
                        break
                    disambig = self._canonical_live_env_action(
                        disambig,
                        source="ambiguous_resolution",
                    )
                    if not disambig:
                        self.logger.warning(
                            "CWME discarded ambiguous resolution outside live "
                            "admissible actions"
                        )
                        break
                    s_next, r, info, termination = await self.step(
                        disambig,
                        store,
                        skip_kg=skip_kg,
                        skip_action_reflect=skip_action_reflect,
                        skip_wm_refine=skip_wm_refine,
                    )
                    metrics = getattr(self, "execution_metrics", None)
                    if metrics is not None and hasattr(metrics, "record_ambiguous_resolution"):
                        metrics.record_ambiguous_resolution()
                print(info['moves'], termination)
            except KeyboardInterrupt:
                self.episode_status = "interrupted"
                print("Manual interruption detected. Stopping...")
                return scores, t

            scores.append(info['score'])
            t.append(time.time())
            print(f't = {time.time()}: Exectued action `{self.past_actions[-1:]}` | received response `{self.action_res[-1:]}` | number of token sent so far: {self.token_sent} | number of token received so far: {self.token_received}')
            self.logger.info(f"t = {time.time()}: Exectued action `{self.past_actions[-1:][0]}` | received response `{self.action_res[-1:][0]} | Total Score {info['score']} | number of token sent so far: {self.token_sent} | number of token received so far: {self.token_received}`")
            prev_score = self.last_score
            current_score = info['score']
            env_failed = self._record_execution_outcome(
                prev_score=prev_score,
                current_score=current_score,
                response=r,
                pattern_milestone=milestone,
            )
            if termination:
                # Commit the terminal transition before final HKG/pattern
                # consolidation; the normal post-step path is skipped below.
                terminal_score = int(info.get('score', self.last_score) or 0)
                self.last_score = terminal_score
                self.episode_progress["score_max"] = max(
                    int(self.episode_progress.get("score_max", 0) or 0),
                    terminal_score,
                )
                self.episode_status = "success" if terminal_score >= 100 else "terminated"
                break

            # past_actions[0] is '<START>'; step budget counts real env steps only.
            steps_taken = max(0, len(self.past_actions) - 1)
            if steps_taken >= self.step_limit:
                final_step_score = int(info.get('score', self.last_score) or 0)
                self.last_score = final_step_score
                self.episode_progress["score_max"] = max(
                    int(self.episode_progress.get("score_max", 0) or 0),
                    final_step_score,
                )
                self.episode_status = "step_limit"
                self.logger.info(f"Reached step limit of {self.step_limit}. Finished with a score of {max(scores)}")
                break

            sar = {
                'state': s,
                'action': a_hat,
                'env response': r,
                'next state': s_next,
                'score': current_score,
            }
            if env_failed:
                self.recent_failed_actions.add((a_hat or "").strip().lower())
                self.recent_failed_actions = set(list(self.recent_failed_actions)[-16:])
            elif "stove appears broken" in (r or "").lower():
                self.recent_failed_actions.add("activate stove")
            elif current_score > prev_score:
                self.recent_failed_actions.clear()
            # A flat score is not sufficient evidence of stagnation in AlfWorld,
            # but a flat score plus an unchanged transition is.  Counting every
            # process verb as progress used to hide repeated no-op actions and
            # delayed recovery until the 50-step limit.
            if current_score > prev_score:
                self.stagnant_steps = 0
            elif (is_alfworld(self.env) or is_scienceworld(self.env)) and not env_failed:
                transition = getattr(self, "_last_transition", None)
                meaningful = bool(
                    transition is not None
                    and getattr(transition, "meaningful_change", False)
                )
                if meaningful:
                    self.stagnant_steps = 0
                else:
                    self.stagnant_steps += 1
            elif current_score <= prev_score:
                self.stagnant_steps += 1
            else:
                self.stagnant_steps = 0
            self.episode_progress = update_episode_progress(
                self.episode_progress,
                prev_score,
                current_score,
                a_hat,
                progress_kind=(
                    getattr(getattr(self, "_last_transition", None), "progress_kind", "")
                    or ""
                ),
                meaningful_change=bool(
                    getattr(getattr(self, "_last_transition", None), "meaningful_change", False)
                ),
            )
            self.last_score = current_score
            # Outer ring: READ Memory Hub + CWME evolution (pattern + HKG).
            self.evolution_engine.after_step(
                self,
                score=current_score,
                prev_score=prev_score,
                transition=getattr(self, "_last_transition", None),
            )

            planned_str = (preds[0] if preds else "").strip().lower()
            executed_str = (a_hat or "").strip().lower()
            replan_eval = self.orchestrator.evaluate_replan(
                sar_hat,
                sar,
                current_score,
                prev_score,
                self.stagnant_steps,
                env_failed,
                self.replan_count,
                planned_str,
                executed_str,
                {},
                self.episode_progress,
                action_history=self.past_actions,
                agent=self,
            )
            if replan_eval.skip_refiner_llm:
                res = Orchestrator.default_refiner_result(
                    self._bounded_text(self.past_reflections, 6000), episode_task,
                )
                token_count = {"sent": 0, "received": 0}
            else:
                state_block = self._bounded_text(
                    self.dual_memory.refiner_block(self, score=current_score),
                    10000,
                )
                prompt = self.refiner_prompt.format(
                    self._bounded_text(self.past_reflections, 6000),
                    episode_task,
                    info['score'],
                    self._bounded_text(sar_hat, 6000),
                    self._bounded_text(sar, 6000),
                ) + Orchestrator.refiner_context_suffix(state_block)
                res, token_count = self.router.retry(
                    cwme_refiner_max_attempts(self),
                    "refiner", sm.Replanning, prompt, json=True, schema=sm.Replanning,
                )
                replan_eval = self.orchestrator.evaluate_replan(
                    sar_hat,
                    sar,
                    current_score,
                    prev_score,
                    self.stagnant_steps,
                    env_failed,
                    self.replan_count,
                    planned_str,
                    executed_str,
                    res,
                    self.episode_progress,
                    action_history=self.past_actions,
                    agent=self,
                )
            self.token_sent += token_count['sent']
            self.token_received += token_count['received']

            should_replan = replan_eval.should_replan
            replan_reason = replan_eval.reason
            replan_cap = replan_eval.replan_cap

            if should_replan and self.replan_count < replan_cap:
                self.replan_count += 1
                self.past_reflections = str(res.get('reflection', '') or '').strip()
                planning_reflection = self._bounded_text(
                    (
                        f"{self.past_reflections}\n"
                        "The main task and all named target entities are immutable. "
                        "Do not replace or reinterpret them.\n"
                        "Suggested local subtask from Refiner: "
                        f"{str(res.get('updated_subtask', '') or '').strip()}"
                    ).strip(),
                    8000,
                )
                print(f'<REPLANNING ({replan_reason}, count={self.replan_count})>')
                pprint(res)

                self.trajectory = self.trajectory[:i] + [sar]

                try:
                    if swift_env and not use_cwme_plan and not grounding_only_mode(self):
                        # Legacy: heuristic replan seed.
                        new_trajectory, printout = fallback_replan_trajectory(
                            self.task,
                            self.observation,
                            self.inventory,
                            self.trajectory,
                            max_steps=replan_eval.look_ahead_steps,
                            action_history=self.past_actions,
                            current_score=current_score,
                            action_seeder=lambda acts, sc=current_score: (
                                self.orchestrator.seed_heuristic_plan(
                                    self, acts, score=sc,
                                )
                            ),
                        )
                        total_token_sent = last_token_sent
                        total_token_received = last_token_received
                    elif swift_env and (use_cwme_plan or grounding_only_mode(self)):
                        new_trajectory, printout, total_token_sent, total_token_received = self.kg.get_trajectory(
                            valid_objects=self.possible_objects,
                            task=episode_task,
                            observation=self.observation,
                            inventory=self.inventory,
                            sequence=self.trajectory[-12:],
                            reflection=planning_reflection,
                            MAX_STEPS=replan_eval.look_ahead_steps,
                            MAX_QUERIES=self.max_query,
                            model=reason_model,
                        )
                        if self.logger:
                            self.logger.info(
                                "CWME replan: 9B get_trajectory (%s steps)", len(new_trajectory),
                            )
                    else:
                        new_trajectory, printout, total_token_sent, total_token_received = self.kg.get_trajectory(
                            valid_objects=self.possible_objects,
                            task=episode_task,
                            observation=self.observation,
                            inventory=self.inventory,
                            sequence=self.trajectory[-12:],
                            reflection=planning_reflection,
                            MAX_STEPS=replan_eval.look_ahead_steps,
                            MAX_QUERIES=self.max_query,
                            model=reason_model
                        )
                except Exception as e:
                    self.logger.error(
                        f"Replan get_trajectory failed: {e}; using fallback trajectory"
                    )
                    if (use_cwme_plan or grounding_only_mode(self)) and swift_env:
                        new_trajectory, printout = minimal_cognition_trajectory(
                            self.observation,
                            self.inventory,
                            max_steps=replan_eval.look_ahead_steps,
                            task=self.task,
                        )
                    else:
                        new_trajectory, printout = fallback_replan_trajectory(
                            self.task,
                            self.observation,
                            self.inventory,
                            self.trajectory,
                            max_steps=self.look_ahead,
                            action_history=self.past_actions,
                            current_score=current_score,
                            action_seeder=lambda acts, sc=current_score: (
                                self.orchestrator.seed_heuristic_plan(
                                    self, acts, score=sc,
                                )
                            ),
                        )
                    total_token_sent = last_token_sent
                    total_token_received = last_token_received

                self.token_sent += total_token_sent - last_token_sent
                self.token_received += total_token_received - last_token_received

                last_token_sent = total_token_sent
                last_token_received = total_token_received

                signature = _trajectory_signature(new_trajectory)
                if signature and signature in replan_signatures:
                    self.logger.warning(
                        "CWME rejected duplicate replan: reason=%s steps=%s",
                        replan_reason,
                        len(new_trajectory) - 1,
                    )
                    self.trajectory[i] = sar
                    self.replan_count = max(self.replan_count, replan_cap)
                else:
                    if signature:
                        replan_signatures.add(signature)
                    self.trajectory = new_trajectory
                print(f'New trajectory: \n{printout}')
                print('<FINISHED REPLANNING - RESUMING>')
                self.stagnant_steps = 0
            elif should_replan:
                self.logger.info(
                    f"Replan skipped: cap reached ({replan_cap}). Reason would have been: {replan_reason}"
                )
                self.trajectory[i] = sar
            else:
                self.trajectory[i] = sar
            i += 1
            s = s_next

            extend, extend_reason, extend_steps = should_extend_trajectory_for_subgoals(
                self.task,
                self.past_actions,
                self.observation,
                self.inventory,
                current_score,
                self.episode_progress.get("score_max", 0),
                at_trajectory_end=(i >= len(self.trajectory) - 1),
                termination=termination,
                steps_taken=max(0, len(self.past_actions) - 1),
                step_limit=self.step_limit,
                base_look_ahead=self.look_ahead,
                force_incomplete=is_alfworld(self.env) or is_scienceworld(self.env),
            )
            # AlfWorld / ScienceWorld: ensure we keep stepping until episode budget / success.
            steps_taken = max(0, len(self.past_actions) - 1)
            remain = max(0, int(self.step_limit) - steps_taken)
            swift_env = is_alfworld(self.env) or is_scienceworld(self.env)
            if (
                not extend
                and swift_env
                and (i >= len(self.trajectory) - 1)
                and not termination
                and current_score < 100
                and remain > 0
            ):
                extend = True
                extend_reason = (
                    "alfworld episode budget"
                    if is_alfworld(self.env)
                    else "scienceworld episode budget"
                )
                extend_steps = min(remain, max(15, int(self.look_ahead or 5) * 3))
            if extend:
                before_len = len(self.trajectory)
                if (use_cwme_plan or grounding_only_mode(self)) and swift_env:
                    self.trajectory, extend_msg = pad_cognition_trajectory(
                        self.trajectory,
                        observation=self.observation,
                        inventory=self.inventory,
                        pad_steps=extend_steps,
                        task=self.task,
                    )
                    if len(self.trajectory) > before_len:
                        self.logger.info(
                            "Extended trajectory to %s steps (%s; CWME cognition pad)",
                            len(self.trajectory),
                            extend_reason,
                        )
                else:
                    extra, _ = fallback_replan_trajectory(
                        self.task,
                        self.observation,
                        self.inventory,
                        self.trajectory,
                        max_steps=extend_steps,
                        action_history=self.past_actions,
                        current_score=current_score,
                        action_seeder=lambda acts, sc=current_score: (
                            self.orchestrator.seed_heuristic_plan(self, acts, score=sc)
                        ),
                    )
                    if len(extra) > before_len:
                        self.trajectory = extra
                        self.logger.info(
                            f"Extended trajectory to {len(extra)} steps ({extend_reason})"
                        )
                    elif swift_env and remain > 0:
                        # Guaranteed padding when fallback did not grow the sequence.
                        pad_n = min(remain, max(10, int(extend_steps or 10)))
                        cur_state = (
                            (self.trajectory[-1].get("next state") if self.trajectory else None)
                            or {"observation": self.observation, "inventory": self.inventory}
                        )
                        for _ in range(pad_n):
                            pad_action = "look" if is_alfworld(self.env) else ""
                            self.trajectory.append({
                                "state": cur_state,
                                "action": pad_action,
                                "reward (env response)": "Continue toward the task goal.",
                                "next state": cur_state,
                                "done or termination": False,
                            })
                        self.logger.info(
                            f"Extended trajectory to {len(self.trajectory)} steps "
                            f"({extend_reason}; padded)"
                        )

            stop_threshold = stagnant_early_stop_threshold(
                self.task,
                self.episode_progress,
                self.past_actions,
                f"{self.observation or ''} {self.inventory or ''}",
                current_score,
            )
            if (
                self.episode_progress.get(
                    "steps_since_relevant_state_change",
                    self.episode_progress.get("steps_since_gain", 0),
                ) >= stop_threshold
                and self.episode_progress.get("score_max", 0) > 0
                and current_score < 100
                and not termination
            ):
                self.logger.info(
                    f"Early stop: no relevant effect for "
                    f"{self.episode_progress.get('steps_since_relevant_state_change', 0)} "
                    f"steps at score {current_score} (max {self.episode_progress['score_max']})"
                )
                self.episode_status = "stagnation"
                break

        print('FINISHED RUN!')
        final_score = max(scores) if scores else self.last_score
        if getattr(self, "episode_status", "running") == "running":
            self.episode_status = "success" if final_score >= 100 else "incomplete"
        self.evolution_engine.finalize_episode(self, final_score=int(final_score))
        from core.trajectory_collector import maybe_export_episode
        maybe_export_episode(self, final_score=int(final_score))
        metrics = getattr(self, "execution_metrics", None)
        src_log = getattr(self, "action_source_log", None)
        if self.logger and metrics is not None:
            self.logger.info("[CWME] execution_metrics=%s", metrics.summary())
        if self.logger and src_log is not None:
            self.logger.info("[CWME] action_source=%s", src_log.summary())
        return scores, t

    def refine(self, reason_model, sar_hat, sar):
        score = sar.get('score', 0) if isinstance(sar, dict) else 0
        state_block = self._bounded_text(
            self.dual_memory.refiner_block(self, score=score),
            10000,
        )
        prompt = self.refiner_prompt.format(
            self._bounded_text(self.past_reflections, 6000),
            self._bounded_text(getattr(self, "_episode_task", self.task), 2000),
            score,
            self._bounded_text(sar_hat, 6000),
            self._bounded_text(sar, 6000),
        ) + Orchestrator.refiner_context_suffix(state_block)
        res, _ = self.router.retry(
            cwme_refiner_max_attempts(self),
            "refiner", sm.Replanning, prompt, json=True, schema=sm.Replanning,
        )
        return res

    def _insert_trajectory_step(self, pointer, sar_hat, action, plan_hint=""):
        """Insert a single resolved trajectory step at pointer."""
        act = (action or "").strip().lower().replace("green house", "greenhouse")
        responses = [plan_hint or ""]
        next_states = [sar_hat.get("next state") or sar_hat.get("state") or {}]
        cur_s = sar_hat["state"]
        sar = {
            "state": cur_s,
            "action": act,
            "reward (env response)": responses[0],
            "next state": next_states[0],
            "done or termination": False,
        }
        sar["next state"] = sar_hat.get("next state")
        sar["done or termination"] = sar_hat.get("done or termination")
        self.trajectory.insert(pointer, sar)

    @staticmethod
    def _bounded_text(value, limit: int = 8000) -> str:
        """Bound cognition context while retaining both recent ends."""
        text = str(value or "")
        limit = max(256, int(limit or 8000))
        if len(text) <= limit:
            return text
        head = max(1, int(limit * 0.6))
        tail = max(1, limit - head)
        return text[:head] + "\n...[context truncated]...\n" + text[-tail:]

    def _canonical_live_env_action(self, action: str, *, source: str = "") -> str:
        """Return the environment's current spelling of an exact action.

        Action grounding may normalize commands to lowercase, while an
        adapter can expose a different spelling.  More importantly, this
        gate is evaluated immediately before automatic execution, after all
        planning and recovery code has run.  No inferred sibling or stale
        action can cross this boundary.
        """
        proposed = (action or "").strip()
        if not proposed:
            return ""
        try:
            live_actions = list(self.env.getValidActionObjectCombinations() or [])
        except Exception:
            live_actions = []
        by_lower = {
            str(value).strip().lower(): str(value).strip()
            for value in live_actions
            if str(value).strip()
        }
        canonical = by_lower.get(proposed.lower(), "")
        if not canonical and self.logger:
            self.logger.warning(
                "CWME action rejected by final live-action gate: %r source=%s "
                "valid_count=%s",
                proposed,
                source,
                len(by_lower),
            )
        return canonical

    def _mark_pattern_execution_for_action(self, action: str, decision=None) -> None:
        """Attach Pattern stage metadata to every artifact milestone execution."""
        plib = getattr(self, "pattern_library", None)
        if plib is None:
            return
        milestone = (
            getattr(decision, "milestone", "") if decision is not None else ""
        ) or getattr(plib, "last_served_milestone", "") or ""
        pattern_id = (
            getattr(decision, "pattern_id", "") if decision is not None else ""
        ) or getattr(plib, "last_active_pattern_id", "") or ""
        stage_index = (
            int(getattr(plib, "last_pattern_stage_index", -1) or -1)
        )
        milestone = str(milestone).strip().lower()
        if not milestone or not pattern_id:
            return
        try:
            from core.cwme_agent_hooks import _pattern_action_matches

            if not _pattern_action_matches(milestone, action):
                return
        except Exception:
            return
        self._last_pattern_execution = {
            "pattern_id": str(pattern_id),
            "stage_index": stage_index,
            "milestone": milestone,
            "action": (action or "").strip(),
        }

    def _recover_grounding_only_actor_action(
        self,
        blocked_action: str,
        proposals,
        valid_actions,
    ) -> str:
        """Recover one safe sibling after a grounding-only rejection.

        This is deliberately a bounded, context-only repair. It never calls a
        task policy or invents an action: candidates must be admissible, share
        the rejected command's verb family, pass the final grounding safety
        gate, and have lexical evidence in the live observation/proposals.
        """
        try:
            from core.action_primitives import action_content_tokens
            from core.scienceworld_grounding import is_safe_grounding_only_execution_action

            blocked = (blocked_action or "").strip().lower()
            # Use the shared primitive vocabulary.  The ALFWorld helper does
            # not know ScienceWorld's ``focus on`` family and can therefore
            # recover a rejected focus as an unrelated move/pour command.
            family = action_verb_family(blocked)
            if not blocked or not family:
                return ""
            recent = {
                (str(a) or "").strip().lower()
                for a in (getattr(self, "past_actions", []) or [])[-6:]
                if a
            }
            context = f"{self.env.look() or ''} {self.inventory or ''}".lower()
            context_tokens = action_content_tokens(context)
            proposal_tokens = set()
            for proposal in list(proposals or []):
                proposal_tokens.update(action_content_tokens(str(proposal)))
            ranked = []
            for candidate in sorted(valid_actions or set()):
                candidate = (str(candidate) or "").strip().lower()
                if (
                    not candidate
                    or candidate == blocked
                    or candidate in recent
                    or action_verb_family(candidate) != family
                ):
                    continue
                if not is_safe_grounding_only_execution_action(
                    candidate,
                    task=self.task,
                    past_actions=getattr(self, "past_actions", []) or [],
                    observation=context,
                ):
                    continue
                candidate_tokens = action_content_tokens(candidate)
                context_overlap = len(candidate_tokens & context_tokens)
                proposal_overlap = len(candidate_tokens & proposal_tokens)
                # Live context alone is too weak after a bad Actor proposal:
                # it can turn an unsafe target into another arbitrary sibling.
                if proposal_overlap <= 0:
                    continue
                evidence = 2 * context_overlap + proposal_overlap
                if evidence > 0:
                    ranked.append((evidence, context_overlap, proposal_overlap, -len(candidate), candidate))
            if not ranked:
                return ""
            return max(ranked)[-1]
        except Exception:
            return ""

    def _artifact_recovery_action(self, valid_actions, live_context: str = "") -> str:
        """Return the current Pattern/HKG action before generic Actor recovery.

        A legal action is not enough for recovery: selecting the first legal
        ``pick up`` or ``focus on`` command can move the episode away from the
        learned procedure.  Reuse the exact current artifact stage, grounded
        against the live environment, and fail closed when its HKG order
        evidence rejects it.
        """
        plib = getattr(self, "pattern_library", None)
        if plib is None or not getattr(plib, "enable_read", False):
            return ""
        milestone = (
            getattr(plib, "last_served_milestone", "")
            or getattr(plib, "last_pattern_template", "")
            or ""
        ).strip().lower()
        pattern_id = (getattr(plib, "last_active_pattern_id", "") or "").strip()
        stage_index = int(getattr(plib, "last_pattern_stage_index", -1) or -1)
        if not milestone or not pattern_id:
            return ""
        if (
            stage_index >= 0
            and hasattr(plib, "is_pattern_stage_failed")
            and plib.is_pattern_stage_failed(pattern_id, stage_index)
        ):
            return ""

        valid = {
            (str(value) or "").strip().lower()
            for value in (valid_actions or [])
            if (str(value) or "").strip()
        }
        if not valid:
            return ""
        try:
            from core.scienceworld_grounding import (
                ground_pattern_action,
                is_safe_grounding_only_execution_action,
            )

            evidence = list(getattr(plib, "last_pattern_preconditions", []) or [])
            expected = getattr(plib, "last_expected_transition", "") or ""
            if expected:
                evidence.append(expected)
            action = ground_pattern_action(
                milestone,
                valid,
                task=getattr(self, "task", "") or "",
                past_actions=getattr(self, "past_actions", []) or [],
                observation=live_context or (self.env.look() or ""),
                pattern_evidence=evidence,
                environment="scienceworld" if is_scienceworld(self.env) else "",
            )
            action = (action or "").strip().lower()
            if not action or action not in valid:
                return ""
            if action in {
                (str(item) or "").strip().lower()
                for item in (getattr(self, "past_actions", []) or [])[-4:]
                if item
            }:
                return ""
            if is_scienceworld(self.env) and not is_safe_grounding_only_execution_action(
                action,
                task=getattr(self, "task", "") or "",
                past_actions=getattr(self, "past_actions", []) or [],
                observation=live_context or (self.env.look() or ""),
                pattern_evidence=evidence,
            ):
                return ""
        except Exception:
            return ""

        hkg = None
        try:
            from core.world_model.agent_bridge import get_combined_historical_kg

            hkg = get_combined_historical_kg(self)
        except Exception:
            hkg = getattr(self, "historical_kg", None)
        hkg_retriever = getattr(self, "hkg_retriever", None)
        if hkg is not None and hkg_retriever is not None:
            try:
                signal = hkg_retriever.action_constraint_signal(
                    hkg,
                    action,
                    past_actions=getattr(self, "past_actions", []) or [],
                    observation=live_context or (self.env.look() or ""),
                )
                if signal is False:
                    if self.logger:
                        self.logger.info(
                            "CWME artifact recovery rejected by HKG: %r", action,
                        )
                    return ""
            except Exception:
                return ""
        return action

    def _invoke_cognition_actor(self, preds, *, plan_hint: str = "") -> str:
        """9B Actor for CWME cold-start / empty grounded action (Plan-Act step)."""
        planned = ""
        if preds:
            planned = (preds[0] if isinstance(preds[0], str) else str(preds[0])).strip()
        try:
            actor_cache_key = (
                len(getattr(self, "past_actions", []) or []),
                planned.lower(),
                (plan_hint or "").strip(),
                (self.env.look() or "").strip(),
            )
        except Exception:
            actor_cache_key = None
        cached_actor = getattr(self, "_cwme_actor_decision_cache", None)
        if actor_cache_key is not None and cached_actor:
            if cached_actor[0] == actor_cache_key:
                self._step_used_actor_llm = True
                return cached_actor[1]

        cognition_route = self.orchestrator.route_cognition(self)
        goal_text = f"Experiment task: {self.task}\nStep goal (from planner): {planned}"
        if plan_hint:
            goal_text += f"\nPlanner expected outcome: {plan_hint}"
        actor_ctx = self._bounded_text(self.dual_memory.actor_block(self), 10000)
        route_hint = self.orchestrator.cognition.actor_priority_hint(cognition_route)
        if route_hint:
            goal_text += f"\n\n{route_hint}"
        if actor_ctx:
            goal_text += f"\n\nStructured context:\n{actor_ctx}"

        # The Actor previously saw possible object names but not the current
        # executable action set.  That made semantically plausible commands
        # such as ``focus on mercury`` look valid even when the environment
        # could not execute them, after which the recovery loop spent steps on
        # repeated observations.  Keep this as grounding context: it does not
        # select a task-specific action for the Actor.
        try:
            valid_actions = sorted({
                (str(v) or "").strip().lower()
                for v in self.env.getValidActionObjectCombinations()
                if (str(v) or "").strip()
            })
        except Exception:
            valid_actions = []
        if valid_actions:
            max_actor_actions = 256
            shown = valid_actions[:max_actor_actions]
            suffix = (
                "\n- ... (truncated; return an empty list if the needed action "
                "is not shown)"
                if len(valid_actions) > max_actor_actions else ""
            )
            goal_text += (
                "\n\nCurrently admissible environment actions. Select the exact "
                "string from this list; do not invent an action or object:\n"
                + "\n".join(f"- {item}" for item in shown)
                + suffix
            )
        else:
            goal_text += (
                "\n\nNo admissible action list is available. Return an empty list "
                "rather than inventing a command."
            )
        prompt = self.actor_prompt.format(
            goal_text,
            self.env.look(),
            self.inventory,
            self.env.getPossibleObjects(),
        )
        res, token_count = self.router.retry(
            cwme_actor_max_attempts(self), "actor", sm.Actor, prompt,
            json=True, schema=sm.Actor,
        )
        self.token_sent += token_count["sent"]
        self.token_received += token_count["received"]
        self._step_used_actor_llm = True
        if isinstance(res, dict):
            res = normalize_actor_dict(res)
        actions = list((res or {}).get("actions") or [])
        if not actions:
            actions = [planned]

        # Actor output is model language, not an environment command.  Always
        # pass it through the same admissible-action grounding used by the
        # execution path so an unrecognized string is never sent to env.step.
        recent = {
            (str(a) or "").strip().lower()
            for a in (getattr(self, "past_actions", []) or [])[-6:]
            if a
        }
        live_context = self._bounded_text(
            f"{self.env.look() or ''} {self.inventory or ''}",
            12000,
        )
        # The final ScienceWorld safety check must see the same artifact
        # evidence that grounded the candidate.  Previously this variable was
        # referenced only inside the closure below and was never initialized;
        # the broad exception handler then rejected every candidate and sent
        # execution into unsafe sibling recovery.
        try:
            from core.execution.artifact_evidence import execution_pattern_evidence

            evidence = execution_pattern_evidence(
                self,
                task=getattr(self, "task", "") or "",
                observation=live_context,
            )
        except Exception:
            evidence = []
        try:
            from core.substitute_engine import ground_cwme_action

            chosen = ""

            def _science_candidate_allowed(candidate: str) -> bool:
                """Apply task safety after env grounding, before env.step."""
                if not is_scienceworld(self.env):
                    return True
                candidate = (candidate or "").strip().lower()
                if not candidate or candidate in recent:
                    # ``wait`` can legitimately be repeated to advance time;
                    # all other repeats are a common source of plateau loops.
                    if action_verb_family(candidate) != "wait":
                        return False
                try:
                    from core.scienceworld_policy import sci_action_allowed

                    if not sci_action_allowed(
                        candidate,
                        self.task,
                        env_valid=valid_actions,
                        task_aware=True,
                        action_history=getattr(self, "past_actions", []) or [],
                        ctx_text=live_context,
                        inventory=self.inventory or "",
                        current_score=int(getattr(self, "last_score", 0) or 0),
                        is_blocked=lambda act: act in (
                            getattr(self, "recent_failed_actions", set()) or set()
                        ),
                    ):
                        return False
                except Exception:
                    return False
                if grounding_only_mode(self):
                    from core.scienceworld_grounding import (
                        is_safe_grounding_only_execution_action,
                    )
                    return is_safe_grounding_only_execution_action(
                        candidate,
                        task=self.task,
                        past_actions=getattr(self, "past_actions", []) or [],
                        observation=live_context,
                        pattern_evidence=evidence,
                    )
                return True

            # A directly executable planner command is authoritative.  The
            # Actor is a repair path for an ungroundable plan, not a second
            # policy that can replace an already legal action with a weaker
            # ``focus``/observation command.
            grounded_plan = ground_cwme_action(
                [planned],
                self.env,
                logger=self.logger,
                blocked_actions=getattr(self, "recent_failed_actions", set()),
            ) if planned else ""
            if grounded_plan and _science_candidate_allowed(grounded_plan):
                chosen = grounded_plan
                if self.logger and chosen.strip().lower() != planned.strip().lower():
                    self.logger.info(
                        "CWME Actor kept grounded planner action %r -> %r",
                        planned,
                        chosen,
                    )
            if not chosen and is_scienceworld(self.env):
                artifact_action = self._artifact_recovery_action(
                    valid_actions,
                    live_context,
                )
                if artifact_action and _science_candidate_allowed(artifact_action):
                    chosen = artifact_action
                    try:
                        artifact_plib = getattr(self, "pattern_library", None)
                        self._last_pattern_execution = {
                            "pattern_id": getattr(
                                artifact_plib, "last_active_pattern_id", ""
                            ),
                            "stage_index": int(getattr(
                                artifact_plib, "last_pattern_stage_index", -1
                            ) or -1),
                            "milestone": getattr(
                                artifact_plib, "last_served_milestone", ""
                            ),
                            "action": chosen,
                        }
                    except Exception:
                        pass
                    if self.logger:
                        self.logger.info(
                            "CWME Actor recovery kept artifact action %r", chosen,
                        )
            if not chosen:
                # Ground each proposal separately.  Passing the whole list to
                # the substitute engine lets its lexical ranking select an
                # admissible but task-wrong focus before the safety gate sees it.
                for proposal in actions:
                    grounded = ground_cwme_action(
                        [proposal],
                        self.env,
                        logger=self.logger,
                        blocked_actions=getattr(self, "recent_failed_actions", set()),
                    )
                    if grounded and _science_candidate_allowed(grounded):
                        chosen = grounded
                        break
                if not chosen and not is_scienceworld(self.env):
                    chosen = ground_cwme_action(
                        actions,
                        self.env,
                        logger=self.logger,
                        blocked_actions=getattr(self, "recent_failed_actions", set()),
                    )
        except Exception:
            chosen = ""
        if not chosen:
            self._cwme_invalid_actor_streak = int(
                getattr(self, "_cwme_invalid_actor_streak", 0) or 0
            ) + 1
            if self.logger:
                self.logger.warning(
                    "CWME Actor produced no admissible action; candidates=%r "
                    "invalid_streak=%s",
                    actions[:3], self._cwme_invalid_actor_streak,
                )
            # Never allow an empty command to reach env.step().  This is a
            # policy-free execution guard: prefer an explicitly admissible
            # observation command, then a stable lexical fallback only when
            # the environment exposes no observation command.
            try:
                valid = {
                    (str(v) or "").strip().lower()
                    for v in self.env.getValidActionObjectCombinations()
                    if v
                }
            except Exception:
                valid = set()

            # For ScienceWorld, prefer a fresh task-safe admissible action over
            # an observation placeholder after rejecting an Actor proposal.
            # This gives the stall detector a chance to make real progress and
            # prevents ``look around`` from becoming the default recovery loop.
            if is_scienceworld(self.env) and valid:
                try:
                    chosen = self._artifact_recovery_action(valid, live_context)
                    if chosen and self.logger:
                        self.logger.info(
                            "CWME artifact recovery selected %r", chosen,
                        )
                    if chosen:
                        artifact_plib = getattr(self, "pattern_library", None)
                        self._last_pattern_execution = {
                            "pattern_id": getattr(
                                artifact_plib, "last_active_pattern_id", ""
                            ),
                            "stage_index": int(getattr(
                                artifact_plib, "last_pattern_stage_index", -1
                            ) or -1),
                            "milestone": getattr(
                                artifact_plib, "last_served_milestone", ""
                            ),
                            "action": chosen,
                        }
                    elif not chosen and not grounding_only_mode(self):
                        from core.scienceworld_policy import sci_nonempty_fallback

                        chosen = sci_nonempty_fallback(
                            self.task or "",
                            valid,
                            recent,
                            live_context,
                            getattr(self, "past_actions", []) or [],
                            failed=getattr(self, "recent_failed_actions", set()) or set(),
                            task_aware=True,
                        ) or ""
                except Exception:
                    chosen = ""

            # Recover a useful navigation command when the Actor uses a
            # non-environment surface form, e.g. ``move mug to desk 2`` while
            # only ``go to desk 2`` is admissible. The recovered command must
            # be admissible and share concrete entity tokens with the proposal.
            try:
                # An empty Actor response is common when the prompt asks for
                # a high-level operation (e.g. ``pick up magazine``) but the
                # environment exposes only a navigation primitive at the
                # current room.  Recover the task-directed legal action before
                # falling back to look/inventory.
                if is_alfworld(self.env) and cwme_heuristic_fallback_enabled(self):
                    from core.alfworld_policy import alfworld_fast_substitute

                    valid_actions = {
                        (str(v) or "").strip().lower()
                        for v in self.env.getValidActionObjectCombinations()
                        if v
                    }
                    recent_actions = {
                        (str(a) or "").strip().lower()
                        for a in (getattr(self, "past_actions", []) or [])[-5:]
                        if a
                    }
                    policy_action = alfworld_fast_substitute(
                        valid_actions,
                        recent_actions,
                        f"{self.env.look() or ''} {self.inventory or ''}",
                        self.task or "",
                        list(getattr(self, "past_actions", []) or []),
                        look=self.env.look() or "",
                        inventory=self.inventory or "",
                        logger=self.logger,
                    )
                    if policy_action:
                        chosen = policy_action
                        if self.logger:
                            self.logger.info(
                                "CWME Actor empty-output recovery %r -> %r",
                                planned,
                                chosen,
                            )

                from core.action_primitives import action_content_tokens

                proposal_tokens = set()
                for candidate in actions + [planned]:
                    proposal_tokens.update(action_content_tokens(candidate))
                recent = {
                    (str(a) or "").strip().lower()
                    for a in (getattr(self, "past_actions", []) or [])[-4:]
                    if a
                }
                recovery = []
                for candidate in valid:
                    cl = candidate.strip().lower()
                    if cl in recent or cl in {"look", "look around", "inventory", "task"}:
                        continue
                    if not (
                        cl.startswith("go to ")
                        or action_verb_family(cl) in ("move", "open")
                    ):
                        continue
                    overlap = len(proposal_tokens & action_content_tokens(cl))
                    if overlap:
                        recovery.append((overlap, -len(cl), cl))
                if not chosen and recovery:
                    chosen = max(recovery)[2]
                    if self.logger:
                        self.logger.info(
                            "CWME grounded navigation recovery %r -> %r",
                            planned,
                            chosen,
                        )
            except Exception:
                pass

            # Permit one observation to refresh the state, but do not cycle
            # through look/inventory/task after an invalid Actor proposal.
            # A second invalid proposal is returned to the caller as empty so
            # the episode can terminate/replan instead of paying another full
            # cognition + perception cycle with no state change.
            observation_actions = {"look", "look around", "inventory", "task"}
            recent_observation = any(
                (str(a) or "").strip().lower() in observation_actions
                for a in (getattr(self, "past_actions", []) or [])[-3:]
                if a
            )
            if not chosen and not recent_observation:
                for preferred in ("look", "look around", "inventory"):
                    if preferred in valid:
                        chosen = preferred
                        break
            if not chosen and recent_observation:
                if self.logger:
                    self.logger.warning(
                        "CWME suppressed ungrounded navigation recovery after "
                        "invalid Actor output"
                    )
            if not chosen and is_scienceworld(self.env) and valid:
                # Formal grounding has intentionally disabled the task-family
                # fallback.  It must still never turn an invalid Actor answer
                # into an empty env.step: choose only a structural, untried
                # action that cannot replay a named pattern target.  Navigation
                # and opening a visible door preserve information/progress and
                # defer task choice to the next cognition pass.
                history = {
                    (str(a) or "").strip().lower()
                    for a in (getattr(self, "past_actions", []) or [])[-8:]
                    if a
                }
                failed = {
                    (str(a) or "").strip().lower()
                    for a in (getattr(self, "recent_failed_actions", set()) or set())
                    if a
                }
                structural = []
                for candidate in sorted(valid):
                    cl = candidate.strip().lower()
                    if not cl or cl in history or cl in failed:
                        continue
                    if cl.startswith(("go to ", "open door to ", "teleport to ")):
                        structural.append((0, cl))
                    elif cl.startswith("open "):
                        structural.append((1, cl))
                    elif cl in {"look", "look around"} and not recent_observation:
                        structural.append((2, cl))
                if structural:
                    structural.sort()
                    chosen = structural[0][1]
                    if self.logger:
                        self.logger.info(
                            "CWME structural recovery selected %r after invalid Actor output",
                            chosen,
                        )
            if not chosen and self.logger:
                self.logger.error("CWME has no admissible fallback action")
            if not chosen:
                return ""
        if is_scienceworld(self.env) and grounding_only_mode(self):
            from core.scienceworld_grounding import (
                is_safe_grounding_only_execution_action,
            )
            if not is_safe_grounding_only_execution_action(
                chosen,
                task=self.task,
                past_actions=getattr(self, "past_actions", []) or [],
                observation=f"{self.env.look() or ''} {self.inventory or ''}",
                pattern_evidence=evidence,
            ):
                blocked = chosen
                self._grounding_blocked_actions = int(
                    getattr(self, "_grounding_blocked_actions", 0) or 0
                ) + 1
                if self.logger:
                    self.logger.info(
                        "CWME grounding-only blocked Actor action: %r; "
                        "discarding it", chosen,
                    )
                # A sibling with the same verb is not a valid recovery: for
                # ``move A to B`` both entities are part of the action's
                # semantics.  Replanning is the only safe recovery path.
                return ""
        if self.logger:
            self.logger.info("CWME cognition actor chose %r (planned=%r)", chosen, planned)
        if chosen not in {"look", "look around", "inventory", "task"}:
            self._cwme_invalid_actor_streak = 0
        if actor_cache_key is not None:
            self._cwme_actor_decision_cache = (actor_cache_key, chosen)
        return chosen

    def act(self, pointer, model):
        """
        Actor: translate planner goal to env action via cognition override,
        planner bypass, or 9B Actor. Grounding is unified in resolve_execution.

        ALFWorld: treat 9B Actor as stuck-only Sage — prefer fast/heuristic
        actions and skip Actor LLM unless genuinely stuck.
        """
        sar_hat = self.trajectory.pop(pointer)
        res = None
        try:
            planned = sar_hat.get("action", "")
            if isinstance(planned, list):
                planned = planned[0] if planned else ""
            plan_hint = sar_hat.get("reward (env response)", "") or ""
            planned_norm = (planned or "").strip().lower().replace("green house", "greenhouse")
            try:
                live_look = (self.env.look() or "").strip()
            except Exception:
                live_look = ""
            raw_look = live_look or (self.observation or "").strip()
            ctx_bypass = f"{raw_look} {self.inventory or ''}".lower()

            step_decision = self.orchestrator.resolve_trajectory_step(
                self,
                planned_norm,
                self.env,
                ctx_text=ctx_bypass,
                logger=self.logger,
                score=self.last_score,
            )
            # Swift-mode: if planner text blocked fast path, retry with empty plan.
            if (
                (is_alfworld(self.env) or is_scienceworld(self.env))
                and (step_decision is None or not step_decision.action)
            ):
                step_decision = self.orchestrator.resolve_trajectory_step(
                    self,
                    "",
                    self.env,
                    ctx_text=ctx_bypass,
                    logger=self.logger,
                    score=self.last_score,
                )
            if step_decision is not None and step_decision.action:
                action = step_decision.action
                self.last_exec_source = getattr(step_decision, "source", "") or ""
                if step_decision.source == "cognition_route":
                    self.logger.info(
                        f"Cognition route ({step_decision.routed_from}): "
                        f"using grounded action {action!r} instead of planner {planned_norm!r}"
                    )
                self._insert_trajectory_step(pointer, sar_hat, action, plan_hint)
                print(f"replaced action {[sar_hat['action']]} with {[action]}")
                return True

            # Swift-mode: Actor LLM only when stuck; otherwise a cheap legal look.
            # CWME-on: no env-heuristic placeholder — fall through to 9B Actor.
            if (
                (is_alfworld(self.env) or is_scienceworld(self.env))
                and not _pattern_read_enabled(self)
                and not grounding_only_mode(self)
            ):
                stagnant = int(getattr(self, "stagnant_steps", 0) or 0)
                failed_n = len(getattr(self, "recent_failed_actions", None) or [])
                stuck_threshold = cwme_actor_stuck_threshold(self)
                stuck = stagnant >= stuck_threshold or failed_n >= 3
                if not stuck:
                    fallback = "look" if is_alfworld(self.env) else "look around"
                    if is_scienceworld(self.env):
                        try:
                            env_valid = {
                                (str(v) or "").strip().lower()
                                for v in (
                                    self.env.getValidActionObjectCombinations() or []
                                )
                            }
                            from core.scienceworld_policy import sci_no_actor_fallback
                            fallback = sci_no_actor_fallback(
                                planned_norm,
                                self.task or "",
                                env_valid,
                                ctx_text=ctx_bypass,
                                action_history=list(
                                    getattr(self, "past_actions", []) or [],
                                ),
                                inventory=self.inventory or "",
                                current_score=int(
                                    getattr(self, "last_score", 0) or 0,
                                ),
                            )
                        except Exception:
                            pass
                    self.last_exec_source = (
                        "alfworld_no_actor"
                        if is_alfworld(self.env)
                        else "sciworld_no_actor"
                    )
                    if self.logger:
                        env_tag = "AlfWorld" if is_alfworld(self.env) else "ScienceWorld"
                        self.logger.info(
                            "%s skip Actor (not stuck); placeholder %r "
                            "(planned=%r stagnant=%s failed=%s)",
                            env_tag,
                            fallback, planned_norm, stagnant, failed_n,
                        )
                    self._insert_trajectory_step(pointer, sar_hat, fallback, plan_hint)
                    print(f"replaced action {[sar_hat['action']]} with {[fallback]}")
                    return True

            chosen = self._invoke_cognition_actor([planned_norm], plan_hint=plan_hint)
            self.last_exec_source = "actor_llm"
            self._insert_trajectory_step(pointer, sar_hat, chosen, plan_hint)
            print(f"replaced action {[sar_hat['action']]} with {[chosen]}")
            return True
        except Exception as e:
            self.logger.error(f"ERROR ACTING: {e}")
            if res is not None:
                print(f"Exception: {e} \n {res}")
            self.trajectory.insert(pointer, sar_hat)
            return False

    def reset_memory(self):
        """
        Removes entries from the fact_tuples table where agent_uuid matches self.uuid.
        Also removes any orphaned entities and relationships that are no longer referenced.
        """
        conn = None
        cur = None
        try:
            # Establish database connection and cursor
            conn = self.connection_pool.getconn()
            cur = conn.cursor()

            # Delete records from fact_tuples where agent_uuid matches self.uuid
            cur.execute("DELETE FROM fact_tuples WHERE agent_uuid = %s RETURNING source_entity_id, target_entity_id, relationship_id;", (self.this_uuid,))

            # Collect the ids of affected entities and relationships
            affected = cur.fetchall()
            entity_ids = {row[0] for row in affected}.union({row[2] for row in affected})
            relationship_ids = {row[1] for row in affected}

            # Commit the deletions in fact_tuples
            conn.commit()

            # Now remove any orphaned entities and relationships
            if entity_ids:
                placeholders = ', '.join(['%s'] * len(entity_ids))
                cur.execute(f"DELETE FROM entities WHERE id NOT IN (SELECT DISTINCT source_entity_id FROM fact_tuples) AND id NOT IN (SELECT DISTINCT target_entity_id FROM fact_tuples) AND id IN ({placeholders});", tuple(entity_ids))

            if relationship_ids:
                placeholders = ', '.join(['%s'] * len(relationship_ids))
                cur.execute(f"DELETE FROM relationships WHERE id NOT IN (SELECT DISTINCT relationship_id FROM fact_tuples) AND id IN ({placeholders});", tuple(relationship_ids))

            # Commit the final deletions
            conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Failed to reset memory for uuid {self.uuid}. Error: {e}")
        finally:
            if cur:
                cur.close()
            if conn:
                self.connection_pool.putconn(conn)

    async def interactive_mode(self, store=False):
        """
        Allows for direct human interaction with the agent in a while loop for data collection purposes.
        """
        print("Entering interactive mode. Type 'exit' to quit.")
        try:
            while True:
                human_action = input("Enter action: ").strip()
                
                if human_action.lower() == 'exit':
                    print("Exiting interactive mode.")
                    break
                
                # Simulate the human action in the environment and update the agent's state
                response, _, termination, info = self.env.step(human_action)
                print(f"Environment Response: {response}")
                
                # Update agent's internal state
                if "move to" in response or "teleport" in response:
                    # print('new')
                    await self.update(store=store)
                else:
                    await self.update(action = True, store=store)
                
                if termination:
                    print("The task has been terminated based on the action taken.")
                    break
                
                
        except KeyboardInterrupt:
            print("Interactive mode interrupted.")

    def clear_run_kg(self) -> int:
        """Delete all fact_tuples for this agent's UUID (empty dedicated run store)."""
        conn = None
        cur = None
        try:
            conn = self.connection_pool.getconn()
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM fact_tuples WHERE agent_uuid = %s;",
                (self.this_uuid,),
            )
            n = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
            conn.commit()
            if self.logger:
                self.logger.info("clear_run_kg: deleted %s facts for %s", n, self.this_uuid)
            return int(n or 0)
        except Exception as e:
            if conn:
                conn.rollback()
            if self.logger:
                self.logger.error("clear_run_kg failed: %s", e)
            print(f"Failed to clear run KG for {self.this_uuid}: {e}")
            return 0
        finally:
            if cur:
                cur.close()
            if conn:
                self.connection_pool.putconn(conn)

    def seed_kg_from_base(self, base_uuid: str) -> int:
        """
        Copy pretrained fact_tuples from ``base_uuid`` into this agent's UUID.

        Entities/relationships tables are shared globally; only agent-scoped facts
        are duplicated. The base graph is never modified. Returns rows inserted.
        """
        base_uuid = (base_uuid or "").strip()
        run_uuid = self.this_uuid
        if not base_uuid or not run_uuid or base_uuid == run_uuid:
            return 0
        conn = None
        cur = None
        try:
            conn = self.connection_pool.getconn()
            cur = conn.cursor()
            # Ensure run namespace is empty before seeding (idempotent re-entry).
            cur.execute("DELETE FROM fact_tuples WHERE agent_uuid = %s;", (run_uuid,))
            cur.execute(
                """
                INSERT INTO fact_tuples (
                    source_entity_id, relationship_id, target_entity_id,
                    agent_uuid, recall_strength
                )
                SELECT
                    source_entity_id, relationship_id, target_entity_id,
                    %s, COALESCE(recall_strength, 1)
                FROM fact_tuples
                WHERE agent_uuid = %s
                """,
                (run_uuid, base_uuid),
            )
            n = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
            conn.commit()
            if self.logger:
                self.logger.info(
                    "seed_kg_from_base: inserted %s facts (%s -> %s)",
                    n, base_uuid, run_uuid,
                )
            return int(n or 0)
        except Exception as e:
            if conn:
                conn.rollback()
            if self.logger:
                self.logger.error("seed_kg_from_base failed: %s", e)
            print(f"Failed to seed KG from base {base_uuid} -> {run_uuid}: {e}")
            return 0
        finally:
            if cur:
                cur.close()
            if conn:
                self.connection_pool.putconn(conn)

    def get_uuid(self):
        return self.this_uuid
