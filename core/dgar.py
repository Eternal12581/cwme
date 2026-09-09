"""
DGAR: Detect → Generate → Apply → Remove knowledge update engine.

Outer-ring roles:
  WRITE  — observation / action reflection into working memory
  STORE  — optional persistent KG triplet store

Wraps AriGraph working memory and persistent KG extraction behind one
task-agnostic perception step used by ReasoningAgent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DGARResult:
    wm_subgraph: list[str] = field(default_factory=list)
    episodic_hints: list[str] = field(default_factory=list)
    new_triplet_strings: list[str] = field(default_factory=list)
    kg_prompt: str = ""
    room_changed: bool = False
    current_room: str = ""
    wm_context_text: str = ""


class DGAREngine:
    """Per-step dynamic graph update: WM triplets + optional KG store."""

    def __init__(self, logger=None):
        self.logger = logger
        self._last_room: str = ""

    @staticmethod
    def _parse_location(agent, ctx_text: str) -> str:
        if hasattr(agent, "_parse_location_hint"):
            return (agent._parse_location_hint(ctx_text) or "").lower()
        return ""

    async def perceive_step(
        self,
        agent,
        *,
        action: bool = False,
        store_kg: bool = True,
        skip_action_reflect: bool = False,
        skip_wm_refine: bool = False,
    ) -> DGARResult:
        """
        Run one perception cycle after an env step (or relocation-only update).

        Detect: room / observation change
        Generate: WM triplet extraction (+ optional action reflect)
        Apply: KG relation storage
        Remove: WM refine prunes stale triplets (inside arigraph_wm.update)
        """
        result = DGARResult()

        # STORE_KG=False plus skip_wm_refine=True explicitly requests a cheap
        # observation refresh. Previously this still called the 4B reflector
        # and AriGraph extractor, so read-only CWME episodes paid several
        # perception calls per environment step.
        if not store_kg and skip_wm_refine:
            raw_look = agent.env.look()
            agent.observation = raw_look or getattr(agent, "observation", "")
            new_task = agent.env.getTaskDescription()
            if new_task != agent.task:
                agent.past_reflections = ""
                agent.replan_count = 0
            agent.task = new_task
            agent.inventory = agent.env.inventory()
            agent.possible_actions = agent.env.getPossibleActions()
            agent.possible_objects = agent.env.getPossibleObjects()
            curr_loc = self._parse_location(agent, raw_look or "")
            prev_loc = (getattr(agent, "_wm_prev_location", None) or curr_loc or "").lower()
            result.room_changed = bool(curr_loc and prev_loc and curr_loc != prev_loc)
            result.current_room = curr_loc or ""
            if curr_loc:
                agent._wm_prev_location = curr_loc
            result.wm_subgraph = list(getattr(agent, "_wm_subgraph", []) or [])
            result.episodic_hints = list(getattr(agent, "_wm_episodic", []) or [])
            result.wm_context_text = getattr(agent, "_wm_context_text", "") or ""
            mf = getattr(agent, "memory_fusion", None)
            if mf is not None:
                mf.observe_step(agent)
            return result

        agent.observation, _ = agent.reflect(agent.env.look())
        new_task = agent.env.getTaskDescription()
        if new_task != agent.task:
            agent.past_reflections = ""
            agent.replan_count = 0
        agent.task = new_task
        agent.inventory = agent.env.inventory()
        agent.possible_actions = agent.env.getPossibleActions()
        agent.possible_objects = agent.env.getPossibleObjects()

        if action and not skip_action_reflect:
            prompt = (
                f"\nRECENT ACTION: {agent.past_actions[-1:]}\n"
                f"RESPONSE: {agent.action_res[-1:]}\n"
            )
            prompt, _ = agent.reflect(prompt)
        else:
            prompt = agent.observation

        raw_look = agent.env.look()
        obs_for_wm = (agent.observation or "")[:6000]
        loc_ctx = f"{raw_look or ''}\n{obs_for_wm}"
        curr_loc = self._parse_location(agent, loc_ctx)
        prev_loc = (getattr(agent, "_wm_prev_location", None) or curr_loc or "").lower()
        result.room_changed = bool(
            curr_loc and prev_loc and curr_loc != prev_loc
        )
        result.current_room = curr_loc or ""

        try:
            action_str = agent.past_actions[-1] if agent.past_actions else "start"
            items1 = agent._build_items1()
            if curr_loc:
                agent._known_locations.add(curr_loc)
            locs = {str(x).lower() for x in agent._known_locations}
            subgraph, top_ep = agent.arigraph_wm.update(
                observation=obs_for_wm,
                observations_hist=agent._obs_hist,
                plan=(agent.task or "")[:4000],
                prev_subgraph=agent._wm_prev_subgraph,
                locations=locs,
                curr_location=curr_loc,
                previous_location=prev_loc,
                action=action_str,
                items1=items1,
                topk_episodic=2,
                skip_refine=skip_wm_refine,
            )
            agent._wm_prev_subgraph = list(subgraph)[:48]
            agent._wm_subgraph = list(subgraph)[:28]
            agent._wm_episodic = list(top_ep)[:6]
            if curr_loc:
                agent._wm_prev_location = curr_loc
            result.wm_subgraph = list(agent._wm_subgraph)
            result.episodic_hints = list(agent._wm_episodic)
            result.wm_context_text = (
                "WORKING_MEMORY_SUBGRAPH:\n- "
                + "\n- ".join(subgraph[:28])
                + "\n\nEPISODIC_HINTS (past observations):\n- "
                + "\n- ".join(
                    (t[:500] + ("…" if len(t) > 500 else "")) for t in top_ep[:6]
                )
            )
            agent._wm_context_text = result.wm_context_text
            agent._obs_hist = (agent._obs_hist + [obs_for_wm])[-8:]
            wm_facts = agent.arigraph_wm.get_latest_triplet_strings()[:24]
            result.new_triplet_strings = list(wm_facts)
            if wm_facts:
                prompt = (
                    (prompt or "")
                    + "\n\nShort-term memory triplets (may overlap DB KG):\n"
                    + "; ".join(wm_facts)
                )
        except Exception as exc:
            if self.logger:
                self.logger.error(f"DGAR WM update failed: {exc}")
            agent._wm_context_text = ""
            agent._wm_subgraph = []
            agent._wm_episodic = []

        result.kg_prompt = prompt or ""
        if store_kg and result.kg_prompt:
            await agent.kg.relations_extraction(
                result.kg_prompt, coref_resolve=True, store=True, iterations=1,
            )
            if self.logger:
                self.logger.info(
                    f"DGAR apply: stored KG relations (room_changed={result.room_changed})"
                )

        self._last_room = curr_loc or self._last_room
        mf = getattr(agent, "memory_fusion", None)
        if mf is not None:
            mf.observe_step(agent)
        return result
