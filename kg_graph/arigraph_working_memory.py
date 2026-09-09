"""
AriGraph-style working memory for DAVIS: incremental triplets, stale-fact pruning,
lightweight retrieval (SentenceTransformer), and episodic hints.
Uses the same local LLM stack as DAVIS (`utils.get_response`).
"""
from __future__ import annotations

import os
import re
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure project root on path when this module is imported as kg_graph.*
_DAVIS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _DAVIS_ROOT not in sys.path:
    sys.path.insert(0, _DAVIS_ROOT)

from utils import get_response  # noqa: E402

# --- Prompts (condensed from AriGraph `prompts/prompts.py`) ---

PROMPT_EXTRACTION_CURRENT = """Objective: Build a small factual knowledge graph from the observation as triplets "subject, relation, object".

Rules:
- Use semicolon to separate triplets. Each triplet has exactly 3 comma-separated parts.
- Max ~7 words per subject/object; relation can be slightly longer.
- Extract only concrete facts; use "could be" style for guesses.
- If something is taken into inventory: "item, is in, inventory".
- Do not include agent location as "you, are in, room".
- Do not use 'none' as entity.

Previously extracted triplets (avoid duplicates): {example}

Observation:
{observation}

Extracted triplets:"""

PROMPT_REFINING_ITEMS = """You will be provided with existing triplets and new triplets (all as "subject, relation, object").
Mark outdated existing triplets that are clearly superseded by new triplets (e.g. item moved from locker to inventory).

If nothing should be removed, output exactly: []

Output ONLY in this format (no explanation):
[[outdated_triplet_1 -> actual_triplet_1], [outdated_triplet_2 -> actual_triplet_2], ...]

Existing triplets: {ex_triplets}
New triplets: {new_triplets}
Replacing:"""


def clear_triplet(triplet: List) -> List:
    if triplet[0] == "I":
        triplet = ["inventory", triplet[1], triplet[2]]
    if triplet[1] == "I":
        triplet = [triplet[0], "inventory", triplet[2]]
    if triplet[0] == "P":
        triplet = ["player", triplet[1], triplet[2]]
    if triplet[1] == "P":
        triplet = [triplet[0], "player", triplet[2]]
    return [
        triplet[0].lower().strip('\"\' .`;:'),
        triplet[1].lower().strip('\"\' .`;:'),
        {"label": triplet[2]["label"].lower().strip('\"\' .`;:')},
    ]


def process_triplets(raw: str) -> List[List]:
    if not raw:
        return []
    parts = raw.split(";")
    triplets = []
    for triplet in parts:
        if len(triplet.split(",")) != 3:
            continue
        if triplet and triplet[0] in "123456789":
            triplet = triplet[2:]
        subj, relation, obj = triplet.split(",", 2)
        subj = subj.split(":")[-1].strip(' \'\n"')
        relation = relation.strip(' \'\n"')
        obj = obj.strip(' \'\n"')
        if not subj or not relation or not obj:
            continue
        triplets.append([subj, obj, {"label": relation}])
    return triplets


def parse_triplets_removing(text: str) -> List[List]:
    if not text:
        return []
    t = text.split("[[")[-1] if "[[" in text else text.split("[\n[")[-1]
    t = t.replace("[", "").strip("]")
    pairs = t.split("],")
    parsed = []
    for pair in pairs:
        sp = pair.split("->")
        if len(sp) != 2:
            continue
        first = sp[0].split(",")
        if len(first) != 3:
            continue
        subj, rel, obj = (
            first[0].strip(' "\'\n'),
            first[1].strip(' "\'\n'),
            first[2].strip(' "\'\n'),
        )
        if subj and rel and obj:
            parsed.append([subj, obj, {"label": rel}])
    return parsed


def triplet_str(t: List) -> str:
    return f"{t[0]}, {t[2]['label']}, {t[1]}"


def find_direction(action: str) -> str:
    a = (action or "").lower()
    if "north" in a:
        return "is north of"
    if "east" in a:
        return "is east of"
    if "south" in a:
        return "is south of"
    if "west" in a:
        return "is west of"
    return "can be achieved from"


def find_opposite_direction(action: str) -> str:
    a = (action or "").lower()
    if "north" in a:
        return "is south of"
    if "east" in a:
        return "is west of"
    if "south" in a:
        return "is north of"
    if "west" in a:
        return "is east of"
    return "can be achieved from"


class AriGraphWorkingMemory:
    """In-memory triplet graph + episodic store; mirrors AriGraph ContrieverGraph.update core logic."""

    def __init__(
        self,
        model: str,
        config,
        embedder,
        logger=None,
        router=None,
        max_episodic: int = 48,
    ):
        self.model = model
        self.config = config
        self.embedder = embedder
        self.logger = logger
        self.router = router
        self.max_episodic = max_episodic
        self.triplets: List[List] = []
        self.obs_episodic: Dict[str, Tuple[List[str], np.ndarray]] = {}
        self._last_new_strings: List[str] = []

    def _llm_call(self, task: str, prompt: str):
        if self.router:
            return self.router.call(task, prompt, json=False)
        return get_response(self.model, prompt, json=False, config=self.config)

    def clear(self) -> None:
        self.triplets = []
        self.obs_episodic = {}
        self._last_new_strings = []

    def get_all_triplet_strings(self) -> List[str]:
        return [triplet_str(t) for t in self.triplets]

    def get_latest_triplet_strings(self) -> List[str]:
        return list(self._last_new_strings)

    def _exclude_new_vs_existing(self, raw: List[List]) -> List[str]:
        out = []
        for t in raw:
            ct = clear_triplet(t)
            if ct[2]["label"] == "free":
                continue
            if ct not in self.triplets:
                out.append(triplet_str(ct))
        return out

    def _add_triplets(self, raw: List[List]) -> None:
        for t in raw:
            if t[2].get("label") == "free":
                continue
            ct = clear_triplet(t)
            if ct not in self.triplets:
                self.triplets.append(ct)

    def _delete_triplets(self, to_remove: List[List], locations: Set[str]) -> None:
        for t in to_remove:
            ct = clear_triplet(t)
            if ct[0] in locations and ct[1] in locations:
                continue
            if ct in self.triplets:
                self.triplets.remove(ct)

    def _associated_one_hop(self, items: Set[str]) -> List[str]:
        items_l = {i.lower() for i in items}
        assoc = []
        for t in self.triplets:
            s, o, rel = t[0], t[1], t[2]["label"]
            if (s in items_l or o in items_l) and triplet_str(t) not in assoc:
                assoc.append(triplet_str(t))
        return assoc

    def _retrieve(self, query: str, triplet_strings: List[str], topk: int = 6, thr: float = 0.25) -> List[str]:
        if not triplet_strings:
            return []
        q = self.embedder.encode(query, normalize_embeddings=True)
        embs = self.embedder.encode(triplet_strings, normalize_embeddings=True)
        if embs.ndim == 1:
            sims = np.array([float(np.dot(embs, q))])
        else:
            sims = np.dot(embs, q)
        order = np.argsort(-sims)
        out = []
        for i in order[:topk]:
            if sims[int(i)] >= thr:
                out.append(triplet_strings[int(i)])
        return out

    def _top_episodic(
        self, prev_subgraph: List[str], plan: str, observations: List[str], topk: int
    ) -> List[str]:
        if not self.obs_episodic or topk <= 0:
            return []
        plan_emb = self.embedder.encode((plan or "")[:2000], normalize_embeddings=True)
        scored = []
        for obs_text, (trip_strs, emb) in self.obs_episodic.items():
            if obs_text in observations:
                continue
            sim = float(np.dot(plan_emb, emb))
            overlap = 0
            if prev_subgraph:
                ps = set(" ".join(prev_subgraph).lower().split())
                for ts in trip_strs:
                    for tok in ts.lower().split(","):
                        tok = tok.strip()
                        if len(tok) > 2 and tok in ps:
                            overlap += 1
            scored.append((obs_text, sim + 0.02 * min(overlap, 20)))
        scored.sort(key=lambda x: -x[1])
        return [k for k, _ in scored[:topk]]

    def _trim_episodic(self) -> None:
        if len(self.obs_episodic) <= self.max_episodic:
            return
        # drop oldest by insertion order (dict preserves order in Py3.7+)
        while len(self.obs_episodic) > self.max_episodic:
            first_key = next(iter(self.obs_episodic))
            del self.obs_episodic[first_key]

    def update(
        self,
        observation: str,
        observations_hist: List[str],
        plan: str,
        prev_subgraph: List[str],
        locations: Set[str],
        curr_location: str,
        previous_location: str,
        action: str,
        items1: Dict[str, int],
        topk_episodic: int = 2,
        skip_refine: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        Returns:
            associated_subgraph: list of triplet strings
            top_episodic: list of past observation strings
        """
        example = [re.sub(r"Step \d+: ", "", x) for x in (prev_subgraph or [])][:12]
        prompt = PROMPT_EXTRACTION_CURRENT.format(
            observation=observation or "",
            example=example,
        )
        try:
            response, _ = self._llm_call("wm_extract", prompt)
        except Exception as e:
            if self.logger:
                self.logger.error(f"[AriGraph WM] extraction LLM failed: {e}")
            return [], []

        new_raw = process_triplets(response or "")
        new_triplets = self._exclude_new_vs_existing(new_raw)
        new_triplets_str = [triplet_str(clear_triplet(t)) for t in new_raw if t[2].get("label") != "free"]

        items_ = {clear_triplet(t)[0] for t in new_raw} | {clear_triplet(t)[1] for t in new_raw}
        associated_local = self._associated_one_hop(items_)
        words_to_exclude = [
            "west",
            "east",
            "south",
            "north",
            "associated with",
            "used for",
            "to be",
        ]
        associated_local = [
            x for x in associated_local if not any(w in x for w in words_to_exclude)
        ]

        predicted_outdated = []
        if not skip_refine:
            refine_prompt = PROMPT_REFINING_ITEMS.format(
                ex_triplets=", ".join(associated_local[:40]),
                new_triplets=", ".join(new_triplets[:40]),
            )
            try:
                refine_resp, _ = self._llm_call("wm_refine", refine_prompt)
                predicted_outdated = parse_triplets_removing(refine_resp or "")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"[AriGraph WM] refine LLM failed: {e}")
                predicted_outdated = []

        self._delete_triplets(predicted_outdated, locations)

        al = (action or "").lower()
        if "go to" not in al and "move to" not in al and "teleport" not in al:
            if curr_location and previous_location and curr_location != previous_location:
                new_raw.append(
                    [curr_location, previous_location, {"label": find_direction(action)}]
                )
                new_raw.append(
                    [previous_location, curr_location, {"label": find_opposite_direction(action)}]
                )

        self._add_triplets(new_raw)
        self._last_new_strings = new_triplets_str[:]

        triplets_all = self.get_all_triplet_strings()
        associated_subgraph: Set[str] = set()
        for query, _depth in (items1 or {}).items():
            for s in self._retrieve(str(query), triplets_all, topk=6, thr=0.22):
                associated_subgraph.add(s)

        for x in list(associated_subgraph):
            if x in new_triplets_str:
                associated_subgraph.discard(x)

        top_episodic = self._top_episodic(
            prev_subgraph or [],
            plan or "",
            observations_hist or [],
            topk_episodic,
        )

        try:
            obs_emb = self.embedder.encode((observation or "")[:2000], normalize_embeddings=True)
        except Exception:
            obs_emb = np.zeros_like(
                self.embedder.encode("empty", normalize_embeddings=True)
            )
        obs_key = (observation or "")[:2400] if observation else f"_obs_{len(self.obs_episodic)}"
        self.obs_episodic[obs_key] = (new_triplets_str, obs_emb)
        self._trim_episodic()

        return list(associated_subgraph), top_episodic
