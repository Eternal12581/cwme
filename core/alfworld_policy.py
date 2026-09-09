"""
AlfWorld household policy — SciWorld-style deterministic grounding / fast path.

Mirrors ScienceWorld's architecture_fast_substitute idea for household tasks:
  explore → take goal object → go to destination → put/move → done

Also provides verb-family admissible matching so SBERT cannot map
`pick up mug` → `use desklamp` / `go to drawer`.
"""
from __future__ import annotations

import re
from typing import Iterable


# Canonical AlfWorld portable / receptacle tokens (lowercase stems).
_OBJECT_STEMS = (
    "alarmclock", "apple", "book", "bowl", "box", "bread", "butterknife", "cd",
    "candle", "cellphone", "cloth", "creditcard", "cup", "dishsponge", "egg",
    "fork", "glassbottle", "handtowel", "kettle", "keychain", "knife", "ladle",
    "laptop", "lettuce", "mug", "newspaper", "pan", "pen", "pencil",
    "peppershaker", "pillow", "plate", "pot", "potato", "remotecontrol",
    "saltshaker", "soapbar", "soapbottle", "spatula", "spoon", "spraybottle",
    "statue", "tissuebox", "toiletpaper", "tomato", "vase", "watch",
    "wateringcan", "winebottle",
)

_RECEP_STEMS = (
    "armchair", "bed", "cabinet", "cart", "coffeemachine", "coffeetable",
    "countertop", "desk", "desklamp", "diningtable", "drawer", "dresser",
    "fridge", "garbagecan", "laundryhamper", "microwave", "ottoman", "safe",
    "shelf",     "sidetable", "sinkbasin", "sofa", "stoveburner", "toilet",
    "toiletpaperhanger", "handtowelholder", "towelholder",
)

# Goal phrasing → preferred stems
_GOAL_OBJECT_ALIASES = {
    "coffee mug": "mug",
    "hand soap": "soapbottle",
    "soap bottle": "soapbottle",
    "bar of soap": "soapbar",
    "soap bar": "soapbar",
    "soap": "soapbar",
    "rack": "handtowelholder",
    "towel rack": "handtowelholder",
    "soap rack": "handtowelholder",
    "remote control": "remotecontrol",
    "credit card": "creditcard",
    "cell phone": "cellphone",
    "alarm clock": "alarmclock",
    "clock": "alarmclock",
    "desk lamp": "desklamp",
    "lamp": "desklamp",
    "phone": "cellphone",
    "remote": "remotecontrol",
    "garbage can": "garbagecan",
    "garbage bin": "garbagecan",
    "trash can": "garbagecan",
    "trash": "garbagecan",
    "bin": "garbagecan",
    "book shelf": "shelf",
    "bookshelf": "shelf",
    "shelves": "shelf",
    "shelf": "shelf",
    "desk": "desk",
    "table": "diningtable",
    "dining table": "diningtable",
    "side table": "sidetable",
    "coffee table": "coffeetable",
    "counter": "countertop",
    "countertop": "countertop",
    "fridge": "fridge",
    "refrigerator": "fridge",
    "microwave": "microwave",
    "cabinet": "cabinet",
    "drawer": "drawer",
    "safe": "safe",
    "vault": "safe",
    "sofa": "sofa",
    "couch": "sofa",
    "bed": "bed",
    "toilet": "toilet",
    "sink": "sinkbasin",
    # Portable paraphrases (plural + synonym → canonical AlfWorld stems)
    "disc": "cd",
    "discs": "cd",
    "disk": "cd",
    "disks": "cd",
    "cds": "cd",
    "cd": "cd",
    "key": "keychain",
    "keys": "keychain",
    "set of keys": "keychain",
    "key chain": "keychain",
    "keychain": "keychain",
    "pepper shaker": "peppershaker",
    "salt shaker": "saltshaker",
    "shaker": "peppershaker",
    "shakers": "peppershaker",
}


def _plural_forms(stem: str) -> tuple[str, ...]:
    """Common English plurals for short AlfWorld object stems."""
    s = (stem or "").strip().lower()
    if not s:
        return ()
    forms = [s, s + "s"]
    if s.endswith(("s", "x", "z", "ch", "sh")):
        forms.append(s + "es")
    elif s.endswith("y") and len(s) > 1 and s[-2] not in "aeiou":
        forms.append(s[:-1] + "ies")
    elif s.endswith("f"):
        forms.append(s[:-1] + "ves")
    elif s.endswith("fe"):
        forms.append(s[:-2] + "ves")
    return tuple(dict.fromkeys(forms))


def _token_matches_object_stem(tok: str, stem: str) -> bool:
    """True when a surface token is the stem or a simple plural of it."""
    t = (tok or "").strip().lower()
    s = (stem or "").strip().lower()
    if not t or not s:
        return False
    if t == s or t in _plural_forms(s):
        return True
    # Suffix aliases already in catalog (clock→alarmclock) handled separately.
    return False

_TAKE_VERBS = ("take", "pick up", "pickup", "get", "grab")
_PUT_VERBS = ("put", "place", "move", "drop", "set")
_GO_VERBS = ("go to", "goto", "walk to", "navigate to", "move to")
_OPEN_VERBS = ("open",)
_CLOSE_VERBS = ("close",)
_LOW_VALUE = ("look", "inventory", "help", "examine")
_META_CMDS = frozenset({"help", "look", "inventory"})

# Lighting / heating / cooling / washing stations inferred from wording + admissible cmds.
# Heat defaults to microwave first: stoveburner rarely admits ``heat … with``.
_PROCESS_STATIONS = {
    "look": ("use", ("desklamp", "lamp", "light")),
    # AlfWorld heat verb is ``heat X with microwave``; stoveburner hosts caused
    # holding agents to thrash burners and never emit a heat action.
    "heat": ("heat", ("microwave",)),
    "cool": ("cool", ("fridge", "refrigerator")),
    "clean": ("clean", ("sinkbasin", "sink")),
}


def _norm(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("green house", "greenhouse")
    t = re.sub(r"\s+", " ", t)
    return t


def _matches_dest_lock(action: str, lock: str) -> bool:
    """True when action targets the exact locked receptacle instance (not desk1⊂desk10)."""
    lock_n = _norm(lock)
    if not lock_n:
        return False
    al = _norm(action)
    if al in (f"go to {lock_n}", f"open {lock_n}"):
        return True
    m = re.search(r"\bto\s+(.+)$", al)
    if m and _norm(m.group(1)) == lock_n:
        return True
    if al.startswith("open ") and al[len("open "):].strip() == lock_n:
        return True
    if al.startswith("go to ") and al[len("go to "):].strip() == lock_n:
        return True
    return False


def _stemify(token: str) -> str:
    t = _norm(token).replace(" ", "")
    # strip trailing instance ids: mug1 / mug 1
    t = re.sub(r"\d+$", "", t)
    return t


def _strip_task_wrapper(text: str) -> str:
    """Drop env wrappers so household vs lab is decided by the goal clause."""
    t = (text or "").strip()
    t = re.sub(r"^your task is to:?\s*", "", t, flags=re.I).strip()
    return t


def task_looks_alfworld(task: str) -> bool:
    """
    True for household put/heat/cool/clean/look instructions.

    Do not use the shared wrapper "Your task is to" as a ScienceWorld marker:
    AlfWorld TextWorld feedback uses the same prefix. Lab tasks are identified
    by focus / phase-change / genetics / circuit wording instead.
    """
    t = _strip_task_wrapper(task).lower()
    if not t:
        return False
    if re.search(
        r"\b(?:first,?\s+focus on|state of matter|"
        r"unknown substance|melting\s+point|boiling\s+point|"
        r"freezing\s+point|electrically conductive|inclined\s+plane|"
        r"life stages?|dominant trait|recessive trait|allele|mendel|"
        r"flower pot|metal pot|blast furnace)\b"
        r"|find a\(n\)"
        r"|use chemistry"
        r"|create the substance"
        r"|focus on the (?:substance|thing)"
        r"|\bgrow a\b",
        t,
    ):
        return False
    return bool(re.search(
        r"\b(?:put|place|move|heat|cool|clean|look at|examine|find two|"
        r"pick up a|take a)\b",
        t,
    ))


def parse_alfworld_goal(task: str) -> dict:
    """
    Extract object / source / destination stems from a natural-language goal.

    Returns dict with keys: objects, sources, destinations (lists of stems).
    """
    if not task_looks_alfworld(task):
        return {
            "objects": [],
            "sources": [],
            "destinations": [],
            "put_back": False,
            "kind": "",
            "stations": [],
        }
    t = _norm(_strip_task_wrapper(task))
    objects: list[str] = []
    sources: list[str] = []
    destinations: list[str] = []

    def _uniq(xs: list[str]) -> list[str]:
        out = []
        for x in xs:
            if x and x not in out:
                out.append(x)
        return out

    def _stems_in_chunk(chunk: str) -> list[str]:
        chunk_n = _norm(chunk)
        compact = chunk_n.replace(" ", "")
        found: list[str] = []
        for phrase, stem in sorted(_GOAL_OBJECT_ALIASES.items(), key=lambda x: -len(x[0])):
            if re.search(rf"\b{re.escape(phrase)}\b", chunk_n) or compact == stem:
                if stem not in found:
                    found.append(stem)
        catalog = list(_RECEP_STEMS) + list(_OBJECT_STEMS)
        for stem in catalog:
            plurals = "|".join(re.escape(p) for p in _plural_forms(stem))
            if re.search(rf"\b(?:{plurals})\b", chunk_n) or (
                len(stem) >= 4 and stem in compact
            ):
                if stem not in found:
                    found.append(stem)
        # Suffix overlap: "clock" → alarmclock, "lamp" → desklamp (not read → bread)
        for tok in re.findall(r"[a-z]+", chunk_n):
            if len(tok) < 3:
                continue
            for stem in catalog:
                if _token_matches_object_stem(tok, stem):
                    hit = True
                elif stem.endswith(tok) and (len(tok) >= 5 or tok in _GOAL_OBJECT_ALIASES):
                    hit = True
                else:
                    hit = False
                if hit and stem not in found:
                    found.append(stem)
        return found

    # Canonical: (to) put/place/move <obj...> (on|in|onto|into|to|inside) <dest...>
    m = re.search(
        r"(?:to\s+)?(?:put|place|move|drop)\s+(?:a|an|some|the|my)?\s*(.+?)\s+"
        r"(?:on(?:to)?|in(?:to|side)?|to)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s*$|\.|\,)",
        t,
    )
    if m:
        obj_chunk, dest_chunk = m.group(1), m.group(2)
        dest_chunk = re.sub(r"\b(?:left|right)\s+of\b.*$", " ", dest_chunk)
        for stem in _stems_in_chunk(obj_chunk):
            if stem in _OBJECT_STEMS and stem not in objects:
                objects.append(stem)
        if re.search(r"\bsoap\b", obj_chunk):
            for extra in ("soapbar", "soapbottle"):
                if extra not in objects:
                    objects.append(extra)
        for stem in _stems_in_chunk(dest_chunk):
            if stem in _RECEP_STEMS and stem not in destinations:
                destinations.append(stem)

    # from / off → sources
    for m in re.finditer(
        r"(?:from|off)\s+(?:the\s+)?([a-z0-9 ]+?)(?:\s+to|\s+and|\s*,|\s*$|\.)",
        t,
    ):
        for stem in _stems_in_chunk(m.group(1)):
            if stem in _RECEP_STEMS and stem not in sources:
                sources.append(stem)

    # Object stems mentioned anywhere (longer stems first; avoid pen⊂pencil)
    compact = t.replace(" ", "")
    for phrase, stem in sorted(_GOAL_OBJECT_ALIASES.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", t) or phrase.replace(" ", "") in compact:
            if stem in _OBJECT_STEMS and stem not in objects:
                objects.append(stem)
    # "shaker(s)" implies both pepper and salt variants in AlfWorld.
    if re.search(r"\bshakers?\b", t):
        for extra in ("peppershaker", "saltshaker"):
            if extra not in objects:
                objects.append(extra)
    for stem in sorted(_OBJECT_STEMS, key=len, reverse=True):
        plurals = "|".join(re.escape(p) for p in _plural_forms(stem))
        if re.search(rf"\b(?:{plurals})\b", t) or re.search(
            rf"(?<![a-z]){re.escape(stem)}(?![a-z])", compact
        ):
            if any(stem != o and stem in o for o in objects):
                continue
            if stem not in objects:
                objects.append(stem)
    for tok in re.findall(r"[a-z]+", t):
        if len(tok) < 3:
            continue
        for stem in _OBJECT_STEMS:
            if _token_matches_object_stem(tok, stem) or (
                stem.endswith(tok) and (len(tok) >= 5 or tok in _GOAL_OBJECT_ALIASES)
            ):
                if stem not in objects and not any(stem != o and stem in o for o in objects):
                    objects.append(stem)
    # Explicit "two/another <noun>" when catalog scan still missed paraphrases.
    if not objects:
        for m in re.finditer(
            r"\b(?:two|another|both|pair of|sets? of)\s+([a-z]+)\b",
            t,
        ):
            tok = m.group(1)
            for stem in _OBJECT_STEMS:
                if _token_matches_object_stem(tok, stem) or (
                    stem.endswith(tok) and len(tok) >= 4
                ):
                    if stem not in objects:
                        objects.append(stem)
            alias = _GOAL_OBJECT_ALIASES.get(tok) or _GOAL_OBJECT_ALIASES.get(tok + "s")
            if alias and alias in _OBJECT_STEMS and alias not in objects:
                objects.append(alias)
            if tok in ("shaker", "shakers"):
                for extra in ("peppershaker", "saltshaker"):
                    if extra not in objects:
                        objects.append(extra)
    # Drop shorter stems that are prefixes/substrings of longer accepted objects
    objects = [
        o for o in objects
        if not any(o != other and o in other for other in objects)
    ]

    # If still no destination, look for "to/on/in the X" receptacles.
    # Mask activation phrasing so "turn on the lamp" is not a put destination.
    t_dest = re.sub(r"\b(?:turn(?:ing)?|switch(?:ing)?|toggle|focus|look)\s+on\b", " ", t)
    # Which-cabinet landmarks are not destinations: "cabinet left of the microwave"
    t_dest = re.sub(r"\b(?:left|right)\s+of\b.*$", " ", t_dest)
    # Process frames are not put destinations: "heat X in/with microwave", "cool in fridge".
    t_dest = re.sub(
        r"\b(?:heat|cool|clean|wash|rinse|microwave|cook)\b.{0,40}?"
        r"(?:in|with|using|on)\s+(?:the\s+|a\s+|an\s+)?[a-z ]+",
        " ",
        t_dest,
    )
    # Source frames: "take/find from/in the fridge" should not become destinations.
    t_dest = re.sub(
        r"\b(?:from|take|find|get|grab|pick(?:\s+up)?)\b.{0,24}?"
        r"(?:in|on|inside|from)\s+(?:the\s+|a\s+|an\s+)?[a-z ]+",
        " ",
        t_dest,
    )
    if not destinations:
        for m in re.finditer(
            r"(?:to|onto|on|in|into|inside)\s+(?:the\s+|a\s+|an\s+)?([a-z ]+?)(?:\s*$|\s*\.|\,)",
            t_dest,
        ):
            for stem in _stems_in_chunk(m.group(1)):
                if stem in _RECEP_STEMS and stem not in destinations:
                    destinations.append(stem)

    # "shelves" as source only when phrased as origin (from / find on shelves / ...)
    if re.search(r"(?:from|off|find|locate).{0,24}(?:shelf|shelves|bookshelf)", t):
        if "shelf" not in sources:
            sources.append("shelf")
    # Ambiguous shelf mention with desk dest → shelf is source
    if "desk" in destinations and ("shelf" in t or "shelves" in t) and "shelf" not in sources:
        sources.append("shelf")

    # "put it back on the X" / "from the X ... on the X" → same stem is both source & dest
    put_back = bool(
        re.search(r"put\s+(?:it|them|that)?\s*back\b", t)
        or re.search(r"put\s+.+\s+back\s+on\b", t)
    )
    if put_back:
        # Prefer the receptacle mentioned after "back on/in"
        m = re.search(r"back\s+(?:on|in|onto|into)\s+(?:the\s+|a\s+)?([a-z ]+)", t)
        if m:
            for stem in _stems_in_chunk(m.group(1)):
                if stem in _RECEP_STEMS:
                    if stem not in destinations:
                        destinations.append(stem)
                    if stem not in sources:
                        sources.append(stem)
        # Also from "from the X"
        m = re.search(r"from\s+(?:the\s+|a\s+)?([a-z ]+)", t)
        if m:
            for stem in _stems_in_chunk(m.group(1)):
                if stem in _RECEP_STEMS:
                    if stem not in sources:
                        sources.append(stem)
                    if stem not in destinations:
                        destinations.append(stem)

    # Never keep the same stem as both source and destination unless put-back / explicit from-to same
    if not put_back:
        for stem in list(destinations):
            if stem in sources and not re.search(
                rf"from\s+(?:the\s+)?{re.escape(stem)}.{{0,20}}to\s+(?:the\s+)?{re.escape(stem)}",
                t,
            ):
                if re.search(
                    rf"(?:put|place|move).{{0,40}}(?:on|in|onto|into|to)\s+(?:the\s+|a\s+)?{re.escape(stem)}",
                    t,
                ):
                    sources = [s for s in sources if s != stem]

    # Drop landmark nouns that are only relative references ("next to the lettuce").
    objects = [
        o for o in objects
        if not re.search(
            rf"(?:next to|beside|near|by)\s+(?:the\s+|a\s+|an\s+)?{re.escape(o)}\b",
            t,
        )
    ]

    kind = _infer_alfworld_kind(t)
    stations = _infer_process_stations(t, kind)
    # Process appliances are not put destinations unless the put/place clause
    # explicitly names them (e.g. "put the egg in the microwave").
    appliance_stems = {
        "microwave", "stove", "stoveburner", "fridge", "refrigerator",
        "sink", "sinkbasin", "desklamp", "lamp",
    }
    explicit_put_appliances: set[str] = set()
    for m in re.finditer(
        r"(?:to\s+)?(?:put|place|move|drop)\s+.{0,48}?(?:on|in|onto|into|to|inside)\s+"
        r"(?:the\s+|a\s+|an\s+)?([a-z ]+?)(?:\s*$|\s*\.|\,)",
        t,
    ):
        for stem in _stems_in_chunk(m.group(1)):
            if stem in appliance_stems:
                explicit_put_appliances.add(stem)
    strip_appliances = set(appliance_stems) - explicit_put_appliances
    if kind in _PROCESS_STATIONS:
        strip_appliances |= set(stations)
    if strip_appliances:
        destinations = [d for d in destinations if d not in strip_appliances]

    return {
        "objects": _uniq(objects),
        "sources": _uniq(sources),
        "destinations": _uniq(destinations),
        "put_back": put_back,
        "kind": kind,
        "stations": _uniq(list(stations)),
    }


def alf_goal_has_constraints(task: str) -> bool:
    """Return whether ``task`` contains a parsed AlfWorld goal.

    Experiment runners sometimes expose only a task-type id (for example
    ``pick_and_place_simple``), while the natural-language instruction is
    available only through the environment. Safety gates must not interpret
    an unparsed task id as evidence that every navigation action is stale.
    """
    goal = parse_alfworld_goal(task or "")
    return bool(
        goal.get("objects")
        or goal.get("sources")
        or goal.get("destinations")
        or goal.get("stations")
        or goal.get("kind")
        or goal.get("put_back")
    )


# Receptacles that are rarely true goal sources; CL must not force takes from
# them unless the instruction names them as an origin.
_ALF_TAKE_DISTRACTOR_SOURCES = frozenset({"toilet", "garbagecan"})


def _object_stem_from_household_action(act: str) -> str:
    """Portable object type stem from take/pick/process commands."""
    a = (act or "").strip().lower()
    m = re.match(r"^(?:take|pick up|put|heat|cool|clean)\s+([a-z]+)", a)
    return m.group(1) if m else ""


def _take_source_stem(act: str) -> str:
    a = (act or "").strip().lower()
    m = re.search(r"\bfrom\s+(?:the\s+)?([a-z]+)", a)
    return m.group(1) if m else ""


def _stem_explicit_in_task(stem: str, task: str) -> bool:
    """True when the goal stem (or a known alias) appears in the instruction."""
    s = (stem or "").strip().lower()
    t = (task or "").strip().lower()
    if not s or not t:
        return False
    compact = t.replace(" ", "")
    if re.search(rf"(?<![a-z]){re.escape(s)}(?![a-z])", compact):
        return True
    if re.search(rf"\b{re.escape(s)}\b", t):
        return True
    for phrase, alias in _GOAL_OBJECT_ALIASES.items():
        if alias == s and re.search(rf"\b{re.escape(phrase)}\b", t):
            return True
    return False


def _take_object_resolves_goal(action: str, obj_stems: list[str], task: str) -> bool:
    """
    Stricter take alignment than substring-in-compact matching.

    Rejects CL takes like ``soapbottle`` when the task only says generic
    ``bottle`` and multiple *bottle compound stems were suffix-inferred.
    """
    act = (action or "").strip().lower()
    if not act.startswith(("take ", "pick up ")):
        return True
    if not obj_stems:
        return True
    obj_stem = _object_stem_from_household_action(act)
    compact = act.replace(" ", "")
    if obj_stem and obj_stem not in obj_stems:
        if not any(s and s in compact for s in obj_stems):
            return False
        obj_stem = next(s for s in obj_stems if s and s in compact)
    elif not obj_stem:
        hits = [s for s in obj_stems if s and s in compact]
        if not hits:
            return False
        obj_stem = hits[0]

    explicit = [s for s in obj_stems if _stem_explicit_in_task(s, task)]
    if explicit:
        return obj_stem in explicit

    task_l = (task or "").lower()
    for tok in re.findall(r"[a-z]+", task_l):
        if len(tok) < 4:
            continue
        suffix_stems = [
            s for s in obj_stems
            if s.endswith(tok) and s != tok
        ]
        if obj_stem in suffix_stems and len(suffix_stems) > 1:
            prefix = obj_stem[: -len(tok)] if obj_stem.endswith(tok) else ""
            if prefix and prefix not in task_l.replace(" ", ""):
                return False
    return True


def _take_source_allowed(action: str, goal: dict, task: str) -> bool:
    """Reject forced takes from fixture distractors not named as goal sources."""
    act = (action or "").strip().lower()
    if not act.startswith(("take ", "pick up ")):
        return True
    src = _take_source_stem(act)
    if not src or src not in _ALF_TAKE_DISTRACTOR_SOURCES:
        return True
    sources = list(goal.get("sources") or [])
    if src in sources:
        return True
    return _stem_explicit_in_task(src, task)


def action_matches_alfworld_goal(action: str, task: str) -> bool:
    """
    True when a take/put/process command aligns with parsed goal stems.

    Used by CL pattern_force so stored ``take`` / ``put`` templates cannot
    ground onto visible distractor objects when goal objects are known.
    """
    act = (action or "").strip().lower()
    if not act:
        return False
    goal = parse_alfworld_goal(task or "")
    kind = (goal.get("kind") or "").strip().lower()
    obj_stems = list(goal.get("objects") or [])
    dest_stems = list(goal.get("destinations") or [])
    station_stems = list(goal.get("stations") or [])
    compact = act.replace(" ", "")
    if kind == "look":
        if act.startswith(("put ", "move ")) and " to " in act:
            return False
        if act.startswith(("take ", "pick up ")):
            return not obj_stems or any(s and s in compact for s in obj_stems)
        if act.startswith("use "):
            if station_stems and any(s and s in compact for s in station_stems):
                return True
            return not bool(obj_stems)
        if cmd_verb_family(act) in ("go", "open"):
            return bool(station_stems) and any(s and s in compact for s in station_stems)
        return cmd_verb_family(act) not in ("put",)
    if act.startswith(("take ", "pick up ")):
        if not _take_source_allowed(act, goal, task):
            return False
        if not obj_stems:
            return True
        hits = [s for s in obj_stems if s and s in compact]
        if not hits:
            return False
        task_l = (task or "").lower()
        if len(obj_stems) > 1 and "soap" in task_l:
            if re.search(r"\b(?:soap\s*bar|bar\s+of\s+soap|soapbar)\b", task_l):
                return "soapbar" in compact
            if re.search(r"\b(?:soap\s*bottle|hand\s+soap|soapbottle)\b", task_l):
                return "soapbottle" in compact
        if not _take_object_resolves_goal(act, obj_stems, task):
            return False
        return bool(hits)
    if act.startswith(("put ", "move ")) and " to " in act:
        if dest_stems and not any(s and s in compact for s in dest_stems):
            return False
        if obj_stems and not any(s and s in compact for s in obj_stems):
            return False
        return True
    if act.startswith(("heat ", "cool ", "clean ")):
        if obj_stems and not any(s and s in compact for s in obj_stems):
            return False
        return True
    if act.startswith("use "):
        if station_stems and any(s and s in compact for s in station_stems):
            return True
        if obj_stems and any(s and s in compact for s in obj_stems):
            return True
        return not bool(obj_stems)
    return True


def alf_manipulation_would_thrash(
    action: str,
    task: str,
    past_actions: list[str] | None = None,
) -> bool:
    """Reject puts/moves that undo progress or violate look / pick_two intent."""
    act = (action or "").strip().lower()
    if not act.startswith(("put ", "move ")) or " to " not in act:
        return False
    goal = parse_alfworld_goal(task or "")
    kind = (goal.get("kind") or "").strip().lower()
    if kind == "look":
        return True
    put_back = bool(goal.get("put_back"))
    destinations = goal.get("destinations") or []
    if kind == "pick_two" and destinations:
        if not any(_matches_dest_lock(act, d) for d in destinations):
            return True
    if not put_back and past_actions:
        last_src = _source_of_last_take(past_actions)
        if last_src:
            dest_m = re.search(r"\bto\s+(.+)$", _norm(act))
            dest_phrase = dest_m.group(1).strip() if dest_m else ""
            if dest_phrase and (
                dest_phrase == last_src
                or re.search(rf"\b{re.escape(last_src)}\b", _norm(dest_phrase))
            ):
                return True
    return False


def alf_take_would_undo_recent_put(action: str, past_actions) -> bool:
    """
    True when a pattern-suggested take would immediately undo a recent put/move.

    Prevents CL loops like put soapbottle → take soapbottle from same receptacle.
    Covers both ``move X to Y`` and ALFWorld ``put X in/on Y`` deposits.
    """
    act = (action or "").strip().lower()
    if not act.startswith(("take ", "pick up ")):
        return False
    m = re.match(r"^(?:take|pick up)\s+([a-z]+\s+\d+)", act)
    if not m:
        return False
    obj_id = m.group(1).replace(" ", "")
    # Optional source of the take (``from coffeemachine 1``).
    take_src = ""
    src_m = re.search(r"\bfrom\s+(.+)$", act)
    if src_m:
        take_src = src_m.group(1).strip().replace(" ", "")
    for past in reversed(list(past_actions or [])[-8:]):
        pl = (past or "").strip().lower()
        compact = pl.replace(" ", "")
        if obj_id not in compact:
            continue
        if pl.startswith("move ") and " to " in pl:
            return True
        if pl.startswith("put "):
            # Same object deposited; if take names a source, require that source
            # matches the put destination (in/on …).
            if not take_src:
                return True
            dest_m = re.search(r"\b(?:in/on|in|on)\s+(.+)$", pl)
            if dest_m and dest_m.group(1).strip().replace(" ", "") == take_src:
                return True
            if take_src and take_src in compact:
                return True
    return False


def alf_pattern_reuse_ok(action: str, task: str, past_actions=None) -> bool:
    """Combined AlfWorld CL reuse gate (goal alignment + no undo loops)."""
    act = (action or "").strip().lower()
    if not act:
        return False
    if alf_manipulation_would_thrash(act, task, past_actions):
        return False
    if alf_take_would_undo_recent_put(act, past_actions):
        return False
    # Process-family polarity: heat/cool/clean from a sibling episode must not
    # land on pick/put/look tasks (CL-on negative transfer).
    try:
        goal = parse_alfworld_goal(task or "")
        constrained = alf_goal_has_constraints(task)
        kind = (goal.get("kind") or "").strip().lower()
        fam = cmd_verb_family(act)
        if constrained and fam in ("heat", "cool", "clean") and kind not in _PROCESS_STATIONS:
            return False
        if constrained and fam in ("heat", "cool", "clean") and kind in _PROCESS_STATIONS:
            want = _PROCESS_STATIONS[kind][0]
            if want and fam != want:
                return False
        objects = list(goal.get("objects") or [])
        if constrained:
            holding = _holding_goal_for_process(
                [], list(past_actions or []), objects,
            )
            # Heat/clean/use/put and station/dest navigation need the object in hand.
            # Source search (cabinet/drawer) may run before take.
            if fam in ("heat", "cool", "clean", "use", "put") and not holding:
                return False
            if fam in ("go", "open"):
                if not holding and not alf_search_nav_ok(act, task):
                    return False
    except Exception:
        pass
    return bool(action_matches_alfworld_goal(act, task))


def alf_on_task_navigation(planned: str, task: str) -> bool:
    """True when go/open names a goal object, source, destination, or station."""
    act = _norm(planned)
    fam = cmd_verb_family(act)
    if fam not in ("go", "open"):
        return False
    goal = parse_alfworld_goal(task or "")
    kind = (goal.get("kind") or "").strip().lower()
    stems: list[str] = []
    for key in ("objects", "sources", "destinations", "stations"):
        stems.extend(list(goal.get(key) or []))
    if kind in _PROCESS_STATIONS:
        stems.extend(list(_PROCESS_STATIONS[kind][1] or []))
    if any(s and _action_mentions_stem(act, s) for s in stems if s):
        return True
    # Compound stations: ``go to desk`` is on-task for a ``desklamp`` station.
    rest = ""
    if act.startswith("go to "):
        rest = act[len("go to "):]
    elif act.startswith("open "):
        rest = act[len("open "):]
    dest_stem = re.sub(r"\d+$", "", _norm(rest).replace(" ", ""))
    if dest_stem and len(dest_stem) >= 4:
        for s in stems:
            st = (s or "").replace(" ", "")
            if not st:
                continue
            if dest_stem in st or st in dest_stem:
                return True
    return False


def alf_search_nav_ok(action: str, task: str) -> bool:
    """go/open toward a source or object hideout — not the put-destination or process station.

    CL-on may reuse this before the object is in hand. Deposit / heat-station
    navigation must wait until holding so search is not skipped.
    """
    act = _norm(action)
    fam = cmd_verb_family(act)
    if fam not in ("go", "open"):
        return False
    if not alf_on_task_navigation(act, task):
        return False
    goal = parse_alfworld_goal(task or "")
    kind = (goal.get("kind") or "").strip().lower()
    sources = list(goal.get("sources") or []) + list(goal.get("objects") or [])
    dests = list(goal.get("destinations") or [])
    stations = list(goal.get("stations") or [])
    if kind in _PROCESS_STATIONS:
        stations.extend(list(_PROCESS_STATIONS[kind][1] or []))
    mentions_src = any(s and _action_mentions_stem(act, s) for s in sources if s)
    mentions_dest = any(d and _action_mentions_stem(act, d) for d in dests if d)
    mentions_station = any(s and _action_mentions_stem(act, s) for s in stations if s)
    if mentions_src and not (mentions_dest or mentions_station):
        return True
    if mentions_src:
        return True
    return False


def alf_planned_is_weak(planned: str, task: str = "") -> bool:
    """True when planner output is look/examine/empty or off-task wander.

    On-task go/open (to a goal receptacle or process station) is *not* weak, so
    CL-on must not override it with a generic take/use from a prior episode.
    """
    act = (planned or "").strip().lower()
    fam = cmd_verb_family(act)
    if not act or fam in ("look", "examine", "other", ""):
        return True
    if fam in ("go", "open"):
        return not (task and alf_on_task_navigation(act, task))
    return False


def _infer_process_stations(task: str, kind: str) -> list[str]:
    """Stations named in the instruction or implied by the process family."""
    t = _norm(task)
    spec = _PROCESS_STATIONS.get((kind or "").strip().lower())
    out: list[str] = []
    if spec:
        out.extend(spec[1])
    allowed = set(out)
    for phrase, stem in _GOAL_OBJECT_ALIASES.items():
        if stem in allowed:
            allowed.add(phrase.replace(" ", ""))
    # Only "use/with/turn on" name a station. "in the cabinet" is a put dest, not a station.
    for m in re.finditer(
        r"(?:turn(?:ing)? on|toggle|use|with)\s+(?:the\s+|a\s+|an\s+)?([a-z ]+?)(?:\s*$|\.|\,| and )",
        t,
    ):
        chunk = _norm(m.group(1))
        compact = chunk.replace(" ", "")
        for stem in list(allowed):
            if stem and (stem in compact or re.search(rf"\b{re.escape(stem)}\b", chunk)):
                if stem in _RECEP_STEMS and stem not in out:
                    out.append(stem)
        for phrase, stem in _GOAL_OBJECT_ALIASES.items():
            if phrase in chunk and stem in allowed and stem in _RECEP_STEMS and stem not in out:
                out.append(stem)
    out = [s for s in out if not any(s != o and s in o for o in out)]
    return out


def _process_verb_satisfied(
    past: list[str],
    kind: str,
    stations: list[str] | None = None,
) -> bool:
    """True once the required process verb ran (incl. use-on-appliance for heat/cool)."""
    spec = _PROCESS_STATIONS.get((kind or "").strip().lower())
    if not spec:
        return False
    verb, default_stations = spec
    st = list(stations or default_stations)
    for act in past or []:
        al = _norm(act)
        if al.startswith("<"):
            continue
        fam = cmd_verb_family(al)
        if fam == verb:
            return True
        if kind == "heat" and fam == "use" and any(_action_mentions_stem(al, s) for s in st):
            return True
        if kind == "cool" and fam == "use" and any(
            _action_mentions_stem(al, s) for s in st + ["fridge", "refrigerator"]
        ):
            return True
        if kind == "look" and fam == "use" and any(_action_mentions_stem(al, s) for s in st):
            return True
    return False


def _process_verb_commands(
    kind: str,
    *,
    heats: list[str],
    cools: list[str],
    cleans: list[str],
    uses: list[str],
    stations: list[str],
) -> list[str]:
    """Admissible commands that complete a process step (heat/cool/clean/look)."""
    spec = _PROCESS_STATIONS.get((kind or "").strip().lower())
    if not spec:
        return []
    verb, _ = spec
    base = {
        "heat": heats,
        "cool": cools,
        "clean": cleans,
        "use": uses,
    }.get(verb, [])
    if kind in ("heat", "cool", "look") and stations:
        via_use = [
            u for u in uses
            if any(_action_mentions_stem(u, s) for s in stations)
        ]
        out: list[str] = []
        for a in list(base) + via_use:
            if a not in out:
                out.append(a)
        return out
    return list(base)


def alfworld_ground_planned_keep(
    planned: str,
    fast: str,
    env_valid: set | list,
    task: str,
    action_history=None,
) -> str | None:
    """
    Prefer a grounded planner process/manip action over generic explore.

    Returns the admissible command to keep, or None to let fast path win.
    Grounds SciWorld-style ``use microwave on apple`` onto AlfWorld
    ``heat apple with microwave`` when that form is admissible.
    """
    p = _norm(planned)
    f = _norm(fast)
    if not p:
        return None
    admissible = [str(a) for a in (env_valid or [])]
    if not admissible:
        return None
    grounded = match_admissible_alfworld(p, admissible, task=task)
    if not grounded:
        return None
    g = _norm(grounded)
    goal = parse_alfworld_goal(task)
    kind = (goal.get("kind") or "pick").strip().lower()
    objects = list(goal.get("objects") or [])
    gf = cmd_verb_family(g)
    ff = cmd_verb_family(f)
    if gf in _LOW_VALUE:
        return None
    stations = list(goal.get("stations") or [])
    if kind in _PROCESS_STATIONS:
        for s in _PROCESS_STATIONS[kind][1]:
            if s not in stations:
                stations.append(s)
    past = list(action_history or [])
    holding = _holding_goal_for_process(admissible, past, objects, inventory="")
    # Never keep put before the required process verb (heat/cool/clean/use).
    if kind in _PROCESS_STATIONS and gf == "put":
        if not _process_verb_satisfied(past, kind, stations):
            return None
    # After the required process verb, do not keep a *different* process verb
    # (e.g. cool after heat) — that thrashing blocked put-to-dest in overnight logs.
    if kind in _PROCESS_STATIONS and gf in ("heat", "cool", "clean", "use"):
        want = _PROCESS_STATIONS[kind][0]
        if _process_verb_satisfied(past, kind, stations):
            if gf != want and not (kind == "look" and gf == "use"):
                return None
    # Process verb only after the goal object is in hand.
    if kind in _PROCESS_STATIONS and gf in ("use", "heat", "cool", "clean") and not holding:
        return None
    # Do not keep takes of non-goal objects when goal stems are known.
    if gf == "take" and objects and not any(_action_mentions_stem(g, o) for o in objects):
        return None
    process_like = gf in ("take", "put", "heat", "cool", "clean", "use", "open")
    if gf == "use" and stations and not any(_action_mentions_stem(g, s) for s in stations):
        if kind not in ("look",):
            process_like = False
    if not process_like:
        return None
    if not f:
        return grounded
    if ff == "go" and process_like:
        return grounded
    if kind in _PROCESS_STATIONS and gf in ("take", "heat", "cool", "clean", "use"):
        return grounded
    return None


def alfworld_keep_planned_over_fast(
    planned: str,
    fast: str,
    env_valid: set | list,
    task: str,
    action_history=None,
) -> bool:
    """Prefer a valid planner manipulation over generic explore hideout."""
    return alfworld_ground_planned_keep(
        planned, fast, env_valid, task, action_history=action_history,
    ) is not None


def _infer_alfworld_kind(task: str) -> str:
    """Classify AlfWorld goal family from instruction semantics, not task ids."""
    t = _norm(task)
    # Past-participle / result phrasing: cooked/heated/microwaved/warmed-up.
    # Do NOT match the noun "microwave" (e.g. "cabinet left of the microwave").
    if re.search(
        r"\b(?:hot|heat(?:ed)?|heat up|cooked|cook|warm(?:ed)?(?:\s+up)?|microwaved)\b",
        t,
    ):
        return "heat"
    if re.search(r"\b(?:cool(?:ed)?|chill(?:ed)?)\b", t):
        return "cool"
    if re.search(r"\b(?:clean(?:ed)?|wash(?:ed)?|rinsed)\b", t) or re.search(
        r"\bfilled\b.{0,48}\bwater\b|\bwater\b.{0,24}\b(?:bowl|cup|mug|sink)\b",
        t,
    ):
        return "clean"
    # Hold/examine an object while activating a light — not pick-and-place.
    if re.search(
        r"(?:look at|examine).{0,80}(?:lamp|light)"
        r"|under the (?:desk\s*)?(?:lamp|light)"
        r"|by the light"
        r"|(?:take|bring|carry|get|pick(?:\s+up)?).{0,64}(?:to|under|near|by)\s+"
        r"(?:the\s+|a\s+|an\s+)?(?:desk\s*)?(?:lamp|light)"
        r"|(?:hold|take|pick(?:ing)?).{0,80}(?:turn(?:ing)? on|toggle|use).{0,40}(?:lamp|light)"
        r"|(?:turn(?:ing)? on|toggle|use)\s+(?:the\s+|a\s+)?(?:desk\s*)?(?:lamp|light)"
        r"|desklamp",
        t,
    ):
        return "look"
    if re.search(r"\b(?:two|another)\b", t):
        return "pick_two"
    return "pick"


def cmd_verb_family(cmd: str) -> str:
    c = _norm(cmd)
    if c.startswith("take ") or c.startswith("pick up "):
        return "take"
    if c.startswith("put ") or c.startswith("move ") or c.startswith("place "):
        return "put"
    if c.startswith("go to ") or c.startswith("goto "):
        return "go"
    if c.startswith("open "):
        return "open"
    if c.startswith("close "):
        return "close"
    if c.startswith("heat "):
        return "heat"
    if c.startswith("cool "):
        return "cool"
    if c.startswith("clean "):
        return "clean"
    if c.startswith("use ") or c.startswith("toggle "):
        return "use"
    if c == "look" or c.startswith("look "):
        return "look"
    if c == "inventory" or c.startswith("inventory"):
        return "inventory"
    if c.startswith("examine "):
        return "examine"
    if c in ("help",) or c.startswith("help "):
        return "meta"
    return "other"


def _pred_verb_family(pred: str) -> str:
    p = _norm(pred)
    for v in _TAKE_VERBS:
        if p.startswith(v + " ") or p == v:
            return "take"
    for v in _PUT_VERBS:
        if p.startswith(v + " ") or p == v:
            return "put"
    for v in _GO_VERBS:
        if p.startswith(v + " ") or p == v:
            return "go"
    for v in _OPEN_VERBS:
        if p.startswith(v + " ") or p == v:
            return "open"
    for v in _CLOSE_VERBS:
        if p.startswith(v + " ") or p == v:
            return "close"
    if p.startswith("heat "):
        return "heat"
    if p.startswith("cool "):
        return "cool"
    if p.startswith("clean "):
        return "clean"
    if p.startswith("use ") or p.startswith("toggle "):
        return "use"
    if p == "look" or p.startswith("look "):
        return "look"
    if p.startswith("examine "):
        return "examine"
    if p in ("help",) or p.startswith("help "):
        return "meta"
    return "other"


# Appliance stems → AlfWorld process verb (LLM often says "use microwave on apple").
_USE_APPLIANCE_TO_PROCESS = (
    (("microwave", "stove", "stoveburner"), "heat"),
    (("fridge", "refrigerator"), "cool"),
    (("sinkbasin", "sink"), "clean"),
)


def normalize_alfworld_prediction(pred: str) -> str:
    """Rewrite common LLM phrasings toward AlfWorld command surface."""
    p = _norm(pred)
    p = re.sub(r"^(please\s+|then\s+|next\s+)", "", p)
    # pick up X → take X
    p = re.sub(r"^pick\s+up\s+", "take ", p)
    p = re.sub(r"^pickup\s+", "take ", p)
    p = re.sub(r"^get\s+", "take ", p)
    p = re.sub(r"^grab\s+", "take ", p)
    # place/put/set X on/in/onto Y → move X to Y (AlfWorld often uses move)
    m = re.match(
        r"^(?:put|place|set|drop)\s+(.+?)\s+(?:on|in|onto|into|over)\s+(?:the\s+|a\s+)?(.+)$",
        p,
    )
    if m:
        return f"move {m.group(1).strip()} to {m.group(2).strip()}"
    m = re.match(r"^move\s+(.+?)\s+(?:on|in|onto|into)\s+(?:the\s+|a\s+)?(.+)$", p)
    if m:
        return f"move {m.group(1).strip()} to {m.group(2).strip()}"
    # use <appliance> on <object> → heat/cool/clean <object> with <appliance>
    m = re.match(r"^use\s+(.+?)\s+on\s+(.+)$", p)
    if m:
        appliance, obj = m.group(1).strip(), m.group(2).strip()
        app_c = appliance.replace(" ", "")
        for stems, verb in _USE_APPLIANCE_TO_PROCESS:
            if any(s in app_c for s in stems):
                return f"{verb} {obj} with {appliance}"
    # go/walk/navigate
    p = re.sub(r"^(?:walk|navigate|head)\s+to\s+", "go to ", p)
    p = re.sub(r"^goto\s+", "go to ", p)
    # look around → look
    if p in ("look around", "look around.", "lookaround"):
        return "look"
    # ScienceWorld-style FOCUS → AlfWorld examine (never put / go-hideout)
    if p.startswith("focus on "):
        return "examine " + p[len("focus on "):].strip()
    if p.startswith("toggle "):
        return "use " + p[len("toggle "):].strip()
    return p


def _action_mentions_stem(action: str, stem: str) -> bool:
    a = _norm(action).replace(" ", "")
    return stem in a


def _filter_by_stems(actions: Iterable[str], stems: list[str]) -> list[str]:
    if not stems:
        return list(actions)
    out = []
    for a in actions:
        if any(_action_mentions_stem(a, s) for s in stems):
            out.append(a)
    return out


def _unmatched_take_object(past_actions: list[str], objects: list[str]) -> str | None:
    """Last take that has not been followed by a put — the agent still holds it.

    When ``objects`` (goal stems) is non-empty, only a take that mentions a goal
    stem counts as holding. Unmatched non-goal takes must not be treated as
    holding — that caused wrong-object ``use``/``heat`` on process tasks.
    """
    for act in reversed(past_actions or []):
        al = _norm(act)
        if al.startswith("<"):
            continue
        fam = cmd_verb_family(al)
        if fam == "put":
            return None
        if fam == "take":
            for obj in objects:
                if obj and _action_mentions_stem(al, obj):
                    return obj
            if objects:
                return None
            m = re.match(r"take\s+(.+?)\s+from\s+", al)
            if m:
                return _stemify(m.group(1))
            return None
    return None


def _carrying_goal_object(
    admissible: list[str],
    past_actions: list[str],
    objects: list[str],
    inventory: str = "",
) -> str | None:
    """Return object stem if agent appears to be holding a goal object."""
    inv = _norm(inventory)
    inv_empty = (not inv) or ("nothing" in inv)
    taken = _unmatched_take_object(past_actions, objects)

    if inv and not inv_empty:
        compact = inv.replace(" ", "")
        for obj in objects:
            if obj and (obj in compact or re.search(rf"\b{re.escape(obj)}\b", inv)):
                return obj
        if taken:
            return taken

    # Stale "You are carrying: nothing" must not erase a take that has not been put.
    if taken:
        return taken

    if inv_empty:
        for a in admissible:
            if cmd_verb_family(a) != "put":
                continue
            for obj in objects:
                if not (obj and _action_mentions_stem(a, obj)):
                    continue
                # If a take for the same object is still admissible, we are not holding it.
                still_takeable = any(
                    cmd_verb_family(t) == "take" and _action_mentions_stem(t, obj)
                    for t in admissible
                )
                if still_takeable:
                    continue
                return obj
    return None


def _source_of_last_take(past_actions: list[str]) -> str | None:
    for act in reversed(past_actions or []):
        al = _norm(act)
        m = re.match(r"take\s+.+\s+from\s+(.+)$", al)
        if m:
            return m.group(1).strip()
    return None


def _is_meta_cmd(cmd: str) -> bool:
    c = _norm(cmd)
    if c in _META_CMDS or c.startswith("help"):
        return True
    return cmd_verb_family(c) == "meta"


def match_admissible_alfworld(
    pred: str,
    admissible: list[str],
    *,
    task: str = "",
    recent: set[str] | None = None,
) -> str | None:
    """Verb-family + entity constrained match (SciWorld-style grounding)."""
    if not admissible:
        return None
    recent = recent or set()
    goal = parse_alfworld_goal(task)
    p = normalize_alfworld_prediction(pred)
    fam = _pred_verb_family(p)
    adm_map = {_norm(a): a for a in admissible}

    if p in adm_map and p not in recent and not _is_meta_cmd(p):
        return adm_map[p]

    # Never ground a contentful prediction onto help/look/inventory.
    contentful = fam in (
        "take", "put", "go", "open", "close", "heat", "cool", "clean", "use", "examine",
    )
    family_cmds = [a for a in admissible if cmd_verb_family(a) == fam] if fam != "other" else []
    if contentful:
        family_cmds = [a for a in family_cmds if not _is_meta_cmd(a)]
        if not family_cmds:
            return None
    elif not family_cmds:
        family_cmds = [a for a in admissible if not _is_meta_cmd(a)]

    # Prefer commands sharing object/recep tokens from prediction + goal
    pred_tokens = set(re.findall(r"[a-z]+", p))
    goal_stems = set(goal["objects"] + goal["sources"] + goal["destinations"] + goal.get("stations", []))

    def _score(cmd: str) -> tuple:
        cl = _norm(cmd)
        c_compact = cl.replace(" ", "")
        overlap_pred = sum(1 for t in pred_tokens if len(t) > 2 and t in c_compact)
        overlap_goal = sum(1 for s in goal_stems if s and s in c_compact)
        obj_hit = 0
        for obj in goal["objects"]:
            if obj in c_compact:
                obj_hit += 3
        dest_hit = 0
        if fam == "put":
            for d in goal["destinations"]:
                if d in c_compact:
                    dest_hit += 4
            for s in goal["sources"]:
                if s in c_compact:
                    dest_hit -= 2
        elif fam == "take":
            for s in goal["sources"]:
                if s in c_compact:
                    dest_hit += 2
        elif fam == "use":
            for s in goal.get("stations") or []:
                if s in c_compact:
                    dest_hit += 5
            for tok in pred_tokens:
                if len(tok) > 3 and tok in c_compact:
                    dest_hit += 2
        recent_pen = 5 if cl in recent else 0
        low_pen = 8 if _is_meta_cmd(cl) or cmd_verb_family(cl) in _LOW_VALUE else 0
        return (obj_hit + dest_hit + overlap_goal + overlap_pred - recent_pen - low_pen, -len(cl))

    candidates = family_cmds or [a for a in admissible if not _is_meta_cmd(a)]
    ranked = sorted(candidates, key=_score, reverse=True)
    if ranked:
        best = ranked[0]
        best_s = _score(best)
        if fam in ("take", "put", "go", "open", "close", "use", "heat", "cool", "clean") and (
            cmd_verb_family(best) != fam or _is_meta_cmd(best)
        ):
            same = [a for a in admissible if cmd_verb_family(a) == fam and not _is_meta_cmd(a)]
            if same:
                best = sorted(same, key=_score, reverse=True)[0]
                best_s = _score(best)
            else:
                return None
        if fam in ("take", "put", "use") and best_s[0] < 1:
            for a in candidates:
                if (p in _norm(a) or _norm(a) in p) and not _is_meta_cmd(a):
                    if _norm(a) not in recent:
                        return a
            return None
        if _is_meta_cmd(best):
            return None
        if _norm(best) not in recent or all(_norm(a) in recent for a in candidates):
            return best
    return None


# Station fixtures that live on another receptacle (lamp is on a desk, not `go to desklamp`).
_STATION_HOST_STEMS = {
    "desklamp": ("desk", "sidetable", "dresser", "shelf", "coffeetable"),
    "lamp": ("desk", "sidetable", "dresser", "shelf", "coffeetable"),
    "light": ("desk", "sidetable", "dresser", "shelf", "coffeetable"),
    "sink": ("sinkbasin",),
    "sinkbasin": ("sinkbasin",),
    "microwave": ("microwave",),
    "stove": ("stoveburner",),
    "fridge": ("fridge",),
    "refrigerator": ("fridge",),
}


def _host_stems_for_stations(stations: list[str]) -> list[str]:
    out: list[str] = []
    for s in stations or []:
        hosts = _STATION_HOST_STEMS.get(s) or (s,)
        for h in hosts:
            if h and h not in out:
                out.append(h)
    return out


def _station_go_priority(kind: str, stations: list[str]) -> list[str]:
    """
    Ordered stems to walk toward while holding a process object.

    Prefer destinations that typically admit the process verb (microwave for heat)
    over side appliances that cause host thrashing (stoveburner).
    """
    kind_l = (kind or "").strip().lower()
    hosts = _host_stems_for_stations(stations)
    if kind_l == "heat":
        preferred = ["microwave"]
    elif kind_l == "cool":
        preferred = ["fridge", "refrigerator"]
    elif kind_l == "clean":
        preferred = ["sinkbasin", "sink"]
    else:
        preferred = list(hosts)
    out: list[str] = []
    for s in preferred:
        if s not in out:
            out.append(s)
    # Heat: never append stoveburner hosts — they do not admit ``heat … with``.
    demote_heat = {"stoveburner", "stove"} if kind_l == "heat" else set()
    for h in hosts:
        if h in demote_heat:
            continue
        if h not in out:
            out.append(h)
    for s in stations or []:
        if s in demote_heat:
            continue
        if s not in out:
            out.append(s)
    return out


def _pick_station_go(
    goes: list[str],
    *,
    kind: str,
    stations: list[str],
    memory_text: str,
    recent: set | None = None,
) -> str | None:
    """
    Next go toward a process station while holding.

    Appliance stations: prefer unvisited microwave/fridge/sink; never oscillate
    stoveburner when a microwave go is available.
    Lamp stations: may revisit remembered desk hosts.
    """
    if not goes:
        return None
    recent_n = {_norm(x) for x in (recent or set()) if x}
    priority = _station_go_priority(kind, stations)

    def _rank_goes(cands: list[str]) -> list[str]:
        def _key(g: str) -> tuple:
            gn = _norm(g)
            pri = 99
            for i, stem in enumerate(priority):
                if _action_mentions_stem(g, stem):
                    pri = i
                    break
            return (0 if gn not in recent_n else 1, pri, gn)

        return sorted(cands, key=_key)

    direct = _filter_by_stems(goes, priority) if priority else []
    if direct:
        ranked = _rank_goes(direct)
        if (kind or "").strip().lower() == "heat":
            micro = [g for g in ranked if _action_mentions_stem(g, "microwave")]
            if micro:
                # Prefer unvisited microwave; never fall through to stoveburner
                # just because microwave was already tried once this episode.
                fresh_m = [g for g in micro if _norm(g) not in recent_n]
                return (fresh_m or micro)[0]
            # Heat: if microwave goes exist anywhere in ``goes``, ignore stove
            # candidates that leaked into ``direct`` via loose stem filters.
            any_micro = [g for g in goes if _action_mentions_stem(g, "microwave")]
            if any_micro:
                ranked_m = _rank_goes(any_micro)
                fresh_m = [g for g in ranked_m if _norm(g) not in recent_n]
                return (fresh_m or ranked_m)[0]
            # No microwave go admissible: still refuse stoveburner thrash.
            ranked = [
                g for g in ranked
                if not _action_mentions_stem(g, "stoveburner")
                and not _action_mentions_stem(g, "stove")
            ]
            if not ranked:
                return None
        fresh = [g for g in ranked if _norm(g) not in recent_n] or ranked
        if fresh:
            return fresh[0]

    if _lamp_like_stations(stations):
        host_goes = _rank_station_host_goes(goes, stations, memory_text)
        return host_goes[0] if host_goes else None

    # Appliance process: only rank hosts that survive priority (no stoveburner for heat).
    host_goes = _rank_station_host_goes(goes, priority or stations, memory_text)
    if not host_goes:
        host_goes = _filter_by_stems(goes, priority) if priority else []
    if not host_goes:
        return None
    # Drop demoted heat hosts that may leak in via memory ranking.
    if (kind or "").strip().lower() == "heat":
        host_goes = [
            g for g in host_goes
            if not _action_mentions_stem(g, "stoveburner")
            and not _action_mentions_stem(g, "stove")
        ] or host_goes
    ranked = _rank_goes(host_goes)
    fresh = [g for g in ranked if _norm(g) not in recent_n] or ranked
    return fresh[0] if fresh else None


def _goes_to_station_host(
    goes: list[str],
    ctx_text: str,
    stations: list[str],
) -> list[str]:
    """Prefer `go to desk 1` when observation mentioned a lamp/station on that desk."""
    hosts = _host_stems_for_stations(stations)
    if not goes or not hosts:
        return []
    ctx_n = _norm(ctx_text)
    near: list[str] = []
    fallback = _filter_by_stems(goes, hosts)
    if not ctx_n:
        return fallback
    station_needles = [s for s in (stations or []) if s]
    for g in goes:
        m = re.search(r"go to (.+)$", _norm(g))
        if not m:
            continue
        loc = m.group(1).strip()
        if not loc or not any(_action_mentions_stem(g, h) for h in hosts):
            continue
        for hit in re.finditer(re.escape(loc), ctx_n):
            window = ctx_n[max(0, hit.start() - 96): hit.end() + 160]
            compact = window.replace(" ", "")
            if any(s in compact or s in window for s in station_needles):
                near.append(g)
                break
    return near or fallback


def _lamp_like_stations(stations: list[str]) -> bool:
    return any(s in {"desklamp", "lamp", "light"} for s in (stations or []))


def _extract_station_host_locations(text: str, stations: list[str]) -> list[str]:
    """
    Parse observation text for receptacle locations that co-locate a process station.

    Generic over lamp/microwave/sink/etc.: ``On the desk 1, you see a desklamp 1``.
    """
    raw = _norm(text)
    if not raw or not stations:
        return []
    needles = [s for s in stations if s]
    found: list[str] = []
    # "on the desk 1, you see ... desklamp"
    for m in re.finditer(
        r"\bon(?:to)?\s+(?:the\s+)?([a-z]+(?:\s+\d+)?)\b[^.]{0,160}",
        raw,
    ):
        window = m.group(0)
        compact = window.replace(" ", "")
        if not any(s in compact or s in window for s in needles):
            continue
        loc = m.group(1).strip()
        if loc and loc not in found:
            found.append(loc)
    # "you see a desklamp 1" near a prior go/arrive mention kept in memory blob
    for m in re.finditer(
        r"(?:arrive(?:d)? at|go to)\s+([a-z]+(?:\s+\d+)?)\b[^.]{0,200}",
        raw,
    ):
        window = m.group(0)
        compact = window.replace(" ", "")
        if not any(s in compact or s in window for s in needles):
            continue
        loc = m.group(1).strip()
        if loc and loc not in found:
            found.append(loc)
    return found


def _rank_station_host_goes(
    goes: list[str],
    stations: list[str],
    memory_text: str,
) -> list[str]:
    """
    Order host receptacles for process stations.

    Prefer locations where the station was actually seen (memory), then
    primary hosts (desk/sidetable for lamps; appliance itself otherwise).
    Demote shelf/garbage for lamps so holding agents do not wander.
    """
    if not goes:
        return []
    remembered_locs = _extract_station_host_locations(memory_text, stations)
    remembered_hit: list[str] = []
    for loc in remembered_locs:
        for g in goes:
            if _norm(g) == f"go to {loc}" or _norm(g).endswith(f" {loc}"):
                if g not in remembered_hit:
                    remembered_hit.append(g)
    # Text-window fallback (older memory blobs without explicit "on the X").
    if not remembered_hit:
        remembered = _goes_to_station_host(goes, memory_text, stations)
        mem_n = _norm(memory_text)
        if mem_n:
            station_needles = [s for s in (stations or []) if s]
            for g in remembered:
                m = re.search(r"go to (.+)$", _norm(g))
                if not m:
                    continue
                loc = m.group(1).strip()
                if not loc:
                    continue
                for hit in re.finditer(re.escape(loc), mem_n):
                    window = mem_n[max(0, hit.start() - 96): hit.end() + 160]
                    compact = window.replace(" ", "")
                    if any(s in compact or s in window for s in station_needles):
                        remembered_hit.append(g)
                        break
    hosts = _host_stems_for_stations(stations)
    all_hosts = _filter_by_stems(goes, hosts)
    primary = ("desk", "sidetable") if _lamp_like_stations(stations) else tuple(hosts[:2] or hosts)
    demote = ("shelf", "garbagecan", "sofa", "bed") if _lamp_like_stations(stations) else ()
    primary_goes = _filter_by_stems(all_hosts, list(primary))
    demoted = [g for g in all_hosts if any(_action_mentions_stem(g, d) for d in demote)]
    mid = [g for g in all_hosts if g not in primary_goes and g not in demoted]
    out: list[str] = []
    for bucket in (remembered_hit, primary_goes, mid):
        for g in bucket:
            if g not in out:
                out.append(g)
    # Only append demoted hosts when nothing better exists.
    if not out:
        for g in demoted:
            if g not in out:
                out.append(g)
    elif remembered_hit or primary_goes:
        pass  # deliberately omit demoted while a better host exists
    else:
        for g in demoted:
            if g not in out:
                out.append(g)
    return out


def _holding_goal_for_process(
    admissible: list[str],
    past: list[str],
    objects: list[str],
    inventory: str = "",
) -> str | None:
    """
    Holding state for process tasks (look/heat/cool/clean).

    When goal object stems are known, never treat an unmatched take of a
    *different* object as holding — that caused wrong-object ``use``/``heat``.
    """
    holding = _carrying_goal_object(admissible, past, objects, inventory=inventory)
    if holding:
        return holding
    if objects:
        return _unmatched_take_object(past, objects)
    # Goal stems unknown: any unmatched take is the best signal we have.
    return _unmatched_take_object(past, [])


def _takes_for_visible_goals(
    takes: list[str],
    objects: list[str],
    look_text: str = "",
) -> list[str]:
    """Prefer takes whose object is both a goal stem and mentioned in the current look."""
    if not takes:
        return []
    goals = [o for o in (objects or []) if o]
    if goals:
        goal_takes = _filter_by_stems(takes, goals)
    else:
        goal_takes = list(takes)
    look_n = _norm(look_text)
    if not look_n or not goal_takes:
        return goal_takes
    visible = [
        a for a in goal_takes
        if any(_action_mentions_stem(look_n, o) or o in look_n.replace(" ", "") for o in goals)
        or any(
            re.search(rf"\b{re.escape(tok)}\b", look_n)
            for tok in re.findall(r"[a-z]+", _norm(a))
            if len(tok) > 3 and tok not in {"take", "from", "the"}
        )
    ]
    return visible or goal_takes


def _alfworld_process_station(
    *,
    kind: str,
    objects: list[str],
    holding,
    past,
    recent: set,
    goes: list[str],
    opens: list[str],
    takes: list[str],
    uses: list[str],
    heats: list[str],
    cools: list[str],
    cleans: list[str],
    pick,
    logger=None,
    extra_stations: list[str] | None = None,
    ctx_text: str = "",
    memory_text: str = "",
) -> str | None:
    """Process-then-place: fetch object, go to station while holding, then verb. Never put."""
    spec = _PROCESS_STATIONS.get((kind or "").strip().lower())
    if not spec:
        return None
    verb, default_stations = spec
    stations = list(default_stations)
    for s in extra_stations or []:
        if not s or s in stations:
            continue
        # Ignore stove stems appended from loose goal parsing — they thrash heat.
        if kind == "heat" and s in ("stove", "stoveburner"):
            continue
        stations.append(s)
    if _process_verb_satisfied(past, kind, stations):
        return None
    verb_cmds = _process_verb_commands(
        kind,
        heats=heats,
        cools=cools,
        cleans=cleans,
        uses=uses,
        stations=stations,
    )
    prefer = list(objects) + list(stations)
    mem = (memory_text or ctx_text or "").strip()

    def _pick_fresh(cands: list[str], *, prefer_stems: list[str] | None = None) -> str | None:
        return pick(cands, prefer_stems=prefer_stems)

    # Execute the process verb only when the object is in hand (or already at station).
    if holding:
        ready = _filter_by_stems(verb_cmds, objects) if objects else list(verb_cmds)
        if objects:
            ready = ready or _filter_by_stems(verb_cmds, stations)
        # Any admissible process command while holding (stem mismatch / use-appliance).
        ready = ready or list(verb_cmds)
        if verb == "use":
            ready = _filter_by_stems(verb_cmds, stations) or list(verb_cmds)
        choice = _pick_fresh(ready, prefer_stems=prefer if verb != "use" else stations)
        if choice:
            if logger:
                logger.info("AlfWorld fast path (%s-now): %r", verb, choice)
            return choice
        open_st = _filter_by_stems(opens, stations + _host_stems_for_stations(stations))
        choice = _pick_fresh(open_st, prefer_stems=_station_go_priority(kind, stations))
        if choice:
            if logger:
                logger.info("AlfWorld fast path (open-%s): %r", kind, choice)
            return choice
        # Walk to the process appliance / remembered host. Respect recent so we
        # do not thrash stoveburner↔stoveburner while microwave is available.
        choice = _pick_station_go(
            goes, kind=kind, stations=stations, memory_text=mem, recent=recent,
        )
        if choice:
            if logger:
                logger.info("AlfWorld fast path (go-%s-host): %r", kind, choice)
            return choice
        return None

    # Look-at-light: if a lamp use is already admissible and we recently took a
    # goal object (inventory lag / stem mismatch), prefer use over host thrash.
    if kind == "look" and verb_cmds:
        recent_goal_take = False
        for act in reversed(past or []):
            al = _norm(act)
            fam = cmd_verb_family(al)
            if fam == "put":
                break
            if fam == "take" and (
                not objects or any(_action_mentions_stem(al, o) for o in objects)
            ):
                recent_goal_take = True
                break
        if recent_goal_take:
            lamp_uses = _filter_by_stems(verb_cmds, stations) or list(verb_cmds)
            choice = _pick_fresh(lamp_uses, prefer_stems=stations)
            if choice:
                if logger:
                    logger.info("AlfWorld fast path (use-after-take-look): %r", choice)
                return choice

    # Empty-handed: take the goal object (or walk to it / its host). Never put.
    # Prefer takes that are visible in the current look before leaving the receptacle.
    look_blob = f"{ctx_text or ''} {mem or ''}"
    goal_takes = _takes_for_visible_goals(takes, objects, look_blob)
    if not objects and kind in _PROCESS_STATIONS:
        goal_takes = []
    elif not objects:
        goal_takes = list(takes)
    choice = _pick_fresh(goal_takes, prefer_stems=objects)
    if choice:
        if logger:
            logger.info("AlfWorld fast path (take-before-%s): %r", kind, choice)
        return choice
    # Visible goal object in look but take somehow missing from filtered list:
    # still refuse host-search leave — fall through to go-object if possible.
    src_goes = _filter_by_stems(goes, objects) if objects else []
    choice = _pick_fresh(src_goes, prefer_stems=objects)
    if choice:
        if logger:
            logger.info("AlfWorld fast path (go-object-%s): %r", kind, choice)
        return choice
    # Empty-handed: only search co-location hosts for lamp-like stations
    # (object often sits on the same desk as the lamp). Do NOT walk to the
    # heat/cool/clean appliance before holding the goal object.
    # Never leave when a goal take is still admissible (even if pick skipped it).
    if objects and _filter_by_stems(takes, objects):
        return None
    if _lamp_like_stations(stations):
        # If look already shows a goal object here, do not wander to another host.
        look_n = _norm(look_blob)
        if look_n and objects and any(
            o in look_n.replace(" ", "") or re.search(rf"\b{re.escape(o)}\b", look_n)
            for o in objects
        ):
            return None
        host_goes = _rank_station_host_goes(goes, stations, mem)
        # Cap revisits so desk↔dresser cannot consume the episode.
        def _vc(a: str) -> int:
            an = _norm(a)
            return sum(1 for p in (past or []) if _norm(p) == an)
        fresh = [g for g in host_goes if _vc(g) < 2]
        if not fresh:
            fresh = [g for g in host_goes if _vc(g) < 3]
        choice = _pick_fresh(
            fresh, prefer_stems=_host_stems_for_stations(stations),
        ) if fresh else None
        if choice:
            if logger:
                logger.info("AlfWorld fast path (go-host-search-%s): %r", kind, choice)
            return choice
    return None


def alfworld_fast_substitute(
    env_valid: set | list,
    recent: set,
    ctx_text: str,
    task: str,
    action_history=None,
    *,
    look: str = "",
    inventory: str = "",
    logger=None,
    memory_text: str = "",
    blacklist: set | None = None,
    is_blocked=None,
    pattern_milestone: str = "",
    cl_read: bool = False,
) -> str | None:
    """
    Deterministic next action for AlfWorld (SciWorld architecture_fast analogue).

    Priority:
      1) put/move goal object onto destination (win)
      2) go to destination while carrying
      3) take goal object when visible
      4) open closed goal-related receptacles
      5) explore go-to source then destination receptacles
      6) avoid look loops

    When ``is_blocked`` / ``blacklist`` is set (CL-on failure memory), skip those
    candidates so Pattern BLACKLIST affects the dominant AlfWorld policy.
    CL-on ``pattern_milestone`` is tried first (grounded + goal-safe) so on/off
    diverge on the same Swift backbone; CL-off leaves it empty.
    """
    admissible = [str(a) for a in (env_valid or [])]
    if not admissible:
        return None
    recent = {_norm(x) for x in (recent or set()) if x}
    blocked = {_norm(x) for x in (blacklist or set()) if x}

    def _is_blocked(a: str) -> bool:
        an = _norm(a)
        # Never let failure memory suppress process / acquire verbs — that made
        # CL-on identical to CL-off when the only admissible take was blacklisted.
        fam = cmd_verb_family(an)
        if fam in ("take", "put", "heat", "cool", "clean", "use"):
            return False
        if blocked and an in blocked:
            return True
        if callable(is_blocked):
            try:
                return bool(is_blocked(an))
            except Exception:
                return False
        return False

    past = list(action_history or [])
    ms_l = (pattern_milestone or "").strip().lower()
    if cl_read and ms_l and not (ms_l == "wait" or re.match(r"^wait\d*$", ms_l)):
        try:
            from core.pattern_library import (
                ground_pattern_action as _ground_pattern_action,
                is_safe_task_pattern_action,
            )
            g = (_ground_pattern_action(
                ms_l, env_valid, task=task, past_actions=past,
                environment="alfworld",
            ) or "").strip().lower()
            if (
                g
                and is_safe_task_pattern_action(g, task=task)
                and not _is_blocked(g)
                and alf_pattern_reuse_ok(g, task, past)
            ):
                fam_g = cmd_verb_family(g)
                # Process/put/station-nav before the object is in hand skips hideout search.
                # Source/object receptacle nav is the CL-on search differentiator.
                if fam_g in ("use", "heat", "cool", "clean", "put"):
                    goal_early = parse_alfworld_goal(task)
                    holding_early = _holding_goal_for_process(
                        admissible,
                        past,
                        list(goal_early.get("objects") or []),
                        inventory=inventory,
                    )
                    if not holding_early:
                        g = ""
                elif fam_g in ("go", "open"):
                    goal_early = parse_alfworld_goal(task)
                    holding_early = _holding_goal_for_process(
                        admissible,
                        past,
                        list(goal_early.get("objects") or []),
                        inventory=inventory,
                    )
                    if (
                        alf_goal_has_constraints(task)
                        and not holding_early
                        and not alf_search_nav_ok(g, task)
                    ):
                        g = ""
                    # Visible goal take beats replaying source nav.
                    objs = list(goal_early.get("objects") or [])
                    if g and objs and any(
                        str(a).startswith("take ")
                        and _take_object_resolves_goal(str(a), objs, task)
                        for a in admissible
                    ):
                        g = ""
                if g:
                    if logger:
                        logger.info("AlfWorld fast path (pattern): %r", g)
                    return g
        except Exception:
            pass
    goal = parse_alfworld_goal(task)
    objects = goal["objects"] or []
    sources = goal["sources"] or []
    destinations = goal["destinations"] or []
    put_back = bool(goal.get("put_back"))
    kind = (goal.get("kind") or "pick").strip().lower()

    def _visit_count(a: str) -> int:
        an = _norm(a)
        return sum(1 for p in past if _norm(p) == an)

    def _pick(cands: list[str], *, prefer_stems: list[str] | None = None) -> str | None:
        pool = list(cands)
        if prefer_stems:
            narrowed = _filter_by_stems(pool, prefer_stems)
            if narrowed:
                pool = narrowed
        if blocked or callable(is_blocked):
            cleared = [a for a in pool if not _is_blocked(a)]
            if cleared:
                pool = cleared
        pool = [a for a in pool if _norm(a) not in recent] or pool
        if not pool:
            return None

        def _rank(a: str) -> tuple:
            stem_i = 999
            if prefer_stems:
                for i, s in enumerate(prefer_stems):
                    if _action_mentions_stem(a, s):
                        stem_i = i
                        break
            # Prefer rarely visited goes so hideout search cannot stick on bed/desk.
            return (_visit_count(a), stem_i, a)

        pool = sorted(pool, key=_rank)
        return pool[0]

    puts = [a for a in admissible if cmd_verb_family(a) == "put"]
    takes = [a for a in admissible if cmd_verb_family(a) == "take"]
    goes = [a for a in admissible if cmd_verb_family(a) == "go"]
    opens = [a for a in admissible if cmd_verb_family(a) == "open"]
    uses = [a for a in admissible if cmd_verb_family(a) == "use"]
    heats = [a for a in admissible if cmd_verb_family(a) == "heat"]
    cools = [a for a in admissible if cmd_verb_family(a) == "cool"]
    cleans = [a for a in admissible if cmd_verb_family(a) == "clean"]

    holding = _holding_goal_for_process(admissible, past, objects, inventory=inventory)
    last_src = _source_of_last_take(past)
    mem_blob = (memory_text or ctx_text or look or "").strip()

    proc = _alfworld_process_station(
        kind=kind,
        objects=objects,
        holding=holding,
        past=past,
        recent=recent,
        goes=goes,
        opens=opens,
        takes=takes,
        uses=uses,
        heats=heats,
        cools=cools,
        cleans=cleans,
        pick=_pick,
        logger=logger,
        extra_stations=goal.get("stations") or [],
        ctx_text=ctx_text or look or "",
        memory_text=mem_blob,
    )
    if proc:
        return proc
    station_stems = list(goal.get("stations") or [])
    if kind in _PROCESS_STATIONS:
        for s in _PROCESS_STATIONS[kind][1]:
            if s not in station_stems:
                station_stems.append(s)
    # Heat verbs require microwave; drop stove stems that goal parsing may inject.
    if kind == "heat":
        station_stems = [s for s in station_stems if s not in ("stove", "stoveburner")]
        if "microwave" not in station_stems:
            station_stems.insert(0, "microwave")
    process_incomplete = False
    if kind in _PROCESS_STATIONS:
        process_incomplete = not _process_verb_satisfied(past, kind, station_stems)

    # Holding a goal object with an unfinished process step: never fall through to
    # random hideout exploration (shelf thrashing). Stay on station hosts.
    if process_incomplete and holding:
        choice = _pick_station_go(
            goes,
            kind=kind,
            stations=station_stems,
            memory_text=mem_blob,
            recent=recent,
        )
        if choice:
            if logger:
                logger.info("AlfWorld fast path (go-process-host-holding): %r", choice)
            return choice
        # Still holding: prefer any non-demoted go over shelf/garbage thrashing.
        demote = ["shelf", "garbagecan", "sofa", "bed"]
        if _lamp_like_stations(station_stems):
            pass
        elif kind == "heat":
            demote.extend(["stoveburner", "stove"])
        safe_goes = [
            g for g in goes
            if not any(_action_mentions_stem(g, d) for d in demote)
        ] or list(goes)
        # Prefer primary process stations when still available.
        prefer_st = _station_go_priority(kind, station_stems)
        if kind == "heat":
            mw = [g for g in safe_goes if _action_mentions_stem(g, "microwave")]
            if mw:
                safe_goes = mw
        choice = _pick(safe_goes, prefer_stems=prefer_st or station_stems)
        if choice:
            if logger:
                logger.info("AlfWorld fast path (go-holding-safe): %r", choice)
            return choice

    # Tried put targets (full receptacle phrase after "to ")
    tried_puts: set[str] = set()
    for act in past:
        al = _norm(act)
        if cmd_verb_family(al) == "put":
            m = re.search(r"\bto\s+(.+)$", al)
            if m:
                tried_puts.add(m.group(1).strip())

    # After a successful put-to-dest, do NOT immediately re-take the same object
    # (prevents desk1↔desk2 ping-pong). Exception: put-back still needs care.
    last_put_dest = None
    for act in reversed(past):
        al = _norm(act)
        if cmd_verb_family(al) == "put":
            m = re.search(r"\bto\s+(.+)$", al)
            if m:
                last_put_dest = m.group(1).strip()
            break
        if cmd_verb_family(al) == "take":
            break
    # Multi-object place: remember the receptacle instance used for the first
    # successful goal put even after a subsequent take (second instance).
    locked_put_dest = None
    place_quota_hint = 2 if kind == "pick_two" else 1
    if place_quota_hint > 1:
        fallback_lock = None
        for act in past:
            al = _norm(act)
            if cmd_verb_family(al) != "put":
                continue
            if objects and not any(_action_mentions_stem(al, o) for o in objects):
                continue
            m = re.search(r"\bto\s+(.+)$", al)
            if not m:
                continue
            phrase = m.group(1).strip()
            if fallback_lock is None:
                fallback_lock = phrase
            if destinations and not any(_action_mentions_stem(al, d) for d in destinations):
                continue
            locked_put_dest = phrase
            break
        if not locked_put_dest:
            locked_put_dest = fallback_lock
    dest_lock = locked_put_dest or (
        last_put_dest if (
            place_quota_hint > 1
            and last_put_dest
            and (
                not destinations
                or any(_action_mentions_stem(last_put_dest, d) for d in destinations)
            )
        ) else None
    )

    # Multi-object place: after N-1 successful dest puts, stop go-dest thrash
    # while empty-handed and fetch another instance first. Quota from goal kind
    # (pick_two / "two|another"), not task ids.
    def _place_quota() -> int:
        return 2 if kind == "pick_two" else 1

    def _goal_dest_put_count() -> int:
        n = 0
        for act in past:
            al = _norm(act)
            if cmd_verb_family(al) != "put":
                continue
            if objects and not any(_action_mentions_stem(al, o) for o in objects):
                continue
            if destinations and not any(_action_mentions_stem(al, d) for d in destinations):
                # Still count put when destination stems unknown but object matches.
                if destinations:
                    continue
            n += 1
        return n

    places_done = _goal_dest_put_count()
    places_needed = _place_quota()
    need_another_object = bool(
        places_needed > 1
        and not holding
        and places_done < places_needed
    )
    # Holding with unfinished place quota: never leave the put/go-dest path.
    must_deposit = bool(holding and places_done < places_needed)

    # 1-2) Put / go-dest only after any required process verb has run.
    # Skip while pick_two still needs another object in hand.
    # look-at-light: never put — the object must stay in hand under the lamp.
    win_puts = []
    if kind == "look":
        pass
    elif not process_incomplete and not need_another_object:
        for a in puts:
            if objects and not any(_action_mentions_stem(a, o) for o in objects):
                continue
            if destinations and not any(_action_mentions_stem(a, d) for d in destinations):
                continue
            dest_m = re.search(r"\bto\s+(.+)$", _norm(a))
            dest_phrase = dest_m.group(1).strip() if dest_m else ""
            if last_src and not put_back:
                if last_src == dest_phrase or re.search(
                    rf"\bto\s+{re.escape(last_src)}\b", _norm(a)
                ):
                    continue
            if dest_phrase and dest_phrase in tried_puts and len(puts) > 1:
                # pick_two: the locked receptacle must stay eligible for the 2nd object.
                if not (dest_lock and dest_phrase == dest_lock):
                    continue
            win_puts.append(a)

        if not win_puts and destinations:
            for a in puts:
                if any(_action_mentions_stem(a, d) for d in destinations):
                    dest_m = re.search(r"\bto\s+(.+)$", _norm(a))
                    dest_phrase = dest_m.group(1).strip() if dest_m else ""
                    if last_src and not put_back and dest_phrase == last_src:
                        continue
                    if dest_phrase and dest_phrase in tried_puts and len(puts) > 1:
                        if not (dest_lock and dest_phrase == dest_lock):
                            continue
                    win_puts.append(a)

        if put_back and last_src and puts:
            same = [a for a in puts if re.search(rf"\bto\s+{re.escape(last_src)}\b", _norm(a))]
            if same:
                win_puts = same

        choice = _pick(win_puts, prefer_stems=destinations)
        if choice:
            # Multi-object place: keep depositing into the *same* receptacle instance.
            if dest_lock:
                same_inst = [a for a in win_puts if _matches_dest_lock(a, dest_lock)]
                if same_inst:
                    choice = _pick(same_inst, prefer_stems=destinations) or choice
            if logger:
                logger.info("AlfWorld fast path (put-to-dest): %r", choice)
            return choice

        # Holding anything placeable: if a put to the goal dest is admissible,
        # put immediately (do not open/go thrash first).
        if (holding or must_deposit) and puts:
            held_obj = holding or ""
            held_puts = [
                a for a in puts
                if (not held_obj or _action_mentions_stem(a, held_obj))
                and (
                    not destinations
                    or any(_action_mentions_stem(a, d) for d in destinations)
                )
            ]
            if not held_puts and destinations:
                held_puts = [
                    a for a in puts
                    if any(_action_mentions_stem(a, d) for d in destinations)
                ]
            if kind == "pick_two" and destinations:
                dest_puts = [
                    a for a in held_puts
                    if any(_matches_dest_lock(a, d) for d in destinations)
                ]
                if dest_puts:
                    held_puts = dest_puts
            if not held_puts and must_deposit and dest_lock:
                held_puts = [a for a in puts if _matches_dest_lock(a, dest_lock)]
            if last_src and not put_back:
                held_puts = [
                    a for a in held_puts
                    if not re.search(rf"\bto\s+{re.escape(last_src)}\b", _norm(a))
                ]
            if dest_lock:
                same_inst = [a for a in held_puts if _matches_dest_lock(a, dest_lock)]
                if same_inst:
                    held_puts = same_inst
            choice = _pick(held_puts or [], prefer_stems=destinations)
            if choice:
                if logger:
                    logger.info("AlfWorld fast path (put-holding-dest): %r", choice)
                return choice

        if holding or puts or must_deposit:
            if put_back and last_src:
                back_goes = [g for g in goes if last_src in _norm(g)]
                choice = _pick(back_goes)
                if choice:
                    if logger:
                        logger.info("AlfWorld fast path (go-back-source): %r", choice)
                    return choice
            dest_goes = _filter_by_stems(goes, destinations) if destinations else []
            if dest_lock:
                locked_goes = [g for g in goes if _matches_dest_lock(g, dest_lock)]
                if locked_goes:
                    dest_goes = locked_goes
            elif tried_puts and dest_goes:
                fresh = [
                    g for g in dest_goes
                    if not any(tp == _norm(g.replace("go to ", "", 1)) or tp in _norm(g)
                               for tp in tried_puts)
                ]
                # Prefer exact receptacle match over substring demotion.
                fresh_exact = [
                    g for g in dest_goes
                    if not any(_matches_dest_lock(g, tp) for tp in tried_puts)
                ]
                if fresh_exact:
                    dest_goes = fresh_exact
                elif fresh:
                    dest_goes = fresh
            # Open a closed destination before wandering to another receptacle.
            open_dest = _filter_by_stems(opens, destinations) if destinations else []
            if dest_lock:
                locked_open = [o for o in opens if _matches_dest_lock(o, dest_lock)]
                if locked_open:
                    open_dest = locked_open
            choice = _pick(open_dest, prefer_stems=destinations)
            if choice:
                if logger:
                    logger.info("AlfWorld fast path (open-dest while carrying): %r", choice)
                return choice
            # Visit-cap destination approach — shelf/fridge ping-pong after heat.
            fresh_dest = [g for g in (dest_goes or []) if _visit_count(g) < 2]
            if not fresh_dest:
                fresh_dest = [g for g in (dest_goes or []) if _visit_count(g) < 3]
            choice = _pick(fresh_dest, prefer_stems=destinations) if fresh_dest else None
            if choice:
                if logger:
                    logger.info("AlfWorld fast path (go-dest while carrying): %r", choice)
                return choice
            # Destinations exhausted without put: stop thrashing; let Actor decide.
            if dest_goes and not fresh_dest:
                return None
            # Only wander while carrying after the process step is done (or non-process).
            # Prefer goes that match an admissible put destination over random explore.
            # When goal destinations are known, NEVER fall back to arbitrary put
            # targets (safe/garbage/microwave) — that broke multi-object place.
            if (holding or must_deposit) and not process_incomplete:
                put_dest_stems: list[str] = []
                if destinations:
                    put_dest_stems = list(destinations)
                else:
                    # Unknown destinations: only stems that appear on admissible puts
                    # and are not generic thrash receptacles.
                    demote = {
                        "garbagecan", "safe", "microwave", "fridge", "stoveburner",
                        "stove", "sinkbasin", "sink",
                    }
                    for a in puts:
                        dm = re.search(r"\bto\s+(.+)$", _norm(a))
                        if not dm:
                            continue
                        phrase = dm.group(1).strip()
                        for tok in re.findall(r"[a-z]+", phrase):
                            if tok in _RECEP_STEMS and tok not in put_dest_stems:
                                if tok in demote:
                                    continue
                                put_dest_stems.append(tok)
                    # If demote filtered everything, allow explicit put stems anyway.
                    if not put_dest_stems:
                        for a in puts:
                            dm = re.search(r"\bto\s+(.+)$", _norm(a))
                            if not dm:
                                continue
                            phrase = dm.group(1).strip()
                            for tok in re.findall(r"[a-z]+", phrase):
                                if tok in _RECEP_STEMS and tok not in put_dest_stems:
                                    put_dest_stems.append(tok)
                if put_dest_stems:
                    put_goes = _filter_by_stems(goes, put_dest_stems)
                    choice = _pick(put_goes, prefer_stems=put_dest_stems)
                    if choice:
                        if logger:
                            logger.info(
                                "AlfWorld fast path (go-put-dest while carrying): %r",
                                choice,
                            )
                        return choice
                # Do not random-explore while carrying when destinations are known
                # or a multi-place deposit is still required.
                if destinations or must_deposit:
                    return None
                choice = _pick(goes)
                if choice:
                    if logger:
                        logger.info("AlfWorld fast path (go-explore carrying): %r", choice)
                    return choice

    # 3) Take goal object if visible — but not right after we just put it on a dest
    if objects:
        goal_takes = _filter_by_stems(takes, objects)
    elif kind in _PROCESS_STATIONS:
        # Unknown object stems: do not steal random items during process tasks.
        goal_takes = []
    elif kind in ("pick_two", "pick"):
        # Place tasks with unresolved object stems: never take arbitrary items.
        goal_takes = []
    else:
        goal_takes = list(takes)
    if last_put_dest and not put_back:
        # Suppress re-take from the receptacle we just placed onto
        goal_takes = [
            a for a in goal_takes
            if last_put_dest not in _norm(a)
        ]
        if destinations and any(_action_mentions_stem(last_put_dest, d) for d in destinations):
            other_dest_goes = _filter_by_stems(goes, destinations) if destinations else []
            other_dest_goes = [
                g for g in other_dest_goes
                if last_put_dest not in _norm(g)
                and not any(tp in _norm(g) for tp in tried_puts)
            ]
            # Single-place done: avoid re-take ping-pong. Multi-place still
            # needing another instance must keep admissible goal takes.
            if not other_dest_goes and not (
                places_needed > 1 and places_done < places_needed
            ):
                goal_takes = []
        # Multi-place: after a successful put, take a *different* instance
        if places_needed > 1 and places_done < places_needed:
            last_taken = None
            for act in reversed(past):
                if cmd_verb_family(_norm(act)) == "take":
                    last_taken = _norm(act)
                    break
            if last_taken:
                goal_takes = [a for a in goal_takes if _norm(a) != last_taken]
            # Prefer takes that are not from the destination we already filled.
            if last_put_dest and goal_takes:
                not_from_dest = [
                    a for a in goal_takes
                    if last_put_dest not in _norm(a)
                ]
                if not_from_dest:
                    goal_takes = not_from_dest
    if sources and not put_back and not need_another_object:
        src_takes = _filter_by_stems(goal_takes, sources)
        if src_takes:
            goal_takes = src_takes
    # Same stem/source gates as pattern_force: generic ``bottle`` must not
    # take soapbottle; toilet/garbagecan are distractors unless named.
    goal_takes = [
        a for a in goal_takes
        if _take_object_resolves_goal(a, objects or [], task)
        and _take_source_allowed(a, goal, task)
    ]
    # put-back: take from source desks; prefer not from a dest we already successfully...
    # just take the visible goal object
    choice = _pick(goal_takes, prefer_stems=objects)
    if choice:
        if logger:
            tag = "take-goal-second" if need_another_object else "take-goal"
            logger.info("AlfWorld fast path (%s): %r", tag, choice)
        return choice

    # Multi-place still missing an object: briefly re-check sources, then hideouts.
    # Do NOT loop forever on go-source-second when the next instance lives elsewhere.
    if need_another_object:
        explore_two = _filter_by_stems(goes, sources) if sources else []
        # Prefer never-visited sources first; after one visit each, leave sources.
        fresh_src = [g for g in explore_two if _visit_count(g) < 1]
        if not fresh_src:
            fresh_src = [g for g in explore_two if _visit_count(g) < 2]
        choice = _pick(fresh_src, prefer_stems=sources)
        if choice:
            if logger:
                logger.info("AlfWorld fast path (go-source-second): %r", choice)
            return choice
        open_src = _filter_by_stems(
            opens, (sources or []) + ["drawer", "cabinet", "shelf", "safe"],
        )
        fresh_open = [o for o in (open_src or opens) if _visit_count(o) < 1]
        choice = _pick(fresh_open, prefer_stems=sources)
        if choice and not _filter_by_stems(takes, objects):
            if logger:
                logger.info("AlfWorld fast path (open-second): %r", choice)
            return choice
        # Second instance often elsewhere: hideout sweep with visit caps.
        hide2 = [
            "desk", "shelf", "drawer", "cabinet", "sidetable", "dresser",
            "coffeetable", "countertop", "diningtable", "sofa", "bed", "ottoman",
        ]
        if destinations:
            hide2 = [h for h in hide2 if h not in destinations]
        if sources:
            # Prefer non-source hideouts once sources are exhausted.
            hide2 = [h for h in hide2 if h not in sources] + [
                h for h in hide2 if h in sources
            ]
        explore_h = _filter_by_stems(goes, hide2)
        fresh_h = [g for g in explore_h if _visit_count(g) < 2]
        choice = _pick(fresh_h, prefer_stems=hide2)
        if choice:
            if logger:
                logger.info("AlfWorld fast path (go-hideout-second): %r", choice)
            return choice
        open_h = _filter_by_stems(opens, hide2)
        fresh_oh = [o for o in open_h if _visit_count(o) < 2]
        choice = _pick(fresh_oh, prefer_stems=hide2)
        if choice:
            if logger:
                logger.info("AlfWorld fast path (open-hideout-second): %r", choice)
            return choice
        # Exhausted: do not fall through to go-any thrash; let Actor decide.
        return None

    # 4) Open closed receptacles that may hide the object (drawer/cabinet/safe/fridge)
    hide = ["drawer", "cabinet", "safe", "fridge", "microwave", "laundryhamper"]
    if kind == "heat":
        hide = ["fridge", "cabinet", "drawer", "microwave", "safe"]
    open_goal = _filter_by_stems(opens, sources + hide)
    choice = _pick(open_goal or opens, prefer_stems=sources + hide)
    # Only open if we still lack a *goal* take (non-goal takes do not count).
    recent_looks = sum(1 for a in past[-4:] if cmd_verb_family(_norm(a)) == "look")
    has_goal_take = bool(_filter_by_stems(takes, objects)) if objects else bool(takes)
    if choice and not has_goal_take:
        if logger:
            logger.info("AlfWorld fast path (open): %r", choice)
        return choice

    # 5) Explore: prefer source receptacles; skip for look (parsed sources mislead).
    if kind != "look":
        explore = _filter_by_stems(goes, sources) if sources else []
        choice = _pick(explore, prefer_stems=sources)
        if choice:
            if logger:
                logger.info("AlfWorld fast path (go-source): %r", choice)
            return choice
    # look: sweep hideouts before lamp-host ping-pong when the object is not visible.
    if kind == "look" and not holding:
        look_goal_takes = _filter_by_stems(takes, objects) if objects else []
        if not look_goal_takes:
            look_hideouts = [
                "desk", "sidetable", "dresser", "shelf", "drawer", "coffeetable",
                "sofa", "bed",
            ]
            explore_lk = _filter_by_stems(goes, look_hideouts)
            fresh_lk = [g for g in explore_lk if _visit_count(g) < 2]
            if not fresh_lk:
                fresh_lk = [g for g in explore_lk if _visit_count(g) < 3]
            choice = _pick(fresh_lk, prefer_stems=look_hideouts) if fresh_lk else None
            if choice:
                if logger:
                    logger.info("AlfWorld fast path (go-hideout-look): %r", choice)
                return choice
            open_lk = _filter_by_stems(opens, look_hideouts)
            fresh_olk = [o for o in open_lk if _visit_count(o) < 2]
            choice = _pick(fresh_olk or open_lk, prefer_stems=look_hideouts)
            if choice:
                if logger:
                    logger.info("AlfWorld fast path (open-hideout-look): %r", choice)
                return choice
    # Process incomplete + empty-handed: only walk co-location hosts when the
    # station typically shares a receptacle with the object (lamp-like).
    # Do NOT visit heat/cool/clean appliances before the goal object is held.
    if (
        process_incomplete
        and not holding
        and station_stems
        and _lamp_like_stations(station_stems)
    ):
        host_goes = _rank_station_host_goes(goes, station_stems, mem_blob)
        # Visit-cap lamp-host search (desk↔dresser / desk1↔desk2 thrash).
        fresh_host = [g for g in host_goes if _visit_count(g) < 2]
        if not fresh_host:
            fresh_host = [g for g in host_goes if _visit_count(g) < 3]
        choice = _pick(fresh_host, prefer_stems=_host_stems_for_stations(station_stems)) if fresh_host else None
        if choice:
            if logger:
                logger.info("AlfWorld fast path (go-process-host): %r", choice)
            return choice
        prefer_hosts = _host_stems_for_stations(station_stems)
        host_explore = _filter_by_stems(goes, prefer_hosts) if prefer_hosts else []
        fresh_he = [g for g in host_explore if _visit_count(g) < 2]
        if not fresh_he:
            fresh_he = [g for g in host_explore if _visit_count(g) < 3]
        choice = _pick(fresh_he, prefer_stems=prefer_hosts) if fresh_he else None
        if choice:
            if logger:
                logger.info("AlfWorld fast path (go-process-host-any): %r", choice)
            return choice
        # Hosts exhausted: fall through to hideout / Actor instead of oscillating.
    # Common hideouts when goal sources unknown (pick & place / process fetch).
    # Use *goal* takes — non-goal takes (glassbottle etc.) must not block search.
    goal_takes_now = _filter_by_stems(takes, objects) if objects else []
    if not holding and (not takes if not objects else not goal_takes_now):
        hideouts = [
            "desk", "shelf", "drawer", "cabinet", "sidetable", "dresser", "coffeetable",
            "countertop", "diningtable", "sofa", "bed", "ottoman", "garbagecan",
            "safe", "fridge", "microwave",
        ]
        # Heat/cool: food often starts in fridge / on counters before the station.
        if kind == "heat":
            hideouts = [
                "fridge", "countertop", "diningtable", "cabinet", "drawer", "sinkbasin",
                "microwave", "garbagecan",
            ] + [h for h in hideouts if h not in {
                "fridge", "countertop", "diningtable", "cabinet", "drawer", "sinkbasin",
                "microwave", "garbagecan",
            }]
        elif kind == "cool":
            hideouts = [
                "countertop", "diningtable", "cabinet", "drawer", "stoveburner",
            ] + [h for h in hideouts if h not in {
                "countertop", "diningtable", "cabinet", "drawer", "stoveburner",
            }]
        elif kind == "clean":
            hideouts = [
                "countertop", "cabinet", "drawer", "sidetable", "diningtable", "desk",
                "shelf", "sofa", "bed", "dresser",
            ] + [h for h in hideouts if h not in {
                "countertop", "cabinet", "drawer", "sidetable", "diningtable", "desk",
                "shelf", "sofa", "bed", "dresser",
            }]
            # Garbagecan is rarely the cloth/bowl source and causes 2-node thrash.
            hideouts = [h for h in hideouts if h != "garbagecan"]
        elif kind == "look":
            # Prefer lamp co-location hosts first; bed/sofa last (easy to thrash).
            hideouts = [
                "desk", "sidetable", "dresser", "shelf", "drawer", "coffeetable",
                "sofa", "bed",
            ] + [h for h in hideouts if h not in {
                "desk", "sidetable", "dresser", "shelf", "drawer", "coffeetable",
                "sofa", "bed",
            }]
        # Avoid walking to process stations empty-handed (heat before take).
        # Do NOT exclude goal destinations while searching: clean/heat objects
        # often start inside the same receptacle class as the put target
        # (cabinet/fridge), and excluding them caused 2-node thrash.
        if process_incomplete and station_stems:
            hideouts = [h for h in hideouts if h not in station_stems]
        elif destinations and holding:
            hideouts = [h for h in hideouts if h not in destinations]
        explore = _filter_by_stems(goes, hideouts)
        # Visit-cap hideout search (same idea as pick_two go-hideout-second):
        # prevent countertop↔garbagecan / fridge↔countertop ping-pong.
        fresh_h = [g for g in explore if _visit_count(g) < 2]
        if not fresh_h:
            fresh_h = [g for g in explore if _visit_count(g) < 3]
        choice = _pick(fresh_h, prefer_stems=hideouts) if fresh_h else None
        if choice:
            if logger:
                logger.info("AlfWorld fast path (go-hideout): %r", choice)
            return choice
        # Closed hideouts: open fridge/cabinet/drawer before wandering further.
        open_hide = _filter_by_stems(opens, hideouts)
        fresh_oh = [o for o in open_hide if _visit_count(o) < 2]
        choice = _pick(fresh_oh or open_hide, prefer_stems=hideouts)
        if choice:
            if logger:
                logger.info("AlfWorld fast path (open-hideout): %r", choice)
            return choice
        # After exhausting hideouts, try a fresh look so takes can appear.
        for look_cmd in ("look", "look around"):
            for a in admissible:
                if _norm(a) == look_cmd and _visit_count(a) < 2 and _norm(a) not in recent:
                    if logger:
                        logger.info("AlfWorld fast path (look-hideout): %r", a)
                    return a
        # Broader go pool excluding thrash destinations / already-capped hideouts.
        capped = { _norm(g) for g in explore if _visit_count(g) >= 3 }
        broader = [
            g for g in goes
            if _norm(g) not in capped
            and not (destinations and any(_action_mentions_stem(g, d) for d in destinations))
        ]
        choice = _pick(broader)
        if choice:
            if logger:
                logger.info("AlfWorld fast path (go-hideout-broad): %r", choice)
            return choice
    # Only seek destination receptacles once carrying (handled in step 2); else any unvisited go.
    # Unfinished multi-place quota: never random go-any (that caused bed/shelf thrash).
    if places_done < places_needed and places_needed > 1:
        return None
    go_pool = goes
    if process_incomplete and not holding and station_stems:
        go_pool = [
            g for g in goes
            if not any(_action_mentions_stem(g, s) for s in station_stems)
        ] or goes
    choice = _pick(go_pool)
    if choice:
        if logger:
            logger.info("AlfWorld fast path (go-any): %r", choice)
        return choice

    # 6) Anti-look: only when stuck looking; otherwise return None so Actor can decide
    if recent_looks >= 2:
        for fam_list in (opens, goes, takes, puts, admissible):
            for a in fam_list:
                if cmd_verb_family(a) not in _LOW_VALUE and not _is_meta_cmd(a) and _norm(a) not in recent:
                    if logger:
                        logger.info("AlfWorld fast path (anti-look): %r", a)
                    return a
    return None


def ground_alfworld_prediction(
    predictions,
    admissible: list[str],
    *,
    task: str = "",
    recent_actions: list | None = None,
    look: str = "",
    inventory: str = "",
    sbert_model=None,
    logger=None,
    k: int = 5,
    prefer_fast: bool = True,
) -> str:
    """Full AlfWorld grounding pipeline used by findValidActionNew."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if predictions is None:
        predictions = []
    elif isinstance(predictions, str):
        predictions = [predictions]
    else:
        predictions = [str(p) for p in predictions if p is not None]

    if not admissible:
        return "look"

    recent = {_norm(a) for a in (recent_actions or [])[-6:] if a}
    past = list(recent_actions or [])

    # SciWorld-style: deterministic policy first when confident
    if prefer_fast:
        fast = alfworld_fast_substitute(
            admissible, recent, f"{look} {inventory}", task, past,
            look=look, inventory=inventory, logger=logger,
        )
        if fast:
            return fast

    # Exact / verb-family match for each prediction
    for pred in predictions:
        hit = match_admissible_alfworld(pred, admissible, task=task, recent=recent)
        if hit:
            if logger:
                logger.info("AlfWorld ground match: %r -> %r", pred, hit)
            return hit

    # SBERT only within predicted verb family (if any)
    try:
        if predictions and sbert_model is not None:
            pred0 = normalize_alfworld_prediction(predictions[0] or "")
            fam = _pred_verb_family(pred0)
            cand = [a for a in admissible if _norm(a) not in recent]
            if fam in ("take", "put", "go", "open", "close", "heat", "cool", "clean", "use", "examine"):
                fam_cand = [
                    a for a in cand
                    if cmd_verb_family(a) == fam and not _is_meta_cmd(a)
                ]
                if fam_cand:
                    cand = fam_cand
            cand = [a for a in cand if not _is_meta_cmd(a)] or list(admissible)
            if pred0 and cand:
                emb_p = sbert_model.encode([pred0], normalize_embeddings=True)
                emb_c = sbert_model.encode(cand, normalize_embeddings=True)
                sims = cosine_similarity(emb_p, emb_c)[0]
                order = list(np.argsort(-sims)[: max(1, k)])
                best = cand[int(order[0])]
                if _is_meta_cmd(best):
                    raise ValueError("meta cmd")
                # Guard: reject low-sim cross garbage
                if float(sims[order[0]]) < 0.25 and fam in ("take", "put", "use"):
                    raise ValueError("low sim")
                if logger:
                    logger.info(
                        "AlfWorld ground sbert(family=%s): %r -> %r (sim=%.3f)",
                        fam, pred0, best, float(sims[order[0]]),
                    )
                return best
    except Exception as exc:
        if logger:
            logger.info("AlfWorld sbert skipped: %s", exc)

    # Final fallback: fast path again without recent filter pressure
    fast = alfworld_fast_substitute(
        admissible, set(), f"{look} {inventory}", task, past,
        look=look, inventory=inventory, logger=logger,
    )
    if fast:
        return fast
    return admissible[0]
