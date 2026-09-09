#!/usr/bin/env python3
"""
Collect executable teacher / ground-truth trajectories for offline CWME memory.

The output is one JSON object per line and is directly consumable by
build_historical_kg.py and build_procedural_memory.py.

ScienceWorld obtains the gold action sequence from generateGoldPath=True.
AlfWorld reads the high-level expert plan from each train split
traj_data.json and replays it in the TextWorld adapter.  ALFRED's
low_actions are embodied actions such as LookDown and Rotate, so they are
not used as TextWorld commands.  Replay is deliberate: state_after and
score evidence must come from the environment.

Examples:
  python scripts/collect_ground_truth_trajectories.py \
      --environment scienceworld \
      --output artifacts/gt/science_train.jsonl \
      --tasks boil melt freeze change-the-state-of-matter-of use-thermometer

  python scripts/collect_ground_truth_trajectories.py \
      --environment alfworld \
      --data-root /mnt/workspace/alfworld \
      --output artifacts/gt/alf_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return _text(value[0]) if value else ""
    if value is None:
        return ""
    return str(value).strip()


def _score(info: dict | None, env: Any, fallback: float = 0.0) -> float:
    info = info or {}
    raw = info.get("score", fallback)
    try:
        value = float(_text(raw))
        # The adapter exposes a 0..100 score.  Some native environments use
        # 0..1; ScienceWorld releases may also expose 1 for a solved task
        # while getScore() reports the canonical 100-point score.
        if 0.0 <= value <= 1.0:
            getter = getattr(env, "getScore", None)
            if callable(getter):
                try:
                    env_value = float(getter())
                    if env_value > 1.0:
                        value = env_value
                    else:
                        value *= 100.0
                except (TypeError, ValueError):
                    value *= 100.0
            else:
                value *= 100.0
        return max(0.0, min(100.0, value))
    except (TypeError, ValueError):
        pass
    getter = getattr(env, "getScore", None)
    if callable(getter):
        try:
            value = float(getter())
            return max(0.0, min(100.0, value))
        except (TypeError, ValueError):
            pass
    return float(fallback)


def _unpack_step(result: Any) -> tuple[str, float, bool, dict]:
    if not isinstance(result, tuple):
        return _text(result), 0.0, False, {}
    if len(result) == 4:
        obs, reward, done, info = result
        return _text(obs), float(_text(reward) or 0.0), bool(done), info if isinstance(info, dict) else {}
    if len(result) == 3:
        obs, reward, done = result
        return _text(obs), float(_text(reward) or 0.0), bool(done), {}
    return _text(result[0]) if result else "", 0.0, False, {}


def _write_rows(path: str, rows: Iterable[dict]) -> int:
    output = Path(path).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _episode_row(
    *,
    episode_id: int,
    task: str,
    variation_id: str,
    steps: list[dict],
    final_score: float,
    success: bool,
    source: dict,
) -> dict:
    return {
        "episode_id": episode_id,
        "task": task,
        "task_id": task,
        "variation_id": str(variation_id),
        "split": "train",
        "final_score": float(final_score),
        "success": bool(success),
        "steps": steps,
        "source": source,
    }


def collect_scienceworld(tasks: list[str], variations: list[int] | None, max_steps: int) -> list[dict]:
    try:
        from scienceworld import ScienceWorldEnv
    except ImportError as exc:
        raise RuntimeError("ScienceWorld collection requires the scienceworld package") from exc

    # Some ScienceWorld releases initialize the Java-side variation sets only
    # after a positive step limit is supplied and a task has been loaded.
    env = ScienceWorldEnv(envStepLimit=max_steps)
    rows: list[dict] = []
    episode_id = 0
    try:
        for task in tasks:
            if variations is None:
                # Prime the Java interface before querying train variations.
                env.load(
                    task,
                    variationIdx=0,
                    simplificationStr="easy",
                    generateGoldPath=True,
                )
                available = list(env.getVariationsTrain())
                selected = [int(v) for v in available]
            else:
                available = None
                selected = variations
            for variation in selected:
                if available is not None and int(variation) not in {int(v) for v in available}:
                    raise ValueError(f"ScienceWorld train variation not available: {task}@{variation}")
                env.load(
                    task,
                    variationIdx=int(variation),
                    simplificationStr="easy",
                    generateGoldPath=True,
                )
                initial = env.reset()
                observation = _text(initial[0] if isinstance(initial, tuple) else initial)
                info = initial[1] if isinstance(initial, tuple) and len(initial) > 1 and isinstance(initial[1], dict) else {}
                score = _score(info, env)
                gold = list(env.getGoldActionSequence() or [])
                steps: list[dict] = []
                done = False
                for step_no, action in enumerate(gold[:max_steps]):
                    action = _text(action)
                    if not action or done:
                        continue
                    before = observation
                    score_before = score
                    observation, _, done, next_info = _unpack_step(env.step(action))
                    score = _score(next_info, env, fallback=score_before)
                    steps.append({
                        "step": step_no,
                        "state_before": before,
                        "action": action,
                        "state_after": observation,
                        "score_before": score_before,
                        "score_after": score,
                        "valid": True,
                        "meaningful_change": bool(before != observation or score > score_before),
                    })
                truncated = bool(len(gold) > max_steps and not done)
                rows.append(_episode_row(
                    episode_id=episode_id,
                    task=task,
                    variation_id=str(variation),
                    steps=steps,
                    final_score=score,
                    success=bool(score >= 100),
                    source={
                        "kind": "scienceworld_gold_action_sequence",
                        "split": "train",
                        "gold_action_count": len(gold),
                        "executed_action_count": len(steps),
                        "truncated_by_max_steps": truncated,
                    },
                ))
                episode_id += 1
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return rows


def _low_action_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    for key in ("action", "command", "text", "sentence", "displ", "description", "high_desc"):
        raw_value = item.get(key)
        if isinstance(raw_value, (dict, list, tuple)):
            continue
        value = _text(raw_value)
        if value:
            return value
    return ""


def _entity_tokens(value: Any) -> set[str]:
    """Extract matching aliases from an ALFRED object id."""
    if isinstance(value, (list, tuple)):
        aliases: set[str] = set()
        for item in value:
            aliases.update(_entity_tokens(item))
        return aliases
    raw = _text(value)
    if not raw:
        return set()
    head = raw.split("|", 1)[0]
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", head)
    aliases = {re.sub(r"[^a-z0-9]", "", part.lower()) for part in parts if part}
    compact = re.sub(r"[^a-z0-9]", "", head.lower())
    if compact:
        aliases.add(compact)
    return {alias for alias in aliases if len(alias) > 1}


_ALF_API_VERBS = {
    "gotolocation": ("go to", "go"),
    "pickupobject": ("take", "pick up"),
    "putobject": ("put", "place", "move"),
    "openobject": ("open",),
    "closeobject": ("close",),
    # TextWorld's ALFWorld adapter commonly exposes lamp toggles as
    # ``use desklamp N`` rather than ``toggle desklamp N``.
    "toggleobject": ("toggle", "use"),
    "sliceobject": ("slice",),
    "cleanobject": ("clean",),
    "heatobject": ("heat", "use"),
    "coolobject": ("cool", "use"),
    "lookatobject": ("examine", "look at"),
}

_ALF_NON_TEXT_API_ACTIONS = frozenset({
    "lookdown", "lookup", "rotate", "moveahead", "moveback",
    "turnleft", "turnright", "look", "initialize", "pass",
})


def _command_tokens(command: str) -> set[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", _normalize_action(command))
    stop = {
        "a", "an", "the", "go", "to", "take", "pick", "up", "put",
        "place", "move", "in", "on", "from", "with", "object", "location",
    }
    tokens = [token for token in raw_tokens if token not in stop]
    output = set(tokens)
    # ALFRED ids use concatenated names (e.g. ``sidetable`` and
    # ``AlarmClock``), while TextWorld commands may tokenize them as
    # ``side table`` / ``alarm clock``.  Include ordered compact n-grams.
    for start in range(len(tokens)):
        compact = ""
        for end in range(start, len(tokens)):
            compact += tokens[end]
            if end > start:
                output.add(compact)
    return output


def _alias_overlap(tokens: set[str], aliases: set[str]) -> int:
    """Count entity matches, including ``sidetable`` vs ``side table``."""
    if not tokens or not aliases:
        return 0
    exact = tokens & aliases
    compact_tokens = {re.sub(r"[^a-z0-9]", "", token) for token in tokens}
    compact_aliases = {re.sub(r"[^a-z0-9]", "", alias) for alias in aliases}
    compact = compact_tokens & compact_aliases
    return max(len(exact), len(compact))


def _api_entity_aliases(payload: dict, api: str, entity: str) -> set[str]:
    """Extract the semantic target of an ALFRED API action.

    ALFRED uses ``objectId`` inconsistently across action families.  For
    example, in ``CleanObject`` records it identifies the sink, while the
    object being cleaned is stored in ``cleanObjectId`` and
    ``coordinateObjectId``.  Selecting fields by action semantics prevents a
    receptacle from being mistaken for the manipulated object.
    """
    if entity == "object":
        if api == "cleanobject":
            keys = ("cleanObjectId", "clean_object_id", "coordinateObjectId", "coordinate_object_id")
        elif api in {"heatobject", "coolobject", "sliceobject"}:
            keys = ("coordinateObjectId", "coordinate_object_id", "objectId", "object_id", "target_object_id")
        else:
            keys = ("objectId", "object_id", "target_object_id", "coordinateObjectId", "coordinate_object_id")
    else:
        keys = (
            "receptacleObjectId",
            "receptacle_object_id",
            "receptacle",
            "coordinateReceptacleObjectId",
            "coordinate_receptacle_object_id",
        )
    for key in keys:
        aliases = _entity_tokens(payload.get(key))
        if aliases:
            return aliases
    return set()


def _resolve_api_action(item: dict, admissible: list[str]) -> tuple[str, bool]:
    """Map an ALFRED API action to an admissible TextWorld command.

    ALFWorld's traj_data.json is embodied-data metadata and often has no
    textual action.  The environment's admissible command list is the only
    authoritative command surface, so this resolver never invents a command.
    """
    payload = dict(item)
    for key in ("api_action", "planner_action", "discrete_action", "action"):
        nested = item.get(key)
        if isinstance(nested, dict):
            payload = {**payload, **nested}
    api = _normalize_action(_text(payload.get("api_action") or payload.get("action_type") or payload.get("action") or ""))
    api = re.sub(r"[^a-z0-9]", "", api)
    verbs = _ALF_API_VERBS.get(api, ())
    object_aliases = _api_entity_aliases(payload, api, "object")
    receptacle_aliases = _api_entity_aliases(payload, api, "receptacle")
    # GotoLocation actions usually have no objectId.  Their TextWorld target
    # is carried by discrete_action.args (e.g. ["sidetable"]), while the
    # planner-side location is an opaque ALFRED coordinate token.
    raw_location = payload.get("location") or payload.get("target_location")
    location_text = _normalize_action(_text(raw_location))
    # ``loc|...`` is an ALFRED coordinate identifier, not a TextWorld
    # location name.  Its numeric tokens must never match ``desk 1`` etc.
    location_aliases = (
        _entity_tokens(raw_location)
        if location_text and not (location_text.startswith("loc|") or "|" in location_text)
        else set()
    )
    argument_aliases = _entity_tokens(payload.get("args"))
    target_aliases = object_aliases | receptacle_aliases | location_aliases | argument_aliases
    if not verbs or not admissible:
        return "", False

    candidates: list[tuple[float, str]] = []
    for command in admissible:
        normalized = _normalize_action(command)
        if not any(normalized == verb or normalized.startswith(verb + " ") for verb in verbs):
            continue
        tokens = _command_tokens(command)
        overlap = _alias_overlap(tokens, target_aliases)
        object_overlap = _alias_overlap(tokens, object_aliases)
        receptacle_overlap = _alias_overlap(tokens, receptacle_aliases)
        # A command for the wrong object is unsafe even when its verb matches.
        if object_aliases and object_overlap == 0:
            continue
        score = float(overlap * 10 + object_overlap * 4 + receptacle_overlap * 3)
        if api == "gotolocation" and _alias_overlap(
            tokens, location_aliases | argument_aliases,
        ):
            score += 5
        if api == "putobject" and receptacle_aliases and receptacle_overlap == 0:
            continue
        candidates.append((score, command))
    if not candidates:
        return "", False
    candidates.sort(key=lambda pair: (-pair[0], pair[1]))
    best_score, best = candidates[0]
    tied = [command for score, command in candidates if score == best_score]
    # A unique best match is required for a ground-truth replay.  Ambiguous
    # matches are rejected instead of silently selecting the wrong instance.
    # The ALFRED high-level plan is sometimes underspecified, however: a
    # PickupObject may provide only object/receptacle *types* while TextWorld
    # exposes several numbered instances of that same pair.  In that narrow
    # case, use a deterministic first instance and let replay success/final
    # score validate the resulting trajectory.
    if best_score <= 0:
        return "", False
    if len(tied) != 1:
        if api == "pickupobject":
            without_indices = {
                re.sub(r"\s+\d+\b", "", _normalize_action(command))
                for command in tied
            }
            if len(without_indices) == 1:
                return best, True
        return "", False
    return best, True


def _resolve_alf_prerequisite(item: dict, admissible: list[str]) -> str:
    """Find an admissible container-opening prerequisite for a pickup.

    ALFRED high-level plans may mark an object as visible in a receptacle,
    whereas the TextWorld replay still requires that receptacle to be opened.
    Return only an environment-admissible ``open`` command; the caller then
    retries the same high-level pickup action after executing it.
    """
    if not isinstance(item, dict):
        return ""
    payload = dict(item)
    for key in ("api_action", "planner_action", "discrete_action", "action"):
        nested = item.get(key)
        if isinstance(nested, dict):
            payload = {**payload, **nested}
    api = re.sub(
        r"[^a-z0-9]",
        "",
        _normalize_action(_text(
            payload.get("api_action")
            or payload.get("action_type")
            or payload.get("action")
            or ""
        )),
    )
    if api != "pickupobject":
        return ""
    aliases = _api_entity_aliases(payload, api, "receptacle")
    if not aliases:
        return ""
    candidates: list[tuple[int, str]] = []
    for command in admissible or []:
        normalized = _normalize_action(command)
        if not normalized.startswith("open "):
            continue
        overlap = _alias_overlap(_command_tokens(command), aliases)
        if overlap:
            candidates.append((overlap, command))
    if not candidates:
        return ""
    candidates.sort(key=lambda pair: (-pair[0], pair[1]))
    best_score, best = candidates[0]
    tied = [command for score, command in candidates if score == best_score]
    return best if best_score > 0 and len(tied) == 1 else ""


def _resolve_alf_action(item: Any, admissible: list[str]) -> tuple[str, bool]:
    text_action = _low_action_text(item)
    if text_action:
        return _choose_admissible(text_action, admissible)
    if isinstance(item, dict):
        return _resolve_api_action(item, admissible)
    return "", False


def _normalize_action(action: str) -> str:
    return re.sub(r"\s+", " ", (action or "").strip().lower())


def _choose_admissible(action: str, admissible: list[str]) -> tuple[str, bool]:
    """Resolve text descriptions against commands without inventing actions."""
    wanted = _normalize_action(action)
    for candidate in admissible:
        if _normalize_action(candidate) == wanted:
            return candidate, True
    wanted_tokens = _command_tokens(wanted)
    if not wanted_tokens:
        return action.strip(), False
    wanted_head = wanted.split(" ", 1)[0]
    candidates: list[tuple[int, str]] = []
    for candidate in admissible:
        candidate_norm = _normalize_action(candidate)
        candidate_head = candidate_norm.split(" ", 1)[0]
        if wanted_head not in {"go", "take", "pick", "put", "place", "move", "open", "close", "toggle", "slice", "clean", "heat", "cool", "examine", "look"}:
            continue
        if candidate_head != wanted_head and not ({wanted_head, candidate_head} <= {"take", "pick"}):
            continue
        overlap = len(wanted_tokens & _command_tokens(candidate_norm))
        if wanted_tokens.issubset(_command_tokens(candidate_norm)):
            candidates.append((overlap, candidate))
    if not candidates:
        return action.strip(), False
    candidates.sort(key=lambda pair: (-pair[0], pair[1]))
    best_score, best = candidates[0]
    tied = [candidate for score, candidate in candidates if score == best_score]
    if best_score <= 0 or len(tied) != 1:
        return action.strip(), False
    return best, True


def _load_plan_actions(path: str) -> list[Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    plan = data.get("plan") or {}
    # high_pddl is the level compatible with the TextWorld command surface.
    # low_actions are ALFRED embodied controls and include LookDown/Rotate.
    high_actions = plan.get("high_pddl") if isinstance(plan, dict) else None
    if isinstance(high_actions, list) and high_actions:
        actions = [
            item for item in high_actions
            if _low_action_text(item) or isinstance(item, dict)
        ]
    else:
        low_actions = plan.get("low_actions") if isinstance(plan, dict) else None
        if not isinstance(low_actions, list):
            raise ValueError(f"No plan.high_pddl or plan.low_actions found in {path}")
        actions = []
        for item in low_actions:
            if not isinstance(item, dict):
                if _low_action_text(item):
                    actions.append(item)
                continue
            payload = item.get("api_action")
            api_name = payload.get("action") if isinstance(payload, dict) else payload
            api_name = re.sub(r"[^a-z0-9]", "", _normalize_action(_text(api_name)))
            if api_name in _ALF_NON_TEXT_API_ACTIONS:
                continue
            if _low_action_text(item) or isinstance(payload, dict):
                actions.append(item)
    if not actions:
        keys = sorted(plan.keys()) if isinstance(plan, dict) else []
        raise ValueError(
            f"No usable high-level actions in {path}; plan keys={keys}"
        )
    return actions


def collect_alfworld(data_root: str, tasks: list[str], max_episodes_per_task: int | None, max_steps: int) -> list[dict]:
    from envs.alfworld_adapter import (
        AlfWorldEnvAdapter,
        SPLIT_TRAIN,
        _collect_games,
        normalize_task_name,
    )

    config = {
        "AGENT": {"ENVIRONMENT": "AlfWorld"},
        "EXPERIMENT": {
            "SET": "train",
            "ALFWORLD": {"DATA_ROOT": data_root, "MAX_EPISODE_STEPS": max_steps},
        },
    }
    env = AlfWorldEnvAdapter(config)
    # The adapter defaults to valid_unseen for evaluation.  This collector
    # replays train-split expert plans, so the environment must load the same
    # train game as the traj_data.json source below.
    env.configure_split("train")
    rows: list[dict] = []
    episode_id = 0
    try:
        for raw_task in tasks:
            task = normalize_task_name(raw_task)
            games = _collect_games(env.data_root, SPLIT_TRAIN, task)
            if max_episodes_per_task is not None:
                games = games[:max_episodes_per_task]
            for variation, game_path in enumerate(games):
                traj_path = os.path.join(os.path.dirname(game_path), "traj_data.json")
                actions = _load_plan_actions(traj_path)
                env.load(task, variationIdx=variation)
                initial = env.reset()
                observation = _text(initial[0] if isinstance(initial, tuple) else initial)
                info = initial[1] if isinstance(initial, tuple) and len(initial) > 1 and isinstance(initial[1], dict) else {}
                score = _score(info, env)
                steps: list[dict] = []
                done = False
                executed_steps = 0
                truncated = False
                for plan_step_no, proposed in enumerate(actions[:max_steps]):
                    if done or truncated:
                        break
                    while not done:
                        if executed_steps >= max_steps:
                            truncated = True
                            break
                        admissible = list(info.get("admissible_commands") or info.get("valid") or [])
                        action, exact = _resolve_alf_action(proposed, admissible)
                        prerequisite = False
                        if not exact:
                            action = _resolve_alf_prerequisite(proposed, admissible)
                            exact = bool(action)
                            prerequisite = exact
                        if not exact:
                            keys = sorted(proposed.keys()) if isinstance(proposed, dict) else []
                            raise RuntimeError(
                                f"Cannot resolve ALFWorld expert action at {traj_path} "
                                f"step={plan_step_no}, api_action={proposed.get('api_action') or proposed.get('planner_action') if isinstance(proposed, dict) else ''!r}, "
                                f"keys={keys}, admissible_count={len(admissible)}, "
                                f"admissible={admissible!r}"
                            )
                        before = observation
                        score_before = score
                        observation, _, done, info = _unpack_step(env.step(action))
                        score = _score(info, env, fallback=score_before)
                        steps.append({
                            "step": executed_steps,
                            "plan_step": plan_step_no,
                            "action": action,
                            "state_before": before,
                            "state_after": observation,
                            "score_before": score_before,
                            "score_after": score,
                            "valid": bool(exact),
                            "meaningful_change": bool(before != observation or score > score_before),
                        })
                        executed_steps += 1
                        # A prerequisite does not consume the high-level plan
                        # action.  Re-resolve the same PickupObject now that
                        # the receptacle has been opened.
                        if not prerequisite:
                            break
                replay_valid = bool(steps) and all(bool(step["valid"]) for step in steps)
                rows.append(_episode_row(
                    episode_id=episode_id,
                    task=task,
                    variation_id=str(variation),
                    steps=steps,
                    final_score=score,
                    success=bool((info.get("won") or score >= 100) and replay_valid),
                    source={
                        "kind": "alfworld_train_expert_replay",
                        "split": "train",
                        "game_file": game_path,
                        "traj_data": traj_path,
                        "high_action_count": len(actions),
                        "executed_action_count": len(steps),
                        "truncated_by_max_steps": truncated,
                    },
                ))
                episode_id += 1
    finally:
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect train-split executable ground-truth trajectories")
    parser.add_argument("--environment", choices=("scienceworld", "alfworld"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--variations", nargs="+", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--data-root", default="")
    args = parser.parse_args()

    if args.environment == "scienceworld":
        tasks = args.tasks or [
            "boil", "melt", "freeze", "change-the-state-of-matter-of", "use-thermometer",
        ]
        rows = collect_scienceworld(tasks, args.variations, max(1, args.max_steps))
    else:
        if not args.data_root:
            raise SystemExit("--data-root is required for AlfWorld")
        tasks = args.tasks or [
            "pick_and_place_simple", "pick_clean_then_place_in_recep", "look_at_obj_in_light",
            "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
        ]
        rows = collect_alfworld(
            args.data_root,
            tasks,
            args.max_episodes_per_task,
            max(1, args.max_steps),
        )

    count = _write_rows(args.output, rows)
    successes = sum(bool(row.get("success")) for row in rows)
    print(f"Saved {count} train trajectories -> {args.output}")
    print(f"Successful trajectories: {successes}/{count}")


if __name__ == "__main__":
    main()
