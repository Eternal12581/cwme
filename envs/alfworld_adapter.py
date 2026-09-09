"""
AlfWorld (TextWorld / AlfredTWEnv) adapter with a ScienceWorld-like API.

Config (optional under EXPERIMENT.ALFWORLD or AGENT.ALFWORLD):
  DATA_ROOT: path containing json_2.1.1/ (default: $ALFWORLD_DATA or ~/.cache/alfworld)
  ENV_TYPE: AlfredTWEnv (text only; Thor not required for this adapter)
  MAX_EPISODE_STEPS: int (default 100)
  TASK_TYPES: [1..6] filter when TASKS contains "all"

TASKS in config.yml use AlfWorld task-type strings or ids, e.g.:
  pick_and_place_simple | look_at_obj_in_light | pick_clean_then_place_in_recep |
  pick_heat_then_place_in_recep | pick_cool_then_place_in_recep | pick_two_obj_and_place
  or numeric 1..6, or "all".

SET mapping (via getVariations*):
  train  -> json_2.1.1/train
  test*  -> json_2.1.1/valid_unseen  (eval_out_of_distribution)
  dev    -> json_2.1.1/valid_seen    (eval_in_distribution)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}
TASK_NAME_TO_ID = {v: k for k, v in TASK_TYPES.items()}
# Friendly aliases for config TASKS lists
TASK_ALIASES = {
    "pick": "pick_and_place_simple",
    "pick_and_place": "pick_and_place_simple",
    "pick&place": "pick_and_place_simple",
    "examine": "look_at_obj_in_light",
    "look": "look_at_obj_in_light",
    "clean": "pick_clean_then_place_in_recep",
    "heat": "pick_heat_then_place_in_recep",
    "cool": "pick_cool_then_place_in_recep",
    "pick_two": "pick_two_obj_and_place",
    "picktwo": "pick_two_obj_and_place",
}

SPLIT_TRAIN = "train"
SPLIT_SEEN = "eval_in_distribution"
SPLIT_UNSEEN = "eval_out_of_distribution"


def _alf_cfg(agent_config: dict) -> dict:
    exp = (agent_config or {}).get("EXPERIMENT") or {}
    agent = (agent_config or {}).get("AGENT") or {}
    merged = {}
    merged.update(agent.get("ALFWORLD") or {})
    merged.update(exp.get("ALFWORLD") or {})
    return merged


def _default_data_root() -> str:
    env = (os.environ.get("ALFWORLD_DATA") or "").strip()
    if env:
        return os.path.expanduser(env)
    return os.path.expanduser("~/.cache/alfworld")


def _split_dir(data_root: str, split: str) -> str:
    base = os.path.join(data_root, "json_2.1.1")
    if split == SPLIT_TRAIN:
        return os.path.join(base, "train")
    if split == SPLIT_SEEN:
        return os.path.join(base, "valid_seen")
    if split == SPLIT_UNSEEN:
        return os.path.join(base, "valid_unseen")
    raise ValueError(f"Unknown AlfWorld split: {split}")


def normalize_task_name(task: str | int) -> str:
    if isinstance(task, int) or (isinstance(task, str) and task.strip().isdigit()):
        tid = int(task)
        if tid not in TASK_TYPES:
            raise ValueError(f"AlfWorld task id must be 1..6, got {tid}")
        return TASK_TYPES[tid]
    raw = str(task).strip().lower().replace(" ", "_").replace("-", "_")
    if raw in ("all", "*"):
        return "all"
    if raw in TASK_ALIASES:
        return TASK_ALIASES[raw]
    if raw in TASK_NAME_TO_ID:
        return raw
    # tolerate partial prefixes
    for name in TASK_TYPES.values():
        if raw in name or name.startswith(raw):
            return name
    raise ValueError(
        f"Unknown AlfWorld TASK={task!r}. "
        f"Use one of: {list(TASK_TYPES.values())} or 1..6 or all"
    )


def _extract_task_desc(traj: dict, feedback: str = "") -> str:
    # Prefer templated / human goal from traj_data
    try:
        anns = (traj.get("turk_annotations") or {}).get("anns") or []
        if anns and anns[0].get("task_desc"):
            return str(anns[0]["task_desc"]).strip()
    except Exception:
        pass
    try:
        gd = traj.get("template") or {}
        if isinstance(gd, dict) and gd.get("task_desc"):
            return str(gd["task_desc"]).strip()
    except Exception:
        pass
    m = re.search(r"your task is to:\s*(.+)", feedback or "", flags=re.I | re.S)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    return (feedback or "").strip()[:500]


def _collect_games(data_root: str, split: str, task_name: str) -> list[str]:
    """Return solvable game.tw-pddl paths for one task type (or all)."""
    root = _split_dir(data_root, split)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"AlfWorld data missing: {root}\n"
            "Install/download with: pip install alfworld && alfworld-download\n"
            "Or set EXPERIMENT.ALFWORLD.DATA_ROOT / $ALFWORLD_DATA"
        )
    wanted = None if task_name == "all" else {task_name}
    games: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        if "traj_data.json" not in files:
            continue
        if "movable" in dirpath or "Sliced" in dirpath:
            continue
        game_path = os.path.join(dirpath, "game.tw-pddl")
        if not os.path.isfile(game_path):
            continue
        traj_path = os.path.join(dirpath, "traj_data.json")
        try:
            with open(traj_path, "r", encoding="utf-8") as f:
                traj = json.load(f)
            tt = traj.get("task_type")
            if wanted is not None and tt not in wanted:
                continue
            with open(game_path, "r", encoding="utf-8") as f:
                gamedata = json.load(f)
            if not gamedata.get("solvable", True):
                continue
        except Exception:
            continue
        games.append(game_path)
    games.sort()
    return games


class AlfWorldEnvAdapter:
    """Duck-types ScienceWorldEnv for ExperimentRunner / ReasoningAgent."""

    env_name = "alfworld"

    def __init__(self, agent_config: dict | None = None):
        try:
            import textworld  # noqa: F401
            import textworld.gym  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "AlfWorld adapter requires textworld (+ alfworld). "
                "Install: pip install alfworld"
            ) from exc

        self.agent_config = agent_config or {}
        cfg = _alf_cfg(self.agent_config)
        self.data_root = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(cfg.get("DATA_ROOT") or _default_data_root())
            )
        )
        self.max_episode_steps = int(cfg.get("MAX_EPISODE_STEPS") or 100)
        self._task = "pick_and_place_simple"
        self._active_split = SPLIT_UNSEEN
        self._game_cache: dict[tuple[str, str], list[str]] = {}
        self._gym_env = None
        self._gym_env_id = None
        self._current_game: str | None = None
        self._moves = 0
        self._score = 0
        self._won = False
        self._task_desc = ""
        self._look = ""
        self._inventory = ""
        self._feedback = ""
        self._admissible: list[str] = []
        self._possible_actions: list[str] = []
        self._possible_objects: list[str] = []
        self._traj_meta: dict[str, Any] = {}

    # ------------------------------------------------------------------ index
    def _games(self, task: str, split: str) -> list[str]:
        key = (task, split)
        if key not in self._game_cache:
            self._game_cache[key] = _collect_games(self.data_root, split, task)
        return self._game_cache[key]

    def _variations_for_split(self, split: str) -> list[int]:
        games = self._games(self._task, split)
        return list(range(len(games)))

    def getVariationsTrain(self) -> list[int]:
        self._active_split = SPLIT_TRAIN
        return self._variations_for_split(SPLIT_TRAIN)

    def getVariationsDev(self) -> list[int]:
        self._active_split = SPLIT_SEEN
        return self._variations_for_split(SPLIT_SEEN)

    def getVariationsTest(self) -> list[int]:
        self._active_split = SPLIT_UNSEEN
        return self._variations_for_split(SPLIT_UNSEEN)

    def configure_split(self, set_name: str | None) -> str:
        """Map ExperimentRunner SET string onto train/seen/unseen before load()."""
        key = str(set_name or "").strip().lower()
        if key in ("train",):
            self._active_split = SPLIT_TRAIN
        elif key in ("seen", "eval_id", "eval_in_distribution", "valid_seen", "dev"):
            self._active_split = SPLIT_SEEN
        else:
            # test / test_mini* / unseen / default
            self._active_split = SPLIT_UNSEEN
        return self._active_split

    # ------------------------------------------------------------------ load
    def load(
        self,
        taskName: str,
        variationIdx: int = 0,
        simplificationStr: str = "",
        generateGoldPath: bool = False,
    ) -> None:
        del simplificationStr, generateGoldPath  # ScienceWorld-only knobs
        self._task = normalize_task_name(taskName)
        games = self._games(self._task, self._active_split)
        if not games:
            raise RuntimeError(
                f"No AlfWorld games for task={self._task!r} split={self._active_split} "
                f"under {self.data_root}"
            )
        idx = int(variationIdx)
        if idx < 0 or idx >= len(games):
            raise IndexError(
                f"AlfWorld variationIdx={idx} out of range for "
                f"task={self._task} split={self._active_split} (n={len(games)})"
            )
        self._current_game = games[idx]
        self._close_gym()
        self._open_gym(self._current_game)
        # Prefetch traj metadata for task description
        traj_path = os.path.join(os.path.dirname(self._current_game), "traj_data.json")
        try:
            with open(traj_path, "r", encoding="utf-8") as f:
                self._traj_meta = json.load(f)
        except Exception:
            self._traj_meta = {}

    def _close_gym(self) -> None:
        if self._gym_env is not None:
            try:
                self._gym_env.close()
            except Exception:
                pass
        self._gym_env = None
        self._gym_env_id = None

    def _open_gym(self, game_file: str) -> None:
        import textworld
        import textworld.gym

        try:
            from alfworld.agents.environment.alfred_tw_env import (
                AlfredDemangler,
                AlfredInfos,
            )
            wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
        except Exception:
            wrappers = []

        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            description=True,
            inventory=True,
            feedback=True,
            score=True,
            moves=True,
            extras=["gamefile"] if wrappers else [],
        )
        # batch_size=None → non-batched single env when possible; AlfWorld
        # stack typically uses batch_size=1 (list-wrapped returns).
        self._gym_env_id = textworld.gym.register_games(
            [game_file],
            request_infos,
            max_episode_steps=self.max_episode_steps,
            wrappers=wrappers or None,
            name=f"alfdavis-{abs(hash(game_file)) % (10**8)}",
        )
        self._gym_env = textworld.gym.make(self._gym_env_id)

    # ------------------------------------------------------------------ episode
    def reset(self):
        if self._gym_env is None:
            raise RuntimeError("AlfWorldEnvAdapter.reset() called before load()")
        raw = self._gym_env.reset()
        obs, infos = self._unpack_reset(raw)
        self._moves = 0
        self._score = 0
        self._won = False
        self._apply_obs(obs, infos)
        # ScienceWorld reset does an initial look; keep caches warm.
        return self._look or obs, self._info_dict()

    def step(self, action: str):
        if self._gym_env is None:
            raise RuntimeError("AlfWorldEnvAdapter.step() called before load()")
        # TextworldGymEnv.step expects a str and wraps [command] internally.
        # Passing a list causes: AttributeError: 'list' object has no attribute 'strip'
        act = (action or "").strip()
        raw = self._gym_env.step(act)
        obs, reward, done, infos = self._unpack_step(raw)
        self._moves += 1
        self._apply_obs(obs, infos)
        # Map won / dense score → ScienceWorld-like 0..100
        won = bool(self._unwrap(infos.get("won"), False))
        tw_score = self._unwrap(infos.get("score"), None)
        if won:
            self._score = 100
            self._won = True
            done = True
        elif tw_score is not None:
            try:
                s = float(tw_score)
                self._score = int(round(100.0 * s)) if s <= 1.0 else int(round(s))
                self._score = max(0, min(100, self._score))
            except Exception:
                pass
        elif reward:
            # Sparse progress signal if engine gives reward without score
            try:
                self._score = max(self._score, int(round(100.0 * float(reward))))
            except Exception:
                pass
        info = self._info_dict()
        return self._feedback or obs, float(reward or 0.0), bool(done), info

    def look(self) -> str:
        return self._look or self._feedback or ""

    def inventory(self) -> str:
        return self._inventory or "You are carrying nothing."

    def getTaskDescription(self) -> str:
        if self._task_desc:
            return self._task_desc
        return _extract_task_desc(self._traj_meta, self._feedback)

    def getPossibleActions(self) -> list[str]:
        return list(self._possible_actions)

    def getPossibleObjects(self) -> list[str]:
        return list(self._possible_objects)

    def getValidActionObjectCombinations(self) -> list[str]:
        return list(self._admissible)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _unwrap(value, default=None):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return default if value is None else value

    def _unpack_reset(self, raw):
        if isinstance(raw, tuple) and len(raw) == 2:
            return raw[0], raw[1] if isinstance(raw[1], dict) else {}
        if isinstance(raw, dict):
            return raw.get("feedback") or raw.get("description") or "", raw
        return str(raw), {}

    def _unpack_step(self, raw):
        # Common shapes:
        #  (obs, reward, done, infos)
        #  (obs, scores, dones, infos)  # AlfWorld agents example
        if not isinstance(raw, tuple):
            return str(raw), 0.0, False, {}
        if len(raw) == 4:
            obs, a, b, infos = raw
            infos = infos if isinstance(infos, dict) else {}
            # Distinguish reward(float) vs scores(list)
            if isinstance(a, (list, tuple)):
                reward = float(a[0]) if a else 0.0
            else:
                reward = float(a or 0.0)
            if isinstance(b, (list, tuple)):
                done = bool(b[0]) if b else False
            else:
                done = bool(b)
            return obs, reward, done, infos
        if len(raw) == 3:
            obs, reward, done = raw
            return obs, float(reward or 0.0), bool(done), {}
        return str(raw[0]), 0.0, False, {}

    def _apply_obs(self, obs, infos: dict) -> None:
        feedback = self._unwrap(obs, "")
        if isinstance(feedback, (list, tuple)):
            feedback = feedback[0] if feedback else ""
        self._feedback = str(feedback or "")
        desc = self._unwrap(infos.get("description"), None)
        if desc is None:
            desc = self._feedback
        if isinstance(desc, (list, tuple)):
            desc = desc[0] if desc else ""
        self._look = str(desc or "")
        inv = self._unwrap(infos.get("inventory"), "You are carrying nothing.")
        if isinstance(inv, (list, tuple)):
            inv = inv[0] if inv else "You are carrying nothing."
        self._inventory = str(inv or "You are carrying nothing.")
        adm = infos.get("admissible_commands")
        if isinstance(adm, (list, tuple)) and adm and isinstance(adm[0], (list, tuple)):
            adm = adm[0]
        self._admissible = [str(x) for x in (adm or [])]
        self._possible_actions = sorted(
            {a.split(" ", 1)[0].lower() for a in self._admissible if a}
        )
        self._possible_objects = self._objects_from_admissible(self._admissible)
        self._task_desc = _extract_task_desc(self._traj_meta, self._feedback)
        self._won = bool(self._unwrap(infos.get("won"), False))

    @staticmethod
    def _objects_from_admissible(commands: list[str]) -> list[str]:
        objs: set[str] = set()
        for cmd in commands:
            # e.g. "take apple from countertop", "go to fridge"
            parts = re.split(r"\s+(?:from|in|on|with|to)\s+", cmd.strip(), maxsplit=1)
            head = parts[0]
            tokens = head.split()
            if len(tokens) >= 2:
                objs.add(" ".join(tokens[1:]).strip())
            if len(parts) > 1:
                objs.add(parts[1].strip())
        return sorted(o for o in objs if o)

    def _info_dict(self) -> dict:
        return {
            "score": int(self._score),
            "moves": int(self._moves),
            "won": bool(self._won),
            "look": self._look,
            "inv": self._inventory,
            "taskDesc": self._task_desc,
            "valid": list(self._admissible),
            "admissible_commands": list(self._admissible),
            "gamefile": self._current_game,
            "task_type": self._task,
            "split": self._active_split,
        }

    def close(self) -> None:
        self._close_gym()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
