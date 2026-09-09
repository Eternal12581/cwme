"""Create a ScienceWorld-compatible env from AGENT.ENVIRONMENT."""

from __future__ import annotations


def normalize_environment_name(raw: str | None) -> str:
    name = (raw or "ScienceWorld").strip().lower().replace("-", "").replace("_", "")
    if name in ("scienceworld", "sciworld", "sw"):
        return "scienceworld"
    if name in ("alfworld", "alfred", "alfredtw", "alfredtwenv"):
        return "alfworld"
    raise ValueError(
        f"Unknown AGENT.ENVIRONMENT={raw!r}. "
        "Supported: ScienceWorld, AlfWorld"
    )


def create_env(agent_config: dict | None = None):
    """
    Build an env that exposes the ScienceWorld-like surface used by DAVIS:
    load / reset / step / look / inventory / getTaskDescription /
    getPossibleActions / getPossibleObjects / getValidActionObjectCombinations /
    getVariationsTrain / getVariationsDev / getVariationsTest.
    """
    agent_config = agent_config or {}
    raw = (agent_config.get("AGENT") or {}).get("ENVIRONMENT", "ScienceWorld")
    kind = normalize_environment_name(raw)

    if kind == "scienceworld":
        from scienceworld import ScienceWorldEnv

        env = ScienceWorldEnv()
        env.env_name = "scienceworld"
        return env

    from envs.alfworld_adapter import AlfWorldEnvAdapter

    return AlfWorldEnvAdapter(agent_config)
