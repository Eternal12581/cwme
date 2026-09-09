"""Environment adapters: ScienceWorld + AlfWorld behind one factory."""

from envs.base import is_alfworld, is_scienceworld
from envs.factory import create_env, normalize_environment_name

__all__ = [
    "create_env",
    "normalize_environment_name",
    "is_scienceworld",
    "is_alfworld",
]
