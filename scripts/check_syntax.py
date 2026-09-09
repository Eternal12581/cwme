import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for rel in (
    "core/pattern_library.py",
    "core/scienceworld_grounding.py",
    "core/world_model/hkg_config.py",
    "core/world_model/agent_bridge.py",
    "core/grounding_facade.py",
    "core/state_packet.py",
    "ReasoningAgent.py",
    "scripts/audit_experiment_configs.py",
):
    ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)
print("syntax ok")
