"""Fix indentation and imports in reuse_policy_legacy.py."""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "core" / "reuse_policy_legacy.py"
lines = path.read_text(encoding="utf-8").splitlines()

sig_end = None
body_end = None
for idx, line in enumerate(lines):
    if line.strip() == "log_prefix: str = \"REUSE\",":
        # next non-empty after closing ): 
        pass
for idx, line in enumerate(lines):
    if line == "    ):" and idx > 800:
        sig_end = idx + 1
        break

for idx in range(len(lines) - 1, 0, -1):
    if lines[idx].startswith("pattern_force_action_ok"):
        body_end = idx
        break

if sig_end is None or body_end is None:
    raise SystemExit(f"markers not found sig={sig_end} body={body_end}")

fixed: list[str] = lines[:sig_end]
fixed.append("        from core.grounding_facade import ExecutionDecision")
fixed.append("")
for line in lines[sig_end:body_end]:
    if not line.strip():
        fixed.append("")
    elif line.startswith("        "):
        fixed.append(line)
    elif line.startswith("    "):
        fixed.append("    " + line)
    else:
        fixed.append("        " + line)
fixed.extend(lines[body_end:])

# TYPE_CHECKING hint for resolve_actor_override return
text = "\n".join(fixed) + "\n"
text = text.replace(
    "    ) -> ExecutionDecision | None:",
    "    ):",
)
text = text.replace(
    '    ):\n        """Cognition-route override',
    '    ):\n        """Cognition-route override',
)
# restore return type using comment only - actually use string annotation
text = text.replace(
    "    def resolve_actor_override(\n        facade,",
    "    def resolve_actor_override(\n        facade,",
)

path.write_text(text, encoding="utf-8")
print(f"Fixed resolve_reuse body lines {sig_end}-{body_end}")
