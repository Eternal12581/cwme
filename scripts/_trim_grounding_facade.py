"""Remove legacy blocks from grounding_facade.py after migration."""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "core" / "grounding_facade.py"
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

# Delete 0-indexed ranges (reverse order): helpers 67-396, methods 495-924, resolve inner 990-1506
# After first deletion indices shift - delete from bottom to top
ranges = [(990, 1506), (495, 924), (67, 396)]
for start, end in ranges:
    del lines[start:end]

# Insert delegation in resolve_reuse after mem_fast block
text = "".join(lines)
needle = "        if not legacy_reuse_heuristics_enabled(agent):\n            return None\n"
replacement = needle + (
    "\n        return LegacyReusePolicy.resolve_reuse(\n"
    "            self,\n"
    "            orchestrator,\n"
    "            agent,\n"
    "            planned,\n"
    "            env_valid,\n"
    "            ctx_text,\n"
    "            route=route,\n"
    "            planned_norm=planned_norm,\n"
    "            current_score=current_score,\n"
    "            past=past,\n"
    "            recent=recent,\n"
    "            failed=failed,\n"
    "            match_kind=match_kind,\n"
    "            raw_ms=raw_ms,\n"
    "            steps_sg=steps_sg,\n"
    "            task_for_cl=task_for_cl,\n"
    "            read_on=read_on,\n"
    "            score=score,\n"
    "            logger=logger,\n"
    "            log_prefix=log_prefix,\n"
    "        )\n"
)
if needle not in text:
    raise SystemExit("delegation anchor not found")
text = text.replace(needle, replacement, 1)

# Add import for LegacyReusePolicy in resolve_reuse_action (lazy) - already in replacement via class name
# Add lazy import at top of resolve_reuse_action after legacy check - use inline import
text = text.replace(
    "        return LegacyReusePolicy.resolve_reuse(",
    "        from core.reuse_policy_legacy import LegacyReusePolicy\n\n"
    "        return LegacyReusePolicy.resolve_reuse(",
    1,
)

# Re-export _pattern_force_repeat_ok from legacy for any facade users - keep in facade
# Add _pattern_force_repeat_ok import from legacy for cl_protocol - update cl_protocol separately

# Remove duplicate resolve_reuse_action signature content if any broken
path.write_text(text, encoding="utf-8")
print("Updated grounding_facade.py, lines:", len(text.splitlines()))
