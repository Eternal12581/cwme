#!/usr/bin/env python3
"""
Compare S1/S2/S3 sanity runs and assert architecture invariants.

Usage:
  python scripts/compare_sanity.py \\
    /mnt/workspace/outputs/sanity_s1_empty_pattern \\
    /mnt/workspace/outputs/sanity_s2_full_pattern \\
    /mnt/workspace/outputs/sanity_s3_empty_hkg
"""
from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import importlib.util

_eval_path = os.path.join(_REPO_ROOT, "scripts", "evaluate_cwme.py")
_spec = importlib.util.spec_from_file_location("evaluate_cwme", _eval_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
_scan = _mod._scan_log_root


def _load(base: str) -> dict:
    if not os.path.isabs(base):
        base = os.path.join(_REPO_ROOT, base)
    return _scan(base)


def _cognition_share(stats: dict) -> float:
    counts = stats.get("mean_decision_source_counts") or {}
    if not counts:
        return 0.0
    total = sum(float(v) for v in counts.values())
    if total <= 0:
        return 0.0
    cog = float(counts.get("cognition", 0) or 0)
    return cog / total


def _heuristic_share(stats: dict) -> float:
    counts = stats.get("mean_decision_source_counts") or {}
    if not counts:
        return 0.0
    total = sum(float(v) for v in counts.values())
    if total <= 0:
        return 0.0
    heur_keys = {
        "heuristic", "sciworld_fast", "architecture_fast",
        "alfworld_fast", "alfworld_fast_override", "fallback",
    }
    heur = sum(float(counts.get(k, 0) or 0) for k in heur_keys)
    return heur / total


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: compare_sanity.py <s1_dir> <s2_dir> [s3_dir]")
        sys.exit(2)

    s1 = _load(args[0])
    s2 = _load(args[1])
    s3 = _load(args[2]) if len(args) > 2 else {}

    print("=== Sanity Architecture Check ===")
    print(f"S1 empty pattern: mean_9b={s1.get('mean_9b_calls')} mean_score={s1.get('mean_score')} cognition_share={_cognition_share(s1):.2%} heuristic_share={_heuristic_share(s1):.2%}")
    print(f"S2 full pattern:  mean_9b={s2.get('mean_9b_calls')} mean_score={s2.get('mean_score')} cognition_share={_cognition_share(s2):.2%} heuristic_share={_heuristic_share(s2):.2%}")
    if s3:
        print(f"S3 empty hkg:    mean_9b={s3.get('mean_9b_calls')} mean_score={s3.get('mean_score')}")

    failures = []
    n9b_s1 = float(s1.get("mean_9b_calls", 0) or 0)
    n9b_s2 = float(s2.get("mean_9b_calls", 0) or 0)
    if n9b_s1 <= n9b_s2:
        failures.append(f"S1 9B calls ({n9b_s1}) should exceed S2 ({n9b_s2})")
    if _cognition_share(s1) < 0.05:
        failures.append(f"S1 cognition decision share too low ({_cognition_share(s1):.2%})")
    if _heuristic_share(s1) > 0.10:
        failures.append(f"S1 heuristic decision share too high ({_heuristic_share(s1):.2%})")
    if _heuristic_share(s2) > 0.05:
        failures.append(f"S2 heuristic decision share too high ({_heuristic_share(s2):.2%})")
    if s3 and float(s3.get("mean_score", 0) or 0) > float(s2.get("mean_score", 0) or 0) + 5:
        failures.append("S3 empty HKG scored unexpectedly higher than S2")

    report = {
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "checks": {
            "s1_9b_gt_s2": n9b_s1 > n9b_s2,
            "s1_cognition_share": round(_cognition_share(s1), 4),
            "s2_cognition_share": round(_cognition_share(s2), 4),
            "s1_heuristic_share": round(_heuristic_share(s1), 4),
            "s2_heuristic_share": round(_heuristic_share(s2), 4),
        },
        "passed": not failures,
        "failures": failures,
    }
    out = os.path.join(args[0] if os.path.isabs(args[0]) else os.path.join(_REPO_ROOT, args[0]), "..", "sanity_compare.json")
    out = os.path.normpath(out)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved -> {out}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
