#!/usr/bin/env python3
"""
Aggregate CWME results across seed_* subdirectories (mean ± seed_std).

Usage:
  python scripts/aggregate_multiseed.py /mnt/workspace/outputs/e01_cwme_sci_main
  python scripts/aggregate_multiseed.py /mnt/workspace/outputs/e01_cwme_sci_main --json

Statistical convention:
  - episode_std_* : std across episodes within one seed run (from evaluate_cwme.py)
  - seed_std_*    : std across random seeds of the per-seed mean
"""
from __future__ import annotations

import json
import math
import os
import sys
from glob import glob

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import importlib.util

_eval_path = os.path.join(_REPO_ROOT, "scripts", "evaluate_cwme.py")
_spec = importlib.util.spec_from_file_location("evaluate_cwme", _eval_path)
_eval_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_eval_mod)
_scan_log_root = _eval_mod._scan_log_root


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


# (aggregate_key, per_seed_source_key, is_pct)
PAPER_METRICS = [
    ("mean_score", "mean_score", False),
    ("auc_score", "auc_score", False),
    ("win_rate", "win_rate", True),
    ("mean_steps", "mean_steps", False),
    ("mean_wall_time", "mean_wall_time", False),
    ("mean_cognition_calls", "mean_cognition_calls", False),
    ("mean_perception_calls", "mean_perception_calls", False),
    ("mean_9b_calls", "mean_9b_calls", False),
    ("mean_4b_calls", "mean_4b_calls", False),
    ("mean_27b_calls", "mean_27b_calls", False),
    ("mean_9b_tokens", "mean_9b_tokens", False),
    ("mean_27b_tokens", "mean_27b_tokens", False),
    ("pattern_reuse_rate", "pattern_reuse_rate", True),
    ("pattern_transfer_rate", "pattern_transfer_rate", True),
    ("fast_path_rate", "fast_path_rate", True),
    ("fast_path_attempts", "fast_path_attempts", False),
    ("fast_path_env_success_rate", "fast_path_env_success_rate", True),
    ("fast_success_rate", "fast_success_rate", True),
    ("heuristic_decision_rate", "heuristic_decision_rate", True),
    ("cognition_decision_rate", "cognition_decision_rate", True),
    ("novelty_rate", "novelty_rate", True),
    ("verification_accept_rate", "verification_accept_rate", True),
    ("final_pattern_size", "final_pattern_size", False),
    ("score_per_time", "score_per_time", False),
    ("score_per_9b_call", "score_per_9b_call", False),
]

EPISODE_STD_METRICS = [
    ("mean_score", "episode_std_score"),
    ("mean_steps", "episode_std_steps"),
    ("mean_wall_time", "episode_std_wall_time"),
]


def aggregate_seeds(base: str) -> dict:
    seed_dirs = sorted(glob(os.path.join(base, "seed_*")))
    if not seed_dirs:
        stats = _scan_log_root(base)
        return {"seeds": 1, "per_seed": {"single": stats}, "aggregated": stats}

    per_seed: dict[str, dict] = {}
    for sd in seed_dirs:
        stats = _scan_log_root(sd)
        if stats.get("episodes", 0) > 0:
            per_seed[os.path.basename(sd)] = stats

    if not per_seed:
        return {"seeds": 0, "per_seed": {}, "aggregated": {}}

    aggregated: dict = {"seeds": len(per_seed)}
    for key, src_key, is_pct in PAPER_METRICS:
        vals = [float(r.get(src_key, 0) or 0) for r in per_seed.values()]
        mean, seed_std = _mean_std(vals)
        aggregated[key] = round(mean, 4) if is_pct else round(mean, 2)
        aggregated[f"seed_std_{key}"] = round(seed_std, 4) if is_pct else round(seed_std, 2)

    for mean_key, ep_std_key in EPISODE_STD_METRICS:
        ep_stds = [
            float(r.get(ep_std_key, r.get(f"std_{mean_key.replace('mean_', '')}", 0)) or 0)
            for r in per_seed.values()
        ]
        aggregated[f"episode_std_{mean_key.replace('mean_', '')}"] = round(
            sum(ep_stds) / len(ep_stds), 2
        )

    return {"seeds": len(per_seed), "per_seed": per_seed, "aggregated": aggregated}


def main() -> None:
    flags = set(sys.argv[1:])
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    base = args[0] if args else os.path.join(_REPO_ROOT, "outputs")
    if not os.path.isabs(base):
        base = os.path.join(_REPO_ROOT, base)

    result = aggregate_seeds(base)
    if result["seeds"] == 0:
        print(f"No episode data in seed dirs under {base}")
        return

    out_json = os.path.join(base, "multiseed_summary.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved -> {out_json}")

    if "--json" in flags:
        print(json.dumps(result, indent=2))
        return

    agg = result["aggregated"]
    print(f"Experiment base: {base}")
    print(f"Seeds aggregated: {result['seeds']}")
    print(f"{'metric':<28} {'mean':>12} {'seed_std':>12}")
    print("-" * 54)
    for key, _, is_pct in PAPER_METRICS:
        mean = agg.get(key, 0)
        seed_std = agg.get(f"seed_std_{key}", 0)
        label = key
        if is_pct:
            print(f"{label:<28} {100*mean:>11.2f}% {100*seed_std:>11.2f}%")
        else:
            print(f"{label:<28} {mean:>12.2f} {seed_std:>12.2f}")

    ep_score_std = agg.get("episode_std_score")
    if ep_score_std is not None:
        print(f"{'episode_std_score (avg)':<28} {ep_score_std:>12.2f}")


if __name__ == "__main__":
    main()
