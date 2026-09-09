#!/usr/bin/env python3
"""
Aggregate CWME experiment results (paper-ready statistics).

Usage:
  python scripts/evaluate_cwme.py [results_dir]
  python scripts/evaluate_cwme.py /mnt/workspace/outputs/e02_cwme_continual_50ep --json

Aggregation conventions (see paper protocol):
  - Performance: mean ± episode_std over episodes (macro, within a single run)
  - Win rate: #success / N episodes
  - Model calls: mean per episode (macro)
  - Reuse / fast-path rates: micro (sum reused / sum executed actions)
  - Multi-seed: mean ± seed_std across seeds (via aggregate_multiseed.py)
"""
from __future__ import annotations

import json
import math
import os
import sys
from glob import glob
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _collect_episode_rows(root: str) -> list[dict]:
    """Prefer episode_metrics.json; fall back to legacy meta globs."""
    candidates = [os.path.join(root, "episode_metrics.json")]
    candidates.extend(
        sorted(glob(os.path.join(root, "log", "experiment_*", "episode_metrics.json")))
    )
    for sub in sorted(glob(os.path.join(root, "log", "experiment_*"))):
        candidates.append(os.path.join(sub, "episode_metrics.json"))

    for metrics_path in candidates:
        data = _load_json(metrics_path)
        if isinstance(data, list) and data:
            return [r for r in data if isinstance(r, dict)]

    rows: list[dict] = []
    for meta_path in glob(os.path.join(root, "**", "*meta*.json"), recursive=True):
        blob = _load_json(meta_path)
        if isinstance(blob, list):
            rows.extend(r for r in blob if isinstance(r, dict))
    if rows:
        rows.sort(key=lambda r: int(r.get("global_episode_id", r.get("episode", 0)) or 0))
        return rows

    for result_path in glob(os.path.join(root, "**", "results*.json"), recursive=True):
        blob = _load_json(result_path)
        if not isinstance(blob, list):
            continue
        for task_block in blob:
            if not isinstance(task_block, dict):
                continue
            task = task_block.get("task", "")
            for meta in task_block.get("meta") or []:
                if isinstance(meta, dict):
                    row = dict(meta)
                    row.setdefault("task", task)
                    rows.append(row)
    rows.sort(key=lambda r: int(r.get("global_episode_id", r.get("episode", 0)) or 0))
    return rows


def _infer_size_label(model_name: str) -> str | None:
    upper = (model_name or "").upper()
    for label in ("27B", "9B", "4B"):
        if label in upper:
            return label.lower()
    return None


def _calls_from_row(row: dict) -> tuple[int, int, int]:
    """Return (4b, 9b, 27b) call counts for one episode row."""
    by_name = row.get("model_calls_by_name") or {}
    if by_name:
        buckets = {"4b": 0, "9b": 0, "27b": 0}
        for name, count in by_name.items():
            size = _infer_size_label(str(name))
            if size in buckets:
                buckets[size] += int(count or 0)
        return buckets["4b"], buckets["9b"], buckets["27b"]
    return (
        int(row.get("model_calls_4b", 0) or 0),
        int(row.get("model_calls_9b", row.get("cognition_calls", 0)) or 0),
        int(row.get("model_calls_27b", 0) or 0),
    )


def _tokens_from_row(row: dict) -> tuple[int, int, int]:
    by_name = row.get("tokens_by_model") or {}
    if by_name:
        buckets = {"4b": 0, "9b": 0, "27b": 0}
        for name, count in by_name.items():
            size = _infer_size_label(str(name))
            if size in buckets:
                buckets[size] += int(count or 0)
        return buckets["4b"], buckets["9b"], buckets["27b"]
    return (
        int(row.get("tokens_4b", 0) or 0),
        int(row.get("tokens_9b", 0) or 0),
        int(row.get("tokens_27b", 0) or 0),
    )


def _mean_model_dict(rows: list[dict], key: str) -> dict[str, float]:
    totals: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        payload = row.get(key) or {}
        if not isinstance(payload, dict):
            continue
        for model, count in payload.items():
            totals[str(model)].append(int(count or 0))
    if not rows:
        return {}
    return {model: round(sum(vals) / len(rows), 2) for model, vals in totals.items()}


def _scan_log_root(root: str) -> dict:
    rows = _collect_episode_rows(root)
    if not rows:
        return {"episodes": 0}

    scores = [int(r.get("max_score", r.get("score", 0)) or 0) for r in rows]
    wins = sum(1 for s in scores if s >= 100)
    steps = [int(r.get("action_count", 0) or 0) for r in rows if r.get("action_count")]
    wall_times = [float(r.get("wall_time_s", 0) or 0) for r in rows if r.get("wall_time_s")]

    calls_4b = []
    calls_9b = []
    calls_27b = []
    tokens_4b = []
    tokens_9b = []
    tokens_27b = []
    cognition_calls = [int(r.get("cognition_calls", 0) or 0) for r in rows]
    perception_calls = [int(r.get("perception_calls", 0) or 0) for r in rows]
    for row in rows:
        c4, c9, c27 = _calls_from_row(row)
        t4, t9, t27 = _tokens_from_row(row)
        calls_4b.append(c4)
        calls_9b.append(c9)
        calls_27b.append(c27)
        tokens_4b.append(t4)
        tokens_9b.append(t9)
        tokens_27b.append(t27)

    total_actions = sum(int(r.get("action_count", 0) or 0) for r in rows)
    total_direct = sum(int(r.get("direct_reuse", 0) or 0) for r in rows)
    total_verified = sum(int(r.get("verified_reuse", 0) or 0) for r in rows)
    total_full = sum(int(r.get("full_cognition", 0) or 0) for r in rows)
    total_verify_accept = sum(int(r.get("verify_accept", 0) or 0) for r in rows)
    total_verify_calls = sum(int(r.get("verify_calls", 0) or 0) for r in rows)
    total_fast_attempts = sum(int(r.get("fast_path_attempts", 0) or 0) for r in rows)
    total_fast_env_success = sum(
        int(r.get("fast_path_env_success", 0) or 0) for r in rows
    )
    total_fast_success = sum(int(r.get("fast_path_success", 0) or 0) for r in rows)

    train_actions = sum(int(r.get("train_pattern_actions", 0) or 0) for r in rows)
    ptr_micro = round(train_actions / total_actions, 4) if total_actions > 0 else 0.0
    reuse_micro = round((total_direct + total_verified) / total_actions, 4) if total_actions > 0 else 0.0
    fast_micro = round((total_direct + total_verified) / total_actions, 4) if total_actions > 0 else 0.0

    fast_macro_vals = [float(r.get("fast_path_ratio", 0) or 0) for r in rows if r.get("fast_path_ratio") is not None]
    ptr_macro_vals = [float(r.get("pattern_transfer_rate", 0) or 0) for r in rows if r.get("pattern_transfer_rate") is not None]

    pattern_sizes = [int(r.get("pattern_lib_size_after", 0) or 0) for r in rows]
    delta_sizes = [int(r.get("delta_pattern_count", 0) or 0) for r in rows]

    mean_score, episode_std_score = _mean_std([float(s) for s in scores])
    mean_steps, episode_std_steps = _mean_std([float(s) for s in steps]) if steps else (0.0, 0.0)
    mean_wall, episode_std_wall = _mean_std(wall_times) if wall_times else (0.0, 0.0)
    auc_score = round(sum(float(s) for s in scores) / len(scores), 2) if scores else 0.0

    decision_sources: dict[str, list[int]] = defaultdict(list)
    cognition_steps = [int(r.get("cognition_decision_steps", 0) or 0) for r in rows]
    total_steps = [int(r.get("total_steps", 0) or r.get("steps", 0) or 0) for r in rows]
    for row in rows:
        counts = row.get("decision_source_counts") or {}
        if isinstance(counts, dict):
            for key, val in counts.items():
                decision_sources[str(key)].append(int(val or 0))

    mean_decision_counts = {
        k: round(sum(v) / len(rows), 2) for k, v in decision_sources.items()
    } if rows else {}
    heur_keys = {
        "heuristic", "sciworld_fast", "architecture_fast",
        "alfworld_fast", "alfworld_fast_override", "fallback",
    }
    ds_total = sum(mean_decision_counts.values())
    heuristic_rate = round(
        sum(mean_decision_counts.get(k, 0) for k in heur_keys) / ds_total, 4
    ) if ds_total > 0 else 0.0
    cognition_decision_rate = round(
        mean_decision_counts.get("cognition", 0) / ds_total, 4
    ) if ds_total > 0 else 0.0
    if cognition_decision_rate == 0.0 and rows:
        step_total = sum(total_steps)
        cognition_decision_rate = round(
            sum(cognition_steps) / step_total, 4
        ) if step_total > 0 else 0.0

    return {
        "episodes": len(rows),
        "mean_score": round(mean_score, 2),
        "auc_score": auc_score,
        "episode_std_score": round(episode_std_score, 2),
        "std_score": round(episode_std_score, 2),  # deprecated alias
        "win_rate": round(wins / len(rows), 4) if rows else 0.0,
        "mean_steps": round(mean_steps, 2),
        "episode_std_steps": round(episode_std_steps, 2),
        "std_steps": round(episode_std_steps, 2),  # deprecated alias
        "mean_wall_time": round(mean_wall, 2),
        "episode_std_wall_time": round(episode_std_wall, 2),
        "std_wall_time": round(episode_std_wall, 2),  # deprecated alias
        "mean_4b_calls": round(sum(calls_4b) / len(rows), 2) if rows else 0.0,
        "mean_9b_calls": round(sum(calls_9b) / len(rows), 2) if rows else 0.0,
        "mean_27b_calls": round(sum(calls_27b) / len(rows), 2) if rows else 0.0,
        "mean_cognition_calls": round(sum(cognition_calls) / len(rows), 2) if rows else 0.0,
        "mean_perception_calls": round(sum(perception_calls) / len(rows), 2) if rows else 0.0,
        "mean_calls_by_model": _mean_model_dict(rows, "model_calls_by_name"),
        "mean_tokens_by_model": _mean_model_dict(rows, "tokens_by_model"),
        "mean_4b_tokens": round(sum(tokens_4b) / len(rows), 2) if rows else 0.0,
        "mean_9b_tokens": round(sum(tokens_9b) / len(rows), 2) if rows else 0.0,
        "mean_27b_tokens": round(sum(tokens_27b) / len(rows), 2) if rows else 0.0,
        "pattern_reuse_rate": reuse_micro,
        "pattern_transfer_rate": ptr_micro,
        "pattern_transfer_rate_macro": round(sum(ptr_macro_vals) / len(ptr_macro_vals), 4) if ptr_macro_vals else ptr_micro,
        "verification_accept_rate": round(total_verify_accept / total_verify_calls, 4) if total_verify_calls > 0 else 0.0,
        "fast_path_rate": fast_micro,
        "fast_path_rate_macro": round(sum(fast_macro_vals) / len(fast_macro_vals), 4) if fast_macro_vals else fast_micro,
        # Success is micro-averaged over attempts.  An environment-accepted
        # no-op is reported separately and is not counted as fast progress.
        "fast_path_attempts": total_fast_attempts,
        "fast_path_env_success": total_fast_env_success,
        "fast_path_success": total_fast_success,
        "fast_path_env_success_rate": round(
            total_fast_env_success / total_fast_attempts, 4
        ) if total_fast_attempts > 0 else 0.0,
        "fast_success_rate": round(
            total_fast_success / total_fast_attempts, 4
        ) if total_fast_attempts > 0 else 0.0,
        "mean_decision_source_counts": mean_decision_counts,
        "heuristic_decision_rate": heuristic_rate,
        "cognition_decision_rate": cognition_decision_rate,
        "novelty_rate": round(total_full / total_actions, 4) if total_actions > 0 else 0.0,
        "final_pattern_size": pattern_sizes[-1] if pattern_sizes else 0,
        "final_delta_pattern_size": delta_sizes[-1] if delta_sizes else 0,
        "final_train_pattern_count": int(rows[-1].get("train_pattern_count", 0) or 0) if rows else 0,
        "final_delta_pattern_count": int(rows[-1].get("delta_pattern_count", 0) or 0) if rows else 0,
        "total_direct_reuse": total_direct,
        "total_verified_reuse": total_verified,
        "total_cognition_calls": sum(calls_9b) + sum(calls_27b),
        "score_per_time": round(mean_score / mean_wall, 4) if mean_wall > 0 else 0.0,
        "score_per_9b_call": round(mean_score / (sum(calls_9b) / len(rows)), 4) if rows and sum(calls_9b) > 0 else 0.0,
    }


def _find_experiment_dirs(base: str) -> list[tuple[str, str]]:
    """Return (display_name, scan_path) pairs."""
    if os.path.isfile(os.path.join(base, "episode_metrics.json")):
        return [(os.path.basename(base), base)]

    seed_dirs = sorted(glob(os.path.join(base, "seed_*")))
    if seed_dirs:
        return [(os.path.basename(sd), sd) for sd in seed_dirs]

    rows: list[tuple[str, str]] = []
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            sub = os.path.join(base, name)
            if os.path.isdir(sub):
                stats_probe = _collect_episode_rows(sub)
                if stats_probe:
                    rows.append((name, sub))
        if not rows:
            if _collect_episode_rows(base):
                rows.append((os.path.basename(base), base))
    return rows


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = set(sys.argv[1:])
    base = args[0] if args else os.path.join(_REPO_ROOT, "outputs")
    if not os.path.isabs(base):
        base = os.path.join(_REPO_ROOT, base)

    exp_dirs = _find_experiment_dirs(base)
    if not exp_dirs:
        print(f"No experiment results found under: {base}")
        print("Run experiments first, then:")
        print("  python scripts/evaluate_cwme.py /mnt/workspace/outputs")
        return

    all_stats: dict[str, dict] = {}
    for name, path in exp_dirs:
        stats = _scan_log_root(path)
        if stats.get("episodes", 0) > 0:
            all_stats[name] = stats

    if "--json" in flags:
        print(json.dumps(all_stats, indent=2))
        return

    print(
        f"{'experiment':<28} {'eps':>5} {'mean±std':>14} {'win%':>7} "
        f"{'reuse':>7} {'fast':>7} {'9B/ep':>7} {'|P|':>5}"
    )
    print("-" * 92)
    for name, s in all_stats.items():
        score_txt = f"{s['mean_score']:.1f}±{s.get('episode_std_score', s.get('std_score', 0)):.1f}"
        print(
            f"{name:<28} {s['episodes']:>5} {score_txt:>14} "
            f"{100*s['win_rate']:>6.1f}% {100*s['pattern_reuse_rate']:>6.1f}% "
            f"{100*s['fast_path_rate']:>6.1f}% {s['mean_9b_calls']:>7.1f} "
            f"{s['final_pattern_size']:>5}"
        )


if __name__ == "__main__":
    main()
