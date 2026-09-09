#!/usr/bin/env python3
"""
Artifact statistics for H_train / P_train (paper appendix & sanity checks).

Usage:
  python scripts/analyze_artifacts.py
  python scripts/analyze_artifacts.py --hkg artifacts/historical_kg/historical_world_model.json
  python scripts/analyze_artifacts.py --patterns artifacts/procedural_memory/patterns.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.procedural.pattern_normalizer import task_signature


def _load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze_hkg(path: str) -> dict:
    if not os.path.isfile(path):
        return {"path": path, "error": "file not found"}

    data = _load_json(path)
    edges = data.get("edges") or data.get("triples") or []
    if isinstance(edges, dict):
        edges = list(edges.values())

    tasks = set()
    edge_types = Counter()
    success_episodes = 0
    partial_episodes = 0
    fail_episodes = 0
    meta = data.get("metadata") or data.get("meta") or {}
    split_meta = meta.get("split") if isinstance(meta, dict) else None
    metadata_has_outcomes = False
    if isinstance(meta, dict):
        for key in ("task_ids", "tasks", "task_names"):
            tasks.update(str(t) for t in (meta.get(key) or []) if t)
        metadata_has_outcomes = any(
            key in meta
            for key in ("success_episodes", "partial_ge50_episodes", "failure_episodes")
        )
    if isinstance(split_meta, dict):
        for key in ("task_ids", "tasks", "task_names"):
            tasks.update(str(t) for t in (split_meta.get(key) or []) if t)
        split_has_outcomes = any(
            key in split_meta
            for key in ("success_episodes", "partial_ge50_episodes", "failure_episodes")
        )
        metadata_has_outcomes = metadata_has_outcomes or split_has_outcomes
        # Older HKG builders kept these aggregates only under ``split``.
        if "success_episodes" not in meta:
            success_episodes = int(split_meta.get("success_episodes", 0) or 0)
        if "partial_ge50_episodes" not in meta:
            partial_episodes = int(split_meta.get("partial_ge50_episodes", 0) or 0)
        if "failure_episodes" not in meta:
            fail_episodes = int(split_meta.get("failure_episodes", 0) or 0)

    episode_rows = data.get("episodes") or data.get("source_episodes") or []
    # Prefer authoritative builder aggregates when present. Otherwise derive
    # them from embedded episode rows. This avoids double-counting artifacts
    # that contain both metadata and full source episodes.
    derive_outcomes = not metadata_has_outcomes
    for ep in episode_rows:
        score = int(ep.get("final_score", ep.get("score", 0)) or 0)
        task = str(ep.get("task", ep.get("task_id", "")) or "")
        if task:
            tasks.add(task)
        if derive_outcomes:
            if ep.get("success") or score >= 100:
                success_episodes += 1
            elif score >= 50:
                partial_episodes += 1
            else:
                fail_episodes += 1

    for edge in edges:
        if isinstance(edge, dict):
            et = edge.get("relation") or edge.get("type") or edge.get("predicate") or "unknown"
            edge_types[str(et)] += 1
            for key in ("task", "task_id", "source_task"):
                t = edge.get(key)
                if t:
                    tasks.add(str(t))

    return {
        "path": path,
        "edges": len(edges),
        "nodes": len(data.get("nodes") or []),
        "tasks": len(tasks),
        "task_list": sorted(tasks)[:20],
        "edge_types_top10": edge_types.most_common(10),
        "total_training_episodes": (
            meta.get("episode_count")
            or meta.get("source_episodes")
            or (split_meta or {}).get("num_episodes")
            or len(data.get("episodes") or data.get("source_episodes") or [])
        ),
        "source_episodes": (
            meta.get("source_episodes")
            or (split_meta or {}).get("num_episodes")
            or meta.get("episode_count")
            or len(data.get("episodes") or [])
        ),
        "success_episodes": meta.get("success_episodes", success_episodes),
        "partial_ge50_episodes": meta.get("partial_ge50_episodes", partial_episodes),
        "failure_episodes": meta.get("failure_episodes", fail_episodes),
        "metadata": {k: meta[k] for k in list(meta.keys())[:10]},
    }


def analyze_patterns(path: str) -> dict:
    if not os.path.isfile(path):
        return {"path": path, "error": "file not found"}

    rows = _load_jsonl(path)
    signatures: Counter[str] = Counter()
    task_ids: set[str] = set()
    cross_task_sigs: dict[str, set[str]] = defaultdict(set)
    confidences: list[float] = []
    lengths: list[int] = []
    success_patterns = 0
    partial_patterns = 0

    for row in rows:
        task = str(row.get("task_id") or row.get("task") or "")
        sig = str(row.get("signature") or row.get("task_signature") or "")
        if not sig and task:
            sig = task_signature(task)
        if task:
            task_ids.add(task)
        if sig:
            signatures[sig] += 1
            if task:
                cross_task_sigs[sig].add(task)

        score = int(row.get("source_score", row.get("score", 0)) or 0)
        if score >= 100 or row.get("success"):
            success_patterns += 1
        elif score >= 50:
            partial_patterns += 1

        conf = row.get("confidence")
        if conf is not None:
            confidences.append(float(conf))
        actions = row.get("actions") or row.get("pattern") or []
        if isinstance(actions, list):
            lengths.append(len(actions))

    shared_sigs = sum(1 for tasks in cross_task_sigs.values() if len(tasks) > 1)
    cross_var_patterns = sum(
        1 for row in rows
        if len(str(row.get("variation_id", "") or "").strip()) > 0
        or str(row.get("cross_variation", "")).lower() in ("true", "1", "yes")
    )

    return {
        "path": path,
        "patterns": len(rows),
        "unique_signatures": len(signatures),
        "unique_tasks": len(task_ids),
        "cross_variation_signatures": shared_sigs,
        "cross_variation_patterns": cross_var_patterns,
        "success_patterns": success_patterns,
        "partial_patterns": partial_patterns,
        "failure_patterns": len(rows) - success_patterns - partial_patterns,
        "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        "avg_pattern_length": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "top_signatures": signatures.most_common(10),
    }


def _print_section(title: str, stats: dict) -> None:
    print(f"\n=== {title} ===")
    if stats.get("error"):
        print(f"  ERROR: {stats['error']} ({stats.get('path')})")
        return
    for key, val in stats.items():
        if key in ("path", "task_list", "top_signatures", "edge_types_top10", "metadata"):
            continue
        print(f"  {key}: {val}")
    if stats.get("edge_types_top10"):
        print(f"  edge_types_top10: {stats['edge_types_top10']}")
    if stats.get("top_signatures"):
        print(f"  top_signatures: {stats['top_signatures']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="H_train / P_train artifact statistics")
    parser.add_argument(
        "--hkg",
        default=os.path.join(_REPO_ROOT, "artifacts", "historical_kg", "historical_world_model.json"),
    )
    parser.add_argument(
        "--patterns",
        default=os.path.join(_REPO_ROOT, "artifacts", "procedural_memory", "patterns.jsonl"),
    )
    parser.add_argument(
        "--schedule",
        default=os.path.join(_REPO_ROOT, "artifacts", "schedules", "e02_continual_50ep.json"),
    )
    args = parser.parse_args()

    from core.experiment_protocol import _sha256_file

    hkg_stats = analyze_hkg(args.hkg)
    pat_path = args.patterns
    if not pat_path.endswith(".procedural.jsonl") and os.path.isfile(f"{pat_path}.procedural.jsonl"):
        pat_path = f"{pat_path}.procedural.jsonl"
    pat_stats = analyze_patterns(pat_path if os.path.isfile(pat_path) else args.patterns)
    _print_section("Historical KG (H_train)", hkg_stats)
    _print_section("Pattern Library (P_train)", pat_stats)

    manifest = {
        "h_train": {"path": args.hkg, "sha256": _sha256_file(args.hkg)},
        "p_train": {
            "path": pat_stats.get("path", args.patterns),
            "sha256": _sha256_file(pat_stats.get("path", "")),
        },
        "e02_schedule": {"path": args.schedule, "sha256": _sha256_file(args.schedule)},
        "artifacts": {
            "hkg": _sha256_file(args.hkg),
            "pattern": _sha256_file(pat_stats.get("path", "")),
            "schedule": _sha256_file(args.schedule),
        },
    }

    out = {"historical_kg": hkg_stats, "pattern_library": pat_stats, "manifest": manifest}
    out_path = os.path.join(_REPO_ROOT, "artifacts", "artifact_statistics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved -> {out_path}")

    manifest_path = os.path.join(_REPO_ROOT, "artifacts", "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"Saved -> {manifest_path}")


if __name__ == "__main__":
    main()
