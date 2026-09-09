#!/usr/bin/env python3
"""
Paired E2 continual-curve analysis: Backbone / Static-PM / CWME.

Usage:
  python scripts/paired_episode_analysis.py \\
    /mnt/workspace/outputs/e02_cwme_continual_50ep \\
    /mnt/workspace/outputs/e02_backbone_continual_50ep \\
    --static-pm-dir /mnt/workspace/outputs/e02_static_pm_50ep

Outputs:
  - per-episode paired delta (score, reuse, 9B calls)
  - first vs later episode summary + continual gain (CG)
  - pattern_reuse vs score correlation (association, not causal)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from glob import glob

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_curve(base: str) -> list[dict]:
    for sub in sorted(glob(os.path.join(base, "log", "experiment_*"))):
        path = os.path.join(sub, "continual_curve.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
            return list(blob.get("episodes") or [])
    path = os.path.join(base, "continual_curve.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        return list(blob.get("episodes") or [])

    import importlib.util
    eval_path = os.path.join(_REPO_ROOT, "scripts", "evaluate_cwme.py")
    spec = importlib.util.spec_from_file_location("evaluate_cwme", eval_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    rows = mod._collect_episode_rows(base)
    return [
        {
            "global_episode_id": int(r.get("global_episode_id", 0) or 0),
            "score": int(r.get("max_score", 0) or 0),
            "pattern_size": int(r.get("pattern_lib_size_after", 0) or 0),
            "reuse_rate": float(r.get("pattern_reuse_rate", 0) or 0),
            "cognition_9b_calls": int(r.get("model_calls_9b", r.get("cognition_calls", 0)) or 0),
        }
        for r in rows
    ]


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return round(num / (den_x * den_y), 4)


def _first_vs_later(curve: list[dict], *, later_from: int = 5) -> dict:
    if not curve:
        return {}
    first = curve[0].get("score", 0)
    later = [c.get("score", 0) for c in curve[later_from - 1:] if c.get("score") is not None]
    later_mean = sum(later) / len(later) if later else 0
    ep10 = curve[9].get("score", 0) if len(curve) >= 10 else None
    delta = round(later_mean - first, 2)
    auc = round(sum(float(c.get("score", 0) or 0) for c in curve) / len(curve), 2)
    return {
        "first_episode_score": first,
        "episode_10_score": ep10,
        "later_mean_score_from_ep5": round(later_mean, 2),
        "delta_later_minus_first": delta,
        "continual_gain": delta,
        "auc_score": auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired E2 continual analysis")
    parser.add_argument("cwme_dir", help="CWME continual curve output")
    parser.add_argument("backbone_dir", help="Backbone (no-memory) continual curve output")
    parser.add_argument("--static-pm-dir", default="", help="Optional Static-PM curve output")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    cwme = _load_curve(args.cwme_dir)
    backbone = _load_curve(args.backbone_dir)
    static_pm = _load_curve(args.static_pm_dir) if args.static_pm_dir else []
    n = min(len(cwme), len(backbone))
    if static_pm:
        n = min(n, len(static_pm))
    if n == 0:
        raise SystemExit("No continual curve data found.")

    paired = []
    for i in range(n):
        c = cwme[i]
        b = backbone[i]
        entry = {
            "episode": i,
            "cwme_score": c.get("score", 0),
            "backbone_score": b.get("score", 0),
            "delta_cwme_minus_backbone": int(c.get("score", 0)) - int(b.get("score", 0)),
            "cwme_reuse_rate": c.get("reuse_rate", 0),
            "cwme_pattern_size": c.get("pattern_size", 0),
            "cwme_9b_calls": c.get("cognition_9b_calls", 0),
            "backbone_9b_calls": b.get("cognition_9b_calls", 0),
        }
        if static_pm:
            s = static_pm[i]
            entry["static_pm_score"] = s.get("score", 0)
            entry["delta_cwme_minus_static_pm"] = int(c.get("score", 0)) - int(s.get("score", 0))
        paired.append(entry)

    reuse_rates = [float(c.get("reuse_rate", 0) or 0) for c in cwme[:n]]
    scores = [float(c.get("score", 0) or 0) for c in cwme[:n]]
    corr = _pearson(reuse_rates, scores)

    cwme_fvl = _first_vs_later(cwme)
    backbone_fvl = _first_vs_later(backbone)
    static_fvl = _first_vs_later(static_pm) if static_pm else {}

    report = {
        "num_paired_episodes": n,
        "cwme_first_vs_later": cwme_fvl,
        "backbone_first_vs_later": backbone_fvl,
        "static_pm_first_vs_later": static_fvl,
        "continual_gain": {
            "cwme": cwme_fvl.get("continual_gain", 0),
            "backbone": backbone_fvl.get("continual_gain", 0),
            "static_pm": static_fvl.get("continual_gain", 0),
            "cwme_auc": cwme_fvl.get("auc_score", 0),
            "static_pm_auc": static_fvl.get("auc_score", 0) if static_fvl else None,
            "delta_continual_cwme_minus_static_pm": round(
                float(cwme_fvl.get("continual_gain", 0) or 0)
                - float(static_fvl.get("continual_gain", 0) or 0),
                2,
            ) if static_fvl else None,
        },
        "mean_delta_score_cwme_minus_backbone": round(
            sum(p["delta_cwme_minus_backbone"] for p in paired) / n, 2
        ),
        "mean_delta_score_cwme_minus_static_pm": round(
            sum(p.get("delta_cwme_minus_static_pm", 0) for p in paired) / n, 2
        ) if static_pm else None,
        "reuse_score_correlation_cwme": corr,
        "paired_episodes": paired,
    }

    print("=== E2 Paired Continual Analysis ===")
    print(f"Paired episodes: {n}")
    print(f"Mean ΔScore (CWME - Backbone): {report['mean_delta_score_cwme_minus_backbone']:+.2f}")
    if static_pm:
        print(f"Mean ΔScore (CWME - Static-PM): {report['mean_delta_score_cwme_minus_static_pm']:+.2f}")
    print(f"CWME  first→later (CG): {report['cwme_first_vs_later']}")
    if static_pm:
        print(f"Static-PM first→later (CG): {report['static_pm_first_vs_later']}")
        print(f"ΔContinual (CWME - Static-PM): {report['continual_gain']['delta_continual_cwme_minus_static_pm']}")
    print(f"Backbone first→later: {report['backbone_first_vs_later']}")
    print(f"CWME reuse↔score Pearson r (association): {corr}")

    out = args.json_out or os.path.join(args.cwme_dir, "e2_paired_analysis.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
