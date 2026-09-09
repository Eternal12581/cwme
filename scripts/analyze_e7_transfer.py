#!/usr/bin/env python3
"""
E7 cross-variation transfer analysis: Transfer Gain (TG) and Pattern Transfer Rate (PTR).

Usage:
  python scripts/analyze_e7_transfer.py \\
    /mnt/workspace/outputs/e07_crossvar_test_cwme_v45 \\
    /mnt/workspace/outputs/e07_crossvar_test_cloff_v45

  # Pure transfer (E7-A):
  python scripts/analyze_e7_transfer.py \\
    /mnt/workspace/outputs/e07_crossvar_static_cwme_v45 \\
    /mnt/workspace/outputs/e07_crossvar_test_cloff_v45

Reports:
  TG = Score_CWME(unseen) - Score_Backbone(unseen)
  PTR = # actions from P_train / # executable actions
  Per-variation breakdown: v4 / v5 scores, win rates, PTR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

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
_collect_episode_rows = _eval_mod._collect_episode_rows


def _load_multiseed_or_single(base: str) -> dict:
    ms_path = os.path.join(base, "multiseed_summary.json")
    if os.path.isfile(ms_path):
        with open(ms_path, encoding="utf-8") as fh:
            blob = json.load(fh)
        return blob.get("aggregated") or blob.get("per_seed", {}).get("seed_4901", {})

    seed_dirs = sorted(
        d for d in os.listdir(base)
        if d.startswith("seed_") and os.path.isdir(os.path.join(base, d))
    ) if os.path.isdir(base) else []
    if seed_dirs:
        vals = [_scan_log_root(os.path.join(base, sd)) for sd in seed_dirs]
        vals = [v for v in vals if v.get("episodes")]
        if vals:
            keys = ["mean_score", "win_rate", "pattern_transfer_rate", "pattern_reuse_rate"]
            return {k: sum(v.get(k, 0) for v in vals) / len(vals) for k in keys}
    return _scan_log_root(base)


def _variation_stats(base: str) -> dict[str, dict]:
    """Per-variation score / win / PTR from episode rows."""
    rows = _collect_episode_rows(base)
    if not rows:
        seed_dirs = sorted(
            d for d in os.listdir(base)
            if d.startswith("seed_") and os.path.isdir(os.path.join(base, d))
        ) if os.path.isdir(base) else []
        for sd in seed_dirs:
            rows.extend(_collect_episode_rows(os.path.join(base, sd)))
    if not rows:
        return {}

    by_var: dict[str, list[dict]] = {}
    for row in rows:
        var = str(row.get("variation_id", row.get("variation", "")) or "unknown").strip()
        by_var.setdefault(var, []).append(row)

    out: dict[str, dict] = {}
    for var, var_rows in sorted(by_var.items()):
        scores = [int(r.get("max_score", r.get("score", 0)) or 0) for r in var_rows]
        wins = sum(1 for s in scores if s >= 100)
        total_actions = sum(int(r.get("action_count", 0) or 0) for r in var_rows)
        train_actions = sum(int(r.get("train_pattern_actions", 0) or 0) for r in var_rows)
        ptr = round(train_actions / total_actions, 4) if total_actions > 0 else 0.0
        out[var] = {
            "episodes": len(var_rows),
            "mean_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "win_rate": round(wins / len(var_rows), 4) if var_rows else 0.0,
            "pattern_transfer_rate": ptr,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="E7 transfer gain analysis")
    parser.add_argument("cwme_dir", help="CWME test output (unseen v4/v5)")
    parser.add_argument("backbone_dir", help="Backbone (no-memory) test output (unseen v4/v5)")
    parser.add_argument("--json-out", default="", help="Optional JSON output path")
    parser.add_argument(
        "--require-pure-transfer",
        action="store_true",
        help="Fail if final_delta_pattern_count != 0 (E7-A sanity)",
    )
    args = parser.parse_args()

    cwme = _load_multiseed_or_single(args.cwme_dir)
    backbone = _load_multiseed_or_single(args.backbone_dir)

    if not cwme.get("episodes") and not backbone.get("episodes"):
        raise SystemExit("No episode data found in either directory.")

    tg_score = float(cwme.get("mean_score", 0)) - float(backbone.get("mean_score", 0))
    tg_win = float(cwme.get("win_rate", 0)) - float(backbone.get("win_rate", 0))
    ptr = float(cwme.get("pattern_transfer_rate", 0))

    cwme_by_var = _variation_stats(args.cwme_dir)
    backbone_by_var = _variation_stats(args.backbone_dir)

    per_variation: dict[str, dict] = {}
    all_vars = sorted(set(cwme_by_var) | set(backbone_by_var))
    for var in all_vars:
        c = cwme_by_var.get(var, {})
        b = backbone_by_var.get(var, {})
        per_variation[var] = {
            "cwme_mean_score": c.get("mean_score"),
            "cwme_win_rate": c.get("win_rate"),
            "cwme_PTR": c.get("pattern_transfer_rate"),
            "backbone_mean_score": b.get("mean_score"),
            "backbone_win_rate": b.get("win_rate"),
            "transfer_gain_score": round(
                float(c.get("mean_score", 0) or 0) - float(b.get("mean_score", 0) or 0), 2
            ),
        }

    report = {
        "cwme": {
            "mean_score": cwme.get("mean_score"),
            "win_rate": cwme.get("win_rate"),
            "pattern_transfer_rate": cwme.get("pattern_transfer_rate"),
            "pattern_reuse_rate": cwme.get("pattern_reuse_rate"),
            "final_train_pattern_count": cwme.get("final_train_pattern_count"),
            "final_delta_pattern_count": cwme.get("final_delta_pattern_count"),
            "episodes": cwme.get("episodes"),
        },
        "backbone": {
            "mean_score": backbone.get("mean_score"),
            "win_rate": backbone.get("win_rate"),
            "episodes": backbone.get("episodes"),
        },
        "transfer_gain": {
            "delta_score": round(tg_score, 2),
            "delta_win_rate": round(tg_win, 4),
            "pattern_transfer_rate": round(ptr, 4),
        },
        "per_variation": per_variation,
    }

    print("=== E7 Cross-Variation Transfer ===")
    print(f"CWME     mean_score={cwme.get('mean_score')} win_rate={100*cwme.get('win_rate', 0):.1f}%")
    print(f"Backbone mean_score={backbone.get('mean_score')} win_rate={100*backbone.get('win_rate', 0):.1f}%")
    print(f"Transfer Gain (ΔScore): {tg_score:+.2f}")
    print(f"Transfer Gain (ΔWin%):  {100*tg_win:+.1f}%")
    print(f"Pattern Transfer Rate:  {100*ptr:.1f}%")
    delta_n = cwme.get("final_delta_pattern_count")
    if delta_n is not None:
        print(f"Final ΔP patterns:      {delta_n} (E7-A pure transfer should be 0)")
        if args.require_pure_transfer and int(delta_n or 0) != 0:
            raise SystemExit(
                f"E7-A pure transfer check failed: final_delta_pattern_count={delta_n} (expected 0)"
            )
    for var, stats in per_variation.items():
        print(
            f"  var {var}: CWME={stats['cwme_mean_score']} "
            f"Backbone={stats['backbone_mean_score']} "
            f"TG={stats['transfer_gain_score']:+.1f} "
            f"PTR={100*(stats.get('cwme_PTR') or 0):.1f}%"
        )

    out = args.json_out or os.path.join(args.cwme_dir, "e7_transfer_report.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
