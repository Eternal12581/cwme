#!/usr/bin/env python3
"""
E5-B efficiency metrics including Large-Model Saving.

Usage:
  python scripts/compute_efficiency_metrics.py \\
    /mnt/workspace/outputs/e05_hetero_4b9b \\
    /mnt/workspace/outputs/e05_hetero_all9b

Large-Model Saving = 1 - N_9B(4B+9B) / N_9B(9B+9B)
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


def _load(base: str) -> dict:
    ms = os.path.join(base, "multiseed_summary.json")
    if os.path.isfile(ms):
        with open(ms, encoding="utf-8") as fh:
            blob = json.load(fh)
        return blob.get("aggregated") or {}
    return _scan_log_root(base)


def main() -> None:
    parser = argparse.ArgumentParser(description="E5-B efficiency comparison")
    parser.add_argument("hetero_dir", help="4B+9B output")
    parser.add_argument("homo_dir", help="9B+9B output")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    hetero = _load(args.hetero_dir)
    homo = _load(args.homo_dir)

    n9b_hetero = float(hetero.get("mean_9b_calls", 0) or 0)
    n9b_homo = float(homo.get("mean_9b_calls", 0) or 0)
    saving = round(1.0 - n9b_hetero / n9b_homo, 4) if n9b_homo > 0 else 0.0

    report = {
        "hetero_4b9b": {
            "mean_score": hetero.get("mean_score"),
            "mean_wall_time": hetero.get("mean_wall_time"),
            "mean_9b_calls": n9b_hetero,
            "score_per_time": hetero.get("score_per_time"),
        },
        "homo_9b9b": {
            "mean_score": homo.get("mean_score"),
            "mean_wall_time": homo.get("mean_wall_time"),
            "mean_9b_calls": n9b_homo,
            "score_per_time": homo.get("score_per_time"),
        },
        "large_model_saving": saving,
        "large_model_saving_pct": round(100 * saving, 1),
    }

    print("=== E5-B Efficiency ===")
    print(f"4B+9B  score={hetero.get('mean_score')} 9B/ep={n9b_hetero:.1f} time={hetero.get('mean_wall_time')}s")
    print(f"9B+9B  score={homo.get('mean_score')} 9B/ep={n9b_homo:.1f} time={homo.get('mean_wall_time')}s")
    print(f"Large-Model Saving: {100*saving:.1f}%")

    out = args.json_out or os.path.join(args.hetero_dir, "e5b_efficiency_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
