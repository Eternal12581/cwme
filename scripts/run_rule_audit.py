#!/usr/bin/env python3
"""
Emit CWME rule / heuristic audit table for paper appendix.

Usage:
  python scripts/run_rule_audit.py [--json output.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.world_model.rule_audit import audit_rows, format_audit_table


def main() -> None:
    parser = argparse.ArgumentParser(description="CWME rule audit report")
    parser.add_argument("--json", dest="json_path", default="", help="Optional JSON output path")
    args = parser.parse_args()

    table = format_audit_table()
    print(table)

    if args.json_path:
        out = args.json_path
        if not os.path.isabs(out):
            out = os.path.join(_REPO_ROOT, out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(audit_rows(), fh, ensure_ascii=False, indent=2)
        print(f"\nSaved audit JSON -> {out}")


if __name__ == "__main__":
    main()
