#!/usr/bin/env python3
"""
Freeze H_train / P_train / schedule SHA256 hashes for formal experiments.

Wraps analyze_artifacts.py with pre-flight checks that required files exist.

Usage:
  python scripts/freeze_artifacts.py
  python scripts/freeze_artifacts.py --strict   # fail if any artifact missing
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_PATHS = {
    "hkg": os.path.join(_REPO_ROOT, "artifacts", "historical_kg", "historical_world_model.json"),
    "pattern": os.path.join(_REPO_ROOT, "artifacts", "procedural_memory", "patterns.jsonl.procedural.jsonl"),
    "schedule": os.path.join(_REPO_ROOT, "artifacts", "schedules", "e02_continual_50ep.json"),
}


def _check_paths(strict: bool, paths: dict[str, str] | None = None) -> list[str]:
    paths = paths or DEFAULT_PATHS
    missing = []
    for key, path in paths.items():
        if key == "pattern" and not os.path.isfile(path) and not path.endswith(".procedural.jsonl"):
            procedural = f"{path}.procedural.jsonl"
            if os.path.isfile(procedural):
                continue
        if not os.path.isfile(path):
            missing.append(f"{key}: {path}")
    if missing and strict:
        raise RuntimeError(
            "Cannot freeze artifacts — missing files:\n  "
            + "\n  ".join(missing)
            + "\n\nRun offline prep first (see artifacts/README.md)."
        )
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze artifact SHA256 manifest")
    parser.add_argument("--strict", action="store_true", help="Fail if artifacts missing")
    parser.add_argument("--hkg", default=DEFAULT_PATHS["hkg"])
    parser.add_argument("--patterns", default=os.path.join(_REPO_ROOT, "artifacts", "procedural_memory", "patterns.jsonl"))
    parser.add_argument("--schedule", default=DEFAULT_PATHS["schedule"])
    parser.add_argument(
        "--manifest",
        default=os.path.join(_REPO_ROOT, "artifacts", "manifest.json"),
        help="Manifest output path; allows independent hashes per dataset/artifact set",
    )
    args = parser.parse_args()

    manifest_path = args.manifest
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(_REPO_ROOT, manifest_path)
    check_paths = {
        "hkg": args.hkg,
        "pattern": args.patterns,
        "schedule": args.schedule,
    }
    missing = _check_paths(args.strict, check_paths)
    if missing and not args.strict:
        print("WARN: missing artifacts (non-strict mode):")
        for line in missing:
            print(f"  - {line}")

    cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "scripts", "analyze_artifacts.py"),
        "--hkg", args.hkg,
        "--patterns", args.patterns,
        "--schedule", args.schedule,
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=_REPO_ROOT)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

    default_manifest = os.path.join(_REPO_ROOT, "artifacts", "manifest.json")
    if os.path.isfile(default_manifest):
        if os.path.abspath(manifest_path) != os.path.abspath(default_manifest):
            os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
            shutil.copyfile(default_manifest, manifest_path)
        print(f"\nArtifact freeze complete -> {manifest_path}")
        print("Formal experiments with ARTIFACT_FREEZE.enforce: true will validate against this manifest.")
    else:
        print("ERROR: manifest.json was not created")
        sys.exit(1)


if __name__ == "__main__":
    main()
