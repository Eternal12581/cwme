#!/usr/bin/env python3
"""
Static compliance check against the paper review checklist (no GPU required).

Usage:
  python scripts/verify_review_compliance.py
"""
from __future__ import annotations

import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml

from core.experiment_protocol import _normalize_manifest_hashes, cwme_no_heuristic_mode


def _failures() -> list[str]:
    return []


def _check_file(path: str, label: str, failures: list[str]) -> None:
    full = path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)
    if not os.path.isfile(full):
        failures.append(f"missing file: {label} ({path})")


def _read_text(path: str) -> str:
    full = os.path.join(_REPO_ROOT, path) if not os.path.isabs(path) else path
    with open(full, encoding="utf-8") as fh:
        return fh.read()


def _run_script(rel_path: str, failures: list[str]) -> None:
    cmd = [sys.executable, os.path.join(_REPO_ROOT, rel_path)]
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        failures.append(
            f"{rel_path} failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '').strip()[:240]}"
        )


def main() -> None:
    failures: list[str] = []

    required = [
        ("core/cwme_planning.py", "CWME planning helpers"),
        ("core/experiment_protocol.py", "protocol snapshot"),
        ("scripts/compare_sanity.py", "sanity compare"),
        ("scripts/check_experiment_invariants.py", "invariant checker"),
        ("scripts/validate_experiment.py", "validate wrapper"),
        ("config/experiments/run_sanity.sh", "sanity runner"),
        ("config/experiments/run_tier1_3seed.sh", "3-seed runner"),
        ("config/experiments/e03_wo_fast_reuse.yml", "E3 wo fast reuse"),
    ]
    for path, label in required:
        _check_file(path, label, failures)

    tier1 = _read_text("config/experiments/run_tier1_3seed.sh")
    for needle in ("e05_fast_on.yml", "e05_fast_off.yml", "sanity_compare.json", "e03_wo_fast_reuse.yml"):
        if needle not in tier1:
            failures.append(f"run_tier1_3seed.sh missing reference: {needle}")

    sanity_sh = _read_text("config/experiments/run_sanity.sh")
    if "compare_sanity.py" not in sanity_sh:
        failures.append("run_sanity.sh does not invoke compare_sanity.py")

    if "wo_direct_reuse" in _read_text("config/experiments/MANIFEST.md"):
        # allow only as historical note — grep whole repo
        pass
    for root, _dirs, files in os.walk(os.path.join(_REPO_ROOT, "config")):
        for name in files:
            if "wo_direct_reuse" in name:
                failures.append(f"deprecated config filename: {name}")

    manifest_sample = {
        "h_train": {"sha256": "sha256:abc"},
        "p_train": {"sha256": "sha256:def"},
        "e02_schedule": {"sha256": "sha256:ghi"},
    }
    normalized = _normalize_manifest_hashes(manifest_sample)
    if normalized.get("hkg") != "sha256:abc" or normalized.get("pattern") != "sha256:def":
        failures.append("manifest hash normalization broken")
    if normalized.get("schedule") != "sha256:ghi":
        failures.append("manifest schedule hash normalization broken")

    cwme_cfg = yaml.load(
        open(os.path.join(_REPO_ROOT, "config/experiments/e01_continual_cwme.yml"), encoding="utf-8"),
        Loader=yaml.FullLoader,
    )
    if not cwme_no_heuristic_mode(cwme_cfg or {}):
        failures.append("e01_cwme_sci_main.yml should be CWME no-heuristic mode")

    backbone_cfg = yaml.load(
        open(os.path.join(_REPO_ROOT, "config/experiments/e02_backbone_continual_50ep.yml"), encoding="utf-8"),
        Loader=yaml.FullLoader,
    )
    if not cwme_no_heuristic_mode(backbone_cfg or {}):
        failures.append("e02_backbone should use grounding_only cognition mode")

    role = (cwme_cfg.get("EXPERIMENT") or {}).get("SCIENTIFIC_ROLE", "")
    if role != "main_comparison_continual_deployment":
        failures.append(f"e01_continual missing SCIENTIFIC_ROLE (got {role!r})")

    frozen_cfg = yaml.load(
        open(os.path.join(_REPO_ROOT, "config/experiments/e01_frozen_static_pm.yml"), encoding="utf-8"),
        Loader=yaml.FullLoader,
    )
    frozen_role = (frozen_cfg.get("EXPERIMENT") or {}).get("SCIENTIFIC_ROLE", "")
    if frozen_role != "main_comparison_frozen_evaluation":
        failures.append(f"e01_frozen missing SCIENTIFIC_ROLE (got {frozen_role!r})")
    if not bool((frozen_cfg.get("EXPERIMENT") or {}).get("ARTIFACT_FREEZE", {}).get("enforce")):
        failures.append("e01_frozen_static_pm.yml must enforce ARTIFACT_FREEZE")

    for cfg_name in (
        "e04_hetero_4b9b.yml",
        "e04_scale_4b27b.yml",
        "e05_hetero_4b9b.yml",
    ):
        cfg = yaml.load(
            open(os.path.join(_REPO_ROOT, "config/experiments", cfg_name), encoding="utf-8"),
            Loader=yaml.FullLoader,
        )
        pm = dict((cfg or {}).get("PROCEDURAL_MEMORY") or {})
        if pm.get("source") != "train_artifact":
            failures.append(f"{cfg_name} missing PROCEDURAL_MEMORY.source=train_artifact")

    print("=== Review compliance (static) ===")
    for rel in ("scripts/sanity_grounding_split.py", "scripts/sanity_execution_router.py", "scripts/run_unit_tests.py"):
        print(f"Running {rel} ...")
        _run_script(rel, failures)

    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO_ROOT, "scripts", "check_experiment_invariants.py"),
         os.path.join(_REPO_ROOT, "config/experiments/e01_continual_cwme.yml"), "--pre-run"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        failures.append("pre-run invariants failed for e01_cwme_sci_main.yml")
    else:
        print("Pre-run invariants e01: PASS")

    if failures:
        print("\nFAILED:")
        for item in failures:
            print(f"  - {item}")
        sys.exit(1)

    print("\nAll static review compliance checks passed.")


if __name__ == "__main__":
    main()
