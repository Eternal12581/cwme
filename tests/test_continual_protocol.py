"""Tests for grounding mode resolution from experiment config."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.experiment_protocol import cwme_no_heuristic_mode
from core.grounding.grounding_mode import GroundingMode, resolve_grounding_mode_from_config


def test_backbone_config_is_grounding_only():
    cfg = {
        "EXPERIMENT": {"ENABLE_PATTERN_READ": False},
        "EXECUTION": {
            "grounding_mode": "grounding_only",
            "task_heuristics_enabled": False,
            "legacy_heuristics_enabled": False,
        },
    }
    assert resolve_grounding_mode_from_config(cfg) == GroundingMode.GROUNDING_ONLY
    assert cwme_no_heuristic_mode(cfg)


def test_legacy_mode_when_legacy_heuristics_on():
    cfg = {
        "EXECUTION": {
            "legacy_heuristics_enabled": True,
        },
    }
    assert resolve_grounding_mode_from_config(cfg) == GroundingMode.LEGACY
    assert not cwme_no_heuristic_mode(cfg)


def test_frozen_config_protocol():
    cfg = {
        "EXPERIMENT": {
            "EXPERIMENT_ID": "E1-frozen",
            "SCIENTIFIC_ROLE": "main_comparison_frozen_evaluation",
            "EVALUATION_PROTOCOL": "frozen_standard_test",
            "ENABLE_PATTERN_READ": True,
            "ENABLE_PATTERN_WRITE": False,
            "FORMAL_EXPERIMENT": True,
            "ARTIFACT_FREEZE": {"enforce": True},
        },
        "CONTINUAL": {"enabled": False},
        "PROCEDURAL_MEMORY": {"read": True, "write": False},
        "EXECUTION": {"task_heuristics_enabled": False, "legacy_heuristics_enabled": False},
    }
    from core.experiment_protocol import build_protocol_snapshot, _infer_evaluation_protocol
    assert _infer_evaluation_protocol(cfg) == "frozen_standard_test"
    assert cwme_no_heuristic_mode(cfg)


def test_continual_config_protocol():
    cfg = {
        "EXPERIMENT": {
            "EXPERIMENT_ID": "E1-continual",
            "SCIENTIFIC_ROLE": "main_comparison_continual_deployment",
            "EVALUATION_PROTOCOL": "online_continual_deployment",
            "ENABLE_PATTERN_READ": True,
            "ENABLE_PATTERN_WRITE": True,
            "FORMAL_EXPERIMENT": True,
        },
        "CONTINUAL": {"enabled": True},
        "PROCEDURAL_MEMORY": {"read": True, "write": True},
        "EXECUTION": {"task_heuristics_enabled": False, "legacy_heuristics_enabled": False},
    }
    from core.experiment_protocol import _infer_evaluation_protocol
    from core.cwme_protocol import _is_formal_experiment
    assert _infer_evaluation_protocol(cfg) == "online_continual_deployment"
    assert _is_formal_experiment(cfg["EXPERIMENT"])
    assert cwme_no_heuristic_mode(cfg)
