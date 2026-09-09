"""
CWME shared types and task-agnostic progress helpers (compatibility shim).

Runtime evolution is handled by ``ContinualWorldModelEvolution``; prefer importing
from ``core.evolution.evolution_types`` and ``core.evolution.progress_evaluator``.
"""
from __future__ import annotations

from core.evolution.evolution_types import ContinualStepResult
from core.evolution.progress_evaluator import ProgressEvaluator, structural_progress_credit

__all__ = ["ContinualStepResult", "ProgressEvaluator", "structural_progress_credit"]
