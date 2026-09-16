"""
Synthetic Reasoning Dataset Package (Phase 3)
=============================================

Builds the comparative / transitive reasoning finetuning corpus for Model H
(Hindi) and Model L (Nepali).  The two corpora are generated independently and
never share entity names, templates, phrasing or files.

Modules:
    tasks       Language-neutral task logic and the ground-truth computation.
    seeds       Hand-authored Hindi and Nepali phrase banks (offline fallback).
    phrasebank  Groq / gpt-oss-120b authoring of in-language phrasing, validated.
    generator   Instantiation, splitting, leakage checks and statistics.
"""

from .generator import build_dataset, print_summary
from .phrasebank import (
    QuotaExhausted,
    build_phrasebank,
    check_coverage,
    log_event,
    sanitize_bank,
)
from .tasks import TASKS, TASK_IDS

__all__ = [
    "build_dataset",
    "print_summary",
    "build_phrasebank",
    "check_coverage",
    "sanitize_bank",
    "log_event",
    "QuotaExhausted",
    "TASKS",
    "TASK_IDS",
]
