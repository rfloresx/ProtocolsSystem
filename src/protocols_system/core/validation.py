"""Shared validation utilities for protocol providers.

The canonical implementation now lives in :mod:`protocols_system.core.llm_support`
(the stricter validator used by every LLM provider). This module re-exports it
so existing ``from protocols_system.core.validation import validate_against_schema``
imports keep working against a single source of truth.
"""

from __future__ import annotations

from .llm_support import validate_against_schema

__all__ = ["validate_against_schema"]
