"""Protocols System core — registry, discovery, and types."""

from .discovery import (
    classify_widget,
    get_all_provider_schemas,
    get_protocol_slots,
    get_provider_schema,
    set_excluded_protocols,
)
from .registry import ProtocolsRegistry
from .types import ConfigParam, ProtocolSlot
from .validation import validate_against_schema

__all__ = [
    "ProtocolsRegistry",
    "get_protocol_slots",
    "get_provider_schema",
    "get_all_provider_schemas",
    "classify_widget",
    "set_excluded_protocols",
    "ProtocolSlot",
    "ConfigParam",
    "validate_against_schema",
]
