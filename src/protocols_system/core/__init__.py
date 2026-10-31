"""Protocols System core — registry, discovery, and types."""

from .discovery import (
    classify_widget,
    get_all_provider_schemas,
    get_protocol_slots,
    get_provider_schema,
    set_excluded_protocols,
)
from .llm_support import (
    ImageConversionError,
    detect_mime,
    prepare_image_for_vision,
    validate_against_schema,
)
from .registry import ProtocolsRegistry
from .types import ConfigParam, ProtocolSlot

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
    "prepare_image_for_vision",
    "detect_mime",
    "ImageConversionError",
]
