"""Protocols System — A self-describing, pluggable service locator.

Provides decorator-based provider registration, singleton lifecycle management,
and introspection-driven GUI discovery. Protocol and provider additions
automatically surface in any UI that consumes the discovery layer.
"""

from .core import (
    ConfigParam,
    ProtocolSlot,
    ProtocolsRegistry,
    classify_widget,
    get_all_provider_schemas,
    get_protocol_slots,
    get_provider_schema,
    set_excluded_protocols,
)

__all__ = [
    "ProtocolsRegistry",
    "get_protocol_slots",
    "get_provider_schema",
    "get_all_provider_schemas",
    "classify_widget",
    "set_excluded_protocols",
    "ProtocolSlot",
    "ConfigParam",
]
