"""Introspection-driven discovery for dynamic GUI rendering.

This module provides the bridge between the ProtocolsRegistry catalog and a UI
framework. It introspects protocols (via docstrings) and providers (via __init__
signatures) so that settings forms can be rendered without any per-provider UI code.
"""

from __future__ import annotations

import re
from inspect import Parameter, signature
from typing import Any

from .registry import ProtocolsRegistry
from .types import ConfigParam, ProtocolSlot

# Protocols in this set won't appear in user-facing settings UI.
_EXCLUDED_PROTOCOLS: set[type] = set()

# Hints that indicate a parameter should be rendered as a password field.
_SECRET_HINTS = ("api_key", "password", "secret", "token", "credentials")


def set_excluded_protocols(protocols: set[type]) -> None:
    """Configure which protocols are excluded from the settings UI.

    Call this once at app startup after defining your infrastructure protocols.
    """
    global _EXCLUDED_PROTOCOLS
    _EXCLUDED_PROTOCOLS = protocols


def get_protocol_slots() -> list[ProtocolSlot]:
    """Discover all configurable protocol slots from the registry.

    Returns a list of ProtocolSlot objects suitable for rendering a settings page.
    Infrastructure protocols in the exclusion set are filtered out.
    """
    slots: list[ProtocolSlot] = []
    for protocol_cls in ProtocolsRegistry.get_protocols():
        if protocol_cls in _EXCLUDED_PROTOCOLS:
            continue
        name = _derive_slot_name(protocol_cls)
        label, description = _parse_protocol_doc(protocol_cls)
        slots.append(ProtocolSlot(name, protocol_cls, label, description))
    return slots


def get_provider_schema(protocol: type, provider_name: str) -> list[ConfigParam]:
    """Introspect a provider's __init__ to produce a list of config parameters.

    Args:
        protocol: The protocol type.
        provider_name: The registered provider name.

    Returns:
        List of ConfigParam objects describing each constructor parameter.

    Raises:
        ValueError: If the protocol or provider is not registered.
    """
    providers = ProtocolsRegistry.get_providers(protocol)
    if provider_name not in providers:
        raise ValueError(
            f"No provider named '{provider_name}' for protocol {protocol.__name__}"
        )
    cls = providers[provider_name]
    return _introspect_init(cls)


def get_all_provider_schemas(protocol: type) -> dict[str, list[ConfigParam]]:
    """Return schemas for all providers registered under a protocol.

    Returns:
        Dict mapping provider_name -> list[ConfigParam].
    """
    providers = ProtocolsRegistry.get_providers(protocol)
    return {name: _introspect_init(cls) for name, cls in providers.items()}


def classify_widget(key: str, annotation: str, default: Any) -> str:
    """Classify a parameter into a UI widget kind.

    Args:
        key: Parameter name.
        annotation: Type annotation as string.
        default: Default value (or Parameter.empty).

    Returns:
        One of: "bool", "password", "float", "int", "str".
    """
    # Bool detection
    if "bool" in annotation.lower() or isinstance(default, bool):
        return "bool"

    # Secret-looking parameter names
    if any(hint in key.lower() for hint in _SECRET_HINTS):
        return "password"

    # Numeric types
    if "float" in annotation.lower() or isinstance(default, float):
        return "float"
    if "int" in annotation.lower() or isinstance(default, int):
        return "int"

    return "str"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _derive_slot_name(protocol_cls: type) -> str:
    """Derive a config-friendly slot name from a protocol class name.

    IImageGenerationClient -> "image_generation"
    ILLMClient             -> "llm"
    IStorageClient         -> "storage"
    ICacheManager          -> "cache_manager"
    """
    name = protocol_cls.__name__

    # Strip leading "I"
    if name.startswith("I") and len(name) > 1 and name[1].isupper():
        name = name[1:]

    # Strip trailing "Client" or "Manager" or "Provider"
    for suffix in ("Client", "Manager", "Provider", "Service"):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            break

    # CamelCase to snake_case
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


def _parse_protocol_doc(protocol_cls: type) -> tuple[str, str]:
    """Parse protocol docstring into (label, description).

    Expected format: "Label:Description text here."
    Falls back to class name if docstring is missing or malformed.
    """
    doc = protocol_cls.__doc__ or ""
    first_line = doc.strip().split("\n")[0].strip()

    if ":" in first_line:
        label, description = first_line.split(":", 1)
        return label.strip(), description.strip()

    # Fallback: derive label from class name
    label = _derive_slot_name(protocol_cls).replace("_", " ").title()
    return label, first_line or f"Provider for {label}."


def _introspect_init(cls: type) -> list[ConfigParam]:
    """Extract ConfigParam list from a class's __init__ signature."""
    try:
        sig = signature(cls.__init__)  # type: ignore[misc]
    except (ValueError, TypeError):
        return []

    params: list[ConfigParam] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue

        # Resolve annotation
        if param.annotation is not Parameter.empty:
            annotation = (
                param.annotation.__name__
                if hasattr(param.annotation, "__name__")
                else str(param.annotation)
            )
        else:
            annotation = "str"

        # Resolve default
        has_default = param.default is not Parameter.empty
        default = param.default if has_default else None
        required = not has_default

        # Classify widget
        widget_kind = classify_widget(name, annotation, default)

        params.append(
            ConfigParam(
                key=name,
                annotation=annotation,
                default=default,
                required=required,
                widget_kind=widget_kind,
            )
        )
    return params
