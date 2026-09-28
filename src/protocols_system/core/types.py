"""Data classes used by the Protocols System discovery layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProtocolSlot:
    """A discoverable protocol slot for the settings UI.

    Attributes:
        name: Derived config key (e.g. "llm", "image_generation").
        protocol: The protocol type class.
        label: Human-readable label (parsed from protocol docstring).
        description: Description text (parsed from protocol docstring).
    """

    name: str
    protocol: type
    label: str
    description: str


@dataclass(frozen=True)
class ConfigParam:
    """A single configuration parameter introspected from a provider's __init__.

    Attributes:
        key: Parameter name.
        annotation: Type annotation string (e.g. "str", "int", "float").
        default: Default value, or MISSING sentinel if required.
        required: Whether the parameter must be supplied.
        widget_kind: Suggested UI widget type ("str", "int", "float", "bool", "password").
    """

    key: str
    annotation: str = "str"
    default: Any = field(default=None)
    required: bool = False
    widget_kind: str = "str"
