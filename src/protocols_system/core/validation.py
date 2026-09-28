"""Shared validation utilities for protocol providers."""

from __future__ import annotations

from typing import Any


def validate_against_schema(data: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Validate a dict against a simplified JSON schema.

    Checks required keys and basic type constraints from the schema's
    ``properties`` definitions. Supports types: number, string, boolean, array.

    Args:
        data: The dict to validate.
        schema: A JSON-schema-like dict with optional "required" and "properties".

    Returns:
        None if valid, or an error message string if validation fails.
    """
    required_keys = schema.get("required", [])
    properties = schema.get("properties", {})

    for key in required_keys:
        if key not in data:
            return f"Missing required key: '{key}'"

    for key, prop_schema in properties.items():
        if key not in data:
            continue
        expected_type = prop_schema.get("type")
        value = data[key]
        if expected_type == "number" and not isinstance(value, (int, float)):
            return f"Key '{key}' expected number, got {type(value).__name__}"
        if expected_type == "string" and not isinstance(value, str):
            return f"Key '{key}' expected string, got {type(value).__name__}"
        if expected_type == "boolean" and not isinstance(value, bool):
            return f"Key '{key}' expected boolean, got {type(value).__name__}"
        if expected_type == "array" and not isinstance(value, list):
            return f"Key '{key}' expected array, got {type(value).__name__}"

    return None
