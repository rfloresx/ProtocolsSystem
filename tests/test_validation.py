"""Tests for validate_against_schema utility."""

from __future__ import annotations

from protocols_system.core import validate_against_schema


class TestValidateAgainstSchema:
    """validate_against_schema() checks required keys and types."""

    def test_valid_data(self):
        """No error for conforming data."""
        schema = {
            "required": ["host", "port"],
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "number"},
            },
        }
        result = validate_against_schema({"host": "localhost", "port": 8080}, schema)
        assert result is None

    def test_missing_required_key(self):
        """Reports missing required keys."""
        schema = {
            "required": ["name", "value"],
            "properties": {},
        }
        result = validate_against_schema({"name": "test"}, schema)
        assert result is not None
        assert "value" in result

    def test_type_mismatch_number(self):
        """Reports wrong type for number field."""
        schema = {
            "required": [],
            "properties": {"port": {"type": "number"}},
        }
        result = validate_against_schema({"port": "not_a_number"}, schema)
        assert result is not None
        assert "number" in result

    def test_type_mismatch_string(self):
        """Reports wrong type for string field."""
        schema = {
            "required": [],
            "properties": {"name": {"type": "string"}},
        }
        result = validate_against_schema({"name": 123}, schema)
        assert result is not None
        assert "string" in result

    def test_type_mismatch_boolean(self):
        """Reports wrong type for boolean field."""
        schema = {
            "required": [],
            "properties": {"enabled": {"type": "boolean"}},
        }
        result = validate_against_schema({"enabled": "yes"}, schema)
        assert result is not None
        assert "boolean" in result

    def test_type_mismatch_array(self):
        """Reports wrong type for array field."""
        schema = {
            "required": [],
            "properties": {"items": {"type": "array"}},
        }
        result = validate_against_schema({"items": "not_a_list"}, schema)
        assert result is not None
        assert "array" in result

    def test_optional_key_missing_ok(self):
        """Missing non-required keys are not an error."""
        schema = {
            "required": [],
            "properties": {"optional_field": {"type": "string"}},
        }
        result = validate_against_schema({}, schema)
        assert result is None

    def test_empty_schema(self):
        """Empty schema validates any dict."""
        result = validate_against_schema({"anything": "goes"}, {})
        assert result is None

    def test_int_passes_number_check(self):
        """Both int and float are valid for 'number' type."""
        schema = {
            "required": [],
            "properties": {"value": {"type": "number"}},
        }
        assert validate_against_schema({"value": 42}, schema) is None
        assert validate_against_schema({"value": 3.14}, schema) is None
