"""Tests for protocols_system discovery — introspection and schema extraction."""

from __future__ import annotations

from typing import Protocol

import pytest

from protocols_system.core import (
    ProtocolsRegistry,
    classify_widget,
    get_protocol_slots,
    get_provider_schema,
    set_excluded_protocols,
)


class ILLMClient(Protocol):
    """LLM:Large language model inference provider."""

    def complete(self, prompt: str) -> str: ...


class ICacheManager(Protocol):
    """Cache:Manages application cache."""

    def get(self, key: str) -> str: ...


@pytest.fixture(autouse=True)
def clean_protocols():
    yield
    ProtocolsRegistry.reset()
    set_excluded_protocols(set())


class TestGetProtocolSlots:
    """get_protocol_slots() produces ProtocolSlot list from registry."""

    def test_returns_slots_for_registered(self):
        @ProtocolsRegistry.register("test_llm", ILLMClient)
        class TestLLM:
            def complete(self, prompt: str) -> str:
                return ""

        slots = get_protocol_slots()
        assert len(slots) >= 1
        slot = next(s for s in slots if s.protocol is ILLMClient)
        assert slot.name == "llm"
        assert slot.label == "LLM"

    def test_excluded_protocols_filtered(self):
        @ProtocolsRegistry.register("cache", ICacheManager)
        class TestCache:
            pass

        @ProtocolsRegistry.register("llm", ILLMClient)
        class TestLLM:
            pass

        set_excluded_protocols({ICacheManager})
        slots = get_protocol_slots()
        protocols_in_slots = {s.protocol for s in slots}
        assert ICacheManager not in protocols_in_slots
        assert ILLMClient in protocols_in_slots


class TestGetProviderSchema:
    """get_provider_schema() introspects __init__ signatures."""

    def test_basic_schema(self):
        @ProtocolsRegistry.register("openai", ILLMClient)
        class OpenAI:
            def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
                pass

        params = get_provider_schema(ILLMClient, "openai")
        assert len(params) == 2
        api_key_param = next(p for p in params if p.key == "api_key")
        assert api_key_param.required is True
        assert api_key_param.widget_kind == "password"
        model_param = next(p for p in params if p.key == "model")
        assert model_param.required is False
        assert model_param.default == "gpt-4o-mini"

    def test_unknown_provider_raises(self):
        @ProtocolsRegistry.register("only", ILLMClient)
        class Only:
            pass

        with pytest.raises(ValueError, match="No provider named"):
            get_provider_schema(ILLMClient, "nonexistent")

    def test_numeric_params(self):
        @ProtocolsRegistry.register("numeric", ILLMClient)
        class Numeric:
            def __init__(self, temperature: float = 0.7, max_length: int = 100):
                pass

        params = get_provider_schema(ILLMClient, "numeric")
        temp = next(p for p in params if p.key == "temperature")
        assert temp.widget_kind == "float"
        length = next(p for p in params if p.key == "max_length")
        assert length.widget_kind == "int"

    def test_bool_params(self):
        @ProtocolsRegistry.register("flags", ILLMClient)
        class Flags:
            def __init__(self, stream: bool = False):
                pass

        params = get_provider_schema(ILLMClient, "flags")
        stream = next(p for p in params if p.key == "stream")
        assert stream.widget_kind == "bool"


class TestClassifyWidget:
    """classify_widget() maps param info to widget kinds."""

    def test_bool_annotation(self):
        assert classify_widget("flag", "bool", False) == "bool"

    def test_password_by_name(self):
        assert classify_widget("api_key", "str", None) == "password"
        assert classify_widget("secret_token", "str", None) == "password"

    def test_float(self):
        assert classify_widget("temp", "float", 0.5) == "float"

    def test_int(self):
        assert classify_widget("count", "int", 10) == "int"

    def test_str_default(self):
        assert classify_widget("name", "str", "default") == "str"

    def test_int_by_default_value(self):
        """Integer default value triggers int widget even without annotation."""
        assert classify_widget("port", "str", 8080) == "int"
