"""Tests for ProtocolsRegistry — register, create, instances, reset."""

from __future__ import annotations

from typing import Protocol

import pytest

from protocols_system.core import ProtocolsRegistry


class ITestProtocol(Protocol):
    """Test protocol for unit tests."""

    def greet(self) -> str: ...


class ISecondProtocol(Protocol):
    """Another test protocol."""

    def compute(self) -> int: ...


@pytest.fixture(autouse=True)
def clean_protocols_registry():
    """Reset ProtocolsRegistry after each test."""
    yield
    ProtocolsRegistry.reset()


class TestRegisterDecorator:
    """@ProtocolsRegistry.register() catalogs provider classes."""

    def test_basic_registration(self):
        @ProtocolsRegistry.register("test_impl", ITestProtocol)
        class TestImpl:
            def greet(self) -> str:
                return "hello"

        providers = ProtocolsRegistry.get_providers(ITestProtocol)
        assert "test_impl" in providers
        assert providers["test_impl"] is TestImpl

    def test_multiple_providers_per_protocol(self):
        @ProtocolsRegistry.register("impl_a", ITestProtocol)
        class ImplA:
            def greet(self) -> str:
                return "a"

        @ProtocolsRegistry.register("impl_b", ITestProtocol)
        class ImplB:
            def greet(self) -> str:
                return "b"

        providers = ProtocolsRegistry.get_providers(ITestProtocol)
        assert len(providers) == 2

    def test_register_under_multiple_protocols(self):
        @ProtocolsRegistry.register("multi", ITestProtocol)
        @ProtocolsRegistry.register("multi", ISecondProtocol)
        class MultiImpl:
            def greet(self) -> str:
                return "hi"

            def compute(self) -> int:
                return 42

        assert "multi" in ProtocolsRegistry.get_providers(ITestProtocol)
        assert "multi" in ProtocolsRegistry.get_providers(ISecondProtocol)


class TestGetProtocols:
    """get_protocols() returns registered protocol types."""

    def test_returns_registered_protocols(self):
        @ProtocolsRegistry.register("x", ITestProtocol)
        class X:
            pass

        protocols = ProtocolsRegistry.get_protocols()
        assert ITestProtocol in protocols

    def test_empty_when_nothing_registered(self):
        assert ProtocolsRegistry.get_protocols() == []


class TestCreateInstance:
    """create() instantiates and stores singleton."""

    def test_create_stores_singleton(self):
        @ProtocolsRegistry.register("simple", ITestProtocol)
        class SimpleImpl:
            def __init__(self, name: str = "world"):
                self.name = name

            def greet(self) -> str:
                return f"hello {self.name}"

        instance = ProtocolsRegistry.create(ITestProtocol, "simple", {"name": "test"})
        assert instance.greet() == "hello test"
        assert ProtocolsRegistry.get_instance(ITestProtocol) is instance

    def test_create_unknown_protocol_raises(self):
        with pytest.raises(ValueError, match="No providers registered"):
            ProtocolsRegistry.create(ISecondProtocol, "missing")

    def test_create_unknown_provider_raises(self):
        @ProtocolsRegistry.register("only_one", ITestProtocol)
        class OnlyOne:
            pass

        with pytest.raises(ValueError, match="No provider named"):
            ProtocolsRegistry.create(ITestProtocol, "wrong_name")


class TestGetInstance:
    """get_instance() retrieves stored singleton."""

    def test_returns_none_before_create(self):
        assert ProtocolsRegistry.get_instance(ITestProtocol) is None

    def test_set_instance(self):
        sentinel = object()
        ProtocolsRegistry.set_instance(ITestProtocol, sentinel)
        assert ProtocolsRegistry.get_instance(ITestProtocol) is sentinel


class TestReset:
    """reset() clears catalog and instances."""

    def test_reset_clears_all(self):
        @ProtocolsRegistry.register("temp", ITestProtocol)
        class Temp:
            pass

        ProtocolsRegistry.set_instance(ITestProtocol, object())
        ProtocolsRegistry.reset()
        assert ProtocolsRegistry.get_protocols() == []
        assert ProtocolsRegistry.get_instance(ITestProtocol) is None

    def test_clear_instances_keeps_catalog(self):
        @ProtocolsRegistry.register("kept", ITestProtocol)
        class Kept:
            pass

        ProtocolsRegistry.set_instance(ITestProtocol, object())
        ProtocolsRegistry.clear_instances()
        # Catalog still intact
        assert "kept" in ProtocolsRegistry.get_providers(ITestProtocol)
        # Instance cleared
        assert ProtocolsRegistry.get_instance(ITestProtocol) is None
