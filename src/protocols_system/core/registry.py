"""ProtocolsRegistry — Decorator-based service locator with singleton lifecycle.

Usage:
    @ProtocolsRegistry.register("openai", ILLMClient)
    class OpenAIClient:
        def __init__(self, api_key: str, model: str = "gpt-4o-mini"): ...

    # Later, instantiate with user config:
    client = ProtocolsRegistry.create(ILLMClient, "openai", {"api_key": "sk-..."})

    # Retrieve the live singleton:
    client = ProtocolsRegistry.get_instance(ILLMClient)
"""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


class ProtocolsRegistry:
    """Static registry mapping protocol types to named provider classes.

    Maintains two separate stores:
    - _registry: catalog of protocol -> {provider_name: provider_class}  (permanent)
    - _instances: protocol -> live singleton instance  (clearable between runs)
    """

    _registry: dict[type, dict[str, type]] = {}
    _instances: dict[type, Any] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @staticmethod
    def register(name: str, protocol: type) -> Any:
        """Decorator that registers a class as a named provider for a protocol.

        A single class can be registered under multiple protocols by stacking
        decorators.

        Args:
            name: Provider name (e.g. "openai", "ollama", "s3").
            protocol: The protocol type this class implements.

        Returns:
            A decorator that registers the class and returns it unchanged.
        """

        def decorator(cls: type) -> type:
            if protocol not in ProtocolsRegistry._registry:
                ProtocolsRegistry._registry[protocol] = {}
            ProtocolsRegistry._registry[protocol][name] = cls
            return cls

        return decorator

    # ------------------------------------------------------------------
    # Catalog queries
    # ------------------------------------------------------------------

    @staticmethod
    def get_registry() -> dict[type, dict[str, type]]:
        """Return the full registry mapping (protocol -> {name: class})."""
        return ProtocolsRegistry._registry

    @staticmethod
    def get_protocols() -> list[type]:
        """Return all protocol types that have at least one registered provider."""
        return list(ProtocolsRegistry._registry.keys())

    @staticmethod
    def get_providers(protocol: type) -> dict[str, type]:
        """Return {name: class} for all providers registered under a protocol."""
        return ProtocolsRegistry._registry.get(protocol, {})

    # ------------------------------------------------------------------
    # Instance lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def create(protocol: type[T], name: str, config: dict[str, Any] | None = None) -> T:
        """Instantiate a provider and store it as the active singleton for the protocol.

        Args:
            protocol: The protocol type to resolve.
            name: The registered provider name.
            config: Keyword arguments passed to the provider's __init__.

        Returns:
            The newly created instance.

        Raises:
            ValueError: If the protocol or provider name is not registered.
        """
        if protocol not in ProtocolsRegistry._registry:
            raise ValueError(f"No providers registered for protocol {protocol.__name__}")
        providers = ProtocolsRegistry._registry[protocol]
        if name not in providers:
            available = ", ".join(sorted(providers.keys())) or "(none)"
            raise ValueError(
                f"No provider named '{name}' for protocol {protocol.__name__}. "
                f"Available: {available}"
            )
        instance = providers[name](**(config or {}))
        ProtocolsRegistry._instances[protocol] = instance
        return instance  # type: ignore[no-any-return]

    @staticmethod
    def build_provider(
        protocol: type[T], name: str, config: dict[str, Any] | None = None
    ) -> T:
        """Instantiate a provider WITHOUT storing it as the active singleton.

        Use this when a composite provider (e.g. a multiplexer) needs to build
        the sub-providers it delegates to, without replacing the protocol's live
        singleton (which is the composite itself).

        Args:
            protocol: The protocol type to resolve.
            name: The registered provider name.
            config: Keyword arguments passed to the provider's __init__.

        Returns:
            The newly created instance.

        Raises:
            ValueError: If the protocol or provider name is not registered.
        """
        if protocol not in ProtocolsRegistry._registry:
            raise ValueError(f"No providers registered for protocol {protocol.__name__}")
        providers = ProtocolsRegistry._registry[protocol]
        if name not in providers:
            available = ", ".join(sorted(providers.keys())) or "(none)"
            raise ValueError(
                f"No provider named '{name}' for protocol {protocol.__name__}. "
                f"Available: {available}"
            )
        return providers[name](**(config or {}))  # type: ignore[no-any-return]

    @staticmethod
    def get_instance(protocol: type[T]) -> T | None:
        """Retrieve the live singleton for a protocol, or None if not yet created."""
        return ProtocolsRegistry._instances.get(protocol)

    @staticmethod
    def set_instance(protocol: type, instance: Any) -> None:
        """Manually inject an instance (useful for testing or webgui overrides)."""
        ProtocolsRegistry._instances[protocol] = instance

    @staticmethod
    def clear_instances() -> None:
        """Clear all live instances. Catalog remains intact."""
        ProtocolsRegistry._instances.clear()

    @staticmethod
    def reset() -> None:
        """Clear both catalog and instances. Intended for test isolation."""
        ProtocolsRegistry._registry.clear()
        ProtocolsRegistry._instances.clear()
