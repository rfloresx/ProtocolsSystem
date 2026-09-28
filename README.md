# Protocols System

A self-describing, pluggable service locator for Python.

Protocols System provides decorator-based provider registration, singleton
lifecycle management, and introspection-driven GUI discovery. Protocol and
provider additions automatically surface in any UI that consumes the discovery
layer.

It is shared across projects (PhotoStudio, Smart-Albums) as a standalone
package so provider implementations and the registry live in one place.

## Install

```bash
# Core only (registry, discovery, protocols)
pip install protocols-system

# With built-in provider implementations (Immich, Ollama, llama.cpp, Google Places, ...)
pip install "protocols-system[providers]"
```

For local development against a checkout:

```bash
pip install -e "../ProtocolsSystem[providers]"
```

## Layout

- `protocols_system.core` — registry, discovery, types, validation
- `protocols_system.protocols` — protocol (interface) definitions: cache,
  embedding, geo, health, image_library, llm, progress
- `protocols_system.providers` — concrete provider implementations that
  self-register via the `@ProtocolsRegistry.register` decorator

## Usage

```python
from protocols_system import ProtocolsRegistry, get_protocol_slots
from protocols_system.providers import register_builtin_providers

# Import provider modules so they register themselves. Providers whose
# optional dependencies are missing are skipped silently.
register_builtin_providers()

registry = ProtocolsRegistry()
```

## Development

```bash
pip install -e ".[dev,providers]"
pytest
mypy
ruff check
```
