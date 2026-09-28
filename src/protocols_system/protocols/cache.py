"""Cache protocols — local caching for computed values."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ICache(Protocol):
    """A single named cache store."""

    def get(self, key: str, default: Any = None) -> Any: ...

    def put(self, key: str, value: Any) -> None: ...

    def __contains__(self, key: object) -> bool: ...

    def __len__(self) -> int: ...


@runtime_checkable
class ICacheManager(Protocol):
    """Cache:Local cache for scores and embeddings so re-runs skip already-computed assets."""

    def get_cache(self, cache_name: str) -> ICache | None: ...
