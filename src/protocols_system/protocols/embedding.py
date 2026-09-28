"""Embedding protocol — image embedding computation for clustering and deduplication."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IEmbeddingClient(Protocol):
    """Embedding:Backend that computes image embeddings.

    Used for scene clustering and near-duplicate detection.
    """

    @property
    def embed_model(self) -> str: ...

    async def embed(self, image_bytes: bytes) -> list[float]: ...

    async def embed_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]: ...

    async def close(self) -> None: ...
