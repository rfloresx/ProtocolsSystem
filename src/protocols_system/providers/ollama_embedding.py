"""OllamaEmbeddingClient — registered IEmbeddingClient slot for Ollama (unsupported).

Registered under the "ollama_v2" provider slot for IEmbeddingClient, but
``embed()``/``embed_batch()`` always raise ``NotImplementedError``: Ollama's
``/api/embed`` endpoint embeds *text*, not images, so handing it image bytes
silently produces meaningless vectors. Selecting this provider therefore fails
loudly rather than corrupting every downstream similarity comparison.

``health_check()`` still works — checking connectivity to the Ollama server
doesn't require embedding an image.

Requires: httpx
"""

from __future__ import annotations

import logging

import httpx

from protocols_system.core.registry import ProtocolsRegistry
from protocols_system.protocols.embedding import IEmbeddingClient
from protocols_system.protocols.health import IHealthCheck

logger = logging.getLogger(__name__)


@ProtocolsRegistry.register("ollama_v2", IEmbeddingClient)
@ProtocolsRegistry.register("ollama_v2", IHealthCheck)
class OllamaEmbeddingClient:
    """Registered IEmbeddingClient provider for Ollama — currently unsupported.

    ``embed()``/``embed_batch()`` always raise ``NotImplementedError``; use a
    dedicated image-embedding provider ("huggingface", "image_embedding").
    ``health_check()`` reports real connectivity to the Ollama server.

    Args:
        model: The embedding model name (informational).
        base_url: The Ollama server URL.
    """

    def __init__(
        self,
        model: str = "mxbai-embed-large",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    @property
    def embed_model(self) -> str:
        """The embedding model name."""
        return self._model

    @property
    def base_url(self) -> str:
        """The Ollama server base URL."""
        return self._base_url

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> OllamaEmbeddingClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(60.0),
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def _get_client(self) -> httpx.AsyncClient:
        """Return the active httpx client, creating one lazily if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(60.0),
            )
        return self._client

    # ------------------------------------------------------------------
    # IEmbeddingClient (unsupported — fails loudly)
    # ------------------------------------------------------------------

    async def embed(self, image_bytes: bytes) -> list[float]:
        """Raise: Ollama's ``/api/embed`` endpoint embeds text, not images.

        Raises:
            NotImplementedError: Always. Use a dedicated image-embedding
                provider ("huggingface", "image_embedding").
        """
        raise NotImplementedError(
            "Ollama's /api/embed endpoint embeds text, not images — sending "
            "a base64-encoded image as text silently produces meaningless "
            "vectors rather than an error. Use a dedicated image-embedding "
            "provider (huggingface, image_embedding) instead."
        )

    async def embed_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Raise: see ``embed()`` — Ollama's embed endpoint doesn't do images."""
        raise NotImplementedError(
            "Ollama's /api/embed endpoint embeds text, not images — sending "
            "a base64-encoded image as text silently produces meaningless "
            "vectors rather than an error. Use a dedicated image-embedding "
            "provider (huggingface, image_embedding) instead."
        )

    async def close(self) -> None:
        """Close the underlying httpx client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # IHealthCheck
    # ------------------------------------------------------------------

    @property
    def health_check_url(self) -> str:
        """The Ollama server base URL."""
        return self._base_url

    async def health_check(self) -> bool:
        """Check connectivity to the Ollama server via /api/tags."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
