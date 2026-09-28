"""ImageEmbeddingClient — async HTTP client for a dedicated image embedding service.

Implements IEmbeddingClient and IHealthCheck. Communicates with a standalone
embedding server that exposes /embed-image (single) and /embed-images (batch)
multipart endpoints.

Requires: httpx
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from protocols_system.core.registry import ProtocolsRegistry
from protocols_system.protocols.embedding import IEmbeddingClient
from protocols_system.protocols.health import IHealthCheck

logger = logging.getLogger(__name__)


@ProtocolsRegistry.register("image_embedding", IEmbeddingClient)
@ProtocolsRegistry.register("image_embedding", IHealthCheck)
class ImageEmbeddingClient:
    """Async client for a dedicated image embedding HTTP service.

    Connects to a server implementing the embed-image / embed-images API
    (multipart file upload) and returns normalized embedding vectors.

    Usage::

        async with ImageEmbeddingClient(
            base_url="http://localhost:8000",
            model_name="openai/clip-vit-base-patch32",
        ) as client:
            embedding = await client.embed(image_bytes)
            embeddings = await client.embed_batch([img1, img2, img3])

    Args:
        base_url: The embedding service URL (e.g. "http://localhost:8000").
        model_name: HuggingFace model ID for embedding generation.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "openai/clip-vit-base-patch32",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout = float(timeout)
        self._client: httpx.AsyncClient | None = None

    @property
    def embed_model(self) -> str:
        """The HuggingFace model ID used for embeddings."""
        return self._model_name

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> ImageEmbeddingClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return the active HTTP client, creating one lazily if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    # ------------------------------------------------------------------
    # IEmbeddingClient
    # ------------------------------------------------------------------

    async def embed(self, image_bytes: bytes) -> list[float]:
        """Compute an embedding vector for a single image.

        Sends the image as a multipart file upload to /embed-image.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, WebP, etc.).

        Returns:
            Normalized embedding vector as list of floats.

        Raises:
            ConnectionError: If the server is unreachable or returns 5xx.
            RuntimeError: If the server returns a 4xx error.
        """
        try:
            response = await self._get_client().post(
                "/embed-image",
                params={"model_name": self._model_name},
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = exc.response.text
            if status >= 500:
                raise ConnectionError(
                    f"Embedding server error (HTTP {status}): {detail}"
                ) from exc
            raise RuntimeError(
                f"Embedding request failed (HTTP {status}): {detail}"
            ) from exc
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot reach embedding server at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Timeout connecting to embedding server: {exc}"
            ) from exc

        data: dict[str, Any] = response.json()
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError(
                f"Embedding response missing 'embedding' field: {data}"
            )

        return [float(x) for x in embedding]

    async def embed_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Compute embeddings for multiple images in a single request.

        Sends all images as multipart upload to /embed-images. Falls back
        to sequential embed() calls if the batch endpoint fails.

        Args:
            image_bytes_list: Raw image bytes for each image.

        Returns:
            List of embedding vectors or None for failures.
        """
        if not image_bytes_list:
            return []

        files = [
            ("files", (f"image_{i}.jpg", img_bytes, "image/jpeg"))
            for i, img_bytes in enumerate(image_bytes_list)
        ]

        try:
            response = await self._get_client().post(
                "/embed-images",
                params={"model_name": self._model_name},
                files=files,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise ConnectionError(
                    f"Embedding server error (HTTP {exc.response.status_code})"
                ) from exc
            # 4xx on batch — fall back to sequential
            logger.warning(
                "Batch embed failed (HTTP %d), falling back to sequential",
                exc.response.status_code,
            )
            return await self._fallback_sequential(image_bytes_list)
        except (httpx.ConnectError, httpx.TimeoutException):
            raise
        except httpx.TransportError as exc:
            raise ConnectionError(
                f"Cannot reach embedding server at {self._base_url}: {exc}"
            ) from exc

        data: dict[str, Any] = response.json()
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(image_bytes_list):
            logger.warning(
                "Batch response mismatch (%d vs %d), falling back to sequential",
                len(embeddings) if embeddings else 0,
                len(image_bytes_list),
            )
            return await self._fallback_sequential(image_bytes_list)

        return [[float(x) for x in emb] for emb in embeddings]

    async def _fallback_sequential(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Process images one at a time when batch endpoint fails."""
        results: list[list[float] | None] = []
        for image_bytes in image_bytes_list:
            try:
                embedding = await self.embed(image_bytes)
                results.append(embedding)
            except Exception as exc:
                logger.warning("Sequential embed fallback failed: %s", exc)
                results.append(None)
        return results

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # IHealthCheck
    # ------------------------------------------------------------------

    @property
    def health_check_url(self) -> str:
        """The embedding service base URL."""
        return self._base_url

    async def health_check(self) -> bool:
        """Check connectivity to the embedding server via /health.

        Returns:
            True if the server responds with status "ok", False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    return bool(data.get("status") == "ok")
                return False
        except Exception:
            return False
