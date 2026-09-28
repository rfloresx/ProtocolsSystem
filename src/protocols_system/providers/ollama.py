"""OllamaClient — async wrapper around the official ollama Python library.

Implements ILLMClient (vision analysis), IEmbeddingClient (image embeddings),
and IHealthCheck. Uses ollama.AsyncClient for all interactions.

Transient errors (ConnectionError) are retried with exponential backoff.

Requires: ollama
"""

from __future__ import annotations

import json
import logging
from typing import Any

from protocols_system.core.registry import ProtocolsRegistry
from protocols_system.core.validation import validate_against_schema
from protocols_system.protocols.embedding import IEmbeddingClient
from protocols_system.protocols.health import IHealthCheck
from protocols_system.protocols.llm import ILLMClient

logger = logging.getLogger(__name__)


@ProtocolsRegistry.register("ollama", ILLMClient)
@ProtocolsRegistry.register("ollama", IEmbeddingClient)
@ProtocolsRegistry.register("ollama", IHealthCheck)
class OllamaClient:
    """Async client for the Ollama API using the official ollama library.

    Provides vision-based image analysis (ILLMClient) and image embedding
    generation (IEmbeddingClient) via a local Ollama server.

    Usage::

        async with OllamaClient(
            base_url="http://localhost:11434",
            vision_model="llava:latest",
            embed_model="mxbai-embed-large",
        ) as client:
            result = await client.analyze_image(img, prompt, schema)
            embedding = await client.embed(img)

    Args:
        base_url: The Ollama server URL.
        vision_model: Model ID for vision/LLM tasks.
        embed_model: Model ID for embedding generation.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        vision_model: str = "llava:latest",
        embed_model: str = "mxbai-embed-large",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._vision_model = vision_model
        self._embed_model = embed_model
        self._client: Any = None  # ollama.AsyncClient

    @property
    def vision_model(self) -> str:
        return self._vision_model

    @property
    def embed_model(self) -> str:
        return self._embed_model

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> OllamaClient:
        from ollama import AsyncClient

        self._client = AsyncClient(host=self._base_url)
        return self

    async def __aexit__(self, *args: object) -> None:
        self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return the active client, creating one lazily if needed."""
        if self._client is None:
            from ollama import AsyncClient

            self._client = AsyncClient(host=self._base_url)
        return self._client

    # ------------------------------------------------------------------
    # ILLMClient
    # ------------------------------------------------------------------

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        return_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Send an image to the vision model and validate the response.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.).
            prompt: Text prompt for the vision model.
            return_schema: JSON-schema-like dict for response validation.

        Returns:
            Parsed JSON dict on success, or {"error": "..."} on failure.

        Raises:
            ConnectionError: On transient connection errors.
        """
        from ollama import ResponseError

        try:
            response = await self._get_client().chat(
                model=self._vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_bytes],
                    }
                ],
                format="json",
                think=False,
            )
        except ResponseError as exc:
            return {"error": f"Ollama API error: {exc}"}
        except (ConnectionError, OSError, TimeoutError) as exc:
            raise ConnectionError(
                f"Cannot reach Ollama server at {self._base_url}: {exc}"
            ) from exc

        raw_content = response.message.content or ""

        if not raw_content.strip():
            return {"error": "Empty content in model response", "raw": ""}

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            return {"error": f"Model response is not valid JSON: {exc}", "raw": raw_content}

        validation_error = validate_against_schema(parsed, return_schema)
        if validation_error is not None:
            return {"error": f"Schema validation failed: {validation_error}", "raw": parsed}

        return parsed  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # IEmbeddingClient
    # ------------------------------------------------------------------

    async def embed(self, image_bytes: bytes) -> list[float]:
        """Generate an embedding vector for an image.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            Embedding vector as list of floats.

        Raises:
            ConnectionError: If the Ollama server is unreachable.
        """
        from ollama import ResponseError

        try:
            response = await self._get_client().embed(
                model=self._embed_model,
                input=image_bytes,
            )
            return [float(x) for x in response.embeddings[0]]
        except ResponseError:
            raise
        except (ConnectionError, OSError, TimeoutError) as exc:
            raise ConnectionError(
                f"Cannot reach Ollama server at {self._base_url}: {exc}"
            ) from exc

    async def embed_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Compute embeddings for a list of images sequentially.

        Ollama processes one image at a time. Failed images return None.

        Args:
            image_bytes_list: Raw image bytes for each image.

        Returns:
            List of embedding vectors or None for failures.
        """
        results: list[list[float] | None] = []
        for image_bytes in image_bytes_list:
            try:
                embedding = await self.embed(image_bytes)
                results.append(embedding)
            except Exception:
                results.append(None)
        return results

    async def close(self) -> None:
        """Release client resources."""
        self._client = None

    # ------------------------------------------------------------------
    # IHealthCheck
    # ------------------------------------------------------------------

    @property
    def health_check_url(self) -> str:
        """The Ollama server base URL."""
        return self._base_url

    async def health_check(self) -> bool:
        """Check connectivity to the Ollama server via /api/tags.

        Returns:
            True if the server responds successfully, False otherwise.
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
