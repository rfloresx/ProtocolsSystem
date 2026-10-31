"""OllamaClient — async wrapper around the official ollama Python library.

Implements ILLMClient (vision analysis) and IHealthCheck via ``ollama.AsyncClient``.
Transient errors (ConnectionError, 5xx) are retried via tenacity with exponential
backoff.

Note: this client does NOT implement image embeddings. Ollama's ``/api/embed``
endpoint embeds *text*, so handing it image bytes silently produces meaningless
vectors; ``embed()``/``embed_batch()`` raise ``NotImplementedError``. Use a
dedicated image-embedding provider ("huggingface", "image_embedding") instead.

Requires: ollama, tenacity
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ollama import AsyncClient, ResponseError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from protocols_system.core.llm_support import (
    ImageConversionError,
    prepare_image_for_vision,
    validate_against_schema,
)
from protocols_system.core.registry import ProtocolsRegistry
from protocols_system.protocols.embedding import IEmbeddingClient
from protocols_system.protocols.health import IHealthCheck
from protocols_system.protocols.llm import ILLMClient

logger = logging.getLogger(__name__)


_retry_policy = retry(
    retry=retry_if_exception_type(ConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@ProtocolsRegistry.register("ollama", ILLMClient)
@ProtocolsRegistry.register("ollama", IEmbeddingClient)
@ProtocolsRegistry.register("ollama", IHealthCheck)
class OllamaClient:
    """Async client for the Ollama API using the official ollama library.

    Provides vision-based image analysis (ILLMClient) via a local Ollama
    server. The library handles image encoding, JSON format negotiation, and
    API endpoint routing internally.

    Usage::

        async with OllamaClient(
            base_url="http://localhost:11434",
            vision_model="llava:latest",
            embed_model="mxbai-embed-large",
        ) as client:
            result = await client.analyze_image(img, prompt, schema)

    Args:
        base_url: The Ollama server URL.
        vision_model: Model ID for vision/LLM tasks.
        embed_model: Model ID retained for config/discovery compatibility.
        timeout: Per-request timeout in seconds. Without a timeout the ollama
            AsyncClient would wait forever on a stuck model load, hanging the
            whole pipeline.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        vision_model: str = "llava:latest",
        embed_model: str = "mxbai-embed-large",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._vision_model = vision_model
        self._embed_model = embed_model
        self._timeout = float(timeout)
        self._client: AsyncClient | None = None

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
        self._client = AsyncClient(host=self._base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> AsyncClient:
        """Return the active client, creating one lazily if needed."""
        if self._client is None:
            self._client = AsyncClient(host=self._base_url, timeout=self._timeout)
        return self._client

    # ------------------------------------------------------------------
    # ILLMClient
    # ------------------------------------------------------------------

    @_retry_policy
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
            ConnectionError: On transient connection errors (retried by tenacity).
        """
        # Normalize non-JPEG/PNG formats (HEIC/TIFF/BMP/GIF/WebP) to JPEG
        # before sending, and fail cleanly if conversion isn't possible
        # rather than handing the model undecodable bytes.
        try:
            prepared_bytes, _mime = prepare_image_for_vision(image_bytes)
        except ImageConversionError as exc:
            return {"error": f"Image conversion failed: {exc}"}

        try:
            response = await self._get_client().chat(
                model=self._vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [prepared_bytes],
                    }
                ],
                format="json",
                think=False,
            )
        except ResponseError as exc:
            # A 5xx usually means the server is still loading the model or is
            # transiently out of memory — wrap it so tenacity retries. Keep
            # 4xx (e.g. model not found) permanent.
            status = getattr(exc, "status_code", None)
            if status is not None and status >= 500:
                raise ConnectionError(
                    f"Ollama server error (HTTP {status}): {exc}"
                ) from exc
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

        result_dict: dict[str, Any] = parsed
        return result_dict

    # ------------------------------------------------------------------
    # IEmbeddingClient (unsupported — fails loudly)
    # ------------------------------------------------------------------

    async def embed(self, image_bytes: bytes) -> list[float]:
        """Raise: Ollama's ``/api/embed`` endpoint embeds text, not images.

        Passing image bytes to it would embed the base64 string *as text* —
        a well-formed vector with no relationship to the image's visual
        content, so every downstream cosine comparison would silently operate
        on noise. Fail loudly instead.

        Raises:
            NotImplementedError: Always. Use a dedicated image-embedding
                provider ("huggingface", "image_embedding").
        """
        raise NotImplementedError(
            "Ollama's /api/embed endpoint embeds text, not images — passing "
            "image bytes to it silently produces meaningless vectors rather "
            "than an error. Use a dedicated image-embedding provider "
            "(huggingface, image_embedding) instead."
        )

    async def embed_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Raise: see ``embed()`` — Ollama's embed endpoint doesn't do images."""
        raise NotImplementedError(
            "Ollama's /api/embed endpoint embeds text, not images — passing "
            "image bytes to it silently produces meaningless vectors rather "
            "than an error. Use a dedicated image-embedding provider "
            "(huggingface, image_embedding) instead."
        )

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
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
