"""LlamaCppClient — async HTTP client for llama.cpp server (OpenAI-compatible API).

Implements ILLMClient and IHealthCheck. Communicates via the OpenAI-compatible
chat/completions endpoint for vision tasks with structured JSON output, and the
``/v1/embeddings`` endpoint for image embeddings when the server is serving a
multimodal embedding model.

Transient connection/transport errors are retried via tenacity with exponential
backoff.

Requires: httpx, tenacity
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx
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


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from model output if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()
    return cleaned


@ProtocolsRegistry.register("llamacpp", ILLMClient)
@ProtocolsRegistry.register("llamacpp", IHealthCheck)
class LlamaCppClient:
    """Async client for a llama.cpp server via its OpenAI-compatible API.

    Connects to a llama.cpp server for vision tasks using the
    ``/v1/chat/completions`` endpoint and (optionally) ``/v1/embeddings``. The
    server must be loaded with a multimodal model (e.g. LLaVA, Qwen-VL) for
    image analysis, and started with ``--embedding`` for embeddings.

    Usage::

        async with LlamaCppClient(
            base_url="http://localhost:8080",
            vision_model="qwen-vl",
        ) as client:
            result = await client.analyze_image(img, prompt, schema)

    Args:
        base_url: The llama.cpp server URL (e.g. "http://localhost:8080").
        vision_model: Model identifier (informational — llama.cpp serves one model).
        api_key: Optional API key if server was started with --api-key.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        vision_model: str = "default",
        api_key: str = "",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._vision_model = vision_model
        self._api_key = api_key
        self._timeout = float(timeout)
        self._client: httpx.AsyncClient | None = None

        if vision_model == "default":
            logger.warning(
                "LlamaCppClient created with vision_model='default'. "
                "Set 'vision_model' in config to suppress this warning. "
                "Note: llama.cpp serves whichever model it was started with "
                "regardless of this value."
            )

    @property
    def vision_model(self) -> str:
        return self._vision_model

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def __aenter__(self) -> LlamaCppClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._build_headers(),
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
                headers=self._build_headers(),
                timeout=httpx.Timeout(self._timeout),
            )
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

        Encodes the image as base64 and sends it to the llama.cpp server's
        OpenAI-compatible chat/completions endpoint with JSON response format.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.).
            prompt: Text prompt for the vision model.
            return_schema: JSON-schema-like dict for response validation.

        Returns:
            Parsed JSON dict on success, or {"error": "..."} on failure.

        Raises:
            ConnectionError: On transient connection errors (retried by tenacity).
        """
        # Normalize every non-JPEG/PNG format (HEIC/TIFF/BMP/GIF/WebP),
        # downscale large images, and fail cleanly if conversion isn't
        # possible rather than sending undecodable bytes.
        try:
            image_bytes, mime_type = prepare_image_for_vision(image_bytes)
        except ImageConversionError as exc:
            return {"error": f"Image conversion failed: {exc}"}

        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        logger.debug(
            "analyze_image: sending %d bytes as %s to model %s",
            len(image_bytes), mime_type, self._vision_model,
        )

        payload: dict[str, Any] = {
            "model": self._vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "cache_prompt": True,
        }

        try:
            response = await self._get_client().post(
                "/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise ConnectionError(
                    f"llama.cpp server error (HTTP {exc.response.status_code}): "
                    f"{exc.response.text}"
                ) from exc
            return {
                "error": f"llama.cpp API error (HTTP {exc.response.status_code}): "
                f"{exc.response.text}"
            }
        except httpx.ConnectTimeout as exc:
            # Couldn't even establish the connection — safe to retry.
            raise ConnectionError(
                f"Timeout connecting to llama.cpp server at {self._base_url}: {exc}"
            ) from exc
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
            # A read timeout means the request *was* sent and the server is
            # (still) processing it — retrying would pile a second expensive
            # inference on top of the first. Return a per-image error instead
            # of raising ConnectionError (which tenacity would retry 3x).
            return {
                "error": (
                    f"llama.cpp request timed out after {self._timeout}s "
                    f"(server may still be processing): {exc}"
                )
            }
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot reach llama.cpp server at {self._base_url}: {exc}"
            ) from exc
        except (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError) as exc:
            # Transport-level errors (connection reset mid-stream, protocol
            # violation) are transient and retryable.
            raise ConnectionError(
                f"Transport error talking to llama.cpp server at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            # Any other timeout type — be conservative and don't retry a
            # possibly-still-running inference.
            return {"error": f"llama.cpp request timed out: {exc}"}

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid JSON in server response: {exc}", "raw": response.text}

        try:
            raw_content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            return {"error": f"Unexpected response structure: {exc}", "raw": data}

        if not raw_content or not raw_content.strip():
            return {"error": "Empty content in model response", "raw": ""}

        cleaned = _strip_markdown_fences(raw_content)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return {"error": f"Model response is not valid JSON: {exc}", "raw": raw_content}

        validation_error = validate_against_schema(parsed, return_schema)
        if validation_error is not None:
            return {"error": f"Schema validation failed: {validation_error}", "raw": parsed}

        result_dict: dict[str, Any] = parsed
        return result_dict

    @_retry_policy
    async def embed(self, image_bytes: bytes) -> list[float]:
        """Generate an embedding vector for an image.

        Sends the image as base64 to the llama.cpp server's ``/v1/embeddings``
        endpoint. The server must be running with ``--embedding`` enabled
        **and** loaded with a multimodal embedding model.

        .. warning::
            ``/v1/embeddings`` on a text-only embedding model will embed the
            request payload as text rather than the image's visual content,
            silently producing a vector unrelated to the picture. Only use
            this against a server you know is serving a multimodal embedding
            model; otherwise prefer a dedicated image-embedding provider
            ("huggingface", "image_embedding").

        Raises:
            ConnectionError: If the server is unreachable (retried).
            RuntimeError: If the server returns an error response.
        """
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        payload: dict[str, Any] = {
            "input": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                }
            ],
            "model": self._vision_model,
        }

        try:
            response = await self._get_client().post(
                "/v1/embeddings",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise ConnectionError(
                    f"llama.cpp server error (HTTP {exc.response.status_code}): "
                    f"{exc.response.text}"
                ) from exc
            raise RuntimeError(
                f"llama.cpp embedding error (HTTP {exc.response.status_code}): "
                f"{exc.response.text}"
            ) from exc
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot reach llama.cpp server at {self._base_url}: {exc}"
            ) from exc
        except (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError) as exc:
            raise ConnectionError(
                f"Transport error talking to llama.cpp server at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Timeout connecting to llama.cpp server at {self._base_url}: {exc}"
            ) from exc

        data = response.json()
        try:
            embedding = data["data"][0]["embedding"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected embedding response structure: {exc}"
            ) from exc

        return [float(x) for x in embedding]

    async def close(self) -> None:
        """Release HTTP client resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # IHealthCheck
    # ------------------------------------------------------------------

    @property
    def health_check_url(self) -> str:
        """The llama.cpp server base URL."""
        return self._base_url

    async def health_check(self) -> bool:
        """Check connectivity to the llama.cpp server.

        Tries /health first (native llama.cpp), then falls back to /v1/models
        (OpenAI-compatible endpoint).

        Returns:
            True if the server responds successfully, False otherwise.
        """
        try:
            async with httpx.AsyncClient(
                timeout=10.0, headers=self._build_headers()
            ) as client:
                resp = await client.get(f"{self._base_url}/health")
                if resp.status_code == 200:
                    return True
                resp = await client.get(f"{self._base_url}/v1/models")
                return resp.status_code == 200
        except Exception:
            return False
