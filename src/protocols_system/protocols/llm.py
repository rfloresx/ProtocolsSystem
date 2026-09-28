"""LLM / Vision protocol — image analysis and text generation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ILLMClient(Protocol):
    """Vision / LLM:Vision model backend used for aesthetic quality scoring."""

    async def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        return_schema: dict[str, Any],
    ) -> dict[str, Any]: ...
