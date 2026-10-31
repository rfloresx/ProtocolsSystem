"""HuggingFaceEmbeddingClient — local CLIP-family image embeddings.

Implements IEmbeddingClient and IHealthCheck using a local HuggingFace
Transformers model (AutoModel + AutoProcessor). Any CLIP-family model can be
loaded by its HuggingFace Hub ID (e.g. "openai/clip-vit-base-patch32").

Two embedding APIs are provided:

* ``embed(image_bytes)`` — single-image path; satisfies the base
  ``IEmbeddingClient`` protocol.
* ``embed_batch(image_bytes_list)`` — processes a list of images in batched
  forward passes (``batch_size`` images per pass). On GPU this is significantly
  faster than calling ``embed()`` N times.

Requires the optional ``huggingface`` dependency group (transformers + torch).
Heavy imports (torch, transformers, PIL) are deferred until runtime so the
module stays importable without those packages installed.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING, Any

from protocols_system.core.registry import ProtocolsRegistry
from protocols_system.protocols.embedding import IEmbeddingClient
from protocols_system.protocols.health import IHealthCheck

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

# Upper bound (width, height) for images decoded ahead of the embedding
# forward pass. CLIP-family processors resize to a small fixed size anyway
# (commonly 224x224), so there is no benefit to keeping a decoded image at
# full camera resolution around in memory.
_DECODE_MAX_SIZE = (512, 512)


def _check_dependencies() -> None:
    """Verify that optional HuggingFace dependencies are installed.

    Raises:
        ImportError: With a clear message if transformers or torch are missing.
    """
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "HuggingFaceEmbeddingClient requires the 'transformers' package. "
            "Install the 'huggingface' extra (transformers + torch)."
        ) from exc

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "HuggingFaceEmbeddingClient requires the 'torch' package. "
            "Install the 'huggingface' extra (transformers + torch)."
        ) from exc


@ProtocolsRegistry.register("huggingface", IEmbeddingClient)
@ProtocolsRegistry.register("huggingface", IHealthCheck)
class HuggingFaceEmbeddingClient:
    """Async embedding client using HuggingFace AutoModel + AutoProcessor.

    Loads any CLIP-family (or compatible vision) model via AutoModel from the
    HuggingFace Hub and encodes images into vector embeddings. For CLIP/SigLIP
    models, ``get_image_features()`` is called directly; for generic vision
    models the forward pass output is pooled.

    Since model inference is CPU/GPU-bound, calls are dispatched via
    ``asyncio.to_thread`` to avoid blocking the event loop. The model is loaded
    in ``__aenter__`` and released in ``__aexit__``.

    Args:
        model_name: HuggingFace Hub model ID.
        device: Compute device: "cpu", "cuda", "cuda:N", "mps", or "auto".
        batch_size: Maximum number of images per forward pass in ``embed_batch``.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._device_arg = device
        self._batch_size = int(batch_size)
        self._device: str | None = None
        self._model: AutoModel | None = None
        self._processor: AutoProcessor | None = None
        self._has_get_image_features: bool = False
        # Serializes forward passes on the shared model so several concurrent
        # embed()/embed_batch() calls can't run against it at once (risking
        # VRAM OOM on GPU, MPS thread-safety issues, CPU oversubscription),
        # and guards close() so the model can't be nulled from under an
        # in-flight forward pass.
        self._inference_lock = asyncio.Lock()

    @property
    def embed_model(self) -> str:
        """The HuggingFace model ID."""
        return self._model_name

    @property
    def device(self) -> str | None:
        """The resolved torch device string (available after __aenter__)."""
        return self._device

    @property
    def batch_size(self) -> int:
        """Images per forward pass used by embed_batch."""
        return self._batch_size

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> HuggingFaceEmbeddingClient:
        """Load the model and processor on the configured device."""
        _check_dependencies()
        await asyncio.to_thread(self._load_model)
        logger.info(
            "Loaded HuggingFace model '%s' on device '%s'",
            self._model_name,
            self._device,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        """Release model and free GPU memory."""
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self, torch: "Any") -> str:
        """Resolve the configured device string to a concrete torch device.

        ``auto`` considers CUDA then Apple's Metal backend (``mps``) then CPU.
        An explicit ``mps``/``cuda``/``cuda:N`` is honored and validated for
        availability; an unknown device raises a clear error rather than
        silently running on CPU.
        """
        arg = (self._device_arg or "auto").strip().lower()

        def _cuda_available() -> bool:
            return bool(torch.cuda.is_available())

        def _mps_available() -> bool:
            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None) if backends is not None else None
            return bool(mps is not None and mps.is_available())

        if arg == "auto":
            if _cuda_available():
                return "cuda"
            if _mps_available():
                return "mps"
            return "cpu"
        if arg == "cpu":
            return "cpu"
        if arg == "mps":
            if not _mps_available():
                raise RuntimeError(
                    "device='mps' requested but the MPS (Apple Metal) backend "
                    "is not available. Use device='cpu' or device='auto'."
                )
            return "mps"
        if arg == "cuda" or arg.startswith("cuda:"):
            if not _cuda_available():
                raise RuntimeError(
                    f"device={self._device_arg!r} requested but CUDA is not "
                    "available. Use device='cpu' or device='auto' instead."
                )
            return arg
        raise RuntimeError(
            f"Unknown device {self._device_arg!r}. Expected one of: "
            "'auto', 'cpu', 'cuda', 'cuda:N', 'mps'."
        )

    def _load_model(self) -> None:
        """Load AutoModel + AutoProcessor synchronously.

        Raises:
            RuntimeError: If CUDA/MPS is requested but unavailable, or if the
                model cannot be loaded.
        """
        import torch
        from transformers import AutoModel, AutoProcessor

        self._device = self._resolve_device(torch)

        try:
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{self._model_name}': {exc}"
            ) from exc

        # Detect CLIP/SigLIP — these expose get_image_features() directly.
        self._has_get_image_features = hasattr(self._model, "get_image_features")

    def _run_forward(self, images: list[PILImage.Image]) -> list[list[float]]:
        """Run a single batched forward pass on a list of PIL images.

        Returns one L2-normalized embedding vector per input image.

        Raises:
            RuntimeError: If the model is not loaded or output is unrecognised.
        """
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError(
                "Model not loaded. Use 'async with' or call __aenter__ first."
            )

        inputs = self._processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items() if hasattr(v, "to")}

        with torch.inference_mode():
            if self._has_get_image_features:
                pixel_inputs = {k: v for k, v in inputs.items() if k == "pixel_values"}
                outputs = self._model.get_image_features(**pixel_inputs)
                if hasattr(outputs, "pooler_output"):
                    outputs = outputs.pooler_output
                elif hasattr(outputs, "image_embeds"):
                    outputs = outputs.image_embeds
            else:
                outputs = self._model(**inputs)
                if hasattr(outputs, "image_embeds"):
                    outputs = outputs.image_embeds
                elif hasattr(outputs, "pooler_output"):
                    outputs = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    outputs = outputs.last_hidden_state.mean(dim=1)
                else:
                    raise RuntimeError(
                        f"Model '{self._model_name}' output has no recognised "
                        "embedding attribute (image_embeds, pooler_output, "
                        "last_hidden_state)."
                    )

        # L2-normalize so inner product == cosine similarity, matching the
        # image-embedding server provider — otherwise switching providers
        # silently changes the vector magnitude and what a similarity
        # threshold means downstream.
        normalized = torch.nn.functional.normalize(outputs, p=2, dim=1)
        return [[float(x) for x in row] for row in normalized.detach().cpu()]

    def _encode_image(self, image_bytes: bytes) -> list[float]:
        """Run synchronous inference for a single image."""
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._run_forward([image])[0]

    def _decode_for_batch(self, image_bytes: bytes) -> "PILImage.Image":
        """Decode one image, downscaled for the embedding forward pass."""
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        # draft() is a no-op for non-JPEG formats and only ever shrinks the
        # image, so it is safe to call unconditionally.
        img.draft("RGB", _DECODE_MAX_SIZE)
        rgb_img = img.convert("RGB")
        rgb_img.thumbnail(_DECODE_MAX_SIZE)
        return rgb_img

    def _encode_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Run batched synchronous inference over a list of raw image bytes.

        Images that fail to decode are returned as ``None`` at their original
        index. Processes images in chunks of ``self._batch_size`` per forward
        pass, decoding each chunk just before its pass, so peak memory stays
        bounded by roughly ``batch_size`` decoded images.
        """
        results: list[list[float] | None] = [None] * len(image_bytes_list)

        for chunk_start in range(0, len(image_bytes_list), self._batch_size):
            chunk_bytes = image_bytes_list[chunk_start : chunk_start + self._batch_size]

            indices: list[int] = []
            images: list["PILImage.Image"] = []
            for offset, image_bytes in enumerate(chunk_bytes):
                idx = chunk_start + offset
                try:
                    images.append(self._decode_for_batch(image_bytes))
                    indices.append(idx)
                except Exception as exc:
                    logger.warning(
                        "Failed to decode image at batch index %d: %s", idx, exc
                    )

            if not images:
                continue

            try:
                embeddings = self._run_forward(images)
                for idx, embedding in zip(indices, embeddings):
                    results[idx] = embedding
            except Exception as exc:
                logger.warning(
                    "Forward pass failed for batch chunk starting at index %d: %s",
                    chunk_start,
                    exc,
                )
            finally:
                for img in images:
                    img.close()

        return results

    # ------------------------------------------------------------------
    # IEmbeddingClient
    # ------------------------------------------------------------------

    async def embed(self, image_bytes: bytes) -> list[float]:
        """Compute an embedding vector for a single image.

        Dispatches synchronous model inference to a thread pool so the async
        event loop is not blocked.

        Raises:
            RuntimeError: If the model has not been loaded via __aenter__.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Use 'async with' or call __aenter__ first."
            )
        async with self._inference_lock:
            if self._model is None:
                raise RuntimeError("Model was released while a request was pending.")
            return await asyncio.to_thread(self._encode_image, image_bytes)

    async def embed_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]:
        """Compute embedding vectors for a list of images in batched passes.

        Images that fail to decode are returned as ``None`` at their original
        index.

        Raises:
            RuntimeError: If the model has not been loaded via __aenter__.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Use 'async with' or call __aenter__ first."
            )
        if not image_bytes_list:
            return []
        async with self._inference_lock:
            if self._model is None:
                raise RuntimeError("Model was released while a request was pending.")
            return await asyncio.to_thread(self._encode_batch, image_bytes_list)

    async def close(self) -> None:
        """Release the model, processor, and GPU memory.

        Acquires the inference lock first so the model isn't nulled out from
        under an in-flight forward pass.
        """
        async with self._inference_lock:
            self._model = None
            self._processor = None

            if self._device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

            self._device = None
        logger.info("Released HuggingFace model '%s'", self._model_name)

    # ------------------------------------------------------------------
    # IHealthCheck
    # ------------------------------------------------------------------

    @property
    def health_check_url(self) -> str:
        """The model name (local provider, no URL)."""
        return f"local://{self._model_name}"

    async def health_check(self) -> bool:
        """Return True if the local model and processor are loaded."""
        return self._model is not None and self._processor is not None
