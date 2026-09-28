"""Protocol definitions for the Protocols System.

Each protocol lives in its own module alongside the data classes it produces/consumes.
Import protocols individually or use this package for convenience re-exports.
"""

from .cache import ICache, ICacheManager
from .embedding import IEmbeddingClient
from .geo import IGeoClient, PlaceCandidate
from .health import IHealthCheck
from .image_library import (
    AlbumResult,
    AlbumSummary,
    Asset,
    IImageClient,
)
from .llm import ILLMClient
from .progress import IProgressReporter

__all__ = [
    # Image Library
    "IImageClient",
    "Asset",
    "AlbumResult",
    "AlbumSummary",
    # LLM
    "ILLMClient",
    # Embedding
    "IEmbeddingClient",
    # Geo
    "IGeoClient",
    "PlaceCandidate",
    # Progress
    "IProgressReporter",
    # Health
    "IHealthCheck",
    # Cache
    "ICache",
    "ICacheManager",
]
