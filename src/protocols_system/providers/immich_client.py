"""ImmichClient — async wrapper around immichpy for the IImageClient protocol.

References photos on a local Immich server by asset id and streams their bytes
on demand. Nothing is copied into a PhotoStudio project; only lightweight
``immich://{asset_id}`` references are stored.

Implements this project's IImageClient (photo library source) and IHealthCheck.
Wraps the ``immichpy`` typed client. Transient (5xx / network) errors are
retried with exponential backoff via tenacity.

Requires: immichpy, tenacity
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from typing import Any
from uuid import UUID

from immichpy import AsyncClient
from immichpy.client.generated.exceptions import (
    ForbiddenException,
    NotFoundException,
    ServiceException,
    UnauthorizedException,
)
from immichpy.client.generated.models.asset_media_size import AssetMediaSize
from immichpy.client.generated.models.bulk_ids_dto import BulkIdsDto
from immichpy.client.generated.models.create_album_dto import CreateAlbumDto
from immichpy.client.generated.models.metadata_search_dto import MetadataSearchDto
from immichpy.client.generated.models.smart_search_dto import SmartSearchDto
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from protocols_system.core.registry import ProtocolsRegistry
from protocols_system.protocols.health import IHealthCheck
from protocols_system.protocols.image_library import (
    AlbumResult,
    AlbumSummary,
    Asset,
    IImageClient,
)

logger = logging.getLogger(__name__)


def _is_transient(exc: BaseException) -> bool:
    """Return True for errors worth retrying (5xx, network errors)."""
    if isinstance(exc, ServiceException):
        return True
    if isinstance(exc, (OSError, ConnectionError)):
        return True
    return False


_retry_policy = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _parse_dt(value: Any) -> datetime | None:
    """Parse a datetime that may arrive as a string or datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _map_asset(dto: Any) -> Asset:
    """Convert an immichpy AssetResponseDto to this project's Asset."""
    captured_at = _parse_dt(
        getattr(dto, "local_date_time", None) or getattr(dto, "file_created_at", None)
    )

    latitude: float | None = None
    longitude: float | None = None
    width: int | None = getattr(dto, "width", None)
    height: int | None = getattr(dto, "height", None)
    file_size: int | None = None
    metadata: dict[str, Any] = {}

    checksum = getattr(dto, "checksum", None)
    if checksum:
        metadata["checksum"] = str(checksum)

    exif = getattr(dto, "exif_info", None)
    if exif is not None:
        latitude = getattr(exif, "latitude", None)
        longitude = getattr(exif, "longitude", None)
        width = getattr(exif, "exif_image_width", None) or width
        height = getattr(exif, "exif_image_height", None) or height
        file_size = getattr(exif, "file_size_in_byte", None)
        metadata["exif"] = {
            "date_time_original": getattr(exif, "date_time_original", None),
            "latitude": latitude,
            "longitude": longitude,
            "file_size_in_byte": file_size,
        }

    return Asset(
        id=str(dto.id),
        filename=getattr(dto, "original_file_name", None) or "",
        mime_type=getattr(dto, "original_mime_type", None) or "image/jpeg",
        captured_at=captured_at,
        latitude=float(latitude) if latitude is not None else None,
        longitude=float(longitude) if longitude is not None else None,
        width=int(width) if width is not None else None,
        height=int(height) if height is not None else None,
        file_size=int(file_size) if file_size is not None else None,
        is_favorite=bool(getattr(dto, "is_favorite", False)),
        metadata=metadata,
    )


@ProtocolsRegistry.register("immich", IImageClient)
@ProtocolsRegistry.register("immich", IHealthCheck)
class ImmichClient:
    """Async client wrapping immichpy, implementing IImageClient + IHealthCheck.

    The client lazily opens an immichpy ``AsyncClient`` on first use and reuses
    it. It can also be used explicitly as an async context manager. A small
    in-memory LRU byte cache avoids refetching the same asset within a render
    or export pass.

    Args:
        base_url: The Immich server URL (e.g. "http://localhost:2283").
        api_key: An Immich API key.
        cache_size: Max number of asset byte blobs to keep in memory.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:2283",
        api_key: str = "",
        cache_size: int = 64,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._cache_size = max(0, cache_size)
        self._byte_cache: dict[str, bytes] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _open(self) -> AsyncIterator[AsyncClient]:
        """Yield a fresh immichpy AsyncClient scoped to a single operation.

        A new client is opened (and closed) per call rather than persisted on
        the instance. The underlying transport is bound to the running event
        loop, and this client is driven from synchronous code via a run-in-loop
        bridge that uses a fresh loop per call — so a cached client would be
        bound to an already-closed loop and raise "Event loop is closed".
        """
        client = AsyncClient(base_url=self._base_url, api_key=self._api_key)
        await client.__aenter__()
        try:
            yield client
        finally:
            await client.__aexit__(None, None, None)

    def is_available(self) -> bool:
        """Cheap, non-network check that the client is configured."""
        return bool(self._api_key and self._base_url)

    # ------------------------------------------------------------------
    # IHealthCheck
    # ------------------------------------------------------------------

    @property
    def health_check_url(self) -> str:
        """The Immich server base URL."""
        return self._base_url

    async def health_check(self) -> bool:
        """Check connectivity to the Immich server. Never raises."""
        try:
            async with self._open() as client:
                await client.server.ping_server()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Byte fetching (used by the resource layer)
    # ------------------------------------------------------------------

    @_retry_policy
    async def get_asset_full(self, asset_id: str) -> bytes:
        """Return full-resolution original bytes for an asset.

        Raises:
            FileNotFoundError: If the asset media does not exist (HTTP 404).
        """
        return await self._view_asset(asset_id, AssetMediaSize.ORIGINAL, "orig")

    @_retry_policy
    async def get_asset_thumbnail(self, asset_id: str) -> bytes:
        """Return thumbnail bytes for an asset.

        Raises:
            FileNotFoundError: If the asset media does not exist (HTTP 404).
        """
        return await self._view_asset(asset_id, AssetMediaSize.THUMBNAIL, "thumb")

    async def _view_asset(
        self, asset_id: str, size: Any, cache_prefix: str
    ) -> bytes:
        cache_key = f"{cache_prefix}:{asset_id}"
        if cache_key in self._byte_cache:
            return self._byte_cache[cache_key]

        try:
            async with self._open() as client:
                result = await client.assets.view_asset(id=UUID(asset_id), size=size)
        except NotFoundException as exc:
            raise FileNotFoundError(
                f"Immich asset media not found (asset_id={asset_id})"
            ) from exc
        except (UnauthorizedException, ForbiddenException):
            raise

        data = bytes(result)
        if self._cache_size > 0:
            self._byte_cache[cache_key] = data
            while len(self._byte_cache) > self._cache_size:
                self._byte_cache.pop(next(iter(self._byte_cache)))
        return data

    @_retry_policy
    async def get_asset_metadata(self, asset_id: str) -> Asset:
        """Fetch a single asset's metadata (EXIF, dimensions, date)."""
        try:
            async with self._open() as client:
                dto = await client.assets.get_asset_info(id=UUID(asset_id))
        except NotFoundException as exc:
            raise FileNotFoundError(
                f"Immich asset not found (asset_id={asset_id})"
            ) from exc
        return _map_asset(dto)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @_retry_policy
    async def search_assets(
        self,
        taken_after: datetime,
        taken_before: datetime,
        **kwargs: Any,
    ) -> list[Asset]:
        """Search assets within a capture-date range (paginated)."""
        return await self._paginated_metadata(
            takenAfter=_ensure_utc(taken_after),
            takenBefore=_ensure_utc(taken_before),
            withExif=True,
        )

    @_retry_policy
    async def search_smart(self, query: str, limit: int) -> list[Asset]:
        """CLIP/smart search, returning up to `limit` assets by relevance."""
        if limit <= 0:
            return []

        results: list[Asset] = []
        page = 1
        page_size = min(limit, 1000)

        async with self._open() as client:
            while len(results) < limit:
                dto = SmartSearchDto(
                    query=query, page=page, size=page_size, withExif=True
                )
                response = await client.search.search_smart(dto)
                for asset_dto in response.assets.items:
                    results.append(_map_asset(asset_dto))
                    if len(results) >= limit:
                        break
                if response.assets.next_page is None:
                    break
                page = int(response.assets.next_page)

        return results

    @_retry_policy
    async def search_people_any(self) -> list[Asset]:
        """Return all assets that contain at least one recognized face."""
        return await self._paginated_metadata(withPeople=True, withExif=True)

    @_retry_policy
    async def search_on_this_day(self, run_date: date) -> list[Asset]:
        """Return assets captured on the same month/day across all years."""
        all_assets = await self._paginated_metadata(withExif=True)
        return [
            a
            for a in all_assets
            if a.captured_at is not None
            and a.captured_at.month == run_date.month
            and a.captured_at.day == run_date.day
        ]

    async def _paginated_metadata(self, **fields: Any) -> list[Asset]:
        results: list[Asset] = []
        page = 1
        async with self._open() as client:
            while True:
                dto = MetadataSearchDto(page=page, **fields)
                response = await client.search.search_assets(dto)
                results.extend(_map_asset(a) for a in response.assets.items)
                if response.assets.next_page is None:
                    break
                page = int(response.assets.next_page)
        return results

    # ------------------------------------------------------------------
    # Albums
    # ------------------------------------------------------------------

    @_retry_policy
    async def list_albums(self) -> list[AlbumSummary]:
        """Return a summary of all albums on the server."""
        async with self._open() as client:
            album_dtos = await client.albums.get_all_albums()
        return [
            AlbumSummary(
                id=str(dto.id),
                name=dto.album_name,
                asset_count=int(getattr(dto, "asset_count", 0) or 0),
                created_at=_parse_dt(getattr(dto, "created_at", None)),
            )
            for dto in album_dtos
        ]

    async def get_album_by_name(self, name: str) -> AlbumSummary | None:
        """Find an album by exact name, or None."""
        for album in await self.list_albums():
            if album.name == name:
                return album
        return None

    @_retry_policy
    async def list_album_assets(self, album_id: str) -> list[Asset]:
        """Return all assets contained in the given album."""
        async with self._open() as client:
            album = await client.albums.get_album_info(id=UUID(album_id))
        assets = getattr(album, "assets", None) or []
        return [_map_asset(a) for a in assets]

    @_retry_policy
    async def create_album(self, name: str, asset_ids: list[str]) -> AlbumResult:
        """Create a new album with the given assets."""
        try:
            dto = CreateAlbumDto(
                albumName=name,
                assetIds=[UUID(aid) for aid in asset_ids],
            )
            async with self._open() as client:
                result = await client.albums.create_album(dto)
        except (UnauthorizedException, ForbiddenException) as exc:
            return AlbumResult(id="", name=name, success=False, message=str(exc))
        return AlbumResult(
            id=str(result.id),
            name=result.album_name,
            asset_count=len(asset_ids),
        )

    @_retry_policy
    async def add_assets_to_album(self, album_id: str, asset_ids: list[str]) -> None:
        """Add assets to an existing album."""
        dto = BulkIdsDto(ids=[UUID(aid) for aid in asset_ids])
        async with self._open() as client:
            await client.albums.add_assets_to_album(UUID(album_id), dto)

    @_retry_policy
    async def remove_assets_from_album(
        self, album_id: str, asset_ids: list[str]
    ) -> None:
        """Remove assets from an existing album."""
        dto = BulkIdsDto(ids=[UUID(aid) for aid in asset_ids])
        async with self._open() as client:
            await client.albums.remove_asset_from_album(UUID(album_id), dto)
