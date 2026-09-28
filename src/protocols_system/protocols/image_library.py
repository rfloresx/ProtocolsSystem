"""Image Library protocol — source of photo assets and album management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Asset:
    """A single photo/video asset from the library.

    The capture timestamp is stored on ``captured_at``. ``taken_at`` is kept as
    a synced alias so consumers using either name interoperate.
    """

    id: str
    filename: str
    mime_type: str = "image/jpeg"
    captured_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    width: int | None = None
    height: int | None = None
    file_size: int | None = None
    is_favorite: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def taken_at(self) -> datetime | None:
        """Alias for :attr:`captured_at`."""
        return self.captured_at

    @taken_at.setter
    def taken_at(self, value: datetime | None) -> None:
        self.captured_at = value


@dataclass
class AlbumSummary:
    """Lightweight album reference returned by listing operations."""

    id: str
    name: str
    asset_count: int = 0
    created_at: datetime | None = None


@dataclass
class AlbumResult:
    """Result of an album creation or modification operation."""

    id: str
    name: str
    asset_count: int = 0
    success: bool = True
    message: str = ""
    url: str = ""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IImageClient(Protocol):
    """Image Library:Photo library provider.

    The source of assets and the target for album publishing.
    """

    async def search_assets(
        self,
        taken_after: datetime,
        taken_before: datetime,
        **kwargs: Any,
    ) -> list[Asset]: ...

    async def get_asset_thumbnail(self, asset_id: str) -> bytes: ...

    async def list_albums(self) -> list[AlbumSummary]: ...

    async def create_album(self, name: str, asset_ids: list[str]) -> AlbumResult: ...

    async def search_smart(self, query: str, limit: int) -> list[Asset]: ...

    async def search_people_any(self) -> list[Asset]: ...

    async def search_on_this_day(self, run_date: date) -> list[Asset]: ...

    async def get_album_by_name(self, name: str) -> AlbumSummary | None: ...

    async def list_album_assets(self, album_id: str) -> list[Asset]: ...

    async def add_assets_to_album(self, album_id: str, asset_ids: list[str]) -> None: ...

    async def remove_assets_from_album(self, album_id: str, asset_ids: list[str]) -> None: ...

    async def get_asset_full(self, asset_id: str) -> bytes: ...
