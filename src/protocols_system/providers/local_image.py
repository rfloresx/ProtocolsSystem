"""LocalImageClient — IImageClient over a project's local assets/ folder.

Serves photos that were uploaded directly into a project (the ``asset://``
scheme). Bytes come from ``{project_dir}/assets/`` and thumbnails from
``{project_dir}/thumbnails/`` (generated on demand). This exposes the existing
upload storage behind the same IImageClient protocol used by remote sources,
so the mux can route ``asset://`` URIs here.

The ``asset_id`` used by this client is the on-disk filename (the part after
``asset://``), not a UUID key like Immich uses.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from protocols_system.protocols.image_library import (
    AlbumResult,
    AlbumSummary,
    Asset,
)

logger = logging.getLogger(__name__)

_THUMBNAIL_MAX_SIZE = 400
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}


class LocalImageClient:
    """IImageClient backed by a single project's local asset folders.

    Not registered as a named provider in ProtocolsRegistry because it is
    bound to a specific project directory at construction time. The mux is
    given a fresh instance per open project.

    Args:
        project_dir: The project's root directory (contains assets/ and
            thumbnails/).
    """

    def __init__(self, project_dir: Path) -> None:
        self._dir = Path(project_dir)

    # ------------------------------------------------------------------
    # Byte fetching
    # ------------------------------------------------------------------

    async def get_asset_full(self, asset_id: str) -> bytes:
        """Return full-resolution bytes for a local asset (asset_id = filename)."""
        path = self._dir / "assets" / asset_id
        if not path.is_file():
            raise FileNotFoundError(f"Local asset not found: '{asset_id}'")
        return path.read_bytes()

    async def get_asset_thumbnail(self, asset_id: str) -> bytes:
        """Return thumbnail bytes for a local asset, generating on demand."""
        thumb = self._dir / "thumbnails" / asset_id
        if thumb.is_file():
            return thumb.read_bytes()

        asset = self._dir / "assets" / asset_id
        if not asset.is_file():
            raise FileNotFoundError(f"Local asset not found: '{asset_id}'")

        data = asset.read_bytes()
        generated = self._generate_thumbnail(asset_id, data)
        if generated is not None:
            return generated
        # Fall back to the full image if thumbnail generation failed.
        return data

    def _generate_thumbnail(self, filename: str, data: bytes) -> bytes | None:
        ext = Path(filename).suffix.lower()
        if ext not in _IMAGE_EXTS:
            return None
        try:
            from io import BytesIO

            from PIL import Image

            img = Image.open(BytesIO(data))
            img.thumbnail((_THUMBNAIL_MAX_SIZE, _THUMBNAIL_MAX_SIZE))

            thumb_dir = self._dir / "thumbnails"
            thumb_dir.mkdir(exist_ok=True)
            thumb_path = thumb_dir / filename

            if img.mode in ("RGBA", "P"):
                img.save(thumb_path, format="PNG", optimize=True)
            else:
                img.convert("RGB").save(
                    thumb_path, format="JPEG", quality=80, optimize=True
                )
            return thumb_path.read_bytes()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local thumbnail generation failed for %s: %s", filename, exc)
            return None

    # ------------------------------------------------------------------
    # Listing / search (folder-scan based)
    # ------------------------------------------------------------------

    def list_asset_ids(self) -> list[str]:
        """Return the filenames of all local image assets."""
        assets_dir = self._dir / "assets"
        if not assets_dir.is_dir():
            return []
        return [
            f.name
            for f in sorted(assets_dir.iterdir())
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        ]

    async def search_assets(
        self,
        taken_after: datetime,
        taken_before: datetime,
        **kwargs: Any,
    ) -> list[Asset]:
        """Return local assets as bare Asset records (no date filtering).

        Local uploads don't carry a searchable capture-date index, so this
        returns every local image; callers filter as needed.
        """
        return [
            Asset(id=fid, filename=fid, mime_type=_mime_for(fid))
            for fid in self.list_asset_ids()
        ]

    async def search_smart(self, query: str, limit: int) -> list[Asset]:
        """Smart search is not supported for local assets — returns []."""
        return []

    async def search_people_any(self) -> list[Asset]:
        """People search is not supported for local assets — returns []."""
        return []

    async def search_on_this_day(self, run_date: Any) -> list[Asset]:
        """On-this-day search is not supported for local assets — returns []."""
        return []

    # ------------------------------------------------------------------
    # Albums (not supported for the local folder source)
    # ------------------------------------------------------------------

    async def list_albums(self) -> list[AlbumSummary]:
        return []

    async def get_album_by_name(self, name: str) -> AlbumSummary | None:
        return None

    async def list_album_assets(self, album_id: str) -> list[Asset]:
        return []

    async def create_album(self, name: str, asset_ids: list[str]) -> AlbumResult:
        return AlbumResult(
            id="", name=name, success=False, message="Local source has no albums"
        )

    async def add_assets_to_album(self, album_id: str, asset_ids: list[str]) -> None:
        return None

    async def remove_assets_from_album(self, album_id: str, asset_ids: list[str]) -> None:
        return None


def _mime_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }.get(ext, "image/jpeg")
