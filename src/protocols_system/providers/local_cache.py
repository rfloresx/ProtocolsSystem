"""LocalFileCacheManager — file-based cache using JSON files on disk.

Implements ICacheManager. Each named cache is stored as a separate JSON file
inside the configured cache directory.

Config:
    cache_dir: Path to the directory where cache files are stored.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from protocols_system.core.registry import ProtocolsRegistry
from protocols_system.protocols.cache import ICache, ICacheManager

logger = logging.getLogger(__name__)


class LocalFileCache:
    """A single named cache backed by a JSON file on disk."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load cache data from disk."""
        if not self._path.exists():
            return {}
        try:
            text = self._path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            logger.warning("Cache file %s is not a dict — resetting.", self._path)
            return {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load cache %s: %s — resetting.", self._path, exc)
            return {}

    def _save(self) -> None:
        """Persist cache data to disk atomically (write to temp, then rename)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp_fd = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".tmp",
                dir=self._path.parent,
                delete=False,
                encoding="utf-8",
            )
            with tmp_fd:
                json.dump(self._data, tmp_fd, ensure_ascii=False, indent=2)
            Path(tmp_fd.name).replace(self._path)
        except OSError as exc:
            logger.error("Failed to save cache %s: %s", self._path, exc)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key, or return default if not found."""
        return self._data.get(key, default)

    def put(self, key: str, value: Any) -> None:
        """Store a value and persist to disk."""
        self._data[key] = value
        self._save()

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


@ProtocolsRegistry.register("local_file", ICacheManager)
class LocalFileCacheManager:
    """File-based cache manager that stores each cache as a JSON file.

    Args:
        cache_dir: Directory where cache JSON files are stored.
    """

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._caches: dict[str, LocalFileCache] = {}

    def get_cache(self, cache_name: str) -> ICache | None:
        """Get or create a named cache store.

        Args:
            cache_name: Logical name for the cache (used as the filename stem).

        Returns:
            An ICache instance backed by a JSON file.
        """
        if cache_name not in self._caches:
            # Sanitize the cache name for use as a filename
            safe_name = cache_name.replace("/", "_").replace("\\", "_")
            path = self._cache_dir / f"{safe_name}.json"
            self._caches[cache_name] = LocalFileCache(path)
        return self._caches[cache_name]
