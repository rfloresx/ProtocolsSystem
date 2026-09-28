"""MuxImageClient — routes IImageClient calls to the right backing provider.

A multiplexing IImageClient. Byte operations receive a full resource URI
(e.g. ``asset://uuid.jpg`` or ``immich://a1b2``) and are dispatched by scheme
to the provider registered for that scheme:

    {"asset": LocalImageClient(project_dir), "immich": ImmichClient(...)}

The project resource layer talks only to the mux, so callers never branch on
where a photo lives. Search/album operations that are not URI-scoped (browsing,
smart search, albums) are forwarded to a configured *primary* source
(the Immich source by default), since those are used to discover new photos.

Registered as the ``mux`` provider for IImageClient. Because the local route is
bound to a specific project, the app rebinds it per open project via
``set_route("asset", LocalImageClient(project_dir))``.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from protocols_system.core.registry import ProtocolsRegistry
from protocols_system.protocols.image_library import (
    AlbumResult,
    AlbumSummary,
    Asset,
    IImageClient,
)

logger = logging.getLogger(__name__)

# Default URI scheme served by each known image provider. The local upload
# source is bound per-project by the app, so it is not built from config here.
_DEFAULT_PROVIDER_SCHEMES = {
    "immich": "immich",
    "local": "asset",
}


def _uri_scheme(uri: str) -> str:
    """Return the scheme of a resource URI (e.g. "asset", "immich"), or ""."""
    if "://" not in uri:
        return ""
    return uri.split("://", 1)[0]


@ProtocolsRegistry.register("mux", IImageClient)
class MuxImageClient:
    """IImageClient that dispatches by URI scheme to backing providers.

    Can be constructed either programmatically (``routes=``) or declaratively
    from persisted settings (``providers=``). The declarative form lets the
    Service Providers UI configure sub-providers by name and scheme:

        MuxImageClient(providers=[
            {"scheme": "immich", "provider": "immich",
             "config": {"base_url": "...", "api_key": "..."}},
        ])

    Args:
        routes: Mapping of URI scheme -> backing IImageClient (programmatic).
        providers: Declarative sub-provider specs, each a dict with keys
            ``scheme`` (optional, defaults from the provider name), ``provider``
            (registered IImageClient name), and ``config`` (its __init__ kwargs).
            Sub-providers are instantiated without replacing the IImageClient
            singleton. Failures are logged and skipped.
        primary: The scheme whose provider handles non-URI-scoped operations
            (search, albums). Defaults to ``"immich"``.
    """

    def __init__(
        self,
        routes: dict[str, IImageClient] | None = None,
        providers: list[dict[str, Any]] | None = None,
        primary: str = "immich",
    ) -> None:
        self._routes: dict[str, IImageClient] = dict(routes or {})
        self._primary = primary
        if providers:
            self._build_from_specs(providers)

    def _build_from_specs(self, specs: list[dict[str, Any]]) -> None:
        """Instantiate declarative sub-provider specs into scheme routes."""
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            provider_name = spec.get("provider")
            if not provider_name:
                continue
            scheme = str(
                spec.get("scheme")
                or _DEFAULT_PROVIDER_SCHEMES.get(provider_name, provider_name)
            )
            config = spec.get("config") or {}
            try:
                client = ProtocolsRegistry.build_provider(
                    IImageClient,  # type: ignore[type-abstract]
                    provider_name,
                    config,
                )
            except Exception as exc:  # noqa: BLE001 — one bad sub-provider must not break the mux
                logger.warning(
                    "Mux: failed to build sub-provider '%s' for scheme '%s': %s",
                    provider_name,
                    scheme,
                    exc,
                )
                continue
            self._routes[scheme] = client
            logger.debug("Mux: routed scheme '%s' -> provider '%s'", scheme, provider_name)

    # ------------------------------------------------------------------
    # Route management
    # ------------------------------------------------------------------

    def set_route(self, scheme: str, client: IImageClient) -> None:
        """Bind (or rebind) the provider handling a URI scheme."""
        self._routes[scheme] = client

    def has_route(self, scheme: str) -> bool:
        """Return True if a provider is registered for the scheme."""
        return scheme in self._routes

    def _route(self, uri: str) -> tuple[IImageClient, str]:
        """Resolve a full URI to (provider, bare_id).

        Raises:
            FileNotFoundError: If no provider is registered for the URI scheme.
        """
        scheme = _uri_scheme(uri)
        provider = self._routes.get(scheme)
        if provider is None:
            raise FileNotFoundError(
                f"No image provider registered for scheme '{scheme}' (uri={uri!r})"
            )
        bare_id = uri.split("://", 1)[1] if "://" in uri else uri
        return provider, bare_id

    def _primary_provider(self) -> IImageClient:
        """Return the provider used for non-URI-scoped operations.

        Raises:
            RuntimeError: If the primary source is not configured.
        """
        provider = self._routes.get(self._primary)
        if provider is None:
            raise RuntimeError(
                f"Primary image source '{self._primary}' is not configured"
            )
        return provider

    # ------------------------------------------------------------------
    # Byte fetching (URI-scoped — routed by scheme)
    # ------------------------------------------------------------------

    async def get_asset_full(self, asset_id: str) -> bytes:
        """Return full-resolution bytes for a URI, routed by scheme."""
        provider, bare = self._route(asset_id)
        return await provider.get_asset_full(bare)

    async def get_asset_thumbnail(self, asset_id: str) -> bytes:
        """Return thumbnail bytes for a URI, routed by scheme."""
        provider, bare = self._route(asset_id)
        return await provider.get_asset_thumbnail(bare)

    # ------------------------------------------------------------------
    # Search / browse (forwarded to the primary source)
    # ------------------------------------------------------------------

    async def search_assets(
        self,
        taken_after: datetime,
        taken_before: datetime,
        **kwargs: Any,
    ) -> list[Asset]:
        return await self._primary_provider().search_assets(
            taken_after, taken_before, **kwargs
        )

    async def search_smart(self, query: str, limit: int) -> list[Asset]:
        return await self._primary_provider().search_smart(query, limit)

    async def search_people_any(self) -> list[Asset]:
        return await self._primary_provider().search_people_any()

    async def search_on_this_day(self, run_date: date) -> list[Asset]:
        return await self._primary_provider().search_on_this_day(run_date)

    # ------------------------------------------------------------------
    # Albums (forwarded to the primary source)
    # ------------------------------------------------------------------

    async def list_albums(self) -> list[AlbumSummary]:
        return await self._primary_provider().list_albums()

    async def get_album_by_name(self, name: str) -> AlbumSummary | None:
        return await self._primary_provider().get_album_by_name(name)

    async def list_album_assets(self, album_id: str) -> list[Asset]:
        return await self._primary_provider().list_album_assets(album_id)

    async def create_album(self, name: str, asset_ids: list[str]) -> AlbumResult:
        return await self._primary_provider().create_album(name, asset_ids)

    async def add_assets_to_album(self, album_id: str, asset_ids: list[str]) -> None:
        await self._primary_provider().add_assets_to_album(album_id, asset_ids)

    async def remove_assets_from_album(self, album_id: str, asset_ids: list[str]) -> None:
        await self._primary_provider().remove_assets_from_album(album_id, asset_ids)
