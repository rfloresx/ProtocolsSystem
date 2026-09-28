"""GooglePlacesGeoClient — reverse geocoding via Google Maps Places API.

Implements IGeoClient and IHealthCheck. Resolves GPS coordinates into ranked
lists of nearby place candidates using the googlemaps SDK.

Requires: googlemaps
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from typing import Any

from protocols_system.core.registry import ProtocolsRegistry
from protocols_system.protocols.cache import ICacheManager
from protocols_system.protocols.geo import IGeoClient, PlaceCandidate
from protocols_system.protocols.health import IHealthCheck

logger = logging.getLogger(__name__)


def _check_dependencies() -> None:
    """Verify that the optional googlemaps package is installed."""
    try:
        import googlemaps  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "GooglePlacesGeoClient requires the 'googlemaps' package. "
            "Install it with: pip install googlemaps"
        ) from exc


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points (meters)."""
    earth_radius = 6_371_000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return earth_radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Pattern to parse plus_code compound_code: "CODE+CODE City, STATE, COUNTRY"
_COMPOUND_CODE_PATTERN = re.compile(
    r"[23456789CFGHJMPQRVWX+]+\s+(.+?),\s*([A-Z]{2,}),\s*([A-Z]{2,})"
)


def _make_cache_key(lat: float, lon: float, radius: int, max_results: int) -> str:
    """Build a deterministic cache key from query parameters.

    Coordinates are rounded to 4 decimal places (~11m precision) so that
    nearby queries within the same block share a cache entry.
    """
    return f"{round(lat, 4)}:{round(lon, 4)}:r{radius}:n{max_results}"


@ProtocolsRegistry.register("google_places", IGeoClient)
@ProtocolsRegistry.register("google_places", IHealthCheck)
class GooglePlacesGeoClient:
    """Reverse geocoding via Google Maps Places API.

    Wraps googlemaps.Client.places_nearby() with pagination support.
    Parses plus_code compound_code for city/state/country extraction.

    If an ICacheManager instance is available in the registry, results are
    cached under the "google_places_geo" cache name to avoid redundant API calls.

    Args:
        api_key: Google Maps API key.
        default_radius: Default search radius in meters.
        max_results: Default maximum number of candidates to return.
    """

    _CACHE_NAME = "google_places_geo"

    def __init__(
        self,
        api_key: str,
        default_radius: int = 1000,
        max_results: int = 10,
    ) -> None:
        _check_dependencies()
        import googlemaps

        self._client = googlemaps.Client(key=api_key)
        self._api_key = api_key
        self._default_radius = default_radius
        self._max_results = max_results

    # ------------------------------------------------------------------
    # IHealthCheck
    # ------------------------------------------------------------------

    @property
    def health_check_url(self) -> str:
        """Display URL for health check status."""
        return "https://maps.googleapis.com/maps/api/place/nearbysearch"

    async def health_check(self) -> bool:
        """Check connectivity by performing a minimal places_nearby request."""
        try:
            await asyncio.to_thread(
                self._client.places_nearby,
                location=(0.0, 0.0),
                radius=1,
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # IGeoClient
    # ------------------------------------------------------------------

    async def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int = 1000,
        max_results: int = 10,
    ) -> list[PlaceCandidate]:
        """Resolve GPS coordinates into ranked nearby place candidates.

        Results are cached via ICacheManager (if available) so repeated queries
        for the same location skip the API call.

        Args:
            latitude: Latitude in degrees.
            longitude: Longitude in degrees.
            radius_meters: Search radius in meters.
            max_results: Maximum candidates to return.

        Returns:
            List of PlaceCandidate sorted by distance from the query point.
        """
        radius = radius_meters or self._default_radius
        limit = max_results or self._max_results

        # --- Cache lookup ---
        cache = self._get_cache()
        cache_key = _make_cache_key(latitude, longitude, radius, limit)

        if cache is not None and cache_key in cache:
            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for %s", cache_key)
                return [PlaceCandidate(**entry) for entry in cached]

        # --- API call ---
        raw_results = await asyncio.to_thread(
            self._fetch_nearby, latitude, longitude, radius, limit
        )

        candidates: list[PlaceCandidate] = []
        for place in raw_results:
            loc = place.get("geometry", {}).get("location", {})
            plat = float(loc.get("lat", 0.0))
            plng = float(loc.get("lng", 0.0))
            distance = _haversine_meters(latitude, longitude, plat, plng)

            name = place.get("name", "")
            city, state, country = self._parse_location(place)

            candidates.append(
                PlaceCandidate(
                    name=name,
                    city=city,
                    state=state,
                    country=country,
                    latitude=plat,
                    longitude=plng,
                    distance_meters=round(distance, 1),
                    place_type=",".join(place.get("types", [])),
                    raw=place,
                )
            )

        candidates.sort(key=lambda c: c.distance_meters)
        candidates = candidates[:limit]

        # --- Cache store (without raw data to keep cache lightweight) ---
        if cache is not None:
            serializable = [
                {
                    "name": c.name,
                    "city": c.city,
                    "state": c.state,
                    "country": c.country,
                    "latitude": c.latitude,
                    "longitude": c.longitude,
                    "distance_meters": c.distance_meters,
                    "place_type": c.place_type,
                }
                for c in candidates
            ]
            cache.put(cache_key, serializable)
            logger.debug("Cached %d candidates for %s", len(candidates), cache_key)

        return candidates

    def _fetch_nearby(
        self, lat: float, lng: float, radius: int, max_results: int
    ) -> list[dict[str, Any]]:
        """Synchronous Places API call with pagination."""
        results: list[dict[str, Any]] = []
        response: dict[str, Any] = self._client.places_nearby(
            location=(lat, lng), radius=radius, open_now=False
        )
        results.extend(response.get("results", []))

        # Follow next_page_token pagination
        while "next_page_token" in response and len(results) < max_results:
            time.sleep(2)  # Token needs a short delay to become valid
            response = self._client.places_nearby(
                page_token=response["next_page_token"]
            )
            results.extend(response.get("results", []))

        return results[:max_results]

    def _parse_location(self, place: dict[str, Any]) -> tuple[str, str, str]:
        """Extract city/state/country from plus_code compound_code or vicinity.

        Returns:
            Tuple of (city, state, country). Empty strings for unavailable fields.
        """
        plus_code = place.get("plus_code", {})
        compound = plus_code.get("compound_code", "")

        if compound:
            match = _COMPOUND_CODE_PATTERN.match(compound)
            if match:
                return (
                    match.group(1).strip(),
                    match.group(2).strip(),
                    match.group(3).strip(),
                )

        # Fallback: parse vicinity string
        vicinity = place.get("vicinity", "")
        if vicinity:
            parts = [p.strip() for p in vicinity.split(",")]
            city = parts[-1] if parts else ""
            return city, "", ""

        return "", "", ""

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cache(self) -> Any:
        """Retrieve the geo cache from ICacheManager, or None if unavailable."""
        cache_manager = ProtocolsRegistry.get_instance(ICacheManager)  # type: ignore[type-abstract]
        if cache_manager is None:
            return None
        return cache_manager.get_cache(self._CACHE_NAME)
