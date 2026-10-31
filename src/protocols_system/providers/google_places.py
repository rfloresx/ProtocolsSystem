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


# Plus-code prefix at the start of a compound_code, e.g. "849VCWC8+R9".
_PLUS_CODE_PREFIX = re.compile(r"^[23456789CFGHJMPQRVWX]+\+[23456789CFGHJMPQRVWX]+\s+")


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
        timeout: float = 10.0,
    ) -> None:
        _check_dependencies()
        import googlemaps

        # Without a timeout the googlemaps client could hang indefinitely
        # while occupying a slot in the shared asyncio.to_thread pool.
        self._timeout = float(timeout)
        self._client = googlemaps.Client(key=api_key, timeout=self._timeout)
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
        """Check connectivity/credentials without a billable Places call.

        A Nearby Search is a *billable* Places API request charged on every
        health poll. Instead, issue a Geocoding request for a fixed well-known
        place — a different, far cheaper product that still exercises auth and
        connectivity. If the SDK/endpoint isn't available, fall back to simply
        confirming the client is configured rather than spending money to
        answer "are we up?".
        """
        try:
            geocode = getattr(self._client, "geocode", None)
            if geocode is None:
                return self._client is not None
            result = await asyncio.to_thread(geocode, "Googleplex, Mountain View, CA")
            return bool(result)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # IGeoClient
    # ------------------------------------------------------------------

    async def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int | None = None,
        max_results: int | None = None,
        place_type: str | None = None,
    ) -> list[PlaceCandidate]:
        """Resolve GPS coordinates into ranked nearby place candidates.

        Results are cached via ICacheManager (if available) so repeated queries
        for the same location skip the API call.

        Args:
            latitude: Latitude in degrees.
            longitude: Longitude in degrees.
            radius_meters: Search radius in meters (default from constructor).
            max_results: Maximum candidates to return (default from constructor).
            place_type: Optional Google Places type filter (e.g.
                "point_of_interest"), passed through to the Nearby Search API's
                ``type`` parameter. This is an extension specific to this
                client, not part of the base ``IGeoClient`` signature.

        Returns:
            List of PlaceCandidate sorted by distance from the query point.
        """
        radius = radius_meters if radius_meters is not None else self._default_radius
        limit = max_results if max_results is not None else self._max_results

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
            self._fetch_nearby, latitude, longitude, radius, place_type
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

    # Hard cap on pages to follow so a pathological query can't paginate
    # forever (the Places API returns up to 20 results/page, 3 pages max).
    _MAX_PAGES = 3

    def _fetch_nearby(
        self, lat: float, lng: float, radius: int, place_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Synchronous Places API call with pagination.

        Returns *all* fetched results (up to the page cap) without truncating
        — the caller (``reverse_geocode``) sorts by distance and then truncates
        to ``max_results``. Truncating here, before that sort, would keep the N
        most prominent places Google returned rather than the N nearest.
        """
        kwargs: dict[str, Any] = {
            "location": (lat, lng),
            "radius": radius,
            "open_now": False,
        }
        if place_type:
            kwargs["type"] = place_type

        results: list[dict[str, Any]] = []
        response: dict[str, Any] = self._client.places_nearby(**kwargs)
        results.extend(response.get("results", []))

        # Follow next_page_token pagination. A freshly-issued page token is not
        # valid immediately; the API returns INVALID_REQUEST until it activates
        # (usually a second or two). Retry the token a few times with backoff
        # and, if it never activates, return what we already have instead of
        # throwing the first page away.
        pages = 1
        while "next_page_token" in response and pages < self._MAX_PAGES:
            token = response["next_page_token"]
            next_response = self._fetch_page_with_token(token)
            if next_response is None:
                break
            response = next_response
            results.extend(response.get("results", []))
            pages += 1

        return results

    def _fetch_page_with_token(self, token: str) -> dict[str, Any] | None:
        """Fetch one more page by token, tolerating token-not-yet-valid.

        Returns the response dict, or ``None`` if the token never became valid
        (so the caller keeps the pages fetched so far rather than failing the
        whole request).
        """
        import googlemaps

        delay = 2.0
        for _attempt in range(3):
            time.sleep(delay)
            try:
                return dict(self._client.places_nearby(page_token=token))
            except googlemaps.exceptions.ApiError as exc:
                # INVALID_REQUEST while the token is still warming up — back
                # off and retry. Any other API error is real; stop paging.
                if getattr(exc, "status", None) == "INVALID_REQUEST":
                    delay *= 1.5
                    continue
                logger.warning("Places pagination failed: %s", exc)
                return None
        logger.warning(
            "Places next_page_token did not activate; returning partial results"
        )
        return None

    def _parse_location(self, place: dict[str, Any]) -> tuple[str, str, str]:
        """Extract city/state/country from plus_code compound_code or vicinity.

        Returns:
            Tuple of (city, state, country). Empty strings for unavailable fields.
        """
        plus_code = place.get("plus_code", {})
        compound = plus_code.get("compound_code", "")

        if compound:
            # Strip the leading plus-code token, then split the remaining
            # "City, Region, Country"-ish string on commas. Splitting
            # positionally handles 2+ parts regardless of casing/length, so
            # localities that don't follow the US "City, ST, US" convention
            # (most of the world) are parsed too.
            remainder = _PLUS_CODE_PREFIX.sub("", compound).strip()
            parts = [p.strip() for p in remainder.split(",") if p.strip()]
            if len(parts) >= 3:
                return parts[0], ", ".join(parts[1:-1]), parts[-1]
            if len(parts) == 2:
                return parts[0], "", parts[1]
            if len(parts) == 1:
                return parts[0], "", ""

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
