"""Geolocation protocol — reverse geocoding from GPS coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaceCandidate:
    """A single reverse-geocoded place result."""

    name: str
    city: str
    state: str
    country: str
    latitude: float
    longitude: float
    distance_meters: float
    place_type: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IGeoClient(Protocol):
    """Geolocation:Reverse geocoding provider.

    Resolves GPS coordinates into place name candidates.
    """

    async def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int = 1000,
        max_results: int = 10,
    ) -> list[PlaceCandidate]: ...
