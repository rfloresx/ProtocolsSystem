"""Health check protocol — connectivity verification for service providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IHealthCheck(Protocol):
    """Health Check:Verifies backing service connectivity status.

    Clients implementing this protocol expose a lightweight health check
    that verifies the backing service is reachable without performing any
    heavy computation. Used by the GUI to display dynamic service status.
    """

    async def health_check(self) -> bool:
        """Return True if the backing service is reachable and healthy.

        Should complete quickly (timeout ~5-10s). Must not raise — returns
        False on any failure.
        """
        ...

    @property
    def health_check_url(self) -> str:
        """The base URL being checked, for display purposes."""
        ...
