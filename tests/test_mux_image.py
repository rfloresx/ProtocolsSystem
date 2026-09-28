"""Tests for MuxImageClient health checks over its configured sub-providers."""

from __future__ import annotations

import pytest

from protocols_system.protocols.health import IHealthCheck
from protocols_system.providers.mux_image import MuxImageClient, SubProviderHealth


class _GoodClient:
    async def health_check(self) -> bool:
        return True

    @property
    def health_check_url(self) -> str:
        return "good"


class _BadClient:
    async def health_check(self) -> bool:
        return False

    @property
    def health_check_url(self) -> str:
        return "bad"


class _RaisingClient:
    async def health_check(self) -> bool:
        raise RuntimeError("boom")

    @property
    def health_check_url(self) -> str:
        return "raise"


class _NoHealthClient:
    """An IImageClient-shaped provider without IHealthCheck support."""


class _EnterableClient:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> _EnterableClient:
        self.entered = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exited = True

    async def health_check(self) -> bool:
        return self.entered

    @property
    def health_check_url(self) -> str:
        return "enterable"


def test_mux_is_health_checkable() -> None:
    mux = MuxImageClient(routes={"immich": _GoodClient()})
    assert isinstance(mux, IHealthCheck)


@pytest.mark.asyncio
async def test_check_providers_reports_per_scheme() -> None:
    mux = MuxImageClient(routes={"immich": _GoodClient(), "asset": _NoHealthClient()})
    results = await mux.check_providers()
    by_scheme = {r.scheme: r for r in results}
    assert by_scheme["immich"] == SubProviderHealth("immich", True, "Reachable")
    assert by_scheme["asset"].ok is None


@pytest.mark.asyncio
async def test_health_check_passes_when_checkable_are_reachable() -> None:
    mux = MuxImageClient(routes={"immich": _GoodClient(), "asset": _NoHealthClient()})
    assert await mux.health_check() is True


@pytest.mark.asyncio
async def test_health_check_fails_when_any_sub_provider_down() -> None:
    mux = MuxImageClient(routes={"immich": _BadClient(), "asset": _GoodClient()})
    assert await mux.health_check() is False


@pytest.mark.asyncio
async def test_health_check_treats_raising_provider_as_down() -> None:
    mux = MuxImageClient(routes={"immich": _RaisingClient()})
    results = await mux.check_providers()
    assert results[0].ok is False
    assert await mux.health_check() is False


@pytest.mark.asyncio
async def test_health_check_false_when_no_providers() -> None:
    mux = MuxImageClient(routes={})
    assert await mux.health_check() is False


@pytest.mark.asyncio
async def test_health_check_true_when_only_uncheckable() -> None:
    mux = MuxImageClient(routes={"asset": _NoHealthClient()})
    assert await mux.health_check() is True


@pytest.mark.asyncio
async def test_health_check_enters_async_context() -> None:
    client = _EnterableClient()
    mux = MuxImageClient(routes={"immich": client})
    assert await mux.health_check() is True
    assert client.entered and client.exited
