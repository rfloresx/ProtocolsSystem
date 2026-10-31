"""Tests for the shared web-auth primitives."""

from __future__ import annotations

import pytest

pytest.importorskip("bcrypt")

from protocols_system.webauth import (  # noqa: E402
    MAX_PASSWORD_BYTES,
    hash_password,
    hash_password_async,
    password_too_long,
    verify_password,
    verify_password_async,
)


class TestHashAndVerify:
    def test_hash_then_verify_roundtrip(self):
        hashed = hash_password("correct horse battery staple")
        assert verify_password("correct horse battery staple", hashed) is True

    def test_wrong_password_fails(self):
        hashed = hash_password("right")
        assert verify_password("wrong", hashed) is False

    def test_hashes_are_salted(self):
        assert hash_password("same") != hash_password("same")

    def test_malformed_hash_returns_false(self):
        assert verify_password("anything", "not-a-bcrypt-hash") is False


class TestPasswordLength:
    def test_password_too_long_boundary(self):
        assert password_too_long("a" * MAX_PASSWORD_BYTES) is False
        assert password_too_long("a" * (MAX_PASSWORD_BYTES + 1)) is True

    def test_hash_rejects_overlong_password(self):
        with pytest.raises(ValueError):
            hash_password("a" * (MAX_PASSWORD_BYTES + 1))

    def test_verify_rejects_overlong_password(self):
        hashed = hash_password("short")
        assert verify_password("a" * (MAX_PASSWORD_BYTES + 1), hashed) is False


class TestAsync:
    async def test_hash_and_verify_async(self):
        hashed = await hash_password_async("secret")
        assert await verify_password_async("secret", hashed) is True
        assert await verify_password_async("nope", hashed) is False

    async def test_unknown_user_hash_none_is_false(self):
        # None hash (unknown username) still runs a comparison and returns False.
        assert await verify_password_async("whatever", None) is False
