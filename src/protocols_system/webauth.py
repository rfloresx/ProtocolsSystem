"""Shared web-auth primitives: bcrypt hashing, timing-safe verification.

Both consuming web apps (PhotoStudio and Smart-Albums) need the exact same
security-sensitive password handling:

* bcrypt hash/verify with the CPU-bound work dispatched off the event loop so
  it never blocks a single-threaded async UI;
* a fixed dummy-hash comparison so an unknown username costs the same bcrypt
  round as a known username with a wrong password (no username enumeration by
  timing);
* the bcrypt 72-byte input limit guarded explicitly (bcrypt>=5 raises beyond
  it, and only the first 72 bytes are ever hashed).

This module deliberately owns ONLY these primitives. Each app keeps its own
user store (sync sqlite3 vs async aiosqlite), its own login rate-limiting
policy, and its own session/page layer — those are genuinely different and
should not be force-shared.

Requires: bcrypt
"""

from __future__ import annotations

import asyncio

import bcrypt

# bcrypt only hashes the first 72 bytes of input; bcrypt>=5 raises ValueError
# for longer input rather than silently truncating.
MAX_PASSWORD_BYTES = 72

# A bcrypt hash of a random, never-used password (cost 12, matching
# ``bcrypt.gensalt()``), computed once at import. verify_password_async()
# checks an unknown username against this so an unknown-username response
# burns roughly the same CPU as a known-username/wrong-password response —
# otherwise an attacker could enumerate valid usernames by timing alone.
# Comparing a real password against this hash always fails.
_DUMMY_HASH = bcrypt.hashpw(bcrypt.gensalt(), bcrypt.gensalt()).decode("utf-8")


def password_too_long(password: str) -> bool:
    """Return True if ``password`` exceeds bcrypt's 72-byte input limit (UTF-8)."""
    return len(password.encode("utf-8")) > MAX_PASSWORD_BYTES


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    Raises:
        ValueError: If the password exceeds :data:`MAX_PASSWORD_BYTES`.
    """
    if password_too_long(password):
        raise ValueError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes (UTF-8)."
        )
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash. Never raises.

    An over-long password or a malformed stored hash returns False rather than
    raising, so callers can treat it as an ordinary auth failure.
    """
    if password_too_long(password):
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        # Malformed stored hash.
        return False


async def hash_password_async(password: str) -> str:
    """Hash a password off the event loop.

    bcrypt's hashpw is a synchronous CPU-bound call (~100-300ms). Running it
    directly in an async handler blocks the whole event loop for that long,
    freezing every other connected client and the background scheduler.

    Raises:
        ValueError: If the password exceeds :data:`MAX_PASSWORD_BYTES`.
    """
    return await asyncio.to_thread(hash_password, password)


async def verify_password_async(password: str, hashed: str | None) -> bool:
    """Verify a password off the event loop, without leaking via timing.

    When ``hashed`` is ``None`` (unknown username), a bcrypt comparison is
    still run against a fixed dummy hash rather than returning immediately, so
    an unknown-username response isn't measurably faster than a
    known-username/wrong-password one. The result for an unknown username is
    always False.
    """
    target = hashed if hashed is not None else _DUMMY_HASH
    result = await asyncio.to_thread(verify_password, password, target)
    return result if hashed is not None else False
