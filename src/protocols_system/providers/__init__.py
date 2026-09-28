"""Common provider implementations for the Protocols System.

Providers registered here are available to any project that depends on
protocols_system. Each provider module self-registers via the
@ProtocolsRegistry.register decorator at import time.

Call register_builtin_providers() once at app startup to trigger registration.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path

logger = logging.getLogger(__name__)

_registered = False


def register_builtin_providers() -> None:
    """Import all provider modules to trigger decorator-based registration.

    Safe to call multiple times — only runs once.
    Providers with missing optional dependencies are silently skipped.
    """
    global _registered
    if _registered:
        return

    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{module_info.name}")
            logger.debug("Registered provider module: %s", module_info.name)
        except ImportError as exc:
            # Missing optional dependency — skip silently
            logger.debug(
                "Skipped provider '%s' (missing dependency): %s",
                module_info.name,
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Failed to register provider '%s': %s",
                module_info.name,
                exc,
            )

    _registered = True
