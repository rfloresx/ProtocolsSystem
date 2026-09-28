"""Progress reporting protocol — lifecycle hooks for stage-based progress."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IProgressReporter(Protocol):
    """Progress:Lifecycle hooks for stage progress reporting."""

    def start_stage(self, description: str, total: int | None = None) -> None: ...

    def advance(self, n: int = 1) -> None: ...

    def finish_stage(self) -> None: ...

    def log(self, message: str) -> None: ...

    def warn(self, message: str) -> None: ...
