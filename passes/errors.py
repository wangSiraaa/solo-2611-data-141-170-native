"""Input and propagation exceptions."""

from __future__ import annotations

from typing import Any, Iterable


class PassesError(Exception):
    """Base class for expected solver errors."""


class InputError(PassesError):
    """Malformed or inconsistent input."""


class PropagationError(PassesError):
    """SGP4 returned an error code or a non-finite position.

    The failure is never converted into "invisible".  ``failures`` contains at
    least one located UTC timestamp; the CLI writes this as JSON and exits
    non-zero.
    """

    def __init__(self, message: str, failures: Iterable[dict[str, Any]]):
        self.failures = list(failures)
        super().__init__(message)

    def summary(self) -> str:
        if not self.failures:
            return str(self)
        first = self.failures[0]
        return (
            f"{self} (first failure at {first.get('utc')} / "
            f"t+{first.get('seconds_from_start_s')} s; "
            f"{len(self.failures)} located failure(s))"
        )
