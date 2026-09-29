"""Typed exceptions the backend can catch without importing torch-heavy modules."""


class NowcastError(Exception):
    """Base class for all nowcast_ml errors raised through the public API."""


class InputContractError(NowcastError, ValueError):
    """Inputs do not satisfy the input contract (data/schema.py).

    ``problems`` lists every human-readable violation found.
    """

    def __init__(self, message: str, problems: list[str] | None = None):
        self.problems = list(problems or [])
        detail = "".join(f"\n  - {p}" for p in self.problems)
        super().__init__(message + detail)


class ModelLoadError(NowcastError, RuntimeError):
    """A model artifact is missing, corrupt, or inconsistent (channels, normalization, versions)."""


class OutputContractError(NowcastError, RuntimeError):
    """A forecast failed output-schema validation (a bug, not a user error)."""
