"""Typed errors. The API layer maps ``status_code`` / ``code`` onto HTTP responses."""

from __future__ import annotations


class BackendError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, problems: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.problems = list(problems or [])


class InvalidRequest(BackendError):
    status_code = 400
    code = "invalid_request"


class Unauthorized(BackendError):
    status_code = 401
    code = "unauthorized"


class NotFound(BackendError):
    status_code = 404
    code = "not_found"


class Conflict(BackendError):
    """The request is valid but does not fit the current state (e.g. a different grid)."""

    status_code = 409
    code = "conflict"


class PayloadTooLarge(BackendError):
    status_code = 413
    code = "payload_too_large"


class InvalidObservations(BackendError):
    """Observations violate the input contract; ``problems`` lists every violation."""

    status_code = 422
    code = "invalid_observations"


class EngineError(BackendError):
    """The forecast engine failed on valid inputs."""

    status_code = 500
    code = "engine_error"
