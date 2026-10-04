"""Request-scoped helpers shared by the routes."""

from __future__ import annotations

import hmac

import pandas as pd
from fastapi import Header, Query, Request

from nowcast_backend.container import Container
from nowcast_backend.domain.errors import Unauthorized
from nowcast_backend.domain.timeutil import parse_time
from nowcast_backend.services.nowcast import NowcastService

API_PREFIX = "/api/v1"


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_service(request: Request) -> NowcastService:
    return request.app.state.container.service


def require_api_key(request: Request, x_api_key: str | None = Header(None)) -> None:
    """Guards write endpoints. With no key configured they are open (local development)."""
    expected = request.app.state.container.settings.api_key
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
        raise Unauthorized("missing or wrong X-API-Key header")


def time_or_latest(value: str) -> pd.Timestamp | None:
    """Path segment that is either ``latest`` or a UTC time."""
    return None if value == "latest" else parse_time(value)


SourceQuery = Query("model", description="the trained model, or a reference method")
