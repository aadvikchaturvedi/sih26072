"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nowcast_backend import __version__
from nowcast_backend.api.routes import domains, forecasts, system, warnings
from nowcast_backend.container import build_container
from nowcast_backend.domain.errors import BackendError
from nowcast_backend.ports import ForecastEngine
from nowcast_backend.settings import Settings

log = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


def _error(status: int, code: str, message: str, problems: list | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message, "problems": problems or []}},
        status_code=status,
    )


def create_app(settings: Settings | None = None, engine: ForecastEngine | None = None) -> FastAPI:
    """Build the app. The model is loaded at startup, so a bad artifact stops the
    process instead of failing on the first request. ``engine`` replaces the model in tests."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = build_container(settings, engine)
        app.state.container = container
        if not settings.api_key:
            log.warning("NOWCAST_API_KEY is not set: write endpoints are unauthenticated")
        if container.scheduler:
            container.scheduler.start()
        try:
            yield
        finally:
            if container.scheduler:
                container.scheduler.stop()

    app = FastAPI(
        title="Thunderstorm & lightning nowcasting API",
        version=__version__,
        lifespan=lifespan,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "Content-Type"],
        )

    @app.exception_handler(BackendError)
    async def backend_error(_: Request, e: BackendError):
        if e.status_code >= 500:
            log.error("%s: %s", e.code, e.message)
        return _error(e.status_code, e.code, e.message, e.problems)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, e: RequestValidationError):
        problems = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
        return _error(400, "invalid_request", "invalid request parameters", problems)

    for module in (system, domains, forecasts, warnings):
        app.include_router(module.router, prefix=API_PREFIX)
    return app
