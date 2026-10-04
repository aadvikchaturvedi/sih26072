"""Health, model description, legends and the live event stream."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from nowcast_backend import __version__
from nowcast_backend.api.deps import get_container, get_service
from nowcast_backend.container import Container
from nowcast_backend.domain.models import ModelDescription
from nowcast_backend.services.nowcast import NowcastService
from nowcast_backend.services.render import STYLES

router = APIRouter(tags=["system"])

HEARTBEAT_SECONDS = 15.0


@router.get("/health")
def health(service: NowcastService = Depends(get_service)) -> dict:
    """The process is up and the model is loaded (the app does not start otherwise)."""
    return {
        "status": "ok",
        "version": __version__,
        "model_version": service.model.version,
        "domains": [{"domain": d.domain, "latest_time": d.latest_time} for d in service.domains()],
    }


@router.get("/model")
def model(service: NowcastService = Depends(get_service)) -> ModelDescription:
    return service.model


@router.get("/styles")
def styles() -> dict:
    """Colour scales used by the PNG layers, for drawing legends."""
    return {name: style.legend() for name, style in STYLES.items()}


@router.get("/events")
async def events(request: Request, container: Container = Depends(get_container)):
    """Server-sent events: ``observation.ingested``, ``forecast.ready``, ``warnings.updated``."""
    queue = container.bus.subscribe()

    async def stream():
        try:
            yield ": connected\n\n"
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"
        finally:
            container.bus.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
