"""Endpoints of the operator console (contract: ``web/src/data/DataSource.ts``)."""

from __future__ import annotations

import asyncio
import logging

import pandas as pd
from fastapi import APIRouter, Depends, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from nowcast_backend.api.console import mapping
from nowcast_backend.api.console import schemas as out
from nowcast_backend.api.deps import API_PREFIX, get_container
from nowcast_backend.container import Container
from nowcast_backend.domain.errors import BackendError, NotFound
from nowcast_backend.domain.timeutil import iso, parse_time, stamp
from nowcast_backend.services import cap, render

log = logging.getLogger(__name__)
router = APIRouter(prefix="/console", tags=["console"])

IMAGE_SCALE = 3  # upsampling of map overlays (nearest neighbour keeps pixel edges crisp)
STROKE_WINDOW_MIN = 30


class Console:
    """One request's view: the console domain and the services behind it."""

    def __init__(self, container: Container = Depends(get_container)):
        self.settings = container.settings
        self.service = container.service
        self.domain = container.settings.console_domain

    @property
    def replay(self) -> bool:
        return self.settings.source == "replay" or self.settings.clock == "data"

    def meta(self, t0: str | None):
        return self.service.forecast_meta(self.domain, parse_time(t0) if t0 else None, "model")

    def health(self, t0: str | None) -> out.HealthStatus:
        return mapping.health(self.meta(t0), self.service.model, self.replay)

    def cells(self, t0: str) -> list[out.Cell]:
        meta = self.meta(t0)
        cells = self.service.cells(self.domain, pd.Timestamp(meta.t0), "model")
        return [mapping.cell(c, meta, self.service.model.t_in) for c in cells]

    def warnings(self, t0: str | None) -> list[out.Warning]:
        """Newest first, as the console lists them."""
        found = self.service.district_warnings(self.domain, parse_time(t0) if t0 else None)
        return [mapping.warning(w) for w in reversed(found)]


@router.get("/timeline", response_model=out.Timeline)
def timeline(c: Console = Depends()):
    status = c.service.domain(c.domain)
    b = status.bounds
    # Pixel centres sit half a pixel inside the image edges.
    dy = (b.north - b.south) / max(1, status.shape[0] - 1) / 2
    dx = (b.east - b.west) / max(1, status.shape[1] - 1) / 2
    model = c.service.model
    return out.Timeline(
        domain=c.domain,
        times=[iso(t) for t in c.service.forecast_time_list(c.domain)],
        stepMin=model.step_minutes,
        leads=model.lead_minutes,
        bbox={
            "west": b.west - dx,
            "east": b.east + dx,
            "south": b.south - dy,
            "north": b.north + dy,
        },
    )


@router.get("/health", response_model=out.HealthStatus)
def health_latest(c: Console = Depends()):
    return c.health(None)


@router.get("/health/{t0}", response_model=out.HealthStatus)
def health_at(t0: str, c: Console = Depends()):
    return c.health(t0)


@router.get("/forecast/{t0}", response_model=out.ForecastResponse, response_model_exclude_none=True)
def forecast(t0: str, request: Request, c: Console = Depends()):
    meta = c.meta(t0)
    attrs = c.service.forecast(c.domain, pd.Timestamp(meta.t0), "model").attrs
    base = f"{str(request.base_url).rstrip('/')}{API_PREFIX}"
    layers = f"{base}/domains/{c.domain}/forecasts/{stamp(meta.t0)}/layers"
    return mapping.forecast(meta, attrs, layers, IMAGE_SCALE)


@router.get("/radar/{time}")
def observed_radar(time: str, c: Console = Depends()) -> Response:
    """Observed MAX-Z as a PNG. Times without radar give a transparent image, so the
    map simply shows nothing there."""
    try:
        field = c.service.observation(c.domain, parse_time(time), "maxz")
        body = render.render_png(field, render.REFLECTIVITY, IMAGE_SCALE)
    except NotFound:
        body = render.TRANSPARENT_PIXEL
    return Response(body, media_type="image/png", headers={"Cache-Control": "max-age=60"})


@router.get("/cells/{t0}", response_model=list[out.Cell])
def cells(t0: str, c: Console = Depends()):
    return c.cells(t0)


@router.get("/lightning/{t0}", response_model=list[out.LightningStroke])
def lightning(t0: str, c: Console = Depends()):
    meta = c.meta(t0)
    t = pd.Timestamp(meta.t0)
    return mapping.strokes(
        *c.service.flashes(c.domain, t, STROKE_WINDOW_MIN), c.service.cells(c.domain, t, "model")
    )


@router.get("/districts/{t0}", response_model=list[out.DistrictForecast])
def districts(t0: str, c: Console = Depends()):
    return [mapping.district(d) for d in c.service.districts(c.domain, parse_time(t0))]


@router.get("/warnings/{t0}", response_model=list[out.Warning])
def warnings(t0: str, c: Console = Depends()):
    return c.warnings(t0)


@router.get("/cap/{warning_id}")
def export_cap(warning_id: str, c: Console = Depends()) -> dict:
    """The warning as CAP 1.2 XML plus the same content as JSON: ``{"xml", "json"}``."""
    found = {w.id: w for w in c.service.district_warnings(c.domain, None)}
    w = found.get(warning_id)
    if w is None:
        raise NotFound(f"no warning {warning_id!r}")
    region = next((r for r in c.service.regions() if r.id == w.district_id), None)
    xml, doc = cap.district_alert(w, region, found.get(w.supersedes or ""), c.settings.cap)
    return {"xml": xml, "json": doc}


@router.get("/skill", response_model=out.SkillReport)
def skill(c: Console = Depends()):
    return mapping.skill(c.service.evaluation())


@router.websocket("/live")
async def live(websocket: WebSocket):
    """Pushes a tick whenever a new model forecast for the console domain is ready."""
    container: Container = websocket.app.state.container
    c = Console(container)
    await websocket.accept()
    queue = container.bus.subscribe()

    def tick(t0: str) -> out.LiveTick:
        return out.LiveTick(
            t0=t0,
            newWarnings=[w for w in c.warnings(t0) if w.issuedAt == t0],
            updatedCells=c.cells(t0),
            health=c.health(t0),
        )

    async def push():
        while True:
            event = await queue.get()
            ready = event.type == "forecast.ready" and event.data.get("source") == "model"
            if not ready or event.domain != c.domain:
                continue
            try:
                message = await run_in_threadpool(tick, iso(event.time))
            except BackendError as e:  # e.g. pruned in the meantime
                log.warning("live tick skipped: %s", e.message)
                continue
            await websocket.send_text(message.model_dump_json())

    pusher = asyncio.create_task(push())
    try:
        while True:  # the client never sends; this just notices when it goes away
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        pusher.cancel()
        container.bus.unsubscribe(queue)
