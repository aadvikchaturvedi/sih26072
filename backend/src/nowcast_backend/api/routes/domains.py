"""Domains and their observations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from nowcast_ml.data import channels as ch

from nowcast_backend.adapters import zarr_zip
from nowcast_backend.api.deps import get_container, get_service, require_api_key, time_or_latest
from nowcast_backend.container import Container
from nowcast_backend.domain.errors import NotFound, PayloadTooLarge
from nowcast_backend.domain.models import DomainStatus, IngestResult
from nowcast_backend.services import render
from nowcast_backend.services.nowcast import NowcastService, check_domain_id

router = APIRouter(prefix="/domains", tags=["domains"])


@router.get("")
def list_domains(service: NowcastService = Depends(get_service)) -> list[DomainStatus]:
    return service.domains()


@router.get("/{domain}")
def get_domain(domain: str, service: NowcastService = Depends(get_service)) -> DomainStatus:
    return service.domain(domain)


@router.post("/{domain}/observations", dependencies=[Depends(require_api_key)])
async def ingest_observations(
    domain: str,
    request: Request,
    run: bool | None = Query(None, description="forecast after ingest (default: server setting)"),
    container: Container = Depends(get_container),
) -> IngestResult:
    """Push observation frames: the body is a zipped Zarr store holding an input-contract
    Dataset (see ``nowcast_backend.client.push_frames``). Re-sending a time replaces it."""
    check_domain_id(domain)
    settings = container.settings
    limit = settings.max_upload_mb * 1024 * 1024
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise PayloadTooLarge(f"upload larger than {settings.max_upload_mb} MB")

    def work() -> IngestResult:
        ds = zarr_zip.decode(bytes(body), settings.max_grid_px, settings.buffer_frames)
        return container.service.ingest(domain, ds, run=run)

    return await run_in_threadpool(work)


@router.get("/{domain}/observations/{time}/{channel}.png")
def observation_layer(
    domain: str,
    time: str,
    channel: str,
    scale: int = Query(1, ge=1, le=8),
    service: NowcastService = Depends(get_service),
) -> Response:
    """One observed channel as a PNG overlay (``time`` may be ``latest``)."""
    if channel not in ch.CHANNELS:
        raise NotFound(f"unknown channel {channel!r}")
    field = service.observation(domain, time_or_latest(time), channel)
    unit = ch.get(channel).unit
    style = render.style_for_unit(unit) or render.greyscale_style(field, unit)
    return Response(render.render_png(field, style, scale), media_type="image/png")
