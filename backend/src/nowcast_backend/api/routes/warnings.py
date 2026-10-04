"""Warnings, CAP documents and the CAP feed."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response

from nowcast_backend.api.deps import get_container, get_service
from nowcast_backend.container import Container
from nowcast_backend.domain.models import Warning, WarningStatus
from nowcast_backend.domain.timeutil import aware_utc
from nowcast_backend.services import cap
from nowcast_backend.services.geo import feature_collection
from nowcast_backend.services.nowcast import NowcastService

router = APIRouter(tags=["warnings"])


@router.get("/warnings")
def list_warnings(
    domain: str | None = None,
    status: WarningStatus = "active",
    limit: int = Query(200, ge=1, le=1000),
    format: str = Query("geojson", pattern="^(geojson|json)$"),
    service: NowcastService = Depends(get_service),
) -> dict | list[Warning]:
    """Warnings, newest first. ``active`` ones are those still in force."""
    if status == "active":
        found = service.active_warnings(domain)[:limit]
    else:
        found = service.warning_history(domain, status, limit)
    return feature_collection(found) if format == "geojson" else found


@router.get("/warnings/{warning_id}")
def get_warning(warning_id: str, service: NowcastService = Depends(get_service)) -> Warning:
    return service.warning(warning_id)[0]


@router.get("/warnings/{warning_id}/cap")
def get_warning_cap(warning_id: str, container: Container = Depends(get_container)) -> Response:
    """The warning as a CAP 1.2 alert."""
    warning, previous = container.service.warning(warning_id)
    return Response(
        cap.cap_alert(warning, container.settings.cap, previous), media_type="application/cap+xml"
    )


@router.get("/cap/feed.atom")
def cap_feed(container: Container = Depends(get_container)) -> Response:
    """Atom feed of the active warnings, for CAP aggregators."""
    service, settings = container.service, container.settings
    return Response(
        cap.atom_feed(
            service.active_warnings(), settings.cap, settings.public_url, aware_utc(service.now())
        ),
        media_type="application/atom+xml",
    )
