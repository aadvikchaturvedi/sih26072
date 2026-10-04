"""Forecasts and the products derived from them."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel, Field

from nowcast_backend.api.deps import SourceQuery, get_service, require_api_key, time_or_latest
from nowcast_backend.domain.models import ForecastMeta, ForecastSource, StormCell
from nowcast_backend.domain.timeutil import parse_time
from nowcast_backend.services import render
from nowcast_backend.services.geo import feature_collection
from nowcast_backend.services.nowcast import NowcastService

router = APIRouter(prefix="/domains/{domain}/forecasts", tags=["forecasts"])


class RunRequest(BaseModel):
    t0: str | None = Field(None, description="analysis time (UTC); default: latest observation")
    source: ForecastSource = "model"
    n_members: int = Field(
        0, ge=0, le=50, description="ensemble members (model with refiner, or steps)"
    )
    force: bool = Field(False, description="recompute even if this forecast is stored")


def _style(layer: str) -> render.Style:
    if layer == "reflectivity":
        return render.REFLECTIVITY
    return render.FLAG if layer == "first_flash" else render.PROBABILITY


@router.get("")
def list_forecasts(
    domain: str,
    source: ForecastSource = SourceQuery,
    limit: int = Query(24, ge=1, le=500),
    service: NowcastService = Depends(get_service),
) -> list[ForecastMeta]:
    """Stored forecasts, newest first."""
    return service.forecast_times(domain, source, limit)


@router.post("", dependencies=[Depends(require_api_key)])
def run_forecast(
    domain: str, body: RunRequest | None = None, service: NowcastService = Depends(get_service)
) -> ForecastMeta:
    """Run (or return the stored) forecast for an analysis time."""
    body = body or RunRequest()
    return service.run(
        domain,
        parse_time(body.t0) if body.t0 else None,
        source=body.source,
        n_members=body.n_members,
        force=body.force,
    )


@router.get("/{t0}")
def get_forecast(
    domain: str,
    t0: str,
    source: ForecastSource = SourceQuery,
    service: NowcastService = Depends(get_service),
) -> dict:
    """Summary of one forecast (``t0`` may be ``latest``) and the layers it offers."""
    meta = service.forecast_meta(domain, time_or_latest(t0), source)
    layers = service.layer_names(service.forecast(domain, time_or_latest(t0), source))
    return {**meta.model_dump(mode="json"), "layers": layers}


@router.get("/{t0}/layers/{layer}.png")
def forecast_layer_png(
    domain: str,
    t0: str,
    layer: str,
    lead: int | None = Query(None, description="minutes; needed for reflectivity / exceedance"),
    threshold: float = Query(35.0, description="dBZ, for the exceedance layer"),
    scale: int = Query(1, ge=1, le=8),
    source: ForecastSource = SourceQuery,
    service: NowcastService = Depends(get_service),
) -> Response:
    """A forecast field as a transparent PNG to stretch over the domain ``bounds``."""
    field = service.layer(domain, time_or_latest(t0), source, layer, lead, threshold)
    # A forecast for a fixed t0 never changes; ``latest`` moves with every run.
    cache = "no-cache" if t0 == "latest" else "public, max-age=3600"
    return Response(
        render.render_png(field, _style(layer), scale),
        media_type="image/png",
        headers={"Cache-Control": cache},
    )


@router.get("/{t0}/layers/{layer}")
def forecast_layer_grid(
    domain: str,
    t0: str,
    layer: str,
    lead: int | None = None,
    threshold: float = 35.0,
    stride: int = Query(1, ge=1, le=16, description="keep every n-th pixel"),
    source: ForecastSource = SourceQuery,
    service: NowcastService = Depends(get_service),
) -> dict:
    """The same field as numbers (row 0 = north); ``null`` where there is no forecast."""
    field = service.layer(domain, time_or_latest(t0), source, layer, lead, threshold)[
        ::stride, ::stride
    ]
    values = np.round(field.astype(np.float64), 3).astype(object)
    values[~np.isfinite(field)] = None
    return {
        "layer": layer,
        "lead": lead,
        "stride": stride,
        "shape": field.shape,
        "values": values.tolist(),
    }


@router.get("/{t0}/point")
def forecast_point(
    domain: str,
    t0: str,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    source: ForecastSource = SourceQuery,
    service: NowcastService = Depends(get_service),
) -> dict:
    """Forecast time series at a location (the pixel containing it)."""
    return service.point(domain, time_or_latest(t0), source, lat, lon)


@router.get("/{t0}/cells")
def forecast_cells(
    domain: str,
    t0: str,
    format: str = Query("geojson", pattern="^(geojson|json)$"),
    source: ForecastSource = SourceQuery,
    service: NowcastService = Depends(get_service),
) -> dict | list[StormCell]:
    """Storm cells at the analysis time with motion, trend and extrapolated track."""
    cells = service.cells(domain, time_or_latest(t0), source)
    return feature_collection(cells) if format == "geojson" else cells
