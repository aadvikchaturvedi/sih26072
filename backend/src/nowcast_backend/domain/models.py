"""Data models shared by services, adapters and the API (they are the JSON schema too)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

#: ``model`` is the trained network; the others are the reference methods of nowcast_ml.
ForecastSource = Literal["model", "persistence", "extrapolation", "steps"]
SOURCES: tuple[str, ...] = ("model", "persistence", "extrapolation", "steps")

Hazard = Literal["lightning", "thunderstorm"]
#: IMD colour codes: yellow = be aware, orange = be prepared, red = take action.
Level = Literal["yellow", "orange", "red"]
LEVELS: tuple[str, ...] = ("yellow", "orange", "red")
WarningStatus = Literal["active", "superseded", "cancelled", "expired"]


class Bounds(BaseModel):
    south: float
    west: float
    north: float
    east: float


class ModelDescription(BaseModel):
    name: str
    version: str
    channels: list[str]
    grid_spacing_km: float
    t_in: int
    step_minutes: int
    lead_minutes: list[int]
    lightning_leads_min: list[int]
    calibrated: bool
    has_ensemble: bool
    backend: str
    device: str
    sources: list[str] = Field(default_factory=lambda: list(SOURCES))


class DomainStatus(BaseModel):
    domain: str
    shape: tuple[int, int]
    grid_spacing_km: float
    bounds: Bounds
    channels: list[str]
    times: list[datetime]  # observed frames in the rolling buffer, oldest first
    latest_time: datetime | None


class ForecastMeta(BaseModel):
    domain: str
    t0: datetime
    source: ForecastSource
    model_name: str
    model_version: str
    mode: Literal["full", "satellite_only"]
    missing_channels: list[str]
    lead_minutes: list[int]
    lightning_leads_min: list[int]
    n_members: int = 0
    inference_ms: float
    created_at: datetime
    shape: tuple[int, int]
    grid_spacing_km: float
    bounds: Bounds
    observed_frames: int  # real (not gap-filled) input frames
    max_reflectivity_dbz: float
    max_lightning_prob: float
    n_cells: int = 0


class TrackPoint(BaseModel):
    time: datetime
    lat: float
    lon: float
    max_dbz: float | None = None


class StormCell(BaseModel):
    id: str
    time: datetime
    #: ``observed`` = found in radar at t0; ``forecast`` = found in the +10 min forecast
    #: because radar was unavailable (satellite-only mode).
    basis: Literal["observed", "forecast"]
    lat: float
    lon: float
    area_km2: float
    max_dbz: float
    speed_kmh: float | None = None
    direction_deg: float | None = None  # bearing the cell moves towards, clockwise from north
    trend: Literal["growing", "steady", "decaying"] = "steady"
    lightning_prob: dict[str, float] = Field(default_factory=dict)  # lead minutes -> max prob
    polygon: dict[str, Any]  # GeoJSON geometry
    history: list[TrackPoint] = Field(default_factory=list)
    forecast_track: list[TrackPoint] = Field(default_factory=list)


class Warning(BaseModel):
    id: str
    domain: str
    hazard: Hazard
    level: Level
    status: WarningStatus = "active"
    msg_type: Literal["Alert", "Update", "Cancel"] = "Alert"
    supersedes: str | None = None
    t0: datetime
    issued_at: datetime
    onset: datetime
    expires: datetime
    headline: str
    description: str
    instruction: str
    peak_value: float
    peak_unit: str
    first_flash: bool = False  # lightning expected where none occurred in the past 30 min
    area_km2: float
    lat: float  # centroid
    lon: float
    polygon: dict[str, Any]  # GeoJSON geometry
    mode: str
    model_version: str


class IngestResult(BaseModel):
    domain: str
    accepted_times: list[datetime]
    latest_time: datetime
    n_frames: int
    forecast: ForecastMeta | None = None


class Event(BaseModel):
    """Something subscribers may care about (pushed over SSE and to webhooks)."""

    type: Literal["observation.ingested", "forecast.ready", "warnings.updated"]
    domain: str
    time: datetime
    data: dict[str, Any] = Field(default_factory=dict)
