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
#: District colour code; ``green`` = no warning.
DistrictLevel = Literal["green", "yellow", "orange", "red"]
InputGroup = Literal["radar", "satellite", "lightning", "nwp"]


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


class InputHealth(BaseModel):
    """State of one input group at a forecast's analysis time."""

    name: InputGroup
    status: Literal["live", "stale", "missing"]
    last_received: datetime | None  # newest input frame carrying this group
    stale_age_seconds: int | None  # None when live


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
    inputs: list[InputHealth] = Field(default_factory=list)


class TrackPoint(BaseModel):
    time: datetime
    lat: float
    lon: float
    max_dbz: float | None = None
    flash_rate: float | None = None  # flashes per minute near the cell (observed points)
    lightning_prob: float | None = None  # forecast points within the lightning leads


class AffectedRegion(BaseModel):
    region_id: str
    name: str
    eta_min: int  # 0 = the cell is over it now


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
    is_new: bool = False  # first seen in this run
    growth_dbz_per_10min: float = 0.0
    flash_rate: float = 0.0  # flashes per minute in the last step
    #: Flash rate rose by more than 2 sigma of its recent changes (kept for 30 min).
    lightning_jump: bool = False
    lightning_jump_at: datetime | None = None
    affected: list[AffectedRegion] = Field(default_factory=list)


class Region(BaseModel):
    """An administrative area (district) that forecasts are summarised over."""

    id: str
    name: str
    geometry: dict[str, Any]  # GeoJSON Polygon / MultiPolygon


class DistrictForecast(BaseModel):
    district_id: str
    name: str
    lightning_prob_30: float
    lightning_prob_60: float
    max_dbz: float  # forecast MAX-Z within the next hour
    area_fraction_above_threshold: float
    first_flash: bool
    level: DistrictLevel
    rule: str  # the condition that set the level
    trigger_value: float
    trigger_threshold: float
    cause_cell: str | None = None


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


class DistrictWarning(BaseModel):
    """A change of a district's level between two consecutive forecasts."""

    id: str
    level: DistrictLevel  # ``green`` = the earlier warning is lifted
    previous_level: DistrictLevel | None
    supersedes: str | None = None  # the district's previous warning
    district_id: str
    district_name: str
    cause_text: str
    cause_cell: str | None
    rule: str
    trigger_value: float
    trigger_threshold: float
    issued_at: datetime
    valid_until: datetime


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
