"""Response models matching ``web/src/data/types.ts`` field for field (camelCase)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

Level = Literal["green", "yellow", "orange", "red"]
Mode = Literal["full", "satellite_only"]
Risk = Literal["high", "moderate", "low"]


class Timeline(BaseModel):
    """What the console needs to lay out its time controls and map."""

    domain: str
    times: list[str]  # analysis times with a forecast, oldest first (ISO-8601 UTC)
    stepMin: int
    leads: list[int]
    bbox: dict[str, float]  # west / east / south / north


class InputHealth(BaseModel):
    name: Literal["radar", "satellite", "lightning", "nwp"]
    status: Literal["live", "stale", "missing"]
    lastReceived: str | None
    staleAgeSeconds: int | None


class HealthStatus(BaseModel):
    lastRunAt: str
    nextRunAt: str
    mode: Literal["live", "replay"]
    modelMode: Mode
    modelName: str
    modelVersion: str
    missingChannels: list[str]
    inputs: list[InputHealth]
    hasEnsemble: bool
    inferenceMsLast: float | None


class ForecastAttrs(BaseModel):
    model_name: str
    model_version: str
    t0: str
    mode: Mode
    missing_channels: list[str]
    inference_ms: float
    output_contract_version: str
    first_flash_threshold: float


class ForecastResponse(BaseModel):
    reflectivityUrlByLead: dict[str, str]
    lightningProb30Url: str
    lightningProb60Url: str
    firstFlashUrl: str
    ensembleSpreadUrl: str | None = None
    attrs: ForecastAttrs


class CellTrackPoint(BaseModel):
    time: str
    lat: float
    lon: float
    maxDbz: float
    flashRate: float


class ForecastPathPoint(BaseModel):
    leadMin: int
    lat: float
    lon: float
    expectedDbz: float
    risk: Risk
    riskBasis: Literal["lightning_prob", "reflectivity_based"]


class Cell(BaseModel):
    id: str
    isNewCell: bool
    status: Literal["initiating", "growing", "mature", "decaying"]
    lat: float
    lon: float
    maxDbz: float
    growthDbzPer10Min: float
    lightningTrend: Literal["rapidly_increasing", "increasing", "steady", "decreasing"]
    headingDeg: float
    speedKmh: float
    lightningJumpFlag: bool
    lightningJumpAt: str | None
    track: list[CellTrackPoint]
    forecastPath: list[ForecastPathPoint]
    uncertaintyCone: list[dict[str, float]]
    affectedDistricts: list[dict[str, Any]]
    # ``confidence`` is optional in the console's contract and not supplied: there is no
    # verified definition for it yet, and the console hides it when absent.
    modelMode: Mode
    dataQuality: float
    flashRateSeries: list[dict[str, Any]]
    dbzSeries: list[dict[str, Any]]


class LightningStroke(BaseModel):
    id: str
    time: str
    lat: float
    lon: float
    cellId: str | None
    # polarity and peak current are not part of the input contract's flash table


class DistrictForecast(BaseModel):
    districtId: str
    name: str
    lightningProb30: float
    lightningProb60: float
    maxDbz: float
    areaFractionAboveThreshold: float
    warningLevel: Level


class Warning(BaseModel):
    id: str
    level: Level
    districtId: str
    districtName: str
    causeText: str
    causeCell: str | None
    ruleTriggered: str
    triggerValue: float
    triggerThreshold: float
    issuedAt: str
    validUntil: str
    issuedBy: Literal["system", "forecaster"] = "system"
    previousLevel: Level | None
    isLevelChange: bool = True


class ModelSkill(BaseModel):
    modelId: str
    label: str
    color: str
    csiByThreshold: dict[str, list[dict[str, float]]]
    reliabilityDiagram: list[dict[str, float]]
    brierSkillScore: float | None
    rocAuc: float | None
    pod: float | None
    far: float | None
    firstFlashHitRate: float | None
    firstFlashFAR: float | None
    medianFirstFlashLeadMin: float | None


class SkillReport(BaseModel):
    reportAt: str
    medianFirstFlashLeadMin: float | None
    models: list[ModelSkill]
    eventLabel: str
    eventStartAt: str | None
    eventEndAt: str | None
    synthetic: bool


class LiveTick(BaseModel):
    t0: str
    newWarnings: list[Warning]
    updatedCells: list[Cell]
    health: HealthStatus
