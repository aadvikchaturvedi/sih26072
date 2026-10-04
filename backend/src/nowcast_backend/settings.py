"""Configuration, read from ``NOWCAST_*`` environment variables (and an optional ``.env``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CellSettings(BaseModel):
    threshold_dbz: float = 35.0
    min_area_km2: float = 20.0
    max_speed_kmh: float = 100.0
    track_minutes: int = 60  # length of the extrapolated track
    trend_dbz: float = 3.0  # change at +30 min that counts as growing / decaying


class WarningSettings(BaseModel):
    #: Lightning probability (next 30 min) for yellow / orange / red.
    lightning_prob: tuple[float, float, float] = (0.3, 0.5, 0.7)
    #: Forecast MAX-Z (any lead up to ``thunderstorm_horizon_min``) for yellow / orange / red.
    thunderstorm_dbz: tuple[float, float, float] = (40.0, 50.0, 60.0)
    thunderstorm_horizon_min: int = 60
    buffer_km: float = 4.0  # warning polygons are grown by this margin
    min_area_km2: float = 40.0


class CapSettings(BaseModel):
    sender: str = "nowcast@example.invalid"
    sender_name: str = "Thunderstorm Nowcasting System"
    #: CAP ``status``. Keep ``Test`` until the model has been verified on real data.
    status: Literal["Actual", "Exercise", "Test"] = "Test"
    language: str = "en-IN"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NOWCAST_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- model
    model_path: Path = Path("../ml/artifacts/models/nowcast/latest")
    device: str = "auto"
    engine_backend: Literal["eager", "torchscript"] = "eager"

    # --- storage
    data_dir: Path = Path("var")
    buffer_frames: int = Field(36, ge=7)  # rolling observation buffer per domain (6 h)
    forecast_retention: int = Field(144, ge=1)  # forecast times kept per domain (24 h)
    #: Real (not gap-filled) frames required among the model's input frames.
    min_observed_frames: int = Field(4, ge=1)

    # --- HTTP
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None  # required on write endpoints when set
    cors_origins: list[str] = Field(default_factory=list)
    max_upload_mb: int = 256
    max_grid_px: int = 2048

    # --- automatic operation
    auto_forecast: bool = True  # run the model whenever a newer frame is ingested
    source: Literal["none", "watch", "replay"] = "none"
    source_path: Path | None = None  # watch: dir of <domain>.zarr; replay: one event store
    replay_domain: str = "demo"
    poll_seconds: float = Field(30.0, gt=0)
    #: ``wall`` = warnings expire by the system clock; ``data`` = by the latest observation
    #: time of the domain (for replaying archived events).
    clock: Literal["wall", "data"] = "wall"

    # --- products
    cells: CellSettings = Field(default_factory=CellSettings)
    warnings: WarningSettings = Field(default_factory=WarningSettings)
    cap: CapSettings = Field(default_factory=CapSettings)
    webhook_url: str | None = None  # POSTed a JSON event when warnings change
    public_url: str = "http://127.0.0.1:8000"  # used for links in the CAP feed
