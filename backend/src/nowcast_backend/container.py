"""Composition root: the one place that picks concrete adapters for the ports."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from nowcast_backend.adapters.evaluation import JsonEvaluation
from nowcast_backend.adapters.forecast_store import ZarrForecastStore
from nowcast_backend.adapters.notifiers import (
    CompositeNotifier,
    EventBus,
    LogNotifier,
    WebhookNotifier,
)
from nowcast_backend.adapters.observations import RollingObservationStore
from nowcast_backend.adapters.regions import GeoJsonRegions, NoRegions
from nowcast_backend.adapters.sources import ReplaySource, ZarrDirectorySource
from nowcast_backend.adapters.warning_repo import SqliteWarningRepository
from nowcast_backend.ports import ForecastEngine, FrameSource
from nowcast_backend.scheduler import Scheduler
from nowcast_backend.services.nowcast import NowcastService
from nowcast_backend.services.warnings import WarningService
from nowcast_backend.settings import Settings

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    service: NowcastService
    bus: EventBus
    scheduler: Scheduler | None


def _source(settings: Settings, t_in: int) -> FrameSource | None:
    if settings.source == "none":
        return None
    if settings.source_path is None:
        raise ValueError(f"NOWCAST_SOURCE={settings.source} needs NOWCAST_SOURCE_PATH")
    if settings.source == "watch":
        return ZarrDirectorySource(settings.source_path, initial_frames=settings.buffer_frames)
    return ReplaySource(
        settings.source_path,
        settings.replay_domain,
        initial_frames=settings.replay_initial_frames or t_in,
    )


def build_container(settings: Settings, engine: ForecastEngine | None = None) -> Container:
    """Wire the application. Pass ``engine`` to substitute the model (tests)."""
    if engine is None:
        from nowcast_backend.adapters.engine_ml import PredictorEngine  # imports torch

        engine = PredictorEngine.load(
            settings.model_path, device=settings.device, backend=settings.engine_backend
        )
    model = engine.describe()
    log.info("model %s %s on %s", model.name, model.version, model.device)

    bus = EventBus()
    notifiers = [LogNotifier(), bus]
    if settings.webhook_url:
        notifiers.append(WebhookNotifier(settings.webhook_url))
    service = NowcastService(
        settings,
        engine,
        RollingObservationStore(
            settings.data_dir / "observations", settings.buffer_frames, model.step_minutes
        ),
        ZarrForecastStore(settings.data_dir / "forecasts"),
        WarningService(SqliteWarningRepository(settings.data_dir / "warnings.sqlite3")),
        CompositeNotifier(notifiers),
        GeoJsonRegions(settings.regions_path) if settings.regions_path else NoRegions(),
        JsonEvaluation(settings.skill_report_path),
    )
    source = _source(settings, model.t_in)
    scheduler = Scheduler(service, source, settings.poll_seconds) if source else None
    return Container(settings, service, bus, scheduler)
