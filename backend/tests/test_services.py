"""Unit tests for the pieces with real logic: cells, geometry, rendering, warnings,
the rolling buffer, sources and the event bus."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from nowcast_backend.adapters.notifiers import EventBus
from nowcast_backend.adapters.observations import RollingObservationStore
from nowcast_backend.adapters.sources import ReplaySource, ZarrDirectorySource
from nowcast_backend.domain.errors import InvalidRequest, NotFound
from nowcast_backend.domain.grid import Grid
from nowcast_backend.domain.models import Event
from nowcast_backend.scheduler import Scheduler
from nowcast_backend.services import cells, render
from nowcast_backend.services.geo import bbox, mask_polygon
from nowcast_backend.settings import CellSettings
from tests.conftest import FakeEngine, frames, t


@pytest.fixture()
def grid() -> Grid:
    lat = np.linspace(23.0, 22.0, 51)[:, None].repeat(51, axis=1)
    lon = np.linspace(88.0, 89.0, 51)[None, :].repeat(51, axis=0)
    return Grid(lat, lon, 2.0)


def blob_field(cy: float, cx: float, peak: float = 50.0, size: int = 51) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return (peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 18.0)).astype(np.float32)


def test_grid_lookup(grid):
    assert grid.nearest_index(22.5, 88.5) == (25, 25)
    with pytest.raises(InvalidRequest):
        grid.nearest_index(10.0, 88.5)
    b = grid.bounds
    assert (b.south, b.north, b.west, b.east) == (22.0, 23.0, 88.0, 89.0)


def test_mask_polygon_follows_pixel_edges(grid):
    mask = np.zeros(grid.shape, bool)
    mask[10:20, 30:40] = True
    west, south, east, north = bbox(mask_polygon(mask, grid))
    # pixel edges are half a pixel (0.01 deg) outside the outermost pixel centres
    assert west == pytest.approx(grid.lon[0, 30] - 0.01, abs=2e-3)
    assert east == pytest.approx(grid.lon[0, 39] + 0.01, abs=2e-3)
    assert north == pytest.approx(grid.lat[10, 0] + 0.01, abs=2e-3)
    assert south == pytest.approx(grid.lat[19, 0] - 0.01, abs=2e-3)
    single = np.zeros(grid.shape, bool)
    single[5, 5] = True
    assert mask_polygon(single, grid)["type"] == "Polygon"
    assert mask_polygon(np.zeros(grid.shape, bool), grid) is None


def test_colorize():
    field = np.array([[np.nan, 0.0, 22.0, 52.0]])
    rgba = render.colorize(field, render.REFLECTIVITY)
    assert rgba[0, 0, 3] == 0 and rgba[0, 1, 3] == 0  # NaN and "no echo" are transparent
    assert tuple(rgba[0, 2]) == (100, 200, 255, 200) and tuple(rgba[0, 3]) == (200, 0, 200, 250)
    prob = render.colorize(np.array([[0.0, 0.3, 1.0]]), render.PROBABILITY)
    assert prob[0, 0, 3] == 0 and tuple(prob[0, 1]) == (254, 204, 92, 190)


def test_cell_tracking_speed_and_direction(grid, event):
    from nowcast_ml.inference.schema import ForecastArrays, build_forecast_dataset

    from tests.conftest import LEADS

    # one cell moving 3 px (6 km) east per 10-min frame = 36 km/h towards 90 degrees
    times = pd.date_range("2026-05-12T10:00", periods=4, freq="10min")
    observed = [(ts, blob_field(25, 10 + 3 * i)) for i, ts in enumerate(times)]
    refl = np.stack([blob_field(25, 19 + 3 * (k + 1), peak=50 + 2 * (k + 1)) for k in range(12)])
    forecast = build_forecast_dataset(
        ForecastArrays(refl, np.full((2, 51, 51), 0.4, np.float32)),
        lat=grid.lat, lon=grid.lon, t0=times[-1], lead_minutes=LEADS, ltg_leads_min=[30, 60],
        past_ltg=np.zeros((51, 51), bool), model_name="m", model_version="v", mode="full",
        missing_channels=[], inference_ms=1.0,
    )  # fmt: skip
    cfg = CellSettings(min_area_km2=8.0, track_minutes=60)
    found = cells.analyse(forecast, observed, grid, cfg, step_minutes=10)
    assert len(found) == 1
    cell = found[0]
    assert cell.speed_kmh == pytest.approx(36.0, abs=1.0)
    assert cell.direction_deg == pytest.approx(90.0, abs=2.0)
    assert len(cell.history) == 3 and len(cell.forecast_track) == 6
    assert cell.forecast_track[-1].lon > cell.lon and cell.trend == "growing"
    assert cell.lightning_prob == {"30": 0.4, "60": 0.4}
    assert not cell.is_new and cell.growth_dbz_per_10min == pytest.approx(0.0, abs=0.5)
    assert all(p.lightning_prob == pytest.approx(0.4) for p in cell.forecast_track)

    # the next run recognises the same storm and keeps its id; a far-away one gets a new id
    later = [
        (ts + pd.Timedelta(minutes=10), blob_field(25, 13 + 3 * i)) for i, ts in enumerate(times)
    ]
    forecast.attrs["t0"] = "2026-05-12T10:40:00Z"
    again = cells.analyse(forecast, later, grid, cfg, 10, previous=found)
    assert again[0].id == cell.id and not again[0].is_new
    elsewhere = [(ts, blob_field(45, 45)) for ts, _ in later]
    assert cells.analyse(forecast, elsewhere, grid, cfg, 10, previous=found)[0].id != cell.id

    # no radar at t0: cells come from the forecast itself
    blind = cells.analyse(forecast, [(times[-1], np.full((51, 51), np.nan))], grid, cfg, 10)
    assert blind[0].basis == "forecast" and blind[0].speed_kmh == pytest.approx(36.0, abs=1.0)


def test_rolling_store_trims_and_restores(tmp_path, event):
    store = RollingObservationStore(tmp_path, max_frames=8)
    store.ingest("d", frames(event, 0, 12))
    status = store.status("d")
    assert len(status.times) == 8 and pd.Timestamp(status.times[0]).tz_localize(None) == t(event, 4)
    window, observed = store.window("d", None, 7)
    assert observed == 7 and window.sizes["time"] == 7
    assert FakeEngine().validate(window) == []
    with pytest.raises(NotFound):
        store.window("d", t(event, 0), 7)  # trimmed away
    # flashes older than the buffer are dropped with their frames
    assert window["flash_time"].values.min() > np.datetime64(t(event, 3))

    restored = RollingObservationStore(tmp_path, max_frames=8)
    assert restored.status("d") == status
    np.testing.assert_array_equal(
        restored.frame("d", t(event, 11), "maxz"), store.frame("d", t(event, 11), "maxz")
    )


def test_sources_and_scheduler(tmp_path, event, service):
    store = tmp_path / "live" / "kolkata.zarr"
    frames(event, 0, 10).to_zarr(str(store), mode="w", consolidated=True)

    watch = ZarrDirectorySource(store.parent, initial_frames=36)
    scheduler = Scheduler(service, watch, interval_seconds=60)
    assert scheduler.tick() == 1 and scheduler.tick() == 0
    assert len(service.domain("kolkata").times) == 10
    frames(event, 0, 12).to_zarr(str(store), mode="w", consolidated=True)  # producer adds 2 frames
    assert scheduler.tick() == 1
    assert len(service.domain("kolkata").times) == 12
    assert (
        service.forecast_meta("kolkata", None, "model").t0 == service.domain("kolkata").latest_time
    )

    replay = ReplaySource(store, "demo", initial_frames=7)
    sizes = [ds.sizes["time"] for _ in range(7) for _, ds in replay.poll()]
    assert sizes == [7, 1, 1, 1, 1, 1]  # then exhausted


def test_event_bus_delivers_across_threads():
    async def scenario():
        bus = EventBus(queue_size=2)
        queue = bus.subscribe()
        events = [
            Event(type="forecast.ready", domain=f"d{i}", time=pd.Timestamp("2026-05-12", tz="UTC"))
            for i in range(3)
        ]
        await asyncio.gather(*(asyncio.to_thread(bus.publish, e) for e in events))
        await asyncio.sleep(0.05)
        assert queue.qsize() == 2  # a full queue drops the oldest event, not the newest
        bus.unsubscribe(queue)
        bus.publish(events[0])
        await asyncio.sleep(0.01)
        assert queue.qsize() == 2

    asyncio.run(scenario())
