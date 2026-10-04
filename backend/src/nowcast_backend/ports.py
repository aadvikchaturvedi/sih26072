"""Interfaces between the services and the outside world.

Services depend only on these protocols; ``adapters`` implements them and
``container`` chooses the implementations.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import pandas as pd
import xarray as xr

from nowcast_backend.domain.grid import Grid
from nowcast_backend.domain.models import (
    DomainStatus,
    Event,
    ForecastMeta,
    ModelDescription,
    Region,
    Warning,
)


class ForecastEngine(Protocol):
    """Turns input-contract observations into an output-contract forecast."""

    def describe(self) -> ModelDescription: ...

    def validate(self, inputs: xr.Dataset) -> list[str]:
        """Input-contract violations for this model (empty = valid)."""

    def forecast(
        self, inputs: xr.Dataset, t0: pd.Timestamp, *, source: str = "model", n_members: int = 0
    ) -> xr.Dataset: ...


class ObservationStore(Protocol):
    """Rolling buffer of the latest observation frames per domain."""

    def ingest(self, domain: str, ds: xr.Dataset) -> list[pd.Timestamp]:
        """Add (or replace) the frames of a contract Dataset; returns their times."""

    def domains(self) -> list[str]: ...

    def status(self, domain: str) -> DomainStatus: ...

    def grid(self, domain: str) -> Grid: ...

    def window(self, domain: str, t0: pd.Timestamp | None, n_frames: int) -> tuple[xr.Dataset, int]:
        """Contract Dataset of ``n_frames`` regular frames ending at ``t0`` (default: latest).

        Gaps are filled with fully-missing frames. Also returns the number of real frames.
        """

    def frame(self, domain: str, time: pd.Timestamp, channel: str): ...

    def flashes(self, domain: str, start: pd.Timestamp, stop: pd.Timestamp):
        """Flash points in ``(start, stop]`` as (time, lat, lon) arrays."""


class ForecastStore(Protocol):
    def save(self, meta: ForecastMeta, forecast: xr.Dataset, products: dict[str, list]) -> None:
        """``products`` are named JSON documents derived from the forecast (cells, districts)."""

    def exists(self, domain: str, t0: pd.Timestamp, source: str) -> bool: ...

    def meta(self, domain: str, t0: pd.Timestamp, source: str) -> ForecastMeta: ...

    def dataset(self, domain: str, t0: pd.Timestamp, source: str) -> xr.Dataset: ...

    def product(self, domain: str, t0: pd.Timestamp, source: str, name: str) -> list:
        """A stored product as plain JSON data (empty if the forecast has none by that name)."""

    def times(self, domain: str, source: str) -> list[pd.Timestamp]:
        """Stored forecast times, oldest first."""

    def prune(self, domain: str, keep: int) -> None: ...


class WarningRepository(Protocol):
    def add(self, warnings: Iterable[Warning]) -> None: ...

    def set_status(self, warning_id: str, status: str) -> None: ...

    def get(self, warning_id: str) -> Warning | None: ...

    def find(
        self, domain: str | None = None, status: str | None = None, limit: int = 200
    ) -> list[Warning]:
        """Newest first."""


class RegionProvider(Protocol):
    def regions(self) -> list[Region]: ...


class EvaluationSource(Protocol):
    def metrics(self) -> dict | None:
        """The latest ``nowcast-eval`` metrics, or None if the model was never evaluated."""


class Notifier(Protocol):
    def publish(self, event: Event) -> None:
        """Must not raise and must not block for long."""


class FrameSource(Protocol):
    """Where observation frames come from when the backend pulls them itself."""

    def poll(self) -> Iterable[tuple[str, xr.Dataset]]:
        """New ``(domain, contract Dataset)`` pairs since the last call."""
