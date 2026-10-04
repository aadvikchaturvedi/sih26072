"""Background polling of a :class:`~nowcast_backend.ports.FrameSource`."""

from __future__ import annotations

import logging
import threading

from nowcast_backend.domain.errors import BackendError
from nowcast_backend.ports import FrameSource
from nowcast_backend.services.nowcast import NowcastService

log = logging.getLogger(__name__)


class Scheduler:
    """Polls the source on a fixed interval and ingests whatever it yields (which, in
    turn, runs the forecast). One failing frame never stops the loop."""

    def __init__(self, service: NowcastService, source: FrameSource, interval_seconds: float):
        self._service = service
        self._source = source
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick(self) -> int:
        """One poll; returns the number of datasets ingested."""
        n = 0
        for domain, frames in self._source.poll():
            try:
                self._service.ingest(domain, frames)
                n += 1
            except BackendError as e:
                log.error("domain %s: rejected frames: %s %s", domain, e.message, e.problems)
            except Exception:  # noqa: BLE001 - keep polling whatever happens
                log.exception("domain %s: ingest failed", domain)
        return n

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.exception("frame source poll failed")
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="frame-source", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5)
