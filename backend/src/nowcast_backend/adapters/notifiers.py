"""Notifier implementations: in-process event bus (SSE), webhook, and a fan-out."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx

from nowcast_backend.domain.models import Event
from nowcast_backend.ports import Notifier

log = logging.getLogger(__name__)


class EventBus:
    """Hands events from worker threads to asyncio subscribers (the SSE endpoint)."""

    def __init__(self, queue_size: int = 100):
        self._subscribers: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
        self._queue_size = queue_size
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(self._queue_size)
        with self._lock:
            self._subscribers.add((asyncio.get_running_loop(), queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = {s for s in self._subscribers if s[1] is not queue}

    @staticmethod
    def _offer(queue: asyncio.Queue, event: Event) -> None:
        if queue.full():  # slow consumer: drop its oldest event rather than block producers
            queue.get_nowait()
        queue.put_nowait(event)

    def publish(self, event: Event) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for loop, queue in subscribers:
            try:
                loop.call_soon_threadsafe(self._offer, queue, event)
            except RuntimeError:  # loop already closed
                self.unsubscribe(queue)


class WebhookNotifier:
    """POSTs warning changes as JSON. Delivery is best effort and never blocks a forecast."""

    def __init__(self, url: str, timeout: float = 5.0, types=("warnings.updated",)):
        self._url = url
        self._timeout = timeout
        self._types = set(types)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="webhook")

    def _send(self, event: Event) -> None:
        try:
            httpx.post(
                self._url, json=event.model_dump(mode="json"), timeout=self._timeout
            ).raise_for_status()
        except httpx.HTTPError as e:
            log.warning("webhook delivery to %s failed: %s", self._url, e)

    def publish(self, event: Event) -> None:
        if event.type in self._types:
            self._pool.submit(self._send, event)


class LogNotifier:
    def publish(self, event: Event) -> None:
        log.info("%s domain=%s time=%s", event.type, event.domain, event.time.isoformat())


class CompositeNotifier:
    def __init__(self, notifiers: list[Notifier]):
        self._notifiers = notifiers

    def publish(self, event: Event) -> None:
        for n in self._notifiers:
            try:
                n.publish(event)
            except Exception:  # noqa: BLE001 - a broken notifier must not break forecasting
                log.exception("notifier %s failed", type(n).__name__)
