"""Time handling. Internally times are tz-naive UTC ``pd.Timestamp`` (as in the data
contract); API models carry tz-aware UTC ``datetime``."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from nowcast_backend.domain.errors import InvalidRequest


def naive_utc(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def aware_utc(t) -> datetime:
    return naive_utc(t).to_pydatetime().replace(tzinfo=UTC)


def now_utc() -> pd.Timestamp:
    return naive_utc(datetime.now(UTC))


def stamp(t) -> str:
    """Compact, path-safe form: ``20260512T1050Z``."""
    return naive_utc(t).strftime("%Y%m%dT%H%MZ")


def iso(t) -> str:
    return naive_utc(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: str) -> pd.Timestamp:
    """Parse an ISO-8601 time or the compact ``stamp`` form (UTC unless an offset is given)."""
    try:
        ts = naive_utc(value)
    except (ValueError, TypeError) as e:
        raise InvalidRequest(f"cannot parse time {value!r}; use e.g. 2026-05-12T10:50Z") from e
    if pd.isnull(ts):
        raise InvalidRequest(f"cannot parse time {value!r}; use e.g. 2026-05-12T10:50Z")
    return ts
