"""Small client for producers that push observation frames to the backend."""

from __future__ import annotations

import httpx
import xarray as xr

from nowcast_backend.adapters.zarr_zip import encode


def push_frames(
    base_url: str,
    domain: str,
    frames: xr.Dataset,
    api_key: str | None = None,
    run: bool | None = None,
    timeout: float = 120.0,
) -> dict:
    """POST input-contract frames; returns the ingest result (with the forecast summary
    if one was run). Raises ``httpx.HTTPStatusError`` with the server's problem list."""
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/v1/domains/{domain}/observations",
        content=encode(frames),
        params={} if run is None else {"run": str(run).lower()},
        headers={
            "Content-Type": "application/zip",
            **({"X-API-Key": api_key} if api_key else {}),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()
