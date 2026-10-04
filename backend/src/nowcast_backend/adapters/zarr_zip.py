"""Wire format for pushing observation frames over HTTP: a zipped Zarr store holding
an input-contract Dataset."""

from __future__ import annotations

import tempfile
import warnings
import zipfile
from pathlib import Path

import xarray as xr
import zarr

from nowcast_backend.domain.errors import InvalidObservations


def encode(ds: xr.Dataset) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "frames.zip"
        store = zarr.ZipStore(str(path), mode="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # zip "duplicate name" on attr rewrites
            ds.to_zarr(store, mode="w", consolidated=False)
        store.close()
        return path.read_bytes()


def decode(payload: bytes, max_grid_px: int, max_frames: int) -> xr.Dataset:
    """Nothing is extracted to disk by name, and sizes are checked before data is read."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "frames.zip"
        path.write_bytes(payload)
        try:
            store = zarr.ZipStore(str(path), mode="r")
        except (zipfile.BadZipFile, OSError) as e:
            raise InvalidObservations(f"body is not a zip archive: {e}") from e
        try:
            ds = xr.open_zarr(store, consolidated=False)
            sizes = dict(ds.sizes)
            if max(sizes.get("y", 0), sizes.get("x", 0)) > max_grid_px:
                raise InvalidObservations(f"grid larger than {max_grid_px} x {max_grid_px} pixels")
            if sizes.get("time", 0) > max_frames:
                raise InvalidObservations(
                    f"{sizes['time']} frames in one upload; send at most {max_frames}"
                )
            return ds.load()
        except InvalidObservations:
            raise
        except Exception as e:  # noqa: BLE001 - any unreadable store is a client error
            raise InvalidObservations(
                f"body is not a readable Zarr store: {type(e).__name__}: {e}"
            ) from e
        finally:
            store.close()
