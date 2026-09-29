#!/usr/bin/env python
"""Convert downloaded SEVIR events (``download_sevir_subset.py`` .npz) into contract Zarr stores.

    python scripts/sevir_to_zarr.py --raw data/sevir_raw --out data/sevir_events --size 128
    python scripts/sevir_to_zarr.py --raw data/sevir_raw --check-lightning 30

``--check-lightning N`` compares flash-placement modes on N events and reports, for
each, the fraction of flashes that land on reflectivity >= 30 dBZ (within 1 pixel).
Lightning comes from convective cores, so the correct mode scores clearly highest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nowcast_ml.data.schema import validate_event
from nowcast_ml.data.sevir import LIGHTNING_MODE, read_catalog, sevir_event_from_arrays


def _entries(raw: Path):
    entries = {e.event_id: e for e in read_catalog(raw, "catalog.csv", require_lightning=True)}
    return [entries[p.stem] for p in sorted(raw.glob("*.npz")) if p.stem in entries]


def _convert(args):
    raw, out, entry, size, mode = args
    dst = Path(out) / f"sevir_{entry.event_id}.zarr"
    if dst.exists():
        return str(dst), []
    a = np.load(Path(raw) / f"{entry.event_id}.npz")
    ds = sevir_event_from_arrays(
        entry, a["vil"], a["ir107"], a["ir069"], a["lght"], size=size, lightning_mode=mode
    )
    tmp = dst.with_name(dst.name + ".tmp")
    ds.to_zarr(tmp, mode="w", consolidated=True)
    tmp.rename(dst)
    return str(dst), validate_event(dst, expected_spacing_km=ds.attrs["grid_spacing_km"])


def check_lightning(raw: Path, n: int, size: int) -> str:
    from scipy.ndimage import maximum_filter

    scores = {}
    for mode in ("xy", "xy_noflip", "latlon"):
        hits = total = 0
        for e in _entries(raw)[:n]:
            a = np.load(raw / f"{e.event_id}.npz")
            ds = sevir_event_from_arrays(
                e, a["vil"], a["ir107"], a["ir069"], a["lght"], size=size, lightning_mode=mode
            )
            z = np.nan_to_num(ds["x"].sel(channel="maxz").values)
            fd = np.nan_to_num(ds["x"].sel(channel="flash_density").values)
            zmax = maximum_filter(z, size=(1, 3, 3))
            hits += int(((fd > 0) & (zmax >= 30)).sum())
            total += int((fd > 0).sum())
        scores[mode] = hits / max(total, 1)
        print(f"{mode:10s} flash pixels on >=30 dBZ: {scores[mode]:.3f}  ({total} flash pixels)")
    best = max(scores, key=scores.get)
    print(f"best: {best}")
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw", default="data/sevir_raw")
    ap.add_argument("--out", default="data/sevir_events")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument(
        "--lightning-mode", default=LIGHTNING_MODE, choices=["xy", "xy_noflip", "latlon"]
    )
    ap.add_argument("--check-lightning", type=int, default=0, metavar="N")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    raw = Path(args.raw)
    if args.check_lightning:
        check_lightning(raw, args.check_lightning, args.size)
        return
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(str(raw), args.out, e, args.size, args.lightning_mode) for e in _entries(raw)]
    bad = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (path, problems) in enumerate(ex.map(_convert, jobs), 1):
            if problems:
                bad += 1
                print(f"INVALID {path}: {problems[:3]}")
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} converted", flush=True)
    print(f"done: {len(jobs) - bad} valid, {bad} invalid -> {args.out}")


if __name__ == "__main__":
    main()
