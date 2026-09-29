#!/usr/bin/env python
"""Download a SEVIR storm-event subset from the public AWS bucket (anonymous).

SEVIR files are multi-GB and store each image type as one contiguous,
uncompressed HDF5 dataset (N, H, W, 49). Instead of downloading whole files,
this script reads each file's HDF5 metadata once (dataset byte offsets), then
fetches only the selected events with parallel HTTP range requests and saves
one compressed ``<id>.npz`` per event (raw SEVIR units), plus ``catalog.csv``.

    pip install -e ".[sevir]"
    python scripts/download_sevir_subset.py --out data/sevir_raw --n-events 600 --workers 8

Resumable: events already on disk are skipped. Convert to contract Zarr stores
with ``scripts/sevir_to_zarr.py``. Needs network access; tests never call it.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET_URL = "https://sevir.s3.us-west-2.amazonaws.com"
TYPES = ("vil", "ir107", "ir069", "lght")
DEFAULT_EVENT_TYPES = [
    "Thunderstorm Wind",
    "Hail",
    "Heavy Rain",
    "Tornado",
    "Lightning",
    "Funnel Cloud",
]


def _range(url: str, start: int, nbytes: int, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{start + nbytes - 1}"}
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) != nbytes:
                raise OSError(f"short read {len(data)} != {nbytes}")
            return data
        except Exception:  # noqa: BLE001 - retry any network error
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def _file_layout(fs, file_name: str, img_type: str, ids: list[str] | None = None) -> dict:
    """Byte offset/shape/dtype of the image dataset, or of each lightning dataset in ``ids``."""
    import h5py

    with fs.open(f"sevir/data/{file_name}", "rb", block_size=2**16) as fo, h5py.File(fo, "r") as h:
        if img_type != "lght":
            d = h[img_type]
            if d.chunks is not None or d.compression is not None:
                raise RuntimeError(
                    f"{file_name}: dataset is chunked/compressed; range reads unsupported"
                )
            return {"offset": d.id.get_offset(), "shape": d.shape, "dtype": d.dtype.str}
        out = {}
        for i in ids or []:
            if i in h:
                d = h[i]
                out[i] = {"offset": d.id.get_offset(), "shape": d.shape, "dtype": d.dtype.str}
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="data/sevir_raw")
    ap.add_argument("--n-events", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--min-flashes", type=int, default=1, help="require at least this many GLM flashes"
    )
    ap.add_argument("--event-types", nargs="*", default=DEFAULT_EVENT_TYPES)
    args = ap.parse_args(argv)

    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("reading catalog ...", flush=True)
    with fs.open("sevir/CATALOG.csv") as f:
        cat = pd.read_csv(f, low_memory=False)
    cat = cat[cat["event_id"].notna() & cat["img_type"].isin(TYPES)]
    if args.event_types:
        cat = cat[cat["event_type"].isin(args.event_types)]
    have = cat.groupby("id")["img_type"].apply(set)
    ids = sorted(i for i, s in have.items() if set(TYPES) <= s)
    lmax = cat[cat.img_type == "lght"].set_index("id")["data_max"]
    ids = [i for i in ids if lmax.get(i, 0) >= args.min_flashes]
    rng = np.random.default_rng(args.seed)
    ids = sorted(rng.choice(ids, size=min(args.n_events, len(ids)), replace=False).tolist())
    sub = cat[cat["id"].isin(ids)].copy()
    sub.to_csv(out / "catalog.csv", index=False)
    todo = [i for i in ids if not (out / f"{i}.npz").exists()]
    print(f"selected {len(ids)} events, {len(todo)} to download", flush=True)
    if not todo:
        return

    # One metadata read per source file.
    layouts: dict[tuple[str, str], dict] = {}
    for (t, fn), g in sub[sub["id"].isin(todo)].groupby(["img_type", "file_name"]):
        t0 = time.time()
        layouts[(t, fn)] = _file_layout(fs, fn, t, sorted(g["id"]) if t == "lght" else None)
        print(f"  layout {fn} ({time.time() - t0:.0f}s)", flush=True)

    def fetch(eid: str) -> tuple[str, int]:
        rows = sub[sub["id"] == eid].set_index("img_type")
        arrays, nbytes = {}, 0
        for t in TYPES:
            r = rows.loc[t]
            lay = layouts[(t, r.file_name)]
            url = f"{BUCKET_URL}/data/{r.file_name}"
            if t == "lght":
                if eid not in lay:
                    arrays[t] = np.zeros((0, 5), np.float32)
                    continue
                lay = lay[eid]
                dt = np.dtype(lay["dtype"])
                n = int(np.prod(lay["shape"])) * dt.itemsize
                arrays[t] = np.frombuffer(_range(url, lay["offset"], n), dt).reshape(lay["shape"])
            else:
                dt = np.dtype(lay["dtype"])
                per = int(np.prod(lay["shape"][1:])) * dt.itemsize
                buf = _range(url, lay["offset"] + int(r.file_index) * per, per)
                arrays[t] = np.frombuffer(buf, dt).reshape(lay["shape"][1:])
            nbytes += arrays[t].nbytes
        tmp = out / f".{eid}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.rename(out / f"{eid}.npz")
        return eid, nbytes

    t0, done, total = time.time(), 0, 0
    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(fetch, e) for e in todo]):
            try:
                _, nb = fut.result()
            except Exception as e:  # noqa: BLE001 - keep going; rerun resumes
                print(f"  FAILED: {e}", flush=True)
                continue
            done += 1
            total += nb
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t0
                print(
                    f"  {done}/{len(todo)} events, {total / 1e9:.2f} GB, {total / el / 1e6:.2f} MB/s, "
                    f"eta {(len(todo) - done) * el / done / 60:.0f} min",
                    flush=True,
                )
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
