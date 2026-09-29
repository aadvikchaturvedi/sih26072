#!/usr/bin/env python
"""Download a small SEVIR subset from the public AWS bucket (anonymous, s3://sevir).

Full SEVIR files are multi-GB; this reads only the selected events' slices over S3
(h5py on an s3fs file object) and writes compact local HDF5 files plus a rewritten
CATALOG.csv that ``nowcast_ml.data.sevir`` can read directly.

    pip install -e ".[sevir]"
    python scripts/download_sevir_subset.py --out data/sevir --n-events 200 --seed 0

Needs network access; nothing in the test suite calls it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET = "sevir"
TYPES = ("vil", "ir107", "ir069", "lght")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="data/sevir")
    ap.add_argument("--n-events", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--event-types",
        nargs="*",
        default=["Thunderstorm Wind", "Hail", "Heavy Rain", "Tornado"],
        help="NOAA storm-event types to keep (catalog column event_type)",
    )
    args = ap.parse_args(argv)

    import h5py
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("reading catalog ...")
    with fs.open(f"{BUCKET}/CATALOG.csv") as f:
        cat = pd.read_csv(f, low_memory=False)
    cat = cat[cat["event_id"].notna()]
    if args.event_types:
        cat = cat[cat["event_type"].isin(args.event_types)]
    have = cat.groupby("id")["img_type"].apply(set)
    ids = sorted(i for i, s in have.items() if set(TYPES) <= s)
    rng = np.random.default_rng(args.seed)
    ids = sorted(rng.choice(ids, size=min(args.n_events, len(ids)), replace=False).tolist())
    sub = cat[cat["id"].isin(ids) & cat["img_type"].isin(TYPES)].copy()
    print(f"selected {len(ids)} events")

    new_rows = []
    for img_type in ("vil", "ir107", "ir069"):
        rows = sub[sub["img_type"] == img_type]
        by_file = defaultdict(list)
        for r in rows.itertuples():
            by_file[r.file_name].append(r)
        local = out / f"{img_type}_subset.h5"
        arrays, idlist = [], []
        for fname, rs in sorted(by_file.items()):
            print(f"  {img_type}: {fname} ({len(rs)} events)")
            with fs.open(f"{BUCKET}/data/{fname}", "rb") as fo, h5py.File(fo, "r") as h:
                for r in sorted(rs, key=lambda r: r.file_index):
                    arrays.append(h[img_type][int(r.file_index)])
                    idlist.append(r.id)
        with h5py.File(local, "w") as h:
            h.create_dataset(
                img_type, data=np.stack(arrays), compression="gzip", compression_opts=4
            )
            h.create_dataset("id", data=np.array(idlist, dtype="S"))
        for k, eid in enumerate(idlist):
            row = rows[rows["id"] == eid].iloc[0].to_dict()
            row.update(file_name=local.name, file_index=k)
            new_rows.append(row)

    lrows = sub[sub["img_type"] == "lght"]
    local = out / "lght_subset.h5"
    with h5py.File(local, "w") as hout:
        for fname, rs in lrows.groupby("file_name"):
            print(f"  lght: {fname} ({len(rs)} events)")
            with fs.open(f"{BUCKET}/data/{fname}", "rb") as fo, h5py.File(fo, "r") as h:
                for r in rs.itertuples():
                    if r.id in h:
                        hout.create_dataset(r.id, data=h[r.id][:])
                    row = r._asdict()
                    row.pop("Index", None)
                    row.update(file_name=local.name, file_index=0)
                    new_rows.append(row)
    pd.DataFrame(new_rows).to_csv(out / "CATALOG.csv", index=False)
    print(f"wrote subset to {out}")


if __name__ == "__main__":
    main()
