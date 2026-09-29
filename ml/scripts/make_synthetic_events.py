#!/usr/bin/env python
"""Write contract-valid synthetic event Zarr stores (for smoke training / e2e / docs)."""

from __future__ import annotations

import argparse

from nowcast_ml.data.schema import validate_event
from nowcast_ml.data.synthetic import write_events


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-events", type=int, default=12)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--radar-missing-prob", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    paths = write_events(
        args.out,
        args.n_events,
        seed=args.seed,
        size=args.size,
        n_frames=args.frames,
        radar_missing_prob=args.radar_missing_prob,
    )
    for p in paths:
        problems = validate_event(p)
        if problems:
            raise SystemExit(f"{p}: {problems}")
    print(f"wrote {len(paths)} valid events to {args.out}")


if __name__ == "__main__":
    main()
