"""``nowcast-predict``: CLI mirror of ``Predictor.predict``.

    nowcast-predict --model artifacts/models/nowcast/latest --event path.zarr \
                    --t0 2026-05-12T10:30Z --out forecast.zarr [--baseline extrapolation]

``--t0 auto`` uses the last frame of the event. Exit code 2 on input-contract
errors, 3 on model-load errors.
"""

from __future__ import annotations

import argparse
import json
import sys

import xarray as xr

from nowcast_ml.inference.errors import InputContractError, ModelLoadError


def _to_zarr(ds: xr.Dataset, out: str) -> None:
    ds = ds.copy()
    ds.attrs = {
        k: (json.dumps(v) if isinstance(v, list | dict) else v) for k, v in ds.attrs.items()
    }
    ds.to_zarr(out, mode="w", consolidated=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="nowcast-predict",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--event", required=True, help="input-contract Zarr store")
    ap.add_argument(
        "--t0", required=True, help="ISO-8601 UTC time of the last input frame, or 'auto'"
    )
    ap.add_argument("--out", required=True, help="output Zarr path")
    ap.add_argument(
        "--baseline",
        choices=["persistence", "extrapolation", "steps"],
        help="run a baseline instead",
    )
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    from nowcast_ml.inference.predictor import Predictor

    try:
        pred = Predictor.load(args.model, device=args.device)
    except ModelLoadError as e:
        print(f"model load error: {e}", file=sys.stderr)
        sys.exit(3)
    try:
        ds_in = xr.open_zarr(args.event, consolidated=None)
    except Exception as e:  # noqa: BLE001
        print(f"cannot open {args.event}: {e}", file=sys.stderr)
        sys.exit(2)
    t0 = ds_in["time"].values[-1] if args.t0 == "auto" else args.t0
    try:
        if args.baseline:
            fc = pred.predict_baseline(ds_in, t0, kind=args.baseline)
        else:
            fc = pred.predict(ds_in, t0)
    except InputContractError as e:
        print(f"input contract error: {e}", file=sys.stderr)
        sys.exit(2)
    _to_zarr(fc, args.out)
    a = fc.attrs
    print(
        f"{a['model_name']} {a['model_version']} t0={a['t0']} mode={a['mode']} "
        f"inference_ms={a['inference_ms']:.0f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
