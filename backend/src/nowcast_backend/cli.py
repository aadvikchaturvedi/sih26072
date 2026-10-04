"""``nowcast-backend``: run the API server, or push frames to a running one."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _serve(args) -> None:
    import uvicorn
    from nowcast_ml.inference.errors import ModelLoadError

    from nowcast_backend.adapters.engine_ml import PredictorEngine
    from nowcast_backend.api.app import create_app
    from nowcast_backend.settings import Settings

    settings = Settings()
    try:  # fail with a readable message, before the server starts, if the model is unusable
        engine = PredictorEngine.load(
            settings.model_path, device=settings.device, backend=settings.engine_backend
        )
    except ModelLoadError as e:
        sys.exit(f"cannot start nowcasting backend: {e}")
    uvicorn.run(
        create_app(settings, engine),
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_level="info",
    )


def _push(args) -> None:
    import httpx
    import xarray as xr

    from nowcast_backend.client import push_frames

    ds = xr.open_zarr(args.store, consolidated=None)
    if args.last:
        ds = ds.isel(time=slice(-args.last, None))
    try:
        result = push_frames(args.url, args.domain, ds.load(), api_key=args.api_key)
    except httpx.HTTPStatusError as e:
        sys.exit(f"rejected ({e.response.status_code}): {e.response.text}")
    print(json.dumps(result, indent=2))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(prog="nowcast-backend", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the API server (configured by NOWCAST_* variables)")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=_serve)

    push = sub.add_parser("push", help="push frames of an input-contract Zarr store to a server")
    push.add_argument("store", help="path to a contract .zarr store")
    push.add_argument("--domain", required=True)
    push.add_argument("--url", default="http://127.0.0.1:8000")
    push.add_argument("--api-key")
    push.add_argument("--last", type=int, help="send only the last N frames")
    push.set_defaults(func=_push)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
