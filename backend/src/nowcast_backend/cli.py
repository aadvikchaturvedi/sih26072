"""``nowcast-backend``: run the API server (on real or dummy data), or push frames to one."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _run(settings, host: str | None, port: int | None) -> None:
    import uvicorn
    from nowcast_ml.inference.errors import ModelLoadError

    from nowcast_backend.adapters.engine_ml import PredictorEngine
    from nowcast_backend.api.app import create_app

    try:  # fail with a readable message, before the server starts, if the model is unusable
        engine = PredictorEngine.load(
            settings.model_path, device=settings.device, backend=settings.engine_backend
        )
    except ModelLoadError as e:
        sys.exit(f"cannot start nowcasting backend: {e}")
    uvicorn.run(
        create_app(settings, engine),
        host=host or settings.host,
        port=port or settings.port,
        log_level="info",
    )


def _serve(args) -> None:
    from nowcast_backend.settings import Settings

    _run(Settings(), args.host, args.port)


def _demo(args) -> None:
    """Everything on dummy data: a model trained on synthetic storms replaying a
    synthetic afternoon over Odisha, with the web console's origins allowed."""
    import shutil
    from pathlib import Path

    from nowcast_backend import demo
    from nowcast_backend.settings import Settings

    root = Path(args.dir)
    demo.setup(root, retrain=args.retrain)
    if args.fresh:
        shutil.rmtree(root / "state", ignore_errors=True)
    settings = Settings(
        model_path=root / "models" / "nowcast" / "latest",
        data_dir=root / "state",
        source="replay",
        source_path=root / "events" / f"{demo.SCENARIO_DOMAIN}.zarr",
        replay_domain=demo.SCENARIO_DOMAIN,
        console_domain=demo.SCENARIO_DOMAIN,
        # --live-from N holds frames back so they arrive one per poll, like live data
        replay_initial_frames=args.live_from or demo.SCENARIO_FRAMES,
        poll_seconds=args.poll_seconds,
        clock="data",
        regions_path=demo.DISTRICTS_GEOJSON,
        skill_report_path=root / "report" / "metrics.json",
        cors_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    )
    _run(settings, args.host, args.port)


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

    demo = sub.add_parser("demo", help="prepare dummy data and a dummy model, then serve them")
    demo.add_argument("--dir", default="var/demo", help="where the dummy data and model live")
    demo.add_argument("--retrain", action="store_true", help="train the dummy model again")
    demo.add_argument("--fresh", action="store_true", help="discard stored forecasts first")
    demo.add_argument(
        "--live-from",
        type=int,
        metavar="N",
        help="preload only the first N frames; the rest arrive one per poll",
    )
    demo.add_argument("--poll-seconds", type=float, default=10.0)
    demo.add_argument("--host")
    demo.add_argument("--port", type=int)
    demo.set_defaults(func=_demo)

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
