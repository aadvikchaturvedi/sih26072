"""Dummy data for running the whole application before real data exists.

``setup`` produces, under one directory:

* ``train_events/``  synthetic storm events (``nowcast_ml.data.synthetic``)
* ``models/``        a small model trained and calibrated on them
* ``report/``        its evaluation against the baselines (feeds the skill page)
* ``events/odisha.zarr``  one synthetic afternoon over Odisha to replay

Nothing here says anything about real skill: storms, satellite, lightning and
NWP fields are all generated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
ML_CONFIGS = REPO / "ml" / "configs"
#: District boundaries shared with the web console (community GeoJSON, demo only).
DISTRICTS_GEOJSON = REPO / "web" / "src" / "assets" / "geo" / "odisha_districts.json"

#: The console's map extent (same box the frontend's mock scenario uses).
ODISHA_BBOX = {"west": 84.0, "east": 87.5, "south": 19.5, "north": 22.5}
SCENARIO_DOMAIN = "odisha"
SCENARIO_START = "2026-04-18T09:00"  # 14:30 IST, a pre-monsoon afternoon
SCENARIO_FRAMES = 22  # to 12:30 UTC
RADAR_OUTAGE_FROM = "2026-04-18T11:30"  # shows the satellite-only fallback


def write_scenario(path: Path, seed: int = 28) -> Path:
    """One synthetic storm afternoon on a ~2 km grid covering ``ODISHA_BBOX``."""
    from nowcast_ml.data import channels as ch
    from nowcast_ml.data.schema import validate_dataset
    from nowcast_ml.data.synthetic import generate_event, to_dataset

    b = ODISHA_BBOX
    mid_lat = (b["south"] + b["north"]) / 2
    H = int(round((b["north"] - b["south"]) * 111.19 / 2.0))
    W = int(round((b["east"] - b["west"]) * 111.19 * np.cos(np.deg2rad(mid_lat)) / 2.0))
    dlat, dlon = (b["north"] - b["south"]) / H, (b["east"] - b["west"]) / W
    size = max(H, W)

    # Storms drift south-east at about 30 km/h, like a nor'wester line.
    ev = generate_event(
        seed,
        size=size,
        n_frames=SCENARIO_FRAMES,
        n_storms=(5, 5),
        motion=(1.6, 1.9),
        start=SCENARIO_START,
        event_id=SCENARIO_DOMAIN,
    )
    # Pixel centres on a regular lat/lon grid (row 0 = north), so images stretch exactly
    # over the bounding box. The generator's square grid is cropped to the box afterwards.
    ev.lat = (b["north"] - (np.arange(size) + 0.5) * dlat)[:, None].repeat(size, axis=1)
    ev.lon = (b["west"] + (np.arange(size) + 0.5) * dlon)[None, :].repeat(size, axis=0)
    outage = ev.times >= pd.Timestamp(RADAR_OUTAGE_FROM)
    ev.missing["radar"][outage] = True
    for name in ch.channels_in_group("radar"):
        ev.fields[name][outage] = np.nan

    ds = to_dataset(ev).isel(y=slice(0, H), x=slice(0, W))
    problems = validate_dataset(ds)
    if problems:
        raise RuntimeError(f"scenario violates the input contract: {problems}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(str(path), mode="w", consolidated=True)
    return path


def train_model(root: Path, n_events: int = 24, epochs: int = 12) -> Path:
    """Train, calibrate and evaluate a small model on synthetic events; returns ``latest``."""
    from nowcast_ml.calibration import fit_cli
    from nowcast_ml.config import load_config
    from nowcast_ml.data.synthetic import write_events
    from nowcast_ml.evaluation import evaluate
    from nowcast_ml.training.train import run

    events = root / "train_events"
    if not any(events.glob("*.zarr")):
        log.info("writing %d synthetic training events", n_events)
        write_events(events, n_events, seed=5, size=160, n_frames=26, radar_missing_prob=0.1)
    cfg = load_config(
        ML_CONFIGS / "train" / "smoke.yaml",
        [
            f"data.events_dir={events}",
            "data.crop=64",
            "data.batch_size=8",
            "data.sample_stride=2",
            "model.hid_s=32",
            "model.hid_t=64",
            "model.n_t=3",
            "model.lightning_head.hidden=32",
            f"train.max_epochs={epochs}",
            "train.accelerator=auto",
            f"train.output_dir={root / 'train'}",
            f"registry.root={root / 'models'}",
        ],
    )
    run(cfg)
    latest = root / "models" / "nowcast" / "latest"
    fit_cli.main(["--model", str(latest), "--events", str(events), "--split", "val"])
    evaluate.main(
        [
            "--model", str(latest), "--events", str(events), "--split", "test",
            "--baselines", "all", "--out", str(root / "report"),
            "eval.steps_members=5", "eval.sample_stride=6",
        ]
    )  # fmt: skip
    return latest


def setup(root: Path, retrain: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if retrain or not (root / "models" / "nowcast" / "latest").exists():
        train_model(root)
    scenario = root / "events" / f"{SCENARIO_DOMAIN}.zarr"
    if retrain or not scenario.exists():
        write_scenario(scenario)
    log.info("demo ready: model in %s, scenario %s", root / "models", scenario)
