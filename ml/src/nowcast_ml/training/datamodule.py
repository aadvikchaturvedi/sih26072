"""LightningDataModule over synthetic, India (Zarr) or SEVIR events.

Splits are by event (``data/splits.py``). Normalization statistics are loaded
from ``data.norm_stats`` or fitted on the *train* split only.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Dataset

from nowcast_ml.config import Config
from nowcast_ml.data import synthetic as syn
from nowcast_ml.data.event import Event
from nowcast_ml.data.splits import split_events
from nowcast_ml.data.transforms import NormStats, fit_norm_stats, random_crop, random_flip
from nowcast_ml.data.zarr_events import EventWindowDataset, WindowSpec, list_event_paths
from nowcast_ml.utils.logging import get_logger

log = get_logger(__name__)


class EventSources:
    """Resolves the configured data source into per-split event openers."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        d = cfg.data
        self.channels = list(d.channels)
        self.openers: dict[str, object] = {}
        self.n_times: dict[str, int] = {}
        random_window = False
        if d.source == "sevir" and not d.events_dir:
            from nowcast_ml.data.sevir import read_catalog, sevir_event_openers, sevir_n_frames

            entries = read_catalog(d.sevir.root, d.sevir.catalog)
            if d.sevir.max_events:
                entries = entries[: d.sevir.max_events]
            ops = sevir_event_openers(d, entries, self.channels)
            for e, op in zip(entries, ops, strict=True):
                eid = f"sevir_{e.event_id}"
                self.openers[eid] = op
                self.n_times[eid] = sevir_n_frames(d.sevir.frame_stride)
            random_window = True
        elif d.events_dir:
            import xarray as xr

            for p in list_event_paths(d.events_dir):
                self.openers[p.stem] = partial(
                    Event.open, p, self.channels
                )  # picklable for workers
                self.n_times[p.stem] = xr.open_zarr(str(p), consolidated=None).sizes["time"]
        elif d.source == "synthetic":
            s = d.synthetic
            for ev in syn.generate_events(
                s.n_events,
                seed=s.seed,
                size=s.size,
                n_frames=s.n_frames,
                n_storms=tuple(s.n_storms),
                speed_px=tuple(s.speed_px),
                radar_missing_prob=s.radar_missing_prob,
                grid_spacing_km=d.grid_spacing_km,
            ):
                e = Event(syn.to_dataset(ev), self.channels)
                self.openers[ev.event_id] = e
                self.n_times[ev.event_id] = ev.fields["maxz"].shape[0]
        else:
            raise ValueError(f"data.source={d.source!r} requires data.events_dir")
        if not self.openers:
            raise ValueError("no events found for the configured data source")
        self.random_window_train = random_window
        self.splits = split_events(sorted(self.openers), d.splits)

    def dataset(
        self, split: str, transform=None, stride: int | None = None, random_window=None
    ) -> EventWindowDataset:
        ids = self.splits[split]
        d = self.cfg.data
        rw = (
            self.random_window_train and split == "train"
            if random_window is None
            else random_window
        )
        return EventWindowDataset(
            [self.openers[i] for i in ids],
            WindowSpec.from_config(d),
            stride=stride or d.sample_stride,
            random_window=rw,
            transform=transform,
            seed=self.cfg.train.seed,
            n_times=[self.n_times[i] for i in ids],
        )


class TrainTransform:
    def __init__(self, crop: int | None, flips: bool):
        self.crop, self.flips = crop, flips

    def __call__(self, sample: dict, rng: np.random.Generator) -> dict:
        if self.crop:
            sample = random_crop(sample, self.crop, rng)
        if self.flips:
            sample = random_flip(sample, rng)
        return sample


def fit_stats_from_dataset(
    ds: Dataset, channels: list[str], max_samples: int, seed: int = 0
) -> NormStats:
    n = len(ds)
    idx = np.random.default_rng(seed).permutation(n)[: max(1, min(max_samples, n))]
    return fit_norm_stats((ds[int(i)]["x"] for i in idx), channels)


class NowcastDataModule(L.LightningDataModule):
    def __init__(self, cfg: Config, norm_stats: NormStats | None = None):
        super().__init__()
        self.cfg = cfg
        self.norm_stats = norm_stats
        self.sources: EventSources | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.sources is not None:
            return
        d = self.cfg.data
        self.sources = EventSources(self.cfg)
        log.info(
            "events: %s",
            {k: len(v) for k, v in self.sources.splits.items()},
        )
        self.train_ds = self.sources.dataset("train", TrainTransform(d.crop, d.flips))
        self.val_ds = self.sources.dataset("val")
        if self.norm_stats is None:
            if d.norm_stats:
                self.norm_stats = NormStats.load(d.norm_stats)
            else:
                self.norm_stats = fit_stats_from_dataset(
                    self.sources.dataset("train"),
                    d.channels,
                    d.max_norm_samples,
                    self.cfg.train.seed,
                )
        self.norm_stats.check_channels(d.channels)

    @property
    def splits(self) -> dict[str, list[str]]:
        return self.sources.splits

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        d = self.cfg.data
        return DataLoader(
            ds,
            batch_size=d.batch_size,
            shuffle=shuffle,
            num_workers=d.num_workers,
            persistent_workers=d.num_workers > 0,
            drop_last=shuffle and len(ds) > d.batch_size,
            pin_memory=False,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)


def save_splits(splits: dict, path: str | Path) -> None:
    from nowcast_ml.utils.io import write_json

    write_json(path, splits)
