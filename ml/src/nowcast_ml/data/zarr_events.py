"""India event Zarr dataset: sliding windows of (7 input, 12 target) frames.

A sample is a dict of numpy arrays::

    x            (t_in, C, H, W)  float32  physical units, NaN where unavailable
    avail        (t_in, C, H, W)  bool
    y_refl       (t_out, H, W)    float32  target maxz (dBZ), NaN where unavailable
    y_refl_valid (t_out, H, W)    bool
    y_ltg        (n_leads, H, W)  float32  lightning labels (+30, +60)
    y_ltg_valid  (n_leads, H, W)  bool
    past_ltg     (H, W)           bool     flash within radius in the past 30 min
    t0_index     int              index of the last input frame in the event
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from nowcast_ml.data import channels as ch
from nowcast_ml.data import labels as lb
from nowcast_ml.data.event import Event


@dataclass(frozen=True)
class WindowSpec:
    t_in: int = 7
    t_out: int = 12
    step_minutes: int = 10
    leads_min: tuple[int, ...] = (30, 60)
    lightning_radius_km: float = 10.0

    @classmethod
    def from_config(cls, data_cfg) -> WindowSpec:
        return cls(
            t_in=data_cfg.t_in,
            t_out=data_cfg.t_out,
            step_minutes=data_cfg.step_minutes,
            leads_min=tuple(data_cfg.lightning_leads_min),
            lightning_radius_km=data_cfg.lightning_radius_km,
        )


def list_event_paths(events_dir: str | Path) -> list[Path]:
    p = Path(events_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"events directory {p} not found")
    return sorted(q for q in p.iterdir() if q.suffix == ".zarr" and q.is_dir())


def valid_t0_indices(n_times: int, spec: WindowSpec, stride: int = 1) -> list[int]:
    """Forecast times with a full input history and full target horizon."""
    return list(range(spec.t_in - 1, n_times - spec.t_out, max(1, stride)))


def make_sample(event: Event, t0: int, spec: WindowSpec, with_targets: bool = True) -> dict:
    x, avail = event.window(t0 - spec.t_in + 1, t0 + 1)
    radius = lb.radius_px(spec.lightning_radius_km, event.grid_spacing_km)
    occ = event.lightning_occurrence
    sample = {
        "x": x,
        "avail": avail,
        "past_ltg": lb.past_lightning(occ, t0, 30, spec.step_minutes, radius),
        "t0_index": t0,
    }
    if with_targets:
        y, yv = event.target(t0 + 1, t0 + 1 + spec.t_out)
        sample["y_refl"] = y
        sample["y_refl_valid"] = yv
        sample["y_ltg"] = lb.lightning_labels(occ, t0, spec.leads_min, spec.step_minutes, radius)
        sample["y_ltg_valid"] = lb.lightning_label_valid(
            event.lightning_missing, t0, spec.leads_min, spec.step_minutes
        )
    return sample


class EventWindowDataset(Dataset):
    """Windows over a list of events.

    ``events`` are either :class:`Event` objects or zero-arg callables returning one
    (lazy open inside DataLoader workers). With ``random_window=True`` the dataset
    has one item per event and a random valid t0 is drawn on each access (SEVIR style).
    """

    def __init__(
        self,
        events: Sequence[Event | Callable[[], Event]],
        spec: WindowSpec,
        stride: int = 1,
        random_window: bool = False,
        transform: Callable[[dict, np.random.Generator], dict] | None = None,
        seed: int = 0,
        n_times: Sequence[int] | None = None,
    ):
        self._events = list(events)
        self._cache: dict[int, Event] = {}
        self.spec = spec
        self.random_window = random_window
        self.transform = transform
        self.seed = seed
        self._epoch_rng = np.random.default_rng(seed)
        if n_times is None:
            n_times = [self._get(i).n_times for i in range(len(self._events))]
        self._t0s = [valid_t0_indices(n, spec, stride) for n in n_times]
        if random_window:
            self.index = [(i, -1) for i, t in enumerate(self._t0s) if t]
        else:
            self.index = [(i, t0) for i, ts in enumerate(self._t0s) for t0 in ts]

    def _get(self, i: int) -> Event:
        ev = self._events[i]
        if isinstance(ev, Event):
            return ev
        if i not in self._cache:
            self._cache[i] = ev()
        return self._cache[i]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> dict:
        i, t0 = self.index[k]
        rng = np.random.default_rng((self.seed, k, int(self._epoch_rng.integers(1 << 31))))
        if t0 < 0:
            t0 = int(rng.choice(self._t0s[i]))
        ev = self._get(i)
        s = make_sample(ev, t0, self.spec)
        s["event_index"] = i
        if self.transform is not None:
            s = self.transform(s, rng)
        return s


def zarr_event_dataset(
    paths: Sequence[Path],
    channels: list[str],
    spec: WindowSpec,
    stride: int = 1,
    transform=None,
    seed: int = 0,
) -> EventWindowDataset:
    import xarray as xr

    n_times = [xr.open_zarr(str(p), consolidated=None).sizes["time"] for p in paths]
    openers = [partial(Event.open, p, channels) for p in paths]
    return EventWindowDataset(
        openers, spec, stride=stride, transform=transform, seed=seed, n_times=n_times
    )


def default_channels() -> list[str]:
    return list(ch.ALL_CHANNELS)
