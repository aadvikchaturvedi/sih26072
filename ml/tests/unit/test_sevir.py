"""SEVIR loader against a fake file with the real SEVIR layout (no download)."""

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from nowcast_ml.data import channels as ch
from nowcast_ml.data.event import Event
from nowcast_ml.data.schema import validate_dataset
from nowcast_ml.data.sevir import (
    ir_raw_to_kelvin,
    load_sevir_event,
    read_catalog,
    sevir_n_frames,
    vil_digital_to_kgm2,
    vil_to_dbz,
)
from nowcast_ml.data.zarr_events import WindowSpec, make_sample


def _write_fake_sevir(root: Path, n=2):
    r = np.random.default_rng(0)
    ids = [f"S{i:06d}" for i in range(n)]
    vil = np.zeros((n, 384, 384, 49), np.uint8)
    vil[:, 150:230, 150:230, :] = 150  # a strong core
    ir = np.full((n, 192, 192, 49), 2000, np.int16)  # 20 C
    ir[:, 70:120, 70:120, :] = -6000  # -60 C anvil
    with h5py.File(root / "vil.h5", "w") as f:
        f["vil"] = vil
        f["id"] = np.array(ids, dtype="S")
    for t in ("ir107", "ir069"):
        with h5py.File(root / f"{t}.h5", "w") as f:
            f[t] = ir
            f["id"] = np.array(ids, dtype="S")
    rows = []
    with h5py.File(root / "lght.h5", "w") as f:
        for eid in ids:
            # flashes at the core centre (lat 35.0, lon -97.0), 30 min after the reference time
            f[eid] = np.column_stack(
                [
                    np.full(20, 1800.0),
                    np.full(20, 35.0),
                    np.full(20, -97.0),
                    r.random(20),
                    r.random(20),
                ]
            )
    for i, eid in enumerate(ids):
        for t, fn in (
            ("vil", "vil.h5"),
            ("ir107", "ir107.h5"),
            ("ir069", "ir069.h5"),
            ("lght", "lght.h5"),
        ):
            rows.append(
                dict(
                    id=eid,
                    file_name=fn,
                    file_index=i,
                    img_type=t,
                    time_utc="2019-06-01 18:00:00",
                    event_id=100 + i,
                    event_type="Hail",
                    llcrnrlat=33.27,
                    llcrnrlon=-99.1,
                    urcrnrlat=36.73,
                    urcrnrlon=-94.9,
                )
            )
    pd.DataFrame(rows).to_csv(root / "CATALOG.csv", index=False)


def test_unit_conversions():
    assert vil_digital_to_kgm2(np.array([0, 5]))[1] == 0
    v = vil_digital_to_kgm2(np.arange(256))
    assert np.all(np.diff(v) >= 0)
    dbz = vil_to_dbz(np.array([0.0, 0.1, 1.0, 10.0]))
    assert dbz[0] == 0 and 10 < dbz[1] < dbz[2] < dbz[3] < 60
    assert ir_raw_to_kelvin(np.array([0]))[0] == pytest.approx(273.15)


def test_fake_sevir_event(tmp_path):
    _write_fake_sevir(tmp_path)
    entries = read_catalog(tmp_path)
    assert len(entries) == 2
    ds = load_sevir_event(tmp_path, entries[0], size=64)
    assert validate_dataset(ds, expected_spacing_km=None) == []
    assert ds.sizes["time"] == sevir_n_frames() == 24
    ev = Event(ds, list(ch.ALL_CHANNELS))
    assert set(ev.missing_channels) == set(ch.ALL_CHANNELS) - set(ch.SEVIR_CHANNELS)
    s = make_sample(ev, 10, WindowSpec())
    assert s["x"].shape == (7, 14, 64, 64)
    maxz = s["x"][-1, 0]
    assert maxz[32, 32] > 40 and maxz[2, 2] == 0
    assert 200 < s["x"][-1, 2, 32, 32] < 220  # tir1_bt of the anvil in K
    # flashes 30 min after reference -> frame 14 (offsets are -110 + 10k min); visible in the +30 label of t0 = 11 (frames 12..14)
    assert (
        ev.lightning_occurrence[14].sum() > 0
        and ev.lightning_occurrence.sum() == ev.lightning_occurrence[14].sum()
    )
    s12 = make_sample(ev, 11, WindowSpec(lightning_radius_km=10.0))
    assert s12["y_ltg"][0].sum() > 0


@pytest.mark.sevir
def test_real_sevir_lightning_colocated_with_vil():
    """Orientation check on real downloaded events: flashes must sit on strong echoes.

    Run with: pytest -m sevir  (needs scripts/download_sevir_subset.py output in data/sevir_raw)
    """
    from scipy.ndimage import maximum_filter

    from nowcast_ml.data.sevir import sevir_event_from_arrays

    raw = Path(
        os.environ.get("SEVIR_RAW", Path(__file__).resolve().parents[2] / "data" / "sevir_raw")
    )
    if not (raw / "catalog.csv").exists():
        pytest.skip("SEVIR subset not downloaded")
    entries = {e.event_id: e for e in read_catalog(raw, "catalog.csv", require_lightning=True)}
    hits = total = 0
    for p in sorted(raw.glob("*.npz"))[:20]:
        a = np.load(p)
        ds = sevir_event_from_arrays(entries[p.stem], a["vil"], a["ir107"], a["ir069"], a["lght"])
        z = maximum_filter(np.nan_to_num(ds["x"].sel(channel="maxz").values), size=(1, 3, 3))
        fd = np.nan_to_num(ds["x"].sel(channel="flash_density").values)
        hits += int(((fd > 0) & (z >= 30)).sum())
        total += int((fd > 0).sum())
    assert total > 0 and hits / total > 0.8, hits / max(total, 1)
