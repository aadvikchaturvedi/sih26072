"""The console API: the shapes the web console validates (``web/src/data/schemas.ts``)."""

from __future__ import annotations

import json
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nowcast_backend.adapters.zarr_zip import encode
from nowcast_backend.api.app import create_app
from nowcast_backend.domain.grid import Grid
from nowcast_backend.domain.models import DistrictForecast, Region
from nowcast_backend.services import cells as cell_service
from nowcast_backend.services import districts as district_service
from nowcast_backend.services.cap import CAP_NS
from tests.conftest import frames

API = "/api/v1/console"
T0 = "2026-05-12T08:10:00Z"  # frame 13 of the synthetic event

METRICS = {
    "meta": {
        "leads_min": [10, 20],
        "n_events": 2,
        "n_samples": 4,
        "split": "test",
        "synthetic": True,
    },
    "forecasters": {
        name: {
            "reflectivity": {
                "csi": {"20.0": [0.8, 0.6], "35.0": [0.5, 0.3]},
                "pod": {"35": [0.7, 0.5]},
                "far": {"35": [0.2, float("nan")]},
            },
            "lightning": {
                "30": {
                    "bss_climatology": 0.4,
                    "roc_auc": 0.9,
                    "reliability": {
                        "mean_forecast": [0.05, float("nan"), 0.8],
                        "observed_freq": [0.02, float("nan"), 0.7],
                        "count": [100, 0, 20],
                    },
                }
            },
            "first_flash": {
                "hit_rate": 0.5,
                "false_alarm_ratio": float("nan"),
                "median_lead_min": 20.0,
            },
        }
        for name in ("persistence", "model_full")
    },
}


def square(west, east, south, north) -> dict:
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {"type": "Polygon", "coordinates": [ring]}


@pytest.fixture()
def console(settings, engine, event, tmp_path):
    """A client whose console domain holds 14 frames; two districts split the grid west / east."""
    lat, lon = event["lat"].values, event["lon"].values
    mid = float(lon.mean())
    regions = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": name, "name": name.title()},
                "geometry": square(west, east, lat.min() - 0.1, lat.max() + 0.1),
            }
            for name, west, east in (("west", lon.min() - 0.1, mid), ("east", mid, lon.max() + 0.1))
        ],
    }
    (tmp_path / "regions.json").write_text(json.dumps(regions))
    (tmp_path / "metrics.json").write_text(json.dumps(METRICS))
    settings = settings.model_copy(
        update={
            "console_domain": "kolkata",
            "regions_path": tmp_path / "regions.json",
            "skill_report_path": tmp_path / "metrics.json",
        }
    )
    with TestClient(create_app(settings, engine)) as client:
        body = encode(frames(event, 0, 14))
        assert client.post("/api/v1/domains/kolkata/observations", content=body).status_code == 200
        yield client


def test_timeline_and_health(console, event):
    timeline = console.get(f"{API}/timeline").json()
    assert len(timeline["times"]) == 11 and timeline["times"][-1] == T0
    assert timeline["stepMin"] == 10 and timeline["leads"][-1] == 120
    box = timeline["bbox"]  # image edges: half a pixel beyond the outermost pixel centres
    assert box["north"] > event["lat"].values.max() and box["west"] < event["lon"].values.min()

    health = console.get(f"{API}/health/{T0}").json()
    assert health == console.get(f"{API}/health").json()
    assert health["lastRunAt"] == T0 and health["nextRunAt"] == "2026-05-12T08:20:00Z"
    assert health["mode"] == "replay" and health["modelMode"] == "full"
    assert {i["name"]: i["status"] for i in health["inputs"]} == {
        "radar": "live", "satellite": "live", "lightning": "live", "nwp": "live",
    }  # fmt: skip


def test_radar_outage_shows_in_health(console, event):
    blind = frames(event, 14, 16).copy(deep=True)
    blind["missing"].loc[{"group": "radar"}] = 1
    console.post("/api/v1/domains/kolkata/observations", content=encode(blind))
    health = console.get(f"{API}/health").json()
    radar = next(i for i in health["inputs"] if i["name"] == "radar")
    assert health["modelMode"] == "satellite_only"
    assert radar == {
        "name": "radar",
        "status": "stale",
        "lastReceived": T0,
        "staleAgeSeconds": 1200,
    }
    cells = console.get(f"{API}/cells/{health['lastRunAt']}").json()
    assert all(c["modelMode"] == "satellite_only" and c["dataQuality"] < 1 for c in cells)
    # the observed-radar image of a time without radar is simply empty
    assert console.get(f"{API}/radar/{health['lastRunAt']}").status_code == 200
    assert console.get(f"{API}/radar/2020-01-01T00:00:00Z").headers["content-type"] == "image/png"


def test_forecast_urls_resolve(console):
    fc = console.get(f"{API}/forecast/{T0}").json()
    assert fc["attrs"]["t0"] == T0 and fc["attrs"]["output_contract_version"] == "1.0"
    assert sorted(map(int, fc["reflectivityUrlByLead"])) == list(range(10, 121, 10))
    assert "ensembleSpreadUrl" not in fc
    for url in (fc["reflectivityUrlByLead"]["60"], fc["lightningProb30Url"], fc["firstFlashUrl"]):
        image = console.get(url)
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert console.get(f"{API}/forecast/2020-01-01T00:00:00Z").status_code == 404


def test_cells_and_strokes(console):
    cells = console.get(f"{API}/cells/{T0}").json()
    assert cells
    cell = cells[0]
    assert set(cell) == {
        "id", "isNewCell", "status", "lat", "lon", "maxDbz", "growthDbzPer10Min", "lightningTrend",
        "headingDeg", "speedKmh", "lightningJumpFlag", "lightningJumpAt", "track", "forecastPath",
        "uncertaintyCone", "affectedDistricts", "modelMode", "dataQuality", "flashRateSeries",
        "dbzSeries",
    }  # fmt: skip
    assert cell["track"][-1]["time"] == T0 and cell["dataQuality"] == 1.0
    leads = [p["leadMin"] for p in cell["forecastPath"]]
    assert leads == sorted(leads) and leads[0] == 10
    basis = {p["leadMin"]: p["riskBasis"] for p in cell["forecastPath"]}
    assert basis[10] == "lightning_prob" and all(
        b == "reflectivity_based" for m, b in basis.items() if m > 60
    )
    assert cell["affectedDistricts"][0]["etaMin"] == 0
    assert [p["isForecast"] for p in cell["dbzSeries"]].count(False) == len(cell["track"])

    strokes = console.get(f"{API}/lightning/{T0}").json()
    assert strokes and set(strokes[0]) == {"id", "time", "lat", "lon", "cellId"}
    assert all("2026-05-12T07:40:00Z" < s["time"] <= T0 for s in strokes)
    assert {s["cellId"] for s in strokes} & {c["id"] for c in cells}


def test_districts_warnings_and_cap(console):
    districts = console.get(f"{API}/districts/{T0}").json()
    assert {d["districtId"] for d in districts} == {"west", "east"}
    assert all(d["warningLevel"] in ("green", "yellow", "orange", "red") for d in districts)

    warnings = console.get(f"{API}/warnings/{T0}").json()
    assert warnings, "the synthetic storms should raise at least one district warning"
    issued = [w["issuedAt"] for w in warnings]
    assert issued == sorted(issued, reverse=True)
    # the newest warning per district is that district's current level
    for d in districts:
        latest = next((w for w in warnings if w["districtId"] == d["districtId"]), None)
        assert (latest["level"] if latest else "green") == d["warningLevel"]
    # warnings accumulate with time and never change once issued
    earlier = console.get(f"{API}/warnings/2026-05-12T07:00:00Z").json()
    assert earlier == [w for w in warnings if w["issuedAt"] <= "2026-05-12T07:00:00Z"]

    w = warnings[-1]
    assert w["previousLevel"] is None and w["id"].endswith(w["districtId"])
    cap = console.get(f"{API}/cap/{w['id']}").json()
    alert = ET.fromstring(cap["xml"])
    ns = {"cap": CAP_NS}
    assert alert.findtext("cap:identifier", namespaces=ns) == w["id"] == cap["json"]["identifier"]
    assert alert.findtext("cap:msgType", namespaces=ns) == "Alert"
    assert "draft" in alert.findtext("cap:note", namespaces=ns)
    assert len(alert.findtext("cap:info/cap:area/cap:polygon", namespaces=ns).split()) == 5
    assert console.get(f"{API}/cap/nope").status_code == 404


def test_skill(console, settings, engine):
    report = console.get(f"{API}/skill").json()
    assert report["synthetic"] and report["eventLabel"].startswith("SYNTHETIC DATA")
    assert [m["modelId"] for m in report["models"]] == ["persistence", "F3F4"]
    model = report["models"][1]
    assert model["csiByThreshold"]["35"] == [
        {"leadMin": 10, "csi": 0.5},
        {"leadMin": 20, "csi": 0.3},
    ]
    assert model["pod"] == pytest.approx(0.6) and model["far"] == pytest.approx(0.2)
    assert model["firstFlashFAR"] is None  # NaN is not valid JSON
    assert len(model["reliabilityDiagram"]) == 2  # the empty bin is dropped
    assert report["medianFirstFlashLeadMin"] == 20.0
    with TestClient(create_app(settings, engine)) as bare:  # no evaluation configured
        assert bare.get(f"{API}/skill").status_code == 404


def test_live_tick(console, event):
    with console.websocket_connect(f"{API}/live") as ws:
        body = encode(frames(event, 14, 15))
        assert console.post("/api/v1/domains/kolkata/observations", content=body).status_code == 200
        tick = ws.receive_json()
    assert tick["t0"] == "2026-05-12T08:20:00Z" == tick["health"]["lastRunAt"]
    assert tick["updatedCells"] and all(w["issuedAt"] == tick["t0"] for w in tick["newWarnings"])


# ----------------------------------------------------------------------------- units
def test_region_index_and_affected(event):
    lat, lon = event["lat"].values, event["lon"].values
    grid = Grid(lat, lon, 2.0)
    mid = float(lon[0, 24])
    index = district_service.RegionIndex(
        [
            Region(
                id="w", name="W", geometry=square(lon.min() - 1, mid, lat.min() - 1, lat.max() + 1)
            ),
            Region(
                id="e", name="E", geometry=square(mid, lon.max() + 1, lat.min() - 1, lat.max() + 1)
            ),
        ],
        grid,
    )
    assert (index.labels[:, :24] == 1).all() and (index.labels[:, 25:] == 2).all()
    assert index.region_at(grid, lat[10, 5], lon[10, 5]).id == "w"
    assert index.region_at(grid, 0.0, 0.0) is None
    row, col = grid.pixel_of(lat[7, 30], lon[7, 30])
    assert (round(float(row[0])), round(float(col[0]))) == (7, 30)


def test_level_changes():
    def run(minute, **levels):
        districts = [
            DistrictForecast(
                district_id=d,
                name=d,
                lightning_prob_30=0.5,
                lightning_prob_60=0.5,
                max_dbz=30,
                area_fraction_above_threshold=0,
                first_flash=False,
                level=level,
                rule="r",
                trigger_value=0.5,
                trigger_threshold=0.3,
            )  # fmt: skip
            for d, level in levels.items()
        ]
        return pd.Timestamp("2026-05-12T08:00") + pd.Timedelta(minutes=minute), districts, []

    changes = district_service.level_changes(
        [
            run(0, a="green", b="yellow"),  # initial green is not a warning
            run(10, a="green", b="yellow"),  # nothing changed
            run(20, a="orange", b="green"),  # a rises; b is lifted
        ],
        valid_minutes=60,
    )
    assert [(w.district_id, w.previous_level, w.level) for w in changes] == [
        ("b", None, "yellow"), ("a", "green", "orange"), ("b", "yellow", "green"),
    ]  # fmt: skip
    assert changes[2].supersedes == changes[0].id and changes[0].supersedes is None
    assert changes[1].id == "W-20260512T0820Z-a"
    assert changes[1].valid_until - changes[1].issued_at == pd.Timedelta(minutes=60)


def test_lightning_jump_and_flash_rate(event):
    assert cell_service.lightning_jump([1.0, 1.2, 1.1, 1.3, 6.0])
    assert not cell_service.lightning_jump([1.0, 2.0, 3.0, 4.0, 5.0])  # steady rise
    assert not cell_service.lightning_jump([0.0, 0.0, 0.0, 0.0, 0.3])  # too small to matter
    assert not cell_service.lightning_jump([1.0, 6.0])  # too little history

    grid = Grid(event["lat"].values, event["lon"].values, 2.0)
    when = pd.Timestamp("2026-05-12T08:00")
    times = np.array([when - pd.Timedelta(minutes=m) for m in (1, 5, 15)], dtype="datetime64[ns]")
    lat = np.array([grid.lat[20, 20], grid.lat[21, 21], grid.lat[20, 20]])
    lon = np.array([grid.lon[20, 20], grid.lon[21, 21], grid.lon[20, 20]])
    counter = cell_service.FlashCounter((times, lat, lon), grid, step_minutes=10)
    assert counter.rate(when, 20, 20) == pytest.approx(0.2)  # two flashes in the last 10 min
    assert counter.rate(when, 45, 45) == 0.0  # too far away
