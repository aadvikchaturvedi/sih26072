"""The HTTP API end to end, against the fake engine."""

from __future__ import annotations

import io
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from nowcast_backend.adapters.zarr_zip import encode
from nowcast_backend.api.app import create_app
from nowcast_backend.domain.timeutil import stamp
from nowcast_backend.services.cap import CAP_NS
from tests.conftest import frames, t

API = "/api/v1"


def push(client, event, start, stop, domain="kolkata", **params):
    return client.post(
        f"{API}/domains/{domain}/observations",
        content=encode(frames(event, start, stop)),
        params=params,
    )


def test_health_and_model(client):
    assert client.get(f"{API}/health").json()["status"] == "ok"
    model = client.get(f"{API}/model").json()
    assert model["t_in"] == 7 and model["lead_minutes"][-1] == 120
    assert "reflectivity" in client.get(f"{API}/styles").json()


def test_ingest_runs_forecast(client, event, engine):
    r = push(client, event, 0, 12)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_frames"] == 12 and len(body["accepted_times"]) == 12
    fc = body["forecast"]
    assert fc["source"] == "model" and fc["mode"] == "full"
    assert fc["t0"] == "2026-05-12T07:50:00Z" and fc["shape"] == [48, 48]

    domain = client.get(f"{API}/domains/kolkata").json()
    assert domain["latest_time"] == fc["t0"] and len(domain["times"]) == 12
    assert [d["domain"] for d in client.get(f"{API}/domains").json()] == ["kolkata"]

    # re-sending the same frames stores nothing new and does not re-run the model
    calls = engine.calls
    assert push(client, event, 6, 12).json()["forecast"] is None
    assert engine.calls == calls


def test_ingest_waits_for_enough_history(client, event):
    body = push(client, event, 0, 2).json()
    assert body["forecast"] is None and body["n_frames"] == 2
    assert client.get(f"{API}/domains/kolkata/forecasts/latest").status_code == 404
    assert push(client, event, 2, 5).json()["forecast"] is not None


def test_gap_frames_are_filled(client, event):
    push(client, event, 0, 5, run="false")
    body = push(client, event, 7, 9).json()  # frames 5 and 6 never arrived
    assert body["n_frames"] == 7
    assert body["forecast"]["observed_frames"] == 5


def test_rejects_bad_input(client, event):
    r = client.post(f"{API}/domains/kolkata/observations", content=b"not a zip")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_observations"

    bad = frames(event, 0, 8).copy(deep=True)
    bad["x"].values[:, 0] += 500.0  # dBZ far outside the plausible range
    r = client.post(f"{API}/domains/kolkata/observations", content=encode(bad))
    assert r.status_code == 422 and any("maxz" in p for p in r.json()["error"]["problems"])

    assert push(client, event, 0, 8, domain="Bad Name!").status_code == 400

    push(client, event, 0, 8)
    other_grid = frames(event, 8, 10).isel(y=slice(0, 32), x=slice(0, 32))
    r = client.post(f"{API}/domains/kolkata/observations", content=encode(other_grid))
    assert r.status_code == 409


def test_forecast_products(client, event):
    push(client, event, 0, 14)
    base = f"{API}/domains/kolkata/forecasts"
    meta = client.get(f"{base}/latest").json()
    assert meta["layers"] == [
        "reflectivity",
        "lightning_prob_30",
        "lightning_prob_60",
        "first_flash",
    ]
    by_time = client.get(f"{base}/{stamp(t(event, 13))}").json()
    assert by_time["t0"] == meta["t0"]
    # every pushed time with enough history got its forecast (frames 3..13), newest first
    stored = [m["t0"] for m in client.get(base).json()]
    assert len(stored) == 11 and stored[0] == meta["t0"] and stored == sorted(stored, reverse=True)

    png = client.get(f"{base}/latest/layers/reflectivity.png", params={"lead": 30, "scale": 2})
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    image = Image.open(io.BytesIO(png.content))
    assert image.size == (96, 96) and image.mode == "RGBA"
    assert np.asarray(image)[..., 3].max() >= 200  # storms are drawn (nearly) opaque
    assert client.get(f"{base}/latest/layers/lightning_prob_30.png").status_code == 200
    assert client.get(f"{base}/latest/layers/reflectivity.png").status_code == 400  # no lead
    assert client.get(f"{base}/latest/layers/nope.png").status_code == 404

    grid = client.get(f"{base}/latest/layers/reflectivity", params={"lead": 10, "stride": 4}).json()
    assert grid["shape"] == [12, 12] and len(grid["values"]) == 12

    status = client.get(f"{API}/domains/kolkata").json()
    b = status["bounds"]
    point = client.get(
        f"{base}/latest/point",
        params={"lat": (b["south"] + b["north"]) / 2, "lon": (b["west"] + b["east"]) / 2},
    ).json()
    assert len(point["reflectivity"]) == 12 and set(point["lightning_prob"]) == {"30", "60"}
    outside = client.get(f"{base}/latest/point", params={"lat": 0, "lon": 0})
    assert outside.status_code == 400

    obs = client.get(f"{API}/domains/kolkata/observations/latest/maxz.png")
    assert obs.status_code == 200
    assert client.get(f"{API}/domains/kolkata/observations/latest/nope.png").status_code == 404


def test_run_baseline_and_earlier_time(client, event, engine):
    push(client, event, 0, 14)
    base = f"{API}/domains/kolkata/forecasts"
    r = client.post(base, json={"source": "persistence", "t0": str(t(event, 10))})
    assert r.status_code == 200, r.text
    assert r.json()["model_name"] == "baseline_persistence"
    assert client.get(f"{base}/latest", params={"source": "persistence"}).status_code == 200
    calls = engine.calls
    client.post(base, json={"source": "persistence", "t0": str(t(event, 10))})
    assert engine.calls == calls  # served from the store
    client.post(base, json={"source": "persistence", "t0": str(t(event, 10)), "force": True})
    assert engine.calls == calls + 1
    assert client.post(base, json={"n_members": 3, "force": True}).status_code == 400
    assert client.post(base, json={"t0": "2020-01-01T00:00Z"}).status_code == 404
    assert client.post(base, json={"source": "magic"}).status_code == 400


def test_cells(client, event):
    push(client, event, 0, 14)
    base = f"{API}/domains/kolkata/forecasts/latest/cells"
    geo = client.get(base).json()
    assert geo["type"] == "FeatureCollection" and geo["features"]
    cell = geo["features"][0]
    assert cell["geometry"]["type"] in ("Polygon", "MultiPolygon")
    props = cell["properties"]
    assert props["basis"] == "observed" and props["max_dbz"] >= 30
    assert props["speed_kmh"] is not None and props["history"] and props["forecast_track"]
    ids = {f["id"] for f in geo["features"]}

    push(client, event, 14, 15)  # next frame: the same storms keep their ids
    after = {c["id"] for c in client.get(base, params={"format": "json"}).json()}
    assert ids & after


def test_warnings_and_cap(client, event):
    push(client, event, 0, 14)
    geo = client.get(f"{API}/warnings", params={"domain": "kolkata"}).json()
    assert geo["features"], "the synthetic storms should trigger warnings"
    first = geo["features"][0]["properties"]
    assert first["status"] == "active" and first["level"] in ("yellow", "orange", "red")
    assert first["msg_type"] == "Alert"

    xml = client.get(f"{API}/warnings/{first['id']}/cap")
    assert xml.headers["content-type"].startswith("application/cap+xml")
    alert = ET.fromstring(xml.text)
    ns = {"cap": CAP_NS}
    assert alert.findtext("cap:identifier", namespaces=ns) == first["id"]
    assert alert.findtext("cap:status", namespaces=ns) == "Test"
    polygon = alert.findtext("cap:info/cap:area/cap:polygon", namespaces=ns).split()
    assert len(polygon) >= 4 and polygon[0] == polygon[-1]

    push(client, event, 14, 15)  # the next run updates the same storms' warnings
    current = client.get(f"{API}/warnings", params={"format": "json"}).json()
    assert current and all(w["t0"] != first["t0"] for w in current)
    updates = [w for w in current if w["msg_type"] == "Update"]
    assert updates
    cap_update = ET.fromstring(client.get(f"{API}/warnings/{updates[0]['id']}/cap").text)
    assert updates[0]["supersedes"] in cap_update.findtext("cap:references", namespaces=ns)
    old = client.get(f"{API}/warnings", params={"status": "superseded", "format": "json"}).json()
    assert first["id"] in {w["id"] for w in old}

    feed = client.get(f"{API}/cap/feed.atom")
    assert feed.status_code == 200 and feed.text.count("<entry>") == len(current)
    assert client.get(f"{API}/warnings/nope").status_code == 404


def test_satellite_only_mode(client, event):
    blind = frames(event, 0, 12).copy(deep=True)
    blind["missing"].loc[{"group": "radar"}] = 1
    r = client.post(f"{API}/domains/kolkata/observations", content=encode(blind))
    assert r.json()["forecast"]["mode"] == "satellite_only"
    assert client.get(f"{API}/domains/kolkata/observations/latest/maxz.png").status_code == 200


def test_api_key_guards_writes(settings, engine, event):
    from fastapi.testclient import TestClient

    secured = settings.model_copy(update={"api_key": "s3cret"})
    with TestClient(create_app(secured, engine)) as client:
        assert push(client, event, 0, 8).status_code == 401
        r = client.post(
            f"{API}/domains/kolkata/observations",
            content=encode(frames(event, 0, 8)),
            headers={"X-API-Key": "s3cret"},
        )
        assert r.status_code == 200
        assert client.get(f"{API}/domains").status_code == 200  # reads stay open


def test_state_survives_restart(settings, engine, event):
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings, engine)) as client:
        push(client, event, 0, 14)
        warnings = client.get(f"{API}/warnings", params={"format": "json"}).json()
    with TestClient(create_app(settings, engine)) as client:
        assert len(client.get(f"{API}/domains/kolkata").json()["times"]) == 14
        assert client.get(f"{API}/domains/kolkata/forecasts/latest").status_code == 200
        assert client.get(f"{API}/warnings", params={"format": "json"}).json() == warnings
        assert client.get(f"{API}/domains/kolkata/forecasts/latest/cells").json()["features"]
