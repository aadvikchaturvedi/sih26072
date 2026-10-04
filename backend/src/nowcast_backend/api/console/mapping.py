"""Domain models -> console response models. Pure functions, no I/O."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nowcast_backend.api.console import schemas as out
from nowcast_backend.domain import models as dm
from nowcast_backend.domain.grid import haversine_km
from nowcast_backend.domain.timeutil import iso

#: Positional uncertainty of an extrapolated cell: a provisional rule of thumb
#: (km), to be replaced by ensemble spread once it is verified.
CONE_BASE_KM, CONE_KM_PER_MIN = 5.0, 0.3
STROKE_CELL_RADIUS_KM = 25.0
MAX_STROKES = 5000

#: nowcast-eval row -> (id, label, colour). The ids are the console's: ``F3F4`` is the
#: nowcast model with its lightning head, the row its headline numbers come from.
_SKILL_ROWS = (
    ("persistence", "persistence", "Persistence", "#6B7280"),
    ("extrapolation", "extrapolation", "Optical flow", "#38BDF8"),
    ("steps", "pysteps", "pySTEPS", "#8B5CF6"),
    ("model_full", "F3F4", "Nowcast model", "#22C55E"),
    ("model_satellite_only", "satellite_only", "Nowcast model (satellite only)", "#F59E0B"),
    ("model_ensemble", "ensemble", "Nowcast model (ensemble)", "#EF4444"),
)
HEADLINE_MODEL = "F3F4"


def _number(value) -> float | None:
    """JSON has no NaN: undefined scores become null."""
    return None if value is None or not math.isfinite(value) else float(value)


def health(meta: dm.ForecastMeta, model: dm.ModelDescription, replay: bool) -> out.HealthStatus:
    return out.HealthStatus(
        lastRunAt=iso(meta.t0),
        nextRunAt=iso(pd.Timestamp(meta.t0) + pd.Timedelta(minutes=model.step_minutes)),
        mode="replay" if replay else "live",
        modelMode=meta.mode,
        modelName=meta.model_name,
        modelVersion=meta.model_version,
        missingChannels=meta.missing_channels,
        inputs=[
            out.InputHealth(
                name=i.name,
                status=i.status,
                lastReceived=iso(i.last_received) if i.last_received else None,
                staleAgeSeconds=i.stale_age_seconds,
            )
            for i in meta.inputs
        ],
        hasEnsemble=model.has_ensemble,
        inferenceMsLast=round(meta.inference_ms, 1),
    )


def forecast(
    meta: dm.ForecastMeta, attrs: dict, layers_url: str, scale: int
) -> out.ForecastResponse:
    def layer(name: str, **query) -> str:
        params = "&".join(f"{k}={v}" for k, v in {**query, "scale": scale}.items())
        return f"{layers_url}/{name}.png?{params}"

    return out.ForecastResponse(
        reflectivityUrlByLead={str(m): layer("reflectivity", lead=m) for m in meta.lead_minutes},
        lightningProb30Url=layer("lightning_prob_30"),
        lightningProb60Url=layer("lightning_prob_60"),
        firstFlashUrl=layer("first_flash"),
        attrs=out.ForecastAttrs(
            model_name=meta.model_name,
            model_version=meta.model_version,
            t0=iso(meta.t0),
            mode=meta.mode,
            missing_channels=meta.missing_channels,
            inference_ms=meta.inference_ms,
            output_contract_version=str(attrs.get("output_contract_version", "1.0")),
            first_flash_threshold=float(attrs.get("first_flash_threshold", 0.5)),
        ),
    )


def _risk(point: dm.TrackPoint) -> tuple[str, str]:
    if point.lightning_prob is not None:
        p = point.lightning_prob
        return ("high" if p > 0.6 else "moderate" if p > 0.3 else "low"), "lightning_prob"
    dbz = point.max_dbz or 0.0
    return ("high" if dbz >= 50 else "moderate" if dbz >= 40 else "low"), "reflectivity_based"


def _lightning_trend(rates: list[float]) -> str:
    if len(rates) < 2:
        return "steady"
    before, now = rates[-2], rates[-1]
    change = now - before
    if change >= max(1.0, before):  # at least doubled, and by a flash a minute
        return "rapidly_increasing"
    return "increasing" if change > 0.2 else "decreasing" if change < -0.2 else "steady"


def _status(cell: dm.StormCell) -> str:
    """From the observed change over the last step where radar saw the cell."""
    if cell.basis == "forecast":  # no radar: only the forecast trend is known
        return {"growing": "growing", "decaying": "decaying"}.get(cell.trend, "mature")
    if not cell.history:
        return "initiating"
    growth = cell.growth_dbz_per_10min
    return "growing" if growth >= 1.0 else "decaying" if growth <= -1.0 else "mature"


def _data_quality(meta: dm.ForecastMeta, t_in: int) -> float:
    """Share of input frames observed, averaged with the share of input groups live at t0."""
    expected = [i for i in meta.inputs if i.last_received is not None or i.status != "missing"]
    live = sum(i.status == "live" for i in meta.inputs)
    groups = live / max(1, len(expected) or len(meta.inputs))
    return round((meta.observed_frames / t_in + min(1.0, groups)) / 2, 2)


def cell(c: dm.StormCell, meta: dm.ForecastMeta, t_in: int) -> out.Cell:
    now = dm.TrackPoint(
        time=c.time, lat=c.lat, lon=c.lon, max_dbz=c.max_dbz, flash_rate=c.flash_rate
    )
    observed = [*c.history, now]
    path = []
    for p in c.forecast_track:
        risk, basis = _risk(p)
        path.append(
            out.ForecastPathPoint(
                leadMin=int(round((p.time - meta.t0).total_seconds() / 60)),
                lat=p.lat,
                lon=p.lon,
                expectedDbz=round(p.max_dbz or 0.0, 1),
                risk=risk,
                riskBasis=basis,
            )
        )
    return out.Cell(
        id=c.id,
        isNewCell=c.is_new,
        status=_status(c),
        lat=c.lat,
        lon=c.lon,
        maxDbz=round(c.max_dbz, 1),
        growthDbzPer10Min=round(c.growth_dbz_per_10min, 1),
        lightningTrend=_lightning_trend([p.flash_rate or 0.0 for p in observed]),
        headingDeg=round(c.direction_deg or 0.0, 0),
        speedKmh=round(c.speed_kmh or 0.0, 0),
        lightningJumpFlag=c.lightning_jump,
        lightningJumpAt=iso(c.lightning_jump_at) if c.lightning_jump_at else None,
        track=[
            out.CellTrackPoint(
                time=iso(p.time),
                lat=p.lat,
                lon=p.lon,
                maxDbz=round(p.max_dbz or 0.0, 1),
                flashRate=round(p.flash_rate or 0.0, 2),
            )
            for p in observed
        ],
        forecastPath=path,
        uncertaintyCone=[
            {"leadMin": p.leadMin, "radiusKm": CONE_BASE_KM + CONE_KM_PER_MIN * p.leadMin}
            for p in path
        ],
        affectedDistricts=[
            {"districtId": a.region_id, "name": a.name, "etaMin": a.eta_min} for a in c.affected
        ],
        modelMode=meta.mode,
        dataQuality=_data_quality(meta, t_in),
        flashRateSeries=[
            {"time": iso(p.time), "rate": round(p.flash_rate or 0.0, 2)} for p in observed
        ],
        dbzSeries=[
            {"time": iso(p.time), "dbz": round(p.max_dbz or 0.0, 1), "isForecast": False}
            for p in observed
        ]
        + [
            {"time": iso(p.time), "dbz": round(p.max_dbz or 0.0, 1), "isForecast": True}
            for p in c.forecast_track
        ],
    )


def strokes(times, lats, lons, cells: list[dm.StormCell]) -> list[out.LightningStroke]:
    """Flash points, each tagged with the nearest storm cell if one is close."""
    if len(times) > MAX_STROKES:  # keep the newest
        keep = np.argsort(times)[-MAX_STROKES:]
        times, lats, lons = times[keep], lats[keep], lons[keep]
    owner: list[str | None] = [None] * len(times)
    if cells and len(times):
        dist = np.stack([haversine_km(lats, lons, c.lat, c.lon) for c in cells])
        nearest = dist.argmin(axis=0)
        close = dist.min(axis=0) <= STROKE_CELL_RADIUS_KM
        owner = [cells[j].id if ok else None for j, ok in zip(nearest, close, strict=True)]
    return [
        out.LightningStroke(
            id=f"L{pd.Timestamp(t).value}-{i}",
            time=iso(t),
            lat=float(la),
            lon=float(lo),
            cellId=cell_id,
        )
        for i, (t, la, lo, cell_id) in enumerate(zip(times, lats, lons, owner, strict=True))
    ]


def district(d: dm.DistrictForecast) -> out.DistrictForecast:
    return out.DistrictForecast(
        districtId=d.district_id,
        name=d.name,
        lightningProb30=d.lightning_prob_30,
        lightningProb60=d.lightning_prob_60,
        maxDbz=d.max_dbz,
        areaFractionAboveThreshold=d.area_fraction_above_threshold,
        warningLevel=d.level,
    )


def warning(w: dm.DistrictWarning) -> out.Warning:
    return out.Warning(
        id=w.id,
        level=w.level,
        districtId=w.district_id,
        districtName=w.district_name,
        causeText=w.cause_text,
        causeCell=w.cause_cell,
        ruleTriggered=w.rule,
        triggerValue=w.trigger_value,
        triggerThreshold=w.trigger_threshold,
        issuedAt=iso(w.issued_at),
        validUntil=iso(w.valid_until),
        previousLevel=w.previous_level,
    )


def _mean(values) -> float | None:
    finite = [v for v in values or [] if v is not None and math.isfinite(v)]
    return sum(finite) / len(finite) if finite else None


def skill(metrics: dict) -> out.SkillReport:
    """``nowcast-eval`` metrics -> the skill page's report.

    POD / FAR are for reflectivity >= 35 dBZ averaged over all leads; Brier skill,
    ROC and reliability are for lightning within 30 minutes.
    """
    meta = metrics.get("meta", {})
    leads = meta.get("leads_min", [])
    models = []
    for key, model_id, label, colour in _SKILL_ROWS:
        row = metrics.get("forecasters", {}).get(key)
        if row is None:
            continue
        refl, ltg, ff = row["reflectivity"], row["lightning"].get("30", {}), row["first_flash"]
        rel = ltg.get("reliability", {})
        models.append(
            out.ModelSkill(
                modelId=model_id,
                label=label,
                color=colour,
                csiByThreshold={
                    str(int(float(thr))): [
                        {"leadMin": lead, "csi": _number(v) or 0.0}
                        for lead, v in zip(leads, values, strict=False)
                    ]
                    for thr, values in refl["csi"].items()
                },
                reliabilityDiagram=[
                    {"forecastProb": f, "observedFreq": o, "count": n}
                    for f, o, n in zip(
                        rel.get("mean_forecast", []),
                        rel.get("observed_freq", []),
                        rel.get("count", []),
                        strict=False,
                    )
                    if n and _number(f) is not None and _number(o) is not None
                ],
                brierSkillScore=_number(ltg.get("bss_climatology")),
                rocAuc=_number(ltg.get("roc_auc")),
                pod=_mean(refl["pod"].get("35")),
                far=_mean(refl["far"].get("35")),
                firstFlashHitRate=_number(ff.get("hit_rate")),
                firstFlashFAR=_number(ff.get("false_alarm_ratio")),
                medianFirstFlashLeadMin=_number(ff.get("median_lead_min")),
            )
        )
    headline = next(
        (m.medianFirstFlashLeadMin for m in models if m.modelId == HEADLINE_MODEL), None
    )
    synthetic = bool(meta.get("synthetic"))
    label = f"{meta.get('n_events', '?')} {meta.get('split', '')} events, {meta.get('n_samples', '?')} forecasts"
    if synthetic:
        label = f"SYNTHETIC DATA ({label}): not indicative of real skill"
    return out.SkillReport(
        reportAt=meta.get("report_at", ""),
        medianFirstFlashLeadMin=headline,
        models=models,
        eventLabel=label,
        eventStartAt=None,
        eventEndAt=None,
        synthetic=synthetic,
    )
