"""Common Alerting Protocol 1.2 documents (and an Atom index) for warnings."""

from __future__ import annotations

from datetime import datetime
from xml.etree import ElementTree as ET

from nowcast_backend.domain.models import DistrictWarning, Region, Warning
from nowcast_backend.services.geo import rings
from nowcast_backend.settings import CapSettings

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
ATOM_NS = "http://www.w3.org/2005/Atom"

_SEVERITY = {"yellow": "Moderate", "orange": "Severe", "red": "Extreme"}
_RESPONSE = {"yellow": "Monitor", "orange": "Prepare", "red": "Shelter"}
_EVENT = {"lightning": "Lightning", "thunderstorm": "Thunderstorm"}
#: What people should do, by colour code.
INSTRUCTIONS = {
    "yellow": "Be aware. Keep watching for updates and plan to move indoors if the sky darkens.",
    "orange": "Be prepared. Finish or postpone outdoor work; stay away from open fields, "
    "tall trees and water.",
    "red": "Take action now. Go inside a solid building or a hard-top vehicle and stay there; "
    "unplug sensitive equipment and keep away from windows.",
}


def _time(t: datetime) -> str:
    """CAP requires a numeric offset ('Z' is not allowed)."""
    return t.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _add(parent: ET.Element, tag: str, text: str | None = None) -> ET.Element:
    el = ET.SubElement(parent, f"{{{CAP_NS}}}{tag}")
    if text is not None:
        el.text = text
    return el


def _certainty(w: Warning) -> str:
    if w.hazard == "lightning":
        return "Likely" if w.peak_value >= 0.5 else "Possible"
    return "Likely"


def cap_alert(w: Warning, cfg: CapSettings, previous: Warning | None = None) -> str:
    """CAP XML for a warning. A cancelled warning yields a ``Cancel`` message that
    references the original; an update references the warning it supersedes."""
    ET.register_namespace("", CAP_NS)
    cancel = w.status == "cancelled"
    alert = ET.Element(f"{{{CAP_NS}}}alert")
    _add(alert, "identifier", f"{w.id}-cancel" if cancel else w.id)
    _add(alert, "sender", cfg.sender)
    _add(alert, "sent", _time(w.issued_at))
    _add(alert, "status", cfg.status)
    _add(alert, "msgType", "Cancel" if cancel else w.msg_type)
    _add(alert, "scope", "Public")
    referenced = w if cancel else previous
    if referenced is not None:
        _add(alert, "references", f"{cfg.sender},{referenced.id},{_time(referenced.issued_at)}")

    info = _add(alert, "info")
    _add(info, "language", cfg.language)
    _add(info, "category", "Met")
    _add(info, "event", _EVENT[w.hazard])
    _add(info, "responseType", "AllClear" if cancel else _RESPONSE[w.level])
    minutes_to_onset = (w.onset - w.t0).total_seconds() / 60
    _add(info, "urgency", "Immediate" if minutes_to_onset <= 30 else "Expected")
    _add(info, "severity", _SEVERITY[w.level])
    _add(info, "certainty", _certainty(w))
    _add(info, "onset", _time(w.onset))
    _add(info, "expires", _time(w.expires))
    _add(info, "senderName", cfg.sender_name)
    _add(info, "headline", f"Cancelled: {w.headline}" if cancel else w.headline)
    _add(info, "description", w.description)
    _add(info, "instruction", w.instruction)
    for name, value in (
        ("colourCode", w.level),
        (f"peak_{w.peak_unit}", f"{w.peak_value:g}"),
        ("forecastMode", w.mode),
        ("modelVersion", w.model_version),
    ):
        p = _add(info, "parameter")
        _add(p, "valueName", name)
        _add(p, "value", value)
    area = _add(info, "area")
    _add(area, "areaDesc", f"Area around {w.lat:.2f}N {w.lon:.2f}E ({w.area_km2:.0f} km2)")
    for ring in rings(w.polygon):
        _add(area, "polygon", " ".join(f"{lat},{lon}" for lon, lat in ring))
    return ET.tostring(alert, encoding="unicode", xml_declaration=True)


NOTE = "Warning thresholds are draft placeholders, not IMD operational criteria."
_DISTRICT_SEVERITY = {**_SEVERITY, "green": "Minor"}
_DISTRICT_RESPONSE = {**_RESPONSE, "green": "AllClear"}


def _decimate(ring: list, max_points: int = 60) -> list:
    """Thin a long boundary ring; CAP consumers choke on thousands of vertices."""
    step = max(1, len(ring) // max_points)
    thin = ring[:-1:step]
    return [*thin, thin[0]]


def district_alert(
    w: DistrictWarning, region: Region | None, previous: DistrictWarning | None, cfg: CapSettings
) -> tuple[str, dict]:
    """A district warning as CAP 1.2 XML and as the same content in JSON."""
    lifted = w.level == "green"
    headline = (
        f"Thunderstorm warning lifted: {w.district_name}"
        if lifted
        else f"{w.level.upper()} thunderstorm warning: {w.district_name}"
    )
    doc = {
        "identifier": w.id,
        "sender": cfg.sender,
        "sent": _time(w.issued_at),
        "status": cfg.status,
        "msgType": "Cancel" if lifted and previous else "Update" if previous else "Alert",
        "scope": "Public",
        "note": NOTE,
        "info": {
            "language": cfg.language,
            "category": "Met",
            "event": "Thunderstorm/Lightning Nowcast",
            "responseType": _DISTRICT_RESPONSE[w.level],
            "urgency": "Immediate" if w.level == "red" else "Expected",
            "severity": _DISTRICT_SEVERITY[w.level],
            "certainty": "Likely" if w.level in ("orange", "red") else "Possible",
            "onset": _time(w.issued_at),
            "expires": _time(w.valid_until),
            "senderName": cfg.sender_name,
            "headline": headline,
            "description": f"{w.cause_text}. Rule: {w.rule}.",
            "instruction": "No action needed." if lifted else INSTRUCTIONS[w.level],
            "area": {"areaDesc": f"{w.district_name} district"},
        },
    }
    if previous is not None:
        doc["references"] = f"{cfg.sender},{previous.id},{_time(previous.issued_at)}"

    ET.register_namespace("", CAP_NS)
    alert = ET.Element(f"{{{CAP_NS}}}alert")
    for key in ("identifier", "sender", "sent", "status", "msgType", "scope", "note", "references"):
        if key in doc:
            _add(alert, key, doc[key])
    info = _add(alert, "info")
    for key, value in doc["info"].items():
        if key != "area":
            _add(info, key, value)
    area = _add(info, "area")
    _add(area, "areaDesc", doc["info"]["area"]["areaDesc"])
    if region is not None:
        for ring in rings(region.geometry):
            _add(area, "polygon", " ".join(f"{lat:.4f},{lon:.4f}" for lon, lat in _decimate(ring)))
    return ET.tostring(alert, encoding="unicode", xml_declaration=True), doc


def atom_feed(warnings: list[Warning], cfg: CapSettings, base_url: str, updated: datetime) -> str:
    """Atom index of active warnings; each entry links to its CAP document."""
    ET.register_namespace("", ATOM_NS)

    def add(parent, tag, text=None, **attrs):
        el = ET.SubElement(parent, f"{{{ATOM_NS}}}{tag}", attrs)
        if text is not None:
            el.text = text
        return el

    base = base_url.rstrip("/")
    feed = ET.Element(f"{{{ATOM_NS}}}feed")
    add(feed, "id", f"{base}/api/v1/cap/feed.atom")
    add(feed, "title", f"{cfg.sender_name}: active warnings")
    add(feed, "updated", updated.strftime("%Y-%m-%dT%H:%M:%SZ"))
    add(add(feed, "author"), "name", cfg.sender_name)
    for w in warnings:
        url = f"{base}/api/v1/warnings/{w.id}/cap"
        entry = add(feed, "entry")
        add(entry, "id", url)
        add(entry, "title", w.headline)
        add(entry, "updated", w.issued_at.strftime("%Y-%m-%dT%H:%M:%SZ"))
        add(entry, "summary", w.description)
        add(entry, "link", href=url, type="application/cap+xml")
    return ET.tostring(feed, encoding="unicode", xml_declaration=True)
