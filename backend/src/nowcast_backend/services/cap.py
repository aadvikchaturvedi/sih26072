"""Common Alerting Protocol 1.2 documents (and an Atom index) for warnings."""

from __future__ import annotations

from datetime import datetime
from xml.etree import ElementTree as ET

from nowcast_backend.domain.models import Warning
from nowcast_backend.services.geo import rings
from nowcast_backend.settings import CapSettings

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
ATOM_NS = "http://www.w3.org/2005/Atom"

_SEVERITY = {"yellow": "Moderate", "orange": "Severe", "red": "Extreme"}
_RESPONSE = {"yellow": "Monitor", "orange": "Prepare", "red": "Shelter"}
_EVENT = {"lightning": "Lightning", "thunderstorm": "Thunderstorm"}


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
