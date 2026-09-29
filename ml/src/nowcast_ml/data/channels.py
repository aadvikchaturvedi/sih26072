"""Canonical channel registry.

Every channel the system knows about is declared here once, with its unit, its
group (used by the ``missing`` mask and modality dropout) and a physically
plausible range used by the input validator. Models store the ordered subset
they were trained on in ``channels.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Group = Literal["radar", "satellite", "lightning", "nwp"]
GROUPS: tuple[Group, ...] = ("radar", "satellite", "lightning", "nwp")


@dataclass(frozen=True)
class Channel:
    name: str
    unit: str
    group: Group
    valid_min: float
    valid_max: float
    description: str
    # Value used for "no signal" (e.g. no echo). Not a fill for missing data.
    background: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


_CHANNELS: tuple[Channel, ...] = (
    Channel("maxz", "dBZ", "radar", -35.0, 85.0, "Column-maximum reflectivity (MAX-Z)", 0.0),
    Channel("cappi3km", "dBZ", "radar", -35.0, 85.0, "Reflectivity CAPPI at 3 km", 0.0),
    Channel(
        "tir1_bt",
        "K",
        "satellite",
        150.0,
        340.0,
        "INSAT TIR1 (10.8 um) brightness temperature",
        290.0,
    ),
    Channel(
        "tir2_bt",
        "K",
        "satellite",
        150.0,
        340.0,
        "INSAT TIR2 (12.0 um) brightness temperature",
        288.0,
    ),
    Channel(
        "wv_bt", "K", "satellite", 150.0, 300.0, "INSAT WV (6.8 um) brightness temperature", 240.0
    ),
    Channel(
        "mir_bt", "K", "satellite", 150.0, 360.0, "INSAT MIR (3.9 um) brightness temperature", 295.0
    ),
    Channel(
        "tir1_cooling",
        "K/10min",
        "satellite",
        -100.0,
        100.0,
        "TIR1 BT change over previous 10 min",
        0.0,
    ),
    Channel("tir1_minus_wv", "K", "satellite", -40.0, 100.0, "TIR1 minus WV BT difference", 50.0),
    Channel(
        "flash_density",
        "flashes/km2/10min",
        "lightning",
        0.0,
        100.0,
        "Lightning flash density",
        0.0,
    ),
    Channel("cape", "J/kg", "nwp", 0.0, 10000.0, "Convective available potential energy", 0.0),
    Channel("cin", "J/kg", "nwp", -1500.0, 0.0, "Convective inhibition (<= 0)", 0.0),
    Channel("shear_0_6km", "m/s", "nwp", 0.0, 80.0, "0-6 km bulk wind shear", 0.0),
    Channel("pwat", "mm", "nwp", 0.0, 100.0, "Precipitable water", 0.0),
    Channel("freezing_level", "m", "nwp", 0.0, 7000.0, "Freezing level height AGL", 4500.0),
)

CHANNELS: dict[str, Channel] = {c.name: c for c in _CHANNELS}
ALL_CHANNELS: tuple[str, ...] = tuple(c.name for c in _CHANNELS)

#: Channel that the reflectivity head predicts.
TARGET_CHANNEL = "maxz"
#: Channel used for lightning labels when no flash point table is available.
LIGHTNING_CHANNEL = "flash_density"

#: SEVIR provides only this subset (see ``data/sevir.py`` for the mapping).
SEVIR_CHANNELS: tuple[str, ...] = (
    "maxz",
    "tir1_bt",
    "wv_bt",
    "tir1_cooling",
    "tir1_minus_wv",
    "flash_density",
)


def get(name: str) -> Channel:
    try:
        return CHANNELS[name]
    except KeyError as e:
        raise KeyError(f"unknown channel {name!r}; known: {', '.join(ALL_CHANNELS)}") from e


def group_of(name: str) -> Group:
    return get(name).group


def channels_in_group(group: Group, subset: tuple[str, ...] | list[str] | None = None) -> list[str]:
    names = ALL_CHANNELS if subset is None else subset
    return [n for n in names if get(n).group == group]


def validate_channel_list(names: list[str] | tuple[str, ...]) -> list[str]:
    """Return problems with an ordered channel list (unknown / duplicate names)."""
    problems = []
    seen = set()
    for n in names:
        if n not in CHANNELS:
            problems.append(f"unknown channel {n!r}")
        if n in seen:
            problems.append(f"duplicate channel {n!r}")
        seen.add(n)
    return problems
