import pytest

from nowcast_ml.data import channels as ch


def test_registry_has_all_contract_channels():
    expected = {
        "maxz",
        "cappi3km",
        "tir1_bt",
        "tir2_bt",
        "wv_bt",
        "mir_bt",
        "tir1_cooling",
        "tir1_minus_wv",
        "flash_density",
        "cape",
        "cin",
        "shear_0_6km",
        "pwat",
        "freezing_level",
    }
    assert set(ch.ALL_CHANNELS) == expected
    assert all(c.group in ch.GROUPS for c in ch.CHANNELS.values())
    assert set(ch.SEVIR_CHANNELS) <= set(ch.ALL_CHANNELS)


def test_groups():
    assert ch.channels_in_group("radar") == ["maxz", "cappi3km"]
    assert ch.group_of("flash_density") == "lightning"
    assert ch.group_of("cape") == "nwp"


def test_unknown_and_duplicate():
    assert ch.validate_channel_list(["maxz", "maxz", "nope"]) == [
        "duplicate channel 'maxz'",
        "unknown channel 'nope'",
    ]
    with pytest.raises(KeyError, match="unknown channel"):
        ch.get("nope")
