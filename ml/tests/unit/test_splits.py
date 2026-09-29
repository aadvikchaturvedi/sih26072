import pytest

from nowcast_ml.config import SplitConfig
from nowcast_ml.data.splits import SplitLeakError, assert_no_leakage, split_events


def test_disjoint_complete_deterministic():
    ids = [f"ev{i:03d}" for i in range(40)]
    s = split_events(ids, SplitConfig(seed=3))
    assert sorted(s["train"] + s["val"] + s["test"]) == ids
    assert len(s["val"]) == 6 and len(s["test"]) == 6
    assert s == split_events(list(reversed(ids)), SplitConfig(seed=3))
    assert s != split_events(ids, SplitConfig(seed=4))


def test_small_sets_nonempty():
    s = split_events(["a", "b", "c"])
    assert all(len(v) == 1 for v in s.values())


def test_explicit_and_leak():
    s = split_events(["a", "b", "c", "d"], SplitConfig(explicit={"val": ["a"], "test": ["b"]}))
    assert s == {"train": ["c", "d"], "val": ["a"], "test": ["b"]}
    with pytest.raises(SplitLeakError):
        assert_no_leakage({"train": ["a"], "test": ["a"]})
    with pytest.raises(ValueError, match="unknown"):
        split_events(["a"], SplitConfig(explicit={"val": ["zzz"]}))
