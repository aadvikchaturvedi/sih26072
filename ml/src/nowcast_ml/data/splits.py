"""Train/val/test splits BY EVENT (never by frame).

Adjacent frames of one storm are highly correlated; splitting by frame leaks
the test set into training. Assignment is a deterministic function of the
event id and seed, so adding new events does not reshuffle old ones much.
"""

from __future__ import annotations

import hashlib

from nowcast_ml.config import SplitConfig

SPLITS = ("train", "val", "test")


class SplitLeakError(ValueError):
    pass


def _key(event_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{event_id}".encode()).hexdigest()


def split_events(event_ids: list[str], cfg: SplitConfig | None = None) -> dict[str, list[str]]:
    cfg = cfg or SplitConfig()
    ids = sorted(set(event_ids))
    if len(ids) != len(event_ids):
        raise ValueError("duplicate event ids")
    if cfg.explicit is not None:
        out = {s: sorted(cfg.explicit.get(s, [])) for s in SPLITS}
        assigned = [e for s in SPLITS for e in out[s]]
        unknown = sorted(set(assigned) - set(ids))
        if unknown:
            raise ValueError(f"explicit split names unknown events: {unknown[:5]}")
        assert_no_leakage(out)
        unassigned = sorted(set(ids) - set(assigned))
        out["train"] = sorted(out["train"] + unassigned)
        return out

    ordered = sorted(ids, key=lambda e: _key(e, cfg.seed))
    n = len(ordered)
    n_val = round(n * cfg.val)
    n_test = round(n * cfg.test)
    if n >= 3:  # keep every split non-empty when there are enough events
        n_val = max(n_val, 1 if cfg.val > 0 else 0)
        n_test = max(n_test, 1 if cfg.test > 0 else 0)
    n_train = n - n_val - n_test
    out = {
        "train": sorted(ordered[:n_train]),
        "val": sorted(ordered[n_train : n_train + n_val]),
        "test": sorted(ordered[n_train + n_val :]),
    }
    assert_no_leakage(out)
    return out


def assert_no_leakage(splits: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for s, ids in splits.items():
        for e in ids:
            if e in seen:
                raise SplitLeakError(f"event {e!r} is in both {seen[e]!r} and {s!r}")
            seen[e] = s
