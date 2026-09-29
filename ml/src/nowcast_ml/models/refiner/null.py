from __future__ import annotations


class NullRefiner:
    """Default: no ensemble. ``Predictor.predict(n_members>0)`` raises a clear error."""

    name = "none"

    def sample(self, cond, deterministic, n_members, seed=None):
        return None
