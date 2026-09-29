from nowcast_ml.models.refiner.base import Refiner
from nowcast_ml.models.refiner.null import NullRefiner


def build_refiner(name: str | None, t_out: int = 12, cond_channels: int = 0, cfg=None) -> Refiner:
    """``None``/``"none"`` -> :class:`NullRefiner`; ``"diffusion"`` -> :class:`DiffusionRefiner`."""
    if name in (None, "none"):
        return NullRefiner()
    if name == "diffusion":
        from nowcast_ml.models.refiner.diffusion import DiffusionRefiner

        return DiffusionRefiner(t_out, cond_channels, cfg)
    raise ValueError(f"unknown refiner {name!r}")


__all__ = ["NullRefiner", "Refiner", "build_refiner"]
