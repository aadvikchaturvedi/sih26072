from nowcast_ml.models.refiner.base import Refiner
from nowcast_ml.models.refiner.null import NullRefiner


def build_refiner(name: str | None, **kwargs) -> Refiner:
    if name in (None, "none"):
        return NullRefiner()
    if name == "diffusion":
        from nowcast_ml.models.refiner.diffusion import DiffusionRefiner

        return DiffusionRefiner(**kwargs)
    raise ValueError(f"unknown refiner {name!r}")


__all__ = ["NullRefiner", "Refiner", "build_refiner"]
