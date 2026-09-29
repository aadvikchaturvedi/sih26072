"""Package logger."""

from __future__ import annotations

import logging

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "nowcast_ml") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger("nowcast_ml").handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FMT))
        root = logging.getLogger("nowcast_ml")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logger
