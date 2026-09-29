"""Eulerian persistence: the last observation is the forecast for every lead."""

from __future__ import annotations

import numpy as np

from nowcast_ml.baselines.base import ForecastInput, last_reflectivity
from nowcast_ml.inference.schema import ForecastArrays


class PersistenceForecaster:
    name = "persistence"

    def forecast(
        self, inp: ForecastInput, n_leads: int = 12, ltg_leads_min=(30, 60)
    ) -> ForecastArrays:
        z = last_reflectivity(inp)
        refl = np.repeat(z[None], n_leads, axis=0)
        ltg = np.repeat(inp.past_ltg[None].astype(np.float32), len(ltg_leads_min), axis=0)
        return ForecastArrays(reflectivity=refl, lightning=ltg)
