"""Backend service for the thunderstorm & lightning nowcasting system.

Layers (each depends only on the ones above it):

* ``domain``    plain data models and typed errors
* ``ports``     interfaces the services need (engine, stores, repository, notifier, source)
* ``services``  nowcast orchestration, storm cells, warnings, CAP, rendering
* ``adapters``  implementations of the ports (nowcast_ml, Zarr, SQLite, webhooks, ...)
* ``api``       FastAPI routes; ``container`` wires everything from ``Settings``
"""

__version__ = "0.1.0"
