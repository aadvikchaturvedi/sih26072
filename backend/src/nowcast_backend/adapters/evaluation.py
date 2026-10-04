"""Evaluation results written by ``nowcast-eval`` (``metrics.json``)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


class JsonEvaluation:
    def __init__(self, path: Path | None):
        self._path = path

    def metrics(self) -> dict | None:
        if self._path is None or not self._path.exists():
            return None
        data = json.loads(self._path.read_text())
        data.setdefault("meta", {})["report_at"] = datetime.fromtimestamp(
            self._path.stat().st_mtime, UTC
        ).isoformat()
        return data
