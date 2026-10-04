"""Warnings in SQLite (stdlib). One connection per call keeps it thread-safe."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

from nowcast_backend.domain.models import Warning

_SCHEMA = """
CREATE TABLE IF NOT EXISTS warnings (
    id        TEXT PRIMARY KEY,
    domain    TEXT NOT NULL,
    status    TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    payload   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS warnings_domain_status ON warnings (domain, status, issued_at);
"""


class SqliteWarningRepository:
    def __init__(self, path: Path):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=10)

    def add(self, warnings: Iterable[Warning]) -> None:
        rows = [
            (w.id, w.domain, w.status, w.issued_at.isoformat(), w.model_dump_json())
            for w in warnings
        ]
        with closing(self._connect()) as con, con:
            con.executemany("INSERT OR REPLACE INTO warnings VALUES (?, ?, ?, ?, ?)", rows)

    def set_status(self, warning_id: str, status: str) -> None:
        w = self.get(warning_id)
        if w is not None:
            self.add([w.model_copy(update={"status": status})])

    def get(self, warning_id: str) -> Warning | None:
        with closing(self._connect()) as con:
            row = con.execute("SELECT payload FROM warnings WHERE id = ?", (warning_id,)).fetchone()
        return Warning.model_validate_json(row[0]) if row else None

    def find(
        self, domain: str | None = None, status: str | None = None, limit: int = 200
    ) -> list[Warning]:
        where, args = [], []
        if domain is not None:
            where.append("domain = ?")
            args.append(domain)
        if status is not None:
            where.append("status = ?")
            args.append(status)
        sql = "SELECT payload FROM warnings"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY issued_at DESC, id LIMIT ?"
        with closing(self._connect()) as con:
            rows = con.execute(sql, (*args, limit)).fetchall()
        return [Warning.model_validate_json(r[0]) for r in rows]
