"""SQLite write-ahead log for FilmSense chunks.

The atomscale ``TimeseriesStreamer.push`` is fire-and-forget: HTTP failures land
in debug logs and never surface to Python (see
``src/atomscale/streaming/src/timeseries.rs:583`` — ``spawn_upload`` drops the
JoinHandle). If we relied on it alone, a 503 mid-run would silently drop a
chunk.

The WAL records every chunk *before* it is dispatched and tracks an
``acknowledged`` flag. On startup, on a transient error, or on operator demand
the runner can replay un-acknowledged chunks via the streamer's blocking
``run()`` mode (which *does* surface upload errors).

This is intentionally minimal — one SQLite file per adapter instance, single
writer thread, std-library only.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    data_id         TEXT    NOT NULL,
    chunk_index     INTEGER NOT NULL,
    channel_name    TEXT    NOT NULL,
    units           TEXT,
    timestamps_json TEXT    NOT NULL,
    values_json     TEXT    NOT NULL,
    created_at      REAL    NOT NULL,
    acknowledged    INTEGER NOT NULL DEFAULT 0,
    UNIQUE(data_id, chunk_index, channel_name)
);

CREATE INDEX IF NOT EXISTS idx_chunks_unacked
    ON chunks(data_id, acknowledged)
    WHERE acknowledged = 0;
"""


@dataclass(frozen=True)
class WalChunk:
    id: int
    data_id: str
    chunk_index: int
    channel_name: str
    units: str | None
    timestamps: list[float]
    values: list[float]


class ChunkWal:
    """Thread-safe SQLite write-ahead log for streamer chunks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.path),
            isolation_level=None,  # autocommit; explicit transactions when needed
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> ChunkWal:
        return self

    def __exit__(self, *_exc):
        self.close()

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def record(
        self,
        data_id: str,
        chunk_index: int,
        channel_name: str,
        timestamps: list[float],
        values: list[float],
        units: str | None = None,
    ) -> int:
        """Persist a chunk to the WAL. Returns the row id."""
        if len(timestamps) != len(values):
            msg = "timestamps and values must have the same length"
            raise ValueError(msg)
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                  (data_id, chunk_index, channel_name, units,
                   timestamps_json, values_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data_id,
                    chunk_index,
                    channel_name,
                    units,
                    json.dumps(timestamps),
                    json.dumps(values),
                    time.time(),
                ),
            )
            row_id = cur.lastrowid
        if row_id is None:
            msg = "WAL insert returned no row id"
            raise RuntimeError(msg)
        return int(row_id)

    def acknowledge(self, row_id: int) -> None:
        """Mark a chunk as successfully delivered."""
        with self._lock:
            self._conn.execute(
                "UPDATE chunks SET acknowledged = 1 WHERE id = ?", (row_id,)
            )

    def acknowledge_all(self, data_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE chunks SET acknowledged = 1 WHERE data_id = ?", (data_id,)
            )

    def pending(self, data_id: str | None = None) -> list[WalChunk]:
        """Return un-acknowledged chunks, ordered by (data_id, chunk_index)."""
        with self._lock:
            if data_id is None:
                cur = self._conn.execute(
                    """
                    SELECT id, data_id, chunk_index, channel_name, units,
                           timestamps_json, values_json
                    FROM chunks
                    WHERE acknowledged = 0
                    ORDER BY data_id, chunk_index, id
                    """
                )
            else:
                cur = self._conn.execute(
                    """
                    SELECT id, data_id, chunk_index, channel_name, units,
                           timestamps_json, values_json
                    FROM chunks
                    WHERE acknowledged = 0 AND data_id = ?
                    ORDER BY chunk_index, id
                    """,
                    (data_id,),
                )
            rows = cur.fetchall()
        return [
            WalChunk(
                id=row[0],
                data_id=row[1],
                chunk_index=row[2],
                channel_name=row[3],
                units=row[4],
                timestamps=json.loads(row[5]),
                values=json.loads(row[6]),
            )
            for row in rows
        ]

    def prune(self, retention_seconds: float) -> int:
        """Drop acknowledged chunks older than retention_seconds. Returns count."""
        cutoff = time.time() - retention_seconds
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM chunks WHERE acknowledged = 1 AND created_at < ?",
                (cutoff,),
            )
            return cur.rowcount or 0
