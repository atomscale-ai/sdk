import time

import pytest

from atomscale.adapters.filmsense.wal import ChunkWal, WalChunk


def test_record_and_pending(tmp_path):
    wal = ChunkWal(tmp_path / "wal.sqlite")
    wal.record("data-1", 0, "psi_465", [0.1, 0.2], [21.5, 21.6], "deg")
    wal.record("data-1", 1, "psi_465", [0.3, 0.4], [21.7, 21.8], "deg")

    pending = wal.pending("data-1")
    assert len(pending) == 2
    assert pending[0].chunk_index == 0
    assert pending[0].timestamps == [0.1, 0.2]
    assert pending[0].values == [21.5, 21.6]
    assert pending[0].units == "deg"
    wal.close()


def test_acknowledge_removes_from_pending(tmp_path):
    wal = ChunkWal(tmp_path / "wal.sqlite")
    row_id = wal.record("data-1", 0, "psi_465", [0.1], [21.5], "deg")
    wal.acknowledge(row_id)
    assert wal.pending("data-1") == []
    wal.close()


def test_acknowledge_all(tmp_path):
    wal = ChunkWal(tmp_path / "wal.sqlite")
    wal.record("data-1", 0, "psi_465", [0.1], [21.5], "deg")
    wal.record("data-1", 1, "psi_465", [0.2], [21.6], "deg")
    wal.record("data-2", 0, "psi_465", [0.1], [99.9], "deg")
    wal.acknowledge_all("data-1")
    assert wal.pending("data-1") == []
    pending2 = wal.pending("data-2")
    assert len(pending2) == 1
    assert pending2[0].values == [99.9]
    wal.close()


def test_record_overwrites_same_chunk_index(tmp_path):
    """Re-recording with the same (data_id, chunk_index, channel_name) replaces."""
    wal = ChunkWal(tmp_path / "wal.sqlite")
    wal.record("data-1", 0, "psi_465", [0.1], [21.5], "deg")
    wal.record("data-1", 0, "psi_465", [0.1, 0.2], [21.5, 21.6], "deg")
    pending = wal.pending("data-1")
    assert len(pending) == 1
    assert pending[0].values == [21.5, 21.6]
    wal.close()


def test_pending_filters_by_data_id(tmp_path):
    wal = ChunkWal(tmp_path / "wal.sqlite")
    wal.record("data-1", 0, "psi", [0.1], [1.0])
    wal.record("data-2", 0, "psi", [0.1], [2.0])
    p1 = wal.pending("data-1")
    p2 = wal.pending("data-2")
    assert len(p1) == 1 and p1[0].values == [1.0]
    assert len(p2) == 1 and p2[0].values == [2.0]
    wal.close()


def test_pending_returns_walchunk(tmp_path):
    wal = ChunkWal(tmp_path / "wal.sqlite")
    wal.record("data-1", 0, "psi", [0.1], [1.0], "deg")
    [chunk] = wal.pending("data-1")
    assert isinstance(chunk, WalChunk)
    wal.close()


def test_prune_drops_old_acknowledged(tmp_path, monkeypatch):
    wal = ChunkWal(tmp_path / "wal.sqlite")
    rid_old = wal.record("data-1", 0, "psi", [0.1], [1.0])

    # Force the row's created_at into the past
    wal._conn.execute(
        "UPDATE chunks SET created_at = ? WHERE id = ?", (time.time() - 3600, rid_old)
    )

    # Unacknowledged: prune should NOT remove
    assert wal.prune(retention_seconds=10) == 0
    assert len(wal.pending()) == 1

    wal.acknowledge(rid_old)
    # Acknowledged + old: prune should remove
    assert wal.prune(retention_seconds=10) == 1
    assert wal.pending() == []
    wal.close()


def test_record_validates_lengths(tmp_path):
    wal = ChunkWal(tmp_path / "wal.sqlite")
    with pytest.raises(ValueError, match="same length"):
        wal.record("data-1", 0, "psi", [0.1, 0.2], [21.5])
    wal.close()


def test_persists_across_reopen(tmp_path):
    path = tmp_path / "wal.sqlite"
    wal = ChunkWal(path)
    wal.record("data-1", 0, "psi", [0.1], [21.5], "deg")
    wal.close()

    wal2 = ChunkWal(path)
    pending = wal2.pending("data-1")
    assert len(pending) == 1
    assert pending[0].values == [21.5]
    wal2.close()
