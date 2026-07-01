from pathlib import Path

import pytest

from atomscale.adapters.filmsense.config import AdapterConfig


def _minimal_dict(tmp_path: Path) -> dict:
    return {
        "fs1": {"host": "10.0.0.5", "port": 4000, "acquisition_seconds": 0.4},
        "streamer": {"api_key": "test-key", "push_interval_seconds": 3.0},
        "lifecycle": {"watch_dir": str(tmp_path / "watch")},
        "wal": {"path": str(tmp_path / "wal.sqlite")},
    }


def test_from_dict_minimal(tmp_path: Path):
    cfg = AdapterConfig.from_dict(_minimal_dict(tmp_path))
    assert cfg.fs1.host == "10.0.0.5"
    assert cfg.fs1.acquisition_seconds == 0.4
    assert cfg.streamer.api_key == "test-key"
    assert cfg.streamer.push_interval_seconds == 3.0
    assert cfg.lifecycle.watch_dir == tmp_path / "watch"
    assert cfg.wal.path == tmp_path / "wal.sqlite"
    assert cfg.archive.enabled is False
    assert cfg.dry_run is False


def test_api_key_falls_back_to_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AS_API_KEY", "from-env")
    data = _minimal_dict(tmp_path)
    del data["streamer"]["api_key"]
    cfg = AdapterConfig.from_dict(data)
    assert cfg.streamer.api_key == "from-env"


def test_missing_api_key_raises(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AS_API_KEY", raising=False)
    data = _minimal_dict(tmp_path)
    del data["streamer"]["api_key"]
    with pytest.raises(ValueError, match="api_key"):
        AdapterConfig.from_dict(data)


def test_from_toml(tmp_path: Path):
    cfg_path = tmp_path / "filmsense.toml"
    # Embed paths as TOML *literal* strings (single quotes): on Windows a path
    # like C:\Users\... contains backslash sequences that a basic (double-quoted)
    # string would parse as escapes, failing with "Invalid hex value" on \U.
    watch_dir = tmp_path / "watch"
    wal_path = tmp_path / "wal.sqlite"
    cfg_path.write_text(
        f"""
dry_run = true

[fs1]
host = "169.254.1.1"
acquisition_seconds = 0.5

[streamer]
api_key = "toml-key"

[lifecycle]
watch_dir = '{watch_dir}'

[wal]
path = '{wal_path}'

[archive]
enabled = true
folder = "Hinkle"
"""
    )
    cfg = AdapterConfig.from_toml(cfg_path)
    assert cfg.streamer.api_key == "toml-key"
    assert cfg.lifecycle.watch_dir == watch_dir
    assert cfg.wal.path == wal_path
    assert cfg.archive.enabled is True
    assert cfg.archive.folder == "Hinkle"
    assert cfg.dry_run is True
