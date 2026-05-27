"""Tests for the generic adapter host CLI (``python -m atomscale.adapters``)."""

from __future__ import annotations

import json

import pytest

from atomscale.adapters import __main__ as host
from atomscale.adapters import registry


def _events(captured_out: str) -> list[dict]:
    return [json.loads(line) for line in captured_out.splitlines() if line.strip()]


def test_list_outputs_filmsense_manifest(capsys):
    rc = host.main(["list"])
    assert rc == 0

    manifests = json.loads(capsys.readouterr().out)
    by_id = {m["id"]: m for m in manifests}
    assert "filmsense" in by_id

    fs = by_id["filmsense"]
    assert fs["version"]
    schema = fs["config_schema"]
    assert schema["type"] == "object"
    assert set(schema["required"]) >= {"lifecycle", "wal"}
    # The API key is a secret supplied via env — it must never appear in the
    # operator-facing config schema.
    assert "api_key" not in json.dumps(schema)


def test_registry_get_unknown_raises():
    with pytest.raises(KeyError):
        registry.get("does-not-exist")


def test_run_unknown_adapter_emits_fatal_error(capsys):
    rc = host.main(["run", "does-not-exist"])
    assert rc == 2

    events = _events(capsys.readouterr().out)
    assert events
    assert events[-1]["event"] == "error"
    assert events[-1]["fatal"] is True


def test_run_with_bad_config_path_returns_2(capsys, tmp_path):
    rc = host.main(["run", "filmsense", "--config", str(tmp_path / "missing.json")])
    assert rc == 2

    events = _events(capsys.readouterr().out)
    assert events[-1]["event"] == "error"
    assert events[-1]["fatal"] is True
