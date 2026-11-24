"""Tools (formerly growth instruments) against mocked HTTP helpers.

Covers the AtomCloud rename of instruments to tools: the new response keys, the
deprecated keys that still read, and the catalogue and sample frames that carry a tool.
"""

import warnings

import pytest

from atomscale import Client
from atomscale.tools import ToolRecord, as_tool

TOOL_ID = "8d3e7f10-2b4c-4d6e-9f01-3a5b7c9d1e2f"
HOST_ID = "11111111-2222-4333-8444-555555555555"
SAMPLE_ID = "287badaa-0000-0000-0000-000000000000"
DATA_ID = "d661bcdc-0000-0000-0000-000000000000"
TARGET_ID = "aaaaaaaa-0000-0000-0000-000000000000"


def _tool(**overrides):
    tool = {
        "id": TOOL_ID,
        "legacy_id": 136,
        "name": "RHEED on Main MBE",
        "tool_type": "rheed",
        "make_id": 4,
        "manufacturer": "Staib",
        "model": "RH 30",
        "tool_metadata": {"label": "RHEED on Main MBE", "serial_id": "SN-1"},
        "host_tool_id": HOST_ID,
        "user_id": None,
        "user_name": None,
        "detail_note_content": None,
        "detail_note_last_updated": None,
        "last_updated": "2026-11-20T10:00:00",
    }
    tool.update(overrides)
    return tool


def _legacy_tool():
    """The same instrument as AtomCloud sent it before instruments were renamed tools."""
    return {
        "id": 136,
        "synth_source_id": TOOL_ID,
        "legacy_id": 136,
        "source_name": "Main MBE",
        "synth_source_type": "mbe",
        "source_makes_id": 4,
        "source_manufacturer": "Veeco",
        "source_model": "GEN10",
        "source_metadata": {"label": "Main MBE"},
        "user_id": "99999999-0000-0000-0000-000000000000",
        "user_name": "Ada",
    }


@pytest.fixture
def client():
    return Client(api_key="key_test", endpoint="http://example.com/")


def test_list_growth_instruments_returns_tools(client, monkeypatch):
    monkeypatch.setattr(client, "_get", lambda **_: [_tool()])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        [tool] = client.list_growth_instruments()

        assert tool["id"] == TOOL_ID
        assert tool["name"] == "RHEED on Main MBE"
        assert tool["tool_type"] == "rheed"
        assert tool["host_tool_id"] == HOST_ID
        assert tool["user_name"] is None
    assert tool == _tool()


def test_list_growth_instruments_when_there_are_none(client, monkeypatch):
    monkeypatch.setattr(client, "_get", lambda **_: None)

    assert client.list_growth_instruments() == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("synth_source_id", "id"),
        ("source_name", "name"),
        ("synth_source_type", "tool_type"),
        ("source_makes_id", "make_id"),
        ("source_metadata", "tool_metadata"),
        ("source_model", "model"),
        ("source_manufacturer", "manufacturer"),
    ],
)
def test_a_tool_still_reads_by_its_old_keys_with_a_warning(old, new):
    tool = as_tool(_tool())

    with pytest.warns(DeprecationWarning, match=f"{old!r} is deprecated; use {new!r}"):
        assert tool[old] == tool[new]
    with pytest.warns(DeprecationWarning, match=f"{old!r} is deprecated"):
        assert tool.get(old) == tool[new]


def test_an_unknown_key_is_still_missing():
    tool = as_tool(_tool())

    with pytest.raises(KeyError):
        tool["nope"]
    assert tool.get("nope", "fallback") == "fallback"


def test_an_old_key_whose_field_is_absent_is_missing():
    tool = ToolRecord(id=TOOL_ID)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(KeyError):
            tool["source_name"]
        assert tool.get("source_name") is None


def test_a_tool_from_before_the_rename_reads_by_the_new_keys():
    tool = as_tool(_legacy_tool())

    assert isinstance(tool, ToolRecord)
    assert tool["id"] == TOOL_ID, "the UUID wins over the old integer id"
    assert tool["name"] == "Main MBE"
    assert tool["tool_type"] == "mbe"
    assert tool["make_id"] == 4
    assert tool["manufacturer"] == "Veeco"
    assert tool["model"] == "GEN10"
    assert tool["tool_metadata"] == {"label": "Main MBE"}
    assert "synth_source_id" not in tool


def test_as_tool_passes_a_missing_tool_through():
    assert as_tool(None) is None


def test_create_growth_instrument_returns_the_tool_id(client, monkeypatch):
    sent = {}

    def post(**kwargs):
        sent.update(kwargs)
        return _tool()

    monkeypatch.setattr(client, "_post_or_put", post)

    tool_id = client.create_growth_instrument(
        label="RHEED on Main MBE",
        name="Staib RH 30",
        instrument_type="rheed",
        host_tool_id=HOST_ID,
    )

    assert tool_id == TOOL_ID
    assert sent["sub_url"] == "instruments/synthesis"
    assert sent["body"] == {
        "label": "RHEED on Main MBE",
        "name": "Staib RH 30",
        "type": "rheed",
        "serial_id": None,
        "host_tool_id": HOST_ID,
    }


def test_create_growth_instrument_leaves_out_an_unset_host(client, monkeypatch):
    sent = {}

    def post(**kwargs):
        sent.update(kwargs)
        return _tool(host_tool_id=None)

    monkeypatch.setattr(client, "_post_or_put", post)

    client.create_growth_instrument(
        label="Main MBE", name="Veeco GEN10", instrument_type="mbe"
    )

    assert "host_tool_id" not in sent["body"]


def test_delete_growth_instrument_by_tool_id(client, monkeypatch):
    sent = {}
    monkeypatch.setattr(client, "_delete", lambda **kwargs: sent.update(kwargs))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        client.delete_growth_instrument(tool_id=TOOL_ID)

    assert sent == {
        "sub_url": "instruments/synthesis",
        "params": {"synthesis_instrument_ids": TOOL_ID},
    }


def test_delete_growth_instrument_by_its_deprecated_keyword(client, monkeypatch):
    sent = {}
    monkeypatch.setattr(client, "_delete", lambda **kwargs: sent.update(kwargs))

    with pytest.warns(DeprecationWarning, match="synth_source_id"):
        client.delete_growth_instrument(synth_source_id=TOOL_ID)

    assert sent["params"] == {"synthesis_instrument_ids": TOOL_ID}


def test_delete_growth_instrument_refuses_both_names(client, monkeypatch):
    monkeypatch.setattr(client, "_delete", lambda **_: pytest.fail("deleted"))

    with pytest.raises(TypeError, match="not both"):
        client.delete_growth_instrument(TOOL_ID, synth_source_id=TOOL_ID)
    with pytest.raises(TypeError, match="needs a tool_id"):
        client.delete_growth_instrument()


def _catalogue_row(**overrides):
    row = {
        "data_id": DATA_ID,
        "user_id": None,
        "upload_datetime": "2026-11-20 10:00:00.000000",
        "collected_datetime": None,
        "last_updated": "2026-11-20 10:00:00.000000",
        "file_metadata": {},
        "processed_file_type": None,
        "char_source_type": "rheed",
        "raw_name": "run.imm",
        "pipeline_status": "success",
        "raw_file_type": "imm",
        "tool_name": "Main MBE",
        "tool_id": HOST_ID,
        "sample_name": "GaAs",
        "target_material_id": TARGET_ID,
        "tags": [],
        "name": None,
        "physical_sample_id": SAMPLE_ID,
        "physical_sample_name": "Wafer-A1",
        "detail_note_content": None,
        "detail_note_last_updated": None,
        "bucket_file_name": "x",
        "projects": [],
        "sha3_256": "abc",
    }
    row.update(overrides)
    return row


def test_search_shows_the_tool_name_as_the_instrument_source(client, monkeypatch):
    monkeypatch.setattr(client, "_get", lambda **_: [_catalogue_row()])

    catalogue = client.search()

    assert catalogue["Instrument Source"].tolist() == ["Main MBE"]
    assert catalogue["Sample Name"].tolist() == ["GaAs"]
    assert catalogue["Owner"].tolist() == [None], "the creator's account is deleted"
    for hidden in ("tool_id", "tool_name", "target_material_id", "user_id"):
        assert hidden not in catalogue.columns


def test_search_reads_a_catalogue_from_before_the_rename(client, monkeypatch):
    row = _catalogue_row()
    row["source_name"] = row.pop("tool_name")
    row["synth_source_id"] = row.pop("tool_id")
    monkeypatch.setattr(client, "_get", lambda **_: [row])

    catalogue = client.search()

    assert catalogue["Instrument Source"].tolist() == ["Main MBE"]
    assert "synth_source_id" not in catalogue.columns


def _sample(**overrides):
    sample = {
        "id": SAMPLE_ID,
        "name": "Wafer-A1",
        "user_id": None,
        "owner_id": None,
        "owner_name": None,
        "target_material_id": TARGET_ID,
        "target_material": {"substrate": "GaAs", "sample_name": "GaAs(001)"},
        "tool": {**_tool(), "organization_id": None, "version": "1"},
        "geometry": None,
        "projects": [],
        "detail_notes": None,
        "num_data_items": 3,
        "last_updated": "2026-11-20T10:00:00",
        "created_datetime": "2026-11-20T10:00:00",
    }
    sample.update(overrides)
    return sample


def test_list_physical_samples_shows_the_sample_tool(client, monkeypatch):
    monkeypatch.setattr(client, "_get", lambda **_: [_sample(), _sample(tool=None)])

    samples = client.list_physical_samples()

    first, second = samples["Growth Instrument"].tolist()
    assert isinstance(first, ToolRecord)
    assert first["id"] == TOOL_ID
    with pytest.warns(DeprecationWarning, match="'source_name' is deprecated"):
        assert first["source_name"] == "RHEED on Main MBE"
    assert second is None
    assert samples["Owner"].tolist() == [None, None]
    assert samples["Target Material"].tolist()[0] == {
        "substrate": "GaAs",
        "sample_name": "GaAs(001)",
    }
    assert "tool" not in samples.columns


def test_list_physical_samples_from_before_the_rename(client, monkeypatch):
    sample = _sample()
    del sample["tool"]
    sample["growth_instrument"] = _legacy_tool()
    sample["growth_instrument_id"] = TOOL_ID
    monkeypatch.setattr(client, "_get", lambda **_: [sample])

    samples = client.list_physical_samples()

    [tool] = samples["Growth Instrument"].tolist()
    assert tool["id"] == TOOL_ID
    assert tool["name"] == "Main MBE"
    assert "growth_instrument_id" not in samples.columns
