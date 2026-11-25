"""Live fixture discovery must distinguish catalogue success from feature data."""

from unittest.mock import Mock

import pytest
from pandas import DataFrame
from requests.exceptions import ConnectionError as RequestsConnectionError

from atomscale.core import ClientError
from atomscale.timeseries.rheed import RHEEDProvider

from .conftest import find_populated_rheed_video


def _series(points):
    return {"series_by_angle": [{"angle": "0", "series": points}]}


def test_discovery_passes_empty_success_entries_and_deduplicates(monkeypatch):
    # The development catalogue had twelve newest synthetic success entries
    # with MP4 metadata but no feature rows before a populated analysis.
    empty_ids = [f"synthetic-{i}" for i in range(12)]
    client = Mock()
    client.search.side_effect = [
        DataFrame({"Data ID": [*empty_ids, None]}),
        DataFrame({"Data ID": [*empty_ids, "analyzed"]}),
    ]
    client._get.side_effect = [
        None,
        {"series_by_angle": []},
        *[_series([]) for _ in range(10)],
        _series([{"frame_number": 0, "specular_intensity": 5.0}]),
    ]
    # Eligibility must not depend on the parser under test. A parser failure
    # must surface after discovery rather than making it choose another item.
    monkeypatch.setattr(
        RHEEDProvider, "to_dataframe", Mock(side_effect=AssertionError("parser called"))
    )

    data_id = find_populated_rheed_video(client, preferred_id=empty_ids[0])

    assert data_id == "analyzed"
    assert client._get.call_count == 13
    assert [call.kwargs["sub_url"] for call in client._get.call_args_list] == [
        f"rheed/timeseries/{data_id}/" for data_id in [*empty_ids, "analyzed"]
    ]
    assert all(
        call.kwargs["params"] == {"last_n": 1} for call in client._get.call_args_list
    )
    assert [call.kwargs for call in client.search.call_args_list] == [
        {"data_type": "rheed", "status": "success", "include_organization_data": False},
        {"data_type": "rheed", "status": "success", "include_organization_data": True},
    ]


def test_discovery_reuses_populated_preferred_id_without_searching():
    client = Mock()
    client._get.return_value = _series([{"frame_number": 0}])

    assert find_populated_rheed_video(client, preferred_id="analyzed") == "analyzed"

    client.search.assert_not_called()
    client._get.assert_called_once_with(
        sub_url="rheed/timeseries/analyzed/", params={"last_n": 1}
    )


def test_discovery_fails_with_diagnostics_at_candidate_limit():
    client = Mock()
    client.search.return_value = DataFrame({"Data ID": ["empty-1", "empty-2", "later"]})
    client._get.return_value = _series([])

    with pytest.raises(
        pytest.fail.Exception, match="2 distinct success candidates"
    ) as exc:
        find_populated_rheed_video(client, max_candidates=2)

    assert "['empty-1', 'empty-2']" in str(exc.value)
    assert client._get.call_count == 2
    assert client.search.call_count == 1


def test_discovery_fails_when_both_catalogues_are_empty():
    client = Mock()
    client.search.return_value = DataFrame()

    with pytest.raises(pytest.fail.Exception, match="0 distinct success candidates"):
        find_populated_rheed_video(client)

    assert client.search.call_count == 2
    client._get.assert_not_called()


@pytest.mark.parametrize("operation", ["search", "_get"])
@pytest.mark.parametrize(
    "error",
    [
        ClientError("Unauthorized", status_code=401),
        ClientError("Backend failed", status_code=500),
        RequestsConnectionError("Connection failed"),
    ],
)
def test_discovery_propagates_api_failures(operation, error):
    client = Mock()
    client.search.return_value = DataFrame({"Data ID": ["analyzed"]})
    getattr(client, operation).side_effect = error

    with pytest.raises(type(error)) as exc:
        find_populated_rheed_video(client)

    assert exc.value is error


@pytest.mark.parametrize(
    "raw", [{}, [], {"series_by_angle": None}, {"series_by_angle": [{"points": []}]}]
)
def test_discovery_fails_on_changed_response_schema(raw):
    client = Mock()
    client._get.return_value = raw

    with pytest.raises(AssertionError, match="Malformed RHEED timeseries.*analyzed"):
        find_populated_rheed_video(client, preferred_id="analyzed")

    client.search.assert_not_called()
