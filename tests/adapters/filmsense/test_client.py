import pytest

from atomscale.adapters.filmsense.client import (
    CMD_GET_PARMS,
    CMD_SAVE_DYNAMIC,
    CMD_START_DYNAMIC,
    FilmSenseClient,
    FilmSenseError,
)


def test_connect_and_close(mock_fs1):
    with FilmSenseClient(mock_fs1.host, mock_fs1.port):
        pass
    # No-op: just verifying the context manager handles its own cleanup.


def test_set_acq_time_sets_state_on_server(mock_fs1):
    with FilmSenseClient(mock_fs1.host, mock_fs1.port) as client:
        client.set_acquisition_time(0.4)
    assert mock_fs1.state.acq_time == pytest.approx(0.4, abs=1e-6)


def test_get_models(mock_fs1):
    with FilmSenseClient(mock_fs1.host, mock_fs1.port) as client:
        models = client.get_models()
    assert models == mock_fs1.state.models


def test_start_dynamic_and_lifecycle(mock_fs1):
    with FilmSenseClient(mock_fs1.host, mock_fs1.port) as client:
        client.start_dynamic_measurements()
        client.stop_dynamic_measurements()
    assert mock_fs1.state.started is True
    assert mock_fs1.state.stopped is True
    cmds = [r.cmd for r in mock_fs1.state.received]
    assert CMD_START_DYNAMIC in cmds


def test_get_parameters_round_trips_values(mock_fs1):
    mock_fs1.state.queue_parms(
        [("Psi_465", 21.5), ("Delta_465", 170.25), ("Thickness", 12.34)]
    )
    with FilmSenseClient(mock_fs1.host, mock_fs1.port) as client:
        params = client.get_parameters()
    assert [p[0] for p in params] == ["Psi_465", "Delta_465", "Thickness"]
    assert params[0][1] == pytest.approx(21.5, abs=1e-4)
    assert params[1][1] == pytest.approx(170.25, abs=1e-4)
    assert params[2][1] == pytest.approx(12.34, abs=1e-4)


def test_get_parameters_empty(mock_fs1):
    with FilmSenseClient(mock_fs1.host, mock_fs1.port) as client:
        assert client.get_parameters() == []


def test_save_dynamic_measurements(mock_fs1):
    with FilmSenseClient(mock_fs1.host, mock_fs1.port) as client:
        client.save_dynamic_measurements("Default", "run-001")
    assert mock_fs1.state.saved == ("Default", "run-001")
    assert any(r.cmd == CMD_SAVE_DYNAMIC for r in mock_fs1.state.received)


def test_connect_failure_raises_filmsense_error():
    # Try a closed port; expect FilmSenseError, not bare OSError.
    client = FilmSenseClient("127.0.0.1", 1, connect_timeout=0.5)
    with pytest.raises(FilmSenseError):
        client.connect()


def test_get_parms_command_id_constant():
    # Sanity: the public constant is what the protocol uses.
    assert CMD_GET_PARMS == 14
