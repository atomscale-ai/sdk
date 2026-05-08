"""FilmSense FS-1 TCP/binary protocol client.

The FS-1 exposes a TCP socket on port 4000 with a fixed-prefix command
protocol::

    [3, 2, 1, <CmdID>, <optional payload>]

Each command echoes a status byte; some commands return additional structured
data (model lists, parameter pairs, etc.). This module wraps the protocol in
context-managed methods, exposes only the commands the adapter uses, and stays
deliberately thin so it can be exercised end-to-end against a mock TCP server
in tests (``tests/adapters/filmsense/conftest.py``).

Reference: ``FS-API_Test Python.txt`` from the customer materials.
"""

from __future__ import annotations

import logging
import socket
import struct
from collections.abc import Iterable
from contextlib import suppress
from types import TracebackType

logger = logging.getLogger(__name__)

# Fixed command-prefix bytes
_PREFIX = bytes([3, 2, 1])

# Command IDs we use (subset of the full FS-1 API)
CMD_LOCK = 10
CMD_GET_MODELS = 11
CMD_SET_MODEL = 12
CMD_GET_PARMS = 14
CMD_SET_ACQ_TIME = 18
CMD_SAVE_DYNAMIC = 20
CMD_START_DYNAMIC = 17
CMD_STOP_DYNAMIC = 21
CMD_PAUSE_DYNAMIC = 22
CMD_RESUME_DYNAMIC = 23
CMD_TRIGGER_DYNAMIC = 24
CMD_NEXT_LAYER = 26


class FilmSenseError(Exception):
    """Raised when the FS-1 returns a non-zero status or the socket dies."""


class FilmSenseClient:
    """Synchronous TCP client for the FilmSense FS-1.

    Use as a context manager so the socket is reliably closed on shutdown::

        with FilmSenseClient("169.254.1.1", 4000) as fs:
            fs.set_acquisition_time(0.4)
            fs.start_dynamic_measurements()
            params = fs.get_parameters()
    """

    def __init__(
        self,
        host: str,
        port: int = 4000,
        connect_timeout: float = 5.0,
        recv_timeout: float = 30.0,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout
        self._sock: socket.socket | None = None

    # ------------------------------------------------------------------ lifecycle

    def connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout)
        try:
            sock.connect((self.host, self.port))
        except OSError as e:
            sock.close()
            msg = f"FS-1 connect to {self.host}:{self.port} failed: {e}"
            raise FilmSenseError(msg) from e
        sock.settimeout(self.recv_timeout)
        self._sock = sock
        logger.debug("FS-1 connected: %s:%s", self.host, self.port)

    def close(self) -> None:
        if self._sock is not None:
            with suppress(OSError):
                self._sock.shutdown(socket.SHUT_RDWR)
            with suppress(OSError):
                self._sock.close()
            self._sock = None
            logger.debug("FS-1 socket closed")

    def __enter__(self) -> FilmSenseClient:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ low-level

    def _send(self, cmd_id: int, payload: bytes | Iterable[int] | None = None) -> None:
        if self._sock is None:
            msg = "FS-1 socket not connected"
            raise FilmSenseError(msg)
        frame = bytearray(_PREFIX)
        frame.append(cmd_id)
        if payload is not None:
            frame.extend(payload)
        try:
            self._sock.sendall(bytes(frame))
        except OSError as e:
            msg = f"FS-1 send failed: {e}"
            raise FilmSenseError(msg) from e

    def _recv_exact(self, n: int) -> bytes:
        if self._sock is None:
            msg = "FS-1 socket not connected"
            raise FilmSenseError(msg)
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self._sock.recv(n - len(buf))
            except OSError as e:
                msg = f"FS-1 recv failed: {e}"
                raise FilmSenseError(msg) from e
            if not chunk:
                msg = (
                    f"FS-1 closed connection while reading "
                    f"({len(buf)}/{n} bytes received)"
                )
                raise FilmSenseError(msg)
            buf.extend(chunk)
        return bytes(buf)

    def _recv_status(self) -> int:
        return self._recv_exact(1)[0]

    def _check_status(self, status: int, what: str) -> None:
        if status != 0:
            msg = f"FS-1 {what} returned non-zero status: {status}"
            raise FilmSenseError(msg)

    # ------------------------------------------------------------------ commands

    def set_lock_mode(self, lock: bool) -> int:
        self._send(CMD_LOCK, [1 if lock else 0])
        return self._recv_status()

    def get_models(self) -> list[str]:
        self._send(CMD_GET_MODELS, None)
        n = self._recv_status()
        models: list[str] = []
        for _ in range(n):
            length = self._recv_status()
            models.append(self._recv_exact(length).decode("utf-8", errors="replace"))
        return models

    def set_model(self, index: int) -> None:
        self._send(CMD_SET_MODEL, [index])
        self._check_status(self._recv_status(), "SetModel")

    def set_acquisition_time(self, seconds: float) -> None:
        self._send(CMD_SET_ACQ_TIME, struct.pack("<f", seconds))
        self._check_status(self._recv_status(), "SetAcqTime")

    def start_dynamic_measurements(self) -> None:
        self._send(CMD_START_DYNAMIC, None)
        self._check_status(self._recv_status(), "StartDynamicMeasurements")

    def stop_dynamic_measurements(self) -> None:
        self._send(CMD_STOP_DYNAMIC, None)
        self._check_status(self._recv_status(), "StopDynamicMeasurements")

    def trigger_dynamic_measurement(self) -> None:
        self._send(CMD_TRIGGER_DYNAMIC, None)
        self._check_status(self._recv_status(), "TriggerDynamicMeasurement")

    def next_layer(self) -> None:
        self._send(CMD_NEXT_LAYER, None)
        self._check_status(self._recv_status(), "NextLayer")

    def get_parameters(self) -> list[tuple[str, float]]:
        """Return the FS-1's current named-parameter snapshot.

        Each element is ``(parameter_name, float_value)``. Names are
        model-dependent and may include wavelength suffixes (e.g.
        ``Psi_465``).
        """
        self._send(CMD_GET_PARMS, None)
        n = self._recv_status()
        out: list[tuple[str, float]] = []
        for _ in range(n):
            name_len = self._recv_status()
            name = self._recv_exact(name_len).decode("utf-8", errors="replace")
            (value,) = struct.unpack("<f", self._recv_exact(4))
            out.append((name, float(value)))
        return out

    def save_dynamic_measurements(self, folder: str, filename: str) -> None:
        """Trigger the FS-1 to dump the current dynamic run to its local disk."""
        payload = bytearray()
        folder_b = folder.encode("utf-8")
        filename_b = filename.encode("utf-8")
        payload.append(len(folder_b))
        payload.extend(folder_b)
        payload.append(len(filename_b))
        payload.extend(filename_b)
        self._send(CMD_SAVE_DYNAMIC, payload)
        self._check_status(self._recv_status(), "SaveDynamicMeasurements")
