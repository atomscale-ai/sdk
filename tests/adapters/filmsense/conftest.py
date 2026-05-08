"""In-process mock FS-1 TCP server for deterministic adapter tests.

The fixture spins up a threaded server on an ephemeral 127.0.0.1 port and
returns a handle the test can use to enqueue ``GetParms`` responses or assert
which commands were received. The mock implements only the subset of the FS-1
protocol the adapter exercises.
"""

from __future__ import annotations

import socket
import struct
import threading
from collections import deque
from contextlib import closing
from dataclasses import dataclass, field

import pytest

# Mirror IDs from atomscale.adapters.filmsense.client (kept local to avoid an
# import cycle in the fixture module)
_PREFIX = bytes([3, 2, 1])
_CMD_LOCK = 10
_CMD_GET_MODELS = 11
_CMD_SET_MODEL = 12
_CMD_GET_PARMS = 14
_CMD_SET_ACQ_TIME = 18
_CMD_SAVE_DYNAMIC = 20
_CMD_START_DYNAMIC = 17
_CMD_STOP_DYNAMIC = 21
_CMD_TRIGGER_DYNAMIC = 24
_CMD_NEXT_LAYER = 26


@dataclass
class _Recv:
    cmd: int
    payload: bytes = b""


@dataclass
class MockFSState:
    received: list[_Recv] = field(default_factory=list)
    parm_queue: deque[list[tuple[str, float]]] = field(default_factory=deque)
    models: list[str] = field(default_factory=lambda: ["SiO2 single layer", "Al2O3 ALD"])
    acq_time: float | None = None
    started: bool = False
    stopped: bool = False
    saved: tuple[str, str] | None = None
    locked: bool = False

    def queue_parms(self, parms: list[tuple[str, float]]) -> None:
        """Enqueue a single GetParms response. Pops on each GetParms call."""
        self.parm_queue.append(parms)


class MockFS1Server:
    def __init__(self) -> None:
        self.state = MockFSState()
        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.host = "127.0.0.1"
        self.port: int = 0

    def __enter__(self) -> MockFS1Server:
        self.start()
        return self

    def __exit__(self, *_exc):  # noqa: ANN001
        self.stop()

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, 0))
        sock.listen(1)
        self.port = sock.getsockname()[1]
        self._server_sock = sock
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._server_sock is not None:
            with closing(self._server_sock):
                # Wake the accept() by connecting to ourselves
                try:
                    with socket.create_connection((self.host, self.port), timeout=0.5):
                        pass
                except OSError:
                    pass
            self._server_sock = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    # ---------------------------------------------------------------- internals

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        self._server_sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                client, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                self._handle_client(client)
            finally:
                with closing(client):
                    pass

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                msg = "client closed mid-frame"
                raise ConnectionError(msg)
            buf.extend(chunk)
        return bytes(buf)

    def _handle_client(self, sock: socket.socket) -> None:
        sock.settimeout(2.0)
        while not self._stop.is_set():
            try:
                prefix = self._recv_exact(sock, 4)
            except (ConnectionError, OSError):
                return
            if prefix[:3] != _PREFIX:
                msg = f"bad command prefix: {prefix[:3]!r}"
                raise AssertionError(msg)
            cmd = prefix[3]

            if cmd == _CMD_LOCK:
                payload = self._recv_exact(sock, 1)
                self.state.locked = bool(payload[0])
                self.state.received.append(_Recv(cmd, payload))
                sock.sendall(b"\x00")
            elif cmd == _CMD_GET_MODELS:
                self.state.received.append(_Recv(cmd))
                resp = bytearray([len(self.state.models)])
                for m in self.state.models:
                    b = m.encode("utf-8")
                    resp.append(len(b))
                    resp.extend(b)
                sock.sendall(bytes(resp))
            elif cmd == _CMD_SET_MODEL:
                payload = self._recv_exact(sock, 1)
                self.state.received.append(_Recv(cmd, payload))
                sock.sendall(b"\x00")
            elif cmd == _CMD_SET_ACQ_TIME:
                payload = self._recv_exact(sock, 4)
                (self.state.acq_time,) = struct.unpack("<f", payload)
                self.state.received.append(_Recv(cmd, payload))
                sock.sendall(b"\x00")
            elif cmd == _CMD_START_DYNAMIC:
                self.state.started = True
                self.state.received.append(_Recv(cmd))
                sock.sendall(b"\x00")
            elif cmd == _CMD_STOP_DYNAMIC:
                self.state.stopped = True
                self.state.received.append(_Recv(cmd))
                sock.sendall(b"\x00")
            elif cmd == _CMD_TRIGGER_DYNAMIC:
                self.state.received.append(_Recv(cmd))
                sock.sendall(b"\x00")
            elif cmd == _CMD_NEXT_LAYER:
                self.state.received.append(_Recv(cmd))
                sock.sendall(b"\x00")
            elif cmd == _CMD_SAVE_DYNAMIC:
                folder_len = self._recv_exact(sock, 1)[0]
                folder = self._recv_exact(sock, folder_len).decode("utf-8")
                fname_len = self._recv_exact(sock, 1)[0]
                fname = self._recv_exact(sock, fname_len).decode("utf-8")
                self.state.saved = (folder, fname)
                self.state.received.append(_Recv(cmd))
                sock.sendall(b"\x00")
            elif cmd == _CMD_GET_PARMS:
                self.state.received.append(_Recv(cmd))
                if self.state.parm_queue:
                    parms = self.state.parm_queue.popleft()
                else:
                    parms = []
                resp = bytearray([len(parms)])
                for name, value in parms:
                    b = name.encode("utf-8")
                    resp.append(len(b))
                    resp.extend(b)
                    resp.extend(struct.pack("<f", value))
                sock.sendall(bytes(resp))
            else:  # pragma: no cover
                msg = f"mock FS-1 received unsupported command: {cmd}"
                raise AssertionError(msg)


@pytest.fixture
def mock_fs1():
    """Spin up a mock FS-1 server on an ephemeral port for the test."""
    with MockFS1Server() as server:
        yield server
