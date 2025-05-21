import io
from pathlib import Path


class _FileSlice(io.RawIOBase):
    """File-like object exposing exactly *length* bytes starting at *offset*.

    Works with `requests.put(..., data=obj)` so the body is streamed
    in ~8 KiB blocks rather than copied into memory.
    """

    def __init__(self, path: str | Path, offset: int, length: int) -> None:
        super().__init__()
        file_ref = Path(path) if isinstance(path, str) else path
        self._fh = file_ref.open("rb")
        self._fh.seek(offset)
        self._remaining = length

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if size < 0 or size > self._remaining:  # clamp to remaining
            size = self._remaining
        data = self._fh.read(size)
        self._remaining -= len(data)
        return data

    def close(self) -> None:
        self._fh.close()
        super().close()
