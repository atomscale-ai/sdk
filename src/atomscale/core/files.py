import io
from pathlib import Path


class _FileSlice(io.RawIOBase):
    """Read-only window of a file (offset…offset+length-1)."""

    def __init__(self, path: str | Path, offset: int, length: int) -> None:
        super().__init__()
        p = Path(path) if isinstance(path, str) else path
        self._fh = p.open("rb", buffering=0)
        self._start = offset  # absolute offset in the file
        self._length = length
        self._pos = 0  # position *within* the slice
        self._fh.seek(self._start)

    # --- RawIOBase API --------------------------------------------------
    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._length:
            return b""
        if size < 0 or size > self._length - self._pos:
            size = self._length - self._pos
        data = self._fh.read(size)
        self._pos += len(data)
        return data

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:  # type: ignore[override]
        if whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = self._length + offset
        else:  # SEEK_SET
            new_pos = offset
        if not (0 <= new_pos <= self._length):
            raise ValueError("seek out of range")
        self._fh.seek(self._start + new_pos)
        self._pos = new_pos
        return self._pos

    def tell(self) -> int:  # type: ignore[override]
        return self._pos

    def readable(self) -> bool:  # type: ignore[override]
        return True

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()

    # Let requests see the size so it sets Content-Length
    def __len__(self) -> int:
        return self._length
