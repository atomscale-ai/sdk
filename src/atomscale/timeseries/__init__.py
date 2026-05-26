"""Atomscale timeseries package.

Convention for time-valued fields:
  * On the wire (JSON): numeric (int or float). Backend response models
    use ``list[float | None]`` for ``unix_timestamp_ms`` because orjson
    does not serialize ``Decimal`` natively.
  * In typed Python surfaces (helper signatures, dataclass fields):
    ``Decimal`` is the canonical type for unix-second / millisecond
    values to preserve precision and signal "this is a fractional
    seconds quantity, not just a number".
  * In pandas DataFrame columns: ``int64`` for whole-millisecond
    ``UNIX Timestamp`` values (lossless and fast). Magnitude is
    milliseconds for metrology / optical / ellipsometry, seconds for
    similarity. Column name disambiguates the unit via
    :func:`align._infer_absolute_time`.

Helpers in :func:`provider.properties_payload_to_dataframe` and
:func:`align._infer_absolute_time` honor this convention.
"""

from .align import align_timeseries
from .ellipsometry import EllipsometryProvider
from .metrology import MetrologyProvider
from .optical import OpticalProvider
from .polling import (
    aiter_poll,
    iter_poll,
    start_polling_task,
    start_polling_thread,
)
from .provider import TimeseriesProvider, properties_payload_to_dataframe
from .registry import get_provider
from .rheed import RHEEDProvider

__all__ = [
    "EllipsometryProvider",
    "MetrologyProvider",
    "OpticalProvider",
    "RHEEDProvider",
    "TimeseriesProvider",
    "aiter_poll",
    "align_timeseries",
    "get_provider",
    "iter_poll",
    "properties_payload_to_dataframe",
    "start_polling_task",
    "start_polling_thread",
]
