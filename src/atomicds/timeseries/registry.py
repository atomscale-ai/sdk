from __future__ import annotations

from atomicds.timeseries import (
    MetrologyProvider,
    OpticalProvider,
    RHEEDProvider,
    TimeseriesProvider,
)

_PROVIDER_CLASSES: dict[str, type[TimeseriesProvider]] = {
    RHEEDProvider.TYPE: RHEEDProvider,
    OpticalProvider.TYPE: OpticalProvider,
    MetrologyProvider.TYPE: MetrologyProvider,
}


def get_provider(data_type: str) -> TimeseriesProvider:
    try:
        return _PROVIDER_CLASSES[data_type]()  # type: ignore[call-arg]
    except KeyError:
        raise ValueError(f"Unsupported timeseries type: '{data_type}'")  # noqa: B904
