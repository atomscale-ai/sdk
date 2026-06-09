"""Registry for data providers."""

from __future__ import annotations

from atomscale.similarity.provider import SimilarityTrajectoryProvider

from .ellipsometry import EllipsometryProvider
from .metrology import MetrologyProvider
from .optical import OpticalProvider
from .provider import TimeseriesProvider
from .recipe import RecipeProvider
from .rheed import RHEEDProvider

_PROVIDER_CLASSES: dict[str, type[TimeseriesProvider]] = {
    RHEEDProvider.TYPE: RHEEDProvider,
    OpticalProvider.TYPE: OpticalProvider,
    MetrologyProvider.TYPE: MetrologyProvider,
    # The backend renamed the "metrology" char-source/data-stream to "tool_state".
    # Accept both so dispatch works whether the API returns the legacy or new value.
    "tool_state": MetrologyProvider,
    # Recipe (process plan) is a distinct first-class type, served from its own
    # /recipe endpoint — not folded into tool-state.
    RecipeProvider.TYPE: RecipeProvider,
    EllipsometryProvider.TYPE: EllipsometryProvider,
    SimilarityTrajectoryProvider.TYPE: SimilarityTrajectoryProvider,
}


def get_provider(data_type: str) -> TimeseriesProvider:
    try:
        return _PROVIDER_CLASSES[data_type]()  # type: ignore[call-arg]
    except KeyError:
        raise ValueError(f"Unsupported timeseries type: '{data_type}'")  # noqa: B904
