"""Registry for data providers."""

from __future__ import annotations

from atomscale.similarity.provider import SimilarityTrajectoryProvider

from .ellipsometry import EllipsometryProvider
from .optical import OpticalProvider
from .provider import TimeseriesProvider
from .recipe import RecipeProvider
from .rheed import RHEEDProvider
from .tool_state import ToolStateProvider

_PROVIDER_CLASSES: dict[str, type[TimeseriesProvider]] = {
    RHEEDProvider.TYPE: RHEEDProvider,
    OpticalProvider.TYPE: OpticalProvider,
    ToolStateProvider.TYPE: ToolStateProvider,
    # Accept data catalogue entries created before the backend enum migration.
    "metrology": ToolStateProvider,
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
