"""FilmSense FS-1 ellipsometer adapter for atomscale.

Bridges the FS-1's binary TCP API (port 4000) to ``TimeseriesStreamer`` so that
real-time deposition measurements (psi, delta, thickness, n, k, ...) land in
atomscale as a property-centric ellipsometry stream.

Typical entry point::

    python -m atomscale.adapters.filmsense --config /etc/filmsense.toml
"""

from atomscale.adapters.filmsense.client import FilmSenseClient
from atomscale.adapters.filmsense.config import AdapterConfig
from atomscale.adapters.filmsense.lifecycle import SentinelWatcher
from atomscale.adapters.filmsense.mapping import normalize_param_name
from atomscale.adapters.filmsense.runner import FilmSenseRunner, RunMetadata

__all__ = [
    "AdapterConfig",
    "FilmSenseClient",
    "FilmSenseRunner",
    "RunMetadata",
    "SentinelWatcher",
    "normalize_param_name",
]
