"""FilmSense FS-1 ``GetParms`` name → atomscale channel-name normalization.

The FS-1 returns model-dependent parameter names like ``Psi_465``, ``Thickness``,
``Thickness_1``, ``MSE``. Atomscale's ellipsometry domain expects snake_case
names with no embedded units; wavelengths are integer-nm suffixes, multi-layer
thickness uses ``thickness_layer_<N>`` to disambiguate from the wavelength rule.

Unmapped names pass through unchanged (lowercased) so the adapter never silently
drops data it doesn't yet recognize — the canonical mapping is updated as we
encounter new model outputs.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Wavelength-suffixed parameters: ``Psi_465`` → ``psi_465``.
_WAVELENGTH_RE = re.compile(r"^(Psi|Delta|Depol|Intensity|n|k)_(\d+)$")

# Multi-layer thickness: ``Thickness_1`` → ``thickness_layer_1``.
_THICKNESS_LAYER_RE = re.compile(r"^Thickness_(\d+)$")

# Static one-off renames. Keys are matched case-insensitively against the FS-1
# parameter name; values are the atomscale channel name.
_STATIC_RENAMES: dict[str, str] = {
    "thickness": "thickness",
    "mse": "mse_fit",
    "fitdiff": "mse_fit",
    "aveint": "intensity_avg",
    "alignx": "align_x",
    "aligny": "align_y",
    "tilt_x": "align_tilt_x",
    "tilt_y": "align_tilt_y",
    "srcrot": "align_src_rot",
    "srctilt": "align_src_tilt",
    "dettilt": "align_det_tilt",
    "frontx": "align_front_x",
    "fronty": "align_front_y",
    "frontz": "align_front_z",
    "temp": "detector_temperature",
}

# Default unit per channel name. Empty string means "no units" / dimensionless.
# Resolved AFTER normalization so it's keyed by the atomscale channel name.
_DEFAULT_UNITS: dict[str, str] = {
    # Optical constants per wavelength (filled dynamically by prefix below)
    # Static channels:
    "thickness": "nm",
    "mse_fit": "",
    "intensity_avg": "counts",
    "align_x": "mm",
    "align_y": "mm",
    "align_tilt_x": "arcmin",
    "align_tilt_y": "arcmin",
    "align_src_rot": "arcmin",
    "align_src_tilt": "arcmin",
    "align_det_tilt": "arcmin",
    "align_front_x": "mm",
    "align_front_y": "mm",
    "align_front_z": "mm",
    "detector_temperature": "C",
}

# Per-prefix unit lookup (applied to wavelength-suffixed and layer-suffixed
# names where the suffix doesn't change units).
_PREFIX_UNITS: dict[str, str] = {
    "psi": "deg",
    "delta": "deg",
    "depol": "",
    "intensity": "counts",
    "n": "",
    "k": "",
    "thickness_layer": "nm",
}


class NormalizedParam(NamedTuple):
    """Result of normalizing a single FS-1 parameter name."""

    channel_name: str
    units: str


def normalize_param_name(fs_name: str) -> NormalizedParam:
    """Map an FS-1 parameter name to its atomscale channel name and unit.

    Examples:
        >>> normalize_param_name("Psi_465").channel_name
        'psi_465'
        >>> normalize_param_name("Psi_465").units
        'deg'
        >>> normalize_param_name("Thickness_2").channel_name
        'thickness_layer_2'
        >>> normalize_param_name("MSE").channel_name
        'mse_fit'
        >>> normalize_param_name("UnknownParam").channel_name
        'unknownparam'
    """
    name = fs_name.strip()

    m = _WAVELENGTH_RE.match(name)
    if m:
        prefix, lam = m.group(1).lower(), int(m.group(2))
        channel = f"{prefix}_{lam}"
        return NormalizedParam(channel, _PREFIX_UNITS.get(prefix, ""))

    m = _THICKNESS_LAYER_RE.match(name)
    if m:
        return NormalizedParam(
            f"thickness_layer_{int(m.group(1))}",
            _PREFIX_UNITS["thickness_layer"],
        )

    static = _STATIC_RENAMES.get(name.lower())
    if static is not None:
        return NormalizedParam(static, _DEFAULT_UNITS.get(static, ""))

    fallback = name.lower()
    return NormalizedParam(fallback, _DEFAULT_UNITS.get(fallback, ""))
