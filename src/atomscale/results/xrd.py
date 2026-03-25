from __future__ import annotations

from uuid import UUID

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from monty.json import MSONable


class XRDResult(MSONable):
    """X-Ray Diffraction result."""

    def __init__(
        self,
        data_id: UUID | str,
        xrd_id: UUID | str,
        two_theta: list[float],
        intensities: list[float],
        detected_peaks: list[dict] | None = None,
        wavelength_angstrom: float = 1.5406,
        two_theta_unit: str = "degrees",
        spectral_metadata: dict | None = None,
        last_updated: str | None = None,
    ):
        """Initializes an XRD result.

        Args:
            data_id: Data catalogue identifier.
            xrd_id: Unique identifier for the XRD result.
            two_theta: 2-theta angle values in degrees.
            intensities: Intensity values aligned with `two_theta`.
            detected_peaks: Optional list of detected peak dicts with keys
                two_theta, intensity, d_spacing_angstrom, prominence, fwhm_degrees.
            wavelength_angstrom: X-ray source wavelength in angstroms (default Cu Ka).
            two_theta_unit: Unit for two_theta values.
            spectral_metadata: Optional file and acquisition metadata.
            last_updated: Optional last-updated timestamp string.
        """
        self.data_id = data_id
        self.xrd_id = xrd_id
        self.two_theta = two_theta
        self.intensities = intensities
        self.detected_peaks = detected_peaks or []
        self.wavelength_angstrom = wavelength_angstrom
        self.two_theta_unit = two_theta_unit
        self.spectral_metadata = spectral_metadata or {}
        self.last_updated = last_updated

    def get_plot(self) -> Figure:
        """Returns a Matplotlib figure of the XRD pattern."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.two_theta, self.intensities, color="#348ABD", linewidth=1)
        ax.set_xlabel(f"2\u03b8 ({self.two_theta_unit})", fontsize=12)
        ax.set_ylabel("Intensity", fontsize=12)
        ax.grid(color="#E0E0E0", linestyle="--", linewidth=0.5)
        ax.tick_params(axis="both", which="major", labelsize=10)
        plt.close()
        return fig
