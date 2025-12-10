from __future__ import annotations

from uuid import UUID

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from monty.json import MSONable


class PhotoluminescenceResult(MSONable):
    """Photoluminescence spectrum result."""

    def __init__(
        self,
        data_id: UUID | str,
        photoluminescence_id: UUID | str,
        energies: list[float],
        intensities: list[float],
        detected_peaks: dict[str, float | str] | None = None,
        last_updated: str | None = None,
    ):
        """Initializes a photoluminescence result.

        Args:
            data_id: Data catalogue identifier.
            photoluminescence_id: Unique identifier for the photoluminescence result.
            energies: Energy axis values.
            intensities: Intensity values aligned with `energies`.
            detected_peaks: Optional mapping of peak labels to positions/metadata.
            last_updated: Optional last-updated timestamp string.
        """
        self.data_id = data_id
        self.photoluminescence_id = photoluminescence_id
        self.energies = energies
        self.intensities = intensities
        self.detected_peaks = detected_peaks or {}
        self.last_updated = last_updated

    def get_plot(self) -> Figure:
        """Returns a Matplotlib figure of the spectrum."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.energies, self.intensities, color="#348ABD", linewidth=1)
        ax.set_xlabel("Energy", fontsize=12)
        ax.set_ylabel("Intensity", fontsize=12)
        ax.grid(color="#E0E0E0", linestyle="--", linewidth=0.5)
        ax.tick_params(axis="both", which="major", labelsize=10)
        plt.close()
        return fig
