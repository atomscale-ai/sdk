from __future__ import annotations

from uuid import UUID

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from monty.json import MSONable


class RamanResult(MSONable):
    """Raman spectrum result."""

    def __init__(
        self,
        data_id: UUID | str,
        raman_id: UUID | str,
        raman_shift: list[float],
        intensities: list[float],
        detected_peaks: dict[str, float | str] | None = None,
        last_updated: str | None = None,
    ):
        """Initializes a Raman result.

        Args:
            data_id: Data catalogue identifier.
            raman_id: Unique identifier for the Raman result.
            raman_shift: Raman shift axis values.
            intensities: Intensity values aligned with `raman_shift`.
            detected_peaks: Optional mapping of peak labels to positions/metadata.
            last_updated: Optional last-updated timestamp string.
        """
        self.data_id = data_id
        self.raman_id = raman_id
        self.raman_shift = raman_shift
        self.intensities = intensities
        self.detected_peaks = detected_peaks or {}
        self.last_updated = last_updated

    def get_plot(self) -> Figure:
        """Returns a Matplotlib figure of the spectrum."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.raman_shift, self.intensities, color="#348ABD", linewidth=1)
        ax.set_xlabel("Raman Shift", fontsize=12)
        ax.set_ylabel("Intensity", fontsize=12)
        ax.grid(color="#E0E0E0", linestyle="--", linewidth=0.5)
        ax.tick_params(axis="both", which="major", labelsize=10)
        plt.close()
        return fig
