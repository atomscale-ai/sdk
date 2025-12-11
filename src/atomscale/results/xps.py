from __future__ import annotations

from uuid import UUID

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from monty.json import MSONable


class XPSResult(MSONable):
    def __init__(
        self,
        data_id: UUID | str,
        xps_id: UUID | str,
        binding_energies: list[float],
        intensities: list[float],
        predicted_composition: dict[str, float],
        detected_peaks: dict[str, float | str],
        elements_manually_set: bool,
    ):
        """XPS result

        Args:
            data_id (UUID | str): Data ID for the entry in the data catalogue.
            xps_id (UUID | str): Unique ID for this specific XPS result.
            binding_energies (list[float]): List of binding energy values in eV.
            intensities (list[float]): List of intensity values.
            predicted_composition (dict[str, float]): Mapping between element symbols and
                predicted fractional composition values.
            detected_peaks (dict[str, float | str]): Mapping with peak transition labels.
            elements_manually_set (bool): Whether the elements represented in the predicted composition
                were manually specified by the user.
        """
        self.data_id = data_id
        self.xps_id = xps_id
        self.binding_energies = binding_energies
        self.intensities = intensities
        self.predicted_composition = predicted_composition
        self.detected_peaks = detected_peaks
        self.elements_manually_set = elements_manually_set

    def get_plot(self) -> Figure:
        """Get plot of X-ray Photoelectron Spectrum

        Returns:
            (Figure): Matplotlib Figure object containing plot data
        """
        # TODO: Add option to include peak identification data
        # TODO: Add option for different backend including plotly for interactivity
        x = self.binding_energies
        y = self.intensities

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, color="#348ABD", linewidth=1)
        ax.set_xlabel("Binding Energy [eV]", fontsize=12)
        ax.set_ylabel("Intensity", fontsize=12)

        ax.grid(color="#E0E0E0", linestyle="--", linewidth=0.5)
        ax.tick_params(axis="both", which="major", labelsize=10)

        ax.set_xlim(max(x), 0)
        plt.close()
        return fig
