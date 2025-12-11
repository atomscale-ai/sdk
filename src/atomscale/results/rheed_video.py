from __future__ import annotations

from uuid import UUID

from monty.json import MSONable
from pandas import DataFrame

from atomscale.results import RHEEDImageResult


class RHEEDVideoResult(MSONable):
    def __init__(
        self,
        data_id: UUID | str,
        timeseries_data: DataFrame,
        snapshot_image_data: list[RHEEDImageResult] | None,
        rotating: bool,
    ):
        """RHEED video result

        Args:
            data_id (UUID | str): Data ID for the entry in the data catalogue.
            timeseries_data (DataFrame): Pandas DataFrame with timeseries data associated with the video.
                Includes cluster assignments, specular intensity, strain, etc...
            snapshot_image_data (list[atomscale.results.rheed_image.RHEEDImageResult]): List of
                :class:`atomscale.results.rheed_image.RHEEDImageResult` objects containing data for images associated
                with each user extracted snapshot in the video.
            rotating (bool): Whether the video was taken of a rotating stage.
        """
        self.data_id = data_id
        self.timeseries_data = timeseries_data
        self.snapshot_image_data = snapshot_image_data
        self.rotating = rotating

    # NOTE: This is temporarily deprecated
    #
    # def get_plot(self) -> Figure:
    #     """Get plot of timeseries data associated with this RHEED video
    #
    #     Returns:
    #         (Figure): Matplotlib Figure object containing plot data
    #     """
    #     fig, axes = plt.subplots(nrows=6, sharex=True, figsize=(10, 10))
    #
    #     time = self.timeseries_data["Time"]
    #
    #     timeseries_data = self.timeseries_data.drop(columns=["Time"])
    #     timeseries_data = timeseries_data.rename(
    #         columns={"Oscillation Period": "Oscillation Period [s]"}
    #     )
    #     colors = {
    #         "Cluster ID": "black",
    #         "Specular Intensity": "#0D74CE",
    #         "First Order Intensity": "#0588F0",
    #         "Cumulative Strain": "#CA244D",
    #         "Relative Strain": "#DC3B5D",
    #         "Oscillation Period [s]": "#AB4ABA",
    #         "Diffraction Spot Count": "#CC4E00",
    #         "Lattice Spacing": "#CC4E00",
    #     }
    #
    #     linewidth = 3
    #     for col, axis in zip(timeseries_data.columns, axes):
    #         (line,) = axis.plot(
    #             time,
    #             timeseries_data[col].values,
    #             label=col,
    #             color=colors[col],
    #             linewidth=linewidth,
    #         )
    #         axis.grid(color="#E0E0E0", linestyle="--", linewidth=0.5)
    #         axis.legend([line], [col])
    #
    #     axes[-1].set_xlabel("Time [s]", fontsize=12)
    #     plt.close()
    #     return fig
