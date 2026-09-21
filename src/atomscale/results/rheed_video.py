"""Result objects for RHEED video analysis."""

from __future__ import annotations

import warnings
from uuid import UUID

from monty.json import MSONable
from pandas import DataFrame

from atomscale.results import RHEEDImageResult
from atomscale.rheed_metadata import legacy_views_frame


class RHEEDVideoResult(MSONable):
    def __init__(
        self,
        data_id: UUID | str,
        timeseries_data: DataFrame,
        snapshot_image_data: list[RHEEDImageResult] | None,
        views: DataFrame | bool | None = None,
        collected_datetime: str | None = None,
        rotating: bool | None = None,
    ):
        """RHEED video result

        Args:
            data_id (UUID | str): Data ID for the entry in the data catalogue.
            timeseries_data (DataFrame): Pandas DataFrame with per-frame RHEED features, indexed
                against a "Time" column. Columns are capitalized labels such as "Cluster ID",
                "Specular Intensity", "Strain", "Cumulative Strain", "Oscillation Period",
                "Diffraction Spot Count", and "Lattice Spacing".
            snapshot_image_data (list[atomscale.results.rheed_image.RHEEDImageResult] | None): One
                :class:`atomscale.results.rheed_image.RHEEDImageResult` per snapshot extracted from the
                video, or None if no snapshots were extracted.
            views (DataFrame | None): Motion intervals and effective azimuth
                annotations. Omitted or None means no views are known, which
                reads as not rotating.
            collected_datetime (str | None): Datetime when the data was collected.
            rotating (bool | None): Deprecated. This argument held a bare
                rotating flag before rotation became a stored rate; it is
                accepted so callers written against that signature — and
                ``MSONable`` payloads serialized under it, which carry no
                ``views`` — still construct. It is translated to the placeholder
                view :func:`atomscale.rheed_metadata.legacy_views_frame` builds
                for the matching legacy type, so :attr:`rotating` answers the
                same. Ignored when ``views`` is given.
        """
        # The flag used to sit in the ``views`` position, so a caller passing it
        # positionally lands here. A DataFrame is never a bool, so this is
        # unambiguous.
        if isinstance(views, bool):
            views, rotating = None, views

        if views is None:
            if rotating is not None:
                warnings.warn(
                    "RHEEDVideoResult(rotating=...) is deprecated; pass the "
                    "`views` frame from Client.get_rheed_azimuths instead. "
                    "Rotation is now read from the stored per-view rpm.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            # An unrecognized type yields the same empty frame, so a result
            # built with neither argument is simply one with no views.
            views = legacy_views_frame(
                "rheed_rotating"
                if rotating
                else "rheed_stationary"
                if rotating is not None
                else ""
            )

        self.data_id = data_id
        self.timeseries_data = timeseries_data
        self.snapshot_image_data = snapshot_image_data
        self.views = views
        self.collected_datetime = collected_datetime

    @property
    def rotating(self) -> bool:
        """Whether the stage turned during this recording.

        Read from the stored per-view ``rpm`` rather than from the catalogue
        type, so a recording answers the same before and after the RHEED types
        were unified — ``rheed_stationary`` is simply rpm 0. Kept because
        callers relied on this attribute before rotation became a stored rate.

        A NaN rpm comes only from the legacy fallback in
        :func:`atomscale.rheed_metadata.legacy_views_frame` — a backend that
        recorded rotation as a type name and never a rate. That still reads as
        rotating.
        """
        if self.views.empty or "rpm" not in self.views:
            return False
        rpm = self.views["rpm"].astype("float64")
        return bool((rpm > 0).any() or rpm.isna().all())

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
