from contextlib import nullcontext

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)
from rich.table import Column
from rich.text import Text


def normalize_pixel_dimensions(
    points: npt.NDArray, image_shape: tuple[int, int]
) -> npt.NDArray:
    """
    Rescale pixel dimensions to a new image shape.

    Args:
        points (NDArray): Numpy array containing a list of points with columns width, height.
        image_shape (tuple[int, int]): Image shape.

    Returns:
        NDArray: Numpy array containing the rescaled points.
    """
    height, width = image_shape

    points[:, 0] = points[:, 0] / width
    points[:, 1] = points[:, 1] / height

    return points


def boxes_overlap(box1, box2) -> bool:
    """Check if two bounding boxes overlap

    Args:
        box1 (list[float]): List of xmin, ymin, xmax, ymax coordinates defining first box
        box2 (list[float]): List of xmin, ymin, xmax, ymax coordinates defining second box

    Returns:
        (bool): True if the boxes overlap
    """
    # Unpack coordinates
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    # Check for overlap
    return not (xmax1 < xmin2 or xmax2 < xmin1 or ymax1 < ymin2 or ymax2 < ymin1)


def regions_horizontal_overlapping(
    node_df: pd.DataFrame, start_node: int, end_node: int
) -> bool:
    """Check if two regions are horizontally overlapping"""
    start_node_row = node_df.loc[node_df["node_id"] == start_node].iloc[0]
    end_node_row = node_df.loc[node_df["node_id"] == end_node].iloc[0]

    left_node = (
        start_node_row
        if start_node_row["bbox_minc"] < end_node_row["bbox_minc"]
        else end_node_row
    )
    right_node = (
        start_node_row
        if start_node_row["bbox_minc"] > end_node_row["bbox_minc"]
        else end_node_row
    )
    left_node_max = left_node["bbox_maxc"]
    right_node_min = right_node["bbox_minc"]
    return left_node_max > right_node_min


def rescale_cartesian_coordinates(
    points: npt.NDArray, origin=(0, 0), scale: float = 1.0
) -> npt.NDArray:
    """
    Normalize radius in polar coordinates, then convert back to cartesian to get rescaled cartesian coordinates in image dimensions.
    Args:
        points (NDArray): Numpy array containing a list of points.
        origin (tuple[int, int]): Origin point.
        scale (float): Scaling number.
    Returns:
        NDArray: Numpy array containing the rescaled points.
    """

    # Convert the points to polar coordinates
    polar_coordinates = convert_to_polar_coordinates(points, origin=origin, scale=scale)

    scaled_1 = polar_coordinates[:, 0] * np.cos(polar_coordinates[:, 1])
    scaled_0 = polar_coordinates[:, 0] * np.sin(polar_coordinates[:, 1])

    return np.stack([scaled_0, scaled_1], axis=1)


def convert_to_polar_coordinates(
    points: npt.NDArray, origin=(0, 0), scale=1.0
) -> npt.NDArray:
    """
    Convert a set of 2D points to polar coordinates with radius and angle.

    Args:
        points (NDArray): Numpy array containing a list of points.
        origin (tuple[int, int]): Origin point.
        scale (float): Scaling number.
    """

    # Calculate the relative position of the points to the origin
    relative_points = points - origin

    # Calculate the radius and angle of the points
    intermediate = np.sum(np.square(relative_points), axis=1)
    radius = np.sqrt(intermediate) / scale
    angle = np.arctan2(relative_points[:, 1], relative_points[:, 0])

    # Stack the radius and angle into a single array
    return np.stack([radius, angle], axis=1)


def generate_graph_from_nodes(node_df: pd.DataFrame) -> nx.Graph:
    """Update a pattern graph with new node data from a DataFrame object"""

    pattern_graph = nx.Graph()

    for _, row in node_df.iterrows():
        node_id = row["node_id"]
        # Use all other columns as attributes
        attributes = row.drop("node_id").to_dict()
        pattern_graph.add_node(node_id, **attributes)

    edge_df = (
        node_df[["node_id", "centroid_1", "centroid_0"]]
        .copy(deep=True)
        .merge(
            node_df[["node_id", "centroid_1", "centroid_0"]].copy(deep=True),
            how="cross",
        )
    )
    edge_df = edge_df.loc[edge_df["node_id_x"] < edge_df["node_id_y"]]
    edge_df = edge_df.rename(
        columns={"node_id_x": "start_node", "node_id_y": "end_node"}
    )

    if len(edge_df) == 0:
        edge_df["horizontal_overlap"] = False
    else:
        edge_df["horizontal_overlap"] = edge_df.apply(
            lambda x: regions_horizontal_overlapping(
                node_df, x["start_node"], x["end_node"]
            ),
            axis=1,
        )

    edge_df["weight"] = np.sqrt(
        (edge_df["centroid_1_x"] - edge_df["centroid_1_y"]) ** 2
        + (edge_df["centroid_0_x"] - edge_df["centroid_0_y"]) ** 2
    )
    edge_df["horizontal_weight"] = np.abs(
        edge_df["centroid_1_x"] - edge_df["centroid_1_y"]
    )
    edge_df["vertical_weight"] = np.abs(
        edge_df["centroid_0_x"] - edge_df["centroid_0_y"]
    )
    edge_df = edge_df[
        [
            "start_node",
            "end_node",
            "weight",
            "horizontal_weight",
            "vertical_weight",
            "horizontal_overlap",
        ]
    ].copy()

    edge_df = edge_df.drop_duplicates(
        subset=["start_node", "end_node"], keep="first"
    ).reset_index(drop=True)

    pattern_graph.add_edges_from(edge_df[["start_node", "end_node"]].to_numpy())

    return pattern_graph


def _make_progress(mute: bool, transient: bool) -> Progress:
    """
    If `muted` is True return (nullcontext(), None),
    else return (progress, progress).

    Transient determines if it hides after completion.
    """
    if mute:
        return Progress(disable=True)

    class PercentOrTotal(ProgressColumn):
        """Render either % or completed/total depending on task flags."""

        _percent = TaskProgressColumn()

        def render(self, task) -> Text:
            if task.fields.get("show_percent", False):  # 42.0
                return self._percent.render(task)
            if task.fields.get("show_total", True):  # 12/37
                return Text(f"{int(task.completed)}/{int(task.total)}")  # type: ignore  # noqa: PGH003
            return Text("")  # blank cell

    class MaybeSpinner(SpinnerColumn):
        """Show the spinner only when task.fields['show_spinner'] is truthy."""

        def render(self, task) -> Text:
            if task.fields.get("show_spinner", True):
                return super().render(task)  # type: ignore  # noqa: PGH003
            return Text("")

    return Progress(
        MaybeSpinner(),
        TextColumn("[bold]{task.fields[pad]}{task.description}"),
        BarColumn(),
        PercentOrTotal(),
        TimeElapsedColumn(),
        transient=transient,
        refresh_per_second=30,
    )
