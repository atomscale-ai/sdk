from __future__ import annotations

from io import BytesIO
from uuid import UUID

import networkx as nx
import numpy as np
import pandas as pd
import trackpy as tp
from monty.json import MSONable
from networkx import Graph
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import ImageDraw
from PIL.Image import Image
from pycocotools import mask as mask_util

from atomicds.core import BaseClient, boxes_overlap, generate_graph_from_nodes

tp.quiet()
# test comment


class RHEEDImageResult(MSONable):
    def __init__(
        self,
        data_id: UUID | str,
        processed_data_id: UUID | str,
        processed_image: Image,
        mask: NDArray | None,
        pattern_graph: Graph | None,
        metadata: dict | None = None,
    ):
        """RHEED image result

        Args:
            data_id (UUID | str): Data ID for the entry in the data catalogue.
            processed_data_id (UUID | str): Processed data ID for the entry in the catalogue.
            processed_image (Image): Processed image data in a PIL Image format.
            mask (NDArray | None): Array containing binary segmentation mask.
            pattern_graph (Graph | None): NetworkX Graph object for the extracted diffraction pattern.
            metadata (dict): Generic metadata (e.g. timestamp, cluster_id, etc...).
        """

        metadata = metadata or {}

        self.data_id = data_id
        self.processed_data_id = processed_data_id
        self.processed_image = processed_image
        self.mask = mask
        self.pattern_graph = pattern_graph
        self.metadata = metadata

    def get_plot(
        self,
        show_mask: bool = True,
        show_spot_nodes: bool = True,
        symmetrize: bool = False,
        alpha: float = 0.2,
    ) -> Image:
        """Get diffraction pattern image with optional overlays

        Args:
            show_mask (bool): Whether to show mask overlay of identified pattern. Defaults to True.
            show_spot_nodes (bool): Whether to show identified diffraction node overlays. Defaults to True.

        Returns:
            (Image): PIL Image object with optional overlays

        """
        image = self.processed_image.copy().convert("RGBA")
        draw = ImageDraw.Draw(image)

        if symmetrize and self.pattern_graph is not None:
            node_df = pd.DataFrame.from_dict(
                dict(self.pattern_graph.nodes(data=True)), orient="index"
            )
            _, pattern_graph = self._symmetrize(node_df)
        else:
            pattern_graph = self.pattern_graph

        if pattern_graph:
            for node_id, node_data in pattern_graph.nodes.data():  # type: ignore  # noqa: PGH003
                if show_spot_nodes:
                    # Draw nodes
                    y = node_data.get("centroid_0")
                    x = node_data.get("centroid_1")

                    if x and y:
                        center = (x, y)
                        radius = 0.02 * max(image.width, image.height)
                        color = (255, 0, 0, 255) if node_id > 0 else (0, 255, 0, 255)

                        draw.ellipse(
                            (
                                center[0] - radius,
                                center[1] - radius,
                                center[0] + radius,
                                center[1] + radius,
                            ),
                            fill=color,
                        )

                        draw.text(
                            xy=(center[0] - (radius / 2), center[1] - radius),
                            text=str(node_id),
                            fill=(255, 255, 255, 255),
                        )

            if show_mask and self.mask is not None:
                overlay = np.zeros((*self.mask.shape, 4), dtype=np.uint8)
                overlay[np.where(self.mask)] = [255, 0, 0, int(alpha * (255))]

                overlay_image = PILImage.fromarray(overlay)
                image.paste(overlay_image, mask=overlay_image)

        return image

    def get_laue_zero_radius(self) -> tuple[float, tuple[float, float]]:
        """
        Get the radius of the zeroth order Laue zone. Note that the data is symmetrized across the
        vertical axis before the Laue zone is searched for.

        Returns:
            (tuple[float, tuple[float, float]]): Tuple containing the best fit radius and center point.
        """

        node_data = []
        if self.pattern_graph is not None:
            node_data.append(
                pd.DataFrame.from_dict(
                    dict(self.pattern_graph.nodes(data=True)), orient="index"
                )
            )

        node_df = pd.concat(node_data, axis=0).reset_index(drop=True)
        node_df, _ = self._symmetrize(node_df)

        image_array = np.array(self.processed_image)

        thetas = np.linspace(0, 2 * np.pi, 1440)
        radiis = np.linspace(100, 1400, 2000)

        intersection_point = (
            node_df.loc[node_df["node_id"] == node_df["node_id"].min()][
                ["intensity_centroid_0", "intensity_centroid_1"]
            ]
            .to_numpy()
            .squeeze()
            + node_df.loc[node_df["node_id"] == node_df["node_id"].min()][
                ["specular_origin_0", "specular_origin_1"]
            ]
            .to_numpy()
            .squeeze()
        )

        result_matrix = []

        for r0 in radiis:
            x_center, y_center = intersection_point[1], intersection_point[0] - r0
            x_points = r0 * np.cos(thetas) + x_center
            y_points = r0 * np.sin(thetas) + y_center
            int_x_points = np.round(x_points).astype(int)
            int_y_points = np.round(y_points).astype(int)
            coords = np.hstack(
                [int_x_points.reshape(-1, 1), int_y_points.reshape(-1, 1)]
            )
            coords = coords[coords[:, 1] >= 0]
            coords = coords[coords[:, 0] >= 0]
            coords = coords[coords[:, 1] < image_array.shape[0]]
            coords = coords[coords[:, 0] < image_array.shape[1]]
            if len(coords) == 0:
                continue
            intensities = image_array[coords[:, 1], coords[:, 0]]
            result_matrix.append(np.quantile(intensities, 0.9))

        best_fit_radius = np.mean(radiis[np.argsort(result_matrix)][-30:])

        return best_fit_radius, [intersection_point[0] - best_fit_radius, x_center]  # type: ignore  # noqa: PGH003

    def get_pattern_dataframe(
        self,
        extra_data: dict | None = None,
        symmetrize: bool = False,
        return_as_features: bool = False,
    ) -> pd.DataFrame:
        """Featurize the RHEED image collection into a dataframe of node features and edge features.

        Args:
            extra_data (dict | None): Dictionary containing field names and values of extra data to be included in the DataFrame object.
                Defaults to None.
            symmetrize (bool): Whether to symmetrize the data across the vertical axis. Defaults to False.
            return_as_features (bool): Whether to return a feature-foward version of the DataFrame. Defaults to False.

        Returns:
            (DataFrame): Pandas DataFrame object of RHEED node and edge features.
        """

        extra_data = extra_data or {}

        node_feature_cols = [
            "spot_area",
            "streak_area",
            "relative_centroid_0",
            "relative_centroid_1",
            "intensity_centroid_0",
            "intensity_centroid_1",
            "specular_origin_0",
            "specular_origin_1",
            "fwhm_0",
            "fwhm_1",
            "area",
            "eccentricity",
            "center_distance",
            "axis_major_length",
            "axis_minor_length",
            "roughness_metric",
        ]

        # TODO: add edge features
        # edge_feature_cols = ["weight", "horizontal_weight", "vertical_weight", "horizontal_overlap"]

        node_data = []
        if self.pattern_graph is not None:
            node_data.append(
                pd.DataFrame.from_dict(
                    dict(self.pattern_graph.nodes(data=True)), orient="index"
                )
            )

        node_df = pd.concat(node_data, axis=0).reset_index(drop=True)

        if symmetrize:
            node_df, _ = self._symmetrize(node_df)

        extra_data_df = pd.DataFrame.from_records(
            [
                {"data_id": self.data_id, "processed_data_id": self.processed_data_id}
                | extra_data
            ]
        )

        node_df["data_id"] = self.data_id
        feature_df: pd.DataFrame = node_df.pivot_table(
            index="data_id", columns="node_id", values=node_feature_cols
        )

        feature_df.columns = feature_df.columns.to_flat_index()

        feature_df = feature_df.merge(
            extra_data_df, left_index=True, right_on="data_id", how="inner"
        )
        feature_df = feature_df.rename(
            columns={col: (col, "") for col in extra_data_df.columns}
        )
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns)  # type: ignore  # noqa: PGH003

        keep_cols = node_feature_cols + list(extra_data.keys())

        if return_as_features:
            return feature_df[keep_cols]  # type: ignore  # noqa: PGH003

        return node_df  # type: ignore  # noqa: PGH003

    @staticmethod
    def _symmetrize(node_df: pd.DataFrame):
        """Symmetrize a DataFrame object containing RHEED image node data"""

        def reflect_mask(
            mask_obj: str, height: int, width: int, origin: float
        ) -> str | bytes:
            """Reflect a list of RLE masks across the vertical axis"""

            mask_array: np.ndarray = mask_util.decode(
                {"counts": mask_obj, "size": (height, width)}  # type: ignore  # noqa: PGH003
            )
            origin = int(np.round(origin, 0))
            num_cols_to_mirror = mask_array.shape[1] - origin

            reflected_array = mask_array.copy()

            # For columns before the origin position, swap them with their mirrored counterparts
            for i in range(origin, origin - num_cols_to_mirror, -1):
                swap_index = origin + (origin - i)
                if swap_index < mask_array.shape[1]:
                    reflected_array[:, i], reflected_array[:, swap_index] = (
                        mask_array[:, swap_index].copy(),
                        mask_array[:, i].copy(),
                    )

            return mask_util.encode(np.asfortranarray(reflected_array))["counts"]

        def merge_masks(masks: list[str], height, width) -> str:
            """Merge a list of RLE masks using logical OR"""
            mask_objs = [
                mask_util.decode({"counts": mm, "size": (height, width)})  # type: ignore  # noqa: PGH003
                for mm in masks
            ]
            merged_mask = np.asfortranarray(np.logical_or.reduce(mask_objs))
            return mask_util.encode(merged_mask)  # type: ignore  # noqa: PGH003

        def merge_overlaps(node_df):
            """Merge overlapping nodes in a DataFrame object. Use recursively until no overlaps remain."""

            first_row = node_df.iloc[[0]]
            original_dtypes = first_row.dtypes

            agg_dict = {
                "centroid_0": "mean",
                "centroid_1": "mean",
                "intensity_centroid_0": "mean",
                "intensity_centroid_1": "mean",
                "relative_centroid_0": "mean",
                "relative_centroid_1": "mean",
                "relative_intensity_centroid_0": "mean",
                "relative_intensity_centroid_1": "mean",
                "bbox_minc": "mean",
                "bbox_minr": "mean",
                "bbox_maxc": "mean",
                "bbox_maxr": "mean",
                "area": "mean",
                "node_id": "min",
                "pattern_id": lambda x: x.iloc[0],
                "specular_origin_0": "mean",
                "specular_origin_1": "mean",
                "center_distance": "mean",
                "fwhm_0": "mean",
                "fwhm_1": "mean",
                "mask_rle": lambda x: x.tolist(),
                "mask_width": lambda x: x.iloc[0],
                "mask_height": lambda x: x.iloc[0],
                "eccentricity": "mean",
                "axis_major_length": "mean",
                "axis_minor_length": "mean",
                "bbox_intensity": "mean",
                "spot_area": "mean",
                "streak_area": "mean",
                "roughness_metric": "mean",
                "distances": "mean",
            }

            new_df = pd.DataFrame()
            for i in range(len(node_df)):
                current_bbox = node_df.iloc[i][
                    ["bbox_minc", "bbox_minr", "bbox_maxc", "bbox_maxr"]
                ]
                overlapping_nodes = node_df[
                    node_df.apply(
                        lambda row, current_bbox=current_bbox: boxes_overlap(
                            current_bbox,
                            row[["bbox_minc", "bbox_minr", "bbox_maxc", "bbox_maxr"]],
                        ),
                        axis=1,
                    )
                ]

                merged_row = overlapping_nodes.agg(agg_dict)

                new_mask = merge_masks(  # type: ignore  # noqa: PGH003
                    merged_row["mask_rle"],
                    merged_row["mask_height"],
                    merged_row["mask_width"],
                )["counts"]
                merged_row["mask_rle"] = new_mask

                new_df = pd.concat([new_df, merged_row], axis=1)

            new_df = new_df.T.astype(original_dtypes).reset_index(drop=True)

            agg_dict["mask_rle"] = lambda x: merge_masks(  # type: ignore  # noqa: PGH003
                x,
                new_df["mask_height"].iloc[0],  # type: ignore  # noqa: PGH003
                new_df["mask_width"].iloc[0],  # type: ignore  # noqa: PGH003
            )[
                "counts"
            ]

            new_df = new_df.groupby("node_id").agg(agg_dict).reset_index(drop=True)

            # relabel node_id > 1000 to monotonically increase from the largest ID < 1000
            while new_df["node_id"].max() > 1000:
                max_id = new_df.loc[new_df["node_id"] < 1000]["node_id"].max()
                new_df.loc[new_df["node_id"] == new_df["node_id"].max(), "node_id"] = (
                    max_id + 1
                )

            return new_df

        reflection_plane = node_df["specular_origin_1"].mean()

        left_nodes = node_df.loc[node_df["centroid_1"] < reflection_plane]
        right_nodes = node_df.loc[node_df["centroid_1"] > reflection_plane]

        # TODO: The repeat code here can be condensed.
        left_to_right = left_nodes.copy()
        left_to_right["centroid_1"] = reflection_plane + (
            reflection_plane - left_to_right["centroid_1"]
        )
        left_to_right["intensity_centroid_1"] = -left_to_right["intensity_centroid_1"]
        left_to_right["relative_centroid_1"] = -left_to_right["relative_centroid_1"]
        left_to_right["mask_rle"] = left_to_right["mask_rle"].apply(
            lambda x: reflect_mask(
                x,
                left_to_right["mask_height"].iloc[0],
                left_to_right["mask_width"].iloc[0],
                left_to_right["specular_origin_1"].iloc[0],
            )
        )

        new_max = (
            reflection_plane + (reflection_plane - left_to_right["bbox_minc"])
        ).astype(int)
        new_min = (
            reflection_plane + (reflection_plane - left_to_right["bbox_maxc"])
        ).astype(int)

        left_to_right["bbox_maxc"] = new_max
        left_to_right["bbox_minc"] = new_min

        left_to_right["node_id"] = left_to_right["node_id"] + 1000

        right_to_left = right_nodes.copy()
        right_to_left["centroid_1"] = reflection_plane - (
            right_to_left["centroid_1"] - reflection_plane
        )
        right_to_left["intensity_centroid_1"] = -right_to_left["intensity_centroid_1"]
        right_to_left["relative_centroid_1"] = -right_to_left["relative_centroid_1"]
        right_to_left["mask_rle"] = right_to_left["mask_rle"].apply(
            lambda x: reflect_mask(
                x,
                right_to_left["mask_height"].iloc[0],
                right_to_left["mask_width"].iloc[0],
                right_to_left["specular_origin_1"].iloc[0],
            )
        )

        new_max = (
            reflection_plane - (right_to_left["bbox_minc"] - reflection_plane)
        ).astype(int)
        new_min = (
            reflection_plane - (right_to_left["bbox_maxc"] - reflection_plane)
        ).astype(int)

        right_to_left["bbox_minc"] = new_min
        right_to_left["bbox_maxc"] = new_max
        right_to_left["node_id"] = right_to_left["node_id"] + 2000

        node_df = pd.concat(
            [node_df, left_to_right, right_to_left], axis=0
        ).reset_index(drop=True)

        if node_df.empty:
            return node_df

        new_df = merge_overlaps(node_df)
        while len(new_df) != len(node_df):
            node_df = new_df.copy(deep=True)  # type: ignore  # noqa: PGH003
            new_df = merge_overlaps(node_df)

        new_pattern_graph = generate_graph_from_nodes(new_df)  # type: ignore  # noqa: PGH003

        return new_df, new_pattern_graph


# TODO: Add tests for RHEEDImageCollection
class RHEEDImageCollection(MSONable):
    def __init__(
        self,
        rheed_images: list[RHEEDImageResult],
        extra_data: list[dict] | None = None,
        sort_key: str | None = None,
    ):
        """Collection of RHEED images

        Args:
            rheed_images (list[RHEEDImageResult]): List of RHEEDImageResult objects.
            extra_data (list[dict] | None): List of dictionaries containing field names and values of extra data to be included in the DataFrame object.
                Defaults to None.
            sort_key (str | None): Key used to sort the data with.
        """

        self._extra_data = extra_data or []  # type: ignore  # noqa: PGH003

        if len(self._extra_data) > 0 and len(self._extra_data) != len(rheed_images):
            raise ValueError(
                "List of extra data must be the same length as the RHEED image collection."
            )

        self._sort_key = sort_key

        if self._sort_key is not None:
            sorted_indices = self._sort_by_extra_data_key(self._sort_key)
        else:
            sorted_indices = list(range(len(rheed_images)))

        for idx, rheed_image in enumerate(rheed_images):
            if rheed_image.pattern_graph:
                for node in rheed_image.pattern_graph.nodes:
                    rheed_image.pattern_graph.nodes[node]["pattern_id"] = idx

        self._rheed_images = [rheed_images[idx] for idx in sorted_indices]
        self._extra_data = (
            [self._extra_data[idx] for idx in sorted_indices]
            if len(self._extra_data)
            else self._extra_data
        )

    @property
    def rheed_images(self):
        return self._rheed_images

    @property
    def extra_data(self):
        return self._extra_data

    @property
    def sort_key(self):
        return self._sort_key

    def align_fingerprints(
        self,
        node_df: pd.DataFrame | None = None,
        inplace: bool = False,
        search_range=0.2,
    ) -> RHEEDImageCollection:
        """
        Align a collection of RHEED fingerprints by relabeling the nodes to connect the same scattering
        features across RHEED patterns, based on relative position to the center feature.

        Returns:
            (tuple[DataFrame, list[RHEEDImageResult]): Pandas DataFrame object with aligned RHEED fingerprint data
        """

        image_scales = [
            rheed_image.processed_image.size for rheed_image in self.rheed_images
        ]
        image_scale = np.amax(image_scales, axis=0)

        if node_df is None:
            extra_iter = (
                self.extra_data if self.extra_data else [None] * len(self.rheed_images)
            )
            node_dfs = [
                rheed_image.get_pattern_dataframe(
                    extra_data=extra_data, symmetrize=False, return_as_features=False
                )
                for rheed_image, extra_data in zip(
                    self.rheed_images, extra_iter, strict=False
                )
            ]
            node_df = pd.concat(node_dfs, axis=0).reset_index(drop=True)

        labels, _ = pd.factorize(node_df["data_id"])
        node_df["pattern_id"] = labels

        linked_df = tp.link(
            f=node_df,
            search_range=np.sqrt(np.sum(np.square(image_scale))) * search_range,
            memory=1,
            t_column="pattern_id",
            pos_columns=["relative_centroid_1", "relative_centroid_0"],
        )

        rheed_images: list[RHEEDImageResult] = []
        splits = [group for _, group in linked_df.groupby("pattern_id")]
        for split, rheed_image in zip(splits, self.rheed_images, strict=False):
            mapping = dict(zip(split["node_id"], split["particle"], strict=False))
            # don't relabel the specular 0 node.
            if 0 in mapping:
                mapping.pop(0)

            rheed_image.pattern_graph = nx.relabel_nodes(  # type: ignore  # noqa: PGH003
                rheed_image.pattern_graph,  # type: ignore  # noqa: PGH003
                mapping,
            )
            for node in rheed_image.pattern_graph.nodes:
                rheed_image.pattern_graph.nodes[node]["node_id"] = node

            rheed_images.append(rheed_image)

        if inplace:
            self._rheed_images = rheed_images

        return self.__class__(rheed_images, self.extra_data, self.sort_key)  # linked_df

    def get_pattern_dataframe(
        self,
        streamline: bool = True,
        normalize: bool = True,
        symmetrize: bool = False,
        return_as_features: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Featurize the RHEED image collection into a dataframe of node features and edge features.

        Args:
            streamline (bool): Whether to streamline the DataFrame object and remove null values. Defaults to True.
            normalize (bool): Whether to min/max normalize the feature data across all images. Defaults to True.
            symmetrize (bool): Whether to symmetrize the RHEEED images and segmented patterns about the vertical axis before
                obtaining the DataFrame representation. Defaults to False.
            return_as_features (bool): Whether to return the final feature-forward DataFrame. Defaults to True.

        Returns:
            (DataFrame): Pandas DataFrame object of RHEED node and edge features.
        """

        node_feature_cols = [
            "spot_area",
            "streak_area",
            "relative_centroid_0",
            "relative_centroid_1",
            "intensity_centroid_0",
            "intensity_centroid_1",
            "specular_origin_0",
            "specular_origin_1",
            "fwhm_0",
            "fwhm_1",
            "area",
            "eccentricity",
            "center_distance",
            "axis_major_length",
            "axis_minor_length",
            "roughness_metric",
        ]

        # TODO: add edge features
        # edge_feature_cols = ["weight", "horizontal_weight", "vertical_weight", "horizontal_overlap"]

        image_iter = (
            zip(self.rheed_images, self.extra_data, strict=False)
            if self.extra_data
            else zip(self.rheed_images, [None] * len(self.rheed_images), strict=False)
        )

        dfs = [
            rheed_image.get_pattern_dataframe(
                extra_data=extra_data,
                symmetrize=symmetrize,
                return_as_features=return_as_features,
            )
            for rheed_image, extra_data in image_iter
        ]

        data_df = pd.concat(dfs, axis=0).reset_index(drop=True)

        keep_cols = node_feature_cols + list(
            {key for extra_data in self.extra_data for key in extra_data}
        )

        if return_as_features:
            data_df = data_df[keep_cols]

            if streamline:
                data_df = data_df.dropna(axis=1)

            if normalize:
                for col in node_feature_cols:
                    data_df[col] = (data_df[col] - data_df[col].mean()) / (
                        data_df[col].std() + 1e-6
                    )

        return data_df  # type: ignore  # noqa: PGH003

    def _sort_by_extra_data_key(self, key: str):
        """Sort the RHEEDImageCollection by an extra data key"""
        if key not in self.extra_data[0]:
            raise ValueError(
                f"Extra data key {key} not found in the extra data dictionary."
            )

        sort_order = [extra_data[key] for extra_data in self.extra_data]
        return np.argsort(sort_order)

        # self._rheed_images = [self.rheed_images[idx] for idx in sorted_indices]
        # self._extra_data = [self.extra_data[idx] for idx in sorted_indices]

    def __getitem__(self, key: int | slice) -> RHEEDImageResult | RHEEDImageCollection:
        if isinstance(key, int):
            return self.rheed_images[key]

        if isinstance(key, slice):
            return self.__class__(
                self.rheed_images[key], self.extra_data[key], self.sort_key
            )

        return self.__class__(
            self.rheed_images[key],  # type: ignore  # noqa: PGH003
            self.extra_data[key],  # type: ignore  # noqa: PGH003
            self.sort_key,
        )

    def __len__(self) -> int:
        return len(self.rheed_images)


def _get_rheed_image_result(
    client: BaseClient, data_id: str, metadata: dict | None = None
):
    # Get pattern graph data
    metadata = metadata or {}
    graph_data = client._get(sub_url=f"rheed/images/{data_id}/fingerprint")
    graph = (
        nx.node_link_graph(
            graph_data["fingerprint"],  # type: ignore #noqa: PGH003
            source="start_id",  # type: ignore #noqa: PGH003
            target="end_id",  # type: ignore #noqa: PGH003
            link="horizontal_distances",  # type: ignore #noqa: PGH003
        )
        if graph_data
        else None
    )
    processed_data_id = (
        graph_data.get("processed_data_id") if graph_data is not None else data_id  # type: ignore #noqa: PGH003
    )

    # Get mask data
    mask_data: dict = client._get(sub_url=f"rheed/images/{data_id}/mask")  # type: ignore  #noqa: PGH003
    mask_rle = mask_data.get("mask_rle")
    mask_array = None

    if mask_data is not None and mask_rle is not None:
        mask_height = mask_data["mask_height"]
        mask_width = mask_data["mask_width"]

        mask_dict = {
            "counts": mask_rle,
            "size": (mask_height, mask_width),
        }
        mask_array = mask_util.decode(mask_dict)  # type: ignore  # noqa: PGH003

    # Get raw and processed image data
    image_download: dict[str, str] | None = client._get(  # type: ignore  # noqa: PGH003
        sub_url=f"data_entries/processed_data/{data_id}",
        params={"return_as": "url-download"},
    )

    if image_download is None:
        return None

    # Image is pulled from the S3 pre-signed URL
    image_bytes: bytes = client._get(  # type: ignore  # noqa: PGH003
        base_override=image_download["url"], sub_url="", deserialize=False
    )

    image_data = PILImage.open(BytesIO(image_bytes))

    return RHEEDImageResult(
        data_id=data_id,
        processed_data_id=processed_data_id,  # type: ignore  # noqa: PGH003
        processed_image=image_data,
        mask=mask_array,
        pattern_graph=graph,
        metadata=metadata,
    )
