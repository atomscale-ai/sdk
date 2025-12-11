from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Literal

import pandas as pd
from pandas import DataFrame

from atomicds.core import BaseClient, ClientError, _FileSlice
from atomicds.core.utils import _make_progress, normalize_path
from atomicds.results import (
    PhotoluminescenceResult,
    RamanResult,
    RHEEDImageResult,
    RHEEDVideoResult,
    UnknownResult,
    XPSResult,
    _get_rheed_image_result,
)
from atomicds.results.group import PhysicalSampleResult, ProjectResult
from atomicds.timeseries.align import align_timeseries
from atomicds.timeseries.registry import get_provider

TimeseriesDomain = Literal["rheed", "optical", "metrology"]


class Client(BaseClient):
    """Atomic Data Sciences API client"""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        mute_bars: bool = False,
    ):
        """
        Args:
            api_key (str | None): API key. Explicit value takes precedence; if None, falls back to AS_API_KEY environment variable.
            endpoint (str): Root API endpoint. Explicit value takes precedence; if None, falls back to AS_API_ENDPOINT environment variable,
                defaulting to 'https://api.atomscale.ai/' if not set.
            mute_bars (bool): Whether to mute progress bars. Defaults to False.
        """

        if api_key is None:
            api_key = os.environ.get("AS_API_KEY")

        if endpoint is None:
            endpoint = os.environ.get("AS_API_ENDPOINT") or "https://api.atomscale.ai/"

        if api_key is None:
            raise ValueError("No valid ADS API key supplied")

        self.mute_bars = mute_bars

        super().__init__(api_key=api_key, endpoint=endpoint)

    def search(
        self,
        keywords: str | list[str] | None = None,
        include_organization_data: bool = True,
        data_ids: str | list[str] | None = None,
        physical_sample_ids: str | list[str] | None = None,
        project_ids: str | list[str] | None = None,
        data_type: Literal[
            "rheed_image",
            "rheed_stationary",
            "rheed_rotating",
            "xps",
            "photoluminescence",
            "pl",
            "raman",
            "recipe",
            "all",
        ] = "all",
        status: Literal[
            "success",
            "pending",
            "error",
            "running",
            "stream_active",
            "stream_interrupted",
            "stream_finalizing",
            "stream_error",
            "all",
        ] = "all",
        growth_length: tuple[int | None, int | None] = (None, None),
        upload_datetime: tuple[datetime | None, datetime | None] = (None, None),
        last_accessed_datetime: tuple[datetime | None, datetime | None] = (None, None),
    ) -> DataFrame:
        """Search and obtain data catalogue entries

        Args:
            keywords (str | list[str] | None): Keyword or list of keywords to search all data catalogue fields with.
                This searching is applied after all other explicit filters. Defaults to None.
            include_organization_data (bool): Whether to include catalogue entries from other users in
                your organization. Defaults to True.
            data_ids (str | list[str] | None): Data ID or list of data IDs. Defaults to None.
            physical_sample_ids (str | list[str] | None): Physical sample ID or list of IDs. Defaults to None.
            project_ids (str | list[str] | None): Project ID or list of IDs. Defaults to None.
            data_type (Literal["rheed_image", "rheed_stationary", "rheed_rotating", "xps", "photoluminescence", "raman", "all"]): Type of data. Defaults to "all".
            status (Literal["success", "pending", "error", "running", "all"]): Analyzed status of the data. Defaults to "all".
            growth_length (tuple[int | None, int | None]): Minimum and maximum values of the growth length in seconds.
                Defaults to (None, None) which will include all non-video data.
            upload_datetime (tuple[datetime | None, datetime | None]): Minimum and maximum values of the upload datetime.
                Defaults to (None, None).
            last_accessed_datetime (tuple[datetime | None, datetime | None]): Minimum and maximum values of the last accessed datetime.
                Defaults to (None, None).

        Returns:
            (DataFrame): Pandas DataFrame containing matched entries in the data catalogue.

        """
        params = {
            "keywords": keywords,
            "include_organization_data": include_organization_data,
            "data_ids": data_ids,
            "physical_sample_ids": physical_sample_ids,
            "project_ids": project_ids,
            "data_type": None if data_type == "all" else data_type,
            "status": status,
            "growth_length_min": growth_length[0],
            "growth_length_max": growth_length[1],
            "upload_datetime_min": upload_datetime[0],
            "upload_datetime_max": upload_datetime[1],
            "last_accessed_datetime_min": last_accessed_datetime[0],
            "last_accessed_datetime_max": last_accessed_datetime[1],
        }

        data = self._get(
            sub_url="data_entries/",
            params=params,
        )
        column_mapping = {
            "data_id": "Data ID",
            "upload_datetime": "Upload Datetime",
            "last_accessed_datetime": "Last Accessed Datetime",
            "char_source_type": "Type",
            "raw_name": "File Name",
            "pipeline_status": "Status",
            "raw_file_type": "File Type",
            "source_name": "Instrument Source",
            "sample_name": "Sample Name",
            "growth_length": "Growth Length",
            "physical_sample_id": "Physical Sample ID",
            "physical_sample_name": "Physical Sample Name",
            "detail_note_content": "Sample Notes",
            "detail_note_last_updated": "Sample Notes Last Updated",
            "file_metadata": "File Metadata",
            "tags": "Tags",
            "name": "Owner",
            "workspaces": "Workspaces",
            "project_ids": "Project ID",
            "project_names": "Project Name",
        }

        columns_to_drop = [
            "user_id",
            "synth_source_id",
            "sample_id",
            "processed_file_type",
            "bucket_file_name",
            "projects",
        ]
        catalogue = DataFrame(data)

        if "projects" in catalogue.columns:
            catalogue["project_ids"] = catalogue["projects"].apply(
                lambda projects: (projects[0].get("id") if projects else None)
            )
            catalogue["project_names"] = catalogue["projects"].apply(
                lambda projects: (projects[0].get("name") if projects else None)
            )

        if len(catalogue):
            if "detail_note_last_updated" in catalogue.columns:
                catalogue["detail_note_last_updated"] = catalogue[
                    "detail_note_last_updated"
                ].apply(lambda v: None if (pd.isna(v) or v == "NaT") else v)
            drop_cols = [col for col in columns_to_drop if col in catalogue.columns]
            catalogue = catalogue.drop(columns=drop_cols)

        catalogue = catalogue.rename(columns=column_mapping)

        desired_order = [
            "Data ID",
            "File Name",
            "Type",
            "Status",
            "File Type",
            "Instrument Source",
            "Sample Name",
            "Physical Sample ID",
            "Physical Sample Name",
            "Project ID",
            "Project Name",
            "Growth Length",
            "Upload Datetime",
            "Last Accessed Datetime",
            "Sample Notes",
            "Sample Notes Last Updated",
            "File Metadata",
            "Tags",
            "Owner",
            "Workspaces",
        ]
        ordered_cols = [col for col in desired_order if col in catalogue.columns] + [
            col for col in catalogue.columns if col not in desired_order
        ]

        return catalogue[ordered_cols]

    def get(
        self, data_ids: str | list[str]
    ) -> list[
        RHEEDVideoResult
        | RHEEDImageResult
        | XPSResult
        | PhotoluminescenceResult
        | RamanResult
        | UnknownResult
    ]:
        """Get analyzed data results

        Args:
            data_ids (str | list[str]): Data ID or list of data IDs from the data catalogue to obtain analyzed results for.

        Returns:
            list[atomicds.results.RHEEDVideoResult | atomicds.results.RHEEDImageResult | atomicds.results.XPSResult]:
                List of result objects

        """
        if isinstance(data_ids, str):
            data_ids = [data_ids]

        # Chunk requests to avoid overly long query strings
        data: list[dict] = []
        chunk_size = 100
        chunks = [
            data_ids[i : i + chunk_size]  # type: ignore[index]
            for i in range(0, len(data_ids), chunk_size)
        ]

        for chunk in chunks:
            chunk_data: list[dict] | dict | None = self._get(  # type: ignore[assignment]
                sub_url="data_entries/",
                params={
                    "data_ids": chunk,
                    "include_organization_data": True,
                },
            )
            if chunk_data:
                data.extend(
                    chunk_data if isinstance(chunk_data, list) else [chunk_data]
                )

        kwargs_list = []
        for entry in data:
            data_id = entry["data_id"]
            data_type = entry["char_source_type"]
            kwargs_list.append(
                {
                    "data_id": data_id,
                    "data_type": data_type,
                    "catalogue_entry": entry,
                }
            )

        # sort by submission order; this is important to match external labels
        kwargs_list = sorted(kwargs_list, key=lambda x: data_ids.index(x["data_id"]))

        with _make_progress(self.mute_bars, False) as progress:
            return self._multi_thread(
                self._get_result_data,
                kwargs_list,
                progress,
                progress_description="Obtaining data results",
            )

    def list_physical_samples(self) -> DataFrame:
        """List physical samples available to the user."""
        data = self._get(sub_url="physical_samples/")
        if data is None:
            return DataFrame(None)
        samples = DataFrame(data)

        if "projects" in samples.columns:
            samples["project_id"] = samples["projects"].apply(
                lambda projects: (projects[0].get("id") if projects else None)
            )
            samples["project_name"] = samples["projects"].apply(
                lambda projects: (projects[0].get("name") if projects else None)
            )

        if "detail_notes" in samples.columns:
            samples["detail_note_content"] = samples["detail_notes"].apply(
                lambda note: note.get("content") if isinstance(note, dict) else None
            )
            samples["detail_note_last_updated"] = samples["detail_notes"].apply(
                lambda note: note.get("last_updated")
                if isinstance(note, dict)
                else None
            )
            samples["detail_note_last_updated"] = samples[
                "detail_note_last_updated"
            ].apply(lambda v: None if (pd.isna(v) or v == "NaT") else v)

        if "target_material" in samples.columns:
            samples["target_material"] = samples["target_material"].apply(
                lambda tm: {
                    k: tm.get(k)
                    for k in ("substrate", "sample_name")
                    if isinstance(tm, dict) and k in tm
                }
                if isinstance(tm, dict)
                else tm
            )

        columns_to_drop = [
            "sample_id",
            "detail_notes_id",
            "user_id",
            "growth_instrument_id",
            "version",
            "owner_id",
            "projects",
            "detail_notes",
        ]

        column_mapping = {
            "physical_sample_metadata": "Sample Metadata",
            "name": "Physical Sample Name",
            "last_updated": "Last Updated",
            "created_datetime": "Created Datetime",
            "id": "Physical Sample ID",
            "owner_name": "Owner",
            "target_material": "Target Material",
            "growth_instrument": "Growth Instrument",
            "num_data_items": "Data Items",
            "project_id": "Project ID",
            "project_name": "Project Name",
            "detail_note_content": "Sample Notes",
            "detail_note_last_updated": "Sample Notes Last Updated",
        }

        if len(samples):
            drop_cols = [col for col in columns_to_drop if col in samples.columns]
            samples = samples.drop(columns=drop_cols)

        samples = samples.rename(columns=column_mapping)

        desired_order = [
            "Physical Sample ID",
            "Physical Sample Name",
            "Project ID",
            "Project Name",
            "Target Material",
            "Sample Metadata",
            "Sample Notes",
            "Sample Notes Last Updated",
            "Growth Instrument",
            "Data Items",
            "Created Datetime",
            "Last Updated",
            "Owner",
        ]
        ordered_cols = [col for col in desired_order if col in samples.columns] + [
            col for col in samples.columns if col not in desired_order
        ]

        return samples[ordered_cols]

    def list_projects(self) -> DataFrame:
        """List projects available to the user."""
        data = self._get(sub_url="projects/")
        if data is None:
            return DataFrame(None)
        projects = DataFrame(data)

        if "detail_note" in projects.columns:
            projects["detail_note_content"] = projects["detail_note"].apply(
                lambda note: note.get("content") if isinstance(note, dict) else None
            )
            projects["detail_note_last_updated"] = projects["detail_note"].apply(
                lambda note: note.get("last_updated")
                if isinstance(note, dict)
                else None
            )
            projects["detail_note_last_updated"] = projects[
                "detail_note_last_updated"
            ].apply(lambda v: None if (pd.isna(v) or v == "NaT") else v)

        columns_to_drop = [
            "owner_id",
            "detail_note",
        ]

        column_mapping = {
            "id": "Project ID",
            "last_updated": "Last Updated",
            "name": "Project Name",
            "physical_sample_count": "Physical Sample Count",
            "owner_name": "Owner",
            "detail_note_content": "Project Notes",
            "detail_note_last_updated": "Project Notes Last Updated",
        }

        if len(projects):
            drop_cols = [col for col in columns_to_drop if col in projects.columns]
            projects = projects.drop(columns=drop_cols)

        projects = projects.rename(columns=column_mapping)

        desired_order = [
            "Project ID",
            "Project Name",
            "Physical Sample Count",
            "Project Notes",
            "Project Notes Last Updated",
            "Last Updated",
            "Owner",
        ]
        ordered_cols = [col for col in desired_order if col in projects.columns] + [
            col for col in projects.columns if col not in desired_order
        ]

        return projects[ordered_cols]

    def get_physical_sample(
        self,
        physical_sample_id: str,
        *,
        include_organization_data: bool = True,
        align: bool | str = False,
        resample: str | None = None,
    ) -> PhysicalSampleResult:
        """Get all data for a physical sample.

        Args:
            physical_sample_id: Identifier of the physical sample.
            include_organization_data: Whether to include organization data. Defaults to True.
            align: Whether to align timeseries data. If truthy, an aligned DataFrame is returned.
            resample: Optional pandas resample rule applied after alignment.
        """
        physical_samples: list[dict] | None = self._get(  # type: ignore  # noqa: PGH003
            sub_url="physical_samples/",
            params={"physical_sample_id": physical_sample_id},
        )
        if physical_samples and not isinstance(physical_samples, list):
            physical_samples = [physical_samples]

        entries: list[dict] | None = self._get(  # type: ignore  # noqa: PGH003
            sub_url="data_entries/",
            params={
                "physical_sample_ids": [physical_sample_id],
                "include_organization_data": include_organization_data,
            },
        )
        if entries and not isinstance(entries, list):
            entries = [entries]

        data_ids = [e["data_id"] for e in entries] if entries else []

        results = self.get(data_ids=data_ids) if data_ids else []
        join_how = "outer"
        if isinstance(align, str):
            join_how = align

        ts_aligned = (
            align_timeseries(results, how=join_how, resample=resample)
            if align
            else None
        )

        non_timeseries = [
            r
            for r in results
            if not hasattr(r, "timeseries_data") or r.timeseries_data is None
        ]
        sample_name = (
            physical_samples[0].get("name")
            if physical_samples
            else entries[0].get("physical_sample_name")
            if entries
            else None
        )
        sample_id = physical_sample_id
        return PhysicalSampleResult(
            physical_sample_id=sample_id,
            physical_sample_name=sample_name,
            data_results=results,
            aligned_timeseries=ts_aligned,
            non_timeseries=non_timeseries,
        )

    def get_project(
        self,
        project_id: str,
        *,
        include_organization_data: bool = True,
        align: bool | str = False,
        resample: str | None = None,
    ) -> ProjectResult:
        """Get all data grouped by physical sample for a project.

        Args:
            project_id: Identifier of the project.
            include_organization_data: Whether to include organization data. Defaults to True.
            align: Whether to align timeseries at the project level. Defaults to False.
            resample: Optional pandas resample rule applied after alignment.
        """
        # Get physical samples associated with the project, then fetch data per sample.
        project_samples: list[dict] = (
            self._get(sub_url=f"projects/{project_id}/physical_samples") or []
        )
        if not project_samples:
            return ProjectResult(project_id, None, [], None)

        sample_results: list[PhysicalSampleResult] = []
        for sample in project_samples:
            sid = sample.get("id")
            if not sid:
                continue
            sample_results.append(
                self.get_physical_sample(
                    sid,
                    include_organization_data=include_organization_data,
                    align=align,
                    resample=resample,
                )
            )

        project_aligned = None
        if align:
            frames = []
            for sample in sample_results:
                if sample.aligned_timeseries is None:
                    continue
                renamed = sample.aligned_timeseries.copy()
                renamed.columns = pd.MultiIndex.from_tuples(
                    [
                        (sample.physical_sample_id, *tuple(col))
                        if isinstance(col, tuple)
                        else (sample.physical_sample_id, col)
                        for col in renamed.columns
                    ]
                )
                frames.append(renamed)
            if frames:
                project_aligned = frames[0]
                for frame in frames[1:]:
                    project_aligned = project_aligned.join(frame, how="outer")

        project_name = None
        return ProjectResult(
            project_id=project_id,
            project_name=project_name,
            samples=sample_results,
            aligned_timeseries=project_aligned,
        )

    def _get_result_data(
        self,
        data_id: str,
        data_type: Literal[
            "xps",
            "photoluminescence",
            "pl",
            "raman",
            "rheed_image",
            "rheed_stationary",
            "rheed_rotating",
            "rheed_xscan",
            "metrology",
            "recipe",
            "optical",
        ],
        catalogue_entry: dict[str, Any] | None = None,
    ) -> (
        RHEEDVideoResult
        | RHEEDImageResult
        | XPSResult
        | PhotoluminescenceResult
        | RamanResult
        | UnknownResult
        | None
    ):
        if data_type == "xps":
            result: dict = self._get(sub_url=f"xps/{data_id}")  # type: ignore  # noqa: PGH003

            return XPSResult(
                data_id=data_id,
                xps_id=result["xps_id"],
                binding_energies=result["binding_energies"],
                intensities=result["intensities"],
                predicted_composition=result["predicted_composition"],
                detected_peaks=result["detected_peaks"],
                elements_manually_set=bool(result["set_elements"]),
            )

        if data_type in ("photoluminescence", "pl"):
            result = (
                self._get(  # type: ignore  # noqa: PGH003
                    sub_url=f"photoluminescence/{data_id}"
                )
                or {}
            )
            return PhotoluminescenceResult(
                data_id=data_id,
                photoluminescence_id=result.get(
                    "photoluminescence_id", result.get("id")
                ),
                energies=result.get("energies", []),
                intensities=result.get("intensities", []),
                detected_peaks=result.get("detected_peaks", {}),
                last_updated=result.get("last_updated"),
            )

        if data_type == "raman":
            result = self._get(sub_url=f"raman/{data_id}") or {}  # type: ignore  # noqa: PGH003
            return RamanResult(
                data_id=data_id,
                raman_id=result.get("raman_id", result.get("id")),
                raman_shift=result.get("energies", result.get("wavenumbers", [])),
                intensities=result.get("intensities", []),
                detected_peaks=result.get("detected_peaks", {}),
                last_updated=result.get("last_updated"),
            )

        if data_type == "rheed_image":
            return _get_rheed_image_result(self, data_id)

        if data_type in [
            "rheed_stationary",
            "rheed_rotating",
            "rheed_xscan",
            "metrology",
            "recipe",
            "optical",
        ]:
            # recipe timeseries are served from the metrology endpoint; reuse that provider.
            timeseries_type = (
                "rheed"
                if "rheed" in data_type
                else "metrology"
                if data_type == "recipe"
                else data_type
            )
            provider = get_provider(timeseries_type)

            # Get timeseries data
            raw = provider.fetch_raw(self, data_id)
            ts_df = provider.to_dataframe(raw)

            return provider.build_result(self, data_id, data_type, ts_df)

        # Fallback for unknown/unsupported data types
        return UnknownResult(
            data_id=data_id,
            data_type=data_type,
            catalogue_entry=catalogue_entry,
        )

    def upload(self, files: list[str | BinaryIO]):
        """Upload and process files

        Args:
            files (list[str | BinaryIO]): List containing string paths to files, or BinaryIO objects from `open`.
        """
        chunk_size = 40 * 1024 * 1024  # 40 MiB

        # Check to make sure list is valid and get pre-signed URL nums
        file_data = []
        for file in files:
            if isinstance(file, str):
                path = normalize_path(file)
                if not (path.exists() and path.is_file()):
                    raise ClientError(f"{path} is not a file or does not exist")

                # Calculate number of URLs needed for this file
                file_size = path.stat().st_size
                num_urls = -(-file_size // chunk_size)  # Ceiling division
                file_name = path.name

            else:
                # Handle BinaryIO objects
                file.seek(0, 2)  # Seek to the end of the file
                file_size = file.tell()
                file.seek(0)  # Seek back to the beginning of the file
                num_urls = -(-file_size // chunk_size)  # Ceiling division
                file_name = file.name

            file_data.append(
                {
                    "num_urls": num_urls,
                    "file_name": file_name,
                    "file_size": file_size,
                    "file_path": file,
                }
            )

        def __upload_file(
            file_info: dict[
                Literal["num_urls", "file_name", "file_size", "file_path"], int | str
            ],
        ):
            url_data: list[dict[str, str | int]] = self._post_or_put(
                method="POST",
                sub_url="data_entries/raw_data/staged/upload_urls/",
                params={
                    "original_filename": file_info["file_name"],
                    "num_parts": file_info["num_urls"],
                    "staging_type": "core",
                },
            )  # type: ignore  # noqa: PGH003

            # Iterate through data structure above and upload file using multi-part S3 urls. Multithread appropriately.
            # build kwargs_list using only serializable bits:
            kwargs_list = []
            for part in url_data:
                part_no = int(part["part"]) - 1
                offset = part_no * chunk_size
                length = min(chunk_size, int(file_info["file_size"]) - offset)  # type: ignore  # noqa: PGH003
                kwargs_list.append(
                    {
                        "method": "PUT",
                        "sub_url": "",
                        "params": None,
                        "base_override": part["url"],
                        "file_path": file_info["file_path"],
                        "offset": offset,
                        "length": length,
                    }
                )

            def __upload_chunk(
                method: Literal["PUT", "POST"],
                sub_url: str,
                params: dict[str, Any] | None,
                base_override: str,
                file_path: Path,
                offset: int,
                length: int,
            ) -> Any:
                slice_obj = _FileSlice(file_path, offset, length)
                return self._post_or_put(
                    method=method,
                    sub_url=sub_url,
                    params=params,
                    body=slice_obj,  # type: ignore  # noqa: PGH003
                    deserialize=False,
                    return_headers=True,
                    base_override=base_override,
                    headers={
                        "Content-Length": str(length),
                    },
                )

            etag_data = self._multi_thread(
                __upload_chunk,
                kwargs_list=kwargs_list,
                progress_bar=progress,
                progress_description=f"[red]{file_info['file_name']}",
                progress_kwargs={
                    "show_percent": True,
                    "show_total": False,
                    "show_spinner": False,
                    "pad": "",
                },
                transient=True,
            )

            # Complete multipart upload *only* if the backend issued an upload_id
            first_part = url_data[0]
            upload_id = first_part.get("upload_id")
            if upload_id:
                etag_body = [
                    {"ETag": entry["ETag"], "PartNumber": i + 1}
                    for i, entry in enumerate(etag_data)
                ]
                self._post_or_put(
                    method="POST",
                    sub_url="data_entries/raw_data/staged/upload_urls/complete/",
                    params={"staging_type": "core"},
                    body={
                        "upload_id": upload_id,
                        "new_filename": first_part["new_filename"],
                        "etag_data": etag_body,
                    },
                )

        main_task = None
        file_count = len(file_data)
        with _make_progress(self.mute_bars, False) as progress:
            if not progress.disable:
                main_task = progress.add_task(
                    "Uploading files…",
                    total=file_count,
                    show_percent=False,
                    show_total=True,
                    show_spinner=True,
                    pad="",
                )

            max_workers = min(8, len(file_data))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(__upload_file, file_info): file_info  # type: ignore  # noqa: PGH003
                    for file_info in file_data
                }
                for future in as_completed(futures):
                    future.result()  # raise early if anything went wrong
                    if main_task is not None:
                        progress.update(main_task, advance=1, refresh=True)

    def download_videos(
        self,
        data_ids: str | list[str],
        dest_dir: str | Path | None = None,
        data_type: Literal["raw", "processed"] = "processed",
    ):
        """
        Download processed RHEED videos to disk.

        Args:
            data_ids (str | list[str]): One or more data IDs from the data catalogue.
            dest_dir (str | Path | None): Directory to write the files to.
                Defaults to the current working directory.
            data_type (Literal["raw", "processed"]): Whether to download raw or processed data.
        """
        chunk_size: int = 20 * 1024 * 1024  # 20 MiB read chunks

        # Normalise inputs
        if isinstance(data_ids, str):
            data_ids = [data_ids]
        if dest_dir is None:
            dest_dir = Path.cwd()
        else:
            dest_dir = Path(dest_dir).expanduser().resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        def __download_one(data_id: str) -> None:
            # 1) Resolve the presigned URL -------------------------------------
            url_type = "raw_data" if data_type == "raw" else "processed_data"
            meta: dict = self._get(  # type: ignore  # noqa: PGH003
                sub_url=f"data_entries/{url_type}/{data_id}",
                params={"return_as": "url-download"},
            )
            if meta is None:
                raise ClientError(f"No processed data found for data_id '{data_id}'")

            url = meta["url"]
            file_name = (
                meta.get("file_name") or f"{data_id}.{meta.get('file_format', 'mp4')}"
            )
            target = dest_dir / file_name  # type: ignore # noqa: PGH003

            # 2) Open the stream *once* (HEAD not allowed)
            with self._session.get(  # type: ignore  # noqa: PGH003
                url, stream=True, allow_redirects=True, timeout=30
            ) as resp:
                resp.raise_for_status()

                # Attempt to read the size from **this** GET response
                total_size = int(resp.headers.get("Content-Length", 0))

                # 3) Create a nested bar for this file
                if total_size:  # we know the size → percent bar
                    bar_id = progress.add_task(
                        f"[red]{file_name}",
                        total=total_size,
                        show_percent=True,
                        show_total=False,
                        show_spinner=False,
                        pad="",
                    )
                else:  # unknown size → indeterminate spinner
                    bar_id = progress.add_task(
                        f"[red]{file_name}",
                        total=None,
                        show_percent=False,
                        show_total=False,
                        show_spinner=True,
                        pad="",
                    )

                # 4) Stream the bytes to disk with updates
                with Path.open(target, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size):
                        if chunk:  # filter out keep-alive
                            fh.write(chunk)
                            progress.update(bar_id, advance=len(chunk))

        # Download files
        with _make_progress(self.mute_bars, False) as progress:
            # master bar
            master_task = None
            if not progress.disable:
                master_task = progress.add_task(
                    "Downloading videos…",
                    total=len(data_ids),
                    show_percent=False,
                    show_total=True,
                    show_spinner=True,
                    pad="",
                )

            # thread-pool for concurrent downloads
            max_workers = min(8, len(data_ids))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(__download_one, did): did for did in data_ids}
                for fut in as_completed(futures):
                    # propagate any exceptions early
                    fut.result()
                    if master_task is not None:
                        progress.update(master_task, advance=1, refresh=True)
