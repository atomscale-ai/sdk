"""Main client for interacting with the Atomscale API."""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Literal

import pandas as pd
from pandas import DataFrame
from requests.exceptions import RequestException

from atomscale.core import BaseClient, ClientError, _FileSlice
from atomscale.core.utils import _make_progress, normalize_path
from atomscale.results import (
    ChangepointResult,
    EllipsometryResult,
    MetrologyResult,
    OpticalResult,
    PhotoluminescenceResult,
    RamanResult,
    RHEEDImageResult,
    RHEEDVideoResult,
    SimilarityTrajectoryResult,
    UnknownResult,
    XPSResult,
    XRDResult,
    _get_rheed_image_result,
)
from atomscale.results.group import PhysicalSampleResult, ProjectResult
from atomscale.timeseries.align import align_timeseries
from atomscale.timeseries.registry import get_provider

TimeseriesDomain = Literal["rheed", "optical", "metrology", "tool_state"]

_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


def _retry_client_call(
    fn: Callable[..., Any],
    *args: Any,
    attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    **kwargs: Any,
) -> Any:
    """Retry a `_post_or_put`/`_get` style call on transient errors.

    Retries on `ClientError` whose `status_code` is in `_RETRYABLE_STATUSES`
    and on transport-level `RequestException`s (connection drops,
    chunked-encoding errors, etc.). Uses exponential backoff capped at
    `max_delay`. Re-raises the last exception when `attempts` is exhausted.
    """
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except ClientError as exc:
            if exc.status_code not in _RETRYABLE_STATUSES or i == attempts - 1:
                raise
        except RequestException:
            if i == attempts - 1:
                raise
        time.sleep(min(base_delay * (2**i), max_delay))
    # Defensive: loop above always returns or raises, but appease type checkers.
    raise RuntimeError("unreachable")  # pragma: no cover


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
            api_key = os.environ.get("AS_API_KEY", os.environ.get("ATOMSCALE_API_KEY"))

        if endpoint is None:
            endpoint = os.environ.get("AS_API_ENDPOINT") or "https://api.atomscale.ai/"

        if api_key is None:
            raise ValueError("No valid Atomscale API key supplied")

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
            "xrd",
            "photoluminescence",
            "pl",
            "raman",
            "recipe",
            "optical",
            "metrology",
            "tool_state",
            "ellipsometry",
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
        last_updated: tuple[datetime | None, datetime | None] = (None, None),
        last_accessed_datetime: tuple[datetime | None, datetime | None] | None = None,
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
            data_type (Literal["rheed_image", "rheed_stationary", "rheed_rotating", "xps", "xrd", "photoluminescence", "raman", "recipe", "optical", "metrology", "ellipsometry", "all"]): Type of data. Defaults to "all".
            status (Literal["success", "pending", "error", "running", "all"]): Analyzed status of the data. Defaults to "all".
            growth_length (tuple[int | None, int | None]): Minimum and maximum values of the growth length in seconds.
                Defaults to (None, None) which will include all non-video data.
            upload_datetime (tuple[datetime | None, datetime | None]): Minimum and maximum values of the upload datetime.
                Defaults to (None, None).
            last_updated (tuple[datetime | None, datetime | None]): Minimum and maximum values of the last
                updated datetime. Defaults to (None, None).
            last_accessed_datetime: Deprecated alias for ``last_updated``; will be removed in a future release.

        Returns:
            (DataFrame): Pandas DataFrame containing matched entries in the data catalogue.

        """
        if last_accessed_datetime is not None:
            warnings.warn(
                "Client.search(last_accessed_datetime=...) is deprecated; "
                "use last_updated=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            last_updated = last_accessed_datetime

        params = {
            "keywords": keywords,
            "include_organization_data": include_organization_data,
            "data_ids": data_ids,
            "physical_sample_ids": physical_sample_ids,
            "project_ids": project_ids,
            # Map the legacy "metrology" filter to its renamed "tool_state"
            # char-source so it resolves against the current backend, which
            # validates this param as an enum that no longer accepts "metrology".
            "data_type": (
                None
                if data_type == "all"
                else "tool_state"
                if data_type == "metrology"
                else data_type
            ),
            "status": status,
            "growth_length_min": growth_length[0],
            "growth_length_max": growth_length[1],
            "upload_datetime_min": upload_datetime[0],
            "upload_datetime_max": upload_datetime[1],
            "last_updated_min": last_updated[0],
            "last_updated_max": last_updated[1],
        }

        data = self._get(
            sub_url="data_entries/",
            params=params,
        )
        column_mapping = {
            "data_id": "Data ID",
            "upload_datetime": "Upload Datetime",
            "last_updated": "Last Updated",
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
            "sha3_256": "sha256",
            "collected_datetime": "Collected Datetime",
            "has_instrument_logs": "Has Instrument Logs",
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
                lambda projects: projects[0].get("id") if projects else None
            )
            catalogue["project_names"] = catalogue["projects"].apply(
                lambda projects: projects[0].get("name") if projects else None
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
            "Last Updated",
            "Sample Notes",
            "Sample Notes Last Updated",
            "File Metadata",
            "Tags",
            "Owner",
            "Workspaces",
            "sha256",
            "Collected Datetime",
            "Has Instrument Logs",
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
        | XRDResult
        | PhotoluminescenceResult
        | RamanResult
        | OpticalResult
        | MetrologyResult
        | EllipsometryResult
        | UnknownResult
    ]:
        """Get analyzed data results

        Args:
            data_ids (str | list[str]): Data ID or list of data IDs from the data catalogue to obtain analyzed results for.

        Returns:
            list[atomscale.results.RHEEDVideoResult | atomscale.results.RHEEDImageResult | atomscale.results.XPSResult | atomscale.results.XRDResult]:
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

    def get_changepoints(
        self,
        data_ids: str | list[str],
        latest_only: bool = True,
        detection_method: (
            Literal["forecasting", "clustering", "intensity_profile"] | None
        ) = "intensity_profile",
        severity: Literal["info", "warning", "critical"] | None = "critical",
        as_dataframe: bool = True,
    ) -> DataFrame | list[ChangepointResult]:
        """Get changepoint detection records for one or more data IDs.

        Args:
            data_ids (str | list[str]): Data ID or list of data IDs from the data catalogue.
            latest_only (bool): If True (default), only return changepoints from the most
                recently completed detection run for each (data_id, detection_method) pair.
                If False, return all changepoints from every historical run.
            detection_method (str | None): Filter to a single detection method. One of
                "forecasting", "clustering", "intensity_profile". Defaults to "intensity_profile".
                Pass None to include all detection methods.
            severity (str | None): Filter to a single severity level. One of "info",
                "warning", "critical". Defaults to "critical". Pass None to include all
                severities.
            as_dataframe (bool): If True (default) return a pandas DataFrame. If False
                return a list of ChangepointResult objects.

        Returns:
            DataFrame | list[ChangepointResult]: Changepoint records matching the filters.
        """
        if isinstance(data_ids, str):
            data_ids = [data_ids]

        records: list[dict] = []
        chunk_size = 100
        chunks = [
            data_ids[i : i + chunk_size] for i in range(0, len(data_ids), chunk_size)
        ]

        for chunk in chunks:
            payload: dict | None = self._get(  # type: ignore[assignment]
                sub_url="changepoints/",
                params={"data_ids": chunk, "latest_only": latest_only},
            )
            if payload:
                records.extend(payload.get("changepoints", []))

        if detection_method is not None:
            records = [
                r for r in records if r.get("detection_method") == detection_method
            ]
        if severity is not None:
            records = [r for r in records if r.get("severity") == severity]

        if as_dataframe:
            # Keep only the label itself; drop label provenance/metadata fields.
            _drop = {
                "label_category",
                "label_notes",
                "label_source",
                "label_confidence",
                "labeled_at",
                "labeled_by_user_id",
                "similar_neighbor_ids",
            }
            rows = [
                {
                    **{k: v for k, v in r.items() if k not in _drop},
                    "label": r.get("label_category"),
                }
                for r in records
            ]
            return DataFrame(rows)
        return [ChangepointResult.from_api(r) for r in records]

    def get_similarity_trajectory(
        self,
        source_id: str,
        *,
        workflow: str = "rheed_stationary",
        last_n: int | None = None,
        window_span: float | None = None,
        reference_ids: list[str] | None = None,
        softmax_mode: str | None = None,
        reference_n_values: int | None = None,
    ) -> SimilarityTrajectoryResult:
        """Fetch a one-shot similarity trajectory for a source data_id or physical_sample_id.

        Args:
            source_id: Data ID or physical sample ID the trajectory is computed against.
            workflow: Similarity workflow name (e.g. "rheed_stationary"). Defaults to
                "rheed_stationary".
            last_n: If set, only fetch the last N points of the trajectory.
            window_span: Optional window span parameter forwarded to the provider.
            reference_ids: Optional list of reference data IDs to compare against.
            softmax_mode: Optional softmax mode forwarded to the provider.
            reference_n_values: Optional number of reference values forwarded to the provider.

        Returns:
            SimilarityTrajectoryResult with the populated timeseries DataFrame.
        """
        provider = get_provider("similarity_trajectory")
        kwargs: dict[str, Any] = {"workflow": workflow}
        if last_n is not None:
            kwargs["last_n"] = last_n
        if window_span is not None:
            kwargs["window_span"] = window_span
        if reference_ids is not None:
            kwargs["reference_ids"] = reference_ids
        if softmax_mode is not None:
            kwargs["softmax_mode"] = softmax_mode
        if reference_n_values is not None:
            kwargs["reference_n_values"] = reference_n_values

        raw = provider.fetch_raw(self, source_id, **kwargs)
        ts_df = provider.to_dataframe(raw)
        return provider.build_result(
            self,
            source_id,
            "similarity_trajectory",
            ts_df,
            workflow=workflow,
            window_span=window_span or 0.0,
        )

    def iter_poll_similarity_trajectory(
        self,
        source_id: str,
        *,
        interval: float = 1.0,
        last_n: int | None = None,
        **kwargs: Any,
    ) -> Iterator[DataFrame]:
        """Synchronously poll similarity trajectory data, yielding DataFrames.

        Thin wrapper around :func:`atomscale.similarity.iter_poll_trajectory`.
        See that function for the full set of keyword arguments
        (`distinct_by`, `until`, `max_polls`, `fire_immediately`, `jitter`,
        `on_error`).
        """
        from atomscale.similarity import polling as _similarity_polling

        return _similarity_polling.iter_poll_trajectory(
            self, source_id, interval=interval, last_n=last_n, **kwargs
        )

    def aiter_poll_similarity_trajectory(
        self,
        source_id: str,
        *,
        interval: float = 1.0,
        last_n: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[DataFrame]:
        """Asynchronously poll similarity trajectory data without blocking the loop.

        Thin wrapper around :func:`atomscale.similarity.aiter_poll_trajectory`.
        """
        from atomscale.similarity import polling as _similarity_polling

        return _similarity_polling.aiter_poll_trajectory(
            self, source_id, interval=interval, last_n=last_n, **kwargs
        )

    def start_polling_similarity_trajectory_thread(
        self,
        source_id: str,
        *,
        interval: float = 1.0,
        last_n: int | None = None,
        on_result: Callable[[DataFrame], None],
        **kwargs: Any,
    ) -> threading.Event:
        """Start polling similarity trajectory data in a background daemon thread.

        Returns a :class:`threading.Event` that can be set to stop polling.
        """
        from atomscale.similarity import polling as _similarity_polling

        return _similarity_polling.start_polling_trajectory_thread(
            self,
            source_id,
            interval=interval,
            last_n=last_n,
            on_result=on_result,
            **kwargs,
        )

    def start_polling_similarity_trajectory_task(
        self,
        source_id: str,
        *,
        interval: float = 1.0,
        last_n: int | None = None,
        on_result: Callable[[DataFrame], Any] | None = None,
        **kwargs: Any,
    ) -> asyncio.Task[None]:
        """Start polling similarity trajectory data as an :class:`asyncio.Task`."""
        from atomscale.similarity import polling as _similarity_polling

        return _similarity_polling.start_polling_trajectory_task(
            self,
            source_id,
            interval=interval,
            last_n=last_n,
            on_result=on_result,
            **kwargs,
        )

    def list_physical_samples(self) -> DataFrame:
        """List physical samples available to the user."""
        data = self._get(sub_url="physical_samples/")
        if data is None:
            return DataFrame(None)
        samples = DataFrame(data)

        if "projects" in samples.columns:
            samples["project_id"] = samples["projects"].apply(
                lambda projects: projects[0].get("id") if projects else None
            )
            samples["project_name"] = samples["projects"].apply(
                lambda projects: projects[0].get("name") if projects else None
            )

        if "detail_notes" in samples.columns:
            samples["detail_note_content"] = samples["detail_notes"].apply(
                lambda note: note.get("content") if isinstance(note, dict) else None
            )
            samples["detail_note_last_updated"] = samples["detail_notes"].apply(
                lambda note: (
                    note.get("last_updated") if isinstance(note, dict) else None
                )
            )
            samples["detail_note_last_updated"] = samples[
                "detail_note_last_updated"
            ].apply(lambda v: None if (pd.isna(v) or v == "NaT") else v)

        if "target_material" in samples.columns:
            samples["target_material"] = samples["target_material"].apply(
                lambda tm: (
                    {
                        k: tm.get(k)
                        for k in ("substrate", "sample_name")
                        if isinstance(tm, dict) and k in tm
                    }
                    if isinstance(tm, dict)
                    else tm
                )
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
                lambda note: (
                    note.get("last_updated") if isinstance(note, dict) else None
                )
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
    ) -> PhysicalSampleResult:
        """Get all data for a physical sample.

        Args:
            physical_sample_id: Identifier of the physical sample.
            include_organization_data: Whether to include organization data. Defaults to True.
            align: Whether to align timeseries data. If truthy, an aligned DataFrame is returned.
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

        ts_aligned = align_timeseries(results, how=join_how) if align else None

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
    ) -> ProjectResult:
        """Get all data grouped by physical sample for a project.

        Args:
            project_id: Identifier of the project.
            include_organization_data: Whether to include organization data. Defaults to True.
            align: Whether to align timeseries at the project level. Defaults to False.
        """
        # Get physical samples associated with the project, then fetch data per sample.
        project_samples: list[dict] = (
            self._get(sub_url=f"projects/{project_id}/physical_samples") or []
        )
        if not project_samples:
            return ProjectResult(project_id, None, [], None)

        sample_results: list[PhysicalSampleResult] = []
        all_results: list = []
        for sample in project_samples:
            sid = sample.get("id")
            if not sid:
                continue
            # For project-level alignment we align once across all entries, so
            # skip per-sample alignment when align=True.
            sample_align = False if align else align
            sample_results.append(
                self.get_physical_sample(
                    sid,
                    include_organization_data=include_organization_data,
                    align=sample_align,
                )
            )
            if sample_results[-1].data_results:
                all_results.extend(sample_results[-1].data_results)

        project_aligned = None
        if align:
            project_aligned = align_timeseries(all_results, how="outer")

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
            "xrd",
            "photoluminescence",
            "pl",
            "raman",
            "rheed_image",
            "rheed_stationary",
            "rheed_rotating",
            "rheed_xscan",
            "metrology",
            "tool_state",
            "recipe",
            "optical",
            "ellipsometry",
        ],
        catalogue_entry: dict[str, Any] | None = None,
    ) -> (
        RHEEDVideoResult
        | RHEEDImageResult
        | XPSResult
        | PhotoluminescenceResult
        | RamanResult
        | XRDResult
        | OpticalResult
        | MetrologyResult
        | EllipsometryResult
        | UnknownResult
        | None
    ):
        collected_dt = (
            catalogue_entry.get("collected_datetime") if catalogue_entry else None
        )

        if data_type == "xps":
            result: dict = self._get(sub_url=f"xps/{data_id}") or {}  # type: ignore  # noqa: PGH003

            return XPSResult(
                data_id=data_id,
                xps_id=result.get("xps_id"),
                binding_energies=result.get("binding_energies", []),
                intensities=result.get("intensities", []),
                predicted_composition=result.get("predicted_composition", {}),
                detected_peaks=result.get("detected_peaks", {}),
                elements_manually_set=bool(result.get("set_elements", False)),
                collected_datetime=collected_dt,
            )

        if data_type == "xrd":
            result = self._get(sub_url=f"xrd/{data_id}") or {}  # type: ignore  # noqa: PGH003
            return XRDResult(
                data_id=data_id,
                xrd_id=result.get("id"),
                two_theta=result.get("two_theta", []),
                intensities=result.get("intensities", []),
                detected_peaks=result.get("detected_peaks", []),
                wavelength_angstrom=result.get("wavelength_angstrom", 1.5406),
                two_theta_unit=result.get("two_theta_unit", "degrees"),
                spectral_metadata=result.get("spectral_metadata", {}),
                last_updated=result.get("last_updated"),
                collected_datetime=collected_dt,
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
                collected_datetime=collected_dt,
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
                collected_datetime=collected_dt,
            )

        if data_type == "rheed_image":
            result_obj = _get_rheed_image_result(self, data_id)
            if result_obj is not None:
                result_obj.collected_datetime = collected_dt
            return result_obj

        if data_type in [
            "rheed_stationary",
            "rheed_rotating",
            "rheed_xscan",
            "metrology",
            "tool_state",
            "recipe",
            "optical",
            "ellipsometry",
        ]:
            # Each data_type resolves to its own provider/endpoint. "metrology" is
            # the legacy char-source value for "tool_state"; both map to the
            # tool-state provider. recipe is distinct (its own /recipe endpoint).
            timeseries_type = "rheed" if "rheed" in data_type else data_type
            provider = get_provider(timeseries_type)

            # Get timeseries data
            raw = provider.fetch_raw(self, data_id)
            ts_df = provider.to_dataframe(raw)

            result_obj = provider.build_result(self, data_id, data_type, ts_df)
            if catalogue_entry:
                # Store upload datetime for alignment fallback when only relative time is available.
                upload_dt = catalogue_entry.get("upload_datetime")
                if upload_dt:
                    result_obj.upload_datetime = upload_dt
            result_obj.collected_datetime = collected_dt
            return result_obj

        # Fallback for unknown/unsupported data types
        warnings.warn(
            f"Unrecognized data_type '{data_type}' for data_id '{data_id}'; "
            "returning UnknownResult. The SDK may be out of date — consider upgrading.",
            stacklevel=3,
        )
        return UnknownResult(
            data_id=data_id,
            data_type=data_type,
            catalogue_entry=catalogue_entry,
            collected_datetime=collected_dt,
        )

    _UUID_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
    )

    def _resolve_physical_sample(self, physical_sample: str) -> tuple[str, str]:
        """Resolve a physical sample name or UUID to (id, name).

        If UUID: look up existing sample, error if not found.
        If name: case-insensitive match, auto-create if not found.

        Returns:
            Tuple of (physical_sample_id, physical_sample_name).
        """
        physical_sample = physical_sample.strip()
        if not physical_sample:
            raise ClientError("physical_sample cannot be empty")

        samples_df = self.list_physical_samples()

        if self._UUID_RE.match(physical_sample):
            match = samples_df[samples_df["Physical Sample ID"] == physical_sample]
            if match.empty:
                raise ClientError(
                    f"Physical sample with id '{physical_sample}' not found"
                )
            return physical_sample, match.iloc[0]["Physical Sample Name"]

        # Name lookup: case-insensitive exact match
        names_lower = samples_df["Physical Sample Name"].str.strip().str.lower()
        mask = names_lower == physical_sample.lower()
        match = samples_df[mask]

        if not match.empty:
            return match.iloc[0]["Physical Sample ID"], match.iloc[0][
                "Physical Sample Name"
            ]

        # Not found — create a new physical sample
        resp: dict = self._post_or_put(  # type: ignore  # noqa: PGH003
            method="POST",
            sub_url="physical_samples/",
            body={"name": physical_sample},
        )
        return resp["id"], physical_sample

    def _resolve_project(self, project: str) -> tuple[str, str]:
        """Resolve a project name or UUID to (id, name).

        Unlike ``_resolve_physical_sample``, this does **not** auto-create
        missing projects (no SDK-exposed create endpoint). Raises ``ClientError``
        if the project cannot be found.
        """
        project = project.strip()
        if not project:
            raise ClientError("project cannot be empty")

        projects_df = self.list_projects()
        if not len(projects_df):
            raise ClientError(f"Project '{project}' not found")

        if self._UUID_RE.match(project):
            match = projects_df[projects_df["Project ID"] == project]
            if match.empty:
                raise ClientError(f"Project with id '{project}' not found")
            return project, match.iloc[0]["Project Name"]

        names_lower = projects_df["Project Name"].str.strip().str.lower()
        mask = names_lower == project.lower()
        match = projects_df[mask]
        if match.empty:
            raise ClientError(f"Project '{project}' not found")
        return match.iloc[0]["Project ID"], match.iloc[0]["Project Name"]

    def _add_sample_to_project(
        self, project_id: str, physical_sample_id: str, set_active: bool = True
    ) -> None:
        """POST a physical sample onto a project's tracking-samples list.

        Mirrors the streamer-side
        ``POST /projects/{id}/configuration/tracking_samples`` call.
        """
        _retry_client_call(
            self._post_or_put,
            method="POST",
            sub_url=f"projects/{project_id}/configuration/tracking_samples",
            body={
                "physical_sample_id": physical_sample_id,
                "set_active": set_active,
            },
        )

    def upload(
        self,
        files: list[str | BinaryIO],
        physical_sample: str | None = None,
        project: str | None = None,
    ) -> list[str]:
        """Upload and process files.

        Args:
            files (list[str | BinaryIO]): List containing string paths to files, or BinaryIO objects from ``open``.
            physical_sample (str | None): Physical sample name or UUID to link uploads to.
                If a name is given and no matching sample exists, one is created automatically.
            project (str | None): Project name or UUID to associate the uploads with. The
                project must already exist (the SDK does not auto-create projects). When
                provided, ``physical_sample`` is required so the sample can be added to
                the project's tracking list via
                ``POST /projects/{id}/configuration/tracking_samples``.

        Returns:
            list[str]: Data IDs assigned to the uploaded files.
        """
        chunk_size = 40 * 1024 * 1024  # 40 MiB

        if project is not None and physical_sample is None:
            raise ClientError(
                "`project` requires `physical_sample` so the sample can be added "
                "to the project's tracking list."
            )

        # Resolve physical sample before uploading so we fail fast on bad input
        metadata_body: dict[str, str] | None = None
        ps_id: str | None = None
        if physical_sample is not None:
            ps_id, ps_name = self._resolve_physical_sample(physical_sample)
            metadata_body = {
                "physical_sample_id": ps_id,
                "physical_sample_name": ps_name,
            }

        # Resolve project up-front so we fail fast on a bad project name/UUID
        project_id: str | None = None
        if project is not None:
            project_id, _ = self._resolve_project(project)
            if ps_id is None:  # guarded above; defensive against future refactors
                raise ClientError(
                    "`project` requires `physical_sample` so the sample can be added "
                    "to the project's tracking list."
                )
            self._add_sample_to_project(project_id, ps_id)

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
        ) -> str:
            url_data: list[dict[str, str | int]] = _retry_client_call(
                self._post_or_put,
                method="POST",
                sub_url="data_entries/raw_data/staged/upload_urls/",
                params={
                    "original_filename": file_info["file_name"],
                    "num_parts": file_info["num_urls"],
                    "staging_type": "core",
                },
                body=metadata_body,
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
                _retry_client_call(
                    self._post_or_put,
                    method="POST",
                    sub_url="data_entries/raw_data/staged/upload_urls/complete/",
                    params={"staging_type": "core"},
                    body={
                        "upload_id": upload_id,
                        "new_filename": first_part["new_filename"],
                        "etag_data": etag_body,
                    },
                )

            return str(first_part["data_id"])

        data_ids: list[str] = []
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
                    data_ids.append(future.result())
                    if main_task is not None:
                        progress.update(main_task, advance=1, refresh=True)

        return data_ids

    def download(
        self,
        data_ids: str | list[str],
        dest_dir: str | Path | None = None,
        data_type: Literal["raw", "processed"] = "processed",
    ):
        """
        Download raw or processed files for any data type to disk.

        Works for every data_type the platform stores (RHEED video, XPS / XRD /
        PL / Raman / optical / metrology / ellipsometry / etc.) — the underlying
        ``data_entries/{raw_data|processed_data}/{data_id}`` endpoint is
        data-type-agnostic and returns whatever file format the backend has on
        record.

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
            try:
                meta: dict = self._get(  # type: ignore  # noqa: PGH003
                    sub_url=f"data_entries/{url_type}/{data_id}",
                    params={"return_as": "url-download"},
                )
            except ClientError as err:
                if err.status_code in {400, 410, 422}:
                    raise ClientError(
                        f"No processed data found for data_id '{data_id}'"
                    ) from err
                raise
            if not meta or "url" not in meta:
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
                    "Downloading files…",
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

    def download_videos(
        self,
        data_ids: str | list[str],
        dest_dir: str | Path | None = None,
        data_type: Literal["raw", "processed"] = "processed",
    ):
        """Deprecated alias for :meth:`download`. Kept for backwards compatibility."""
        warnings.warn(
            "Client.download_videos is deprecated; use Client.download instead. "
            "It is data-type-agnostic and works for every supported file type.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.download(data_ids=data_ids, dest_dir=dest_dir, data_type=data_type)

    # -------------------------------------------------------------------------
    # Growth Instrument Management
    # -------------------------------------------------------------------------

    def list_growth_instruments(self) -> list[dict[str, Any]]:
        """List all growth instruments accessible by the user.

        Returns instruments within the user's organization.

        Returns:
            list[dict]: List of instruments with keys including:
                - synth_source_id (int): Unique instrument ID
                - source_name (str): Display name
                - synth_source_type (str): Instrument type (mbe, cvd, etc.)
                - source_manufacturer (str | None): Manufacturer name
                - source_model (str | None): Model name

        Example:
            >>> instruments = client.list_growth_instruments()
            >>> for inst in instruments:
            ...     print(f"{inst['synth_source_id']}: {inst['source_name']}")
        """
        result = self._get(sub_url="instruments/synthesis")
        return result if result else []

    def create_growth_instrument(
        self,
        label: str,
        name: str,
        instrument_type: Literal["mbe", "cvd", "pvd", "sputter", "ald", "pld"],
        serial_id: str | None = None,
    ) -> int:
        """Create a new growth instrument.

        Args:
            label: Display name for the instrument (e.g., "Main MBE").
            name: Manufacturer and model (e.g., "Veeco GEN10").
            instrument_type: Type of instrument.
            serial_id: Optional serial number or identifier.

        Returns:
            int: The synth_source_id of the created instrument.

        Example:
            >>> instrument_id = client.create_growth_instrument(
            ...     label="Main MBE",
            ...     name="Veeco GEN10",
            ...     instrument_type="mbe",
            ...     serial_id="SN-12345",
            ... )
        """
        body = {
            "label": label,
            "name": name,
            "type": instrument_type,
            "serial_id": serial_id,
        }
        result = self._post_or_put(
            method="POST",
            sub_url="instruments/synthesis",
            body=body,
        )
        return result["synth_source_id"]

    def delete_growth_instrument(self, synth_source_id: int) -> None:
        """Delete a growth instrument.

        Args:
            synth_source_id: ID of the instrument to delete.

        Raises:
            ClientError: If the instrument is not found or not accessible.

        Example:
            >>> client.delete_growth_instrument(synth_source_id=42)
        """
        self._delete(
            sub_url="instruments/synthesis",
            params={"synthesis_instrument_ids": synth_source_id},
        )
