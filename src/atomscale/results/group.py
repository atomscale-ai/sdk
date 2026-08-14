from __future__ import annotations

from collections.abc import Iterable

from monty.json import MSONable
from pandas import DataFrame


class PhysicalSampleResult(MSONable):
    def __init__(
        self,
        physical_sample_id: str,
        physical_sample_name: str | None,
        data_results: list,
        aligned_timeseries: DataFrame | None,
        non_timeseries: list,
        sample_metrics: DataFrame | None = None,
    ):
        """Aggregated results for a physical sample.

        ``sample_metrics`` holds the sample-scoped computed timeseries results
        (``rheed_quality``, ``composition_metric``, …) in the long form returned
        by :meth:`~atomscale.client.Client.get_physical_sample_timeseries`, or
        ``None`` when they were not fetched. It is distinct from
        ``aligned_timeseries``, which is the per-data-item curated join.
        """
        self.physical_sample_id = physical_sample_id
        self.physical_sample_name = physical_sample_name
        self.data_results = data_results
        self.aligned_timeseries = aligned_timeseries
        self.non_timeseries = non_timeseries
        self.sample_metrics = sample_metrics


class ProjectResult(MSONable):
    def __init__(
        self,
        project_id: str,
        project_name: str | None,
        samples: Iterable[PhysicalSampleResult],
        aligned_timeseries: DataFrame | None,
    ):
        """Aggregated results for a project."""
        self.project_id = project_id
        self.project_name = project_name
        self.samples = list(samples)
        self.aligned_timeseries = aligned_timeseries
