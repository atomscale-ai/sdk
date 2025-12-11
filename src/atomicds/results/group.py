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
    ):
        """Aggregated results for a physical sample."""
        self.physical_sample_id = physical_sample_id
        self.physical_sample_name = physical_sample_name
        self.data_results = data_results
        self.aligned_timeseries = aligned_timeseries
        self.non_timeseries = non_timeseries


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
