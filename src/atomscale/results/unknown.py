from __future__ import annotations

from typing import Any
from uuid import UUID

from monty.json import MSONable


class UnknownResult(MSONable):
    """Fallback result for unsupported or unknown data types.

    Stores minimal information from the data catalogue so callers can
    still inspect the entry even when no domain-specific handler exists.
    """

    def __init__(
        self,
        data_id: UUID | str,
        data_type: str | None,
        catalogue_entry: dict[str, Any] | None = None,
    ) -> None:
        self.data_id = data_id
        self.data_type = data_type
        # keep a shallow copy so later mutation of the source entry doesn't
        # affect the stored snapshot
        self.catalogue_entry = dict(catalogue_entry or {})

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary of catalogue fields if present."""

        keys_of_interest = (
            "raw_name",
            "pipeline_status",
            "raw_file_type",
            "upload_datetime",
            "last_accessed_datetime",
            "char_source_type",
            "physical_sample_id",
            "project_ids",
        )

        return {k: v for k, v in self.catalogue_entry.items() if k in keys_of_interest}
