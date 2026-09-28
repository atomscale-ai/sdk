"""Synthetic API responses. No credentials, catalogue discovery, or shared state.

Only the transport boundary is replaced: Client.get and the real providers still
construct results. Unknown routes fail loudly rather than returning empty data.
"""

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

from PIL import Image


class Sandbox:
    def __init__(self):
        types = (
            "rheed",
            "xps",
            "xrd",
            "raman",
            "photoluminescence",
            "optical",
            "tool_state",
            "recipe",
            "ellipsometry",
        )
        self.ids = SimpleNamespace(**{kind: f"sandbox-{kind}" for kind in types})
        self.ids.rheed_rotating = self.ids.rheed
        self.ids.rheed_stationary = self.ids.rheed
        self.ids.changepoint = self.ids.rheed
        self.ids.similarity_source_id = self.ids.rheed
        self.ids.similarity_workflow = "rheed"
        self.calls = []
        self.catalogue = [
            {
                "data_id": getattr(self.ids, kind),
                "char_source_type": kind,
                "raw_name": f"example-{kind}.vms",
                "pipeline_status": "success",
                "raw_file_type": "vms",
                "source_name": "Synthetic instrument",
                "growth_length": 10,
                "upload_datetime": "2024-01-01T00:00:00Z",
                "last_updated": "2024-01-02T00:00:00Z",
                "collected_datetime": "2024-01-01T00:00:00Z",
                "file_metadata": {},
                "tags": ["sdk-test"],
                "name": "Test owner",
                "physical_sample_id": "sample-1",
                "physical_sample_name": "Test sample",
                "sample_name": "Test sample",
                "detail_note_content": None,
                "detail_note_last_updated": None,
                "projects": [{"id": "project-1", "name": "Test project"}],
                "workspaces": [],
                "sha3_256": "0" * 64,
                "has_instrument_logs": False,
            }
            for kind in types
        ]
        properties = {
            "properties": {
                "signal": {
                    "relative_time_seconds": [0.0, 1.0, 2.0],
                    "unix_timestamp_ms": [1704067200000, 1704067201000, 1704067202000],
                    "values": [1.0, 2.0, 3.0],
                    "units": "a.u.",
                }
            }
        }
        samples = [{"id": "sample-1", "name": "Test sample"}]
        self.responses = {
            "physical_samples/": samples,
            "projects/": [{"id": "project-1", "name": "Test project"}],
            "projects/project-1/physical_samples": samples,
            "physical_samples/sample-1/timeseries/": {"properties": {}},
            "data_entries/processed_data/ffffffff-ffff-ffff-ffff-ffffffffffff": None,
        }
        for kind in ("xps", "xrd", "raman", "photoluminescence"):
            self.responses[f"{kind}/{getattr(self.ids, kind)}"] = {
                "id": f"result-{kind}",
                f"{kind}_id": f"result-{kind}",
                "binding_energies": [1.0, 2.0, 3.0],
                "energies": [1.0, 2.0, 3.0],
                "two_theta": [1.0, 2.0, 3.0],
                "intensities": [10.0, 20.0, 10.0],
                "predicted_composition": {"Si": 1.0},
                "detected_peaks": [],
                "set_elements": False,
            }
        for kind, path in (
            ("optical", "optical/timeseries/{}/"),
            ("tool_state", "tool-state/{}/timeseries/"),
            ("recipe", "recipe/{}/timeseries/"),
            ("ellipsometry", "ellipsometry/{}/timeseries/"),
        ):
            self.responses[path.format(getattr(self.ids, kind))] = deepcopy(properties)
        self.responses[f"rheed/timeseries/{self.ids.rheed}/"] = {
            "series_by_angle": [
                {
                    "angle": "0",
                    "view_id": "view-1",
                    "interval_id": "interval-1",
                    "azimuth_label": "100",
                    "series": [
                        {
                            "frame_number": i,
                            "relative_time_seconds": float(i),
                            "unix_timestamp_ms": 1704067200000 + i * 1000,
                            "specular_intensity": float(i + 1),
                        }
                        for i in range(3)
                    ],
                }
            ]
        }
        self.responses[f"rheed/{self.ids.rheed}/azimuths"] = [
            {
                "data_id": self.ids.rheed,
                "view_id": "view-1",
                "interval_id": "interval-1",
                "seed_frame": 0,
                "rpm": 12.0,
            }
        ]
        self.responses[f"data_entries/video_single_frames/{self.ids.rheed}"] = {
            "frames": []
        }
        self.responses[f"optical/frame/video_single_frames/{self.ids.optical}"] = {
            "frames": [{"image_uuid": "optical-frame"}]
        }
        self.responses["data_entries/processed_data/optical-frame"] = {
            "url": "https://sandbox.invalid/frame.png"
        }
        image = BytesIO()
        Image.new("RGB", (4, 4), "white").save(image, format="PNG")
        self.responses["https://sandbox.invalid/frame.png"] = image.getvalue()
        self.responses[f"similarity/rheed/{self.ids.rheed}/trajectory/"] = {
            "trajectories": [
                {
                    "reference_id": "reference-1",
                    "reference_item_name": "Synthetic reference",
                    "similarity_values": [0.9, 0.8],
                    "real_time_seconds": [0.0, 1.0],
                    "unix_times": [1704067200.0, 1704067201.0],
                    "is_active": False,
                    "averaged_count": 2,
                }
            ]
        }
        self.responses["changepoints/"] = {
            "changepoints": [
                {
                    "id": "changepoint-1",
                    "data_id": self.ids.rheed,
                    "data_modality": "rheed",
                    "property_name": "specular_intensity",
                    "severity": "critical",
                    "score": 0.9,
                    "window_start_elapsed": 0.0,
                    "window_end_elapsed": 1.0,
                    "detection_method": "intensity_profile",
                    "detail": {},
                }
            ]
        }

    def get(self, sub_url, params=None, **kwargs):
        self.calls.append((sub_url, deepcopy(params), deepcopy(kwargs)))
        if sub_url == "data_entries/":
            rows = self.catalogue
            ids = (params or {}).get("data_ids")
            if ids is not None:
                ids = [ids] if isinstance(ids, str) else ids
                rows = [row for row in rows if row["data_id"] in ids]
            return deepcopy(rows)
        route = kwargs.get("base_override") or sub_url
        assert route in self.responses, f"Unexpected sandbox request: {route} {params}"
        return deepcopy(self.responses[route])
