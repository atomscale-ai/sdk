"""Unified view labels keep stationary and rotating identities distinct."""
from atomscale.rheed_metadata import rheed_azimuths_to_dataframe
from atomscale.timeseries.rheed import RHEEDProvider


def test_repeated_labels_retain_view_identity():
    views = [dict(data_id="recording", processed_data_id="derivative", view_id=f"v{i}",
                  interval_id=f"i{i}", seed_frame=i*10, start_frame=i*10, end_frame=(i+1)*10,
                  rpm=0 if i == 0 else 12, azimuth_label="100", azimuth_source="user",
                  confidence=None, annotation={}) for i in range(2)]
    frame = rheed_azimuths_to_dataframe(views)
    assert frame.view_id.tolist() == ["v0", "v1"]
    assert frame.azimuth_label.tolist() == ["100", "100"]
    assert frame.rpm.tolist() == [0, 12]
    payload = {"series_by_angle": [dict(view_id=v["view_id"], interval_id=v["interval_id"],
        angle="100", azimuth_label="100", series=[dict(frame_number=0, specular_intensity=1)]) for v in views]}
    samples = RHEEDProvider().to_dataframe(payload)
    assert samples.index.is_unique
    assert samples.index.names == ["View ID", "Frame Number"]
